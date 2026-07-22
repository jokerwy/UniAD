#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
占用预测头部（OccHead）评估指标模块
===================================

本模块定义了占用预测（Occupancy Prediction）任务中使用的评估指标，包括：

1. **IntersectionOverUnion**：标准的交并比（IoU）指标，用于评估语义分割/占用预测
   在每个类别上的分割质量。支持忽略指定类别、缺失类别评分和自定义归约方式。

2. **PanopticMetric**：全景质量（Panoptic Quality, PQ）指标，用于评估全景分割
   （同时考虑语义分割和实例分割）的质量。计算PQ、SQ（分割质量）和RQ（识别质量）。
   支持时序一致性检查，用于在视频序列中验证实例ID的跨帧一致性。

这些指标基于 PyTorch Lightning 的 Metric 基类实现，支持分布式训练中的
自动聚合（通过 dist_reduce_fx 进行跨进程求和）。
"""

from typing import Optional

import torch
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.functional.classification import stat_scores_multiple_classes
from pytorch_lightning.metrics.functional.reduction import reduce


class IntersectionOverUnion(Metric):
    """交并比（IoU）评估指标。

    计算多类别语义分割的交并比。对于每个类别，IoU = TP / (TP + FP + FN)。

    特性：
    - 支持忽略指定类别（ignore_index），该类别不参与评分计算。
    - 支持缺失类别评分（absent_score）：当某个类别在预测和目标中都不存在时，
      使用该分数替代（默认为0.0）。
    - 支持多种归约方式（reduction）：'none'（逐类别返回）、'mean'（取平均）、
      'sum'（求和）等。

    参数：
        n_classes (int): 类别总数。
        ignore_index (Optional[int]): 需要忽略的类别索引，默认为None。
        absent_score (float): 当某个类别在目标中不存在时的默认分数，默认为0.0。
        reduction (str): 归约方式，默认为'none'（逐类别返回）。
        compute_on_step (bool): 是否在每个step上计算指标，默认为False（在epoch结束时计算）。
    """

    def __init__(
        self,
        n_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        reduction: str = 'none',
        compute_on_step: bool = False,
    ):
        """初始化IoU指标。

        注册四个状态张量用于累积统计量：
        - true_positive: 每个类别的真正例数
        - false_positive: 每个类别的假正例数
        - false_negative: 每个类别的假负例数
        - support: 每个类别的目标像素数

        所有状态张量在分布式训练中通过 'sum' 进行跨进程聚合。

        参数：
            n_classes (int): 类别总数。
            ignore_index (Optional[int]): 忽略的类别索引。
            absent_score (float): 缺失类别分数。
            reduction (str): 归约方式。
            compute_on_step (bool): 是否在每步计算。
        """
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score
        self.reduction = reduction

        # 注册状态张量：每个类别一个统计值，跨进程求和聚合
        self.add_state('true_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('support', default=torch.zeros(n_classes), dist_reduce_fx='sum')

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """更新指标状态：根据当前批次的预测和目标累积统计量。

        调用 pytorch_lightning 的 stat_scores_multiple_classes 函数，
        计算当前批次的 TP、FP、FN 和 support，并累加到状态张量中。

        参数：
            prediction (torch.Tensor): 预测的类别标签，形状为任意维度。
            target (torch.Tensor): 真实的类别标签，形状与 prediction 相同。
        """
        # 计算当前批次的统计量：tp, fp, tn, fn, sup
        # stat_scores_multiple_classes 返回 (tps, fps, tns, fns, sups)
        tps, fps, _, fns, sups = stat_scores_multiple_classes(prediction, target, self.n_classes)

        # 累加到全局状态中
        self.true_positive += tps
        self.false_positive += fps
        self.false_negative += fns
        self.support += sups

    def compute(self):
        """计算最终的IoU得分。

        对每个类别（除忽略类别外）计算 IoU = TP / (TP + FP + FN)。
        如果某个类别在目标和预测中都不存在，则使用 absent_score。

        返回：
            torch.Tensor: IoU得分。如果 reduction='none'，形状为 (n_classes,) 或
                          (n_classes-1,)（如果存在忽略类别）；否则为标量。
        """
        # 初始化得分张量，默认为0
        scores = torch.zeros(self.n_classes, device=self.true_positive.device, dtype=torch.float32)

        for class_idx in range(self.n_classes):
            # 跳过需要忽略的类别
            if class_idx == self.ignore_index:
                continue

            tp = self.true_positive[class_idx]
            fp = self.false_positive[class_idx]
            fn = self.false_negative[class_idx]
            sup = self.support[class_idx]

            # 如果该类在目标中不存在（support=0）且预测中也不存在（tp+fp=0），
            # 则使用 absent_score 作为该类得分
            if sup + tp + fp == 0:
                scores[class_idx] = self.absent_score
                continue

            # 计算 IoU = TP / (TP + FP + FN)
            denominator = tp + fp + fn
            score = tp.to(torch.float) / denominator
            scores[class_idx] = score

        # 从得分中移除忽略类别的索引
        if (self.ignore_index is not None) and (0 <= self.ignore_index < self.n_classes):
            scores = torch.cat([scores[:self.ignore_index], scores[self.ignore_index + 1:]])

        # 根据指定的归约方式返回最终结果
        return reduce(scores, reduction=self.reduction)


class PanopticMetric(Metric):
    """全景质量（Panoptic Quality, PQ）评估指标。

    评估全景分割的质量，同时考虑语义分割和实例分割。计算三个核心指标：
    - **PQ（全景质量）**：PQ = SQ * RQ，综合衡量分割质量和识别质量。
    - **SQ（分割质量）**：所有匹配实例的平均IoU，衡量分割精度。
    - **RQ（识别质量）**：衡量实例检测的F1分数。

    指标计算基于以下统计量：
    - true_positive: 预测与真值匹配的实例数（IoU > 0.5 且类别匹配）。
    - false_positive: 预测中存在但真值中不存在的实例数。
    - false_negative: 真值中存在但预测中不存在的实例数。
    - iou: 所有匹配实例的IoU之和。

    支持时序一致性检查：在视频序列中，vehicle类别的实例ID需要在帧间保持
    一致。如果同一真值ID匹配到不同的预测ID，则计为FP和FN。

    参数：
        n_classes (int): 语义类别总数。
        temporally_consistent (bool): 是否启用时序一致性检查，默认为True。
        vehicles_id (int): 车辆类别的语义ID，用于时序一致性检查，默认为1。
        compute_on_step (bool): 是否在每个step上计算指标，默认为False。
    """

    def __init__(
        self,
        n_classes: int,
        temporally_consistent: bool = True,
        vehicles_id: int = 1,
        compute_on_step: bool = False,
    ):
        """初始化全景质量指标。

        注册四个状态张量用于累积统计量，每个类别一个值。

        参数：
            n_classes (int): 类别总数。
            temporally_consistent (bool): 是否启用时序一致性。
            vehicles_id (int): 车辆类别的语义ID。
            compute_on_step (bool): 是否在每步计算。
        """
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.temporally_consistent = temporally_consistent
        self.vehicles_id = vehicles_id
        self.keys = ['iou', 'true_positive', 'false_positive', 'false_negative']

        # 注册状态张量：每个类别一个统计值，跨进程求和聚合
        self.add_state('iou', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('true_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(n_classes), dist_reduce_fx='sum')

    def update(self, pred_instance, gt_instance):
        """更新指标状态：根据预测和真值实例分割计算并累积全景质量统计量。

        对批次中的每个样本和每个时间步，调用 panoptic_metrics 计算
        单帧的全景质量统计量，然后累加到全局状态中。

        参数：
            pred_instance (torch.Tensor): 时序一致的实例分割预测，
                                          形状为 (B, S, H, W)，
                                          其中B是批次大小，S是序列长度。
                                          值为0表示背景，>0表示实例ID。
            gt_instance (torch.Tensor): 真值实例分割，
                                        形状为 (B, S, H, W)。
                                        值为0表示背景，>0表示实例ID。
        """
        batch_size, sequence_length = gt_instance.shape[:2]

        # 验证真值中ID=0必须是背景
        assert gt_instance.min() == 0, 'ID 0 of gt_instance must be background'

        # 从实例分割中提取语义分割：非零位置为前景
        pred_segmentation = (pred_instance > 0).long()
        gt_segmentation = (gt_instance > 0).long()

        for b in range(batch_size):
            # unique_id_mapping 用于跨时间步追踪实例ID的一致性
            # 键：真值实例ID，值：匹配的预测实例ID
            unique_id_mapping = {}
            for t in range(sequence_length):
                # 计算单帧的全景质量统计量
                result = self.panoptic_metrics(
                    pred_segmentation[b, t].detach(),
                    pred_instance[b, t].detach(),
                    gt_segmentation[b, t],
                    gt_instance[b, t],
                    unique_id_mapping,
                )

                # 累加统计量
                self.iou += result['iou']
                self.true_positive += result['true_positive']
                self.false_positive += result['false_positive']
                self.false_negative += result['false_negative']

    def compute(self):
        """计算最终的全景质量指标。

        根据累积的统计量计算 PQ、SQ 和 RQ：
        - PQ = sum(IoU) / (TP + FP/2 + FN/2)
        - SQ = sum(IoU) / TP
        - RQ = TP / (TP + FP/2 + FN/2)

        返回：
            dict: 包含以下键的字典：
                - 'pq': 全景质量，形状为 (n_classes,)
                - 'sq': 分割质量，形状为 (n_classes,)
                - 'rq': 识别质量，形状为 (n_classes,)
                - 'denominator': 分母值（TP + FP/2 + FN/2），用于调试，形状为 (n_classes,)
        """
        # 计算分母：TP + FP/2 + FN/2，使用 torch.maximum 防止除零
        denominator = torch.maximum(
            (self.true_positive + self.false_positive / 2 + self.false_negative / 2),
            torch.ones_like(self.true_positive)
        )
        # PQ = sum(IoU) / denominator
        pq = self.iou / denominator
        # SQ = sum(IoU) / TP（仅匹配的实例）
        sq = self.iou / torch.maximum(self.true_positive, torch.ones_like(self.true_positive))
        # RQ = TP / denominator（F1分数）
        rq = self.true_positive / denominator

        return {'pq': pq,
                'sq': sq,
                'rq': rq,
                # 如果分母为0，说明没有任何检测
                'denominator': (self.true_positive + self.false_positive / 2 + self.false_negative / 2),
                }

    def panoptic_metrics(self, pred_segmentation, pred_instance, gt_segmentation, gt_instance, unique_id_mapping):
        """计算单帧的全景质量统计量。

        这是全景质量指标的核心算法，执行以下步骤：
        1. 将语义分割和实例分割合并为统一的全景分割格式。
        2. 计算预测和真值之间的IoU混淆矩阵。
        3. 通过IoU阈值（>0.5）匹配预测和真值实例。
        4. 验证类别匹配，并检查时序一致性。
        5. 统计TP、FP、FN和IoU。

        参数：
            pred_segmentation (torch.Tensor): 预测的语义分割，形状为 [H, W]，
                                              值域 {0, ..., n_classes-1}。
            pred_instance (torch.Tensor): 预测的实例分割，形状为 [H, W]，
                                          值域 {0, ..., n_instances}（0表示背景）。
            gt_segmentation (torch.Tensor): 真值语义分割，形状为 [H, W]。
            gt_instance (torch.Tensor): 真值实例分割，形状为 [H, W]。
            unique_id_mapping (dict): 跨帧的实例ID映射字典，
                                      键为真值实例ID，值为预测实例ID。

        返回：
            dict: 包含 'iou', 'true_positive', 'false_positive', 'false_negative'
                  键的字典，每个值形状为 (n_classes,)。
        """
        n_classes = self.n_classes

        # 初始化当前帧的统计结果
        result = {key: torch.zeros(n_classes, dtype=torch.float32, device=gt_instance.device) for key in self.keys}

        # 验证输入维度：所有输入必须是2D且形状一致
        assert pred_segmentation.dim() == 2
        assert pred_segmentation.shape == pred_instance.shape == gt_segmentation.shape == gt_instance.shape

        # 计算总的实例数（预测和真值中最大的实例ID）
        n_instances = int(torch.cat([pred_instance, gt_instance]).max().item())
        # 总的全景ID数 = 语义类别数 + 实例数
        n_all_things = n_instances + n_classes
        # 再加1用于void类别（ID=0）
        n_things_and_void = n_all_things + 1

        # 将语义和实例分割合并为统一的全景分割格式
        # ID=0: void（无效像素），ID=1: background（背景），
        # ID=2: 车辆语义类（与实例重叠，实际上不使用），
        # ID>=3: 实例ID（从3开始）
        prediction, pred_to_cls = self.combine_mask(pred_segmentation, pred_instance, n_classes, n_all_things)
        target, target_to_cls = self.combine_mask(gt_segmentation, gt_instance, n_classes, n_all_things)

        # 计算预测和真值之间的混淆矩阵
        # 技巧：将两个数组编码为一个标量，通过 bincount 高效计算2D混淆矩阵
        x = prediction + n_things_and_void * target
        bincount_2d = torch.bincount(x.long(), minlength=n_things_and_void ** 2)
        if bincount_2d.shape[0] != n_things_and_void ** 2:
            raise ValueError('Incorrect bincount size.')
        conf = bincount_2d.reshape((n_things_and_void, n_things_and_void))
        # 移除void类别（ID=0），只保留有效类别
        conf = conf[1:, 1:]

        # 计算IoU混淆矩阵
        # union = 预测区域 + 真值区域 - 交集
        union = conf.sum(0).unsqueeze(0) + conf.sum(1).unsqueeze(1) - conf
        iou = torch.where(union > 0, (conf.float() + 1e-9) / (union.float() + 1e-9), torch.zeros_like(union).float())

        # IoU矩阵中，第一维是真值索引，第二维是预测索引
        # 找到所有IoU > 0.5的匹配对
        mapping = (iou > 0.5).nonzero(as_tuple=False)

        # 验证匹配对的类别是否一致
        is_matching = pred_to_cls[mapping[:, 1]] == target_to_cls[mapping[:, 0]]
        mapping = mapping[is_matching]

        # 创建TP掩码，标记所有真正例匹配
        tp_mask = torch.zeros_like(conf, dtype=torch.bool)
        tp_mask[mapping[:, 0], mapping[:, 1]] = True

        # 遍历所有匹配对，更新统计量
        # 前n_classes个ID对应"stuff"（语义类别），之后的ID对应实例（已偏移）
        for target_id, pred_id in mapping:
            cls_id = pred_to_cls[pred_id]

            # 时序一致性检查：仅对车辆类别进行
            if self.temporally_consistent and cls_id == self.vehicles_id:
                if target_id.item() in unique_id_mapping and unique_id_mapping[target_id.item()] != pred_id.item():
                    # 同一真值ID匹配到了不同的预测ID，说明时序不一致
                    # 计为FN（真值未被正确跟踪）和FP（预测为错误实例）
                    result['false_negative'][target_to_cls[target_id]] += 1
                    result['false_positive'][pred_to_cls[pred_id]] += 1
                    # 更新映射，以最新的匹配为准
                    unique_id_mapping[target_id.item()] = pred_id.item()
                    continue

            # 正常匹配：更新TP和IoU
            result['true_positive'][cls_id] += 1
            result['iou'][cls_id] += iou[target_id][pred_id]
            unique_id_mapping[target_id.item()] = pred_id.item()

        # 统计假负例（FN）：真值中存在但未被匹配的实例
        for target_id in range(n_classes, n_all_things):
            # 如果该实例已被匹配为TP，跳过
            if tp_mask[target_id, n_classes:].any():
                continue
            # 如果该真值实例存在且未被匹配，计为FN
            if target_to_cls[target_id] != -1:
                result['false_negative'][target_to_cls[target_id]] += 1

        # 统计假正例（FP）：预测中存在但未被匹配的实例
        for pred_id in range(n_classes, n_all_things):
            # 如果该实例已被匹配为TP，跳过
            if tp_mask[n_classes:, pred_id].any():
                continue
            # 如果该预测实例存在且未被匹配，且与真值有重叠，计为FP
            if pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
                result['false_positive'][pred_to_cls[pred_id]] += 1

        return result

    def combine_mask(self, segmentation: torch.Tensor, instance: torch.Tensor, n_classes: int, n_all_things: int):
        """将语义分割和实例分割合并为统一的全景分割掩码。

        合并策略：
        - 实例像素（instance > 0）的ID被偏移 n_classes 以区分于语义类别。
        - 语义类别像素保持在 [0, n_classes-1] 范围。
        - 所有有效ID +1，将ID=0保留给void类别。
        - 构建一个从全景ID到语义类别ID的映射表。

        参数：
            segmentation (torch.Tensor): 语义分割掩码，值域 [0, n_classes-1]。
            instance (torch.Tensor): 实例分割掩码，值域 [0, n_instances]（0为背景）。
            n_classes (int): 语义类别总数。
            n_all_things (int): 全景ID总数 = n_classes + n_instances。

        返回：
            tuple:
                - combined_mask (torch.Tensor): 合并后的全景掩码，形状同输入。
                  ID=0为void，其余为全景ID。
                - instance_id_to_class (torch.Tensor): 从全景ID到语义类别ID的映射，
                  形状为 (n_all_things,)，-1表示无效ID。
        """
        instance = instance.view(-1)
        instance_mask = instance > 0  # 实例像素掩码（非背景）
        instance = instance - 1 + n_classes  # 实例ID偏移：从0开始改为从n_classes开始

        segmentation = segmentation.clone().view(-1)
        segmentation_mask = segmentation < n_classes  # 有效语义像素掩码（排除void）

        # 构建实例ID到语义类别的映射表
        # 将每个实例像素的实例ID和语义类别配对
        instance_id_to_class_tuples = torch.cat(
            (
                instance[instance_mask & segmentation_mask].unsqueeze(1),
                segmentation[instance_mask & segmentation_mask].unsqueeze(1),
            ),
            dim=1,
        )
        instance_id_to_class = -instance_id_to_class_tuples.new_ones((n_all_things,))
        instance_id_to_class[instance_id_to_class_tuples[:, 0]] = instance_id_to_class_tuples[:, 1]
        # 语义类别（stuff）的映射：每个语义类别ID映射到自身
        instance_id_to_class[torch.arange(n_classes, device=segmentation.device)] = torch.arange(
            n_classes, device=segmentation.device
        )

        # 合并掩码：实例像素使用偏移后的实例ID
        segmentation[instance_mask] = instance[instance_mask]
        segmentation += 1  # 所有有效ID +1，将ID=0留给void
        segmentation[~segmentation_mask] = 0  # 无效像素（void）设为0

        return segmentation, instance_id_to_class