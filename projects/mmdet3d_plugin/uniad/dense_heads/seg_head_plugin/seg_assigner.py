"""
seg_assigner.py - 分割任务中的样本分配器模块

本模块实现了用于分割任务（特别是基于Transformer的分割头）的样本分配器（Assigner）和采样器（Sampler）。
在目标检测/分割任务中，分配器负责将预测的查询（queries）与真实标签（ground truth）进行匹配，
采样器则决定哪些样本用于计算损失。

本模块包含以下核心组件：
1. SamplingResult_segformer: 为分割任务定制的采样结果类，存储正负样本的索引、边界框、掩码等信息。
2. PseudoSampler_segformer: 伪采样器，不执行实际的采样操作，直接返回所有正负样本。
3. HungarianAssigner_filter: 匈牙利算法分配器（带过滤），支持多轮匹配，限制每个真值匹配的预测数量。
4. HungarianAssigner_multi_info: 匈牙利算法分配器（多信息），在匹配代价中额外加入掩码（mask）代价，
   适合需要同时优化检测框和分割掩码的任务。
"""

from mmdet.core import mask
import torch
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core import mask
import torch

from mmdet.utils import util_mixins


# 无穷大常量，用于在匈牙利匹配中屏蔽已匹配的项
INF = 10000000


class SamplingResult_segformer(util_mixins.NiceRepr):
    """
    分割任务专用的采样结果类。

    该类继承自 util_mixins.NiceRepr，用于提供友好的字符串表示。
    与标准的 SamplingResult 相比，该类额外存储了真实掩码（gt_masks）信息，
    以支持分割任务中的掩码相关操作。

    存储的主要信息包括：
    - 正样本索引和负样本索引
    - 正样本和负样本的预测边界框
    - 正样本对应的真实边界框和真实掩码
    - 正样本对应的真实标签
    """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, gt_masks,assign_result,
                 gt_flags):
        """
        初始化采样结果。

        Args:
            pos_inds (Tensor): 正样本的索引，形状为 (num_pos,)。
            neg_inds (Tensor): 负样本的索引，形状为 (num_neg,)。
            bboxes (Tensor): 所有预测的边界框，形状为 (num_pred, 4)。
            gt_bboxes (Tensor): 真实边界框，形状为 (num_gt, 4)。
            gt_masks (Tensor): 真实分割掩码，形状为 (num_gt, H, W)。
            assign_result (AssignResult): 分配结果对象，包含 gt_inds 和 labels 信息。
            gt_flags (Tensor): 标记每个预测框是否为真实框的标志位。
        """
        # 存储正负样本的索引
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        # 根据索引提取正样本和负样本的预测边界框
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        # 标记正样本是否为真实标注框（而非预测框）
        self.pos_is_gt = gt_flags[pos_inds]

        # 真实标注框的数量
        self.num_gts = gt_bboxes.shape[0]
        # 正样本所匹配的真实框索引（gt_inds 从1开始，减1转为0-based索引）
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1


        if gt_bboxes.numel() == 0:
            # 当没有真实框时，处理边界情况，避免索引错误
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)

            #print('pos_gt_bboxes',self.pos_gt_bboxes.shape)
            #print('gt_mask',gt_masks.shape)
            n,h,w = gt_masks.shape
            #n = self.pos_gt_bboxes.shape[0]
            self.pos_gt_masks = torch.empty_like(gt_masks).view(-1, h,w)
        else:
            # 确保 gt_bboxes 是二维张量
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            # 根据匹配索引提取正样本对应的真实框和真实掩码
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
            self.pos_gt_masks = gt_masks[self.pos_assigned_gt_inds, :]

        # 存储正样本对应的真实标签
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        """
        返回所有样本（正样本+负样本）的预测边界框。

        Returns:
            Tensor: 拼接后的正负样本边界框，形状为 (num_pos + num_neg, 4)。
        """
        return torch.cat([self.pos_bboxes, self.neg_bboxes])


    def to(self, device):
        """
        将采样结果中的所有张量移动到指定设备上（原地操作）。

        Args:
            device (torch.device | str): 目标设备，如 'cpu'、'cuda:0' 等。

        Returns:
            self: 返回自身以支持链式调用。

        Example:
            >>> self = SamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        """
        生成友好的字符串表示（供 NiceRepr 使用）。

        Returns:
            str: 包含采样结果信息的格式化字符串。
        """
        data = self.info.copy()
        data['pos_bboxes'] = data.pop('pos_bboxes').shape
        data['neg_bboxes'] = data.pop('neg_bboxes').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """
        返回包含采样结果信息的字典。

        Returns:
            dict: 包含正负样本索引、边界框、标志位等信息的字典。
        """
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_bboxes': self.pos_bboxes,
            'neg_bboxes': self.neg_bboxes,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }

    @classmethod
    def random(cls, rng=None, **kwargs):
        """
        生成一个随机的采样结果（用于测试）。

        Args:
            rng (None | int | numpy.random.RandomState): 随机种子或随机状态。
            kwargs (keyword arguments):
                - num_preds: 预测框的数量
                - num_gts: 真实框的数量
                - p_ignore (float): 预测框被分配为忽略真值的概率
                - p_assigned (float): 预测框未被分配的概率
                - p_use_label (float | bool): 是否使用标签

        Returns:
            SamplingResult: 随机生成的采样结果。

        Example:
            >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
            >>> self = SamplingResult.random()
            >>> print(self.__dict__)
        """
        from mmdet.core.bbox.samplers.random_sampler import RandomSampler
        from mmdet.core.bbox.assigners.assign_result import AssignResult
        from mmdet.core.bbox import demodata
        rng = demodata.ensure_rng(rng)

        # 设置采样参数：总样本数32，正样本比例0.5，负正比上限-1（不限制）
        num = 32
        pos_fraction = 0.5
        neg_pos_ub = -1

        # 随机生成分配结果
        assign_result = AssignResult.random(rng=rng, **kwargs)

        # 随机生成预测框和真实框
        bboxes = demodata.random_boxes(assign_result.num_preds, rng=rng)
        gt_bboxes = demodata.random_boxes(assign_result.num_gts, rng=rng)

        if rng.rand() > 0.2:
            # 有时算法会压缩数据，为保持鲁棒性也做压缩
            gt_bboxes = gt_bboxes.squeeze()
            bboxes = bboxes.squeeze()

        if assign_result.labels is None:
            gt_labels = None
        else:
            gt_labels = None  # todo

        if gt_labels is None:
            add_gt_as_proposals = False
        else:
            add_gt_as_proposals = True  # 以一定概率将真实框也作为候选

        # 使用随机采样器执行采样
        sampler = RandomSampler(
            num,
            pos_fraction,
            neg_pos_ub=neg_pos_ub,
            add_gt_as_proposals=add_gt_as_proposals,
            rng=rng)
        self = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        return self


@BBOX_SAMPLERS.register_module()
class PseudoSampler_segformer(BaseSampler):
    """
    分割任务专用的伪采样器。

    伪采样器不执行实际的采样操作（即不会从所有候选中随机抽取子集），
    而是直接返回所有分配结果中的正样本和负样本。
    在DETR系列模型中，由于使用的是匈牙利算法进行一对一匹配，
    不需要额外的正负样本采样，因此使用伪采样器。

    与标准 PseudoSampler 的区别在于：sample 方法额外接收 gt_masks 参数，
    并返回 SamplingResult_segformer 对象（包含掩码信息）。
    """

    def __init__(self, **kwargs):
        """初始化伪采样器，不需要任何配置参数。"""
        pass

    def _sample_pos(self, **kwargs):
        """
        采样正样本（伪采样器不支持此操作）。

        Raises:
            NotImplementedError: 伪采样器不应调用此方法。
        """
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """
        采样负样本（伪采样器不支持此操作）。

        Raises:
            NotImplementedError: 伪采样器不应调用此方法。
        """
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes,gt_masks, **kwargs):
        """
        直接返回分配结果中的正样本和负样本索引，不进行随机采样。

        工作流程：
        1. 从分配结果中找到 gt_inds > 0 的索引作为正样本（匹配到真实框的预测）。
        2. 从分配结果中找到 gt_inds == 0 的索引作为负样本（未匹配的预测，即背景）。
        3. 创建 gt_flags 标记（全零，表示所有预测框都不是真实框）。
        4. 构建并返回 SamplingResult_segformer 对象。

        Args:
            assign_result (AssignResult): 分配结果，包含 gt_inds 和 labels 信息。
            bboxes (Tensor): 所有预测边界框，形状为 (num_pred, 4)。
            gt_bboxes (Tensor): 真实边界框，形状为 (num_gt, 4)。
            gt_masks (Tensor): 真实分割掩码，形状为 (num_gt, H, W)。

        Returns:
            SamplingResult_segformer: 包含正负样本信息的采样结果对象。
        """
        # 找到正样本索引：gt_inds > 0 表示该预测匹配到了某个真实框
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        # 找到负样本索引：gt_inds == 0 表示该预测未匹配到任何真实框（背景）
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        # 创建标志位：全零表示所有预测框都不是真实框（伪采样器不会将真实框加入候选）
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        # 构建并返回分割任务专用的采样结果
        sampling_result = SamplingResult_segformer(pos_inds, neg_inds, bboxes, gt_bboxes,gt_masks,
                                         assign_result, gt_flags,**kwargs)
        return sampling_result


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner_filter(BaseAssigner):
    """
    带过滤功能的匈牙利算法分配器。

    该类使用匈牙利算法在预测框和真实框之间进行一对一匹配，代价由分类代价、
    回归L1代价和IoU代价三部分加权组成。与标准 HungarianAssigner 的区别在于：

    1. 支持多轮匹配（max_pos 参数控制最大匹配轮数），每轮匹配后将被匹配的预测
       行设置为 INF，防止重复匹配。
    2. 每轮最多匹配 300/num_gts 个预测框，以防止单个真实框匹配过多预测。
    3. 返回正样本索引、负样本索引和分配结果。

    适用场景：分割任务中需要限制每个真实框匹配的预测数量的情况。
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 max_pos = 3
                 ):
        """
        初始化分配器。

        Args:
            cls_cost (dict): 分类代价的配置，默认为 ClassificationCost，权重为1。
            reg_cost (dict): 回归L1代价的配置，默认为 BBoxL1Cost，权重为1.0。
            iou_cost (dict): IoU代价的配置，默认为 IoUCost（GIoU模式），权重为1.0。
            max_pos (int): 最大匹配轮数，默认为3。每轮执行一次匈牙利匹配，
                最多匹配 max_pos 轮或 300/num_gts 轮（取较小值）。
        """
        # 构建各类匹配代价函数
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.max_pos = max_pos

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """
        执行预测框与真实框之间的匹配分配。

        工作流程：
        1. 默认将所有预测分配为 -1（不关心）。
        2. 如果没有真实框或预测框，则将所有预测分配为背景（0），返回空结果。
        3. 计算分类代价、回归L1代价和IoU代价的加权和。
        4. 使用图像尺寸对边界框进行归一化。
        5. 在CPU上执行多轮匈牙利匹配：
           - 每轮执行 linear_sum_assignment 找到最优匹配。
           - 将已匹配的预测行代价设为 INF，防止重复匹配。
           - 最多进行 max_pos 轮或直到所有匹配代价都达到 INF。
        6. 返回正样本索引、负样本索引和分配结果。

        Args:
            bbox_pred (Tensor): 预测的边界框，格式为 (cx, cy, w, h)，归一化到[0,1]。
                形状为 (num_query, 4)。
            cls_pred (Tensor): 预测的分类logits，形状为 (num_query, num_class)。
            gt_bboxes (Tensor): 真实边界框，格式为 (x1, y1, x2, y2)，未归一化。
                形状为 (num_gt, 4)。
            gt_labels (Tensor): 真实标签，形状为 (num_gt,)。
            img_meta (dict): 图像的元信息，包含 img_shape 等字段。
            gt_bboxes_ignore (Tensor, optional): 需要忽略的真实框。默认 None。
            eps (float): 数值稳定性参数，默认 1e-7。

        Returns:
            tuple:
                - pos_ind (Tensor): 正样本的索引。
                - neg_ind (Tensor): 负样本的索引。
                - result (AssignResult): 分配结果对象，包含 gt_inds 和 labels 信息。
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 步骤1: 默认将所有预测分配为 -1（表示未分配/不关心）
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)

        assigned_labels = bbox_pred.new_full((num_bboxes, ),-1,dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # 没有真实框或预测框，返回空分配结果
            if num_gts == 0:
                # 没有真实框时，将所有预测分配为背景（0）
                assigned_gt_inds[:] = 0
                pos_ind = assigned_gt_inds.gt(0).nonzero().squeeze(1)
                neg_ind = assigned_gt_inds.eq(0).nonzero().squeeze(1)
            return pos_ind, neg_ind,  AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 获取图像尺寸，用于边界框归一化
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        # 步骤2: 计算加权代价矩阵
        # 分类代价：基于预测的类别分数和真实标签
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # 回归L1代价：将真实框归一化后计算L1距离
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # 回归IoU代价：将预测框从 (cx,cy,w,h) 转为 (x1,y1,x2,y2) 并还原到图像尺度后计算GIoU
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # 总代价 = 分类代价 + 回归L1代价 + IoU代价
        cost = cls_cost + reg_cost + iou_cost

        # 步骤3: 在CPU上执行匈牙利匹配
        cost = cost.detach()

        assigned_gt_inds[:] = 0

        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        result=None
        # 多轮匹配：最多进行 max_pos 轮，或最多 300/num_gts 轮（取较小值）
        for i in range(min(self.max_pos, 300//num_gts)):
            # 将代价矩阵移到CPU上执行匈牙利算法
            cost = cost.cpu()
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)

            # 将匹配结果移回原设备
            matched_row_inds = torch.from_numpy(matched_row_inds).to(
                bbox_pred.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(
                bbox_pred.device)

            # 将已匹配的预测行代价设为 INF，防止下一轮再次匹配
            cost = cost.to(bbox_pred.device)
            cost[matched_row_inds,:] = INF

            # 记录匹配结果：gt_inds = col_inds + 1（0表示背景，所以加1）
            assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
            assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

            # 第一轮匹配的结果作为最终返回结果
            if i == 0:
                result = AssignResult(num_gts, assigned_gt_inds.clone(), None, labels=assigned_labels.clone())

            # 如果匹配的代价已经达到 INF，说明所有可匹配的预测都已匹配完毕，停止循环
            if cost[matched_row_inds,matched_col_inds].max()>=INF:
                break

        # 提取正样本和负样本的索引
        pos_ind = assigned_gt_inds.gt(0).nonzero().squeeze(1)
        neg_ind = assigned_gt_inds.eq(0).nonzero().squeeze(1)

        return pos_ind, neg_ind, result



@BBOX_ASSIGNERS.register_module()
class HungarianAssigner_multi_info(BaseAssigner):
    """
    多信息匈牙利算法分配器。

    在标准匈牙利分配器的基础上，额外加入了掩码（mask）代价，使得匹配过程
    同时考虑分类代价、回归L1代价、IoU代价和掩码代价（Dice代价）。

    这种设计适用于需要同时预测检测框和分割掩码的任务（如全景分割），
    通过引入掩码代价可以更好地将预测与真实标注进行匹配。

    与标准 HungarianAssigner 的主要区别：
    - 额外接收 mask_pred 和 gt_mask 参数
    - 额外计算 mask_cost（Dice代价）
    - 总代价 = cls_cost + reg_cost + iou_cost + mask_cost
    - 分类代价权重翻倍（cls_cost['weight'] *= 2），以平衡与其他代价的影响
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 mask_cost=dict(type='DiceCost', weight=1.0)

                 ):
        """
        初始化多信息分配器。

        Args:
            cls_cost (dict): 分类代价的配置，默认为 ClassificationCost，权重为1。
                注意：初始化时权重会被乘以2，以增强分类代价在总代价中的影响。
            reg_cost (dict): 回归L1代价的配置，默认为 BBoxL1Cost，权重为1.0。
            iou_cost (dict): IoU代价的配置，默认为 IoUCost（GIoU模式），权重为1.0。
            mask_cost (dict): 掩码代价的配置，默认为 DiceCost（Dice系数代价），权重为1.0。
        """
        # 分类代价权重翻倍，增强分类信息在匹配中的作用
        cls_cost['weight'] *= 2
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.mask_cost = build_match_cost(mask_cost)


    def assign(self,
               bbox_pred,
               cls_pred,
               mask_pred,
               gt_bboxes,
               gt_labels,
               gt_mask,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """
        基于多信息代价（分类+回归+IoU+掩码）执行匈牙利匹配。

        匹配流程：
        1. 默认将所有预测分配为 -1（不关心）。
        2. 如果没有真实框或预测框，将所有预测分配为背景（0），返回空结果。
        3. 计算四部分代价并加权求和：
           - cls_cost: 分类代价
           - reg_cost: 回归L1代价（将真实框归一化到[0,1]后计算）
           - iou_cost: IoU代价（GIoU模式）
           - mask_cost: 掩码代价（Dice系数代价）
        4. 在CPU上执行匈牙利匹配（linear_sum_assignment）。
        5. 将匹配到的预测分配对应的 gt_inds（从1开始），未匹配的分配为0（背景）。

        Args:
            bbox_pred (Tensor): 预测的边界框，格式为 (cx, cy, w, h)，归一化到[0,1]。
                形状为 (num_query, 4)。
            cls_pred (Tensor): 预测的分类logits，形状为 (num_query, num_class)。
            mask_pred (Tensor): 预测的分割掩码，形状为 (num_query, H, W)。
            gt_bboxes (Tensor): 真实边界框，格式为 (x1, y1, x2, y2)，未归一化。
                形状为 (num_gt, 4)。
            gt_labels (Tensor): 真实标签，形状为 (num_gt,)。
            gt_mask (Tensor): 真实分割掩码，形状为 (num_gt, H, W)。
            img_meta (dict): 图像的元信息，包含 img_shape 等字段。
            gt_bboxes_ignore (Tensor, optional): 需要忽略的真实框。默认 None。
            eps (float): 数值稳定性参数，默认 1e-7。

        Returns:
            AssignResult: 分配结果对象，包含：
                - num_gts: 真实框数量
                - gt_inds: 每个预测分配的真实框索引（1-based，0表示背景）
                - labels: 每个预测分配的类别标签
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 步骤1: 默认将所有预测分配为 -1（未分配）
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)

        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # 没有真实框或预测框，返回空分配
            if num_gts == 0:
                # 没有真实框时，将所有预测分配为背景（0）
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 获取图像尺寸，用于归一化
        img_h, img_w, _ = img_meta['img_shape']

        factor = bbox_pred.new_tensor([img_w, img_h, img_w,img_h]).unsqueeze(0)


        # 步骤2: 计算各类代价
        # 分类代价
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # 回归L1代价：将真实框归一化后计算
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # 回归IoU代价：将预测框转为 (x1,y1,x2,y2) 格式并还原到图像尺度
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # 掩码代价：基于 Dice 系数计算
        mask_cost = self.mask_cost(mask_pred,gt_mask)
        # 总代价 = 分类 + 回归L1 + IoU + 掩码
        cost = cls_cost + reg_cost + iou_cost + mask_cost

        # 步骤3: 在CPU上执行匈牙利匹配
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 步骤4: 分配前景和背景
        # 先将所有索引分配为背景（0）
        assigned_gt_inds[:] = 0
        # 将匹配到的预测分配对应的真实框索引（从1开始，+1）
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)