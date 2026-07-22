#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
运动预测头的基础模块 (Base Motion Head)

本模块定义了 UniAD 中运动预测任务的基础类 `BaseMotionHead`。
运动预测是 UniAD 的关键任务之一，其目标是根据检测到的目标智能体（agent）的
历史轨迹和周围环境信息，预测它们未来若干步的轨迹。

该类作为运动预测头的基类，提供了以下核心功能：
1. 损失函数的构建（轨迹回归损失，使用 Softmax 分类后的加权回归）
2. 锚点（Anchor）的加载 -- 从聚类得到的 K-means 锚点用于初始化多模态轨迹预测
3. Transformer 解码器层的构建（MotionFormer）
4. 跟踪查询信息融合层、智能体级别/场景级别嵌入层的构建
5. 分类分支和回归分支的初始化（每个解码器层一对分支，迭代优化轨迹）
6. 从检测结果中提取归一化的目标中心点坐标，用于位置编码

该基类被具体的运动预测头（如 MotionHead）继承并实现完整的 forward 逻辑。
"""

import torch
import copy
import pickle
import torch.nn as nn
from mmdet.models import  build_loss
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence


class BaseMotionHead(nn.Module):
    """
    UniAD 运动预测头的基础类。

    负责运动预测任务中损失函数、锚点、神经网络层（Transformer 解码器、分类/回归分支）
    的构建和初始化。子类继承该类后，只需实现 forward 方法来完成完整的前向推理流程。

    主要组件：
    - loss_traj: 轨迹回归损失函数
    - kmeans_anchors: K-means 聚类得到的锚点轨迹，用于多模态初始化
    - learnable_motion_query_embedding: 可学习的运动查询嵌入
    - motionformer: MotionFormer 解码器（多层 Transformer 解码器序列）
    - layer_track_query_fuser: 将多层检测查询融合为单层跟踪查询的 MLP
    - agent_level_embedding_layer: 智能体级别嵌入（自车视角下的agent位置编码）
    - scene_level_ego_embedding_layer: 场景级别自车嵌入（全局坐标下的自车位置编码）
    - scene_level_offset_embedding_layer: 场景级别偏移嵌入（agent相对于当前位置的偏移）
    - boxes_query_embedding_layer: 检测框查询嵌入层
    - traj_cls_branches: 轨迹分类分支（多层，每层预测轨迹的置信度得分）
    - traj_reg_branches: 轨迹回归分支（多层，每层预测轨迹的偏移量）
    """

    def __init__(self, *args, **kwargs):
        """
        初始化 BaseMotionHead。

        注意：该类本身不执行具体的初始化逻辑，所有参数构建由子类调用
        _build_loss, _load_anchors, _build_layers, _init_layers 等方法完成。
        """
        super(BaseMotionHead, self).__init__()
        pass

    def _build_loss(self, loss_traj):
        """
        构建运动预测任务的损失函数。

        使用 mmdet 的 build_loss 根据配置字典构建损失函数。
        同时创建用于将预测结果展平的工具层。

        关键组件：
        - self.loss_traj: 轨迹回归损失，通常是 SmoothL1Loss 或 L1Loss
        - self.unflatten_traj: 将展平的预测结果还原为 (predict_steps, 5) 的形状，
          其中 5 表示 (dx, dy, score, ...) 等轨迹属性
        - self.log_softmax: 对多模态维度做 LogSoftmax，用于计算分类损失

        Args:
            loss_traj (dict): 损失函数的配置字典，包含 type 和具体参数。
                例如: dict(type='SmoothL1Loss', beta=1.0, reduction='mean')
        """
        self.loss_traj = build_loss(loss_traj)
        self.unflatten_traj = nn.Unflatten(3, (self.predict_steps, 5))
        self.log_softmax = nn.LogSoftmax(dim=2)

    def _load_anchors(self, anchor_info_path):
        """
        从 pickle 文件加载 K-means 聚类锚点轨迹。

        锚点轨迹是通过对训练集中的未来轨迹进行 K-means 聚类得到的。
        这些锚点用于初始化多模态轨迹预测，每个锚点代表一种可能的运动模式
        （如直行、左转、右转等）。

        加载后的锚点形状为 (Nc, Pc, steps, 2)：
        - Nc: 锚点群组数量（不同运动模式的聚类中心数量）
        - Pc: 每个群组的锚点数量（模态数量）
        - steps: 预测步数
        - 2: (x, y) 坐标

        Args:
            anchor_info_path (str): 锚点信息文件的路径（pickle 格式）。
                该文件通常包含 "anchors_all" 键，对应一个列表，
                每个元素是一个 numpy 数组，形状为 (Pc, steps, 2)。
        """
        anchor_infos = pickle.load(open(anchor_info_path, 'rb'))
        self.kmeans_anchors = torch.stack(
            [torch.from_numpy(a) for a in anchor_infos["anchors_all"]])  # Nc, Pc, steps, 2

    def _build_layers(self, transformerlayers, det_layer_num):
        """
        构建运动预测模块的神经网络层。

        包括以下组件：
        1. 可学习的运动查询嵌入 (learnable_motion_query_embedding):
           形状为 (num_anchor * num_anchor_group, embed_dims)，每个锚点对应一个可学习的嵌入向量。
           其中 num_anchor 是每个群组的锚点数量（模态数），num_anchor_group 是锚点群组数。

        2. MotionFormer 解码器 (motionformer):
           使用 mmcv 的 build_transformer_layer_sequence 根据配置构建多层 Transformer 解码器序列。
           每一层依次进行：意图交互、智能体间交互、智能体-地图交互、智能体-BEV 交互。

        3. 跟踪查询融合层 (layer_track_query_fuser):
           将多层检测 Transformer 解码器输出的跟踪查询融合为单一表示。
           输入: embed_dims * det_layer_num 维，输出: embed_dims 维。
           结构: Linear -> LayerNorm -> ReLU

        4. 智能体级别嵌入层 (agent_level_embedding_layer):
           将智能体在自车坐标系下的终点位置编码映射到高维嵌入空间。
           结构: Linear(embed_dims, embed_dims*2) -> ReLU -> Linear(embed_dims*2, embed_dims)

        5. 场景级别自车嵌入层 (scene_level_ego_embedding_layer):
           将自车在全局坐标系下的终点位置编码映射到高维嵌入空间。
           结构: Linear(embed_dims, embed_dims*2) -> ReLU -> Linear(embed_dims*2, embed_dims)

        6. 场景级别偏移嵌入层 (scene_level_offset_embedding_layer):
           将轨迹终点相对于起点的偏移量编码映射到高维嵌入空间。
           结构: Linear(embed_dims, embed_dims*2) -> ReLU -> Linear(embed_dims*2, embed_dims)

        7. 检测框查询嵌入层 (boxes_query_embedding_layer):
           将检测框的信息编码为查询嵌入。
           结构: Linear(embed_dims, embed_dims*2) -> ReLU -> Linear(embed_dims*2, embed_dims)

        Args:
            transformerlayers (dict): Transformer 解码器层的配置字典。
                包含每层的注意力机制、FFN 等配置。
            det_layer_num (int): 检测 Transformer 解码器的层数。
                用于确定融合层输入维度的大小。
        """
        # 可学习的运动查询嵌入: 每个锚点对应一个嵌入向量
        self.learnable_motion_query_embedding = nn.Embedding(
            self.num_anchor * self.num_anchor_group, self.embed_dims)

        # MotionFormer 解码器: 多层 Transformer 解码器序列
        self.motionformer = build_transformer_layer_sequence(
            transformerlayers)

        # 跟踪查询融合层: 将多层检测查询融合为单层表示
        self.layer_track_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * det_layer_num, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True)
        )

        # 智能体级别嵌入层: 编码 agent 在自车坐标系下的终点位置
        self.agent_level_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

        # 场景级别自车嵌入层: 编码自车在全局坐标系下的终点位置
        self.scene_level_ego_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

        # 场景级别偏移嵌入层: 编码轨迹终点相对于起点的偏移
        self.scene_level_offset_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

        # 检测框查询嵌入层: 编码检测框信息
        self.boxes_query_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

    def _init_layers(self):
        """
        初始化运动预测头的分类分支和回归分支。

        分类分支（traj_cls_branch）:
        - 结构: Linear -> LayerNorm -> ReLU -> [num_reg_fcs-1 个 (Linear -> LayerNorm -> ReLU)] -> Linear(embed_dims, 1)
        - 功能: 预测每条轨迹的置信度得分（单个标量），用于从多个模态中选择最佳轨迹
        - 输出维度: 1（每条轨迹一个得分）

        回归分支（traj_reg_branch）:
        - 结构: Linear -> ReLU -> [num_reg_fcs-1 个 (Linear -> ReLU)] -> Linear(embed_dims, predict_steps * 5)
        - 功能: 预测未来 predict_steps 步的轨迹偏移量
        - 输出维度: predict_steps * 5（每步 5 个值，包含 dx, dy 以及可能的其他属性如速度、置信度等）

        使用 _get_clones 函数为 MotionFormer 的每一层创建独立的分类和回归分支，
        实现迭代细化（iterative refinement）的轨迹预测。
        """
        # 构建轨迹分类分支
        traj_cls_branch = []
        traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
        traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
        traj_cls_branch.append(nn.ReLU(inplace=True))
        for _ in range(self.num_reg_fcs-1):
            traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(nn.Linear(self.embed_dims, 1))  # 输出单个得分
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        # 构建轨迹回归分支
        traj_reg_branch = []
        traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
        traj_reg_branch.append(nn.ReLU())
        for _ in range(self.num_reg_fcs-1):
            traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_reg_branch.append(nn.ReLU())
        traj_reg_branch.append(nn.Linear(self.embed_dims, self.predict_steps * 5))  # 输出轨迹坐标
        traj_reg_branch = nn.Sequential(*traj_reg_branch)

        # 辅助函数: 深度克隆模块 N 次，返回 ModuleList
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.motionformer.num_layers
        # 为每一层解码器创建独立的分类和回归分支（迭代细化）
        self.traj_cls_branches = _get_clones(traj_cls_branch, num_pred)
        self.traj_reg_branches = _get_clones(traj_reg_branch, num_pred)

    def _extract_tracking_centers(self, bbox_results, bev_range):
        """
        从检测结果中提取归一化的目标边界框中心点坐标。

        该函数用于将检测到的每个智能体的位置信息编码为位置嵌入，
        作为运动预测 Transformer 的输入之一。

        处理流程：
        1. 遍历每个 batch 样本的检测结果
        2. 提取每个检测框的重力中心点（gravity_center）
        3. 将中心点的 (x, y) 坐标根据 BEV 范围归一化到 [0, 1] 区间
        4. 返回归一化后的中心点坐标张量

        Args:
            bbox_results (List[Tuple[torch.Tensor]]): 检测结果列表，每个元素是一个元组，
                包含 (bboxes, scores, labels, bbox_index, mask)。
                bboxes 对象需要有 gravity_center 属性，返回形状为 (N, 2) 或 (N, 3) 的张量。
            bev_range (List[float]): BEV（鸟瞰图）范围，格式为 [x_min, y_min, z_min, x_max, y_max, z_max]。
                用于将坐标归一化到 [0, 1] 区间。

        Returns:
            torch.Tensor: 归一化后的检测框中心点坐标，形状为 (batch_size, max_num_agents, 2)。
                坐标值在 [0, 1] 范围内，分别表示 x 和 y 方向上的归一化位置。
        """
        batch_size = len(bbox_results)
        det_bbox_posembed = []
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            # 提取重力中心点的 x, y 坐标
            xy = bboxes.gravity_center[:, :2]
            # 对 x 坐标进行归一化: 将 [x_min, x_max] 映射到 [0, 1]
            x_norm = (xy[:, 0] - bev_range[0]) / \
                (bev_range[3] - bev_range[0])
            # 对 y 坐标进行归一化: 将 [y_min, y_max] 映射到 [0, 1]
            y_norm = (xy[:, 1] - bev_range[1]) / \
                (bev_range[4] - bev_range[1])
            det_bbox_posembed.append(
                torch.cat([x_norm[:, None], y_norm[:, None]], dim=-1))
        return torch.stack(det_bbox_posembed)