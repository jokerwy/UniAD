#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
运动预测 Transformer 解码器与交互模块 (Motion Transformer Decoder & Interaction Modules)

本模块实现了 UniAD 运动预测任务的核心 Transformer 解码器架构，包括以下组件：

1. MotionTransformerDecoder: 运动预测的 Transformer 解码器序列。
   该解码器接收来自跟踪头的智能体查询（track query）、地图查询（lane query）、
   BEV 特征（bev_embed）等信息，通过多层交互迭代地预测每个智能体的未来轨迹。

   每层解码器的交互流程如下：
   - 意图交互 (IntentionInteraction): 不同运动模态（锚点）之间的交互
   - 静态-动态意图融合: 将静态意图嵌入与动态意图嵌入融合
   - 智能体间交互 (TrackAgentInteraction): 不同智能体之间的交互
   - 智能体-地图交互 (MapInteraction): 智能体与地图车道线之间的交互
   - 智能体-BEV 交互 (bev_interaction): 智能体与 BEV 特征的交互（通过可变形注意力）
   - 融合所有交互结果并更新查询嵌入
   - 用回归分支更新参考轨迹（迭代细化），并更新相应的位置嵌入

2. TrackAgentInteraction: 智能体间交互模块。
   使用 TransformerDecoderLayer 实现不同智能体之间的信息交换。
   查询（query）是当前智能体的嵌入，键/值（key/value）是所有智能体的嵌入。

3. MapInteraction: 智能体与地图交互模块。
   使用 TransformerDecoderLayer 实现智能体与地图元素（车道线等）之间的信息交换。
   查询（query）是智能体嵌入，键/值（key/value）是地图查询嵌入。

4. IntentionInteraction: 意图/锚点间交互模块。
   使用 TransformerEncoderLayer 实现不同运动模态（锚点）之间的信息交换。
   使模型能够理解不同运动模式之间的关系。
"""

import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner.base_module import BaseModule
from projects.mmdet3d_plugin.models.utils.functional import (
    norm_points,
    pos2posemb2d,
    trajectory_coordinate_transform
)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MotionTransformerDecoder(BaseModule):
    """
    运动预测 Transformer 解码器。

    这是 UniAD 运动预测模块的核心解码器，负责将跟踪查询、地图查询和 BEV 特征
    进行深度融合，以迭代方式预测每个智能体的未来轨迹。

    该解码器包含多层，每层执行以下操作：
    1. 意图交互：在锚点（运动模态）之间进行信息交换
    2. 静态-动态意图融合：融合固定的锚点嵌入和上一层的动态嵌入
    3. 智能体间交互：Cross-Attention，query 为当前智能体嵌入，key/value 为所有智能体
    4. 智能体-地图交互：Cross-Attention，query 为智能体嵌入，key/value 为地图元素
    5. 智能体-BEV 交互：Deformable Attention，在 BEV 特征图上采样与轨迹相关的区域
    6. 多源融合：将上述所有交互结果拼接并通过 MLP 融合
    7. 迭代细化：用回归分支预测轨迹偏移，更新参考轨迹并更新各类位置嵌入

    输入张量说明：
    - track_query: (B, A, D) 跟踪查询，B=batch_size, A=agent数量, D=嵌入维度
    - lane_query: (B, M, D) 地图查询，M=地图元素数量
    - track_query_pos: (B, A, D) 跟踪查询的位置编码
    - lane_query_pos: (B, M, D) 地图查询的位置编码
    - track_bbox_results: 检测框结果列表
    - bev_embed: (B, H*W, D) BEV 特征嵌入
    - reference_trajs: (B, A, P, S, 2) 参考轨迹，P=模态数, S=预测步数
    - agent_level_embedding: (B, A, P, D) 智能体级别嵌入（自车坐标系）
    - scene_level_ego_embedding: (B, A, P, D) 场景级别自车嵌入（全局坐标系）
    - scene_level_offset_embedding: (B, A, P, D) 场景级别偏移嵌入
    - learnable_embed: (B, A, P, D) 可学习的嵌入

    Args:
        pc_range (list): 点云范围，用于坐标归一化。格式: [x_min, y_min, z_min, x_max, y_max, z_max]
        embed_dims (int): 嵌入维度，默认 256
        transformerlayers (dict): Transformer 层的配置字典
        num_layers (int): 解码器层数，默认 3
    """

    def __init__(self, pc_range=None, embed_dims=256, transformerlayers=None, num_layers=3, **kwargs):
        super(MotionTransformerDecoder, self).__init__()
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_layers = num_layers

        # 意图交互层: 在不同锚点（运动模态）之间进行 self-attention
        self.intention_interaction_layers = IntentionInteraction()

        # 智能体间交互层: 每层一个，实现不同智能体之间的 cross-attention
        self.track_agent_interaction_layers = nn.ModuleList(
            [TrackAgentInteraction() for i in range(self.num_layers)])

        # 智能体-地图交互层: 每层一个，实现智能体与地图元素之间的 cross-attention
        self.map_interaction_layers = nn.ModuleList(
            [MapInteraction() for i in range(self.num_layers)])

        # 智能体-BEV 交互层: 每层一个，使用可变形注意力在 BEV 特征上采样
        self.bev_interaction_layers = nn.ModuleList(
            [build_transformer_layer(transformerlayers) for i in range(self.num_layers)])

        # 静态-动态融合器: 将静态意图嵌入和动态意图嵌入融合
        # 输入: embed_dims*2 (静态 + 动态), 输出: embed_dims
        self.static_dynamic_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

        # 动态嵌入融合器: 将智能体级别嵌入、偏移嵌入和自车嵌入三者融合
        # 输入: embed_dims*3, 输出: embed_dims
        self.dynamic_embed_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*3, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

        # 输入查询融合器: 将上一层的查询嵌入与当前层的意图嵌入融合
        # 输入: embed_dims*2 (query_embed + query_embed_intention), 输出: embed_dims
        self.in_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

        # 输出查询融合器: 将四种交互结果（智能体间、地图、BEV、原始查询）融合
        # 输入: embed_dims*4, 输出: embed_dims
        self.out_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*4, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )

    def forward(self,
                track_query,
                lane_query,
                track_query_pos=None,
                lane_query_pos=None,
                track_bbox_results=None,
                bev_embed=None,
                reference_trajs=None,
                traj_reg_branches=None,
                agent_level_embedding=None,
                scene_level_ego_embedding=None,
                scene_level_offset_embedding=None,
                learnable_embed=None,
                agent_level_embedding_layer=None,
                scene_level_ego_embedding_layer=None,
                scene_level_offset_embedding_layer=None,
                **kwargs):
        """
        MotionTransformerDecoder 的前向传播函数。

        对每个智能体，通过多层解码器迭代地预测其未来轨迹。
        每一层执行意图交互、智能体间交互、智能体-地图交互、智能体-BEV 交互，
        然后融合所有交互结果并更新轨迹预测。

        Args:
            track_query (Tensor): 跟踪查询，形状 (B, A, D)
                B=batch_size, A=agent数量, D=embed_dims
            lane_query (Tensor): 地图查询，形状 (B, M, D)
                M=地图元素数量
            track_query_pos (Tensor): 跟踪查询的位置编码，形状 (B, A, D)
            lane_query_pos (Tensor): 地图查询的位置编码，形状 (B, M, D)
            track_bbox_results: 检测框结果列表
            bev_embed (Tensor): BEV 特征嵌入，形状 (B, H*W, D)
            reference_trajs (Tensor): 参考轨迹（初始化为锚点轨迹），形状 (B, A, P, S, 2)
                P=模态数, S=预测步数
            traj_reg_branches (nn.ModuleList): 轨迹回归分支列表，每层一个
            agent_level_embedding (Tensor): 智能体级别嵌入，形状 (B, A, P, D)
            scene_level_ego_embedding (Tensor): 场景级别自车嵌入，形状 (B, A, P, D)
            scene_level_offset_embedding (Tensor): 场景级别偏移嵌入，形状 (B, A, P, D)
            learnable_embed (Tensor): 可学习的嵌入，形状 (B, A, P, D)
            agent_level_embedding_layer (nn.Module): 智能体级别嵌入编码层
            scene_level_ego_embedding_layer (nn.Module): 场景级别自车嵌入编码层
            scene_level_offset_embedding_layer (nn.Module): 场景级别偏移嵌入编码层

        Returns:
            Tuple[Tensor, Tensor]:
                - intermediate: 每层输出的查询嵌入，形状 (num_layers, B, A, P, D)
                - intermediate_reference_trajs: 每层输出的参考轨迹，形状 (num_layers, B, A, P, S, 2)
        """
        intermediate = []  # 存储每层解码器的中间输出
        intermediate_reference_trajs = []  # 存储每层解码器的参考轨迹

        B, _, P, D = agent_level_embedding.shape

        # 将跟踪查询扩展到多模态维度: (B, A, D) -> (B, A, P, D)
        # 每个智能体在每种运动模态下都有一个查询副本
        track_query_bc = track_query.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)
        track_query_pos_bc = track_query_pos.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)

        # 静态意图嵌入: 通过意图交互层处理后，在所有解码器层中保持不变
        # 它融合了智能体级别嵌入、偏移嵌入和可学习嵌入
        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)
        static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed

        # 参考轨迹扩展维度: (B, A, P, S, 2) -> (B, A, P, S, 1, 2)，用于后续的 deformable attention
        reference_trajs_input = reference_trajs.unsqueeze(4).detach()

        # 查询嵌入初始化为零
        query_embed = torch.zeros_like(static_intention_embed)

        for lid in range(self.num_layers):
            # --- 步骤 1: 融合静态和动态意图嵌入 ---
            # 动态意图嵌入: 由智能体级别嵌入、偏移嵌入和自车嵌入三者融合而成
            # 它在每层解码器中都会更新，反映当前层对轨迹的最新预测
            dynamic_query_embed = self.dynamic_embed_fuser(torch.cat(
                [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1))

            # 静态-动态融合: 将不变的静态意图嵌入与变化的动态意图嵌入融合
            query_embed_intention = self.static_dynamic_fuser(torch.cat(
                [static_intention_embed, dynamic_query_embed], dim=-1))  # (B, A, P, D)

            # 将意图嵌入与上一层的查询嵌入融合，作为当前层的输入
            query_embed = self.in_query_fuser(torch.cat([query_embed, query_embed_intention], dim=-1))

            # --- 步骤 2: 智能体间交互 ---
            # 使用 Cross-Attention 让每个智能体感知其他智能体的状态
            # query: 当前智能体的嵌入 (B, A, P, D)
            # key: 所有智能体的嵌入 (B, A, D)
            track_query_embed = self.track_agent_interaction_layers[lid](
                query_embed, track_query, query_pos=track_query_pos_bc, key_pos=track_query_pos)

            # --- 步骤 3: 智能体-地图交互 ---
            # 使用 Cross-Attention 让智能体感知周围的地图元素（车道线等）
            map_query_embed = self.map_interaction_layers[lid](
                query_embed, lane_query, query_pos=track_query_pos_bc, key_pos=lane_query_pos)

            # --- 步骤 4: 智能体-BEV 交互 ---
            # 使用 Deformable Attention 在 BEV 特征图上根据参考轨迹采样
            # 这使智能体能够感知目标位置附近的 BEV 特征
            bev_query_embed = self.bev_interaction_layers[lid](
                query_embed,
                value=bev_embed,
                query_pos=track_query_pos_bc,
                bbox_results=track_bbox_results,
                reference_trajs=reference_trajs_input,
                **kwargs)

            # --- 步骤 5: 融合所有交互结果 ---
            # 将四种信息源拼接: 智能体间交互、地图交互、BEV 交互、原始查询+位置编码
            query_embed = [track_query_embed, map_query_embed, bev_query_embed, track_query_bc+track_query_pos_bc]
            query_embed = torch.cat(query_embed, dim=-1)  # 拼接: (B, A, P, 4*D)
            query_embed = self.out_query_fuser(query_embed)  # 融合: (B, A, P, D)

            if traj_reg_branches is not None:
                # --- 步骤 6: 更新参考轨迹（迭代细化） ---
                # 使用回归分支预测轨迹偏移量
                tmp = traj_reg_branches[lid](query_embed)
                bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape
                tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)

                # 使用累积和 (cumsum) 技巧: 预测的是相邻步之间的速度/位移，
                # 通过 cumsum 得到绝对坐标轨迹
                tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
                new_reference_trajs = torch.zeros_like(reference_trajs)
                new_reference_trajs = tmp[..., :2]  # 提取 (x, y) 坐标
                reference_trajs = new_reference_trajs.detach()  # 阻断梯度，作为下一层的参考
                reference_trajs_input = reference_trajs.unsqueeze(4)  # BS, NUM_AGENT, NUM_MODE, 12, 1, 2

                # --- 步骤 7: 更新各类位置嵌入 ---
                # 根据更新后的参考轨迹，重新计算终点位置嵌入，供下一层使用

                # 偏移嵌入: 轨迹终点相对于起点的偏移（全局坐标系）
                ep_offset_embed = reference_trajs.detach()

                # 自车嵌入: 轨迹终点在全局坐标系下的坐标
                # 将参考轨迹从 agent 坐标系转换到全局坐标系（平移变换）
                ep_ego_embed = trajectory_coordinate_transform(
                    reference_trajs.unsqueeze(2),
                    track_bbox_results,
                    with_translation_transform=True,
                    with_rotation_transform=False
                ).squeeze(2).detach()

                # 智能体嵌入: 轨迹终点在自车坐标系下的坐标
                # 将参考轨迹从 agent 坐标系转换到自车坐标系（旋转变换）
                ep_agent_embed = trajectory_coordinate_transform(
                    reference_trajs.unsqueeze(2),
                    track_bbox_results,
                    with_translation_transform=False,
                    with_rotation_transform=True
                ).squeeze(2).detach()

                # 将终点位置编码为高维嵌入
                # 1. 归一化坐标到 [0, 1] 范围 (norm_points)
                # 2. 将 2D 位置编码为高维正弦/余弦嵌入 (pos2posemb2d)
                # 3. 通过 MLP 映射到嵌入空间 (agent_level_embedding_layer 等)
                agent_level_embedding = agent_level_embedding_layer(pos2posemb2d(
                    norm_points(ep_agent_embed[..., -1, :], self.pc_range)))
                scene_level_ego_embedding = scene_level_ego_embedding_layer(pos2posemb2d(
                    norm_points(ep_ego_embed[..., -1, :], self.pc_range)))
                scene_level_offset_embedding = scene_level_offset_embedding_layer(pos2posemb2d(
                    norm_points(ep_offset_embed[..., -1, :], self.pc_range)))

                # 保存当前层的输出
                intermediate.append(query_embed)
                intermediate_reference_trajs.append(reference_trajs)

        # 将所有层的输出堆叠返回
        return torch.stack(intermediate), torch.stack(intermediate_reference_trajs)


class TrackAgentInteraction(BaseModule):
    """
    智能体间交互模块。

    建模场景中不同智能体之间的交互关系。使用标准的 TransformerDecoderLayer，
    其中查询（query）是当前智能体的嵌入，键（key）和值（value）是所有智能体的嵌入。

    核心思想：通过 Cross-Attention 机制，让每个智能体能够关注其他智能体的状态，
    从而理解多智能体之间的相互影响（如跟车、避让等行为）。

    输入形状：
    - query: (B, A, P, D) -- 批量大小 B, 智能体数 A, 模态数 P, 嵌入维度 D
    - key: (B, A, D) -- 所有智能体的嵌入
    - query_pos: (B, A, P, D) -- 查询的位置编码
    - key_pos: (B, A, D) -- 键的位置编码

    Args:
        embed_dims (int): 嵌入维度，默认 256
        num_heads (int): 多头注意力的头数，默认 8
        dropout (float): Dropout 比率，默认 0.1
        batch_first (bool): 是否 batch 维度在前，默认 True
        norm_cfg (dict): 归一化层配置
        init_cfg (dict): 初始化配置
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        # 使用 PyTorch 内置的 TransformerDecoderLayer 实现 Cross-Attention
        # 前馈网络维度为 embed_dims*2
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims*2,
            batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None):
        """
        智能体间交互的前向传播。

        将多模态查询展平后，通过 Cross-Attention 与所有智能体的嵌入进行交互。

        Args:
            query (Tensor): 当前智能体的查询嵌入，形状 (B, A, P, D)
            query_pos (Tensor): 查询位置编码，形状 (B, A, P, D)
            key (Tensor): 所有智能体的键嵌入，形状 (B, A, D)
            key_pos (Tensor): 键位置编码，形状 (B, A, D)

        Returns:
            Tensor: 交互后的智能体嵌入，形状 (B, A, P, D)
        """
        B, A, P, D = query.shape

        # 添加位置编码
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # 扩展 key 以匹配 batch 维度: (B, A, D) -> (B*A, A, D)
        # 每个 batch 样本中的所有智能体共享键
        mem = key.expand(B*A, -1, -1)

        # 展平查询: (B, A, P, D) -> (B*A, P, D)
        # 将 batch 和 agent 维度合并，方便输入 TransformerDecoderLayer
        query = torch.flatten(query, start_dim=0, end_dim=1)

        # Cross-Attention: query 关注 mem（所有智能体）
        query = self.interaction_transformer(query, mem)

        # 恢复形状: (B*A, P, D) -> (B, A, P, D)
        query = query.view(B, A, P, D)
        return query


class MapInteraction(BaseModule):
    """
    智能体与地图交互模块。

    建模智能体与地图元素（车道线、路口等）之间的交互关系。
    使用标准的 TransformerDecoderLayer，查询（query）是智能体嵌入，
    键（key）和值（value）是地图元素嵌入。

    核心思想：通过 Cross-Attention 机制，让每个智能体能够关注周围的地图信息，
    从而理解道路结构对运动的影响（如车道保持、转弯约束等）。

    与 TrackAgentInteraction 的区别：
    - TrackAgentInteraction: query 关注其他智能体（key 是智能体嵌入）
    - MapInteraction: query 关注地图元素（key 是地图嵌入）

    Args:
        embed_dims (int): 嵌入维度，默认 256
        num_heads (int): 多头注意力的头数，默认 8
        dropout (float): Dropout 比率，默认 0.1
        batch_first (bool): 是否 batch 维度在前，默认 True
        norm_cfg (dict): 归一化层配置
        init_cfg (dict): 初始化配置
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        # 使用 PyTorch 内置的 TransformerDecoderLayer 实现 Cross-Attention
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims*2,
            batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None):
        """
        智能体-地图交互的前向传播。

        将多模态查询展平后，通过 Cross-Attention 与地图元素进行交互。

        Args:
            query (Tensor): 智能体查询嵌入，形状 (B, A, P, D)
            query_pos (Tensor): 查询位置编码，形状 (B, A, P, D)
            key (Tensor): 地图元素嵌入，形状 (B, M, D)，M=地图元素数量
            key_pos (Tensor): 地图元素位置编码，形状 (B, M, D)

        Returns:
            Tensor: 与地图交互后的智能体嵌入，形状 (B, A, P, D)
        """
        B, A, P, D = query.shape

        # 添加位置编码
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # 展平查询: (B, A, P, D) -> (B*A, P, D)
        query = torch.flatten(query, start_dim=0, end_dim=1)

        # 扩展 key 以匹配 batch 维度: (B, M, D) -> (B*A, M, D)
        mem = key.expand(B*A, -1, -1)

        # Cross-Attention: query 关注 mem（地图元素）
        query = self.interaction_transformer(query, mem)

        # 恢复形状: (B*A, P, D) -> (B, A, P, D)
        query = query.view(B, A, P, D)
        return query


class IntentionInteraction(BaseModule):
    """
    意图/锚点间交互模块。

    建模不同运动模态（锚点）之间的交互关系。
    使用标准的 TransformerEncoderLayer，通过 Self-Attention 机制
    让不同的运动模态之间进行信息交换。

    核心思想：不同的运动模态（如直行、左转、右转）之间存在一定的关系，
    通过 Self-Attention 让模型隐式地学习这些关系，从而生成更协调的多模态预测。

    注意：这里使用的是 Encoder（Self-Attention），而非 Decoder（Cross-Attention），
    因为是在模态之间进行交互，没有外部键/值输入。

    Args:
        embed_dims (int): 嵌入维度，默认 256
        num_heads (int): 多头注意力的头数，默认 8
        dropout (float): Dropout 比率，默认 0.1
        batch_first (bool): 是否 batch 维度在前，默认 True
        norm_cfg (dict): 归一化层配置
        init_cfg (dict): 初始化配置
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        # 使用 TransformerEncoderLayer 实现 Self-Attention（模态间交互）
        self.interaction_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims*2,
            batch_first=batch_first)

    def forward(self, query):
        """
        意图交互的前向传播。

        将 batch 和 agent 维度合并后，通过 Self-Attention 在模态之间进行交互。

        Args:
            query (Tensor): 智能体嵌入，形状 (B, A, P, D)

        Returns:
            Tensor: 意图交互后的嵌入，形状 (B, A, P, D)
        """
        B, A, P, D = query.shape

        # 合并 batch 和 agent 维度: (B, A, P, D) -> (B*A, P, D)
        # 这样 Self-Attention 在模态维度 P 上进行交互
        rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)

        # Self-Attention: 模态之间互相交互
        rebatch_x = self.interaction_transformer(rebatch_x)

        # 恢复形状: (B*A, P, D) -> (B, A, P, D)
        out = rebatch_x.view(B, A, P, D)
        return out