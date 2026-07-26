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
                track_query,              # (B, A, D) 来自 Track Head 的跟踪查询
                lane_query,               # (B, M, D) 来自 Map/Seg Head 的地图查询
                track_query_pos=None,     # (B, A, D) 跟踪查询的位置编码
                lane_query_pos=None,      # (B, M, D) 地图查询的位置编码
                track_bbox_results=None,  # List[dict] 检测框信息（坐标/朝向/类别）
                bev_embed=None,           # (B, H*W, D) BEV 特征图
                reference_trajs=None,     # (B, A, P, S, 2) 参考轨迹（K-means 锚点）
                traj_reg_branches=None,   # nn.ModuleList 每层一个轨迹回归 MLP
                agent_level_embedding=None,         # (B, A, P, D) agent 坐标系锚点嵌入
                scene_level_ego_embedding=None,     # (B, A, P, D) 全局坐标系锚点嵌入
                scene_level_offset_embedding=None,  # (B, A, P, D) 偏移量锚点嵌入
                learnable_embed=None,               # (B, A, P, D) 可学习锚点嵌入
                agent_level_embedding_layer=None,   # nn.Module agent 坐标系位置编码 MLP
                scene_level_ego_embedding_layer=None,   # nn.Module 全局坐标系位置编码 MLP
                scene_level_offset_embedding_layer=None,  # nn.Module 偏移量位置编码 MLP
                **kwargs):
        """
        MotionTransformerDecoder 的前向传播函数。

        这是 UniAD 运动预测模块的核心 —— 对每个智能体，通过多层解码器迭代地预测其未来轨迹。
        每一层解码器按顺序执行以下 7 个步骤：

        步骤 1 — 静态-动态意图融合:
            将不变的静态意图嵌入（锚点K-means中心 + 偏移 + 可学习嵌入）与
            当前层的动态意图嵌入（智能体级 + 偏移级 + 自车级）融合，再与上层查询合并。

        步骤 2 — 智能体间交互 (TrackAgentInteraction):
            Cross-Attention: 每个智能体的 query 关注所有其他智能体的 key/value，
            建模多智能体之间的社交影响（跟车、避让、竞速等）。

        步骤 3 — 智能体-地图交互 (MapInteraction):
            Cross-Attention: 每个智能体的 query 关注地图元素(lane_query)的 key/value，
            让智能体理解道路结构约束（车道保持、转弯限制等）。

        步骤 4 — 智能体-BEV 交互 (Deformable Attention):
            以当前预测轨迹的路径点为 2D 参考采样位置，在 BEV 特征图上做可变形注意力，
            直接从原始感知特征中提取路径沿线的环境信息。

        步骤 5 — 多源融合:
            将上述四种交互的输出 + 原始 track_query 拼接，通过 MLP 融合为统一的查询嵌入。

        步骤 6 — 轨迹迭代细化:
            用回归分支预测当前轨迹的偏移量（cumsum 后得到绝对坐标），
            更新参考轨迹供下一层使用。

        步骤 7 — 位置嵌入更新:
            根据更新后的参考轨迹终点，重新计算三类位置嵌入（agent/ego/offset），
            供下一层解码器使用。

        ---
        整体设计思想：
        - 每一层解码器都对轨迹进行一步"修正"，层数越深预测越精确
        - 四种交互从不同角度提供信息：社交(context)、地图(context)、感知(raw feature)、意图(prior)
        - 轨迹采用 cumsum 方式预测（预测相邻步之间的位移增量），保证时序平滑性
        - 三类位置嵌入（agent/ego/offset）在不同坐标系下编码轨迹终点，提供多视角位置信息

        ---
        张量形状约定：
            B  = batch_size（通常为 1，因为 UniAD 逐场景处理）
            A  = agent 数量（包括自车 SDC，约 N_active + 1）
            P  = 模态数/锚点数（num_anchor = 6，6 种可能的运动模式）
            S  = 预测步数（通常为 12，即预测未来 6 秒，0.5 秒/步）
            M  = 地图元素数量（lane_query 的数量）
            D  = 嵌入维度（embed_dims = 256）
            H*W = BEV 特征图的空间尺寸

        Args:
            track_query (Tensor):
                来自 Track Head 的跟踪查询，代表每个智能体的身份和状态特征。
                形状 (B, A, D)。每个 256 维向量编码了该智能体的类别、外观、运动状态等信息。
                注意：自车(SDC)的 query 已拼接在最后一个位置（index = -1）。

            lane_query (Tensor):
                来自 Map/Seg Head 的地图查询，代表车道线、路口等地图元素的特征。
                形状 (B, M, D)。每个 256 维向量编码了对应地图元素的几何和语义信息。

            track_query_pos (Tensor):
                跟踪查询的位置编码，形状 (B, A, D)。
                通过 pos2posemb2d 将每个智能体的 BEV 中心坐标 (x,y) 编码为高维正弦嵌入。

            lane_query_pos (Tensor):
                地图查询的位置编码，形状 (B, M, D)。

            track_bbox_results (List[dict]):
                检测框结果列表，每个元素包含该智能体的 bbox 坐标、朝向角、类别等信息。
                用于将轨迹在不同坐标系之间转换（agent<->ego<->global）。

            bev_embed (Tensor):
                来自 BEVFormer 编码器的 BEV 特征图，形状 (B, H*W, D)。
                是整个 UniAD 系统的共享感知表征，由多相机图像融合得到。

            reference_trajs (Tensor):
                参考轨迹，形状 (B, A, P, S, 2)。
                初始化为 K-means 聚类得到的锚点轨迹，在每层解码器中逐步细化。
                最后一维的 2 表示 (x, y) 坐标。

            traj_reg_branches (nn.ModuleList):
                轨迹回归分支，长度为 num_layers。每层一个 MLP，将 query_embed (D 维)
                映射为轨迹偏移 (S*2 维)，预测相邻步之间的位移增量。

            --- 三类锚点嵌入（均来自 K-means 聚类中心，形状 (B, A, P, D)）---
            agent_level_embedding (Tensor):
                智能体自身坐标系下的锚点轨迹终点嵌入。
                使用智能体朝向角做旋转变换，反映"以智能体为中心"的运动模式。

            scene_level_ego_embedding (Tensor):
                全局场景坐标系下的锚点轨迹终点嵌入。
                使用智能体位置做平移变换，反映"在世界地图中"的运动模式。

            scene_level_offset_embedding (Tensor):
                偏移量嵌入 —— 轨迹终点相对于当前智能体位置的偏移。
                不做坐标变换，反映"从当前位置出发"的运动模式。

            learnable_embed (Tensor):
                可学习的锚点嵌入，形状 (B, A, P, D)。
                通过 nn.Embedding 初始化，在训练中学习，作为静态意图的一部分。

            agent_level_embedding_layer (nn.Module):
                将位置编码映射到嵌入空间的 MLP 层（智能体坐标系用）。

            scene_level_ego_embedding_layer (nn.Module):
                将位置编码映射到嵌入空间的 MLP 层（全局坐标系用）。

            scene_level_offset_embedding_layer (nn.Module):
                将位置编码映射到嵌入空间的 MLP 层（偏移量用）。

        Returns:
            Tuple[Tensor, Tensor]:
                - intermediate (Tensor):
                    每层解码器输出的查询嵌入。
                    形状 (num_layers, B, A, P, D)。
                    intermediate[0] 是第 1 层输出，intermediate[-1] 是最后一层输出。
                    这些查询嵌入会被下游模块（Occ Head、Planning Head）使用。

                - intermediate_reference_trajs (Tensor):
                    每层解码器输出的参考轨迹。
                    形状 (num_layers, B, A, P, S, 2)。
                    最后一层输出的轨迹就是最终的 6 条多模态预测轨迹。
        """
        intermediate = []  # 存储每层解码器的查询嵌入，最终 stack 后返回
        intermediate_reference_trajs = []  # 存储每层解码器细化后的参考轨迹

        B, _, P, D = agent_level_embedding.shape

        # --- 预处理: 将 (B, A, D) 的 track_query 扩展到 (B, A, P, D) ---
        # 每个智能体只有 1 个跟踪查询（代表其身份和状态），但需要预测 P=6 条不同的未来轨迹。
        # 做法: 将同一个 track_query 在模态维度上复制 P 份，每份配合不同的锚点嵌入来产生不同的预测。
        # 这样 6 个模态共享同一个 agent 身份信息，但各自有独立的运动先验。
        track_query_bc = track_query.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)
        track_query_pos_bc = track_query_pos.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)

        # --- 预处理: 构建静态意图嵌入 ---
        # 静态意图 = 锚点（模态间交互后的智能体级嵌入）+ 偏移量 + 可学习嵌入
        # "静态"意味着它在所有解码器层中保持不变，作为不变的先验知识。
        # 1. IntentionInteraction: 让 P=6 个模态之间做 self-attention，学习模态间关系（如直行和右转的差异）
        # 2. 三个嵌入相加: 三者从不同角度描述了锚点终点的位置信息，相加后形成统一的静态意图
        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)
        static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed

        # --- 预处理: 参考轨迹维度扩展 ---
        # Deformable Attention 需要参考轨迹中每个采样点的 (x, y) 坐标来确定在 BEV 特征图上采样的位置。
        # unsqueeze(4) 在最后一维前插入一个维度: (B, A, P, S, 2) -> (B, A, P, S, 1, 2)
        # 这个新增的维度是 deformable attention 的"采样点层级"维度，此处为 1 表示每个轨迹点只有 1 个采样层级。
        # detach() 阻断梯度回传，因为参考轨迹是作为"位置索引"而非"可学习变量"使用的。
        reference_trajs_input = reference_trajs.unsqueeze(4).detach()

        # --- 预处理: 查询嵌入初始化为零 ---
        # 第 0 层没有"上层查询"，所以用零向量作为初始查询。
        # 送入第一层解码器时，query_embed 和意图嵌入通过 in_query_fuser 融合，
        # 相当于第一层主要依赖意图嵌入（锚点先验），后续层则逐步融合上层解码器的输出。
        query_embed = torch.zeros_like(static_intention_embed)

        for lid in range(self.num_layers):
            # =====================================================================
            # 步骤 1: 静态-动态意图融合
            # =====================================================================
            # 静态意图: static_intention_embed，在循环外计算一次，所有层共用
            #   它代表了锚点轨迹的"原始形状"——K-means 聚类出来的 6 种典型运动模式
            # 动态意图: 每层重新计算，因为参考轨迹在不断细化，三类位置嵌入 (agent/ego/offset)
            #   都在变化，动态意图反映了"当前层对轨迹终点的最新猜测"
            # =====================================================================
            dynamic_query_embed = self.dynamic_embed_fuser(torch.cat(
                [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1))

            # 将静态和动态意图融合: 导航系统中"先验路线" + "实时位置修正"的组合
            query_embed_intention = self.static_dynamic_fuser(torch.cat(
                [static_intention_embed, dynamic_query_embed], dim=-1))  # (B, A, P, D)

            # 将意图嵌入与上层查询融合: 第 1 层时 query_embed 是零向量，所以意图占主导
            # 后续层中 query_embed 携带了上层解码器的输出，意图和上层查询各占一定比例
            query_embed = self.in_query_fuser(torch.cat([query_embed, query_embed_intention], dim=-1))

            # =====================================================================
            # 步骤 2: 智能体间交互 (Agent-Agent Cross-Attention)
            # =====================================================================
            # 设计: 每个智能体的 query 关注所有其他智能体的 key/value
            # query: (B, A, P, D) — 当前智能体的嵌入，包含 P 个模态
            # key:   (B, A, D) — 所有智能体的跟踪查询（模态间共享）
            # 效果: 智能体 A 可以"看到"智能体 B 的位置和运动状态，从而调整自己的轨迹预测
            # 例如: 前方车辆减速 → 后方车辆预测轨迹也应减速
            # =====================================================================
            track_query_embed = self.track_agent_interaction_layers[lid](
                query_embed, track_query, query_pos=track_query_pos_bc, key_pos=track_query_pos)

            # =====================================================================
            # 步骤 3: 智能体-地图交互 (Agent-Map Cross-Attention)
            # =====================================================================
            # 设计: 每个智能体的 query 关注所有地图元素 (lane_query) 的 key/value
            # query: (B, A, P, D) — 智能体嵌入
            # key:   (B, M, D) — 地图元素嵌入（车道线、路口等）
            # 效果: 智能体 query 隐式地"学会"了车道保持、跟随弯道等与道路结构相关的行为
            # 注意: 这里不是显式地指定"沿着车道线走"，而是让 cross-attention 自动学习关联
            # =====================================================================
            map_query_embed = self.map_interaction_layers[lid](
                query_embed, lane_query, query_pos=track_query_pos_bc, key_pos=lane_query_pos)

            # =====================================================================
            # 步骤 4: 智能体-BEV 交互 (Deformable Attention on BEV features)
            # =====================================================================
            # 设计: 以当前预测轨迹的路径点为参考点，在 BEV 特征图上做可变形注意力采样
            # 这是四种交互中唯一直接访问原始感知特征的路径，其他三种是 query-to-query 交互
            # reference_trajs_input: (B, A, P, S, 1, 2) — 每个轨迹点提供 1 个 2D 采样位置
            # 效果: 模型可以"查看"预测轨迹沿线的实际环境（是否有障碍物、路面标记等），
            #   从而调整轨迹以避免碰撞或违反交通规则
            # =====================================================================
            bev_query_embed = self.bev_interaction_layers[lid](
                query_embed,
                value=bev_embed,
                query_pos=track_query_pos_bc,
                bbox_results=track_bbox_results,
                reference_trajs=reference_trajs_input,
                **kwargs)

            # =====================================================================
            # 步骤 5: 多源融合 (Fuse all four interaction outputs)
            # =====================================================================
            # 四种信息源:
            #   1. track_query_embed: 来自其他智能体的社交信息
            #   2. map_query_embed:    来自地图元素的道路结构信息
            #   3. bev_query_embed:    来自 BEV 原始特征的感知信息
            #   4. track_query_bc + track_query_pos_bc: 原始 track query（残差连接）
            # 拼接 → 4*D 维 → MLP 压缩回 D 维，形成统一的查询嵌入
            # =====================================================================
            query_embed = [track_query_embed, map_query_embed, bev_query_embed, track_query_bc+track_query_pos_bc]
            query_embed = torch.cat(query_embed, dim=-1)  # 拼接: (B, A, P, 4*D)
            query_embed = self.out_query_fuser(query_embed)  # 融合: (B, A, P, D)

            if traj_reg_branches is not None:
                # =====================================================================
                # 步骤 6: 轨迹迭代细化 (Iterative Trajectory Refinement)
                # =====================================================================
                # 核心思路: 用 MLP 回归分支预测轨迹偏移量，通过 cumsum 得到绝对坐标
                #
                # 为什么用 cumsum?
                #   MLP 输出的是相邻步之间的位移增量 (Δx, Δy)，而不是绝对坐标。
                #   cumsum 将这些增量累加，得到每一步的绝对位置。
                #   这样做的好处:
                #   - 保证轨迹的时序平滑性（每步的位移增量是有界的）
                #   - 位移增量相对于绝对坐标更容易学习（数值范围更稳定）
                #   - 天然保证了轨迹的连续性
                #
                # 示例: 如果 MLP 输出 [Δ1, Δ2, Δ3, ...]，cumsum 后得到
                #   [Δ1, Δ1+Δ2, Δ1+Δ2+Δ3, ...] = 绝对位置序列
                # =====================================================================
                tmp = traj_reg_branches[lid](query_embed)  # (B, A, P, D) -> (B, A, P, S*2)
                bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape
                tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)  # (B, A, P, S, 2)

                # cumsum 在时间维度上累加位移增量，得到绝对坐标轨迹
                tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
                new_reference_trajs = torch.zeros_like(reference_trajs)
                new_reference_trajs = tmp[..., :2]  # 提取 (x, y) 坐标

                # detach() 阻断梯度: 参考轨迹是作为下一层 deformable attention 的"采样位置"
                # 而非需要梯度回传的变量，这样可以避免梯度环（层间解耦，类似 DETR 的做法）
                reference_trajs = new_reference_trajs.detach()
                reference_trajs_input = reference_trajs.unsqueeze(4)  # (B, A, P, S, 1, 2)

                # =====================================================================
                # 步骤 7: 更新位置嵌入 (Update Position Embeddings)
                # =====================================================================
                # 参考轨迹更新后，其终点位置也变了，需要重新计算三类位置嵌入。
                # 只取轨迹的终点 [..., -1, :] 来编码位置信息，因为终点最能代表
                # "这个轨迹最终会到达哪里"。
                #
                # 三种坐标系下的终点嵌入:
                #   ep_agent_embed:  在 agent 自身坐标系下（经过旋转变换，朝向对齐）
                #   ep_ego_embed:    在全局坐标系下（经过平移变换，世界坐标）
                #   ep_offset_embed: 相对于当前 agent 位置的偏移（不做坐标变换）
                #
                # 编码流程: 终点坐标 → norm_points (归一化到 [0,1]) → pos2posemb2d (正弦编码)
                #          → MLP (映射到 256 维嵌入空间)
                # =====================================================================

                # agent 坐标系: 只旋转变换（对齐 agent 朝向），坐标原点在 agent 当前位置
                ep_agent_embed = trajectory_coordinate_transform(
                    reference_trajs.unsqueeze(2),
                    track_bbox_results,
                    with_translation_transform=False,
                    with_rotation_transform=True
                ).squeeze(2).detach()

                # 全局坐标系: 只平移变换（对齐世界坐标），坐标原点在场景原点
                ep_ego_embed = trajectory_coordinate_transform(
                    reference_trajs.unsqueeze(2),
                    track_bbox_results,
                    with_translation_transform=True,
                    with_rotation_transform=False
                ).squeeze(2).detach()

                # 偏移量: 不做坐标变换，即相对于 agent 当前位置的偏移
                ep_offset_embed = reference_trajs.detach()

                # 将终点位置编码为高维嵌入 (pos2posemb2d: 可学习的位置编码）
                # 1. norm_points: 将坐标归一化到 [0, 1] 范围（使用 pc_range 做 min-max 归一化）
                # 2. pos2posemb2d: 将 2D 位置编码为高维正弦/余弦嵌入
                # 3. MLP: 映射到 D=256 维的嵌入空间，与 query 维度对齐
                agent_level_embedding = agent_level_embedding_layer(pos2posemb2d(
                    norm_points(ep_agent_embed[..., -1, :], self.pc_range)))
                scene_level_ego_embedding = scene_level_ego_embedding_layer(pos2posemb2d(
                    norm_points(ep_ego_embed[..., -1, :], self.pc_range)))
                scene_level_offset_embedding = scene_level_offset_embedding_layer(pos2posemb2d(
                    norm_points(ep_offset_embed[..., -1, :], self.pc_range)))

                # 保存当前层的输出
                intermediate.append(query_embed)
                intermediate_reference_trajs.append(reference_trajs)

        # 将所有层的输出在 dim=0 上堆叠: (num_layers, B, A, P, D) 和 (num_layers, B, A, P, S, 2)
        # 下游模块（Occ Head, Planning Head）通常取最后一层 [-1] 的输出
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