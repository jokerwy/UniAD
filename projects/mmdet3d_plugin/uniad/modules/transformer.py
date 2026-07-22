# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
感知 Transformer (PerceptionTransformer)
========================================
这是 BEVFormer 的核心 Transformer 模块，负责将多视角图像特征转换为 BEV 特征，
并在 BEV 特征上进行 3D 目标检测。

整体架构:
    PerceptionTransformer
    ├── Encoder (BEVFormerEncoder)
    │   └── 将多视角图像特征 + BEV queries → BEV 鸟瞰图特征
    │
    └── Decoder (DetectionTransformerDecoder)
        └── 将 BEV 特征 + object queries → 3D 目标检测结果

两个核心方法:
    1. get_bev_features()  → 仅 Encoder: 图像特征 → BEV 特征
    2. forward()            → Encoder + Decoder: 图像特征 → BEV 特征 → 检测结果

时序 BEV 特征对齐:
    - 上一帧 BEV 特征需要旋转对齐到当前帧的坐标系
    - 自车运动信息 (can_bus) 通过 MLP 编码后注入 BEV queries
    - 图像特征添加 camera embeddings 和 level embeddings 以区分不同视角和尺度
"""

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from mmcv.runner import force_fp32, auto_fp16
from nuscenes.utils.data_classes import Quaternion

@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """感知 Transformer: BEVFormer 的核心模块

    将多视角图像特征通过 Encoder 转换为 BEV 特征，
    再通过 Decoder 在 BEV 特征上检测 3D 目标。

    核心设计:
    - Encoder: 空间交叉注意力 (SCA) + 时序自注意力 (TSA)
    - Decoder: 可变形自注意力 + 可变形交叉注意力
    - 时序对齐: 通过自车运动信息旋转和平移上一帧 BEV

    Args:
        num_feature_levels: FPN 输出的特征尺度数，默认 4
        num_cams: 相机数量，默认 6 (nuScenes 的 6 视角)
        two_stage_num_proposals: 两阶段检测时的 proposal 数量，默认 300
        encoder: Encoder 配置 (BEVFormerEncoder)
        decoder: Decoder 配置 (DetectionTransformerDecoder)
        embed_dims: 特征嵌入维度，默认 256
        rotate_prev_bev: 是否根据自车旋转角旋转上一帧 BEV
        use_shift: 是否使用自车位移来偏移 BEV 参考点
        use_can_bus: 是否将 can_bus 信息注入 BEV queries
        can_bus_norm: 是否对 can_bus MLP 输出做 LayerNorm
        use_cams_embeds: 是否为图像特征添加相机嵌入 (camera embeddings)
        rotate_center: 旋转中心坐标
    """

    def __init__(self,
                 num_feature_levels=4,        # FPN 特征尺度数
                 num_cams=6,                   # 相机数量
                 two_stage_num_proposals=300,  # 两阶段 proposal 数
                 encoder=None,                 # Encoder 配置 (BEVFormerEncoder)
                 decoder=None,                 # Decoder 配置 (DetectionTransformerDecoder)
                 embed_dims=256,               # 特征嵌入维度
                 rotate_prev_bev=True,         # 是否旋转上一帧 BEV
                 use_shift=True,               # 是否使用自车位移偏移
                 use_can_bus=True,             # 是否注入 can_bus 信息
                 can_bus_norm=True,            # can_bus MLP 是否加 LayerNorm
                 use_cams_embeds=True,         # 是否添加相机嵌入
                 rotate_center=[100, 100],     # 旋转中心
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)

        # 根据配置构建 Encoder 和 Decoder
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        # 时序对齐相关开关
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """初始化 Transformer 的可学习参数

        初始化以下可学习参数:
        1. level_embeds:  多尺度特征层级嵌入 (4, 256)
           用于区分不同 FPN 层级的特征，帮助模型知道特征来自哪个尺度

        2. cams_embeds:   相机视角嵌入 (6, 256)
           用于区分 6 个不同相机的特征，帮助模型知道特征来自哪个视角

        3. reference_points: 参考点预测器
           根据 query 的位置编码预测 3D 参考点 (cx, cy, cz)

        4. can_bus_mlp:   自车状态编码器
           将 18 维的 can_bus 信号 (位置、速度、加速度、朝向等)
           编码为 256 维嵌入，注入 BEV queries 以提供自车运动信息
        """
        # level_embeds: 多尺度特征层级嵌入
        # 形状 (num_feature_levels=4, embed_dims=256)
        # 每个尺度有一个独立的可学习嵌入向量
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))

        # cams_embeds: 相机视角嵌入
        # 形状 (num_cams=6, embed_dims=256)
        # 每个相机有一个独立的可学习嵌入向量
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))

        # reference_points: 根据 query 位置编码预测 3D 参考点
        # 输入 embed_dims=256 → 输出 3 (cx, cy, cz)
        self.reference_points = nn.Linear(self.embed_dims, 3)

        # can_bus_mlp: 自车状态编码器
        # 输入 18 维 can_bus 信号 → 隐藏层 128 → 输出 256 维嵌入
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),   # 18 → 128
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),  # 128 → 256
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """初始化所有权重

        初始化策略:
        - 所有维度 > 1 的参数: Xavier 均匀初始化
        - 注意力模块 (MSDeformableAttention3D, TemporalSelfAttention,
          CustomMSDeformableAttention): 调用各自的 init_weight 方法
        - level_embeds 和 cams_embeds: 标准正态分布初始化
        - can_bus_mlp: Xavier 均匀初始化，偏置置 0
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,          # 多尺度图像特征列表
            bev_queries,         # BEV 查询向量 (bev_h*bev_w, embed_dims)
            bev_h,               # BEV 特征图高度
            bev_w,               # BEV 特征图宽度
            real_h,              # 真实世界高度范围
            real_w,              # 真实世界宽度范围
            grid_length=[0.512, 0.512],  # BEV 网格实际长度
            bev_pos=None,        # BEV 位置编码
            prev_bev=None,       # 上一帧的 BEV 特征
            img_metas=None):     # 图像元信息
        """获取 BEV 特征 (仅 Encoder 部分)

        这是 BEV 特征生成的完整流程，包含以下步骤:

        Step 1: 准备 BEV Queries
            - 扩展 batch 维度
            - 注入 can_bus 自车运动信息

        Step 2: 时序对齐
            - 计算自车位移 (global → lidar)
            - 旋转上一帧 BEV 到当前帧坐标系

        Step 3: 准备图像特征
            - 展平多尺度特征
            - 添加 camera embeddings (区分视角)
            - 添加 level embeddings (区分尺度)

        Step 4: Encoder 前向传播
            - BEV queries 作为 query
            - 图像特征作为 key/value
            - 通过时序自注意力 (TSA) 和空间交叉注意力 (SCA) 生成 BEV 特征

        Args:
            mlvl_feats: 多尺度图像特征，每个尺度形状 (bs, num_cam, C, H, W)
            bev_queries: BEV 查询向量 (bev_h*bev_w, embed_dims)
            bev_h, bev_w: BEV 网格尺寸
            real_h, real_w: 真实世界范围 (m)
            grid_length: 每个 BEV 网格的实际大小
            bev_pos: BEV 位置编码
            prev_bev: 上一帧 BEV 特征
            img_metas: 图像元信息 (包含 can_bus, l2g_r_mat 等)

        Returns:
            bev_embed: BEV 特征，形状 (bev_h*bev_w, bs, embed_dims)
        """

        bs = mlvl_feats[0].size(0)

        # ============================================================
        # Step 1: 准备 BEV Queries
        # ============================================================
        # bev_queries: (bev_h*bev_w, embed_dims)
        # → (bev_h*bev_w, bs, embed_dims)  扩展 batch 维度
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)

        # bev_pos: (bs, embed_dims, bev_h, bev_w)
        # → (bev_h*bev_w, bs, embed_dims)  展平并转置
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # ============================================================
        # Step 2: 时序对齐 - 计算自车位移
        # ============================================================
        # can_bus[:3]: 自车在 global 坐标系下的位移增量 (Δx, Δy, Δz)
        # 需要将其从 global 坐标系转换到 lidar 坐标系
        delta_global = np.array([each['can_bus'][:3] for each in img_metas])

        # l2g_r_mat: lidar → global 旋转矩阵
        lidar2global_rotation = np.array([each['l2g_r_mat'] for each in img_metas])

        # 将 global 位移转到 lidar 坐标系: Δ_lidar = R^{-1} @ Δ_global
        delta_lidar = []
        for i in range(bs):
            delta_lidar.append(np.linalg.inv(lidar2global_rotation[i]) @ delta_global[i])

        delta_lidar = np.array(delta_lidar)

        # 归一化: 将位移量归一化到 [0, 1] 范围 (相对于 BEV 的实际范围)
        # 用于后续偏移 BEV 参考点
        shift_y = delta_lidar[:, 1] / real_h
        shift_x = delta_lidar[:, 0] / real_w

        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # (bs, 2)  xy 偏移量

        # ============================================================
        # Step 3: 时序对齐 - 旋转上一帧 BEV
        # ============================================================
        # 上一帧 BEV 是在上一帧的 lidar 坐标系下生成的
        # 自车在帧间发生了旋转，需要将上一帧 BEV 旋转对齐到当前帧
        if prev_bev is not None:
            # 统一形状: (bev_h*bev_w, bs, C)
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)

            if self.rotate_prev_bev:
                for i in range(bs):
                    # can_bus[-1]: 自车朝向角变化 (yaw 角增量)
                    rotation_angle = img_metas[i]['can_bus'][-1]

                    # 将 BEV 特征 reshape 为 2D 图像格式，方便旋转
                    # (bev_h*bev_w, 1, C) → (C, bev_h, bev_w)
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)

                    # 使用 torchvision 的 rotate 函数旋转 BEV 特征图
                    # 这相当于将 BEV 特征从上一帧的朝向旋转到当前帧的朝向
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)

                    # 恢复原始形状: (C, bev_h, bev_w) → (bev_h*bev_w, 1, C)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # ============================================================
        # Step 4: 注入 can_bus 自车运动信息
        # ============================================================
        # can_bus: 自车状态信号，包含 18 维信息
        #   - 位置 (x, y, z)
        #   - 速度 (vx, vy, vz)
        #   - 加速度 (ax, ay, az)
        #   - 角速度 (wx, wy, wz)
        #   - 朝向角 (yaw)
        #   - 其他车辆状态
        # 通过 MLP 编码为 256 维嵌入，注入到 BEV queries 中
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in img_metas])  # (bs, 18)
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]  # (1, bs, 256)
        bev_queries = bev_queries + can_bus * self.use_can_bus

        # ============================================================
        # Step 5: 准备图像特征 (多尺度 + 视角嵌入 + 层级嵌入)
        # ============================================================
        feat_flatten = []
        spatial_shapes = []

        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)

            # 展平空间维度: (bs, num_cam, C, H, W) → (num_cam, bs, H*W, C)
            feat = feat.flatten(3).permute(1, 0, 3, 2)

            # 添加 camera embeddings: 让模型知道每个特征来自哪个相机
            # cams_embeds 形状 (6, 256)，每个相机一个独立的嵌入
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)

            # 添加 level embeddings: 让模型知道每个特征来自哪个 FPN 层级
            # level_embeds 形状 (4, 256)，每个尺度一个独立的嵌入
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        # 拼接所有尺度的特征
        # 每个尺度: (num_cam, bs, H_i*W_i, C)
        # 拼接后: (num_cam, bs, Σ(H_i*W_i), C)
        feat_flatten = torch.cat(feat_flatten, 2)

        # spatial_shapes: 记录每个尺度的 (H, W)，用于后续可变形注意力
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)

        # level_start_index: 记录每个尺度在拼接后特征中的起始位置
        # 例如: [0, H1*W1, H1*W1+H2*W2, H1*W1+H2*W2+H3*W3]
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # 调整维度顺序: (num_cam, bs, Σ(H*W), C) → (num_cam, Σ(H*W), bs, C)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        # ============================================================
        # Step 6: Encoder 前向传播
        # ============================================================
        # 调用 BEVFormerEncoder，将图像特征转换为 BEV 特征
        # - bev_queries: BEV 查询 (bev_h*bev_w, bs, C)
        # - feat_flatten: 图像特征作为 key/value (num_cam, Σ(H*W), bs, C)
        # - prev_bev: 上一帧 BEV 用于时序融合
        # - shift: 自车位移用于参考点偏移
        bev_embed = self.encoder(
            bev_queries,          # query: BEV 网格点
            feat_flatten,         # key: 图像特征
            feat_flatten,         # value: 图像特征 (与 key 相同)
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,      # BEV 位置编码
            spatial_shapes=spatial_shapes,        # 多尺度特征空间形状
            level_start_index=level_start_index,  # 多尺度特征起始索引
            prev_bev=prev_bev,    # 上一帧 BEV (时序融合)
            shift=shift,          # 自车位移偏移
            img_metas=img_metas,
        )

        return bev_embed

    def get_states_and_refs(
        self,
        bev_embed,              # BEV 特征 (bev_h*bev_w, bs, C)
        object_query_embed,     # object queries (num_query, 2*C)
        bev_h,                  # BEV 高度
        bev_w,                  # BEV 宽度
        reference_points,       # 初始参考点 (num_query, 3)
        reg_branches=None,      # 回归分支 (用于逐层 refine)
        cls_branches=None,      # 分类分支 (用于两阶段)
        img_metas=None          # 图像元信息
    ):
        """获取 Decoder 的输出状态和参考点

        将 object queries 送入 Decoder，在 BEV 特征上进行交叉注意力，
        逐层迭代 refine 参考点位置和特征表示。

        这个方法被 BEVFormerTrackHead.get_detections() 调用，
        用于跟踪任务中的目标检测。

        Args:
            bev_embed: BEV 特征 (bev_h*bev_w, bs, C)
            object_query_embed: object queries (num_query, 2*C)
                前半部分为位置编码，后半部分为内容编码
            bev_h, bev_w: BEV 空间尺寸
            reference_points: 初始 3D 参考点 (num_query, 3)
            reg_branches: 回归分支模块列表
            cls_branches: 分类分支模块列表

        Returns:
            inter_states: Decoder 各层输出 (num_layers, num_query, bs, C)
            init_reference_out: 初始参考点 (bs, num_query, 3)
            inter_references_out: 各层更新后的参考点 (num_layers, bs, num_query, 3)
        """
        bs = bev_embed.shape[1]

        # 拆分 object_query_embed: 前半为位置编码，后半为内容编码
        # object_query_embed: (num_query, 2*C) → query_pos: (num_query, C), query: (num_query, C)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)

        # 扩展 batch 维度
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # (bs, num_query, C)
        query = query.unsqueeze(0).expand(bs, -1, -1)           # (bs, num_query, C)

        # 参考点从 inverse sigmoid 空间转换到 [0, 1] 范围
        reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.sigmoid()

        init_reference_out = reference_points

        # 调整维度顺序为 (num_query, bs, C) 适配 Decoder 的输入格式
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        # Decoder 前向传播
        # query: object queries (num_query, bs, C)
        # value: BEV 特征 (bev_h*bev_w, bs, C)
        # 在 BEV 特征上做交叉注意力，逐层 refine 检测结果
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,          # BEV 特征作为 value
            query_pos=query_pos,      # query 位置编码
            reference_points=reference_points,  # 初始参考点
            reg_branches=reg_branches,          # 回归分支 (逐层 refine)
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            img_metas=img_metas
        )
        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,            # 多尺度图像特征
                bev_queries,           # BEV 查询向量 (bev_h*bev_w, C)
                object_query_embed,    # 目标查询向量 (num_query, 2*C)
                bev_h,                 # BEV 高度
                bev_w,                 # BEV 宽度
                real_h,                # 真实世界高度范围
                real_w,                # 真实世界宽度范围
                grid_length=[0.512, 0.512],  # BEV 网格长度
                bev_pos=None,          # BEV 位置编码
                reg_branches=None,     # 回归分支 (逐层 refine)
                cls_branches=None,     # 分类分支 (两阶段)
                prev_bev=None,         # 上一帧 BEV
                **kwargs):
        """完整的前向传播: Encoder + Decoder

        这是 BEVFormer 检测流程的完整实现:

        Phase 1 - Encoder: 图像特征 → BEV 特征
            1. 准备 BEV queries (注入 can_bus 信息)
            2. 时序对齐 (旋转上一帧 BEV, 计算位移偏移)
            3. 准备图像特征 (添加 camera/level embeddings)
            4. BEVFormerEncoder 前向传播 (TSA + SCA)

        Phase 2 - Decoder: BEV 特征 → 检测结果
            1. 拆分 object queries (位置编码 + 内容编码)
            2. 预测初始参考点 (sigmoid 到 [0,1])
            3. DetectionTransformerDecoder 前向传播
               (自注意力 + 交叉注意力 + 逐层 refine 参考点)

        Args:
            mlvl_feats: 多尺度图像特征列表
                每个元素形状 (bs, num_cams, C, H, W)
            bev_queries: BEV 查询向量 (bev_h*bev_w, C)
            object_query_embed: 目标查询向量 (num_query, 2*C)
                前半部分为位置编码，后半部分为内容编码
            bev_h, bev_w: BEV 网格尺寸
            real_h, real_w: 真实世界范围
            grid_length: BEV 网格实际大小
            bev_pos: BEV 位置编码 (bs, C, bev_h, bev_w)
            reg_branches: 回归分支列表 (每层 decoder 一个)
            cls_branches: 分类分支列表 (两阶段模式)
            prev_bev: 上一帧 BEV 特征 (时序融合)

        Returns:
            bev_embed:              BEV 特征 (bev_h*bev_w, bs, C)
            inter_states:           Decoder 各层输出 (num_layers, num_query, bs, C)
            init_reference_out:     初始参考点 (bs, num_query, 3)
            inter_references_out:   各层更新后的参考点 (num_layers, bs, num_query, 3)
        """

        # ============================================================
        # Phase 1: Encoder - 图像特征 → BEV 特征
        # ============================================================
        # 调用 get_bev_features 执行完整的 BEV 特征生成流程
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            real_h,
            real_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: (bev_h*bev_w, bs, embed_dims)

        # ============================================================
        # Phase 2: Decoder - BEV 特征 → 检测结果
        # ============================================================

        bs = mlvl_feats[0].size(0)

        # 拆分 object_query_embed: 前半为位置编码，后半为内容编码
        # (num_query, 2*C) → query_pos: (num_query, C), query: (num_query, C)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)

        # 扩展 batch 维度
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # (bs, num_query, C)
        query = query.unsqueeze(0).expand(bs, -1, -1)           # (bs, num_query, C)

        # 根据 query 的位置编码预测初始 3D 参考点
        # reference_points: Linear(256 → 3) 预测 (cx, cy, cz)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()  # 归一化到 [0, 1]
        init_reference_out = reference_points

        # 调整维度顺序为 (num_query, bs, C)，适配 Decoder 输入格式
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)  # (bev_h*bev_w, bs, C) → (bs, bev_h*bev_w, C)

        # Decoder 前向传播
        # query: object queries (num_query, bs, C)
        # value: BEV 特征 (bs, bev_h*bev_w, C)
        # 每层 decoder 执行:
        #   1. 自注意力: queries 之间交互，避免重复检测
        #   2. 交叉注意力: queries 在 BEV 特征上采样
        #   3. FFN: 特征变换
        #   4. 回归分支: 更新参考点位置
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out