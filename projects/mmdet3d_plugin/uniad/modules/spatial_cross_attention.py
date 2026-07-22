# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
空间交叉注意力 (Spatial Cross-Attention, SCA)
==============================================
负责将 BEV queries 与多视角图像特征进行交互，是 BEVFormer 的核心注意力机制。

包含两个类:
1. SpatialCrossAttention:    空间交叉注意力的高层封装
   负责将 BEV query 通过投影参考点与对应相机图像的特征交互

2. MSDeformableAttention3D:  3D 可变形注意力的底层实现
   负责在图像特征上以可变形方式采样特征并聚合

SCA 的核心流程:
    BEV Query (3D 空间中的查询点)
        │
        ├── Step 1: 参考点投影
        │   └── 将 3D BEV 参考点通过 lidar2img 矩阵投影到各相机图像平面
        │
        ├── Step 2: 按相机分组
        │   └── 每个 BEV query 只与能"看到"它的相机交互 (节省 GPU 显存)
        │
        ├── Step 3: 可变形注意力
        │   └── 在图像特征上采样参考点周围的区域
        │
        └── Step 4: 多相机聚合
            └── 对多个相机的输出取平均，得到最终的 BEV query 特征
"""

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """空间交叉注意力 (SCA)

    将 BEV 空间中的查询点投影到多视角图像上，在对应图像特征上采样。
    这是 BEVFormer 实现图像特征 → BEV 特征转换的核心机制。

    关键设计:
    1. 按相机分组: 每个 BEV query 只与能"看到"它的相机交互
       这大大减少了计算量，因为大多数 BEV 点只有 2-3 个相机能看到
    2. 多相机特征平均: 对不同相机的输出取平均而非求和
       避免因可见相机数量不同导致的特征尺度不一致

    Args:
        embed_dims: 特征嵌入维度，默认 256
        num_cams: 相机数量，默认 6 (nuScenes 的 6 视角)
        pc_range: 点云范围，用于 3D 参考点投影
        dropout: Dropout 概率
        deformable_attention: 底层可变形注意力模块的配置
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False

        # 底层的可变形注意力模块
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams

        # 输出投影层
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """初始化输出投影层权重"""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """空间交叉注意力前向传播

        核心流程:
        1. 根据 bev_mask 确定每个 BEV query 对哪些相机可见
        2. 按相机重新分组 query 和参考点 (rebatch)
        3. 在每个相机上执行可变形注意力
        4. 将多相机结果聚合回 BEV 空间

        Args:
            query: BEV 查询 (bs, num_query, C)  -- batch_first=True
            key/value: 图像特征 (num_cam, ΣH*W, bs, C)
            reference_points: 3D 参考点 (bs, num_query, ...)
            reference_points_cam: 投影到图像平面的参考点 (num_cam, bs, num_query, D, 2)
            bev_mask: 每个 BEV 点在各相机上的可见性 (num_cam, bs, num_query, D)
            spatial_shapes: 多尺度图像特征的空间形状
            level_start_index: 多尺度特征起始索引

        Returns:
            slots: 更新后的 BEV 特征 (bs, num_query, C)
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)  # 用于累积多相机输出
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        # ============================================================
        # Step 1: 按相机分组 - 确定每个 BEV query 对哪些相机可见
        # ============================================================
        # bev_mask: (num_cam, bs, num_query, D) 其中 D 是每个 pillar 的采样点数
        # 对于每个相机，找出所有能被该相机看到的 BEV queries
        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            # 找到在相机 i 中可见的 BEV query 索引
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # ============================================================
        # Step 2: Rebatch - 按相机重新组织 query 和参考点
        # ============================================================
        # 每个相机只与能"看到"的 BEV queries 交互，大大节省 GPU 显存
        # queries_rebatch: (bs, num_cams, max_len, C)
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        # reference_points_rebatch: (bs, num_cams, max_len, D, 2)
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        # ============================================================
        # Step 3: 可变形注意力 - 在每个相机上采样图像特征
        # ============================================================
        # 将 key/value 从 (num_cam, l, bs, C) 变换为 (bs*num_cams, l, C)
        num_cams, l, bs, embed_dims = key.shape
        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        # 在图像特征上执行可变形注意力
        # query: (bs*num_cams, max_len, C)
        # key/value: (bs*num_cams, l, C)  -- 多尺度图像特征
        # reference_points: (bs*num_cams, max_len, D, 2) -- 投影后的图像坐标
        queries = self.deformable_attention(
            query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims),
            key=key, value=value,
            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)

        # ============================================================
        # Step 4: 多相机聚合 - 将各相机输出累加回 BEV 空间
        # ============================================================
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # 计算每个 BEV query 被多少个相机看到
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)

        # 取平均 (而非求和)，避免可见相机数量不同的影响
        slots = slots / count[..., None]

        # 输出投影
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """3D 可变形注意力模块

    用于空间交叉注意力 (SCA) 的底层实现。
    在图像特征上以可变形方式采样，支持多尺度特征和 3D 参考点投影。

    与 Decoder 中的 CustomMSDeformableAttention 的区别:
    - 这里用于 Encoder 的 SCA (query=BEV, key/value=图像特征)
    - 支持 3D 参考点 (高度维度采样)
    - 每个 BEV query 有 num_Z_anchors 个不同高度的参考点

    3D 参考点采样:
    每个 BEV grid 位置在高度方向上有 num_Z_anchors 个采样点。
    在投影到图像后，每个采样点对应图像上的一个位置。
    总共 num_points * num_Z_anchors 个采样点。

    Args:
        embed_dims: 特征嵌入维度，默认 256
        num_heads: 注意力头数，默认 8
        num_levels: 多尺度特征层数，默认 4 (FPN 的 4 个尺度)
        num_points: 每个 query 在每个 head 的采样点数，默认 8
        im2col_step: CUDA 实现的并行参数
        dropout: Dropout 概率
        batch_first: 是否 batch 维度在第一维
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        # 预测采样偏移量 (每个采样点 2D 偏移)
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)

        # 预测注意力权重
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)

        # value 特征投影
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """初始化权重

        - sampling_offsets: bias 初始化为放射状分布
        - attention_weights: 初始化为 0 (均匀注意力)
        - value_proj 和 output_proj: Xavier 均匀初始化
        """
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """3D 可变形注意力前向传播

        支持 3D 参考点: 每个 BEV query 在高度方向上有 num_Z_anchors 个采样点。
        投影到图像后，每个 3D 参考点对应图像上的一个 2D 位置。

        Args:
            query: 查询向量 (bs, num_query, C)
            key/value: 图像特征 (bs, num_value, C)
            reference_points: 归一化 3D 参考点 (bs, num_query, num_Z_anchors, 2)
            spatial_shapes: 多尺度特征空间形状
            level_start_index: 多尺度特征起始索引

        Returns:
            output: 注意力输出 (bs, num_query, C)
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # 预测采样偏移量: (bs, num_query, num_heads, num_levels, num_points, 2)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        # 预测注意力权重: (bs, num_query, num_heads, num_levels * num_points)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        # ---- 计算采样位置 ----
        # reference_points 形状: (bs, num_query, num_Z_anchors, 2)
        # 其中 num_Z_anchors 是每个 BEV query 在高度方向的采样点数
        if reference_points.shape[-1] == 2:
            """
            对于每个 BEV query，它在 3D 空间中有 num_Z_anchors 个不同高度的采样点。
            投影到每个 2D 图像后，每个采样点对应图像上的一个位置。
            每个参考点周围采样 num_points 个点。
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            # 扩展维度: (bs, num_query, 1, 1, 1, num_Z_anchors, 2)
            reference_points = reference_points[:, :, None, None, None, :, :]

            # 归一化偏移量
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]

            # 将采样偏移量按 num_Z_anchors 分组
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels,
                num_all_points // num_Z_anchors, num_Z_anchors, xy)

            # 采样位置 = 参考点 + 偏移量
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            # 合并 num_points 和 num_Z_anchors 维度
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        # 执行可变形注意力 (CUDA 加速)
        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output