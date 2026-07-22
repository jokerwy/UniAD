# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
时序自注意力 (Temporal Self-Attention, TSA)
============================================
负责将当前帧 BEV queries 与上一帧 BEV 特征进行交互，实现时序信息融合。

核心思想:
    当前帧的 BEV query 通过可变形注意力，在上一帧 BEV 特征图上的对应位置采样。
    由于自车在两帧之间发生了运动，需要对上一帧的 BEV 参考点施加位移补偿。

TSA 与 SCA 的对比:
    TSA (时序): query=当前BEV, key/value=上一帧BEV, 在BEV平面上采样
    SCA (空间): query=BEV点,   key/value=图像特征, 投影到图像上采样

时序融合机制:
    1. 将上一帧 BEV 和当前帧 BEV query 拼接 (bs*2, len_bev, C)
    2. 对上一帧 BEV 的参考点施加 shift 偏移 (自车运动补偿)
    3. 当前帧 BEV query 可以关注上一帧 BEV 中的相关空间位置
    4. 融合历史信息和当前帧信息: 对两个 BEV 的输出取平均

Args:
    embed_dims: 特征嵌入维度，默认 256
    num_heads: 注意力头数，默认 8
    num_levels: 多尺度特征层数，TSA 中通常为 1 (只有 BEV 特征)
    num_points: 每个 query 在每个 head 的采样点数，默认 4
    num_bev_queue: BEV 队列长度，默认为 2 (当前帧 + 上一帧)
    im2col_step: CUDA 实现的并行参数
    dropout: Dropout 概率
    batch_first: 是否 batch 维度在第一维
"""

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class TemporalSelfAttention(BaseModule):
    """时序自注意力 (TSA)

    基于可变形注意力实现，当前帧 BEV query 在上一帧 BEV 特征上采样。
    通过自车运动补偿，对齐两帧之间的空间位置关系。

    关键设计:
    1. num_bev_queue=2: 同时处理当前帧和上一帧 BEV
    2. query 由当前帧 query 和上一帧 value 拼接而成
       这样 query 同时包含两帧的信息，可以更好地决定关注哪些位置
    3. 融合方式: 对两帧的输出取平均 (mean)

    与普通可变形注意力的区别:
    - 普通: query 预测采样偏移量，在 value 上采样
    - TSA: query 由 [prev_value, current_query] 拼接而成
           采样偏移量按 num_bev_queue 分组，分别对应两帧

    Args:
        embed_dims: 特征嵌入维度，默认 256
        num_heads: 注意力头数，默认 8
        num_levels: 特征层级数，TSA 中为 1 (只有 BEV 特征)
        num_points: 每个 head 的采样点数，默认 4
        num_bev_queue: BEV 队列长度，默认 2 (当前帧 + 上一帧)
        im2col_step: 图像到列步长
        dropout: Dropout 概率
        batch_first: 是否 batch 维度在第一维
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
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
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
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
        self.num_bev_queue = num_bev_queue  # BEV 队列长度 (当前帧 + 历史帧)

        # sampling_offsets: 输入是 [prev_value, query] 的拼接 (2*C)
        # 输出按 num_bev_queue 分组，分别对应两帧的采样偏移量
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue,
            num_bev_queue*num_heads * num_levels * num_points * 2)

        # attention_weights: 类似地按 num_bev_queue 分组
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)

        # value 投影和输出投影
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """初始化权重

        - sampling_offsets: bias 初始化为放射状分布
           考虑 num_bev_queue 维度，每个 queue 有不同的初始化
        - attention_weights: 初始化为 0
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
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

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
                flag='decoder',
                **kwargs):
        """时序自注意力前向传播

        核心流程:
        1. 构建 value: 将 prev_bev 和当前 query 拼接
        2. 构建 query: 将 prev_value 和当前 query 拼接
        3. 预测采样偏移量 (按 num_bev_queue 分组)
        4. 在 BEV 特征上执行可变形注意力
        5. 融合两帧的输出 (取平均)

        Args:
            query: 当前帧 BEV 查询 (bs, num_query, C)
            key/value: 上一帧 BEV 特征 (bs*2, num_query, C) 或 None
                如果为 None，内部自动构建
            identity: 残差连接的输入
            query_pos: 查询位置编码 (bev_pos)
            reference_points: 2D 参考点 (bs*2, num_query, num_levels, 2)
                前半部分为上一帧参考点 (带 shift 偏移)
                后半部分为当前帧参考点
            spatial_shapes: BEV 特征图空间形状 [[bev_h, bev_w]]
            level_start_index: 特征起始索引 [0]

        Returns:
            output: 融合后的 BEV 特征 (bs, num_query, C)
        """

        # 如果没有提供 value，自动构建: 将 query 复制两份
        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        # ---- 构建 query: 拼接上一帧 value 和当前帧 query ----
        # 这样 query 同时包含两帧的信息
        query = torch.cat([value[:bs], query], -1)  # (bs, num_query, 2*C)

        # value 投影
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        # value reshape: (bs*2, num_value, num_heads, C_per_head)
        value = value.reshape(bs*self.num_bev_queue,
                              num_value, self.num_heads, -1)

        # ---- 预测采样偏移量 ----
        # (bs, num_query, num_heads, num_bev_queue, num_levels, num_points, 2)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels, self.num_points, 2)

        # ---- 预测注意力权重 ----
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        # 重组维度以匹配 CUDA kernel 的输入格式
        # (bs, num_bev_queue, num_query, num_heads, num_levels, num_points)
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads,
                     self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads,
                     self.num_levels, self.num_points, 2)

        # ---- 计算采样位置 ----
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        # ---- 执行可变形注意力 ----
        if torch.cuda.is_available() and value.is_cuda:
            # fp16 不稳定，统一使用 fp32
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

        # ---- 融合两帧输出 ----
        # output shape: (bs*num_bev_queue, num_query, C)
        # → (num_query, C, bs*num_bev_queue) → (num_query, C, bs, num_bev_queue)
        output = output.permute(1, 2, 0)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)

        # 取平均融合两帧的输出
        output = output.mean(-1)

        # (num_query, C, bs) → (bs, num_query, C)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        # 残差连接 + Dropout
        return self.dropout(output) + identity