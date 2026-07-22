# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
检测 Decoder (DetectionTransformerDecoder)
===========================================
负责在 BEV 特征上进行 3D 目标检测。由多层 Transformer Decoder Layer 堆叠而成，
每层包含自注意力和可变形交叉注意力，并逐层迭代 refine 检测框位置。

核心流程 (Decoder 单层):
    Object Query (可学习的目标查询向量)
        │
        ├── 自注意力 (Self-Attention): 300 个 queries 之间交互
        │   └── 避免多个 queries 检测同一个目标
        │
        ├── 交叉注意力 (Cross-Attention): query 在 BEV 特征上采样
        │   └── 通过可变形注意力在参考点周围采样 BEV 特征
        │
        └── 回归分支: 更新参考点位置
            └── 预测的偏移量加到当前参考点上，作为下一层 decoder 的输入

逐层 refine 机制:
    Layer 0: 初始参考点 → 预测偏移 → 更新参考点
    Layer 1: 更新后的参考点 → 预测偏移 → 再次更新
    ...
    Layer 5: 最终参考点 → 最终检测框

CustomMSDeformableAttention:
    底层的可变形注意力实现，支持在 BEV 特征图上灵活采样。
"""

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import mmcv
import cv2 as cv
import copy
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def inverse_sigmoid(x, eps=1e-5):
    """Sigmoid 的反函数

    将 [0, 1] 范围内的值映射回无约束空间。
    用于在 Decoder 层之间传递参考点时，保持数值稳定性。

    inverse_sigmoid(sigmoid(x)) = x

    Args:
        x: 输入值，范围 [0, 1]
        eps: 防止数值溢出的最小值

    Returns:
        inverse sigmoid 变换后的值
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetectionTransformerDecoder(TransformerLayerSequence):
    """检测 Transformer Decoder

    由多层 Decoder Layer 堆叠而成，每层包含:
    1. 自注意力: 300 个 object queries 之间交互，避免检测冲突
    2. 交叉注意力: queries 在 BEV 特征上做可变形注意力
    3. FFN: 特征变换
    4. 回归分支: 逐层 refine 参考点位置 (with_box_refine=True 时)

    逐层 refine 的设计思路:
    - 初始参考点是对目标位置的粗略估计
    - 每层 Decoder 预测一个偏移量，加到当前参考点上
    - 下一层 Decoder 使用更新后的参考点，在更精确的位置采样
    - 经过 6 层迭代，参考点逐渐收敛到目标的真实位置

    Args:
        return_intermediate: 是否返回所有中间层的输出
            True → 返回 (num_layers, num_query, bs, C) 用于逐层监督
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Decoder 前向传播

        逐层处理 object queries，每层:
        1. 在 BEV 特征上做交叉注意力
        2. 通过回归分支预测偏移量，更新参考点

        Args:
            query: object queries (num_query, bs, C)
            reference_points: 初始 3D 参考点 (bs, num_query, 3)
                在 [0, 1] 范围内，分别表示归一化的 cx, cy, cz
            reg_branches: 回归分支模块列表 (每层 decoder 一个)
                用于预测参考点的偏移量
            key_padding_mask: key 的 padding 掩码

        Returns:
            output: Decoder 各层输出
                (num_layers, num_query, bs, C) 如果 return_intermediate=True
            reference_points: 各层更新后的参考点
                (num_layers, bs, num_query, 3) 如果 return_intermediate=True
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # 取参考点的前 2 维 (cx, cy) 作为 2D 参考点
            # (BS, NUM_QUERY, 3) → (BS, NUM_QUERY, 1, 2)
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)

            # Decoder 层前向传播: 自注意力 + 交叉注意力 + FFN
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            # 逐层 refine 参考点: 每层预测偏移量并更新参考点
            if reg_branches is not None:
                # 回归分支预测偏移量: (bs, num_query, 10)
                # tmp[0:2] = Δcx, Δcy → 用于更新参考点
                # tmp[4:5] = Δcz → 用于更新参考点高度
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                # 计算新的参考点:
                # new_cx = sigmoid(Δcx + inverse_sigmoid(old_cx))
                # new_cy = sigmoid(Δcy + inverse_sigmoid(old_cy))
                # new_cz = sigmoid(Δcz + inverse_sigmoid(old_cz))
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                # detach: 参考点更新不参与梯度传播到上一层
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class CustomMSDeformableAttention(BaseModule):
    """Decoder 中使用的可变形注意力模块

    基于 Deformable DETR 的可变形注意力机制实现。
    与 Encoder 中的交叉注意力不同，这里的 query 是 object queries，
    key/value 是 BEV 特征。

    工作原理:
    1. 对于每个 query，根据其参考点在 BEV 特征上采样 num_points 个点
    2. 通过 sampling_offsets 预测每个采样点的偏移量
    3. 通过 attention_weights 对采样点做加权聚合
    4. 输出投影后与残差相加

    这允许每个 object query 灵活地关注 BEV 特征中与目标相关的区域，
    而不局限于固定的网格采样。

    Args:
        embed_dims: 特征嵌入维度，默认 256
        num_heads: 注意力头数，默认 8
        num_levels: 多尺度特征层数，Decoder 中通常为 1 (只有 BEV 特征)
        num_points: 每个 query 在每个 head 的采样点数，默认 4
        im2col_step: 图像到列的步长，影响 CUDA 实现的并行度
        dropout: Dropout 概率
        batch_first: 是否 batch 维度在第一维
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
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

        # 每个注意力头的维度需要是 2 的幂次，CUDA 实现更高效
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

        # sampling_offsets: 预测每个采样点相对于参考点的偏移量
        # 输出维度: num_heads * num_levels * num_points * 2 (每个点 2D 偏移)
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)

        # attention_weights: 预测每个采样点的注意力权重
        # 输出维度: num_heads * num_levels * num_points
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)

        # value_proj: 将 value 特征投影到注意力空间
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        # output_proj: 将注意力输出投影回原始空间
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """初始化权重

        - sampling_offsets: bias 初始化为放射状分布 (不同 head 朝向不同角度)
            这样初始时每个注意力头关注参考点周围不同方向的特征
        - attention_weights: 初始化为 0 (经过 softmax 后变为均匀注意力)
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
                flag='decoder',
                **kwargs):
        """可变形注意力前向传播

        核心流程:
        1. 根据 query 预测采样偏移量和注意力权重
        2. 在参考点周围采样 BEV 特征
        3. 加权聚合采样特征
        4. 输出投影 + 残差连接

        Args:
            query: 查询向量 (num_query, bs, C) 或 (bs, num_query, C)
            key/value: BEV 特征 (num_bev, bs, C) 或 (bs, num_bev, C)
            identity: 残差连接的输入
            query_pos: 查询位置编码
            reference_points: 归一化参考点 (bs, num_query, num_levels, 2)
            spatial_shapes: BEV 特征图的空间形状
            level_start_index: 多尺度特征起始索引

        Returns:
            output: 注意力输出 (num_query, bs, C) 或 (bs, num_query, C)
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

        # 将 value 投影到多头注意力空间
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

        # 计算实际采样位置: 参考点 + 偏移量
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

        # 执行可变形注意力 (CUDA 加速或 PyTorch 实现)
        if torch.cuda.is_available() and value.is_cuda:
            # fp16 的可变形注意力不稳定 (多次求和累积误差)，统一使用 fp32
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

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        # 残差连接 + Dropout
        return self.dropout(output) + identity