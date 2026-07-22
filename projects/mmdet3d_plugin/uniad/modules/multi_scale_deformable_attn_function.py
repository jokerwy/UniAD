# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
多尺度可变形注意力 CUDA 函数
=============================
这是可变形注意力的底层 CUDA 实现，定义了前向和反向传播的 CUDA kernel。

包含两个类:
1. MultiScaleDeformableAttnFunction_fp16: fp16 精度版本
   标注了 @custom_fwd(cast_inputs=torch.float16)，自动将输入转为 fp16
   但由于 fp16 在多次求和时不稳定，实际使用时通常强制转为 fp32

2. MultiScaleDeformableAttnFunction_fp32: fp32 精度版本
   标注了 @custom_fwd(cast_inputs=torch.float32)，自动将输入转为 fp32
   这是实际使用的版本，因为 fp32 在多次求和时数值更稳定

为什么 fp16 不稳定:
    可变形注意力涉及大量的求和操作:
    - 多个采样点的特征加权求和
    - 多个注意力头的输出合并
    fp16 的有效精度只有 ~3.3 位十进制数，多次求和累积误差大
    因此即使标注了 fp16 版本，代码中也统一使用 fp32 版本

自定义 autograd Function:
    继承 torch.autograd.Function，手动实现 forward 和 backward
    - forward: 调用 CUDA kernel ms_deform_attn_forward
    - backward: 调用 CUDA kernel ms_deform_attn_backward
    这样做可以避免 PyTorch 自动求导在复杂操作上的性能开销
"""

import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class MultiScaleDeformableAttnFunction_fp16(Function):
    """多尺度可变形注意力 - fp16 版本

    继承 torch.autograd.Function，手动实现前向和反向传播。
    使用 @custom_fwd(cast_inputs=torch.float16) 自动将输入转换为 fp16。

    注意: 虽然定义了 fp16 版本，但实际使用中通常强制使用 fp32 版本，
    因为可变形注意力的多次求和操作在 fp16 下数值不稳定。
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """多尺度可变形注意力前向传播 (GPU)

        Args:
            value: 值特征 (bs, num_keys, num_heads, embed_dims//num_heads)
            value_spatial_shapes: 每层特征图的空间形状 (num_levels, 2)
            value_level_start_index: 每层在 value 中的起始索引 (num_levels,)
            sampling_locations: 采样位置 (bs, num_queries, num_heads, num_levels, num_points, 2)
            attention_weights: 注意力权重 (bs, num_queries, num_heads, num_levels, num_points)
            im2col_step: 图像到列的步长

        Returns:
            output: 注意力输出 (bs, num_queries, embed_dims)
        """
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """多尺度可变形注意力反向传播 (GPU)

        Args:
            grad_output: 输出张量的梯度 (bs, num_queries, embed_dims)

        Returns:
            grad_value: value 的梯度
            grad_sampling_loc: 采样位置的梯度
            grad_attn_weight: 注意力权重的梯度
            None: value_spatial_shapes 不需要梯度
            None: value_level_start_index 不需要梯度
            None: im2col_step 不需要梯度
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformableAttnFunction_fp32(Function):
    """多尺度可变形注意力 - fp32 版本

    继承 torch.autograd.Function，手动实现前向和反向传播。
    使用 @custom_fwd(cast_inputs=torch.float32) 自动将输入转换为 fp32。

    这是实际使用的版本，因为 fp32 的数值精度更高，
    在多次求和操作中不会累积显著误差。
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """多尺度可变形注意力前向传播 (GPU) - fp32 版本

        Args:
            value: 值特征 (bs, num_keys, num_heads, embed_dims//num_heads)
            value_spatial_shapes: 每层特征图的空间形状 (num_levels, 2)
            value_level_start_index: 每层在 value 中的起始索引 (num_levels,)
            sampling_locations: 采样位置 (bs, num_queries, num_heads, num_levels, num_points, 2)
            attention_weights: 注意力权重 (bs, num_queries, num_heads, num_levels, num_points)
            im2col_step: 图像到列的步长

        Returns:
            output: 注意力输出 (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """多尺度可变形注意力反向传播 (GPU) - fp32 版本

        Args:
            grad_output: 输出张量的梯度 (bs, num_queries, embed_dims)

        Returns:
            grad_value: value 的梯度
            grad_sampling_loc: 采样位置的梯度
            grad_attn_weight: 注意力权重的梯度
            None: value_spatial_shapes, value_level_start_index, im2col_step 不需要梯度
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None
