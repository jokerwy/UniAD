#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
占用预测头部（OccHead）插件模块
===============================

本模块定义了占用预测（Occupancy Prediction）任务中使用的各种神经网络组件，包括：

1. **BevFeatureSlicer**：从较大的BEV（鸟瞰图）特征图中采样出较小感受野的区域，
   用于在不同分辨率的BEV特征之间进行对齐和裁剪。
2. **MLP**：简单的多层感知机，用于特征变换。
3. **SimpleConv2d**：轻量级2D卷积模块，用于特征图的通道变换。
4. **CVT_DecoderBlock / CVT_Decoder**：基于CenterNet风格的跨视图变换解码器，
   用于将低分辨率特征逐步上采样到高分辨率，同时融合跳跃连接。
5. **UpsamplingAdd**：上采样后与跳跃连接相加的模块。
6. **Interpolate**：封装了插值上采样操作。
7. **Bottleneck**：带残差连接的瓶颈模块，支持上采样、下采样和空洞卷积。

这些模块共同构成了占用预测头部的特征解码和变换流水线。
"""

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from .utils import calculate_birds_eye_view_parameters
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule, build_conv_layer
from einops import rearrange
from collections import OrderedDict


# =============================================================================
# Grid sampler（网格采样器）
# =============================================================================
# 从较大的BEV特征图中采样出较小感受野的区域，用于不同分辨率BEV特征之间的对齐。
class BevFeatureSlicer(nn.Module):
    """BEV特征切片器：从较大的BEV特征图中采样出较小感受野的区域。

    该模块用于在不同网格分辨率之间进行BEV特征的对齐。例如，当占用预测头部的
    网格配置与地图头部的网格配置不一致时，需要将较大的BEV特征图裁剪/采样
    到较小的目标区域。

    工作原理：
    - 当源网格配置与目标网格配置相同时，直接透传（identity mapping）。
    - 当两者不同时，通过计算目标网格在源网格中的归一化坐标，使用
      grid_sample 进行双线性插值采样。

    参数：
        grid_conf (dict): 源BEV网格的配置字典，包含 'xbound', 'ybound', 'zbound' 键。
        map_grid_conf (dict): 目标（地图）BEV网格的配置字典，格式同上。
    """

    def __init__(self, grid_conf, map_grid_conf):
        """初始化BEV特征切片器。

        如果源网格配置与目标网格配置相同，则设置为恒等映射（直接透传）；
        否则，计算目标网格在源网格中的归一化坐标，用于后续的grid_sample采样。

        参数：
            grid_conf (dict): 源BEV网格配置，包含 xbound/ybound/zbound，
                              每个为 [start, end, step] 格式。
            map_grid_conf (dict): 目标BEV网格配置，格式同上。
        """
        super().__init__()
        if grid_conf == map_grid_conf:
            # 如果两个网格配置相同，直接使用恒等映射，节省计算
            self.identity_mapping = True
        else:
            self.identity_mapping = False

            # 计算源BEV特征图的参数：分辨率、起始位置、维度
            bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
                grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
            )

            # 计算目标BEV特征图的参数：分辨率、起始位置、维度
            map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound']
            )

            # 目标网格在x方向上的物理坐标（从起始位置到边界的等间距网格）
            self.map_x = torch.arange(
                map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])

            # 目标网格在y方向上的物理坐标
            self.map_y = torch.arange(
                map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

            # 将物理坐标归一化到[-1, 1]范围，以便torch.grid_sample使用
            # 归一化因子是源BEV的半宽度/半高度（取负的起始位置，因为起始位置通常为负值）
            self.norm_map_x = self.map_x / (- bev_start_position[0])
            self.norm_map_y = self.map_y / (- bev_start_position[1])

            # 生成2D网格坐标（xy模式），形状为 (H, W, 2)
            tmp_m, tmp_n = torch.meshgrid(
                self.norm_map_x, self.norm_map_y)  # indexing 'ij' 模式
            tmp_m, tmp_n = tmp_m.T, tmp_n.T  # 转换为 'xy' 模式，使形状为 (H, W)
            self.map_grid = torch.stack([tmp_m, tmp_n], dim=2)  # (H, W, 2)

    def forward(self, x):
        """前向传播：从源BEV特征图中采样目标网格区域的特征。

        参数：
            x (torch.Tensor): 源BEV特征图，形状为 (B, C, H, W)。

        返回：
            torch.Tensor: 采样后的特征图，形状为 (B, C, H_target, W_target)。
                          如果为恒等映射，则直接返回输入。
        """
        if self.identity_mapping:
            # 网格配置相同，无需采样，直接返回
            return x
        else:
            # 将网格坐标扩展到batch维度，形状为 (B, H_target, W_target, 2)
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)  # (B, H, W, 2)

            # 使用双线性插值从源特征图中采样，align_corners=True 确保角点对齐
            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)


# =============================================================================
# General layers（通用网络层）
# =============================================================================

class MLP(nn.Module):
    """非常简单的多层感知机（也称前馈网络FFN）。

    由多个全连接层串联而成，除最后一层外均使用ReLU激活函数。

    参数：
        input_dim (int): 输入特征维度。
        hidden_dim (int): 隐藏层特征维度（所有隐藏层统一使用此维度）。
        output_dim (int): 输出特征维度。
        num_layers (int): 总层数（包括输入层和输出层）。
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """初始化MLP。

        构建一个由 `num_layers` 个全连接层组成的网络，其中前 `num_layers-1` 层
        使用 hidden_dim 作为输出维度，最后一层输出维度为 output_dim。

        参数：
            input_dim (int): 输入特征维度。
            hidden_dim (int): 隐藏层统一维度。
            output_dim (int): 输出特征维度。
            num_layers (int): 网络层数（至少为1）。
        """
        super().__init__()
        self.num_layers = num_layers
        # 构建隐藏层维度列表：例如 num_layers=3 时，h = [hidden_dim, hidden_dim]
        h = [hidden_dim] * (num_layers - 1)
        # 串联所有全连接层：输入维度依次为 [input_dim] + h，输出维度依次为 h + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """前向传播。

        除最后一层外，每层全连接后接ReLU激活函数；最后一层仅做线性变换。

        参数：
            x (torch.Tensor): 输入张量，形状为 (..., input_dim)。

        返回：
            torch.Tensor: 输出张量，形状为 (..., output_dim)。
        """
        for i, layer in enumerate(self.layers):
            # 前 num_layers-1 层使用 ReLU 激活，最后一层不使用激活函数
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SimpleConv2d(BaseModule):
    """简单的2D卷积模块。

    由多个 ConvModule（卷积+归一化+激活）组成，最后一层为纯卷积层（无归一化和激活）。
    用于对特征图进行通道维度上的变换，同时保持空间分辨率不变。

    参数：
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        conv_channels (int): 中间卷积层的通道数，默认为64。
        num_conv (int): 卷积层总数，默认为1（即仅一个1x1卷积层）。
        conv_cfg (dict): 卷积层配置，默认为 {'type': 'Conv2d'}。
        norm_cfg (dict): 归一化层配置，默认为 {'type': 'BN2d'}。
        bias (str): 偏置配置，'auto' 表示自动决定。
        init_cfg (dict): 初始化配置，不允许外部设置（内部使用Kaiming初始化）。
    """

    def __init__(self, in_channels,
                       out_channels,
                       conv_channels=64,
                       num_conv=1,
                       conv_cfg=dict(type='Conv2d'),
                       norm_cfg=dict(type='BN2d'),
                       bias='auto',
                       init_cfg=None,
                       ):
        # 禁止外部设置 init_cfg，防止异常的初始化行为
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels
        # 当只有一层卷积时，中间的卷积通道数直接使用输入通道数
        if num_conv == 1:
            conv_channels = in_channels

        conv_layers = []
        c_in = in_channels
        # 构建前 num_conv-1 个卷积模块（带BN和ReLU），使用3x3卷积保持空间分辨率
        for i in range(num_conv - 1):
            conv_layers.append(
                ConvModule(
                    c_in,
                    conv_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,       # padding=1 保持空间分辨率不变
                    bias=bias,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                )
            )
            c_in = conv_channels
        # 最后一层：纯卷积层（无归一化和激活函数），使用1x1卷积进行通道变换
        conv_layers.append(
            build_conv_layer(
                conv_cfg,
                conv_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        )
        self.conv_layers = nn.Sequential(*conv_layers)

        # 如果未指定初始化配置，默认使用Kaiming初始化
        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        """前向传播。

        对输入特征图进行卷积变换，保持空间分辨率 (H, W) 不变，
        仅改变通道数。

        参数：
            x (torch.Tensor): 输入特征图，形状为 (B, C_in, H, W)。

        返回：
            torch.Tensor: 输出特征图，形状为 (B, out_channels, H, W)。
        """
        b, c_in, h_in, w_in = x.size()
        out = self.conv_layers(x)
        # 完整性检查：确保输出空间分辨率与输入一致
        assert out.size() == (b, self.out_channels, h_in, w_in)  # sanity check
        return out


# =============================================================================
# Decoder（解码器）：基于CenterNet风格的CVT解码器
# =============================================================================
# 用于将低分辨率特征逐步上采样到高分辨率，同时融合跳跃连接中的高分辨率细节。

class CVT_DecoderBlock(nn.Module):
    """跨视图变换（CVT）解码器块。

    这是CVT解码器的基本构建块，对输入特征进行上采样和通道变换，
    并可选择性地与跳跃连接（skip connection）相加融合。

    参数：
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        skip_dim (int): 跳跃连接特征的通道数。
        residual (bool): 是否使用残差连接（跳跃连接融合）。
        factor (int): 中间通道的缩减因子，中间通道数 = out_channels // factor。
        upsample (bool): 是否对输入进行2倍上采样。
        with_relu (bool): 是否在输出前应用ReLU激活函数。
    """

    def __init__(self, in_channels, out_channels, skip_dim, residual, factor, upsample, with_relu=True):
        """初始化CVT解码器块。

        构建卷积序列：根据 `upsample` 参数决定是否先进行2倍上采样，
        然后使用3x3卷积（降通道）+ 1x1卷积（升通道到out_channels），
        每层卷积后接BatchNorm。

        参数：
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            skip_dim (int): 跳跃连接来源的通道数（通常等于基础维度dim）。
            residual (bool): 是否启用残差跳跃连接融合。
            factor (int): 瓶颈中间通道的缩减因子。
            upsample (bool): 是否包含2倍上采样。
            with_relu (bool): 输出前是否使用ReLU激活。
        """
        super().__init__()

        dim = out_channels // factor  # 中间瓶颈通道数

        if upsample:
            # 上采样分支：先2倍上采样，再3x3卷积降通道，最后1x1卷积升通道
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            # 不上采样分支：直接3x3卷积降通道，再1x1卷积升通道
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels))

        # 残差连接：将跳跃连接特征的通道数变换到与输出通道数一致
        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.with_relu = with_relu
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        """前向传播：对输入特征进行变换并与跳跃连接融合。

        参数：
            x (torch.Tensor): 输入特征图，形状为 (B, C_in, H, W)。
            skip (torch.Tensor): 跳跃连接特征图，形状为 (B, skip_dim, H_skip, W_skip)。

        返回：
            torch.Tensor: 输出特征图，形状为 (B, out_channels, H_out, W_out)。
        """
        # 主通路：卷积变换（可能包含上采样）
        x = self.conv(x)

        if self.up is not None:
            # 跳跃连接通路：1x1卷积变换通道数，然后插值到与主通路相同的空间尺寸
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])
            # 残差相加融合
            x = x + up

        if self.with_relu:
            return self.relu(x)
        return x


class CVT_Decoder(BaseModule):
    """跨视图变换（CVT）解码器。

    由多个 CVT_DecoderBlock 串联而成，逐步将低分辨率特征图解码到高分辨率。
    每个块接收当前特征和原始跳跃连接特征作为输入，通过残差融合保留细节信息。

    该解码器设计用于处理时序BEV特征：
    - 输入形状为 (B, T, C, H, W)，其中T是时间维度。
    - 内部将 (B, T) 合并为 batch 维度进行处理，处理完成后再拆分回原始形状。

    参数：
        dim (int): 基础特征维度（跳跃连接特征的通道数）。
        blocks (list[int]): 每个解码器块的输出通道数列表。
        residual (bool): 是否使用残差跳跃连接，默认为True。
        factor (int): 瓶颈通道的缩减因子，默认为2。
        upsample (bool): 是否在每个块中进行上采样，默认为True。
        use_checkpoint (bool): 是否使用梯度检查点以节省显存，默认为False。
        init_cfg (dict): 初始化配置，不允许外部设置。
    """

    def __init__(self, dim, blocks, residual=True, factor=2, upsample=True, use_checkpoint=False, init_cfg=None):
        # 禁止外部设置 init_cfg
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        layers = []
        channels = dim  # 初始输入通道数等于基础维度

        for i, out_channels in enumerate(blocks):
            # 最后一个块不使用ReLU激活（with_relu=False），让输出值域更灵活
            with_relu = i < len(blocks) - 1
            layer = CVT_DecoderBlock(channels, out_channels, dim, residual, factor, upsample, with_relu=with_relu)
            layers.append(layer)
            channels = out_channels  # 当前块的输出通道数作为下一个块的输入通道数

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels  # 最终输出通道数
        self.use_checkpoint = use_checkpoint

        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        """前向传播：将时序BEV特征解码到高分辨率。

        参数：
            x (torch.Tensor): 输入特征图，形状为 (B, T, C, H, W)。

        返回：
            torch.Tensor: 解码后的特征图，形状为 (B, T, out_channels, H_out, W_out)。
        """
        b, t = x.size(0), x.size(1)
        # 将 batch 和时间维度合并，便于2D卷积处理
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        y = x  # y 用作跳跃连接特征，保持原始分辨率
        for layer in self.layers:
            if self.use_checkpoint:
                # 使用梯度检查点以节省显存（以计算换内存）
                y = checkpoint.checkpoint(layer, y, x)
            else:
                y = layer(y, x)

        # 恢复 batch 和时间维度
        y = rearrange(y, '(b t) c h w -> b t c h w', b=b, t=t)
        return y


# =============================================================================
# Conv modules（卷积模块）
# =============================================================================

class UpsamplingAdd(nn.Module):
    """上采样加法模块。

    对输入特征进行上采样（默认2倍），然后通过1x1卷积变换通道数，
    最后与跳跃连接特征逐元素相加。

    参数：
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数（也即跳跃连接特征的通道数）。
        scale_factor (int): 上采样倍率，默认为2。
    """

    def __init__(self, in_channels, out_channels, scale_factor=2):
        """初始化上采样加法模块。

        构建上采样层序列：双线性插值上采样 + 1x1卷积通道变换 + BatchNorm。

        参数：
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            scale_factor (int): 上采样倍率。
        """
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        """前向传播：上采样后与跳跃连接相加。

        参数：
            x (torch.Tensor): 待上采样的特征图，形状为 (B, C_in, H, W)。
            x_skip (torch.Tensor): 跳跃连接特征图，形状为 (B, C_out, 2*H, 2*W)。

        返回：
            torch.Tensor: 融合后的特征图，形状为 (B, C_out, 2*H, 2*W)。
        """
        x = self.upsample_layer(x)
        return x + x_skip


class Interpolate(nn.Module):
    """插值上采样模块。

    将 torch.nn.functional.interpolate 封装为 nn.Module，方便集成到 nn.Sequential 中。

    参数：
        scale_factor (int): 上采样倍率，默认为2。
    """

    def __init__(self, scale_factor: int = 2):
        """初始化插值上采样模块。

        参数：
            scale_factor (int): 上采样倍率（空间尺寸放大倍数）。
        """
        super().__init__()
        self._interpolate = nn.functional.interpolate
        self._scale_factor = scale_factor

    def forward(self, x):
        """前向传播：对输入进行双线性插值上采样。

        参数：
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)。

        返回：
            torch.Tensor: 上采样后的特征图，形状为 (B, C, scale_factor*H, scale_factor*W)。
        """
        return self._interpolate(x, scale_factor=self._scale_factor, mode='bilinear', align_corners=False)


class Bottleneck(nn.Module):
    """带残差连接的瓶颈模块。

    该模块实现了经典的瓶颈残差块结构，包含：
    1. 1x1卷积降通道（in_channels -> bottleneck_channels）
    2. 3x3卷积（可选上采样、下采样或空洞卷积）
    3. 1x1卷积升通道（bottleneck_channels -> out_channels）
    4. 残差跳跃连接（当输入输出形状不匹配时，通过投影层对齐）

    参数：
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数，默认为None（等于in_channels）。
        kernel_size (int): 中间卷积的核大小，默认为3。
        dilation (int): 空洞卷积的膨胀率，默认为1（当前仅支持dilation=1）。
        groups (int): 分组卷积的组数，默认为1。
        upsample (bool): 是否进行上采样（使用转置卷积）。
        downsample (bool): 是否进行下采样（使用步长为2的卷积）。
        dropout (float): Dropout概率，默认为0.0。
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        groups=1,
        upsample=False,
        downsample=False,
        dropout=0.0,
    ):
        """初始化瓶颈模块。

        根据配置构建主通路（瓶颈卷积序列）和残差跳跃连接（投影层）。

        参数：
            in_channels (int): 输入通道数。
            out_channels (int, optional): 输出通道数，默认等于in_channels。
            kernel_size (int): 中间卷积核大小。
            dilation (int): 空洞卷积膨胀率（当前仅支持=1）。
            groups (int): 分组卷积组数。
            upsample (bool): 是否上采样。
            downsample (bool): 是否下采样。
            dropout (float): Dropout概率。
        """
        super().__init__()
        self._downsample = downsample
        # 瓶颈通道数 = 输入通道数的一半
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        # 计算padding使输出尺寸与输入一致（步长为1时）
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        # 根据上采样/下采样/普通模式选择中间卷积类型
        assert dilation == 1
        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            # 上采样：使用转置卷积（ConvTranspose2d），步长为2
            bottleneck_conv = nn.ConvTranspose2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=1,
                stride=2,
                output_padding=padding_size,
                padding=padding_size,
                groups=groups,
            )
        elif downsample:
            # 下采样：使用步长为2的普通卷积
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                stride=2,
                padding=padding_size,
                groups=groups,
            )
        else:
            # 普通模式：步长为1的卷积，保持空间分辨率
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                padding=padding_size,
                groups=groups,
            )

        # 主通路：瓶颈结构
        # 1x1降通道 -> BN+ReLU -> 3x3卷积 -> BN+ReLU -> 1x1升通道 -> BN+ReLU -> Dropout
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    # 第一步：1x1卷积降通道（投影层）
                    ('conv_down_project', nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)),
                    ('abn_down_project', nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
                                                       nn.ReLU(inplace=True))),
                    # 第二步：3x3卷积（可选上采样/下采样）
                    ('conv', bottleneck_conv),
                    ('abn', nn.Sequential(nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True))),
                    # 第三步：1x1卷积升通道（投影层）
                    ('conv_up_project', nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)),
                    ('abn_up_project', nn.Sequential(nn.BatchNorm2d(out_channels),
                                                     nn.ReLU(inplace=True))),
                    # 正则化：Dropout
                    ('dropout', nn.Dropout2d(p=dropout)),
                ]
            )
        )

        # 残差跳跃连接：当输入输出形状不匹配时，需要投影层对齐
        if out_channels == in_channels and not downsample and not upsample:
            # 形状完全匹配，无需投影层，直接恒等相加
            self.projection = None
        else:
            # 构建投影层：上采样/下采样 + 1x1卷积 + BatchNorm
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': Interpolate(scale_factor=2)})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update(
                {
                    'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    'bn_skip_proj': nn.BatchNorm2d(out_channels),
                }
            )
            self.projection = nn.Sequential(projection)

    def forward(self, *args):
        """前向传播：瓶颈变换 + 残差连接。

        参数：
            x (torch.Tensor): 输入特征图，形状为 (B, C_in, H, W)。

        返回：
            torch.Tensor: 输出特征图，形状为 (B, C_out, H_out, W_out)。
        """
        (x,) = args
        # 主通路：瓶颈变换
        x_residual = self.layers(x)

        if self.projection is not None:
            if self._downsample:
                # 下采样时，如果输入空间尺寸为奇数，需要padding以匹配残差分支的输出尺寸
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            # 残差通路：投影变换 + 相加
            return x_residual + self.projection(x)
        # 形状匹配，直接恒等残差相加
        return x_residual + x