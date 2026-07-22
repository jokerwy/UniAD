#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
占用预测头部（OccHead）工具函数模块
===================================

本模块提供了占用预测（Occupancy Prediction）任务中使用的各种工具函数，包括：

1. **BEV参数计算**：
   - `calculate_birds_eye_view_parameters`：根据网格边界配置计算BEV（鸟瞰图）的
     分辨率、起始位置和空间维度。
   - `gen_dx_bx`：与上述函数功能相同，但使用不同的变量命名风格（dx, bx, nx），
     用于兼容不同的上游代码。

2. **实例分割工具**：
   - `update_instance_ids`：将实例分割掩码中的旧实例ID替换为新ID。
   - `make_instance_seg_consecutive`：将实例分割掩码中的实例ID重新映射为连续的
     编号（从0开始）。
   - `predict_instance_segmentation_and_trajectories`：将前景掩码和实例sigmoid
     预测结果合并为最终的实例分割掩码。

这些工具函数在占用预测头部的数据预处理、后处理和评估过程中被广泛使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# BEV参数计算（鸟瞰图参数计算）
# =============================================================================
# 这些函数根据网格边界配置（xbound, ybound, zbound）计算BEV特征图的关键参数。
# 每个bound配置为 [start, end, step] 格式，表示该方向上的起始坐标、结束坐标和步长。

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """根据三维空间边界配置计算BEV（鸟瞰图）的关键参数。

    该函数将三个维度（x: 前向, y: 侧向, z: 高度）的边界配置
    转换为BEV特征图的分辨率、起始位置和空间维度。

    参数：
        x_bounds (list): x方向（自车前向）的边界配置，格式为 [start, end, step]。
                         例如 [-50.0, 50.0, 0.5] 表示从-50m到50m，每0.5m一个网格。
        y_bounds (list): y方向（侧向）的边界配置，格式同上。
                         例如 [-50.0, 50.0, 0.5]。
        z_bounds (list): z方向（高度）的边界配置，格式同上。
                         例如 [-5.0, 3.0, 0.5]。

    返回：
        tuple:
            - bev_resolution (torch.Tensor): BEV每个方向的分辨率（网格大小），
              形状为 (3,)，对应 [x_res, y_res, z_res]。
            - bev_start_position (torch.Tensor): BEV每个方向的起始位置（第一个网格中心坐标），
              形状为 (3,)，对应 [x_start, y_start, z_start]。
              计算公式：start + step/2。
            - bev_dimension (torch.Tensor): BEV每个方向的网格数量（空间维度），
              形状为 (3,)，对应 [nx, ny, nz]，类型为 long。
              计算公式：(end - start) / step。

    使用示例：
        >>> res, start, dim = calculate_birds_eye_view_parameters(
        ...     [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-5.0, 3.0, 0.5])
        >>> # res = [0.5, 0.5, 0.5]
        >>> # start = [-49.75, -49.75, -4.75]
        >>> # dim = [200, 200, 16]
    """
    # 分辨率 = 每个方向的步长（网格大小）
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])

    # 起始位置 = 起始坐标 + 半个步长（网格中心位置）
    # 这是为了将网格坐标对齐到体素中心
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])

    # 空间维度 = (结束坐标 - 起始坐标) / 步长，即每个方向上的网格数量
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


def gen_dx_bx(xbound, ybound, zbound):
    """生成BEV特征图的网格参数（dx, bx, nx）。

    与 `calculate_birds_eye_view_parameters` 功能相同，但使用不同的变量命名：
    - dx: 网格分辨率（对应 bev_resolution）
    - bx: 起始位置（对应 bev_start_position）
    - nx: 网格数量（对应 bev_dimension）

    此函数主要用于兼容使用该命名约定的上游代码（如LSS等）。

    参数：
        xbound (list): x方向的边界配置，格式为 [start, end, step]。
        ybound (list): y方向的边界配置，格式为 [start, end, step]。
        zbound (list): z方向的边界配置，格式为 [start, end, step]。

    返回：
        tuple:
            - dx (torch.Tensor): 网格分辨率，形状为 (3,)，类型为 float32。
            - bx (torch.Tensor): 起始位置（网格中心），形状为 (3,)，类型为 float32。
            - nx (torch.Tensor): 网格数量，形状为 (3,)，类型为 int64。
    """
    # dx: 网格分辨率 = 步长
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    # bx: 起始位置 = 起始坐标 + 半个步长（体素中心对齐）
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    # nx: 网格数量 = (结束 - 起始) / 步长
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


# =============================================================================
# 实例分割工具函数
# =============================================================================
# 这些函数用于处理实例分割掩码，包括ID映射、连续化和预测生成。

def update_instance_ids(instance_seg, old_ids, new_ids):
    """将实例分割掩码中的旧实例ID替换为新ID。

    该函数通过构建一个索引映射表，高效地将实例分割掩码中的指定旧ID
    替换为对应的新ID。映射表的大小由最大旧ID决定，对于超出范围的ID
    保持原值不变。

    参数：
        instance_seg (torch.Tensor): 实例分割掩码，形状任意。
                                     值为实例ID，通常0表示背景。
        old_ids (torch.Tensor): 需要替换的旧ID列表，1D张量。
                                必须确保所有old_ids都存在于instance_seg中。
        new_ids (torch.Tensor): 替换后的新ID列表，1D张量，与old_ids对齐。

    返回：
        torch.Tensor: 替换后的实例分割掩码，形状与输入相同，类型为 long。

    使用示例：
        >>> seg = torch.tensor([0, 1, 2, 1, 3])
        >>> update_instance_ids(seg, torch.tensor([1, 3]), torch.tensor([10, 30]))
        tensor([0, 10, 2, 10, 30])
    """
    # 构建索引映射表：大小为最大旧ID+1，初始化为0到max_id的连续值
    indices = torch.arange(old_ids.max() + 1, device=instance_seg.device)
    # 将需要替换的旧ID位置更新为新ID
    for old_id, new_id in zip(old_ids, new_ids):
        indices[old_id] = new_id

    # 使用映射表进行索引替换，转换为long类型
    return indices[instance_seg].long()


def make_instance_seg_consecutive(instance_seg):
    """将实例分割掩码中的实例ID重新映射为连续编号。

    在处理实例分割时，实例ID可能不是连续的（例如存在间隙），
    该函数将实例ID重新映射为从0开始的连续编号，便于后续处理。

    注意：背景（ID=0）也会被包含在映射中，因此映射后的ID从0开始
    且连续。

    参数：
        instance_seg (torch.Tensor): 实例分割掩码，形状任意。

    返回：
        torch.Tensor: 实例ID连续化后的分割掩码，形状与输入相同。

    使用示例：
        >>> seg = torch.tensor([0, 5, 5, 0, 3, 3])
        >>> make_instance_seg_consecutive(seg)
        tensor([0, 1, 1, 0, 2, 2])
    """
    # 找到所有唯一的实例ID（包括背景0）
    unique_ids = torch.unique(instance_seg)  # include background
    # 生成从0开始的连续新ID
    new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
    # 执行ID替换
    instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
    return instance_seg


def predict_instance_segmentation_and_trajectories(
    foreground_masks,
    ins_sigmoid,
    vehicles_id=1,
):
    """从前景掩码和实例预测中生成最终的实例分割掩码。

    该函数是实例分割预测的关键后处理步骤，执行以下操作：
    1. 从前景掩码中提取指定类别（默认车辆，vehicles_id=1）的区域。
    2. 从实例sigmoid预测中取argmax得到每个像素的实例归属。
    3. 将前景掩码与实例预测结合，生成带背景的实例分割。
    4. 将实例ID重映射为连续编号。

    参数：
        foreground_masks (torch.Tensor): 前景语义掩码，形状为 (B, T, H, W)
                                          或 (B, T, 1, H, W)。
                                          值为语义类别ID，其中 vehicles_id
                                          表示车辆类别。
        ins_sigmoid (torch.Tensor): 实例预测的sigmoid输出，形状为
                                    (B, N_instances, T, H, W)。
                                    每个通道对应一个潜在实例的归属概率。
        vehicles_id (int): 车辆类别的语义ID，默认为1。

    返回：
        torch.Tensor: 最终的实例分割掩码，形状为 (B, T, H, W)，类型为 long。
                      值为0表示背景，>=1表示连续的实例ID。

    处理流程：
        (1) 前景掩码筛选：提取 vehicles_id 类别的区域
        (2) 实例argmax：确定每个像素属于哪个实例
        (3) 掩码融合：将背景像素设为0，前景像素使用实例ID
        (4) ID连续化：将实例ID映射为连续编号
    """
    # 步骤1: 处理前景掩码的维度
    # 如果输入是5维 (B, T, 1, H, W)，则压缩掉第3维（通道维）
    if foreground_masks.dim() == 5 and foreground_masks.shape[2] == 1:
        foreground_masks = foreground_masks.squeeze(2)  # [B, T, H, W]

    # 提取车辆类别的前景区域：布尔掩码，形状为 (B, T, H, W)
    # True 表示该像素属于车辆类别
    foreground_masks = foreground_masks == vehicles_id  # [B, T, H, W]

    # 步骤2: 从实例预测中取argmax
    # ins_sigmoid 形状为 (B, N_instances, T, H, W)
    # argmax 在 dim=1（实例维度）上取最大值索引，得到每个像素最可能属于的实例ID
    # 结果形状为 (B, T, H, W)，实例ID从0开始
    argmax_ins = ins_sigmoid.argmax(dim=1)  # long, [B, T, H, W], ins_id从0开始

    # 步骤3: 实例ID偏移+1，将0留给背景
    # 此时 argmax_ins 中 0 表示第一个实例，+1后变为 1 表示第一个实例
    argmax_ins = argmax_ins + 1  # [B, T, H, W], ins_id从1开始

    # 步骤4: 融合前景掩码和实例预测
    # 前景像素（车辆区域）使用实例ID，背景像素（非车辆区域）设为0
    instance_seg = (argmax_ins * foreground_masks.float()).long()  # bg=0, fg从1开始

    # 步骤5: 将实例ID映射为连续编号
    # 由于前景掩码的筛选，部分实例ID可能没有对应的像素，导致ID不连续
    # 此操作将所有存在像素的实例ID重新映射为从0开始的连续编号
    instance_seg = make_instance_seg_consecutive(instance_seg).long()

    return instance_seg