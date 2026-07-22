#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
运动预测工具函数 (Motion Utility Functions)

本模块提供了运动预测任务中使用的工具函数，主要用于对真实轨迹进行非线性平滑处理，
以生成满足车辆动力学约束的训练目标。

核心函数: nonlinear_smoother

该函数的作用是：
1. 接收原始的真实未来轨迹 (ground truth future trajectory)
2. 使用 MotionNonlinearSmoother 对轨迹进行非线性优化平滑
3. 生成满足车辆运动学约束的平滑轨迹作为训练目标
4. 对平滑后的轨迹进行质量检查，过滤掉质量较差的优化结果

使用场景：
在训练运动预测模型时，直接使用原始的真实轨迹作为回归目标可能存在问题，
因为原始轨迹可能包含噪声或不满足车辆动力学约束。通过非线性平滑器对真实轨迹
进行优化，生成符合车辆运动学模型的平滑轨迹，可以提高模型训练的质量。

处理逻辑：
1. 从真实轨迹计算每步的偏航角 (yaw)
2. 检查目标是否满足"动态"条件（位移超过阈值）
3. 检查当前状态与参考轨迹的初始状态差异是否过大
4. 使用 MotionNonlinearSmoother 进行非线性优化
5. 检查优化结果的 ADE (Average Displacement Error) 是否在阈值内
6. 如果优化结果不满足质量要求，回退到原始真实轨迹
"""

import torch
import random
import numpy as np
from .motion_optimization import MotionNonlinearSmoother


def nonlinear_smoother(gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, bbox_tensor):
    """
    对真实未来轨迹进行非线性平滑处理，生成满足车辆动力学约束的平滑轨迹。

    该函数是 UniAD 运动预测中数据预处理的关键步骤。它使用非线性优化平滑器
    对原始真实轨迹进行优化，生成符合车辆运动学模型的平滑轨迹作为训练目标。

    处理流程：
    1. 将输入张量从 GPU 转移到 CPU 并转换为 NumPy 数组
    2. 从真实轨迹中计算每步的偏航角 (yaw)
    3. 对每个样本，判断是否满足平滑条件：
       - 轨迹长度 > 1（至少需要 2 个点才能进行平滑）
       - 目标满足"动态"条件（总位移超过 2 米）
       - 当前状态与参考轨迹初始状态差异不过大（位置差 < 2m，角度差 < 30 度）
    4. 对满足条件的样本，使用 MotionNonlinearSmoother 进行非线性优化
    5. 检查优化结果的 ADE 是否在阈值内（< 1.5 米）
    6. 如果优化结果不满足质量要求，回退到原始真实轨迹
    7. 将轨迹转换为相对于起点的偏移量格式

    Args:
        gt_bboxes_3d (torch.Tensor): 真实的 3D 边界框，形状 (batch_size, 7)。
            7 个维度通常为: [x, y, z, w, l, h, yaw]
            用于提取初始位置和偏航角。
        gt_fut_traj (torch.Tensor): 真实的未来轨迹，形状 (batch_size, 12, 2)。
            12 个时间步（6 秒，0.5s 间隔），每步包含 (x, y) 坐标（绝对坐标）。
        gt_fut_traj_mask (torch.Tensor): 真实未来轨迹的有效性掩码，形状 (batch_size, 12)。
            值为 1 表示该时间步有效，0 表示无效（如目标已离开场景）。
        bbox_tensor (torch.Tensor): 检测框属性张量，形状 (batch_size, 9)。
            9 个维度通常为: [x, y, z, w, l, h, yaw, vx, vy]
            用于提供当前状态（位置、偏航角、速度）。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - traj_perturb (Tensor): 平滑后的轨迹（相对偏移量格式），形状 (batch_size, 12, 2)。
              轨迹值是相对于起始位置的偏移量 (dx, dy)。
            - mask (Tensor): 更新后的掩码，形状 (batch_size, 12)。
              值为布尔值，True 表示该时间步有效。
    """
    device = gt_fut_traj.device
    dtype = gt_fut_traj.dtype

    # 将输入张量从 GPU 转移到 CPU 并转换为 NumPy 数组
    # 因为 CasADi 优化器在 CPU 上运行
    gt_bboxes_3d = gt_bboxes_3d.cpu().detach().numpy()
    gt_fut_traj = gt_fut_traj.cpu().detach().numpy()
    gt_fut_traj_xy_diff = np.zeros((gt_fut_traj.shape[0], 13, 2))

    # 在第 0 步填充 0（表示起始位置），将轨迹从 12 步扩展为 13 步（包括起点）
    gt_fut_traj_xy_diff[:, 1:, :] = gt_fut_traj

    # 计算相邻步之间的 (dx, dy) 差异
    gt_fut_traj_xy_diff = np.diff(gt_fut_traj_xy_diff, axis=1)

    # 从 (dx, dy) 计算每步的偏航角 (yaw)
    # arctan2(dy, dx) 计算从 x 轴正方向到点 (dx, dy) 的角度
    gt_fut_traj_yaw = np.arctan2(
        gt_fut_traj_xy_diff[:, :, 1], gt_fut_traj_xy_diff[:, :, 0])

    # 将边界框的偏航角作为初始偏航角，拼接在轨迹偏航角前面
    # NOTE: 已适配 mmdet3d 1.0.0rc6 格式的偏航角定义
    gt_fut_traj_yaw = np.concatenate(
        [gt_bboxes_3d[:, None, 6:7], gt_fut_traj_yaw[:, :, None]], axis=1)

    # 将边界框的初始位置 (x, y) 拼接在轨迹前面，形成完整的 13 步轨迹
    gt_fut_traj = np.concatenate(
        [gt_bboxes_3d[:, None, :2], gt_fut_traj], axis=1)

    # 将 mask 和 bbox_tensor 转换到 CPU
    gt_fut_traj_mask = gt_fut_traj_mask.cpu().detach().numpy()
    bbox_tensor = bbox_tensor.cpu().detach().numpy()

    # 计算每个样本的有效时间步数限制
    # mask 求和得到每个样本的有效步数
    ts_limit = gt_fut_traj_mask.sum(1)[:, 0]

    # 从 bbox_tensor 中提取当前状态信息
    yaw_preds = bbox_tensor[:, 6]  # 预测的偏航角
    vel_preds = bbox_tensor[:, -2:]  # 预测的速度 (vx, vy)
    speed_preds = np.sqrt(np.sum(vel_preds**2, axis=-1))  # 计算速度大小

    traj_perturb_all = []  # 存储所有样本的平滑轨迹

    def _is_dynamic(traj, ts, dist_thres):
        """
        判断目标是否处于"动态"状态。

        通过检查从起点到终点（ts 步）的总位移是否超过阈值来判断。
        静态目标（如停放的车辆）的轨迹不需要平滑处理。

        Args:
            traj: 轨迹数组，形状 (N, 2)
            ts: 有效时间步数（整数）
            dist_thres: 位移阈值（米），低于此阈值视为静态

        Returns:
            bool: True 表示目标处于动态状态
        """
        return np.sqrt(np.sum((traj[ts, :2] - traj[0, :2])**2)) > dist_thres

    def _check_diff(x_curr, ref_traj):
        """
        检查当前状态与参考轨迹初始状态之间的差异。

        如果差异过大，说明当前状态与参考轨迹的起点不匹配，
        这种情况下平滑优化可能不会产生有意义的结果。

        Args:
            x_curr: 当前状态 [x, y, yaw, speed]
            ref_traj: 参考轨迹，形状 (N, 3)，每行为 (x, y, yaw)

        Returns:
            bool: True 表示差异在可接受范围内
        """
        # 检查位置差异: 当前位置与参考轨迹起点距离不超过 2 米
        if np.sqrt((x_curr[0] - ref_traj[0, 0]) ** 2 + (x_curr[1] - ref_traj[0, 1])**2) > 2:
            return False

        # 检查角度差异: 使用余弦相似度计算角度差，不超过 30 度
        a = np.array([np.cos(x_curr[2]), np.sin(x_curr[2])])
        b = np.array([np.cos(ref_traj[0, 2]), np.sin(ref_traj[0, 2])])
        diff_theta = np.arccos(
            np.sum(a*b)/(np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))))
        if diff_theta > np.pi/180 * 30:  # 30 度转换为弧度
            return False
        return True

    def _check_ade(traj_pert, traj_ref, thres):
        """
        检查平滑轨迹与参考轨迹之间的平均位移误差 (ADE)。

        ADE 是运动预测中常用的评估指标，计算预测轨迹与真实轨迹之间
        各时间步的平均欧氏距离。

        Args:
            traj_pert: 平滑后的轨迹，形状 (N, 2) 或其他包含 (x, y) 坐标的格式
            traj_ref: 参考轨迹，形状 (N, 3) 或 (N, 2)，包含 (x, y) 坐标
            thres: ADE 阈值（米），低于此阈值认为平滑结果质量可接受

        Returns:
            bool: True 表示 ADE 在阈值内
        """
        return np.mean(np.sqrt(np.sum((traj_pert[:, :2] - traj_ref[:, :2])**2, axis=-1))) < thres

    perturb_count = 0       # 成功平滑的样本计数器
    perturb_used_count = 0  # 使用平滑器的样本计数器

    for i in range(gt_fut_traj.shape[0]):
        ts = ts_limit[i]  # 当前样本的有效时间步数

        # 构建当前状态向量: [x, y, yaw, speed]
        x_curr = [bbox_tensor[i, 0], bbox_tensor[i, 1],
                  yaw_preds[i], speed_preds[i]]

        # 构建参考轨迹: 将 (x, y) 坐标和偏航角拼接
        # 形状: (13, 3) 或 (ts+1, 3)，每行为 (x, y, yaw)
        reference_trajectory = np.concatenate(
            [gt_fut_traj[i], gt_fut_traj_yaw[i]], axis=-1)

        # 判断是否需要进行非线性平滑:
        # 1. 有效步数 > 1（至少 2 个点才能构成轨迹）
        # 2. 目标处于动态状态（位移超过 2 米）
        # 3. 当前状态与参考轨迹初始状态差异不大
        if ts > 1 and _is_dynamic(gt_fut_traj[i], int(ts), 2) and _check_diff(x_curr, reference_trajectory):
            # 创建非线性平滑器，轨迹长度为有效步数，时间步长为 0.5s
            smoother = MotionNonlinearSmoother(
                trajectory_len=int(ts), dt=0.5)

            # 截取有效步数的参考轨迹（+1 是因为包含起点）
            reference_trajectory = reference_trajectory[:int(ts)+1, :]

            # 设置参考轨迹并求解
            smoother.set_reference_trajectory(x_curr, reference_trajectory)
            sol = smoother.solve()

            # 提取优化后的 (x, y) 轨迹
            traj_perturb = np.stack(
                [sol.value(smoother.position_x), sol.value(smoother.position_y)], axis=-1)
            perturb_used_count += 1

            # 检查优化结果的质量: ADE 是否在阈值 1.5 米内
            if not _check_ade(traj_perturb, reference_trajectory, thres=1.5):
                # 优化结果质量不佳，回退到原始真实轨迹的偏移量
                traj_perturb = gt_fut_traj[i, 1:, :2] - gt_fut_traj[i, 0:1, :2]
            else:
                # 优化结果质量良好，将绝对坐标转换为相对偏移量
                traj_perturb_tmp = traj_perturb[1:, :2] - traj_perturb[0:1, :2]
                # 创建 12 步的输出数组，填充有效步数的偏移量
                traj_perturb = np.zeros((12, 2))
                traj_perturb[:traj_perturb_tmp.shape[0], :] = traj_perturb_tmp[:, :2]
                perturb_count += 1
        else:
            # 不满足平滑条件，直接使用原始真实轨迹的偏移量
            traj_perturb = gt_fut_traj[i, 1:, :2] - gt_fut_traj[i, 0:1, :2]

        traj_perturb_all.append(traj_perturb)

    # 将结果转换回原始设备和数据类型
    return (
        torch.tensor(traj_perturb_all, device=device, dtype=dtype),
        torch.tensor(gt_fut_traj_mask > 0, device=device)
    )