#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
碰撞优化模块 (Collision Optimization)

本模块实现了基于非线性优化的轨迹碰撞避免功能。
核心思路是：将规划好的轨迹作为参考轨迹，同时结合预测的占用栅格（occupancy）信息，
通过求解一个非线性优化问题，对轨迹进行微调，使其在保持与参考轨迹接近的同时，远离被占用的区域。

优化问题使用 CasADi 框架建模，并采用 IPOPT（内点法）求解器进行求解。
优化目标由两部分组成：
    1. 轨迹跟踪代价（cost_stage）：惩罚优化后的轨迹与参考轨迹之间的偏差。
    2. 碰撞代价（cost_collision）：惩罚优化后的轨迹点落入占用区域的概率，
       使用高斯核函数建模，离占用区域中心越近，代价越高。

该方法参考了 nuPlan (https://github.com/motional/nuplan-devkit) 中的实现，
并针对 UniAD 的规划流程进行了适配。
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from casadi import DM, Opti, OptiSol, cos, diff, sin, sumsqr, vertcat, exp

# 定义位姿类型别名：(x坐标, y坐标, 偏航角)
Pose = Tuple[float, float, float]  # (x, y, yaw)


class CollisionNonlinearOptimizer:
    """
    碰撞非线性优化器

    使用直接多重打靶法（Direct Multiple Shooting）对规划轨迹进行优化，
    以避免与预测的占用区域发生碰撞。优化器基于 CasADi 框架和 IPOPT 求解器。

    工作流程：
        1. 初始化优化问题（决策变量、参数、目标函数、求解器配置）
        2. 设置参考轨迹（作为优化问题的参数和初始猜测值）
        3. 调用 solve() 求解优化问题，返回优化后的轨迹

    参数:
        trajectory_len: 轨迹的时间步数（长度）
        dt: 相邻轨迹点之间的时间间隔（秒）
        sigma: 碰撞代价高斯核函数的标准差，控制碰撞代价的影响范围
        alpha_collision: 碰撞代价的权重系数，越大则越倾向于避开占用区域
        obj_pixel_pos: 每个时间步的占用区域中心像素坐标列表，
                       形状为 [trajectory_len][num_objects][2]，
                       每个元素为 (x, y) 坐标
    """

    def __init__(self, trajectory_len: int, dt: float, sigma, alpha_collision, obj_pixel_pos):
        """
        初始化碰撞非线性优化器。

        参数:
            trajectory_len: 需要优化的轨迹长度（时间步数）。
            dt: 相邻轨迹点之间的时间间隔（秒）。
            sigma: 碰撞代价高斯核函数的标准差。该值决定了碰撞代价的"影响半径"，
                   sigma 越大，碰撞代价的作用范围越广，但峰值越小。
            alpha_collision: 碰撞代价的权重系数。该值越大，优化器越倾向于
                            牺牲轨迹跟踪精度来避免碰撞。
            obj_pixel_pos: 占用区域的位置信息，是一个列表，长度为 trajectory_len，
                          每个元素是一个列表，包含该时间步上所有占用区域中心的 (x, y) 坐标。
        """
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.current_index = 0
        self.sigma = sigma
        self.alpha_collision = alpha_collision
        self.obj_pixel_pos = obj_pixel_pos
        # 使用 dts 数组以兼容不同时间步长可能不相等的情况
        self._dts: npt.NDArray[np.float32] = np.asarray([[dt] * trajectory_len])
        # 初始化优化问题
        self._init_optimization()

    def _init_optimization(self) -> None:
        """
        初始化优化问题的相关变量和约束条件。

        具体步骤：
            1. 设置状态维度 nx = 2（仅优化 x, y 坐标，不优化偏航角）
            2. 创建 CasADi Opti 优化问题实例
            3. 创建决策变量（优化变量）
            4. 创建参数（参考轨迹）
            5. 设置目标函数
            6. 配置 IPOPT 求解器（默认静默模式）
        """
        self.nx = 2  # 状态维度：仅优化 (x, y) 两个坐标

        self._optimizer = Opti()  # 创建 CasADi 优化问题实例
        self._create_decision_variables()  # 创建决策变量（状态轨迹）
        self._create_parameters()  # 创建参数（参考轨迹）
        self._set_objective()  # 设置目标函数

        # 配置默认求解器选项：使用 IPOPT 求解器，输出设为静默模式
        # ipopt.print_level: 0 表示不输出求解日志
        # print_time: 0 表示不输出求解耗时
        # ipopt.sb: "yes" 表示抑制 IPOPT 的 banner 信息
        self._optimizer.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})

    def set_reference_trajectory(self, reference_trajectory: Sequence[Pose]) -> None:
        """
        设置参考轨迹，优化器会尽量贴近该轨迹，同时避免碰撞。

        参考轨迹作为优化问题的参数传入，同时作为优化变量的初始猜测值（warm-start），
        以加速求解收敛。

        参数:
            reference_trajectory: 形状为 N x 3 的参考轨迹，第二维包含 (x, y, yaw)，
                                  其中 N 等于 trajectory_len。注意 yaw 在本优化中
                                  不参与优化，仅取前两维 (x, y)。
        """
        # 将参考轨迹的前两维 (x, y) 转置后设置为优化问题中 ref_traj 参数的值
        # DM 是 CasADi 的稠密矩阵类型，用于高效数值计算
        self._optimizer.set_value(self.ref_traj, DM(reference_trajectory).T)
        # 以参考轨迹作为优化变量的初始猜测值，提供 warm-start
        self._set_initial_guess(reference_trajectory)

    def set_solver_optimizerons(self, options: Dict[str, Any]) -> None:
        """
        设置求解器选项，用于控制求解器的详细行为。

        可以通过此方法设置 IPOPT 求解器的各种参数，例如：
            - ipopt.max_iter: 最大迭代次数
            - ipopt.tol: 收敛容差
            - ipopt.print_level: 日志输出级别

        参数:
            options: 包含求解器配置选项的字典，键值对将直接传递给 IPOPT 求解器。
        """
        self._optimizer.solver("ipopt", options)

    def solve(self) -> OptiSol:
        """
        求解优化问题，获得优化后的轨迹。

        调用此方法前，需要先通过 set_reference_trajectory() 设置参考轨迹。

        返回:
            OptiSol: CasADi 优化解对象，包含优化后的变量值。
                     可以通过 sol.value(self.state) 获取优化后的状态轨迹。
        """
        return self._optimizer.solve()

    def _create_decision_variables(self) -> None:
        """
        创建轨迹优化的决策变量（优化变量）。

        决策变量是优化器需要求解的未知量，这里定义为完整的轨迹状态序列：
            - state: 形状为 (nx, trajectory_len) 的矩阵，即 (2, trajectory_len)
            - position_x: state 的第 0 行，表示所有时间步的 x 坐标
            - position_y: state 的第 1 行，表示所有时间步的 y 坐标

        注意：这里只优化 (x, y) 两个自由度，偏航角 yaw 不在优化范围内。
        """
        # 状态轨迹变量：shape (2, trajectory_len)，分别对应 x 和 y 坐标
        self.state = self._optimizer.variable(self.nx, self.trajectory_len)
        # 提取 x 坐标序列，用于目标函数中方便引用
        self.position_x = self.state[0, :]
        # 提取 y 坐标序列，用于目标函数中方便引用
        self.position_y = self.state[1, :]

    def _create_parameters(self) -> None:
        """
        创建优化问题的参数。

        参数是在优化过程中保持固定不变的量，用于向优化器传递外部信息。
        这里定义了一个参数 ref_traj，用于存储参考轨迹，优化器会尽量贴近该轨迹。

        参数类型为 CasADi 的 parameter，形状为 (2, trajectory_len)，
        两行分别对应 x 和 y 坐标。
        """
        # 参考轨迹参数：shape (2, trajectory_len)，存储 (x, y) 坐标
        self.ref_traj = self._optimizer.parameter(2, self.trajectory_len)  # (x, y)

    def _set_objective(self) -> None:
        """
        设置优化问题的目标函数。

        目标函数由两部分组成，通过加权求和构成总代价：

        1. 轨迹跟踪代价 (cost_stage)：
           使用 sumsqr（平方和）计算优化后的轨迹与参考轨迹之间的欧氏距离平方。
           权重 alpha_xy = 1.0，表示对轨迹跟踪偏差的惩罚力度。

        2. 碰撞代价 (cost_collision)：
           对于每个时间步 t，遍历该时间步的所有占用区域中心 (col_x, col_y)，
           使用高斯核函数 exp(-d^2 / (2*sigma^2)) 计算该轨迹点与占用区域中心的
           邻近程度。轨迹点越靠近占用区域中心，代价越大。
           权重 alpha_collision 控制碰撞代价的整体重要程度。
           normalizer = 1/(2.507*sigma) 是高斯核的归一化因子（近似于 1/(sqrt(2*pi)*sigma)）。

        注意：修改这些权重时需要谨慎，它们直接影响到优化结果的行为。
        """
        # 轨迹跟踪代价权重：控制优化轨迹与参考轨迹的贴合程度
        alpha_xy = 1.0
        # 计算轨迹跟踪代价：所有时间步上优化轨迹与参考轨迹的欧氏距离平方和
        # vertcat 将 position_x 和 position_y 垂直拼接为 (2, trajectory_len) 矩阵
        cost_stage = (
            alpha_xy * sumsqr(self.ref_traj[:2, :] - vertcat(self.position_x, self.position_y))
        )

        # 碰撞代价权重：控制轨迹对占用区域的规避程度
        alpha_collision = self.alpha_collision

        # 碰撞代价初始化为 0
        cost_collision = 0
        # 高斯核归一化因子：1/(sqrt(2*pi)*sigma) ≈ 1/(2.507*sigma)
        normalizer = 1/(2.507*self.sigma)
        # TODO: 将该循环向量化以提升性能
        # 遍历每个时间步
        for t in range(len(self.obj_pixel_pos)):
            # 当前时间步优化轨迹的 (x, y) 坐标
            x, y = self.position_x[t], self.position_y[t]
            # 遍历当前时间步的所有占用区域中心
            for i in range(len(self.obj_pixel_pos[t])):
                # 占用区域中心的坐标
                col_x, col_y = self.obj_pixel_pos[t][i]
                # 累加高斯核碰撞代价：
                # 代价 = alpha_collision * normalizer * exp(-((x-col_x)^2 + (y-col_y)^2) / (2*sigma^2))
                # 当轨迹点 (x,y) 靠近占用区域中心 (col_x, col_y) 时，exp 项接近 1，代价最大
                # 当轨迹点远离占用区域时，exp 项趋近 0，代价可忽略
                cost_collision += alpha_collision * normalizer * exp(-((x - col_x)**2 + (y - col_y)**2)/2/self.sigma**2)
        # 将总代价（轨迹跟踪 + 碰撞避免）设置为优化目标，求解器将最小化该值
        self._optimizer.minimize(cost_stage + cost_collision)

    def _set_initial_guess(self, reference_trajectory: Sequence[Pose]) -> None:
        """
        为求解器设置基于参考轨迹的初始猜测值（warm-start）。

        良好的初始猜测值可以显著加速 IPOPT 的收敛速度，避免陷入局部最优。
        这里直接将参考轨迹作为优化变量 state 的初始值。

        参数:
            reference_trajectory: 形状为 N x 3 的参考轨迹，取前两维 (x, y) 并转置。
        """
        # 将参考轨迹的前两维 (x, y) 转置后作为状态变量的初始猜测值
        # DM(reference_trajectory).T 将 (N, 3) 转为 (3, N)，再取 [:2, :] 得到 (2, N)
        self._optimizer.set_initial(self.state[:2, :], DM(reference_trajectory).T)  # (x, y, yaw)