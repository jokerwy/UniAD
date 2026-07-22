#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
运动轨迹非线性平滑优化器 (Motion Nonlinear Smoother)

本模块实现了基于车辆动力学模型的轨迹非线性优化平滑器，用于对运动预测
生成的轨迹进行后处理优化，使其满足车辆的运动学约束。

该模块改编自 nuPlan-devkit (https://github.com/motional/nuplan-devkit)，
使用 CasADi 优化库和 IPOPT 求解器，通过直接多重打靶法 (direct multiple-shooting)
求解带约束的非线性优化问题。

核心思想：
给定一条参考轨迹（由运动预测模型生成），通过优化找到一条满足车辆动力学约束的
平滑轨迹，使其尽可能接近参考轨迹，同时满足以下约束：
- 车辆动力学约束：dx/dt = v*cos(yaw), dy/dt = v*sin(yaw), dyaw/dt = v*curvature, dv/dt = accel
- 控制量约束：曲率限制（最小转弯半径）、加速度限制
- 状态约束：速度限制、横摆角速度限制、横向加速度限制

优化目标：
- 最小化与参考轨迹的位置偏差（xy 和 yaw）
- 最小化控制量的变化率（曲率变化率和 jerk）
- 最小化控制量的绝对值（曲率和加速度）
- 最小化横向加速度
- 对终点状态施加更高的权重（特别是终点朝向角）
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from casadi import DM, Opti, OptiSol, cos, diff, sin, sumsqr, vertcat

# 位姿类型定义: (x, y, yaw) 三元组
Pose = Tuple[float, float, float]  # (x, y, yaw)


class MotionNonlinearSmoother:
    """
    运动轨迹非线性平滑优化器。

    使用车辆动力学模型对一组 (x, y) 观测点进行平滑优化。
    采用直接多重打靶法 (direct multiple-shooting) 求解。
    改编自 nuPlan-devkit。

    状态变量 (4维):
    - x: 车辆在全局坐标系下的 x 坐标
    - y: 车辆在全局坐标系下的 y 坐标
    - yaw: 车辆的偏航角（朝向角）
    - speed: 车辆的速度

    控制变量 (2维):
    - curvature: 车辆行驶路径的曲率（1/转弯半径）
    - accel: 车辆的加速度

    动力学模型:
    - dx/dt = speed * cos(yaw)
    - dy/dt = speed * sin(yaw)
    - dyaw/dt = speed * curvature
    - dspeed/dt = accel

    Args:
        trajectory_len (int): 需要优化的轨迹长度（时间步数）。
        dt (float): 时间步长（秒），相邻轨迹点之间的时间间隔。
    """

    def __init__(self, trajectory_len: int, dt: float):
        """
        初始化运动非线性平滑优化器。

        Args:
            trajectory_len (int): 轨迹长度（时间步数）。
                注意：状态向量的长度为 trajectory_len + 1（包含起点和终点）。
            dt (float): 时间步长（秒），默认 0.5s。
        """
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.current_index = 0  # 当前时间步的索引（用于初始状态约束）

        # 使用 dt 数组以兼容不同时间步长的情况
        self._dts: npt.NDArray[np.float32] = np.asarray(
            [[dt] * trajectory_len])

        # 初始化优化问题
        self._init_optimization()

    def _init_optimization(self) -> None:
        """
        初始化优化问题的相关变量和约束。

        依次执行以下步骤：
        1. 定义状态维度和控制维度
        2. 创建决策变量（状态轨迹和控制轨迹）
        3. 创建参数（参考轨迹和当前状态）
        4. 设置动力学约束
        5. 设置状态约束
        6. 设置控制约束
        7. 设置目标函数
        8. 配置求解器选项
        """
        self.nx = 4  # 状态维度: (x, y, yaw, speed)
        self.nu = 2  # 控制维度: (curvature, accel)

        self._optimizer = Opti()  # 创建 CasADi 优化问题实例
        self._create_decision_variables()  # 创建决策变量
        self._create_parameters()  # 创建参数
        self._set_dynamic_constraints()  # 设置动力学约束
        self._set_state_constraints()  # 设置状态约束
        self._set_control_constraints()  # 设置控制约束
        self._set_objective()  # 设置目标函数

        # 配置 IPOPT 求解器，默认静默模式
        # ipopt.print_level: 0 表示不输出日志
        # print_time: 0 表示不输出求解时间
        # ipopt.sb: "yes" 表示抑制 IPOPT 的 banner 输出
        self._optimizer.solver(
            "ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})

    def set_reference_trajectory(self, x_curr: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """
        设置平滑器需要跟踪的参考轨迹。

        参考轨迹通常由运动预测模型生成，平滑器会找到一条满足车辆动力学约束的
        平滑轨迹，使其尽可能接近参考轨迹。

        Args:
            x_curr (Sequence[float]): 当前状态向量，长度为 nx (4)。
                格式: [x, y, yaw, speed]
            reference_trajectory (Sequence[Pose]): 参考轨迹，长度为 N+1，每个元素为 (x, y, yaw)。
                注意：长度应为 trajectory_len + 1。
        """
        self._check_inputs(x_curr, reference_trajectory)

        # 设置当前状态参数的值
        self._optimizer.set_value(self.x_curr, DM(x_curr))
        # 设置参考轨迹参数的值（转置为 (3, trajectory_len+1) 格式）
        self._optimizer.set_value(self.ref_traj, DM(reference_trajectory).T)
        # 设置初始猜测值（热启动）
        self._set_initial_guess(x_curr, reference_trajectory)

    def set_solver_optimizerons(self, options: Dict[str, Any]) -> None:
        """
        控制求解器选项，包括输出详细程度。

        可以通过此方法覆盖默认的 IPOPT 求解器选项。

        Args:
            options (Dict[str, Any]): IPOPT 求解器选项字典。
                例如: {"ipopt.print_level": 5, "print_time": 1}
        """
        self._optimizer.solver("ipopt", options)

    def solve(self) -> OptiSol:
        """
        求解优化问题。

        假设参考轨迹已经通过 set_reference_trajectory 设置。

        Returns:
            OptiSol: CasADi 优化求解结果。
                可以通过 sol.value(smoother.position_x) 等方法获取优化后的轨迹。
        """
        return self._optimizer.solve()

    def _create_decision_variables(self) -> None:
        """
        定义轨迹优化的决策变量。

        决策变量是优化器可以自由调整的变量，包括：
        - 状态轨迹 (state): 形状 (nx, trajectory_len + 1)，即 (4, N+1)
          - position_x: 状态轨迹的 x 坐标
          - position_y: 状态轨迹的 y 坐标
          - yaw: 状态轨迹的偏航角
          - speed: 状态轨迹的速度
        - 控制轨迹 (control): 形状 (nu, trajectory_len)，即 (2, N)
          - curvature: 控制轨迹的曲率
          - accel: 控制轨迹的加速度

        派生变量（由决策变量计算得出）：
        - curvature_rate: 曲率变化率 = d(curvature)/dt
        - jerk: 加加速度 = d(accel)/dt
        - lateral_accel: 横向加速度 = speed^2 * curvature
        """
        # 状态轨迹: (x, y, yaw, speed) 共 trajectory_len+1 个时间步
        self.state = self._optimizer.variable(self.nx, self.trajectory_len + 1)
        self.position_x = self.state[0, :]  # x 坐标轨迹
        self.position_y = self.state[1, :]  # y 坐标轨迹
        self.yaw = self.state[2, :]  # 偏航角轨迹
        self.speed = self.state[3, :]  # 速度轨迹

        # 控制轨迹: (curvature, accel) 共 trajectory_len 个时间步
        # 控制轨迹比状态轨迹少一步，因为控制量作用于两个状态之间
        self.control = self._optimizer.variable(self.nu, self.trajectory_len)
        self.curvature = self.control[0, :]  # 曲率轨迹
        self.accel = self.control[1, :]  # 加速度轨迹

        # 派生变量: 控制量的变化率和横向加速度
        # dt[:, 1:] 是因为状态向量比控制向量多一步
        self.curvature_rate = diff(self.curvature) / self._dts[:, 1:]
        self.jerk = diff(self.accel) / self._dts[:, 1:]
        self.lateral_accel = self.speed[: self.trajectory_len] ** 2 * \
            self.curvature

    def _create_parameters(self) -> None:
        """
        定义轨迹优化的参数（优化器不能修改的固定值）。

        参数包括：
        - ref_traj: 参考轨迹，形状 (3, trajectory_len + 1)，包含 (x, y, yaw)
        - x_curr: 当前状态，形状 (nx, 1)，即 (4, 1)，包含 (x, y, yaw, speed)
        """
        self.ref_traj = self._optimizer.parameter(
            3, self.trajectory_len + 1)  # (x, y, yaw) 参考轨迹
        self.x_curr = self._optimizer.parameter(self.nx, 1)  # 当前状态

    def _set_dynamic_constraints(self) -> None:
        r"""
        设置系统动力学约束。

        使用车辆运动学模型描述状态转移:
          dx/dt = f(x, u)
          \dot{x} = speed * cos(yaw)
          \dot{y} = speed * sin(yaw)
          \dot{yaw} = speed * curvature
          \dot{speed} = accel

        使用 Runge-Kutta 4 阶 (RK4) 方法进行数值积分，保证精度。
        RK4 公式:
          k1 = f(x_k, u_k)
          k2 = f(x_k + dt/2 * k1, u_k)
          k3 = f(x_k + dt/2 * k2, u_k)
          k4 = f(x_k + dt * k3, u_k)
          x_{k+1} = x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        state = self.state
        control = self.control
        dt = self.dt

        def process(x: Sequence[float], u: Sequence[float]) -> Any:
            """车辆运动学模型的微分方程。

            Args:
                x: 状态向量 [x, y, yaw, speed]
                u: 控制向量 [curvature, accel]

            Returns:
                状态导数 [dx/dt, dy/dt, dyaw/dt, dspeed/dt]
            """
            return vertcat(x[3] * cos(x[2]), x[3] * sin(x[2]), x[3] * u[0], u[1])

        for k in range(self.trajectory_len):  # 遍历每个控制区间
            # Runge-Kutta 4 阶积分
            k1 = process(state[:, k], control[:, k])
            k2 = process(state[:, k] + dt / 2 * k1, control[:, k])
            k3 = process(state[:, k] + dt / 2 * k2, control[:, k])
            k4 = process(state[:, k] + dt * k3, control[:, k])
            next_state = state[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            # 添加约束: 下一个状态必须等于 RK4 积分的结果
            self._optimizer.subject_to(
                state[:, k + 1] == next_state)

    def _set_control_constraints(self) -> None:
        """设置控制量的硬约束。

        约束项：
        1. 曲率限制: [-1/5, 1/5] 1/m
           - 对应最小转弯半径 5 米
           - 曲率 = 1/转弯半径，正值表示左转，负值表示右转
        2. 加速度限制: [-4.0, 4.0] m/s^2
           - 正值表示加速，负值表示减速
        """
        curvature_limit = 1.0 / 5.0  # 1/m，对应最小转弯半径 5 米
        self._optimizer.subject_to(
            self._optimizer.bounded(-curvature_limit, self.curvature, curvature_limit))

        accel_limit = 4.0  # m/s^2，最大加速度
        self._optimizer.subject_to(
            self._optimizer.bounded(-accel_limit, self.accel, accel_limit))

    def _set_state_constraints(self) -> None:
        """设置状态的硬约束。

        约束项：
        1. 初始边界条件: 当前时间步的状态必须等于给定的当前状态
           - state[:, current_index] == x_curr
        2. 速度限制: [0, 35.0] m/s
           - 只允许前进（速度为非负），不允许倒车
           - 最高速度 35 m/s（约 126 km/h）
        3. 横摆角速度限制: [-1.75, 1.75] rad/s
           - 约 100 度/秒，防止过快的转向
        4. 横向加速度限制: [-4.0, 4.0] m/s^2
           - 基于圆形运动假设: a_lat = speed^2 * curvature
           - 防止过大的侧向力
        """
        # 初始边界条件: 当前时间步的状态必须等于给定的当前状态
        self._optimizer.subject_to(
            self.state[:, self.current_index] == self.x_curr)

        # 速度限制: 只允许前进，最大速度 35 m/s
        max_speed = 35.0  # m/s
        self._optimizer.subject_to(self._optimizer.bounded(
            0.0, self.speed, max_speed))

        # 横摆角速度限制: dyaw/dt 在 [-1.75, 1.75] rad/s 范围内
        max_yaw_rate = 1.75  # rad/s
        self._optimizer.subject_to(
            self._optimizer.bounded(-max_yaw_rate, diff(self.yaw) / self._dts, max_yaw_rate))

        # 横向加速度限制: speed^2 * curvature 在 [-4.0, 4.0] m/s^2 范围内
        max_lateral_accel = 4.0  # m/s^2
        self._optimizer.subject_to(
            self._optimizer.bounded(
                -max_lateral_accel, self.speed[:, : self.trajectory_len] ** 2 *
                self.curvature, max_lateral_accel
            )
        )

    def _set_objective(self) -> None:
        """设置目标函数。

        目标函数是一个加权多目标代价函数，由以下部分组成：

        1. 位置跟踪代价 (alpha_xy=1.0):
           - 最小化优化轨迹与参考轨迹在 (x, y) 位置上的偏差
           - 使用平方和 (sumsqr) 计算偏差

        2. 朝向跟踪代价 (alpha_yaw=0.1):
           - 最小化优化轨迹与参考轨迹在偏航角上的偏差
           - 权重较小，因为位置偏差通常更重要

        3. 控制量变化率代价 (alpha_rate=0.08):
           - 最小化曲率变化率 (curvature_rate) 和加加速度 (jerk)
           - 鼓励平滑的转向和加减速

        4. 控制量绝对值代价 (alpha_abs=0.08):
           - 最小化曲率和加速度的绝对值
           - 鼓励节能、舒适的驾驶

        5. 横向加速度代价 (alpha_lat_accel=0.06):
           - 最小化横向加速度
           - 提高乘坐舒适性

        6. 终点状态代价 (alpha_terminal_xy=1.0, alpha_terminal_yaw=40.0):
           - 对轨迹终点施加更高的权重
           - 终点朝向角 (yaw) 的权重特别高 (40.0)，帮助完成换道等操作
           - 终点代价乘以 trajectory_len/4 进行缩放

        总代价 = 阶段代价 + (trajectory_len / 4.0) * 终点代价
        """
        # 权重系数定义
        alpha_xy = 1.0       # 位置跟踪权重
        alpha_yaw = 0.1      # 朝向跟踪权重
        alpha_rate = 0.08    # 控制量变化率权重
        alpha_abs = 0.08     # 控制量绝对值权重
        alpha_lat_accel = 0.06  # 横向加速度权重

        # 阶段代价: 每个时间步的代价之和
        cost_stage = (
            alpha_xy *
            sumsqr(self.ref_traj[:2, :] -
                   vertcat(self.position_x, self.position_y))
            + alpha_yaw * sumsqr(self.ref_traj[2, :] - self.yaw)
            + alpha_rate * (sumsqr(self.curvature_rate) + sumsqr(self.jerk))
            + alpha_abs * (sumsqr(self.curvature) + sumsqr(self.accel))
            + alpha_lat_accel * sumsqr(self.lateral_accel)
        )

        # 终点代价: 对轨迹终点的位置和朝向施加额外权重
        alpha_terminal_xy = 1.0    # 终点位置权重
        alpha_terminal_yaw = 40.0  # 终点朝向权重（非常高，帮助完成换道）
        cost_terminal = alpha_terminal_xy * sumsqr(
            self.ref_traj[:2, -1] -
            vertcat(self.position_x[-1], self.position_y[-1])
        ) + alpha_terminal_yaw * sumsqr(self.ref_traj[2, -1] - self.yaw[-1])

        # 总代价: 阶段代价 + 缩放的终点代价
        self._optimizer.minimize(
            cost_stage + self.trajectory_len / 4.0 * cost_terminal)

    def _set_initial_guess(self, x_curr: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """
        基于参考轨迹设置求解器的初始猜测值（热启动）。

        良好的初始猜测值可以显著加速优化收敛。这里使用参考轨迹作为初始猜测：
        - 状态 (x, y, yaw) 使用参考轨迹的值
        - 速度使用当前速度的恒定值

        Args:
            x_curr (Sequence[float]): 当前状态，格式 [x, y, yaw, speed]
            reference_trajectory (Sequence[Pose]): 参考轨迹，长度为 N+1
        """
        self._check_inputs(x_curr, reference_trajectory)

        # 初始化状态猜测: 前 3 维 (x, y, yaw) 使用参考轨迹
        self._optimizer.set_initial(self.state[:3, :], DM(
            reference_trajectory).T)  # (x, y, yaw)
        # 速度初始化为当前速度的恒定值
        self._optimizer.set_initial(self.state[3, :], DM(x_curr[3]))  # speed

    def _check_inputs(self, x_curr: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """
        检查输入参数的尺寸是否合法。

        Args:
            x_curr (Sequence[float]): 当前状态向量，长度必须为 nx (4)
            reference_trajectory (Sequence[Pose]): 参考轨迹，长度必须为 trajectory_len + 1

        Raises:
            ValueError: 如果输入参数的尺寸不符合要求
        """
        if len(x_curr) != self.nx:
            raise ValueError(
                f"x_curr length {len(x_curr)} must be equal to state dim {self.nx}")

        if len(reference_trajectory) != self.trajectory_len + 1:
            raise ValueError(
                f"reference traj length {len(reference_trajectory)} must be equal to {self.trajectory_len + 1}"
            )