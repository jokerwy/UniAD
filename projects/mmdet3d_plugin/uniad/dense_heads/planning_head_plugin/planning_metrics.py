#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
规划指标评估模块 (Planning Metrics)

本模块实现了自动驾驶规划任务的多维度评估指标，用于衡量规划轨迹的质量。
评估主要从以下两个维度进行：

1. 碰撞指标 (Collision Metrics)：
   - obj_col: 轨迹点与占用栅格之间的碰撞率，衡量规划轨迹点是否落在被占用的栅格中。
   - obj_box_col: 车辆包围盒（bounding box）与占用栅格之间的碰撞率，
     将车辆建模为矩形包围盒，计算包围盒覆盖区域内是否有占用栅格，更真实地反映碰撞情况。

2. 位移误差指标 (L2 Distance)：
   - L2: 规划轨迹与真实轨迹（GT）之间的欧氏距离（L2 范数误差），衡量轨迹的位移精度。

评估流程：
    1. 将规划轨迹和真实轨迹从世界坐标系转换到 BEV（鸟瞰图）栅格坐标系。
    2. 对于 obj_col：检查每个轨迹点所在的栅格是否为占用状态。
    3. 对于 obj_box_col：将车辆建模为固定尺寸的矩形包围盒，在栅格上绘制多边形，
       检查该多边形覆盖区域内是否存在占用栅格。
    4. 对于 L2：直接计算规划轨迹与真实轨迹各点之间的欧氏距离。

该模块继承自 PyTorch Lightning 的 Metric 类，支持分布式训练中的指标聚合（通过 dist_reduce_fx="sum"）。
"""

import torch
import torch.nn as nn
import numpy as np
from skimage.draw import polygon
from pytorch_lightning.metrics.metric import Metric
from ..occ_head_plugin import calculate_birds_eye_view_parameters, gen_dx_bx


class PlanningMetric(Metric):
    """
    规划指标评估类

    继承自 PyTorch Lightning 的 Metric 基类，实现了规划任务的多维度评估。
    使用 add_state 注册需要在分布式环境下同步的状态变量，
    通过 update() 方法累积每个 batch 的指标，通过 compute() 方法计算最终的平均指标。

    关键参数:
        n_future: 预测/规划的未来时间步数，默认为 6。
        compute_on_step: 是否在每个 step 上计算指标（而非仅在 epoch 结束时计算）。

    关键属性:
        dx: BEV 栅格在 x 和 y 方向上的分辨率（栅格单元大小）。
        bx: BEV 栅格左下角的原点坐标偏移量。
        bev_dimension: BEV 栅格的尺寸 (H, W)。
        W: 车辆包围盒的宽度（米），默认 1.85m。
        H: 车辆包围盒的长度（米），默认 4.084m。
    """

    def __init__(
        self,
        n_future=6,
        compute_on_step: bool = False,
    ):
        """
        初始化规划指标评估器。

        参数:
            n_future: 未来轨迹的时间步数，默认 6。该值决定了指标状态向量的维度。
            compute_on_step: 是否在每一步都计算指标。默认为 False，
                             表示仅在 epoch 结束时汇总计算。
        """
        super().__init__(compute_on_step=compute_on_step)
        # 生成 BEV 栅格的参数：
        # gen_dx_bx 根据 x/y/z 三个维度的范围与分辨率，生成 dx(栅格分辨率)、bx(原点偏移)、nx(栅格尺寸)
        # x 范围: [-50.0, 50.0, 0.5] 表示从 -50m 到 50m，分辨率为 0.5m/格
        # y 范围: 同上
        # z 范围: [-10.0, 10.0, 20.0] 表示从 -10m 到 10m，分辨率为 20m/格
        dx, bx, _ = gen_dx_bx([-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0])
        # 只取 x 和 y 方向的参数（忽略 z 方向）
        dx, bx = dx[:2], bx[:2]
        # 将 dx 和 bx 注册为 nn.Parameter，使它们成为模型参数但不需要梯度更新
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)

        # 计算 BEV 栅格的尺寸（H, W）
        _, _, self.bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )
        self.bev_dimension = self.bev_dimension.numpy()

        # 车辆包围盒尺寸：宽度 1.85m，长度 4.084m
        # 这些值与 nuScenes 数据集中典型车辆的尺寸一致
        self.W = 1.85
        self.H = 4.084

        self.n_future = n_future

        # 注册分布式指标状态变量，使用 dist_reduce_fx="sum" 在分布式节点间进行求和聚合
        # obj_col: 每个时间步的轨迹点碰撞计数（逐点检测）
        self.add_state("obj_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        # obj_box_col: 每个时间步的车辆包围盒碰撞计数（包围盒检测）
        self.add_state("obj_box_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        # L2: 每个时间步的 L2 位移误差累积和
        self.add_state("L2", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        # total: 累计评估的轨迹（样本）总数
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    def evaluate_single_coll(self, traj, segmentation):
        """
        评估单条轨迹的车辆包围盒碰撞情况。

        该方法将车辆建模为固定尺寸的矩形包围盒，在 BEV 栅格上绘制该包围盒的多边形，
        然后检查该多边形覆盖区域内是否存在占用栅格。这比逐点检测更准确地反映了
        车辆行驶过程中可能发生的碰撞。

        工作流程：
            1. 以轨迹点为车辆中心，定义车辆的四个角点坐标。
            2. 将角点从世界坐标系转换到 BEV 栅格坐标系。
            3. 使用 skimage.draw.polygon 绘制多边形，获取覆盖的所有栅格索引。
            4. 对于每个时间步，检查多边形覆盖的栅格中是否存在占用（值为 True 的栅格）。

        参数:
            traj: 规划轨迹，形状为 (n_future, 2)，第二维为 (x, y) 世界坐标。
            segmentation: 占用栅格分割图，形状为 (n_future, 200, 200)，
                          值为 True 表示该栅格被占用。

        返回:
            collision: 布尔张量，形状为 (n_future,)，每个元素表示该时间步是否发生碰撞。
        '''
        gt_segmentation
        traj: torch.Tensor (n_future, 2)
        segmentation: torch.Tensor (n_future, 200, 200)
        '''
        # 定义车辆包围盒的四个角点（相对于车辆中心）
        # 车辆坐标系：x 向前（长度方向），y 向左（宽度方向）
        # 四个角点按逆时针/顺时针顺序排列：
        # 左上角: (-H/2 + 0.5,  W/2), 右上角: ( H/2 + 0.5,  W/2)
        # 右下角: ( H/2 + 0.5, -W/2), 左下角: (-H/2 + 0.5, -W/2)
        # 其中 +0.5 是偏移量，将车辆中心放在轨迹点前方 0.5m 处
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        # 将车辆角点从世界坐标系转换到 BEV 栅格坐标系
        # 公式: grid_coord = (world_coord - bx) / dx
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        # 交换 x 和 y 列，因为 BEV 栅格的行/列索引与 (x, y) 可能需要交换
        pts[:, [0, 1]] = pts[:, [1, 0]]
        # 使用 skimage 的 polygon 函数获取多边形内部的所有栅格索引
        # rr: 行索引（y 方向），cc: 列索引（x 方向）
        rr, cc = polygon(pts[:,1], pts[:,0])
        # 将行索引和列索引拼接为 (N, 2) 数组，每个元素为 (row, col)
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)

        n_future, _ = traj.shape
        # 将轨迹 reshape 为 (n_future, 1, 2)，方便后续广播操作
        trajs = traj.view(n_future, 1, 2)
        # 交换 x 和 y 列，与 BEV 栅格坐标系保持一致
        trajs[:,:,[0,1]] = trajs[:,:,[1,0]]  # 注意：这会修改原始张量
        # 将轨迹点从世界坐标转换为栅格坐标
        trajs = trajs / self.dx
        # 将轨迹点与车辆包围盒的栅格偏移相加，得到每个时间步车辆包围盒覆盖的所有栅格坐标
        # 结果形状: (n_future, num_polygon_cells, 2)
        trajs = trajs.cpu().numpy() + rc  # (n_future, 32, 2)

        # 提取所有栅格坐标的行索引
        r = trajs[:,:,0].astype(np.int32)
        # 将行索引裁剪到有效范围 [0, bev_height - 1]
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        # 提取所有栅格坐标的列索引
        c = trajs[:,:,1].astype(np.int32)
        # 将列索引裁剪到有效范围 [0, bev_width - 1]
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        # 初始化碰撞检测结果数组，全部设为 False
        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]  # 当前时间步车辆包围盒覆盖的行索引
            cc = c[t]  # 当前时间步车辆包围盒覆盖的列索引
            # 过滤掉超出 BEV 范围的索引（理论上 clip 已处理，但这里做双重保险）
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            # 检查当前时间步的占用栅格图中，车辆包围盒覆盖区域内是否有任何占用栅格
            # np.any: 如果任意一个覆盖栅格为 True（被占用），则判定为碰撞
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())

        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        """
        批量评估规划轨迹的碰撞指标。

        该方法对 batch 中的每一条轨迹，分别计算两种碰撞指标：
            1. obj_coll: 轨迹点级别的碰撞，检查每个轨迹点所在的栅格是否被占用。
            2. obj_box_coll: 包围盒级别的碰撞，检查车辆包围盒覆盖区域内是否有占用栅格。

        注意：对于真实轨迹（GT）已被判定为碰撞的时间步，对应的规划轨迹碰撞结果
        将从统计中排除（通过 m1 和 m2 掩码过滤），以避免不公平的评估。

        参数:
            trajs: 规划轨迹，形状为 (B, n_future, 2)，B 为 batch 大小。
            gt_trajs: 真实轨迹，形状为 (B, n_future, 2)。
            segmentation: 占用栅格分割图，形状为 (B, n_future, 200, 200)。

        返回:
            obj_coll_sum: 各时间步的轨迹点碰撞累计次数，形状为 (n_future,)。
            obj_box_coll_sum: 各时间步的包围盒碰撞累计次数，形状为 (n_future,)。
        '''
        trajs: torch.Tensor (B, n_future, 2)
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)
        '''
        B, n_future, _ = trajs.shape
        # 将 x 坐标取反：UniAD 中 x 轴方向可能与 BEV 栅格的 x 方向相反
        # 乘以 [-1, 1] 向量：x 取反，y 保持不变
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        # 初始化各时间步的碰撞累计值
        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            # 步骤 1：计算第 i 条真实轨迹的包围盒碰撞情况
            # gt_box_coll: (n_future,) 布尔张量，表示每个时间步 GT 轨迹是否发生碰撞
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i])

            # 步骤 2：计算轨迹点级别的碰撞 (obj_coll)
            xx, yy = trajs[i,:,0], trajs[i, :, 1]
            # 将世界坐标 (x, y) 转换为 BEV 栅格索引 (xi, yi)
            # 注意：bx 和 dx 的索引对应关系：bx[0]/dx[0] 对应 y 方向，bx[1]/dx[1] 对应 x 方向
            yi = ((yy - self.bx[0]) / self.dx[0]).long()
            xi = ((xx - self.bx[1]) / self.dx[1]).long()

            # 掩码 m1：只统计那些规划轨迹点在有效范围内，且 GT 轨迹未发生碰撞的时间步
            # 条件 1：轨迹点索引在 BEV 栅格有效范围内
            m1 = torch.logical_and(
                torch.logical_and(yi >= 0, yi < self.bev_dimension[0]),
                torch.logical_and(xi >= 0, xi < self.bev_dimension[1]),
            )
            # 条件 2：GT 轨迹在该时间步未发生碰撞（排除 GT 本身就在碰撞区域的样本）
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            # 累加轨迹点碰撞：提取满足 m1 条件的轨迹点所在栅格的占用值
            # segmentation[i, ti[m1], yi[m1], xi[m1]] 返回各时间步对应栅格的占用状态
            ti = torch.arange(n_future, device=segmentation.device)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()

            # 步骤 3：计算包围盒级别的碰撞 (obj_box_coll)
            # 掩码 m2：GT 轨迹未发生碰撞的时间步
            m2 = torch.logical_not(gt_box_coll)
            # 计算规划轨迹的包围盒碰撞
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i])
            # 累加包围盒碰撞：只统计 GT 未碰撞的时间步
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        """
        计算规划轨迹与真实轨迹之间的 L2 位移误差（欧氏距离）。

        该函数计算每个时间步上规划轨迹点与真实轨迹点之间的欧氏距离，
        并用 gt_trajs_mask 进行过滤，只计算有效 GT 轨迹点对应的误差。

        参数:
            trajs: 规划轨迹，形状为 (B, n_future, 3)，
                   第三维为 (x, y, yaw)，但只使用前两维 (x, y) 计算 L2 距离。
            gt_trajs: 真实轨迹，形状为 (B, n_future, 3)。
            gt_trajs_mask: 真实轨迹的有效性掩码，形状为 (B, n_future, 1) 或 (B, n_future)，
                           值为 1 表示该轨迹点有效，0 表示无效（如超出范围）。

        返回:
            L2 距离张量，形状为 (B, n_future)，每个元素表示对应时间步的欧氏距离。
        '''
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        '''
        # 计算每个时间步的欧氏距离：
        # 1. (trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2：计算 (x, y) 坐标差的平方
        # 2. * gt_trajs_mask：用掩码过滤无效的 GT 轨迹点（无效点乘 0 后距离为 0）
        # 3. .sum(dim=-1)：将 x 和 y 方向的平方差求和
        # 4. torch.sqrt：开平方根得到欧氏距离
        return torch.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1))

    def update(self, trajs, gt_trajs, gt_trajs_mask, segmentation):
        """
        更新规划指标的状态（每个 batch 调用一次）。

        这是 PyTorch Lightning Metric 的核心接口方法，在每个 batch 处理后调用，
        用于累积指标统计量。支持分布式训练，通过 dist_reduce_fx="sum" 在各节点间
        进行求和聚合。

        工作流程：
            1. 对坐标进行必要的方向转换（x 取反）。
            2. 计算 L2 位移误差并累积到 self.L2。
            3. 计算两类碰撞指标（点碰撞和包围盒碰撞）并累积到对应的状态变量。
            4. 更新累计样本总数。

        参数:
            trajs: 规划轨迹，形状为 (B, n_future, 3)，第三维为 (x, y, yaw)。
            gt_trajs: 真实轨迹（GT），形状为 (B, n_future, 3)。
            gt_trajs_mask: 真实轨迹的有效性掩码。
            segmentation: 占用栅格分割图，形状为 (B, n_future, 200, 200)。
        '''
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        segmentation: torch.Tensor (B, n_future, 200, 200)
        '''
        # 断言规划轨迹和真实轨迹的形状一致
        assert trajs.shape == gt_trajs.shape
        # 将 x 坐标取反，与 evaluate_coll 中的坐标转换保持一致
        trajs[..., 0] = - trajs[..., 0]
        gt_trajs[..., 0] = - gt_trajs[..., 0]
        # 计算 L2 位移误差
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        # 计算碰撞指标（只取前两维 (x, y)，忽略 yaw）
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], segmentation)

        # 累积各项指标到状态变量（分布式训练中会自动跨节点求和）
        self.obj_col += obj_coll_sum  # 累加点碰撞计数
        self.obj_box_col += obj_box_coll_sum  # 累加包围盒碰撞计数
        self.L2 += L2.sum(dim=0)  # 沿 batch 维度求和，累加 L2 误差
        self.total += len(trajs)  # 累加样本总数

    def compute(self):
        """
        计算并返回最终的规划指标。

        该方法在 epoch 结束时调用，将累积的统计量除以样本总数，得到平均指标值。

        返回:
            dict: 包含以下键值对的字典：
                - 'obj_col': 平均轨迹点碰撞率，形状为 (n_future,)，值越大表示碰撞越多。
                - 'obj_box_col': 平均包围盒碰撞率，形状为 (n_future,)，值越大表示碰撞越多。
                - 'L2': 平均 L2 位移误差（米），形状为 (n_future,)，值越小表示轨迹越准确。
        """
        return {
            'obj_col': self.obj_col / self.total,
            'obj_box_col': self.obj_box_col / self.total,
            'L2': self.L2 / self.total
        }