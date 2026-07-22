"""
规划头 (PlanningHeadSingleMode)
===============================
基于感知结果 (BEV 特征、运动预测、占据预测) 生成自车的未来行驶轨迹。

核心流程:
    1. 融合 SDC 查询: 将轨迹查询、跟踪查询和导航指令融合为规划查询
    2. 交叉注意力: 规划查询在 BEV 特征上做注意力，获取环境信息
    3. 轨迹回归: 预测规划步数的轨迹点
    4. 碰撞优化: (可选) 使用占据掩码优化轨迹，避免碰撞

规划轨迹格式:
    预测 planning_steps=6 个时间步的 (x, y) 坐标
    使用 cumsum 累积位移，输出绝对位置
    使用 bivariate Gaussian 激活处理不确定性

Args:
    bev_h, bev_w: BEV 尺寸
    embed_dims: 嵌入维度
    planning_steps: 规划步数，默认 6 (3 秒)
    use_col_optim: 是否使用碰撞优化
"""

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_loss
from einops import rearrange
from projects.mmdet3d_plugin.models.utils.functional import bivariate_gaussian_activation
from .planning_head_plugin import CollisionNonlinearOptimizer
import numpy as np
import copy

@HEADS.register_module()
class PlanningHeadSingleMode(nn.Module):
    def __init__(self,
                 bev_h=200, bev_w=200,          # BEV 尺寸
                 embed_dims=256,                 # 嵌入维度
                 planning_steps=6,               # 规划步数 (3 秒)
                 loss_planning=None,             # 规划损失 (ADE)
                 loss_collision=None,            # 碰撞损失
                 planning_eval=False,            # 是否评估模式
                 use_col_optim=False,            # 是否使用碰撞优化
                 col_optim_args=dict(            # 碰撞优化参数
                     occ_filter_range=5.0,        # 占据过滤范围
                     sigma=1.0,                   # 平滑参数
                     alpha_collision=5.0,         # 碰撞惩罚系数
                 ),
                 with_adapter=False,             # 是否使用 BEV adapter
                ):
        super(PlanningHeadSingleMode, self).__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w

        # 导航指令嵌入: 3 种指令 (左转/右转/直行)
        self.navi_embed = nn.Embedding(3, embed_dims)

        # 轨迹回归分支: embed_dims → planning_steps * 2
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(),
            nn.Linear(embed_dims, planning_steps * 2))

        self.loss_planning = build_loss(loss_planning)
        self.planning_steps = planning_steps
        self.planning_eval = planning_eval

        # 规划 Transformer: 3 个查询在 BEV 特征上做交叉注意力
        fuser_dim = 3
        attn_module_layer = nn.TransformerDecoderLayer(
            embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)

        # 多模态查询融合器
        self.mlp_fuser = nn.Sequential(
            nn.Linear(embed_dims*fuser_dim, embed_dims),
            nn.LayerNorm(embed_dims), nn.ReLU(inplace=True))

        self.pos_embed = nn.Embedding(1, embed_dims)

        # 碰撞损失
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = nn.ModuleList(self.loss_collision)

        self.use_col_optim = use_col_optim
        self.occ_filter_range = col_optim_args['occ_filter_range']
        self.sigma = col_optim_args['sigma']
        self.alpha_collision = col_optim_args['alpha_collision']

        # BEV adapter: 可选的卷积适配器
        self.with_adapter = with_adapter
        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims//2, kernel_size=3, padding=1),
                nn.ReLU(), nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1))
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)

    def forward_train(self, bev_embed, outs_motion={}, sdc_planning=None,
                      sdc_planning_mask=None, command=None, gt_future_boxes=None):
        """训练前向传播

        Args:
            bev_embed: BEV 特征
            outs_motion: 运动预测输出 (包含 SDC 轨迹/跟踪查询)
            sdc_planning: 自车规划真值
            sdc_planning_mask: 自车规划掩码
            command: 导航指令 (0=右转, 1=左转, 2=直行)
            gt_future_boxes: 未来帧的 GT 框 (用于碰撞损失)
        """
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']
        occ_mask = None

        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes]
        losses = self.loss(*loss_inputs)
        ret_dict = dict(losses=losses, outs_motion=outs_planning)
        return ret_dict

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None):
        """推理前向传播"""
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']
        occ_mask = outs_occflow['seg_out']

        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)
        return outs_planning

    def forward(self, bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command):
        """规划核心前向传播

        流程:
        1. 融合 SDC 查询 (轨迹+跟踪+导航) → 规划查询
        2. 规划查询在 BEV 特征上做 Transformer 交叉注意力
        3. 回归分支预测轨迹
        4. 累积位移 → 绝对位置
        5. (推理时) 碰撞优化
        """
        sdc_track_query = sdc_track_query.detach()
        sdc_traj_query = sdc_traj_query[-1]
        P = sdc_traj_query.shape[1]
        sdc_track_query = sdc_track_query[:, None].expand(-1, P, -1)

        # 导航指令嵌入
        navi_embed = self.navi_embed.weight[command]
        navi_embed = navi_embed[None].expand(-1, P, -1)

        # 融合查询: [轨迹查询, 跟踪查询, 导航指令]
        plan_query = torch.cat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)
        plan_query = self.mlp_fuser(plan_query).max(1, keepdim=True)[0]  # 融合为单个查询
        plan_query = rearrange(plan_query, 'b p c -> p b c')

        # BEV 特征 + 位置编码
        bev_pos = rearrange(bev_pos, 'b c h w -> (h w) b c')
        bev_feat = bev_embed + bev_pos

        # 可选的 BEV adapter
        if self.with_adapter:
            bev_feat = rearrange(bev_feat, '(h w) b c -> b c h w', h=self.bev_h, w=self.bev_w)
            bev_feat = bev_feat + self.bev_adapter(bev_feat)
            bev_feat = rearrange(bev_feat, 'b c h w -> (h w) b c')

        pos_embed = self.pos_embed.weight
        plan_query = plan_query + pos_embed[None]

        # Transformer 交叉注意力: 规划查询关注 BEV 特征
        plan_query = self.attn_module(plan_query, bev_feat)

        # 轨迹回归: 预测 planning_steps * 2 个坐标
        sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))

        # cumsum: 累积位移 → 绝对位置
        sdc_traj_all[..., :2] = torch.cumsum(sdc_traj_all[..., :2], dim=1)
        sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])

        # 推理时碰撞优化
        if self.use_col_optim and not self.training:
            assert occ_mask is not None
            sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)

        return dict(sdc_traj=sdc_traj_all, sdc_traj_all=sdc_traj_all)

    def collision_optimization(self, sdc_traj_all, occ_mask):
        """碰撞优化

        使用占据掩码检测轨迹上的碰撞风险，通过非线性优化调整轨迹。
        在轨迹点周围搜索占据区域，使用优化器找到无碰撞的轨迹。

        Args:
            sdc_traj_all: 预测轨迹 (1, planning_steps, 2)
            occ_mask: 占据掩码 (1, T, 1, H, W)

        Returns:
            优化后的轨迹
        """
        pos_xy_t = []
        valid_occupancy_num = 0

        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)
        occ_horizon = occ_mask.shape[1]
        assert occ_horizon == 5

        for t in range(self.planning_steps):
            cur_t = min(t+1, occ_horizon-1)
            pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25

            # 过滤范围内的占据
            keep_index = torch.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2])**2, axis=-1) < self.occ_filter_range**2
            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
            valid_occupancy_num += torch.sum(keep_index>0)

        if valid_occupancy_num == 0:
            return sdc_traj_all

        col_optimizer = CollisionNonlinearOptimizer(
            self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t)
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack([sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)], axis=-1)
        return torch.tensor(sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)

    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None):
        """规划损失计算

        包括:
        - 碰撞损失: 预测轨迹与未来 GT 框的重叠
        - ADE 损失: 预测轨迹与 GT 轨迹的平均位移误差
        """
        sdc_traj_all = outs_planning['sdc_traj_all']
        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](
                sdc_traj_all,
                sdc_planning[0, :, :self.planning_steps, :3],
                torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1),
                future_gt_bbox[0][1:self.planning_steps+1])
            loss_dict[f'loss_collision_{i}'] = loss_collision
        loss_ade = self.loss_planning(
            sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :2],
            torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1))
        loss_dict.update(dict(loss_ade=loss_ade))
        return loss_dict