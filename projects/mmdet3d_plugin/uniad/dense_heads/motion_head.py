"""
运动预测头 (MotionHead)
=======================
预测每个跟踪目标的未来轨迹，支持多模态预测（每个目标预测多个可能的轨迹）。

核心流程:
    1. 编码跟踪目标中心点 → track_query_pos
    2. 构建可学习的 anchor 查询 (agent-level, scene-level ego, scene-level offset)
    3. 根据目标类别分组选择对应的 anchor
    4. MotionFormer: track_query + lane_query → 轨迹预测
    5. 分类 + 回归: 预测轨迹分数和轨迹点 (使用 bivariate Gaussian 激活)

轨迹预测格式:
    每个目标预测 num_anchor 个模态，每个模态预测 predict_steps 个时间步
    每个时间步输出 5 个参数: (μx, μy, σx, σy, ρ)  (二元高斯分布)

Args:
    predict_steps: 预测步数，默认 12 (6 秒)
    num_anchor: anchor 模态数，默认 6
    det_layer_num: Decoder 层数
    group_id_list: 类别分组，用于 anchor 分配
    vehicle_id_list: 车辆类别 ID，用于过滤非车辆目标
"""

import torch
import copy
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.functional import (
    bivariate_gaussian_activation, norm_points, pos2posemb2d, anchor_coordinate_transform)
from .motion_head_plugin.motion_utils import nonlinear_smoother
from .motion_head_plugin.base_motion_head import BaseMotionHead


@HEADS.register_module()
class MotionHead(BaseMotionHead):
    def __init__(self, *args,
                 predict_steps=12,          # 预测步数 (6 秒，每步 0.5 秒)
                 transformerlayers=None,     # Transformer 层配置
                 bbox_coder=None,            # 框编码器
                 num_cls_fcs=2,              # 分类 FC 层数
                 bev_h=30, bev_w=30,         # BEV 尺寸
                 embed_dims=256,             # 嵌入维度
                 num_anchor=6,               # 每个目标的轨迹模态数
                 det_layer_num=6,            # Decoder 层数
                 group_id_list=[],           # 类别分组
                 pc_range=None,              # 点云范围
                 use_nonlinear_optimizer=False,  # 是否使用非线性优化
                 anchor_info_path=None,      # anchor 信息路径
                 loss_traj=dict(),           # 轨迹损失配置
                 num_classes=0,              # 类别数
                 vehicle_id_list=[0,1,2,3,4,6,7],  # 车辆类别 ID
                 **kwargs):
        super(MotionHead, self).__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_reg_fcs = num_cls_fcs - 1
        self.embed_dims = embed_dims
        self.num_anchor = num_anchor
        self.num_anchor_group = len(group_id_list)

        # 将类别合并到分组中 (用于 anchor 分配)
        self.cls2group = [0 for i in range(num_classes)]
        for i, grouped_ids in enumerate(group_id_list):
            for gid in grouped_ids:
                self.cls2group[gid] = i
        self.cls2group = torch.tensor(self.cls2group)
        self.pc_range = pc_range
        self.predict_steps = predict_steps
        self.vehicle_id_list = vehicle_id_list

        self.use_nonlinear_optimizer = use_nonlinear_optimizer
        self._load_anchors(anchor_info_path)
        self._build_loss(loss_traj)
        self._build_layers(transformerlayers, det_layer_num)
        self._init_layers()

    def forward_train(self, bev_embed, gt_bboxes_3d, gt_labels_3d,
                      gt_fut_traj=None, gt_fut_traj_mask=None,
                      gt_sdc_fut_traj=None, gt_sdc_fut_traj_mask=None,
                      outs_track={}, outs_seg={}):
        """训练前向传播

        与 forward_test 相比，训练时多了 GT 轨迹的拼接和损失计算。
        关键区别：训练时自车(SDC)的 GT 轨迹需要与普通目标的 GT 轨迹合并，
        并扩展 matched_idxes 以包含 SDC。

        ---
        处理流程:
            1. 提取跟踪 query 和匹配索引
            2. 将自车 (SDC) query 拼接到跟踪 query 末尾
               - track_query: 在 agent 维度上拼接 (dim=2)
               - track_boxes: 分别在 bbox/scores/labels/index 上拼接
               - matched_idxes: 追加 SDC 的匹配索引
               - gt_fut_traj: 合并 SDC 的 GT 未来轨迹
            3. 提取地图 query（lane_query, lane_query_pos）
            4. 调用 self.forward() 进行 MotionFormer 预测
            5. 计算轨迹损失
            6. 分离自车 query，过滤只保留车辆目标

        ---
        Args:
            bev_embed: BEV 特征图
            gt_bboxes_3d: GT 3D 检测框
            gt_labels_3d: GT 类别标签
            gt_fut_traj: GT 未来轨迹（普通目标）
            gt_fut_traj_mask: GT 未来轨迹有效掩码
            gt_sdc_fut_traj: GT 自车未来轨迹
            gt_sdc_fut_traj_mask: GT 自车未来轨迹有效掩码
            outs_track: Track Head 的输出，包含:
                - track_query_embeddings: 活跃跟踪目标的 query 嵌入
                - track_query_matched_idxes: 每个 query 匹配的 GT 索引
                - track_bbox_results: 检测框结果
                - sdc_embedding: 自车 query 嵌入
                - sdc_track_bbox_results: 自车检测框结果
            outs_seg: Map/Seg Head 的输出，包含:
                - args_tuple: (memory, memory_mask, memory_pos, lane_query,
                               _, lane_query_pos, hw_lvl)

        Returns:
            dict: 包含:
                - 'losses': 损失字典
                - 'outs_motion': 运动预测输出（只含车辆目标）
                - 'track_boxes': 跟踪框（含 SDC）
        """
        # =====================================================================
        # 步骤 1: 提取跟踪结果
        # =====================================================================
        # track_query 形状: (N_active, D) → 扩展为 (1, 1, N_active, D)
        #   维度含义: (batch=1, temporal_layers=1, agents, dim)
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        all_matched_idxes = [outs_track['track_query_matched_idxes']]
        track_boxes = outs_track['track_bbox_results']

        # =====================================================================
        # 步骤 2: 将自车 (SDC) query 拼接到跟踪 query 末尾
        # =====================================================================
        # 为什么需要拼接 SDC?
        #   Motion Head 需要同时预测所有智能体（包括自车）的未来轨迹，
        #   自车作为第 (N_active+1) 个智能体参与 MotionFormer 的智能体间交互。
        #   拼接后 track_query 形状: (1, 1, N_active+1, D)

        # SDC 的匹配索引: 指向 GT 轨迹中 SDC 对应的索引
        sdc_match_index = torch.zeros((1,), dtype=all_matched_idxes[0].dtype,
                                       device=all_matched_idxes[0].device)
        sdc_match_index[0] = gt_fut_traj[0].shape[0]  # SDC 在所有 GT 中的最后一个位置
        all_matched_idxes = [torch.cat([all_matched_idxes[0], sdc_match_index], dim=0)]

        # 合并 SDC 的 GT 轨迹
        gt_fut_traj[0] = torch.cat([gt_fut_traj[0], gt_sdc_fut_traj[0]], dim=0)
        gt_fut_traj_mask[0] = torch.cat([gt_fut_traj_mask[0], gt_sdc_fut_traj_mask[0]], dim=0)

        # 拼接 SDC query 嵌入: (1, 1, N_active, D) → (1, 1, N_active+1, D)
        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)

        # 拼接 SDC 检测框: 将 SDC 的 bbox/scores/labels/index 分别追加到对应列表末尾
        sdc_track_boxes = outs_track['sdc_track_bbox_results']
        track_boxes[0][0].tensor = torch.cat(
            [track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat(
            [track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat(
            [track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat(
            [track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)

        # =====================================================================
        # 步骤 3: 提取地图查询
        # =====================================================================
        # args_tuple = [memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl]
        # lane_query: (B, M, D) — 地图元素嵌入
        # lane_query_pos: (B, M, D) — 地图元素位置编码
        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']

        # =====================================================================
        # 步骤 4: MotionFormer 前向传播
        # =====================================================================
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        loss_inputs = [gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask,
                       outs_motion, all_matched_idxes, track_boxes]
        losses = self.loss(*loss_inputs)

        # =====================================================================
        # 步骤 5: 分离自车 query，过滤只保留车辆目标
        # =====================================================================
        # 移除 SDC 的匹配索引（后续不再需要）
        all_matched_idxes[0] = all_matched_idxes[0][:-1]

        # 提取 SDC 专属的 query 和轨迹（最后一个 agent, index=-1）
        # 这些将作为 Planning Head 的输入
        outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]      # (num_layers, B, P, D)
        outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]        # (B, D)
        outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]  # (B, D)

        # 移除 SDC，只保留普通目标
        outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]
        outs_motion['track_query'] = outs_motion['track_query'][:, :-1]
        outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]

        # 过滤只保留车辆类别的目标（剔除行人、自行车等）
        outs_motion, all_matched_idxes = self._filter_vehicle_query(
            outs_motion, all_matched_idxes, gt_labels_3d, self.vehicle_id_list)
        outs_motion['all_matched_idxes'] = all_matched_idxes

        ret_dict = dict(losses=losses, outs_motion=outs_motion, track_boxes=track_boxes)
        return ret_dict

    def _filter_vehicle_query(self, outs_motion, all_matched_idxes, gt_labels_3d, vehicle_id_list):
        """过滤车辆 query: 只保留车辆类别的目标（训练时调用）

        为什么需要过滤？
        UniAD 的 Occupancy Head 和 Planning Head 只关心车辆目标的运动，
        行人和自行车等目标的轨迹不需要传递给下游模块。
        但在 MotionFormer 内部，所有智能体都参与交互（因为行人的运动也会影响
        车辆的决策），过滤只在 Motion Head 输出后、传递给下游模块前进行。

        Args:
            outs_motion: 运动预测输出
            all_matched_idxes: 匹配索引
            gt_labels_3d: GT 标签
            vehicle_id_list: 车辆类别 ID 列表

        Returns:
            Tuple[dict, list]: 过滤后的 outs_motion 和 matched_idxes
        """
        query_label = gt_labels_3d[0][-1][all_matched_idxes[0]]
        vehicle_mask = torch.zeros_like(query_label)
        for veh_id in vehicle_id_list:
            vehicle_mask |= query_label == veh_id
        outs_motion['traj_query'] = outs_motion['traj_query'][:, :, vehicle_mask>0]
        outs_motion['track_query'] = outs_motion['track_query'][:, vehicle_mask>0]
        outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, vehicle_mask>0]
        all_matched_idxes[0] = all_matched_idxes[0][vehicle_mask>0]
        return outs_motion, all_matched_idxes

    def forward_test(self, bev_embed, outs_track={}, outs_seg={}):
        """推理前向传播

        与 forward_train 的区别:
        - 不需要 GT 轨迹，不需要计算损失
        - 不需要 matched_idxes
        - 最终调用 get_trajs() 将预测结果转换为可用的轨迹格式

        ---
        处理流程:
            1. 提取跟踪 query 和检测框
            2. 将自车 (SDC) query 拼接到跟踪 query 末尾
            3. 提取地图 query
            4. 调用 self.forward() 进行 MotionFormer 预测
            5. 调用 get_trajs() 将预测转换为轨迹
            6. 分离自车 query，过滤只保留车辆目标

        Args:
            bev_embed: BEV 特征图
            outs_track: Track Head 输出
            outs_seg: Map/Seg Head 输出

        Returns:
            Tuple[list, dict]:
                - traj_results: 每个样本的轨迹预测（含每层解码器结果）
                - outs_motion: 运动预测输出（只含车辆目标，SDC 已分离）
        """
        # =====================================================================
        # 步骤 1: 提取跟踪结果
        # =====================================================================
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        track_boxes = outs_track['track_bbox_results']

        # =====================================================================
        # 步骤 2: 将自车 (SDC) query 拼接到跟踪 query 末尾
        # =====================================================================
        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
        sdc_track_boxes = outs_track['sdc_track_bbox_results']
        track_boxes[0][0].tensor = torch.cat(
            [track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat(
            [track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat(
            [track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat(
            [track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)

        # =====================================================================
        # 步骤 3: 提取地图查询
        # =====================================================================
        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']

        # =====================================================================
        # 步骤 4: MotionFormer 前向传播
        # =====================================================================
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)

        # =====================================================================
        # 步骤 5: 将预测结果转换为轨迹格式
        # =====================================================================
        traj_results = self.get_trajs(outs_motion, track_boxes)

        # 提取分数，并将 SDC 的 label 强制设为 0 (vehicle)
        bboxes, scores, labels, bbox_index, mask = track_boxes[0]
        outs_motion['track_scores'] = scores[None, :]
        labels[-1] = 0  # SDC 标签设为 vehicle

        # =====================================================================
        # 步骤 6: 过滤非车辆目标 & 分离 SDC
        # =====================================================================
        outs_motion = self._filter_vehicle_query_test(outs_motion, labels, self.vehicle_id_list)

        if outs_motion is not None:
            # 提取 SDC 专属 query
            outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]
            outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]
            outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]

            # 移除 SDC，只保留普通目标
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]
            outs_motion['track_query'] = outs_motion['track_query'][:, :-1]
            outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]
            outs_motion['track_scores'] = outs_motion['track_scores'][:, :-1]

        return traj_results, outs_motion

    @auto_fp16(apply_to=('bev_embed', 'track_query', 'lane_query', 'lane_query_pos', 'lane_query_embed', 'prev_bev'))
    def forward(self, bev_embed, track_query, lane_query, lane_query_pos, track_bbox_results):
        """运动预测核心前向传播 (MotionHead.forward)

        这是整个运动预测模块的入口函数，将跟踪 query 和地图 query 通过 MotionFormer
        结合，预测每个智能体未来的多模态轨迹。

        ---
        输入说明:
            bev_embed (Tensor):
                BEV 特征图，形状 (B, H*W, D)。来自 BEVFormer 编码器，
                是整个 UniAD 系统的共享感知表征。

            track_query (Tensor):
                跟踪查询，形状 (1, num_temporal_layers, N_agents, D)。
                来自 Track Head 的输出，最后一个智能体是自车(SDC)。
                num_temporal_layers 通常为 1（只取当前帧）。

            lane_query (Tensor):
                地图查询，形状 (B, M, D)。来自 Map/Seg Head 的 Location Decoder 输出，
                每个 256 维向量编码了一个地图元素（车道线/路口等）的几何和语义信息。

            lane_query_pos (Tensor):
                地图查询的位置编码，形状 (B, M, D)。

            track_bbox_results (List[Tuple]):
                检测框结果，每个 batch 样本一个元组 (bboxes, scores, labels, bbox_index, mask)。
                用于提取智能体中心点位置和在不同坐标系间转换轨迹。

        ---
        处理流程:
            步骤 1 — 编码跟踪目标中心点:
                提取每个智能体的 BEV 中心坐标 (x, y)，归一化后用 pos2posemb2d
                编码为高维正弦嵌入，作为 track_query_pos。

            步骤 2 — 构建三类锚点嵌入 (agent-level / scene-level ego / scene-level offset):
                加载 K-means 聚类得到的锚点轨迹 (num_groups × num_anchor × predict_steps × 2)，
                对每个智能体根据其类别（如 vehicle/pedestrian）选择对应分组的锚点。
                三类嵌入分别在三种坐标系下编码锚点轨迹终点的位置信息。

            步骤 3 — 根据目标类别选择对应的锚点 (group_mode_query_pos):
                使用 cls2group 映射，确保车辆用车辆锚点、行人用行人锚点。

            步骤 4 — MotionFormer 多模态交互:
                调用 self.motionformer (MotionTransformerDecoder) 进行 3 层迭代解码，
                每层执行: 意图交互 → 智能体交互 → 地图交互 → BEV 交互 → 轨迹细化。
                返回 inter_states (num_layers, B, N, P, D) 和 inter_references (num_layers, B, N, P, S, 2)。

            步骤 5 — 分类 + 回归:
                每层解码器的输出分别通过:
                - traj_cls_branches: 预测每条轨迹的置信度分数 (log_softmax)
                - traj_reg_branches: 预测轨迹位移增量，cumsum 后得到绝对坐标
                - bivariate_gaussian_activation: 对每个轨迹点拟合二元高斯分布 (μx, μy, σx, σy, ρ)

        ---
        轨迹格式:
            每个智能体预测 P=num_anchor=6 条可能的轨迹，每条轨迹 S=predict_steps=12 个时间步。
            每个时间步输出 5 个参数 (μx, μy, σx, σy, ρ)，即二元高斯分布参数。
            μx, μy: 轨迹点的期望位置（均值）
            σx, σy: 轨迹点的不确定性（标准差）
            ρ: x 和 y 之间的相关系数

        ---
        Returns:
            dict: 包含以下键的字典:
                - 'all_traj_scores' (Tensor):
                    每层解码器的轨迹置信度分数，形状 (num_layers, B, N, P)。
                    经过 log_softmax，值域 (-∞, 0]，越大表示该模态越可能。

                - 'all_traj_preds' (Tensor):
                    每层解码器的轨迹预测，形状 (num_layers, B, N, P, S, 5)。
                    最后一维: (μx, μy, σx, σy, ρ)，经过 bivariate_gaussian_activation。

                - 'valid_traj_masks' (Tensor):
                    有效轨迹掩码，形状 (B, N)，全 True 表示所有智能体都有效。

                - 'traj_query' (Tensor):
                    每层解码器的查询嵌入，形状 (num_layers, B, N, P, D)。
                    下游模块 (Occ Head, Planning Head) 使用最后一层的 traj_query。

                - 'track_query' (Tensor):
                    跟踪查询，形状 (B, N, D)。用于下游模块。

                - 'track_query_pos' (Tensor):
                    跟踪查询的位置编码，形状 (B, N, D)。用于下游模块。
        """
        dtype = track_query.dtype
        device = track_query.device
        num_groups = self.kmeans_anchors.shape[0]  # 锚点类别分组数（如 vehicle/pedestrian 等）

        # =====================================================================
        # 步骤 1: 编码跟踪目标中心点 → track_query_pos
        # =====================================================================
        # track_query 形状: (1, num_temporal_layers, N_agents, D)
        # 取 index=-1 即最后一帧的查询（通常 num_temporal_layers=1，只有当前帧）
        track_query = track_query[:, -1]  # (1, N_agents, D) → squeeze 后为 (N_agents, D)

        # 从检测框中提取每个智能体在 BEV 上的归一化中心坐标 (x, y) ∈ [0, 1]
        reference_points_track = self._extract_tracking_centers(track_bbox_results, self.pc_range)
        # pos2posemb2d: 将 2D 坐标编码为高维正弦/余弦位置嵌入
        # boxes_query_embedding_layer: MLP 将位置嵌入映射到 D=256 维
        track_query_pos = self.boxes_query_embedding_layer(pos2posemb2d(reference_points_track.to(device)))

        # =====================================================================
        # 步骤 2: 构建三类锚点嵌入
        # =====================================================================
        # 可学习的运动查询嵌入: (num_anchor * num_anchor_group, D)
        # 按 num_anchor 分割后得到 (num_anchor_group, num_anchor, D)
        learnable_query_pos = self.learnable_motion_query_embedding.weight.to(dtype)
        learnable_query_pos = torch.stack(torch.split(learnable_query_pos, self.num_anchor, dim=0))

        # 加载 K-means 锚点轨迹: (num_groups, num_anchor, predict_steps, 2)
        # 锚点轨迹是从训练数据中聚类得到的典型运动模式
        agent_level_anchors = self.kmeans_anchors.to(dtype).to(device).view(
            num_groups, self.num_anchor, self.predict_steps, 2).detach()

        # 对锚点轨迹做坐标变换，得到三种坐标系下的表示:
        #   scene_level_ego_anchors:   全局坐标系（平移变换，对齐世界坐标原点）
        #   scene_level_offset_anchors: 偏移量（不做变换，即相对于 agent 当前位置的偏移）
        #   agent_level_anchors:        agent 自身坐标系（原始锚点，不做额外变换）
        scene_level_ego_anchors = anchor_coordinate_transform(
            agent_level_anchors, track_bbox_results, with_translation_transform=True)
        scene_level_offset_anchors = anchor_coordinate_transform(
            agent_level_anchors, track_bbox_results, with_translation_transform=False)

        # 归一化锚点轨迹的终点坐标到 [0, 1] 范围
        agent_level_norm = norm_points(agent_level_anchors, self.pc_range)
        scene_level_ego_norm = norm_points(scene_level_ego_anchors, self.pc_range)
        scene_level_offset_norm = norm_points(scene_level_offset_anchors, self.pc_range)

        # 只用轨迹终点 [..., -1, :] 编码位置信息
        # pos2posemb2d: 2D 坐标 → 高维正弦/余弦嵌入
        # *_embedding_layer: MLP 将位置嵌入映射到 D 维空间
        # 结果形状: (num_groups, num_anchor, D)
        agent_level_embedding = self.agent_level_embedding_layer(
            pos2posemb2d(agent_level_norm[..., -1, :]))
        scene_level_ego_embedding = self.scene_level_ego_embedding_layer(
            pos2posemb2d(scene_level_ego_norm[..., -1, :]))
        scene_level_offset_embedding = self.scene_level_offset_embedding_layer(
            pos2posemb2d(scene_level_offset_norm[..., -1, :]))

        # =====================================================================
        # 步骤 3: 根据目标类别选择对应的锚点 (group_mode_query_pos)
        # =====================================================================
        # 将锚点嵌入从 (num_groups, num_anchor, D) 扩展到 (B, N_agents, num_groups, num_anchor, D)
        batch_size, num_agents = scene_level_ego_embedding.shape[:2]
        agent_level_embedding = agent_level_embedding[None, None, ...].expand(
            batch_size, num_agents, -1, -1, -1)
        learnable_embed = learnable_query_pos[None, None, ...].expand(
            batch_size, num_agents, -1, -1, -1)

        # group_mode_query_pos: 根据每个智能体的预测类别，从 num_groups 组锚点中
        # 选择对应的一组。例如 vehicle 选 vehicle 锚点，pedestrian 选 pedestrian 锚点。
        # 输入: (B, N, num_groups, num_anchor, ...) → 输出: (B, N, num_anchor, ...)
        # 相当于用 cls2group[label] 做索引，每组智能体只保留自己类别对应的锚点
        scene_level_offset_anchors = self.group_mode_query_pos(
            track_bbox_results, scene_level_offset_anchors)
        agent_level_embedding = self.group_mode_query_pos(
            track_bbox_results, agent_level_embedding)
        scene_level_ego_embedding = self.group_mode_query_pos(
            track_bbox_results, scene_level_ego_embedding)
        scene_level_offset_embedding = self.group_mode_query_pos(
            track_bbox_results, scene_level_offset_embedding)
        learnable_embed = self.group_mode_query_pos(
            track_bbox_results, learnable_embed)

        # 初始参考轨迹: 使用 scene_level_offset_anchors 作为起点
        # 形状: (B, N_agents, num_anchor, predict_steps, 2)
        init_reference = scene_level_offset_anchors.detach()

        outputs_traj_scores = []  # 每层解码器的轨迹置信度
        outputs_trajs = []        # 每层解码器的轨迹预测

        # =====================================================================
        # 步骤 4: MotionFormer — 多模态交互 + 轨迹迭代细化
        # =====================================================================
        # 调用 self.motionformer (即 MotionTransformerDecoder)，进行 3 层迭代解码。
        # 每层执行:
        #   1. 静态-动态意图融合
        #   2. 智能体间交互 (Cross-Attention: agent ↔ all agents)
        #   3. 智能体-地图交互 (Cross-Attention: agent ↔ lane_query)
        #   4. 智能体-BEV 交互 (Deformable Attention: query ↔ BEV features)
        #   5. 多源融合
        #   6. 轨迹迭代细化 (cumsum 位移增量)
        #   7. 位置嵌入更新
        #
        # 返回值:
        #   inter_states:    (num_layers, B, N_agents, num_anchor, D)  查询嵌入
        #   inter_references: (num_layers, B, N_agents, num_anchor, S, 2)  参考轨迹
        inter_states, inter_references = self.motionformer(
            track_query, lane_query,
            track_query_pos=track_query_pos, lane_query_pos=lane_query_pos,
            track_bbox_results=track_bbox_results, bev_embed=bev_embed,
            reference_trajs=init_reference,
            traj_reg_branches=self.traj_reg_branches,
            traj_cls_branches=self.traj_cls_branches,
            agent_level_embedding=agent_level_embedding,
            scene_level_ego_embedding=scene_level_ego_embedding,
            scene_level_offset_embedding=scene_level_offset_embedding,
            learnable_embed=learnable_embed,
            agent_level_embedding_layer=self.agent_level_embedding_layer,
            scene_level_ego_embedding_layer=self.scene_level_ego_embedding_layer,
            scene_level_offset_embedding_layer=self.scene_level_offset_embedding_layer,
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=device),
            level_start_index=torch.tensor([0], device=device))

        # =====================================================================
        # 步骤 5: 分类 + 回归 — 对每层解码器的输出分别预测
        # =====================================================================
        for lvl in range(inter_states.shape[0]):
            # 分类分支: 预测每条轨迹的置信度分数
            # inter_states[lvl]: (B, N_agents, num_anchor, D) → (B, N_agents, num_anchor, 1)
            outputs_class = self.traj_cls_branches[lvl](inter_states[lvl])

            # 回归分支: 预测轨迹位移增量
            # inter_states[lvl]: (B, N_agents, num_anchor, D) → (B, N_agents, num_anchor, S*5)
            tmp = self.traj_reg_branches[lvl](inter_states[lvl])
            tmp = self.unflatten_traj(tmp)  # (B, N_agents, num_anchor, S, 5)

            # cumsum: 将预测的位移增量 (Δx, Δy) 累积为绝对坐标
            # 例如: [Δ1, Δ2, Δ3, ...] → [Δ1, Δ1+Δ2, Δ1+Δ2+Δ3, ...]
            tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)

            # 对置信度做 log_softmax，得到归一化的对数概率（在模态维度上）
            outputs_class = self.log_softmax(outputs_class.squeeze(3))
            outputs_traj_scores.append(outputs_class)

            # 二元高斯激活: 将原始输出转换为高斯分布参数
            #   μx, μy: 使用 sigmoid 确保在合理范围内
            #   σx, σy: 使用 exp 确保为正
            #   ρ: 使用 tanh 确保在 [-1, 1] 之间
            for bs in range(tmp.shape[0]):
                tmp[bs] = bivariate_gaussian_activation(tmp[bs])
            outputs_trajs.append(tmp)

        outputs_traj_scores = torch.stack(outputs_traj_scores)  # (num_layers, B, N, P)
        outputs_trajs = torch.stack(outputs_trajs)              # (num_layers, B, N, P, S, 5)

        B, A_track, D = track_query.shape
        valid_traj_masks = track_query.new_ones((B, A_track)) > 0  # 全 True 掩码

        # =====================================================================
        # 组装输出
        # =====================================================================
        outs = {
            'all_traj_scores': outputs_traj_scores,   # 轨迹置信度（每层都有）
            'all_traj_preds': outputs_trajs,           # 轨迹预测（二元高斯参数）
            'valid_traj_masks': valid_traj_masks,      # 有效轨迹掩码
            'traj_query': inter_states,               # 查询嵌入（供 Occ Head/Planning Head 使用）
            'track_query': track_query,               # 跟踪查询（供下游模块使用）
            'track_query_pos': track_query_pos,       # 跟踪查询位置编码（供下游模块使用）
        }
        return outs

    def group_mode_query_pos(self, bbox_results, mode_query_pos):
        """根据目标类别分组选择对应的锚点嵌入

        UniAD 对不同类别的智能体使用不同的 K-means 锚点轨迹。
        例如车辆有 6 种典型运动模式（直行/左转/右转/掉头/加速/减速），
        行人则有另外 6 种模式（步行/跑步/站立/横穿马路等）。

        该函数使用 cls2group 映射表，从 num_groups 组锚点中为每个智能体
        选择其类别对应的那一组。

        ---
        工作流程:
            1. 对每个 batch 样本，提取该样本中每个智能体的预测类别 label
            2. 用 cls2group[label] 将类别 ID 映射到分组 ID
               （例如 cls2group[car_id] = 0, cls2group[pedestrian_id] = 1）
            3. 用分组 ID 作为索引，从 mode_query_pos 的 num_groups 维中
               取出对应的一组锚点

        Args:
            bbox_results (List[Tuple]): 检测框结果，用于提取每个智能体的类别标签
            mode_query_pos (Tensor): 锚点嵌入，形状 (B, N_agents, num_groups, num_anchor, ...)

        Returns:
            Tensor: 分组后的锚点嵌入，形状 (B, N_agents, num_anchor, ...)
                每个智能体只保留其类别对应的 num_anchor 个锚点
        """
        batch_size = len(bbox_results)
        agent_num = mode_query_pos.shape[1]
        batched_mode_query_pos = []
        self.cls2group = self.cls2group.to(mode_query_pos.device)
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            label = labels.to(mode_query_pos.device)
            # cls2group[label]: 将类别 ID 映射到锚点分组 ID
            grouped_label = self.cls2group[label]  # (N_agents,)
            grouped_mode_query_pos = []
            for j in range(agent_num):
                # 为第 j 个智能体选择其类别对应的锚点
                # mode_query_pos[i, j, grouped_label[j]]: (num_anchor, ...)
                grouped_mode_query_pos.append(
                    mode_query_pos[i, j, grouped_label[j]])
            batched_mode_query_pos.append(torch.stack(grouped_mode_query_pos))
        return torch.stack(batched_mode_query_pos)

    @force_fp32(apply_to=('preds_dicts_motion'))
    def loss(self, gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, preds_dicts_motion, all_matched_idxes, track_bbox_results):
        """轨迹预测损失计算

        对每层解码器的输出都计算损失，但只有最后一层的损失用于反向传播。
        中间层的损失作为辅助监督信号，记录在 loss_dict 中以 '{dN}.loss_*' 命名。

        ---
        损失组成:
            - l_class: 轨迹分类损失（交叉熵），判断哪条轨迹模态最接近 GT
            - l_reg: 轨迹回归损失（负对数似然），用二元高斯分布拟合 GT 轨迹
            - min_ade: Minimum Average Displacement Error（最小平均位移误差）
            - min_fde: Minimum Final Displacement Error（最小终点位移误差）
            - mr: Miss Rate（未命中率），预测的 6 条轨迹中是否有一条足够接近 GT

        Args:
            gt_bboxes_3d: GT 3D 检测框
            gt_fut_traj: GT 未来轨迹
            gt_fut_traj_mask: GT 未来轨迹有效掩码
            preds_dicts_motion: MotionHead 的输出，包含 all_traj_scores 和 all_traj_preds
            all_matched_idxes: 每个 query 匹配的 GT 索引
            track_bbox_results: 检测框结果

        Returns:
            dict: 损失字典，包含:
                - 'loss_traj': 总轨迹损失（最后一层，用于反向传播）
                - 'l_class': 分类损失
                - 'l_reg': 回归损失
                - 'min_ade': 最小平均位移误差（监控指标）
                - 'min_fde': 最小终点位移误差（监控指标）
                - 'mr': 未命中率（监控指标）
                - 'd{0..N-1}.loss_traj' 等: 中间解码器层的损失（仅监控）
        """
        all_traj_scores = preds_dicts_motion['all_traj_scores']
        all_traj_preds = preds_dicts_motion['all_traj_preds']
        num_dec_layers = len(all_traj_scores)

        # 将 GT 轨迹复制 num_dec_layers 份，每层使用相同的 GT
        all_gt_fut_traj = [gt_fut_traj for _ in range(num_dec_layers)]
        all_gt_fut_traj_mask = [gt_fut_traj_mask for _ in range(num_dec_layers)]

        losses_traj = []
        # 根据匹配索引提取 GT 轨迹（只取匹配到的目标，忽略未匹配的）
        gt_fut_traj_all, gt_fut_traj_mask_all = self.compute_matched_gt_traj(
            all_gt_fut_traj[0], all_gt_fut_traj_mask[0], all_matched_idxes,
            track_bbox_results, gt_bboxes_3d)

        for i in range(num_dec_layers):
            loss_traj, l_class, l_reg, l_mindae, l_minfde, l_mr = self.compute_loss_traj(
                all_traj_scores[i], all_traj_preds[i],
                gt_fut_traj_all, gt_fut_traj_mask_all, all_matched_idxes)
            losses_traj.append((loss_traj, l_class, l_reg, l_mindae, l_minfde, l_mr))

        loss_dict = dict()
        # 最后一层作为主损失（用于反向传播）
        loss_dict['loss_traj'] = losses_traj[-1][0]
        loss_dict['l_class'] = losses_traj[-1][1]
        loss_dict['l_reg'] = losses_traj[-1][2]
        loss_dict['min_ade'] = losses_traj[-1][3]    # Minimum Average Displacement Error
        loss_dict['min_fde'] = losses_traj[-1][4]    # Minimum Final Displacement Error
        loss_dict['mr'] = losses_traj[-1][5]          # Miss Rate

        # 中间层损失作为辅助监督信号（仅记录，不用于反向传播）
        num_dec_layer = 0
        for loss_traj_i in losses_traj[:-1]:
            loss_dict[f'd{num_dec_layer}.loss_traj'] = loss_traj_i[0]
            loss_dict[f'd{num_dec_layer}.l_class'] = loss_traj_i[1]
            loss_dict[f'd{num_dec_layer}.l_reg'] = loss_traj_i[2]
            loss_dict[f'd{num_dec_layer}.min_ade'] = loss_traj_i[3]
            loss_dict[f'd{num_dec_layer}.min_fde'] = loss_traj_i[4]
            loss_dict[f'd{num_dec_layer}.mr'] = loss_traj_i[5]
            num_dec_layer += 1

        return loss_dict

    def compute_matched_gt_traj(self, gt_fut_traj, gt_fut_traj_mask, all_matched_idxes, track_bbox_results, gt_bboxes_3d):
        """根据匹配索引提取 GT 轨迹（只取与检测框匹配上的目标）

        由于 Track Head 的 DETR 匹配机制，并非所有检测 query 都有对应的 GT。
        该函数使用 all_matched_idxes 筛选出匹配到的目标，只保留其 GT 轨迹用于损失计算。

        Args:
            gt_fut_traj: 所有 GT 未来轨迹
            gt_fut_traj_mask: GT 轨迹有效掩码
            all_matched_idxes: 每个 query 对应的 GT 索引（-1 表示未匹配）
            track_bbox_results: 检测框结果
            gt_bboxes_3d: GT 3D 检测框

        Returns:
            Tuple[Tensor, Tensor]:
                - gt_fut_traj_all: 匹配到的 GT 轨迹 (N_matched, S, 2)
                - gt_fut_traj_mask_all: 对应的有效掩码 (N_matched,)
        """
        num_imgs = len(all_matched_idxes)
        gt_fut_traj_all = []
        gt_fut_traj_mask_all = []
        for i in range(num_imgs):
            matched_gt_idx = all_matched_idxes[i]
            valid_traj_masks = matched_gt_idx >= 0  # 过滤未匹配的 query
            matched_gt_fut_traj = gt_fut_traj[i][matched_gt_idx][valid_traj_masks]
            matched_gt_fut_traj_mask = gt_fut_traj_mask[i][matched_gt_idx][valid_traj_masks]

            if self.use_nonlinear_optimizer:
                # 可选: 使用非线性优化平滑轨迹
                bboxes = track_bbox_results[i][0].tensor[:len(valid_traj_masks)].to(
                    valid_traj_masks.device)[valid_traj_masks]
                matched_tensor = gt_bboxes_3d[i][-1].tensor
                matched_indices = matched_gt_idx[:-1]
                valid_masks = valid_traj_masks[:-1]
                matched_gt_bboxes_3d = matched_tensor.to(valid_masks.device)[
                    matched_indices.to(valid_masks.device)][valid_masks]
                sdc_gt_fut_traj = matched_gt_fut_traj[-1:]
                sdc_gt_fut_traj_mask = matched_gt_fut_traj_mask[-1:]
                matched_gt_fut_traj = matched_gt_fut_traj[:-1]
                matched_gt_fut_traj_mask = matched_gt_fut_traj_mask[:-1]
                bboxes = bboxes[:-1]
                matched_gt_fut_traj, matched_gt_fut_traj_mask = nonlinear_smoother(
                    matched_gt_bboxes_3d, matched_gt_fut_traj, matched_gt_fut_traj_mask, bboxes)
                matched_gt_fut_traj = torch.cat([matched_gt_fut_traj, sdc_gt_fut_traj], dim=0)
                matched_gt_fut_traj_mask = torch.cat(
                    [matched_gt_fut_traj_mask, sdc_gt_fut_traj_mask], dim=0)

            # 轨迹掩码: 所有时间步都有效才算有效
            matched_gt_fut_traj_mask = torch.all(matched_gt_fut_traj_mask > 0, dim=-1)
            gt_fut_traj_all.append(matched_gt_fut_traj)
            gt_fut_traj_mask_all.append(matched_gt_fut_traj_mask)
        gt_fut_traj_all = torch.cat(gt_fut_traj_all, dim=0)
        gt_fut_traj_mask_all = torch.cat(gt_fut_traj_mask_all, dim=0)
        return gt_fut_traj_all, gt_fut_traj_mask_all

    def compute_loss_traj(self, traj_scores, traj_preds, gt_fut_traj_all, gt_fut_traj_mask_all, all_matched_idxes):
        """计算轨迹损失"""
        num_imgs = traj_scores.size(0)
        traj_prob_all = []
        traj_preds_all = []
        for i in range(num_imgs):
            matched_gt_idx = all_matched_idxes[i]
            valid_traj_masks = matched_gt_idx >= 0
            batch_traj_prob = traj_scores[i, valid_traj_masks, :]
            batch_traj_preds = traj_preds[i, valid_traj_masks, ...]
            traj_prob_all.append(batch_traj_prob)
            traj_preds_all.append(batch_traj_preds)
        traj_prob_all = torch.cat(traj_prob_all, dim=0)
        traj_preds_all = torch.cat(traj_preds_all, dim=0)
        traj_loss, l_class, l_reg, l_minade, l_minfde, l_mr = self.loss_traj(
            traj_prob_all, traj_preds_all, gt_fut_traj_all, gt_fut_traj_mask_all)
        return traj_loss, l_class, l_reg, l_minade, l_minfde, l_mr

    @force_fp32(apply_to=('preds_dicts'))
    def get_trajs(self, preds_dicts, bbox_results):
        """从预测结果生成轨迹"""
        num_samples = len(bbox_results)
        num_layers = preds_dicts['all_traj_preds'].shape[0]
        ret_list = []
        for i in range(num_samples):
            preds = dict()
            for j in range(num_layers):
                subfix = '_' + str(j) if j < (num_layers - 1) else ''
                traj = preds_dicts['all_traj_preds'][j, i]
                traj_scores = preds_dicts['all_traj_scores'][j, i]
                traj_scores, traj = traj_scores.cpu(), traj.cpu()
                preds['traj' + subfix] = traj
                preds['traj_scores' + subfix] = traj_scores
            ret_list.append(preds)
        return ret_list