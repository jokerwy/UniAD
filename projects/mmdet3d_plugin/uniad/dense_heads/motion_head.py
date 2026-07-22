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

        流程:
        1. 提取跟踪 query 和建图 query
        2. 将自车 query 拼接到跟踪 query 末尾
        3. MotionFormer 预测轨迹
        4. 计算轨迹损失
        5. 过滤车辆 query，只保留车辆目标
        """
        # 提取跟踪结果: query、匹配索引、框
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        all_matched_idxes = [outs_track['track_query_matched_idxes']]
        track_boxes = outs_track['track_bbox_results']

        # 将自车 (SDC) query 拼接到跟踪 query 末尾
        sdc_match_index = torch.zeros((1,), dtype=all_matched_idxes[0].dtype, device=all_matched_idxes[0].device)
        sdc_match_index[0] = gt_fut_traj[0].shape[0]
        all_matched_idxes = [torch.cat([all_matched_idxes[0], sdc_match_index], dim=0)]
        gt_fut_traj[0] = torch.cat([gt_fut_traj[0], gt_sdc_fut_traj[0]], dim=0)
        gt_fut_traj_mask[0] = torch.cat([gt_fut_traj_mask[0], gt_sdc_fut_traj_mask[0]], dim=0)
        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
        sdc_track_boxes = outs_track['sdc_track_bbox_results']
        track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)

        # 提取建图结果: 车道线 query
        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']

        # MotionFormer 前向传播
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        loss_inputs = [gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, outs_motion, all_matched_idxes, track_boxes]
        losses = self.loss(*loss_inputs)

        # 分离自车 query
        all_matched_idxes[0] = all_matched_idxes[0][:-1]
        outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]
        outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]
        outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]
        outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]
        outs_motion['track_query'] = outs_motion['track_query'][:, :-1]
        outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]

        # 过滤只保留车辆目标
        outs_motion, all_matched_idxes = self._filter_vehicle_query(outs_motion, all_matched_idxes, gt_labels_3d, self.vehicle_id_list)
        outs_motion['all_matched_idxes'] = all_matched_idxes

        ret_dict = dict(losses=losses, outs_motion=outs_motion, track_boxes=track_boxes)
        return ret_dict

    def _filter_vehicle_query(self, outs_motion, all_matched_idxes, gt_labels_3d, vehicle_id_list):
        """过滤车辆 query: 只保留车辆类别的目标"""
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
        """推理前向传播"""
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        track_boxes = outs_track['track_bbox_results']

        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
        sdc_track_boxes = outs_track['sdc_track_bbox_results']
        track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)

        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        traj_results = self.get_trajs(outs_motion, track_boxes)
        bboxes, scores, labels, bbox_index, mask = track_boxes[0]
        outs_motion['track_scores'] = scores[None, :]
        labels[-1] = 0

        # 过滤非车辆目标
        outs_motion = self._filter_vehicle_query_test(outs_motion, labels, self.vehicle_id_list)

        # 分离自车 query
        if outs_motion is not None:
            outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]
            outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]
            outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]
            outs_motion['track_query'] = outs_motion['track_query'][:, :-1]
            outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]
            outs_motion['track_scores'] = outs_motion['track_scores'][:, :-1]

        return traj_results, outs_motion

    @auto_fp16(apply_to=('bev_embed', 'track_query', 'lane_query', 'lane_query_pos', 'lane_query_embed', 'prev_bev'))
    def forward(self, bev_embed, track_query, lane_query, lane_query_pos, track_bbox_results):
        """运动预测核心前向传播

        使用 MotionFormer 将跟踪 query 和车道线 query 结合，
        预测每个目标的未来轨迹。

        关键步骤:
        1. 编码跟踪目标中心点
        2. 构建 anchor 嵌入 (agent-level, scene-level ego, scene-level offset)
        3. 根据目标类别选择对应的 anchor
        4. MotionFormer: 交叉注意力融合 track + lane 信息
        5. 轨迹预测: cumsum 累积位移 → 绝对位置
        """
        dtype = track_query.dtype
        device = track_query.device
        num_groups = self.kmeans_anchors.shape[0]

        track_query = track_query[:, -1]

        # 编码跟踪目标中心点
        reference_points_track = self._extract_tracking_centers(track_bbox_results, self.pc_range)
        track_query_pos = self.boxes_query_embedding_layer(pos2posemb2d(reference_points_track.to(device)))

        # 可学习的 query 位置编码
        learnable_query_pos = self.learnable_motion_query_embedding.weight.to(dtype)
        learnable_query_pos = torch.stack(torch.split(learnable_query_pos, self.num_anchor, dim=0))

        # 构建 anchor 嵌入 (三种级别)
        agent_level_anchors = self.kmeans_anchors.to(dtype).to(device).view(num_groups, self.num_anchor, self.predict_steps, 2).detach()
        scene_level_ego_anchors = anchor_coordinate_transform(agent_level_anchors, track_bbox_results, with_translation_transform=True)
        scene_level_offset_anchors = anchor_coordinate_transform(agent_level_anchors, track_bbox_results, with_translation_transform=False)

        agent_level_norm = norm_points(agent_level_anchors, self.pc_range)
        scene_level_ego_norm = norm_points(scene_level_ego_anchors, self.pc_range)
        scene_level_offset_norm = norm_points(scene_level_offset_anchors, self.pc_range)

        agent_level_embedding = self.agent_level_embedding_layer(pos2posemb2d(agent_level_norm[..., -1, :]))
        scene_level_ego_embedding = self.scene_level_ego_embedding_layer(pos2posemb2d(scene_level_ego_norm[..., -1, :]))
        scene_level_offset_embedding = self.scene_level_offset_embedding_layer(pos2posemb2d(scene_level_offset_norm[..., -1, :]))

        batch_size, num_agents = scene_level_ego_embedding.shape[:2]
        agent_level_embedding = agent_level_embedding[None,None,...].expand(batch_size, num_agents, -1, -1, -1)
        learnable_embed = learnable_query_pos[None, None, ...].expand(batch_size, num_agents, -1, -1, -1)

        scene_level_offset_anchors = self.group_mode_query_pos(track_bbox_results, scene_level_offset_anchors)
        agent_level_embedding = self.group_mode_query_pos(track_bbox_results, agent_level_embedding)
        scene_level_ego_embedding = self.group_mode_query_pos(track_bbox_results, scene_level_ego_embedding)
        scene_level_offset_embedding = self.group_mode_query_pos(track_bbox_results, scene_level_offset_embedding)
        learnable_embed = self.group_mode_query_pos(track_bbox_results, learnable_embed)

        init_reference = scene_level_offset_anchors.detach()

        outputs_traj_scores = []
        outputs_trajs = []

        # MotionFormer: 交叉注意力融合
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

        for lvl in range(inter_states.shape[0]):
            outputs_class = self.traj_cls_branches[lvl](inter_states[lvl])
            tmp = self.traj_reg_branches[lvl](inter_states[lvl])
            tmp = self.unflatten_traj(tmp)

            # cumsum: 累积位移 → 绝对位置
            tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)

            outputs_class = self.log_softmax(outputs_class.squeeze(3))
            outputs_traj_scores.append(outputs_class)

            for bs in range(tmp.shape[0]):
                tmp[bs] = bivariate_gaussian_activation(tmp[bs])  # 二元高斯激活
            outputs_trajs.append(tmp)

        outputs_traj_scores = torch.stack(outputs_traj_scores)
        outputs_trajs = torch.stack(outputs_trajs)

        B, A_track, D = track_query.shape
        valid_traj_masks = track_query.new_ones((B, A_track)) > 0

        outs = {
            'all_traj_scores': outputs_traj_scores,
            'all_traj_preds': outputs_trajs,
            'valid_traj_masks': valid_traj_masks,
            'traj_query': inter_states,
            'track_query': track_query,
            'track_query_pos': track_query_pos,
        }
        return outs

    def group_mode_query_pos(self, bbox_results, mode_query_pos):
        """根据目标类别分组选择对应的 anchor 嵌入"""
        batch_size = len(bbox_results)
        agent_num = mode_query_pos.shape[1]
        batched_mode_query_pos = []
        self.cls2group = self.cls2group.to(mode_query_pos.device)
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            label = labels.to(mode_query_pos.device)
            grouped_label = self.cls2group[label]
            grouped_mode_query_pos = []
            for j in range(agent_num):
                grouped_mode_query_pos.append(mode_query_pos[i, j, grouped_label[j]])
            batched_mode_query_pos.append(torch.stack(grouped_mode_query_pos))
        return torch.stack(batched_mode_query_pos)

    @force_fp32(apply_to=('preds_dicts_motion'))
    def loss(self, gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, preds_dicts_motion, all_matched_idxes, track_bbox_results):
        """轨迹预测损失计算"""
        all_traj_scores = preds_dicts_motion['all_traj_scores']
        all_traj_preds = preds_dicts_motion['all_traj_preds']
        num_dec_layers = len(all_traj_scores)

        all_gt_fut_traj = [gt_fut_traj for _ in range(num_dec_layers)]
        all_gt_fut_traj_mask = [gt_fut_traj_mask for _ in range(num_dec_layers)]

        losses_traj = []
        gt_fut_traj_all, gt_fut_traj_mask_all = self.compute_matched_gt_traj(
            all_gt_fut_traj[0], all_gt_fut_traj_mask[0], all_matched_idxes, track_bbox_results, gt_bboxes_3d)
        for i in range(num_dec_layers):
            loss_traj, l_class, l_reg, l_mindae, l_minfde, l_mr = self.compute_loss_traj(
                all_traj_scores[i], all_traj_preds[i], gt_fut_traj_all, gt_fut_traj_mask_all, all_matched_idxes)
            losses_traj.append((loss_traj, l_class, l_reg, l_mindae, l_minfde, l_mr))

        loss_dict = dict()
        loss_dict['loss_traj'] = losses_traj[-1][0]
        loss_dict['l_class'] = losses_traj[-1][1]
        loss_dict['l_reg'] = losses_traj[-1][2]
        loss_dict['min_ade'] = losses_traj[-1][3]    # Minimum Average Displacement Error
        loss_dict['min_fde'] = losses_traj[-1][4]    # Minimum Final Displacement Error
        loss_dict['mr'] = losses_traj[-1][5]          # Miss Rate

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
        """根据匹配索引提取 GT 轨迹"""
        num_imgs = len(all_matched_idxes)
        gt_fut_traj_all = []
        gt_fut_traj_mask_all = []
        for i in range(num_imgs):
            matched_gt_idx = all_matched_idxes[i]
            valid_traj_masks = matched_gt_idx >= 0
            matched_gt_fut_traj = gt_fut_traj[i][matched_gt_idx][valid_traj_masks]
            matched_gt_fut_traj_mask = gt_fut_traj_mask[i][matched_gt_idx][valid_traj_masks]
            if self.use_nonlinear_optimizer:
                bboxes = track_bbox_results[i][0].tensor[:len(valid_traj_masks)].to(valid_traj_masks.device)[valid_traj_masks]
                matched_tensor = gt_bboxes_3d[i][-1].tensor
                matched_indices = matched_gt_idx[:-1]
                valid_masks = valid_traj_masks[:-1]
                matched_gt_bboxes_3d = matched_tensor.to(valid_masks.device)[matched_indices.to(valid_masks.device)][valid_masks]
                sdc_gt_fut_traj = matched_gt_fut_traj[-1:]
                sdc_gt_fut_traj_mask = matched_gt_fut_traj_mask[-1:]
                matched_gt_fut_traj = matched_gt_fut_traj[:-1]
                matched_gt_fut_traj_mask = matched_gt_fut_traj_mask[:-1]
                bboxes = bboxes[:-1]
                matched_gt_fut_traj, matched_gt_fut_traj_mask = nonlinear_smoother(
                    matched_gt_bboxes_3d, matched_gt_fut_traj, matched_gt_fut_traj_mask, bboxes)
                matched_gt_fut_traj = torch.cat([matched_gt_fut_traj, sdc_gt_fut_traj], dim=0)
                matched_gt_fut_traj_mask = torch.cat([matched_gt_fut_traj_mask, sdc_gt_fut_traj_mask], dim=0)
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