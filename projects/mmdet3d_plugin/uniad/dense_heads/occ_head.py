"""
占据预测头 (OccHead)
====================
预测未来帧的 3D 占据栅格 (Occupancy Grid)，推断被遮挡区域的状态。

核心流程:
    1. BEV 特征采样: 从 BEV 特征中提取感兴趣区域
    2. 未来帧预测: 通过时序 Transformer 逐帧预测未来占据状态
    3. 实例分割: 预测每个目标的占据掩码

网络架构:
    BEV 特征 → BevFeatureSlicer → 下采样 → Transformer Decoder (逐未来帧) → 上采样 → 占据预测

关键技术:
    - 注意力掩码: 用预测的掩码限制 Transformer 的注意力范围，只关注前景区域
    - 时序查询: 每个未来帧有独立的 temporal_mlp 和 downscale_conv
    - 多模态查询融合: 将轨迹查询、跟踪查询和位置编码融合为占据查询
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule
from einops import rearrange
from mmdet.core import reduce_mean
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
import copy
from .occ_head_plugin import MLP, BevFeatureSlicer, SimpleConv2d, CVT_Decoder, Bottleneck, UpsamplingAdd, \
                             predict_instance_segmentation_and_trajectories

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(self,
                 # 通用参数
                 receptive_field=3,         # 感受野帧数 (过去帧数)
                 n_future=4,                # 未来预测帧数
                 spatial_extent=(50, 50),   # 空间范围
                 ignore_index=255,          # 忽略标签

                 # BEV 参数
                 grid_conf=None,            # 网格配置
                 bev_size=(200, 200),       # BEV 尺寸
                 bev_emb_dim=256,           # BEV 嵌入维度
                 bev_proj_dim=64,           # BEV 投影维度
                 bev_proj_nlayers=1,       # BEV 投影层数

                 # Query 参数
                 query_dim=256,             # 查询维度
                 query_mlp_layers=3,       # 查询 MLP 层数
                 detach_query_pos=True,     # 是否 detach 查询位置
                 temporal_mlp_layer=2,     # 时序 MLP 层数

                 # Transformer 参数
                 transformer_decoder=None,  # Transformer Decoder 配置
                 attn_mask_thresh=0.5,      # 注意力掩码阈值

                 # 损失参数
                 sample_ignore_mode='all_valid',
                 aux_loss_weight=1.,       # 辅助损失权重
                 loss_mask=None,           # 掩码损失
                 loss_dice=None,           # Dice 损失

                 # 配置
                 init_cfg=None,

                 # 评估参数
                 pan_eval=False,           # 是否全景评估
                 test_seg_thresh=0.5,      # 测试分割阈值
                 test_with_track_score=False,  # 测试时是否使用跟踪分数
                 ):
        assert init_cfg is None, 'To prevent abnormal initialization behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)
        self.receptive_field = receptive_field
        self.n_future = n_future
        self.spatial_extent = spatial_extent
        self.ignore_index = ignore_index

        # BEV 特征采样器: 从 BEV 特征中提取感兴趣区域
        bevformer_bev_conf = {
            'xbound': [-51.2, 51.2, 0.512],
            'ybound': [-51.2, 51.2, 0.512],
            'zbound': [-10.0, 10.0, 20.0],
        }
        self.bev_sampler = BevFeatureSlicer(bevformer_bev_conf, grid_conf)

        self.bev_size = bev_size
        self.bev_proj_dim = bev_proj_dim

        # BEV 特征投影 (降维)
        if bev_proj_nlayers == 0:
            self.bev_light_proj = nn.Sequential()
        else:
            self.bev_light_proj = SimpleConv2d(
                in_channels=bev_emb_dim, conv_channels=bev_emb_dim,
                out_channels=bev_proj_dim, num_conv=bev_proj_nlayers)

        # 下采样 BEV 特征 (/4)
        self.base_downscale = nn.Sequential(
            Bottleneck(in_channels=bev_proj_dim, downsample=True),
            Bottleneck(in_channels=bev_proj_dim, downsample=True))

        # 未来帧块数 (包括当前帧)
        self.n_future_blocks = self.n_future + 1

        self.attn_mask_thresh = attn_mask_thresh
        self.num_trans_layers = transformer_decoder.num_layers
        assert self.num_trans_layers % self.n_future_blocks == 0

        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)

        # 时序 MLP: 将上一帧的查询转换为当前帧的查询
        temporal_mlp = MLP(query_dim, query_dim, bev_proj_dim, num_layers=temporal_mlp_layer)
        self.temporal_mlps = _get_clones(temporal_mlp, self.n_future_blocks)

        # 下采样卷积: 每帧进一步下采样
        downscale_conv = Bottleneck(in_channels=bev_proj_dim, downsample=True)
        self.downscale_convs = _get_clones(downscale_conv, self.n_future_blocks)

        # 上采样: 恢复分辨率
        upsample_add = UpsamplingAdd(in_channels=bev_proj_dim, out_channels=bev_proj_dim)
        self.upsample_adds = _get_clones(upsample_add, self.n_future_blocks)

        # 密集解码器: 将特征解码为最终占据预测
        self.dense_decoder = CVT_Decoder(dim=bev_proj_dim, blocks=[bev_proj_dim, bev_proj_dim])

        # Query 融合: 将多模态查询融合为占据查询
        self.mode_fuser = nn.Sequential(
            nn.Linear(query_dim, bev_proj_dim), nn.LayerNorm(bev_proj_dim), nn.ReLU(inplace=True))
        self.multi_query_fuser = nn.Sequential(
            nn.Linear(query_dim * 3, query_dim * 2), nn.LayerNorm(query_dim * 2),
            nn.ReLU(inplace=True), nn.Linear(query_dim * 2, bev_proj_dim))

        self.detach_query_pos = detach_query_pos
        self.query_to_occ_feat = MLP(query_dim, query_dim, bev_proj_dim, num_layers=query_mlp_layers)
        self.temporal_mlp_for_mask = copy.deepcopy(self.query_to_occ_feat)

        # 损失
        self.sample_ignore_mode = sample_ignore_mode
        self.aux_loss_weight = aux_loss_weight
        self.loss_dice = build_loss(loss_dice)
        self.loss_mask = build_loss(loss_mask)

        self.pan_eval = pan_eval
        self.test_seg_thresh = test_seg_thresh
        self.test_with_track_score = test_with_track_score
        self.init_weights()

    def init_weights(self):
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def get_attn_mask(self, state, ins_query):
        """生成注意力掩码

        使用预测的实例掩码限制 Transformer 的交叉注意力范围，
        只让 query 关注前景区域，提高计算效率。

        Args:
            state: BEV 特征 (b, c, h, w)
            ins_query: 实例查询 (b, q, c)

        Returns:
            attn_mask: 注意力掩码 (用于 Transformer)
            upsampled_mask_pred: 上采样后的掩码预测 (用于监督)
            ins_embed: 实例嵌入
        """
        ins_embed = self.temporal_mlp_for_mask(ins_query)
        mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed, state)
        attn_mask = mask_pred.sigmoid() < self.attn_mask_thresh
        attn_mask = rearrange(attn_mask, 'b q h w -> b (h w) q').unsqueeze(1).repeat(
            1, self.num_heads, 1, 1).flatten(0, 1)
        attn_mask = attn_mask.detach()

        # 如果某个掩码全部为 True (全背景)，则设为 False (允许看到全部)
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        upsampled_mask_pred = F.interpolate(mask_pred, self.bev_size, mode='bilinear', align_corners=False)
        return attn_mask, upsampled_mask_pred, ins_embed

    def forward(self, x, ins_query):
        """占据预测前向传播

        逐未来帧预测占据状态，利用 Transformer 融合时序信息。

        Args:
            x: BEV 特征 (h*w, b, d)
            ins_query: 实例查询 (b, q, d)

        Returns:
            mask_preds: 各帧的掩码预测
            ins_occ_logits: 实例占据 logits
        """
        base_state = rearrange(x, '(h w) b d -> b d h w', h=self.bev_size[0])
        base_state = self.bev_sampler(base_state)
        base_state = self.bev_light_proj(base_state)
        base_state = self.base_downscale(base_state)
        base_ins_query = ins_query

        last_state = base_state
        last_ins_query = base_ins_query
        future_states = []
        mask_preds = []
        temporal_query = []
        temporal_embed_for_mask_attn = []
        n_trans_layer_each_block = self.num_trans_layers // self.n_future_blocks
        assert n_trans_layer_each_block >= 1

        for i in range(self.n_future_blocks):
            # 下采样
            cur_state = self.downscale_convs[i](last_state)

            # 时序感知的实例查询
            cur_ins_query = self.temporal_mlps[i](last_ins_query)
            temporal_query.append(cur_ins_query)

            # 生成注意力掩码
            attn_mask, mask_pred, cur_ins_emb_for_mask_attn = self.get_attn_mask(cur_state, cur_ins_query)
            attn_masks = [None, attn_mask]
            mask_preds.append(mask_pred)
            temporal_embed_for_mask_attn.append(cur_ins_emb_for_mask_attn)

            cur_state = rearrange(cur_state, 'b c h w -> (h w) b c')
            cur_ins_query = rearrange(cur_ins_query, 'b q c -> q b c')

            # Transformer 交叉注意力: BEV 特征与实例查询交互
            for j in range(n_trans_layer_each_block):
                trans_layer_ind = i * n_trans_layer_each_block + j
                trans_layer = self.transformer_decoder.layers[trans_layer_ind]
                cur_state = trans_layer(
                    query=cur_state, key=cur_ins_query, value=cur_ins_query,
                    query_pos=None, key_pos=None, attn_masks=attn_masks,
                    query_key_padding_mask=None, key_padding_mask=None)

            cur_state = rearrange(cur_state, '(h w) b c -> b c h w', h=self.bev_size[0]//8)
            cur_state = self.upsample_adds[i](cur_state, last_state)
            future_states.append(cur_state)
            last_state = cur_state

        future_states = torch.stack(future_states, dim=1)
        temporal_query = torch.stack(temporal_query, dim=1)
        mask_preds = torch.stack(mask_preds, dim=2)
        ins_query = torch.stack(temporal_embed_for_mask_attn, dim=1)

        future_states = self.dense_decoder(future_states)
        ins_occ_query = self.query_to_occ_feat(ins_query)

        # 实例占据预测: 查询与特征的内积
        ins_occ_logits = torch.einsum("btqc,btchw->bqthw", ins_occ_query, future_states)

        return mask_preds, ins_occ_logits

    def merge_queries(self, outs_dict, detach_query_pos=True):
        """融合多模态查询

        将轨迹查询、跟踪查询和跟踪位置编码融合为占据查询。
        轨迹查询取最大模态，然后与跟踪查询拼接。
        """
        ins_query = outs_dict.get('traj_query', None)
        track_query = outs_dict['track_query']
        track_query_pos = outs_dict['track_query_pos']

        if detach_query_pos:
            track_query_pos = track_query_pos.detach()

        ins_query = ins_query[-1]
        ins_query = self.mode_fuser(ins_query).max(2)[0]  # 取最大模态
        ins_query = self.multi_query_fuser(torch.cat([ins_query, track_query, track_query_pos], dim=-1))

        return ins_query

    def forward_train(self, bev_feat, outs_dict, gt_inds_list=None,
                      gt_segmentation=None, gt_instance=None, gt_img_is_valid=None):
        """训练前向传播

        流程:
        1. 获取占据标签
        2. 融合查询
        3. 前向传播
        4. 计算实例分割损失
        """
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(
            gt_segmentation, gt_instance, gt_img_is_valid)

        all_matched_gt_ids = outs_dict['all_matched_idxes']
        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)

        mask_preds_batch, ins_seg_preds_batch = self(bev_feat, ins_query=ins_query)
        ins_seg_targets_batch = gt_instance

        # 帧有效性掩码: 只对有效帧计算损失
        img_is_valid = gt_img_is_valid
        frame_valid_mask = img_is_valid.bool()
        past_valid_mask = frame_valid_mask[:, :self.receptive_field]
        future_frame_mask = frame_valid_mask[:, (self.receptive_field-1):]
        past_valid = past_valid_mask.all(dim=1)
        future_frame_mask[~past_valid] = False

        loss_dict = dict()
        loss_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_mask = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_aux_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_aux_mask = ins_seg_preds_batch.new_zeros(1)[0].float()

        bs = ins_query.size(0)
        assert bs == 1
        for ind in range(bs):
            cur_gt_inds = gt_inds_list[ind][-1]
            cur_matched_gt = all_matched_gt_ids[ind]

            # 根据匹配索引重排 GT
            cur_gt_inds = cur_gt_inds[cur_matched_gt]
            cur_gt_inds[cur_matched_gt == -1] = -1  # 未匹配
            cur_gt_inds[cur_matched_gt == -2] = -2  # 无查询

            frame_mask = future_frame_mask[ind]
            ins_seg_preds = ins_seg_preds_batch[ind]
            ins_seg_targets = ins_seg_targets_batch[ind]
            mask_preds = mask_preds_batch[ind]

            # 构建有序的 GT 掩码
            ins_seg_targets_ordered = []
            for ins_id in cur_gt_inds:
                if (ins_seg_targets == self.ignore_index).all().item() is True:
                    ins_tgt = ins_seg_targets.long()
                elif ins_id.item() in [-1, -2]:
                    ins_tgt = torch.ones_like(ins_seg_targets).long() * self.ignore_index
                else:
                    SPECIAL_INDEX = -20
                    if ins_id.item() == self.ignore_index:
                        ins_id = torch.ones_like(ins_id) * SPECIAL_INDEX
                    ins_tgt = (ins_seg_targets == ins_id).long()
                ins_seg_targets_ordered.append(ins_tgt)

            ins_seg_targets_ordered = torch.stack(ins_seg_targets_ordered, dim=0)
            num_total_pos = ins_seg_preds.size(0)
            num_total_pos = ins_seg_preds.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

            cur_dice_loss = self.loss_dice(ins_seg_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask)
            cur_mask_loss = self.loss_mask(ins_seg_preds, ins_seg_targets_ordered, frame_mask=frame_mask)
            cur_aux_dice_loss = self.loss_dice(mask_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask)
            cur_aux_mask_loss = self.loss_mask(mask_preds, ins_seg_targets_ordered, frame_mask=frame_mask)

            loss_dice += cur_dice_loss
            loss_mask += cur_mask_loss
            loss_aux_dice += cur_aux_dice_loss * self.aux_loss_weight
            loss_aux_mask += cur_aux_mask_loss * self.aux_loss_weight

        loss_dict['loss_dice'] = loss_dice / bs
        loss_dict['loss_mask'] = loss_mask / bs
        loss_dict['loss_aux_dice'] = loss_aux_dice / bs
        loss_dict['loss_aux_mask'] = loss_aux_mask / bs

        return loss_dict

    def forward_test(self, bev_feat, outs_dict, no_query=False,
                     gt_segmentation=None, gt_instance=None, gt_img_is_valid=None):
        """推理前向传播"""
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(
            gt_segmentation, gt_instance, gt_img_is_valid)

        out_dict = dict()
        out_dict['seg_gt'] = gt_segmentation[:, :1+self.n_future]
        out_dict['ins_seg_gt'] = self.get_ins_seg_gt(gt_instance[:, :1+self.n_future])

        if no_query:
            out_dict['seg_out'] = torch.zeros_like(out_dict['seg_gt']).long()
            out_dict['ins_seg_out'] = torch.zeros_like(out_dict['ins_seg_gt']).long()
            return out_dict

        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)
        _, pred_ins_logits = self(bev_feat, ins_query=ins_query)

        out_dict['pred_ins_logits'] = pred_ins_logits
        pred_ins_logits = pred_ins_logits[:,:,:1+self.n_future]
        pred_ins_sigmoid = pred_ins_logits.sigmoid()

        if self.test_with_track_score:
            track_scores = outs_dict['track_scores'].to(pred_ins_sigmoid)
            track_scores = track_scores[:, :, None, None, None]
            pred_ins_sigmoid = pred_ins_sigmoid * track_scores

        out_dict['pred_ins_sigmoid'] = pred_ins_sigmoid
        pred_seg_scores = pred_ins_sigmoid.max(1)[0]
        seg_out = (pred_seg_scores > self.test_seg_thresh).long().unsqueeze(2)
        out_dict['seg_out'] = seg_out

        if self.pan_eval:
            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                seg_out, pred_ins_sigmoid)
            out_dict['ins_seg_out'] = pred_consistent_instance_seg

        return out_dict

    def get_ins_seg_gt(self, gt_instance):
        """将非连续的实例 ID 转为连续 ID"""
        ins_gt_old = gt_instance
        ins_gt_new = torch.zeros_like(ins_gt_old).to(ins_gt_old)
        ins_inds_unique = torch.unique(ins_gt_old)
        new_id = 1
        for uni_id in ins_inds_unique:
            if uni_id.item() in [0, self.ignore_index]:
                continue
            ins_gt_new[ins_gt_old == uni_id] = new_id
            new_id += 1
        return ins_gt_new

    def get_occ_labels(self, gt_segmentation, gt_instance, gt_img_is_valid):
        """获取占据标签"""
        if not self.training:
            gt_segmentation = gt_segmentation[0]
            gt_instance = gt_instance[0]
            gt_img_is_valid = gt_img_is_valid[0]
        gt_segmentation = gt_segmentation[:, :self.n_future+1].long().unsqueeze(2)
        gt_instance = gt_instance[:, :self.n_future+1].long()
        gt_img_is_valid = gt_img_is_valid[:, :self.receptive_field + self.n_future]
        return gt_segmentation, gt_instance, gt_img_is_valid