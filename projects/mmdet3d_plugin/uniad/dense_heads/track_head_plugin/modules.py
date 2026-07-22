"""
跟踪头模块组件 (Track Head Modules)
===================================
包含 MemoryBank 和 QueryInteractionModule 两个核心组件。

MemoryBank (记忆库):
    存储每个跟踪目标的历史特征，用于时序一致性建模。
    通过 temporal attention 融合历史特征和当前特征。

QueryInteractionModule (Query 交互模块):
    将历史帧的活跃 query 与当前帧的空 query 融合，
    生成下一帧的初始 query。训练时还包含假阳性 (FP) 注入和随机丢弃策略。
"""

import torch
import torch.nn.functional as F
from torch import nn
from .track_instance import Instances


class MemoryBank(nn.Module):
    """记忆库

    为每个跟踪目标维护一个历史特征队列 (长度为 mem_bank_len)。
    通过时序多头注意力机制融合历史特征和当前特征。

    核心流程:
    1. update(): 将当前帧的目标特征存入记忆库 (FIFO 队列)
    2. temporal_attn(): 当前特征对历史特征做注意力，融合时序信息

    记忆库存储策略:
    - 训练时: 存储所有分数 > 0 的目标
    - 推理时: 每隔 save_period 帧存储一次，分数需 > save_thresh

    Args:
        args: 配置字典，包含:
            - memory_bank_score_thresh: 存入 memory 的分数阈值
            - memory_bank_len: 记忆库长度 (存储的历史帧数)
        dim_in: 输入特征维度
        hidden_dim: 隐藏层维度
        dim_out: 输出特征维度
    """

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        self.save_thresh = args['memory_bank_score_thresh']
        self.save_period = 3  # 推理时每隔 3 帧存储一次
        self.max_his_length = args['memory_bank_len']

        # 存储特征投影层
        self.save_proj = nn.Linear(dim_in, dim_in)

        # 时序注意力模块
        self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
        self.temporal_norm1 = nn.LayerNorm(dim_in)
        self.temporal_norm2 = nn.LayerNorm(dim_in)

    def update(self, track_instances):
        """更新记忆库

        将当前帧的目标特征存入 FIFO 队列:
        - 新特征追加到队尾
        - 最早的旧特征从队首移除
        - mem_padding_mask 标记有效位置 (False=有效, True=填充)

        Args:
            track_instances: 跟踪实例
        """
        embed = track_instances.output_embedding[:, None]  # (N, 1, 256)
        scores = track_instances.scores
        mem_padding_mask = track_instances.mem_padding_mask
        device = embed.device

        save_period = track_instances.save_period
        if self.training:
            # 训练时: 存储所有有分数的目标
            saved_idxes = scores > 0
        else:
            # 推理时: 周期存储 + 分数筛选
            saved_idxes = (save_period == 0) & (scores > self.save_thresh)
            save_period[save_period > 0] -= 1
            save_period[saved_idxes] = self.save_period

        saved_embed = embed[saved_idxes]
        if len(saved_embed) > 0:
            prev_embed = track_instances.mem_bank[saved_idxes]
            save_embed = self.save_proj(saved_embed)
            # 更新 padding mask: 新特征对应位置置为 False (有效)
            mem_padding_mask[saved_idxes] = torch.cat(
                [mem_padding_mask[saved_idxes, 1:],
                 torch.zeros((len(saved_embed), 1), dtype=torch.bool, device=device)], dim=1)
            track_instances.mem_bank = track_instances.mem_bank.clone()
            # FIFO 更新: 移除最旧的，追加最新的
            track_instances.mem_bank[saved_idxes] = torch.cat(
                [prev_embed[:, 1:], save_embed], dim=1)

    def _forward_temporal_attn(self, track_instances):
        """时序注意力

        当前帧的目标特征 (query) 对记忆库中的历史特征 (key/value) 做注意力。
        融合时序信息，输出增强后的目标特征。

        只有记忆库中有有效特征的目标才参与计算 (key_padding_mask[:, -1] == 0)。
        """
        if len(track_instances) == 0:
            return track_instances

        key_padding_mask = track_instances.mem_padding_mask

        # 筛选记忆库中有有效特征的目标
        valid_idxes = key_padding_mask[:, -1] == 0
        embed = track_instances.output_embedding[valid_idxes]  # (n, 256)

        if len(embed) > 0:
            prev_embed = track_instances.mem_bank[valid_idxes]
            key_padding_mask = key_padding_mask[valid_idxes]

            # 时序多头注意力: 当前特征关注历史特征
            embed2 = self.temporal_attn(
                embed[None],                      # query: (1, n, 256)
                prev_embed.transpose(0, 1),        # key: (mem_len, n, 256)
                prev_embed.transpose(0, 1),        # value: (mem_len, n, 256)
                key_padding_mask=key_padding_mask,
            )[0][0]

            # 残差连接 + FFN
            embed = self.temporal_norm1(embed + embed2)
            embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
            embed = self.temporal_norm2(embed + embed2)

            track_instances.output_embedding = track_instances.output_embedding.clone()
            track_instances.output_embedding[valid_idxes] = embed

        return track_instances

    def forward_temporal_attn(self, track_instances):
        return self._forward_temporal_attn(track_instances)

    def forward(self, track_instances: Instances, update_bank=True) -> Instances:
        """记忆库前向传播

        1. 时序注意力: 融合历史特征
        2. 更新记忆库: 存入当前特征

        Args:
            track_instances: 跟踪实例
            update_bank: 是否更新记忆库 (训练时 True, 某些推理场景可能 False)

        Returns:
            更新后的跟踪实例
        """
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances)
        return track_instances


class QueryInteractionBase(nn.Module):
    """Query 交互模块基类"""
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class QueryInteractionModule(QueryInteractionBase):
    """Query 交互模块 (QIM)

    负责将历史帧的活跃 query 与当前帧的空 query 融合，
    生成下一帧的初始 query。

    核心流程:
    1. 筛选活跃 track: 选择 obj_idxes >= 0 的目标
    2. 假阳性注入: 训练时随机选择一些非活跃 query 作为"假阳性"
       模拟推理时可能出现的误检
    3. 随机丢弃: 训练时随机丢弃一些活跃目标
       模拟推理时可能出现的漏检
    4. 更新 query embedding: 通过自注意力 + FFN 更新 query 特征
    5. 合并: 将空 query 和活跃 query 拼接，作为下一帧的 query

    Args:
        args: 配置字典，包含:
            - random_drop: 随机丢弃概率
            - fp_ratio: 假阳性注入比例
            - update_query_pos: 是否更新 query 位置编码
            - merger_dropout: dropout 概率
        dim_in: 输入维度
        hidden_dim: 隐藏维度
        dim_out: 输出维度
    """

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args["random_drop"]
        self.fp_ratio = args["fp_ratio"]
        self.update_query_pos = args["update_query_pos"]

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args["merger_dropout"]

        # 自注意力模块
        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        # 可选: 更新 query 位置编码
        if args["update_query_pos"]:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        # 更新 query 内容编码
        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        """更新跟踪目标的 query embedding

        通过自注意力 + FFN 更新 query 的位置编码和内容编码。

        处理流程:
        1. 自注意力: query 之间交互，融合上下文信息
        2. FFN: 特征变换
        3. 更新 query_pos (位置编码) 和 query_feat (内容编码)
        """
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query[:, :dim // 2]   # 前半部分: 位置编码
        query_feat = track_instances.query[:, dim // 2:]  # 后半部分: 内容编码
        q = k = query_pos + out_embed

        # 自注意力
        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 更新 query 位置编码
        if self.update_query_pos:
            query_pos2 = self.linear_pos2(
                self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query[:, :dim // 2] = query_pos

        # 更新 query 内容编码
        query_feat2 = self.linear_feat2(
            self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query[:, dim // 2:] = query_feat

        return track_instances

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        """随机丢弃跟踪目标 (训练时)

        模拟推理时可能出现的漏检，增强模型鲁棒性。
        """
        drop_probability = self.random_drop
        if drop_probability > 0 and len(track_instances) > 0:
            keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
            track_instances = track_instances[keep_idxes]
        return track_instances

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:
        """添加假阳性 (FP) 跟踪目标 (训练时)

        模拟推理时可能出现的误检，增强模型鲁棒性。
        从非活跃目标中选择分数最高的几个作为假阳性。

        Args:
            track_instances: 所有跟踪实例
            active_track_instances: 活跃跟踪实例

        Returns:
            合并了假阳性的跟踪实例
        """
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # 以 fp_ratio 概率决定是否添加假阳性
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]
        num_fp = len(selected_active_track_instances)

        if len(inactive_instances) > 0 and num_fp > 0:
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                # 选择分数最高的非活跃目标作为假阳性
                fp_indexes = torch.argsort(inactive_instances.scores)[-num_fp:]
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat(
                [active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        """筛选活跃的跟踪目标

        训练时:
        - 选择 obj_idxes >= 0 且 IoU > 0.5 的目标
        - 随机丢弃一些目标 (模拟漏检)
        - 添加假阳性 (模拟误检)

        推理时:
        - 选择 obj_idxes >= 0 的目标
        """
        track_instances: Instances = data["track_instances"]
        if self.training:
            # 训练时: 筛选活跃目标 (有 ID 且 IoU 足够高)
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # 随机丢弃
            active_track_instances = self._random_drop_tracks(active_track_instances)
            # 添加假阳性
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(
                    track_instances, active_track_instances)
        else:
            # 推理时: 简单筛选
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def forward(self, data) -> Instances:
        """Query 交互模块前向传播

        1. 筛选活跃的跟踪目标
        2. 更新目标的 query embedding (自注意力 + FFN)
        3. 将空 query 和活跃 query 拼接，作为下一帧的初始 query

        Args:
            data: 字典，包含:
                - track_instances: 当前帧的跟踪实例
                - init_track_instances: 空的初始跟踪实例

        Returns:
            merged_track_instances: 合并后的跟踪实例
        """
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data["init_track_instances"]
        # 合并: 空 query (探索新目标) + 活跃 query (延续已有目标)
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances