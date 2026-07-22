# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
自定义 Transformer 层基类 (MyCustomBaseTransformerLayer)
========================================================
这是一个灵活的 Transformer 层实现，支持自定义操作顺序和注意力模块配置。

与标准 Transformer 层的区别:
    - 支持任意组合的 self_attn, cross_attn, ffn, norm 操作
    - 支持 pre-norm (先归一化再做注意力) 和 post-norm (先注意力再归一化)
    - 操作的顺序和执行次数完全由 operation_order 配置决定

BEVFormer 中的使用:
    - BEVFormerLayer 继承自此类，operation_order 配置为:
      ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
      即: TSA → Norm → SCA → Norm → FFN → Norm

    - DetectionTransformerDecoder 的每层也使用类似的结构:
      ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
      即: 自注意力 → Norm → 交叉注意力 → Norm → FFN → Norm
"""

import copy
import warnings

import torch

from mmcv import ConfigDict
from mmcv.cnn import build_norm_layer
from mmcv.runner.base_module import BaseModule, ModuleList

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention


@TRANSFORMER_LAYER.register_module()
class MyCustomBaseTransformerLayer(BaseModule):
    """自定义 Transformer 层基类

    支持灵活的 operation_order 配置，可以自由组合:
    - self_attn: 自注意力 (query=key=value)
    - cross_attn: 交叉注意力 (query≠key=value)
    - ffn: 前馈网络
    - norm: 层归一化

    支持 pre-norm 和 post-norm 两种模式:
    - pre-norm (operation_order[0] == 'norm'):
        在注意力/FFN 之前先做归一化，残差连接不加归一化
    - post-norm (operation_order[0] != 'norm'):
        在注意力/FFN 之后做归一化，残差连接包含前面的归一化

    Args:
        attn_cfgs: 注意力模块配置列表
            可以是单个 dict (所有注意力共享配置) 或 list of dict
        ffn_cfgs: FFN 模块配置列表
            可以是单个 dict 或 list of dict
        operation_order: 操作执行顺序
            如 ('self_attn', 'norm', 'ffn', 'norm')
            或 ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        norm_cfg: 归一化层配置，默认 LayerNorm
        batch_first: 是否 batch 维度在第一维，默认 True
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ')
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(MyCustomBaseTransformerLayer, self).__init__(init_cfg)

        self.batch_first = batch_first

        # 验证 operation_order 包含所有必要的操作类型
        assert set(operation_order) & set(
            ['self_attn', 'norm', 'ffn', 'cross_attn']) == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        # 统计注意力模块数量
        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        # 按 operation_order 构建各模块
        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # 标记操作类型，某些注意力模块可能需要区分 self/cross
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        # 构建 FFN 模块
        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims

            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index]))

        # 构建 LayerNorm 层
        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Transformer 层前向传播

        按照 operation_order 依次执行各个操作。
        每个注意力/FFN 操作后都通过残差连接与 identity 相加。

        pre-norm 模式:
            x = x + attention(norm(x))
            x = x + ffn(norm(x))

        post-norm 模式:
            x = norm(x + attention(x))
            x = norm(x + ffn(x))

        Args:
            query: 查询向量 (num_queries, bs, embed_dims) 或 (bs, num_queries, embed_dims)
            key: 键向量 (用于交叉注意力)
            value: 值向量 (用于交叉注意力)
            query_pos: 查询位置编码
            key_pos: 键位置编码
            attn_masks: 注意力掩码列表
            query_key_padding_mask: query/key 的 padding 掩码
            key_padding_mask: key 的 padding 掩码

        Returns:
            query: 更新后的查询向量
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # 处理 attn_masks
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                f'attn_masks {len(attn_masks)} must be equal ' \
                f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        # 按 operation_order 依次执行
        for layer in self.operation_order:
            # ---- 自注意力 ----
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            # ---- 层归一化 ----
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # ---- 交叉注意力 ----
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            # ---- 前馈网络 ----
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
