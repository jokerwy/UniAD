"""
UniAD 模块包 (Modules)
======================
包含 BEVFormer 核心的 Transformer 模块，实现多视角图像到 BEV 特征的转换，
以及在 BEV 特征上进行 3D 目标检测。

模块组成:
    - PerceptionTransformer:   核心 Transformer，包含 Encoder 和 Decoder
    - BEVFormerEncoder:        BEV 特征编码器 (多视角图像 → BEV 特征)
    - BEVFormerLayer:          编码器单层 (TSA + SCA + FFN)
    - DetectionTransformerDecoder: 检测解码器 (BEV 特征 → 检测结果)
    - SpatialCrossAttention:   空间交叉注意力 (BEV 查询 → 图像特征采样)
    - MSDeformableAttention3D: 3D 可变形注意力 (空间交叉注意力的底层实现)
    - TemporalSelfAttention:   时序自注意力 (当前 BEV 查询 → 历史 BEV 特征)
    - MyCustomBaseTransformerLayer: 自定义 Transformer 层基类
"""

from .transformer import PerceptionTransformer
from .spatial_cross_attention import SpatialCrossAttention, MSDeformableAttention3D
from .temporal_self_attention import TemporalSelfAttention
from .encoder import BEVFormerEncoder, BEVFormerLayer
from .decoder import DetectionTransformerDecoder

