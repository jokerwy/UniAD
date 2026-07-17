# UniAD 2.0 代码仓库结构分析文档

> **项目**: UniAD — Planning-oriented Autonomous Driving
> **版本**: v2.0
> **论文**: [arXiv:2212.10156](https://arxiv.org/abs/2212.10156)
> **官方仓库**: [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)
> **分析日期**: 2026-07-15

---

## 目录

1. [项目概述](#1-项目概述)
2. [总体架构](#2-总体架构)
3. [目录结构](#3-目录结构)
4. [核心模块详解](#4-核心模块详解)
   - [4.1 Detectors（检测器）](#41-detectors检测器)
   - [4.2 Dense Heads（任务头）](#42-dense-heads任务头)
   - [4.3 Modules（BEV Transformer 模块）](#43-modulesbev-transformer-模块)
   - [4.4 Datasets（数据集）](#44-datasets数据集)
   - [4.5 Losses（损失函数）](#45-losses损失函数)
   - [4.6 Core（核心工具）](#46-core核心工具)
5. [配置文件解析](#5-配置文件解析)
6. [训练管线](#6-训练管线)
7. [数据流](#7-数据流)
8. [工具脚本](#8-工具脚本)
9. [依赖关系](#9-依赖关系)

---

## 1. 项目概述

UniAD 是一个**以规划为导向的端到端自动驾驶算法框架**，将感知、预测和规划任务统一到一个层次化网络中。它是 CVPR 2023 **最佳论文候选**（12/2360）。

### 核心思想

传统自动驾驶系统将感知、预测、规划拆分为独立模块，导致信息传递损失。UniAD 的核心理念是：

- **以规划为最终目标**，所有上游任务（检测、跟踪、建图、运动预测、占据预测）都为规划服务
- **层次化级联**：下游任务查询（query）基于上游任务的输出进行交互，而非独立处理
- **纯视觉输入**：仅使用 6 个摄像头图像（不使用 LiDAR），通过 BEV（Bird's Eye View）特征转换实现

### 六大任务

| 任务 | 英文名 | 功能 |
|------|--------|------|
| 3D 目标检测 | Detection | 在 BEV 视角下检测车辆、行人等 10 类目标 |
| 多目标跟踪 | Tracking | 跨帧关联同一目标，维护轨迹 ID |
| 在线建图 | Mapping | 分割车道线、可行驶区域等道路结构 |
| 运动预测 | Motion Forecasting | 预测目标未来 12 步（6 秒）的轨迹 |
| 占据栅格预测 | Occupancy Prediction | 预测未来 4 步的占据栅格 |
| 轨迹规划 | Planning | 规划自车未来 6 步（3 秒）的行驶轨迹 |

---

## 2. 总体架构

### 类继承关系

```
MVXTwoStageDetector (mmdet3d)
  └── UniADTrack                 # 阶段一：检测 + 跟踪
        └── UniAD                # 阶段二：完整端到端（六大任务）
                                 # 继承 UniADTrack 的检测/跟踪能力
                                 # 增加 seg_head, motion_head, occ_head, planning_head
```

### 模型组成（以完整 UniAD 为例）

```
输入: 6 个摄像头图像 (3 帧时序)
  │
  ├── img_backbone (ResNet-101 + DCN)
  │     └── 提取多尺度图像特征
  │
  ├── img_neck (FPN)
  │     └── 融合多尺度特征 → 统一维度 256
  │
  ├── PerceptionTransformer (BEVFormer 编码器-解码器)
  │     ├── BEVFormerEncoder (6 层)
  │     │     ├── TemporalSelfAttention   # 时序融合：对齐历史 BEV 特征
  │     │     └── SpatialCrossAttention   # 空间投影：图像特征 → BEV 空间
  │     └── DetectionTransformerDecoder (6 层)
  │           ├── MultiheadAttention      # Object Query 自注意力
  │           └── CustomMSDeformableAttention  # Query 与 BEV 特征交互
  │
  ├── pts_bbox_head (BEVFormerTrackHead)
  │     ├── 检测分支：分类 + 3D 边界框回归
  │     ├── 跟踪分支：Query Interaction Module (QIM) + Memory Bank
  │     └── 输出：track_bboxes, track_scores, track_ids
  │
  ├── seg_head (PansegformerHead)          # 在线建图
  │     ├── DetrTransformerEncoder (6 层)
  │     ├── DeformableDetrTransformerDecoder (6 层)
  │     ├── thing_transformer_head (SegMaskHead)   # Things: 车道线
  │     └── stuff_transformer_head (SegMaskHead)   # Stuff: 可行驶区域
  │
  ├── motion_head (MotionHead)             # 运动预测
  │     ├── MotionTransformerDecoder (3 层)
  │     ├── MotionDeformableAttention
  │     └── 非线性优化精炼轨迹
  │
  ├── occ_head (OccHead)                   # 占据栅格预测
  │     ├── BevFeatureSlicer: 提取 BEV 特征
  │     ├── DetrTransformerDecoder (5 层)
  │     └── 预测未来 4 帧的占据/实例/流向
  │
  └── planning_head (PlanningHeadSingleMode)  # 轨迹规划
        ├── navi_embed: 导航指令嵌入
        ├── reg_branch: 回归规划轨迹
        └── CollisionNonlinearOptimizer: 碰撞优化
```

---

## 3. 目录结构

```
UniAD/
├── README.md                           # 项目说明
├── requirements.txt                    # Python 依赖列表
├── CODE_OF_CONDUCT.md                  # 行为准则
│
├── docs/                               # 文档
│   ├── INSTALL.md                      # 环境安装指南
│   ├── DATA_PREP.md                    # 数据集准备指南
│   └── TRAIN_EVAL.md                   # 训练/评估指南
│
├── ckpts/                              # 预训练权重目录
│   ├── r101_dcn_fcos3d_pretrain.pth    # BEVFormer 骨干网络预训练权重
│   ├── bevformer_r101_dcn_24ep.pth     # BEVFormer 在 nuScenes 上训练 24 epoch 的权重
│   ├── uniad_base_track_map.pth        # 第一阶段：检测+跟踪+建图
│   └── uniad_base_e2e.pth             # 第二阶段：端到端完整权重
│
├── data/                               # 数据集目录
│   ├── nuscenes/                       # nuScenes 原始数据
│   │   ├── can_bus/                    # CAN 总线数据
│   │   ├── maps/                       # 地图扩展 (v1.3)
│   │   ├── samples/                    # 图像样本
│   │   ├── sweeps/                     # 雷达扫描
│   │   ├── v1.0-trainval/              # 训练/验证标注
│   │   ├── v1.0-test/                  # 测试标注
│   │   └── v1.0-mini/                  # 迷你数据集
│   ├── infos/                          # 数据信息文件
│   │   ├── nuscenes_infos_temporal_train.pkl  # 训练集时序信息
│   │   └── nuscenes_infos_temporal_val.pkl    # 验证集时序信息
│   └── others/                         # 辅助文件
│       └── motion_anchor_infos_mode6.pkl      # 运动锚点信息
│
├── projects/                           # 核心代码（mmdet3d 插件形式）
│   ├── __init__.py
│   │
│   ├── configs/                        # 模型配置文件
│   │   ├── _base_/                     # 基础配置模板
│   │   │   ├── datasets/
│   │   │   │   └── nus-3d.py          # 基础数据集配置（被各阶段覆盖）
│   │   │   └── default_runtime.py     # 基础运行时配置
│   │   ├── bevformer/
│   │   │   └── base_bevformer.py      # BEVFormer 骨干网络配置
│   │   ├── stage1_track_map/
│   │   │   └── base_track_map.py      # 第一阶段：检测+跟踪+建图
│   │   └── stage2_e2e/
│   │       └── base_e2e.py            # 第二阶段：端到端全任务
│   │
│   └── mmdet3d_plugin/                # mmdet3d 插件代码
│       ├── __init__.py
│       │
│       ├── core/                      # 核心工具
│       │   ├── bbox/
│       │   │   ├── assigners/          # 目标匹配分配器
│       │   │   │   ├── hungarian_assigner_3d.py       # 匈牙利匹配（3D 检测）
│       │   │   │   └── hungarian_assigner_3d_track.py # 匈牙利匹配（跟踪）
│       │   │   ├── coders/            # 边界框编解码
│       │   │   │   ├── detr3d_track_coder.py  # 跟踪专用编码器
│       │   │   │   └── nms_free_coder.py     # 免 NMS 编码器
│       │   │   ├── match_costs/       # 匹配代价函数
│       │   │   │   └── match_cost.py
│       │   │   └── util.py            # 归一化工具
│       │   └── evaluation/
│       │       └── eval_hooks.py      # 评估钩子
│       │
│       ├── datasets/                  # 数据集
│       │   ├── __init__.py
│       │   ├── builder.py             # 数据集构建器
│       │   ├── nuscenes_bev_dataset.py # 基础 nuScenes 数据集
│       │   ├── nuscenes_e2e_dataset.py # 端到端 nuScenes 数据集（增加时序 + 未来标注）
│       │   ├── nuscenes_eval.py       # 评估入口
│       │   ├── pipelines/             # 数据处理管线
│       │   │   ├── formating.py       # 数据格式化
│       │   │   ├── loading.py         # 数据加载（多视角图像、3D 标注等）
│       │   │   ├── occflow_label.py   # 占据/流向标签生成
│       │   │   └── transform_3d.py    # 3D 数据增强
│       │   ├── samplers/              # 数据采样器
│       │   │   ├── distributed_sampler.py
│       │   │   ├── group_sampler.py
│       │   │   └── sampler.py
│       │   ├── data_utils/            # 数据处理工具
│       │   │   ├── data_utils.py
│       │   │   ├── rasterize.py       # 地图栅格化
│       │   │   ├── trajectory_api.py  # 轨迹 API
│       │   │   └── vector_map.py      # 矢量地图
│       │   └── eval_utils/            # 评估工具
│       │       ├── eval_utils.py
│       │       ├── map_api.py         # 地图评估 API
│       │       ├── metric_utils.py    # 度量计算
│       │       ├── nuscenes_eval.py   # nuScenes 检测/跟踪评估
│       │       └── nuscenes_eval_motion.py  # 运动预测评估
│       │
│       ├── losses/                    # 损失函数
│       │   ├── dice_loss.py           # Dice 损失（分割）
│       │   ├── track_loss.py          # 跟踪损失
│       │   ├── traj_loss.py           # 轨迹预测损失
│       │   ├── occflow_loss.py        # 占据/流向损失
│       │   ├── planning_loss.py       # 规划损失
│       │   └── mtp_loss.py            # 多轨迹预测损失
│       │
│       ├── models/                    # 模型组件
│       │   ├── backbones/
│       │   │   └── vovnet.py          # VoVNet 骨干网络（替代方案）
│       │   ├── hooks/
│       │   │   └── hooks.py           # 训练钩子
│       │   ├── opt/
│       │   │   └── adamw.py           # AdamW 优化器
│       │   └── utils/
│       │       ├── bricks.py          # 网络构建块
│       │       ├── functional.py      # 工具函数（高斯激活等）
│       │       ├── grid_mask.py       # GridMask 数据增强
│       │       └── visual.py          # 可视化工具
│       │
│       └── uniad/                     # UniAD 核心实现
│           ├── __init__.py
│           │
│           ├── apis/                  # 训练/测试 API
│           │   ├── train.py
│           │   ├── test.py
│           │   └── mmdet_train.py     # 训练逻辑
│           │
│           ├── detectors/             # 检测器（模型入口）
│           │   ├── bevformer.py       # BEVFormer 纯检测模型
│           │   ├── uniad_track.py     # UniAD 跟踪模型（阶段一）
│           │   └── uniad_e2e.py      # UniAD 端到端模型（阶段二）
│           │
│           ├── modules/               # Transformer 编码器/解码器
│           │   ├── encoder.py         # BEVFormerEncoder（时序 + 空间注意力）
│           │   ├── decoder.py         # CustomMSDeformableAttention
│           │   ├── transformer.py     # PerceptionTransformer（整体 Transformer）
│           │   ├── temporal_self_attention.py    # 时序自注意力
│           │   ├── spatial_cross_attention.py    # 空间交叉注意力
│           │   ├── multi_scale_deformable_attn_function.py  # CUDA 可变形注意力
│           │   └── custom_base_transformer_layer.py  # 自定义 Transformer 层
│           │
│           ├── dense_heads/           # 任务头
│           │   ├── bevformer_head.py          # BEVFormer 检测头
│           │   ├── track_head.py              # 跟踪头（BEVFormerTrackHead）
│           │   ├── panseg_head.py             # 全景分割头（在线建图）
│           │   ├── motion_head.py             # 运动预测头
│           │   ├── occ_head.py                # 占据栅格预测头
│           │   ├── planning_head.py           # 规划头
│           │   │
│           │   ├── track_head_plugin/         # 跟踪插件
│           │   │   ├── tracker.py             # 运行时跟踪器
│           │   │   ├── track_instance.py      # 跟踪实例
│           │   │   └── modules.py             # MemoryBank, QIM
│           │   │
│           │   ├── seg_head_plugin/           # 分割插件
│           │   │   ├── seg_deformable_transformer.py  # 可变形 Transformer
│           │   │   ├── seg_detr_head.py       # DETR 分割头
│           │   │   ├── seg_mask_head.py       # Mask 预测头
│           │   │   ├── seg_assigner.py        # 分割分配器
│           │   │   └── seg_utils.py           # 分割工具
│           │   │
│           │   ├── motion_head_plugin/        # 运动预测插件
│           │   │   ├── base_motion_head.py
│           │   │   ├── motion_deformable_attn.py  # 运动可变形注意力
│           │   │   ├── motion_optimization.py     # 非线性轨迹优化
│           │   │   ├── motion_utils.py
│           │   │   └── modules.py
│           │   │
│           │   ├── occ_head_plugin/           # 占据预测插件
│           │   │   ├── modules.py             # 网络模块
│           │   │   ├── metrics.py             # 评估指标
│           │   │   └── utils.py
│           │   │
│           │   └── planning_head_plugin/      # 规划插件
│           │       ├── collision_optimization.py  # 碰撞优化
│           │       └── planning_metrics.py        # 规划指标
│           │
│           └── hooks/                 # 训练钩子
│               └── custom_hooks.py
│
├── tools/                             # 训练/评估/可视化脚本
│   ├── train.py                       # 训练入口
│   ├── test.py                        # 测试入口
│   ├── create_data.py                 # 数据预处理
│   ├── uniad_dist_train.sh            # 分布式训练启动脚本
│   ├── uniad_dist_eval.sh             # 分布式评估启动脚本
│   ├── uniad_slurm_train.sh           # Slurm 训练启动脚本
│   ├── uniad_slurm_eval.sh            # Slurm 评估启动脚本
│   ├── uniad_create_data.sh           # 数据创建启动脚本
│   ├── uniad_vis_result.sh            # 可视化启动脚本
│   ├── data_converter/
│   │   └── uniad_nuscenes_converter.py  # nuScenes 数据转换器
│   └── analysis_tools/
│       ├── analyze_logs.py            # 日志分析
│       ├── benchmark.py               # 性能测试
│       └── visualize/                 # 可视化工具
│           ├── run.py                 # 可视化入口
│           ├── bev_visual.py          # BEV 可视化
│           ├── utils.py
│           └── render/
│               ├── base_render.py
│               ├── bev_render.py      # BEV 渲染
│               └── cam_render.py      # 相机视角渲染
│
├── docker/                            # Docker 配置
├── sources/                           # 附件资源（海报等）
└── .github/                           # GitHub 配置
```

---

## 4. 核心模块详解

### 4.1 Detectors（检测器）

#### 4.1.1 `UniADTrack` — 第一阶段模型

**文件**: `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py`

**继承自**: `MVXTwoStageDetector` (mmdet3d)

**核心功能**:
- 图像特征提取（ResNet-101 + DCN + FPN）
- BEV 特征转换（BEVFormer Encoder）
- 3D 目标检测（BEVFormer Decoder）
- 多目标跟踪（QIM + Memory Bank）

**关键组件**:
- `QueryInteractionModule (QIM)`：在检测 query 之间进行信息交互，增强跟踪连续性
- `MemoryBank`：存储历史帧的跟踪信息（位置、外观特征、ID），实现跨帧关联
- `RuntimeTrackerBase`：在线跟踪器，维护轨迹生命周期（创建/更新/删除）

**数据流**:
```
img (B, T, N, C, H, W) → img_backbone → img_neck → BEV Encoder
    → BEV features → Decoder → track_queries
    → QIM + Memory Bank → 检测结果 + 跟踪 ID
```

#### 4.1.2 `UniAD` — 第二阶段完整模型

**文件**: `projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py`

**继承自**: `UniADTrack`

**新增功能**（在 UniADTrack 基础上冻结 BEV 编码器，增加）：

| 任务头 | 类名 | 功能 |
|--------|------|------|
| `seg_head` | `PansegformerHead` | 在线建图（车道线 + 可行驶区域） |
| `motion_head` | `MotionHead` | 多模态轨迹预测 |
| `occ_head` | `OccHead` | 占据栅格预测 |
| `planning_head` | `PlanningHeadSingleMode` | 自车轨迹规划 |

**任务损失权重配置**:
```python
task_loss_weight = dict(
    track=1.0,      # 检测 + 跟踪
    map=1.0,        # 在线建图
    motion=1.0,     # 运动预测
    occ=1.0,        # 占据预测
    planning=1.0,   # 轨迹规划
)
```

**训练模式**: 因冻结 BEV 编码器，第二阶段仅需 ~17GB GPU 显存

#### 4.1.3 `BEVFormer` — 纯检测模型

**文件**: `projects/mmdet3d_plugin/uniad/detectors/bevformer.py`

BEVFormer 的 UniAD 仓库实现，仅包含 3D 检测功能，用作预训练骨干网络。

---

### 4.2 Dense Heads（任务头）

#### 4.2.1 `BEVFormerTrackHead` — 检测 + 跟踪头

**文件**: `projects/mmdet3d_plugin/uniad/dense_heads/track_head.py`

**继承自**: `DETRHead` (mmdet)

**核心功能**:
- 基于 BEV 特征的 3D 目标检测
- 输出目标分类、3D 边界框、速度
- 提供跟踪所需的特征表示

**关键设计**:
- `with_box_refine=True`：在 6 层解码器中逐步精炼边界框
- 使用 `NMSFreeCoder`：免 NMS 后处理，DETR 风格
- 支持 `past_steps` 和 `fut_steps`：为时序交互提供帧索引

#### 4.2.2 `PansegformerHead` — 在线建图头

**文件**: `projects/mmdet3d_plugin/uniad/dense_heads/panseg_head.py`

**核心功能**:
- 全景分割：同时预测 Things（车道线，3 类）和 Stuff（可行驶区域，1 类）
- 基于 BEV 特征，使用 Deformable DETR 架构

**网络结构**:
- `SegDeformableTransformer`：6 层编码器 + 6 层解码器
- `thing_transformer_head`：4 层 Mask Head，预测车道线实例
- `stuff_transformer_head`：6 层 Mask Head（带自注意力），预测可行驶区域
- 损失：Focal Loss（分类）+ L1 Loss（边界框）+ GIoU Loss（IoU）+ Dice Loss（Mask）

**输出**:
- 车道线边界框 + 掩码
- 可行驶区域边界框 + 掩码

#### 4.2.3 `MotionHead` — 运动预测头

**文件**: `projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py`

**继承自**: `BaseMotionHead`

**核心功能**:
- 多模态轨迹预测：为每个检测到的目标预测 6 种可能的未来轨迹
- 预测未来 12 步（6 秒，每步 0.5 秒）

**网络结构**:
- `MotionTransformerDecoder`（3 层）：基于检测 query 和 track query 的交互
- `MotionDeformableAttention`：时序可变形注意力，捕获目标运动模式
- 非线性优化器：使用 `casadi` 库进行轨迹平滑

**锚点机制**:
- 使用 6 个预定义运动锚点（`motion_anchor_infos_mode6.pkl`）
- 覆盖不同的运动模式（直行、左转、右转、加速、减速等）

**损失函数**:
- 分类损失：选择最佳预测模式
- 负对数似然（NLL）损失：回归轨迹分布
- minADE/minFDE：评估指标，不参与训练

#### 4.2.4 `OccHead` — 占据栅格预测头

**文件**: `projects/mmdet3d_plugin/uniad/dense_heads/occ_head.py`

**继承自**: `BaseModule`

**核心功能**:
- 预测未来 4 帧（2 秒）的占据栅格
- 同时预测实例分割和流向（flow）

**网络结构**:
- `BevFeatureSlicer`：从 BEV 特征中提取当前帧和未来帧的信息
- `DetrTransformerDecoder`（5 层）：时序 Transformer 解码器
- `CVT_Decoder`：将 Transformer 输出上采样为高分辨率占据栅格

**输出**（每帧未来）:
- 语义分割：车辆/背景
- 实例 ID：区分不同车辆
- 中心度（centerness）：实例中心置信度
- 偏移（offset）：像素到实例中心的偏移
- 前向/后向流动（flow）：像素运动向量

**损失**:
- `FieryBinarySegmentationLoss`：Top-K 二元分割损失
- `DiceLossWithMasks`：Dice 损失

**栅格配置**:
```python
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],   # 200 像素
    'ybound': [-50.0, 50.0, 0.5],   # 200 像素
    'zbound': [-10.0, 10.0, 20.0],  # 1 个高度层
}
```

#### 4.2.5 `PlanningHeadSingleMode` — 规划头

**文件**: `projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py`

**核心功能**:
- 输出自车未来 6 步（3 秒）的规划轨迹
- 基于导航指令（左转/直行/右转）和场景上下文

**网络结构**:
- `navi_embed`：导航指令嵌入（3 类：左转、直行、右转）
- `reg_branch`：2 层 MLP，回归 12 个值（6 步 × 2 坐标）
- `CollisionNonlinearOptimizer`：基于占据预测的碰撞优化

**损失**:
- `PlanningLoss`：L2 距离损失
- `CollisionLoss`：三段碰撞损失（delta=0.0, 0.5, 1.0），权重分别为 2.5, 1.0, 0.25

---

### 4.3 Modules（BEV Transformer 模块）

#### 4.3.1 `PerceptionTransformer` — 整体 Transformer

**文件**: `projects/mmdet3d_plugin/uniad/modules/transformer.py`

**核心功能**: 组装 BEVFormer 的编码器和解码器

**关键组件**:
- `encoder`: `BEVFormerEncoder` — 6 层 Transformer 编码器
- `decoder`: `DetectionTransformerDecoder` — 6 层 Transformer 解码器
- CAN 总线信息嵌入：将车辆运动信息（速度、加速度、转角等）编码为特征
- 相机姿态嵌入：编码 6 个相机的外参信息

#### 4.3.2 `BEVFormerEncoder` — BEV 编码器

**文件**: `projects/mmdet3d_plugin/uniad/modules/encoder.py`

**每层包含两个注意力模块**:
1. **`TemporalSelfAttention`**（时序自注意力）
   - 将当前帧的 BEV query 与历史 BEV 特征对齐
   - 根据车辆运动补偿旋转和平移
   - 实现时序一致性

2. **`SpatialCrossAttention`**（空间交叉注意力）
   - 基于 `MSDeformableAttention3D`（CUDA 加速的可变形注意力）
   - 将 3D 参考点投影到多相机图像特征上
   - 直接从图像特征中提取 BEV 空间信息

**参考点生成**:
- 在 BEV 空间（[-51.2, 51.2]m × [-51.2, 51.2]m）中均匀采样
- 每个 pillar 采样 4 个高度点
- 投影到 6 个相机的图像平面

#### 4.3.3 注意力模块

| 模块 | 文件 | 功能 |
|------|------|------|
| `TemporalSelfAttention` | `temporal_self_attention.py` | 时序 BEV 特征对齐 |
| `SpatialCrossAttention` | `spatial_cross_attention.py` | 空间交叉注意力（图像→BEV） |
| `MSDeformableAttention3D` | `spatial_cross_attention.py` | 3D 可变形注意力 |
| `CustomMSDeformableAttention` | `decoder.py` | 自定义可变形注意力（解码器使用） |
| `multi_scale_deformable_attn_function` | `multi_scale_deformable_attn_function.py` | CUDA 加速的可变形注意力实现 |

---

### 4.4 Datasets（数据集）

#### 4.4.1 `NuScenesE2EDataset` — 端到端数据集

**文件**: `projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py`

**继承自**: `NuScenesBEVDataset` → `CustomNuScenesDataset`

**核心特性**:
- 支持时序输入：`queue_length` 帧连续图像
- 提供未来标注：用于运动预测和占据预测
- 提供规划标注：自车未来轨迹
- 提供地图标注：车道线、可行驶区域

**输入数据项**:
| 数据项 | 说明 |
|--------|------|
| `img` | 多视角时序图像 (T, N, C, H, W) |
| `timestamp` | 时间戳 |
| `gt_bboxes_3d` | 3D 边界框标注 |
| `gt_labels_3d` | 类别标签 |
| `gt_inds` | 实例 ID（跟踪用） |
| `gt_fut_traj` | 目标未来轨迹 |
| `gt_past_traj` | 目标历史轨迹 |
| `gt_sdc_bbox` | 自车边界框 |
| `gt_sdc_fut_traj` | 自车未来轨迹 |
| `gt_lane_labels/bboxes/masks` | 车道线标注 |
| `gt_segmentation/instance/flow` | 占据栅格标注 |
| `sdc_planning` | 自车规划轨迹 |
| `command` | 导航指令 |

#### 4.4.2 数据处理管线（Pipeline）

**训练管线**:
```
LoadMultiViewImageFromFilesInCeph  # 加载 6 个相机图像
  → PhotoMetricDistortionMultiViewImage  # 光度增强
  → LoadAnnotations3D_E2E  # 加载完整标注
  → GenerateOccFlowLabels  # 生成占据/流向标签
  → ObjectRangeFilterTrack  # 过滤范围外目标
  → ObjectNameFilterTrack  # 过滤类别
  → NormalizeMultiviewImage  # 图像归一化
  → PadMultiViewImage  # 填充到 32 的倍数
  → DefaultFormatBundle3D  # 格式化
  → CustomCollect3D  # 收集指定字段
```

**测试管线**: 类似训练，但去除数据增强，增加 `MultiScaleFlipAug3D`

---

### 4.5 Losses（损失函数）

| 文件 | 功能 | 用途 |
|------|------|------|
| `track_loss.py` | `ClipMatcher`：跟踪匹配损失（Focal Loss + L1） | 检测 + 跟踪 |
| `dice_loss.py` | Dice 损失 | 地图分割 Mask |
| `traj_loss.py` | `TrajLoss`：分类损失 + NLL 损失 + minADE/minFDE | 运动预测 |
| `occflow_loss.py` | `FieryBinarySegmentationLoss`：Top-K 二元分割损失 | 占据预测 |
| `planning_loss.py` | `PlanningLoss`：L2 距离 + 碰撞优化 | 轨迹规划 |
| `mtp_loss.py` | 多轨迹预测损失 | 运动预测（辅助） |

---

### 4.6 Core（核心工具）

#### 4.6.1 目标匹配（Assigners）

| 模块 | 功能 |
|------|------|
| `HungarianAssigner3D` | 匈牙利算法：检测 query 与 GT 3D 框匹配 |
| `HungarianAssigner3DTrack` | 跟踪专用：检测 query 与 GT 框 + 轨迹 ID 匹配 |

#### 4.6.2 边界框编码（Coders）

| 模块 | 功能 |
|------|------|
| `NMSFreeCoder` | DETR 风格解码器，免 NMS 后处理 |
| `DETRTrack3DCoder` | 跟踪专用编码器，支持分数阈值过滤 |

---

## 5. 配置文件解析

### 5.1 配置继承关系

```
_be_base_/datasets/nus-3d.py  ← 基础数据集配置（被完全覆盖）
_be_base_/default_runtime.py  ← 基础运行时配置

bevformer/base_bevformer.py   ← BEVFormer 骨干网络（纯检测，24 epoch）

stage1_track_map/base_track_map.py   ← 第一阶段：检测+跟踪+建图（6 epoch）
    ├── 继承自：_base_ 配置
    ├── load_from: ckpts/bevformer_r101_dcn_24ep.pth
    └── 包含 seg_head, freeze_img_neck=False, freeze_bn=False

stage2_e2e/base_e2e.py             ← 第二阶段：完整端到端（20 epoch）
    ├── 继承自：_base_ 配置
    ├── load_from: ckpts/uniad_base_track_map.pth
    ├── freeze_img_backbone=True, freeze_img_neck=True, freeze_bn=True, freeze_bev_encoder=True
    └── 包含全部 6 个任务头
```

### 5.2 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `point_cloud_range` | `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]` | 检测范围 (m) |
| `voxel_size` | `[0.2, 0.2, 8]` | 体素大小 |
| `bev_h_/bev_w_` | `200×200` | BEV 特征图分辨率 |
| `_dim_` | `256` | 特征维度 |
| `queue_length` | `3` (stage2) / `5` (stage1) | 时序帧数 |
| `predict_steps` | `12` | 运动预测步数 (6s) |
| `planning_steps` | `6` | 规划步数 (3s) |
| `occ_n_future` | `4` | 占据预测未来帧数 (2s) |
| `num_query` | `900` (检测) / `300` (分割/运动) | Query 数量 |
| `optimizer` | AdamW, lr=2e-4, weight_decay=0.01 | 优化器配置 |
| `lr_config` | CosineAnnealing, warmup=500 | 学习率调度 |
| `total_epochs` | `6` (stage1) / `20` (stage2) | 训练轮数 |

---

## 6. 训练管线

### 6.1 三阶段训练流程

```
阶段 0: BEVFormer 预训练（可选）
├── 配置: projects/configs/bevformer/base_bevformer.py
├── 加载: ckpts/r101_dcn_fcos3d_pretrain.pth
├── 训练: 24 epoch
├── 输出: ckpts/bevformer_r101_dcn_24ep.pth
└── 功能: 学习 BEV 特征表示和 3D 检测

阶段 1: 检测 + 跟踪 + 建图
├── 配置: projects/configs/stage1_track_map/base_track_map.py
├── 加载: ckpts/bevformer_r101_dcn_24ep.pth
├── 训练: 6 epoch, ~50GB GPU 显存
├── 输出: ckpts/uniad_base_track_map.pth
└── 功能: 学习时序跟踪和在线建图

阶段 2: 端到端全任务
├── 配置: projects/configs/stage2_e2e/base_e2e.py
├── 加载: ckpts/uniad_base_track_map.pth
├── 训练: 20 epoch, ~17GB GPU 显存
├── 输出: ckpts/uniad_base_e2e.pth
└── 功能: 冻结 BEV 编码器，学习运动预测 + 占据预测 + 规划
```

### 6.2 评估指标

| 任务 | 指标 | 说明 |
|------|------|------|
| 检测 | NDS, mAP | nuScenes 标准检测指标 |
| 跟踪 | AMOTA, AMOTP, RECALL | nuScenes 跟踪指标 |
| 建图 | IoU | 车道线和可行驶区域 IoU |
| 运动预测 | minADE, minFDE, MR | 最小平均位移/终点误差，缺失率 |
| 占据预测 | IoU, VPQ | 占据 IoU，视频全景质量 |
| 规划 | avg.L2, avg.Col | 平均 L2 位移误差，碰撞率 |

---

## 7. 数据流

### 7.1 训练时前向数据流

```
输入: 6 摄像头 × 3 帧时序图像
  │
  ├─ img_backbone (ResNet-101 + DCN, frozen_stages=4)
  │   └─ 多尺度特征: [1/8, 1/16, 1/32]
  │
  ├─ img_neck (FPN)
  │   └─ 统一维度: 256, 4 层
  │
  ├─ BEVFormerEncoder (6 层)
  │   ├─ TemporalSelfAttention: 融合历史 BEV
  │   └─ SpatialCrossAttention: 图像 → BEV 投影
  │   └─ 输出: bev_embed (B, 200×200, 256)
  │
  ├─ Detection Decoder (6 层)
  │   ├─ Self-Attn: 检测 query 间交互
  │   └─ Cross-Attn: query ↔ bev_embed
  │   └─ 输出: track_query (B, 900, 256)
  │
  ├─ QIM + MemoryBank
  │   └─ 输出: 检测结果 + 跟踪 ID
  │
  ├─ seg_head (PansegformerHead)
  │   ├─ 输入: bev_embed + track_query 交互
  │   └─ 输出: 车道线框/Mask + 可行驶区域框/Mask
  │
  ├─ motion_head (MotionHead)
  │   ├─ 输入: track_query + 检测框
  │   └─ 输出: 6 模态 × 12 步轨迹
  │
  ├─ occ_head (OccHead)
  │   ├─ 输入: bev_embed 时序切片
  │   └─ 输出: 4 帧占据栅格 + 实例 + 流向
  │
  └─ planning_head (PlanningHeadSingleMode)
      ├─ 输入: bev_embed + navi_embed
      └─ 输出: 6 步规划轨迹
```

### 7.2 推理时数据流

与训练类似，区别在于：
- 使用 Memory Bank 维护历史轨迹 ID
- 运动预测使用非线性优化器精炼
- 规划使用碰撞优化器调整轨迹
- 输出所有 6 个任务的结果

---

## 8. 工具脚本

| 脚本 | 功能 | 用法 |
|------|------|------|
| `tools/uniad_dist_train.sh` | 分布式训练 | `./tools/uniad_dist_train.sh config.py N_GPUS` |
| `tools/uniad_dist_eval.sh` | 分布式评估 | `./tools/uniad_dist_eval.sh config.py ckpt.pth N_GPUS` |
| `tools/uniad_slurm_train.sh` | Slurm 训练 | `./tools/uniad_slurm_train.sh PARTITION config.py N_GPUS` |
| `tools/uniad_slurm_eval.sh` | Slurm 评估 | `./tools/uniad_slurm_eval.sh PARTITION config.py ckpt.pth N_GPUS` |
| `tools/uniad_create_data.sh` | 生成数据信息文件 | `./tools/uniad_create_data.sh` |
| `tools/uniad_vis_result.sh` | 结果可视化 | `./tools/uniad_vis_result.sh` |
| `tools/analysis_tools/visualize/run.py` | 可视化引擎 | 生成 BEV/相机视角视频 |
| `tools/analysis_tools/analyze_logs.py` | 日志分析 | 绘制训练曲线 |
| `tools/analysis_tools/benchmark.py` | 推理速度测试 | 测试 FPS |

---

## 9. 依赖关系

### 9.1 核心依赖

| 依赖 | 版本 | 作用 |
|------|------|------|
| PyTorch | 2.0.1+cu118 | 深度学习框架 |
| torchvision | 0.15.2 | 图像处理 |
| mmcv-full | 1.7.0 | OpenMMLab 基础库（CUDA 算子、训练框架） |
| mmdet | 2.26.0 | 目标检测框架 |
| mmsegmentation | 0.29.1 | 语义分割框架 |
| mmdet3d | 1.0.0rc6 | 3D 目标检测框架 |

### 9.2 其他依赖（requirements.txt）

| 依赖 | 版本 | 用途 |
|------|------|------|
| numpy | 1.22.4 | 数值计算 |
| opencv-python | 4.8.0.76 | 图像处理 |
| einops | 0.8.1 | 张量操作 |
| casadi | 3.6.7 | 非线性优化（轨迹平滑） |
| pytorch-lightning | 1.2.5 | 训练框架 |
| motmetrics | ≤1.1.3 | 跟踪评估指标 |
| torchmetrics | 0.6.2 | 评估指标 |
| pandas | 1.2.2 | 数据处理 |
| networkx | 2.5 | 图算法 |
| google-cloud-bigquery | latest | 云端数据存储 |

### 9.3 模块依赖图

```
UniAD (uniad_e2e.py)
  ├── UniADTrack (uniad_track.py)
  │     ├── MVXTwoStageDetector (mmdet3d)
  │     │     ├── mmdet (DETECTORS, build_loss, ...)
  │     │     └── mmcv (runner, cnn, ops, ...)
  │     ├── BEVFormerTrackHead (track_head.py)
  │     │     └── DETRHead (mmdet)
  │     ├── PerceptionTransformer (transformer.py)
  │     │     ├── BEVFormerEncoder (encoder.py)
  │     │     │     ├── TemporalSelfAttention
  │     │     │     └── SpatialCrossAttention
  │     │     └── DetectionTransformerDecoder (mmdet)
  │     ├── MemoryBank + QIM (track_head_plugin/modules.py)
  │     └── RuntimeTrackerBase (track_head_plugin/tracker.py)
  │
  ├── PansegformerHead (panseg_head.py)
  │     ├── SegDeformableTransformer (seg_head_plugin/)
  │     └── SegMaskHead (seg_head_plugin/)
  │
  ├── MotionHead (motion_head.py)
  │     ├── BaseMotionHead (motion_head_plugin/)
  │     ├── MotionTransformerDecoder (motion_head_plugin/)
  │     └── nonlinear_smoother (motion_head_plugin/)
  │
  ├── OccHead (occ_head.py)
  │     ├── BevFeatureSlicer (occ_head_plugin/)
  │     ├── DetrTransformerDecoder (mmdet)
  │     └── CVT_Decoder (occ_head_plugin/)
  │
  └── PlanningHeadSingleMode (planning_head.py)
        ├── CollisionNonlinearOptimizer (planning_head_plugin/)
        └── PlanningLoss (losses/planning_loss.py)
```

---

## 附录：关键设计总结

1. **层次化任务设计**：下游任务（运动预测、规划）的 query 基于上游任务（检测、跟踪）的输出进行交互，而非独立处理，使信息在各任务间有效流动

2. **分阶段训练**：先训练感知基础（检测+跟踪+建图），再冻结 BEV 编码器训练高级任务（预测+规划），大幅节省显存

3. **Query 交互机制**：通过 QIM（Query Interaction Module）让检测 query 之间传递信息，增强跟踪连续性；Memory Bank 存储历史帧信息，实现长时关联

4. **纯视觉 BEV 感知**：仅使用 6 个摄像头图像，通过 BEVFormer 的时序自注意力和空间交叉注意力将多视角图像特征转换为统一的 BEV 表示

5. **非线性优化**：运动预测使用 `casadi` 进行轨迹平滑，规划使用碰撞优化器在占据预测基础上调整规划轨迹