# 🚀 Vision-Language-Action (VLA) 模型学习路线

> 一份从零到精通的学习指南，涵盖基础理论、核心论文、实践工具与前沿方向
> 最后更新：2026年7月

---

## 📋 目录

1. [什么是 VLA？](#1-什么是-vla)
2. [第一阶段：基础知识（第 1-2 月）](#2-第一阶段基础知识)
3. [第二阶段：核心理论（第 3-4 月）](#3-第二阶段核心理论)
4. [第三阶段：经典论文精读（第 5-6 月）](#4-第三阶段经典论文精读)
5. [第四阶段：动手实践（第 7 月+）](#5-第四阶段动手实践)
6. [关键数据与工具生态](#6-关键数据与工具生态)
7. [前沿方向与最新趋势](#7-前沿方向与最新趋势)
8. [推荐资源汇总](#8-推荐资源汇总)

---

## 1. 什么是 VLA？

**Vision-Language-Action (VLA)** 模型是机器人领域的一类**基础模型（Foundation Model）**，它将视觉感知、语言理解和动作控制统一到一个模型中。VLA 模型的核心思想是：**将互联网规模预训练的视觉语言模型（VLM）的知识迁移到机器人控制中**，使机器人具备语义理解、场景泛化与多任务执行能力。

### 核心架构范式

```
┌──────────────────────────────────────────────────────┐
│                    VLA 模型架构                        │
│                                                      │
│  视觉输入（图像/视频） ──► 视觉编码器 ──┐              │
│                                        ├──► 多模态融合 ──► 动作解码器 ──► 机器人动作
│  语言指令（自然语言） ──► 语言模型 ────┘              │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### VLA 发展的三个阶段

| 阶段 | 代表工作 | 特点 |
|:---|:---|:---|
| **萌芽期（2023）** | RT-2, RT-1 | 将动作视为文本 token，利用 VLM 进行机器人控制 |
| **开源爆发期（2024）** | OpenVLA, Octo, π0 | 开源模型涌现，扩散策略+流匹配，多具身支持 |
| **效率与规模化（2025+）** | Efficient VLA, VLA+ | 模型压缩、边缘部署、触觉融合、实时在线学习 |

---

## 2. 第一阶段：基础知识

> ⏱ 预计时间：第 1-2 月 | 目标：打下坚实的深度学习与机器人学习基础

### 2.1 编程与框架

| 技能 | 重要程度 | 学习资源 |
|:---|:---|:---|
| **Python 高级编程** | ⭐⭐⭐⭐⭐ | [Python 官方教程](https://docs.python.org/3/tutorial/) |
| **PyTorch** | ⭐⭐⭐⭐⭐ | [PyTorch 官方教程](https://pytorch.org/tutorials/) |
| **ROS 2（机器人操作系统）** | ⭐⭐⭐⭐ | [ROS 2 官方文档](https://docs.ros.org/en/humble/) |
| **Linux 命令行** | ⭐⭐⭐ | [The Linux Command Line](https://linuxcommand.org/) |

### 2.2 深度学习基础

| 主题 | 核心内容 | 学习资源 |
|:---|:---|:---|
| **Transformer 架构** | Self-Attention, Multi-Head Attention, Positional Encoding | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| **视觉 Transformer (ViT)** | 图像分块、位置编码、CLS Token | [ViT 论文](https://arxiv.org/abs/2010.11929) |
| **多模态学习** | CLIP, 图文对齐, 对比学习 | [CLIP 论文](https://arxiv.org/abs/2103.00020) |
| **自监督学习** | DINOv2, MAE, 掩码自编码器 | [DINOv2 论文](https://arxiv.org/abs/2304.07193) |

### 2.3 机器人学习基础

| 主题 | 核心内容 | 学习资源 |
|:---|:---|:---|
| **模仿学习 (Imitation Learning)** | 行为克隆 (Behavior Cloning), DAgger | [CS224R: Imitation Learning](https://web.stanford.edu/class/cs224r/) |
| **强化学习 (RL)** | 策略梯度, PPO, SAC | [Spinning Up in Deep RL](https://spinningup.openai.com/) |
| **机器人运动学** | 正向/逆向运动学, 雅可比矩阵 | [Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/) |
| **示教数据采集** | 遥操作, 键盘控制, VR 示教 | [ALOHA 硬件项目](https://github.com/tonyzhaozh/act) |

### 2.4 推荐在线课程

| 课程 | 机构 | 链接 | 说明 |
|:---|:---|:---|:---|
| **CS224R: Deep Reinforcement Learning** | Stanford | [课程主页](https://web.stanford.edu/class/cs224r/) | 最适合机器人+DL 的交叉课程 |
| **CS231n: CNNs for Visual Recognition** | Stanford | [课程主页](http://cs231n.stanford.edu/) | 视觉基础 |
| **CS224N: NLP with Deep Learning** | Stanford | [课程主页](https://web.stanford.edu/class/cs224n/) | Transformer 与语言模型 |
| **6.S191: Introduction to Deep Learning** | MIT | [课程主页](http://introtodeeplearning.com/) | 深度学习入门 Bootcamp |
| **Deep Reinforcement Learning** | UC Berkeley | [课程主页](https://rail.eecs.berkeley.edu/deeprlcourse/) | Sergey Levine 主讲，机器人学习权威 |

---

## 3. 第二阶段：核心理论

> ⏱ 预计时间：第 3-4 月 | 目标：理解 VLA 的核心概念与架构设计

### 3.1 动作 Tokenization（Action Tokenization）

这是 VLA 最核心的创新之一 —— **如何将连续的机器人动作转化为离散的 token**。

| 方法 | 描述 | 代表工作 |
|:---|:---|:---|
| **语言描述式** | 将动作描述为自然语言（如"向前移动 10cm"） | SayCan |
| **代码式** | 将动作表示为 Python 代码调用 | Code as Policies |
| **离散化 bin** | 将连续动作空间均匀离散化为 token | RT-2, OpenVLA |
| **轨迹式** | 将动作序列整体编码为 token | ACT |
| **潜在表示** | 使用 VAE 将动作压缩到潜在空间 | Diffusion Policy |
| **原始动作** | 直接输出连续动作值 | π0 (流匹配) |

📖 **推荐阅读**: [A Survey on VLA Models: An Action Tokenization Perspective](https://arxiv.org/abs/2507.01925) (arXiv:2507.01925, 2025 年 7 月)

### 3.2 多模态融合架构

| 方法 | 描述 | 代表模型 |
|:---|:---|:---|
| **Cross-Attention 融合** | 视觉特征通过 Cross-Attention 注入语言模型 | RT-2 (PaLM-E) |
| **特征拼接** | 视觉 token + 文本 token 直接拼接输入 LLM | OpenVLA |
| **双塔结构** | 视觉和语言分别编码后融合 | Octo |
| **早期融合** | 在输入层进行多模态融合 | π0 |

### 3.3 动作解码策略

| 策略 | 原理 | 优点 | 缺点 |
|:---|:---|:---|:---|
| **自回归解码** | 逐 token 生成动作 | 简单，与 LLM 一致 | 推理慢，累积误差 |
| **扩散策略 (Diffusion Policy)** | 从噪声中迭代去噪生成动作 | 处理多模态分布 | 推理步数多 |
| **流匹配 (Flow Matching)** | 学习从噪声到动作的连续流 | 速度快 4-12× | 训练复杂 |
| **动作分块 (Action Chunking)** | 一次预测多步动作 | 减少累积误差，运动平滑 | 需要更多训练数据 |

### 3.4 System 1 vs System 2 架构

现代 VLA 通常采用**双系统架构**：

| 系统 | 类比 | 功能 | 实现方式 |
|:---|:---|:---|:---|
| **System 1** | 快速直觉 | 高频低层反应控制（30-100Hz） | 扩散策略、流匹配 |
| **System 2** | 慢速推理 | 高层语义规划与推理 | LLM/VLM 推理 |

---

## 4. 第三阶段：经典论文精读

> ⏱ 预计时间：第 5-6 月 | 目标：深入理解领域里程碑工作

### 4.1 必读论文（按阅读顺序）

#### 🥇 基础篇

| 序号 | 论文 | 年份 | 核心贡献 | 链接 |
|:---|:---|:---|:---|:---|
| 1 | **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** | 2023 | 引入扩散模型生成机器人动作，处理多模态分布 | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) |
| 2 | **ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware** | 2023 | 动作分块 Transformer，解决累积误差问题 | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) |
| 3 | **Mobile ALOHA: Learning Bimanual Mobile Manipulation** | 2024 | 低成本移动双臂遥操作+协同训练 | [arXiv:2401.02117](https://arxiv.org/abs/2401.02117) |

#### 🥈 VLA 核心篇

| 序号 | 论文 | 年份 | 核心贡献 | 链接 |
|:---|:---|:---|:---|:---|
| 4 | **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control** | 2023 | VLA 范式开创者，将机器人动作视为文本 token | [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) |
| 5 | **OpenVLA: An Open-Source Vision-Language-Action Model** | 2024 | 开源 7B VLA 模型（Llama-2 + DINOv2 + SigLIP） | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) |
| 6 | **Octo: An Open-Source Generalist Robot Policy** | 2024 | 基于扩散的轻量开源通用策略，多具身支持 | [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) |
| 7 | **π0: A Vision-Language-Action Flow Model for General Robot Control** | 2024 | Physical Intelligence 出品，流匹配 + VL 基础模型 | [项目页面](https://www.physicalintelligence.company/blog/pi0) |

#### 🥉 进阶篇

| 序号 | 论文 | 年份 | 核心贡献 | 链接 |
|:---|:---|:---|:---|:---|
| 8 | **Open X-Embodiment: Robotic Learning Datasets and RT-X Models** | 2024 | 大规模多机器人数据集，ICRA 2024 最佳论文 | [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) |
| 9 | **OpenVLA-OFT** | 2025 | 优化微调策略：并行解码+动作分块+连续动作 | [项目页面](https://openvla-oft.github.io/) |
| 10 | **RDT-1B** | 2024 | 10 亿参数扩散 Transformer 用于双手操作 | [arXiv:2410.07864](https://arxiv.org/abs/2410.07864) |

### 4.2 综述论文推荐

| 论文 | 视角 | 链接 |
|:---|:---|:---|
| **A Survey on VLA Models: An Action Tokenization Perspective** (2025.07) | 按动作 tokenization 方式分类 | [arXiv:2507.01925](https://arxiv.org/abs/2507.01925) |
| **Vision-Language-Action Models for Embodied AI: A Survey** (2025) | 按控制层级（低层/高层）分类 | IEEE TNNLS |
| **A Comprehensive Review of Efficient VLA Models** (2025.10) | 高效模型设计/训练/数据 | [arXiv:2510.24795](https://arxiv.org/abs/2510.24795) |
| **VLA Models for Robotics: A Review Towards Real-World Applications** (2025) | 面向实际部署的实用指南 | IEEE Access |
| **VLA Models: A Systematic Review** (2025.09) | 300+ 论文的系统性综述 | [arXiv](https://arxiv.org/) |

### 4.3 论文阅读方法建议

1. **第一遍**：快速浏览 Abstract、Introduction、Figures，理解核心思想
2. **第二遍**：精读 Method 部分，理解架构细节与训练流程
3. **第三遍**：结合代码实现（如果有开源），理解工程实现
4. **做笔记**：记录每篇论文的核心创新、局限性和你的思考

---

## 5. 第四阶段：动手实践

> ⏱ 预计时间：第 7 月+ | 目标：在仿真或真实机器人上训练和部署 VLA 模型

### 5.1 实践路径选择

#### 路径 A：纯仿真路径（推荐入门）

```
1. 安装 LeRobot → 2. 在 SIMPLER-env 中评估预训练模型 → 3. 在 LIBERO 中微调 → 4. 收集自定义数据
```

#### 路径 B：仿真+真实路径（有硬件条件）

```
1. 搭建 ALOHA 硬件 → 2. 采集示教数据 → 3. 训练 ACT/扩散策略 → 4. 部署测试
```

#### 路径 C：VLA 模型微调路径（有 GPU 条件）

```
1. 下载 OpenVLA 权重 → 2. LoRA 微调 → 3. 仿真评估 → 4. 真实部署
```

### 5.2 关键实践项目

#### 项目 1：OpenVLA 微调实战

**目标**：在自己的任务上微调 OpenVLA 模型

**步骤**：
1. Clone 仓库并设置环境
2. 准备 RLDS 格式数据
3. 使用 LoRA/QLoRA 进行参数高效微调
4. 在 SIMPLER-env 或 LIBERO 中评估

**关键资源**：
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA-OFT 项目页](https://openvla-oft.github.io/)
- 硬体需求：RTX 3090/4090 (24GB VRAM) 即可运行 LoRA 微调

#### 项目 2：LeRobot 上手

**目标**：使用 HuggingFace LeRobot 工具包训练模仿学习策略

**步骤**：
1. 安装 LeRobot: `pip install lerobot`
2. 加载预训练数据集
3. 训练 ACT 或扩散策略
4. 在仿真或真实机器人上评估

**关键资源**：
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot 文档](https://huggingface.co/lerobot)
- [LeRobot-LIBERO 集成](https://github.com/huggingface/lerobot-libero)

#### 项目 3：Diffusion Policy 从零实现

**目标**：理解扩散策略的底层原理

**步骤**：
1. 阅读 Diffusion Policy 论文
2. 实现一个简单的 DDPM 去噪网络
3. 在模拟环境中训练和评估
4. 对比自回归 vs 扩散策略的效果差异

**关键资源**：
- [Diffusion Policy 官方代码](https://github.com/columbia-ai-robotics/diffusion_policy)
- [Denoising Diffusion Probabilistic Models 论文](https://arxiv.org/abs/2006.11239)

### 5.3 仿真环境对比

| 环境 | 特点 | 适合场景 | 链接 |
|:---|:---|:---|:---|
| **SIMPLER-env** | GPU 加速 (10-15×)，高度模拟真实场景 | VLA 模型评估，Sim-to-Real | [GitHub](https://github.com/simpler-env/SimplerEnv) |
| **LIBERO** | 终身学习基准，130 个操作任务 | 多任务迁移学习 | [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| **ManiSkill** | 基于 SAPIEN 的 GPU 并行仿真 | 大规模策略训练 | [GitHub](https://github.com/haosulab/ManiSkill) |
| **RoboVerse** | 自动化数据采集与 RLDS 格式转换 | 多环境数据收集 | [GitHub](https://github.com/RoboVerseOrg/RoboVerse) |
| **Isaac Sim / Isaac Lab** | NVIDIA 出品，物理真实度高 | 工业级仿真 | [官网](https://developer.nvidia.com/isaac-sim) |
| **MuJoCo** | 轻量级物理引擎 | 经典 RL 研究 | [GitHub](https://github.com/google-deepmind/mujoco) |

### 5.4 硬件平台参考

| 平台 | 成本 | 适合场景 | 链接 |
|:---|:---|:---|:---|
| **ALOHA 2 (双臂静态)** | ~$3K-$5K | 精细双手操作 | [GitHub](https://github.com/tonyzhaozh/act) |
| **Mobile ALOHA** | ~$32K | 移动操作 | [GitHub](https://github.com/MarkFzp/mobile-aloha) |
| **SO-100 (LeRobot)** | ~$100 | 极低成本入门 | [GitHub](https://github.com/huggingface/lerobot) |
| **WidowX 250** | ~$3K | 桌面操作 | [Trossen Robotics](https://www.trossenrobotics.com/widowx-250.aspx) |
| **Franka Emika Panda** | ~$25K | 工业级研究 | [Franka](https://www.franka.de/) |

---

## 6. 关键数据与工具生态

### 6.1 数据集

| 数据集 | 规模 | 特点 | 链接 |
|:---|:---|:---|:---|
| **Open X-Embodiment (OXE)** | 1M+ 轨迹，22 种机器人 | VLA 预训练核心数据集，ICRA 2024 最佳论文 | [项目页](https://robotics-transformer-x.github.io/) |
| **BridgeData V2** | 60K+ 轨迹 | WidowX 桌面操作，使用最广泛 | [GitHub](https://github.com/rail-berkeley/bridge_data_v2) |
| **DROID** | 76K 轨迹，564 个场景 | 多样化操作数据 | [项目页](https://droid-dataset.github.io/) |
| **RoboMIND** | 大规模合成数据 | 多机器人、多任务 | - |
| **ABC-130k** | 130K 轨迹 | 大规模双手操作 | - |

### 6.2 数据格式

- **RLDS (Reinforcement Learning Datasets)**: TensorFlow 生态的数据格式，OpenVLA 使用
- **LeRobotDataset**: Parquet + 视频格式，HuggingFace Hub 托管
- 两种格式可以通过工具互相转换

### 6.3 模型权重与 Hub

| 模型 | 参数量 | 发布方 | 链接 |
|:---|:---|:---|:---|
| **OpenVLA** | 7B | Stanford/UC Berkeley | [HuggingFace](https://huggingface.co/openvla) |
| **Octo** | 27M/93M | UC Berkeley | [HuggingFace](https://huggingface.co/octo-models) |
| **RT-2** | 5B/55B | Google DeepMind | 闭源 |
| **π0** | - | Physical Intelligence | 闭源 |

### 6.4 工具链速查

| 工具 | 用途 | 链接 |
|:---|:---|:---|
| **LeRobot** | 数据采集、训练、部署一体化 | [GitHub](https://github.com/huggingface/lerobot) |
| **RLDS** | 机器人数据标准格式 | [GitHub](https://github.com/google-research/rlds) |
| **TensorFlow Datasets** | OXE 数据集加载 | [TFDS](https://www.tensorflow.org/datasets) |
| **PEFT/LoRA** | 大模型参数高效微调 | [HuggingFace PEFT](https://github.com/huggingface/peft) |
| **Weights & Biases** | 实验跟踪 | [wandb.ai](https://wandb.ai/) |

---

## 7. 前沿方向与最新趋势

### 7.1 高效 VLA（Efficient VLA）

将大模型压缩到边缘设备上运行，目标是在机器人端侧实现 30-100Hz 推理。

| 代表工作 | 方法 | 链接 |
|:---|:---|:---|
| **SmolVLA** | 模型蒸馏 + 量化 | - |
| **TinyVLA** | 轻量化架构设计 | - |
| **OpenVLA-OFT** | 并行解码 + 动作分块 | [项目页](https://openvla-oft.github.io/) |

### 7.2 世界模型增强 VLA（World-Model-Augmented VLA）

不仅预测动作，还预测未来状态，让机器人"预演"动作后果。

### 7.3 多模态感知扩展

| 方向 | 描述 |
|:---|:---|
| **触觉 VLA (VLA+)** | Microsoft Rho-alpha，融合触觉传感器 |
| **3D VLA** | 利用 3D 点云/NeRF 进行空间推理 |
| **音频 VLA** | 融合声音信号进行操作反馈 |

### 7.4 数据瓶颈

当前 VLA 领域最大的瓶颈是**数据质量**而非模型架构或算力：

- 标注质量参差不齐
- 数据同质化严重
- 跨具身数据难以对齐
- 高效数据采集策略（如自动标注、合成数据生成）是重要方向

### 7.5 跨具身泛化（Cross-Embodiment Generalization）

让同一个 VLA 模型控制不同形态的机器人（单臂、双臂、移动底盘、四足等），而不需要为每种机器人重新训练。

---

## 8. 推荐资源汇总

### 8.1 Awesome Lists（持续更新）

| 名称 | 内容 | 链接 |
|:---|:---|:---|
| **awesome-vision-language-action** | 最全面的 VLA 论文分类汇总 | [GitHub](https://github.com/ai4s-research/awesome-vision-language-action) |
| **Awesome-VLA-Papers** | 200+ 论文，按领域分类 | [GitHub](https://github.com/hanjianhua44/Awesome-VLA-Papers) |
| **Awesome-RL-VLA** | RL + VLA 交叉方向 | [GitHub](https://github.com/topics/rl-vla) |

### 8.2 社区与讨论

| 平台 | 话题 | 链接 |
|:---|:---|:---|
| **Reddit r/robotics** | 机器人学习讨论 | [reddit.com/r/robotics](https://reddit.com/r/robotics) |
| **Reddit r/MachineLearning** | VLA 论文讨论 | [reddit.com/r/MachineLearning](https://reddit.com/r/MachineLearning) |
| **HuggingFace 社区** | LeRobot、模型分享 | [huggingface.co/lerobot](https://huggingface.co/lerobot) |
| **Twitter/X** | 关注 @chelseabfinn, @svlevine, @tonyzzhao 等研究者 | - |

### 8.3 博客与教程

| 资源 | 内容 | 链接 |
|:---|:---|:---|
| **Lil'Log** | VLA 相关技术博客 | [lilianweng.github.io](https://lilianweng.github.io/) |
| **HuggingFace Blog** | LeRobot 教程 | [huggingface.co/blog](https://huggingface.co/blog) |
| **Physical Intelligence Blog** | π0 技术细节 | [physicalintelligence.company](https://www.physicalintelligence.company/blog) |
| **Google DeepMind Blog** | RT-2 官方解读 | [deepmind.google](https://deepmind.google/discover/blog/) |

### 8.4 推荐学习路线总览

```
时间线      │  阶段              │  核心任务
───────────┼────────────────────┼─────────────────────────────────────
第 1-2 月   │  🔧 基础夯实       │  PyTorch, Transformer, 模仿学习, ROS 2
第 3-4 月   │  📖 理论深入       │  动作 Tokenization, 多模态融合, 扩散策略
第 5-6 月   │  📄 论文精读       │  RT-2 → OpenVLA → Octo → π0 → 综述
第 7 月+    │  🛠️ 动手实践       │  LeRobot 训练 → OpenVLA 微调 → 仿真评估
───────────┼────────────────────┼─────────────────────────────────────
持续进行   │  🔭 前沿追踪       │  Awesome List 订阅, arxiv 每日更新
```

---

> 💡 **学习建议**：
> 1. **不要跳过基础**：Transformer 和模仿学习是 VLA 的两大支柱，基础不牢后面会很难
> 2. **论文+代码结合**：每篇论文尽量找到代码实现，跑通再看论文效果翻倍
> 3. **从仿真开始**：仿真环境零成本试错，熟练后再上真实硬件
> 4. **加入社区**：关注 GitHub Issue、Reddit 讨论，遇到问题多交流
> 5. **动手实践是王道**：VLA 是工程密集型的领域，光看不练学不会

---

*本文档持续更新，如有新的优质资源欢迎补充。*