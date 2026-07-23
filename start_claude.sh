#!/bin/bash

# 启动 ept claude，使用 kivy-deepseek-v4-pro[1m] 模型
# 自动激活 uniad2.0 conda 环境

# 初始化 conda（适配非交互式 shell）
eval "$(conda shell.bash hook)"

# 激活 uniad2.0 环境
conda activate uniad2.0

# 启动 ept claude
ept claude --model "kivy-deepseek-v4-pro[1m]"