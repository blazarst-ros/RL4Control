# CartPole-RL

这是一个基于强化学习的 CartPole 项目，使用
[Gymnasium](https://gymnasium.farama.org/) 和
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) 实现
**DQN（Deep Q-Network）** 训练、评估、渲染和实验对比流程。

这个仓库主要覆盖三类任务：

1. 在 `CartPole-v1` 环境中训练 DQN 智能体。
2. 加载已保存模型进行评估和可视化。
3. 对关键超参数做对比实验，并生成完整的实验归档。

除了训练代码之外，项目还保留了历史检查点、已训练模型、实验输出
目录和一份中文项目说明文档。

## 项目结构

```text
.
├── scripts/
│   ├── train_dqn.py
│   ├── evaluate_dqn.py
│   ├── run_cartpole_experiments.py
│   ├── test_env.py
│   └── test_render.py
├── CheckPoints/
├── ModelWeight&TrainingStrategy/
├── Matlab_Wheeled_legged/
├── requirements.txt
├── environment.yml
└── CartPole强化学习项目使用指南.docx
```

### 主要目录说明

- `scripts/`：训练、评估、测试和实验批处理的入口脚本。
- `CheckPoints/`：历史训练过程中保存的模型检查点。
- `ModelWeight&TrainingStrategy/`：已训练好的模型权重文件。
- `Matlab_Wheeled_legged/`：轮腿倒立摆 LQR 补充实验，用于和强化学习方法做传统控制对比。
- `requirements.txt`：当前环境导出的 Python 依赖。
- `environment.yml`：Conda 环境定义。
- `CartPole强化学习项目使用指南.docx`：中文项目使用说明。

## CartPole 环境简介

本项目的目标环境是 `CartPole-v1`。这是一个经典控制任务，智能体需
要通过选择左右两个离散动作，让小车上的杆尽可能保持平衡。

### 观测空间

环境返回一个 4 维连续状态：

- 小车位置
- 小车速度
- 杆的角度
- 杆的角速度

### 动作空间

- 向左移动
- 向右移动

当杆倾倒过多、小车越界或达到最大步数时，回合结束。

## 算法说明

项目采用 **DQN** 作为核心算法。

### DQN 在这里做什么

DQN 会学习动作价值函数 `Q(s, a)`，并根据状态选择价值最高的动作。
训练过程中主要包含以下机制：

- **经验回放**：把交互样本存入 replay buffer，再用小批量方式训练。
- **目标网络更新**：使用单独的 target network 提升训练稳定性。
- **epsilon-greedy 探索**：训练早期随机探索，随后逐步偏向贪心策略。

### 为什么适合 CartPole

CartPole 具有以下特点：

- 观测维度小
- 动作空间离散
- 训练速度快
- 奖励信号明确

因此它非常适合用来验证基于价值的强化学习方法。

## Policy 和 Model

训练脚本使用 Stable-Baselines3 的 `DQN` 实现，并采用：

- `policy="MlpPolicy"`：用多层感知机近似 Q 函数
- `device="auto"`：自动选择 GPU 或 CPU
- `seed=42`：默认随机种子

在 DQN 中，“policy” 更多是指由 Q 值推导出的动作选择行为，而不是
单独训练一个随机策略网络。

## 训练配置

默认训练脚本是 [`scripts/train_dqn.py`](./scripts/train_dqn.py)，其主要
参数如下：

- 环境：`CartPole-v1`
- 算法：DQN
- Policy：`MlpPolicy`
- 总训练步数：`50_000`
- 学习率：`1e-3`
- replay buffer 大小：`50_000`
- 开始学习步数：`1_000`
- batch size：`64`
- 折扣因子 `gamma`：`0.99`
- 训练频率：每 `4` 个环境步更新一次
- target network 更新间隔：`1_000`
- exploration fraction：`0.2`
- checkpoint 保存频率：每 `5_000` 步

### 训练输出

运行训练脚本后，会生成：

- `models/cartpole_dqn.zip`：最终模型
- `checkpoints/`：周期性 checkpoint 和 replay buffer
- `logs/dqn/`：TensorBoard 日志
- 终端输出中的评估奖励结果

注意：仓库中已经存在一个历史的 `CheckPoints/` 目录，里面保存了旧
的模型快照。当前训练脚本会使用小写的 `checkpoints/` 目录保存新结
果。

## 实验批处理

更完整的实验流程位于
[`scripts/run_cartpole_experiments.py`](./scripts/run_cartpole_experiments.py)。
它会运行多组 DQN 训练任务，自动归档结果，并生成对比报告。

### 对比组

当前支持三类超参数对比：

- `learning_rate`
- `gamma`
- `exploration`

### 记录内容

每次运行会记录：

- 训练过程中的回合奖励
- 回合长度
- loss
- exploration rate
- 评估奖励随训练步数的变化
- checkpoints
- 最终模型文件
- 运行配置和元数据

### 生成的实验产物

每次 sweep 都会被归档到一个带时间戳的目录中，目录结构包含：

- `sweep_config.json`
- 每个 run 的 `config.json`
- 每个 run 的 `summary.json`
- `metrics/`
- `plots/`
- `checkpoints/`
- `evaluation/`
- `models/`
- `comparison/` 下的对比总结和图表

`comparison/` 目录中会包含：

- `summary.csv`
- `summary.json`
- `summary.md`
- 奖励、loss、探索率和评估曲线的对比图

### 本次报告使用的数据

课程报告使用的实验归档位于工作区：

```text
Data/cartpole_dqn_report/20260712_193429/
```

该归档包含三组参数对比：

- `learning_rate`：对比 `5e-4`、`1e-3`、`2e-3`、`5e-3`。
- `gamma`：对比 `0.90`、`0.95`、`0.99`、`0.995`。
- `exploration`：对比默认 epsilon-greedy、快速衰减、慢速衰减和 greedy。

每组实验下均保存 `comparison/summary.csv`、对比曲线、单次运行配置、
训练指标、评估指标、模型文件和检查点。报告中选用 `gamma_0p95_seed_42`
作为最终分析配置，其独立评估平均奖励为 `248.0`。

## 脚本说明

### 训练

```bash
python scripts/train_dqn.py
```

训练一个 DQN 智能体，保存最终模型，并输出 10 个评估回合的奖励均值
和标准差。

### 评估并渲染

```bash
python scripts/evaluate_dqn.py
```

加载 `models/cartpole_dqn.zip`，执行一个确定性回合，并以窗口方式渲染
环境。

### 环境自检

```bash
python scripts/test_env.py
```

用于快速检查 CartPole 环境是否可用，并打印观测空间、动作空间等信息。

### 渲染自检

```bash
python scripts/test_render.py
```

随机采取动作并打开渲染窗口，验证图形化显示是否正常。

### 实验批处理

```bash
python scripts/run_cartpole_experiments.py
```

也可以通过参数自定义实验范围：

```bash
python scripts/run_cartpole_experiments.py \
  --data-root /path/to/archive/root \
  --sweep-name cartpole_dqn_report \
  --groups learning_rate gamma exploration \
  --seeds 42 7 \
  --total-timesteps 50000 \
  --eval-freq 5000 \
  --eval-episodes 10 \
  --checkpoint-freq 5000
```

### MATLAB LQR 补充实验

```matlab
cd Matlab_Wheeled_legged
run_wheel_leg_lqr_demo
```

该补充实验会建立轮驱倒立摆的小角度线性化模型，使用 `lqr(A, B, Q, R)`
计算反馈增益，并在非线性摆模型上仿真响应。主要文件包括：

- `run_wheel_leg_lqr_demo.m`：主入口脚本。
- `wheel_leg_lqr_config.m`：物理参数、LQR 权重、仿真和可视化配置。
- `linearized_wheel_leg_model.m`：线性化状态空间模型。
- `wheel_leg_dynamics.m`：带饱和控制输入的非线性仿真模型。
- `animate_wheel_leg_lqr.m`：轮腿小车和五连杆腿部可视化。
- `plot_response.m`：位置、俯仰角和控制输入曲线。

## 已包含的模型与检查点

仓库中已经附带了若干训练结果：

- `CheckPoints/cartpole_dqn_30000_steps.zip`
- `CheckPoints/cartpole_dqn_35000_steps.zip`
- `CheckPoints/cartpole_dqn_40000_steps.zip`
- `CheckPoints/cartpole_dqn_45000_steps.zip`
- `CheckPoints/cartpole_dqn_50000_steps.zip`
- `ModelWeight&TrainingStrategy/cartpole_dqn.zip`

如果你只想查看一个已训练好的策略，可以直接使用这些文件，而不必
重新训练。

## 环境安装

### Conda

```bash
conda env create -f environment.yml
conda activate RL
```

### pip

如果你更习惯 `pip`，可以在 Python 3.10 环境中安装
`requirements.txt` 中列出的依赖。

## 备注

- 本项目面向的是 `CartPole-v1`，不是 Atari 或图像输入任务。
- 评估脚本默认从 `models/cartpole_dqn.zip` 读取模型。
- `run_cartpole_experiments.py` 会把实验结果写入可配置的数据根目录，
  适合做可重复的超参数对比。
- GitHub 仓库地址：<https://github.com/blazarst-ros/RL4Control>

## 许可证

仓库当前没有 `LICENSE` 文件。如果你计划公开分发或二次使用，请补充
相应许可证。
