# Physical AI (Embodied AI) Primer

> **Physical AI / Embodied AI Primer**
>
> This document is aimed at beginners. It systematically introduces the basic concepts, history, technology stack, and applications of Physical AI (Embodied AI).

---

## Table of Contents

1. [What Is Physical AI / Embodied AI](#1-what-is-physical-ai--embodied-ai)
2. [Differences from Traditional AI](#2-differences-from-traditional-ai)
3. [History](#3-history)
4. [Core Technology Stack](#4-core-technology-stack)
5. [Application Domains](#5-application-domains)
6. [Suggested Learning Path](#6-suggested-learning-path)
7. [Summary & Outlook](#7-summary--outlook)

---

## 1. What Is Physical AI / Embodied AI

### 1.1 Definition

**Physical AI**, also known as **Embodied AI**, refers to intelligent systems with physical bodies that can perceive, understand, and interact with the world in real or simulated environments.

> 💡 **Core idea**: Intelligence is not just the computing power of a brain — it emerges from the **continuous interaction** between an agent and its physical environment.

### 1.2 Core Concepts

| Concept | Description |
|---------|-------------|
| **Embodiment** | The agent has a physical form; it can affect and be affected by the environment |
| **Perception-Action Loop** | Sense via sensors → decide → act on the environment through actuators |
| **World Model** | The agent's internal representation and prediction of the environment |
| **Emergent Intelligence** | Complex intelligent behavior emerges from simple interaction rules |

### 1.3 The Embodied Hypothesis

```
┌─────────────────────────────────────────────────────────────┐
│                     具身假说                                 │
│                                                             │
│   "智能无法脱离身体而存在，认知是身体与环境的耦合产物"          │
│                                                             │
│   — 认知科学 & 机器人学的共识                                 │
└─────────────────────────────────────────────────────────────┘
```

*(具身假说 = The Embodied Hypothesis — "Intelligence cannot exist apart from a body; cognition is the product of coupling between body and environment." — a consensus of cognitive science & robotics)*

**Key insights**:
- The body is not a container but a **constituent part** of intelligence
- Perception, cognition, and action are an **inseparable whole**
- Intelligence is **situated**, not abstract symbol manipulation

---

## 2. Differences from Traditional AI

### 2.1 Comparison Overview

| Dimension | Traditional AI | Physical AI / Embodied AI |
|-----------|----------------|---------------------------|
| Existence | Pure software / virtual | Physical entity + software |
| Interaction target | Digital data | The real physical world |
| Input form | Structured data | Multimodal sensory data |
| Output form | Predictions / classifications | Physical actions / behaviors |
| Learning source | Static datasets | Real-time interaction experience |
| Environment feedback | Labels / loss functions | Physical outcomes / reward signals |
| Time dimension | Single-step / batch | Continuous sequential decisions |

### 2.2 Architecture Comparison

**Traditional AI (e.g., LLMs)**:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   文本输入    │ ──▶ │   神经网络    │ ──▶ │   文本输出    │
│  (Prompt)    │     │   推理计算    │     │  (Response)  │
└──────────────┘     └──────────────┘     └──────────────┘
                            ▲
                            │
                     ┌──────────────┐
                     │  预训练权重   │
                     │ (静态知识)   │
                     └──────────────┘
```

*(文本输入 = Text input, 神经网络/推理计算 = Neural network / inference, 文本输出 = Text output, 预训练权重 (静态知识) = Pretrained weights (static knowledge))*

**Physical AI / Embodied AI**:

```
                    ┌─────────────────────────────────────┐
                    │           物理世界                    │
                    │  ┌─────────┐     ┌─────────┐        │
                    │  │  物体   │     │   人类  │        │
                    │  └─────────┘     └─────────┘        │
                    └──────────┬──────────────────────────┘
                               │ 力/运动/触觉
                               ▼
┌──────────┐    ┌──────────┐   ┌──────────┐    ┌──────────┐
│  传感器   │───▶│ 感知模块  │──▶│ 决策模块  │───▶│ 执行器   │
│  Sensor  │    │Perception│   │  Policy  │    │Actuator │
└──────────┘    └──────────┘   └──────────┘    └──────────┘
   视觉                              ▲              │
   听觉                              │              │
   触觉                              │              │
   力觉                              │              │
                             ┌──────────┐           │
                             │ 世界模型  │───────────┘
                             │World Model│
                             └──────────┘
```

*(物理世界 = Physical world, 物体 = Objects, 人类 = Humans, 力/运动/触觉 = Force/motion/tactile, 传感器 = Sensors, 感知模块 = Perception, 决策模块 = Policy/Decision, 执行器 = Actuator, 视觉/听觉/触觉/力觉 = Vision/Hearing/Touch/Force, 世界模型 = World Model)*

### 2.3 Key Differences Summary

| Limitation of Traditional AI | Breakthrough of Physical AI |
|------------------------------|------------------------------|
| Lacks physical common sense | Learns physical laws through interaction |
| Cannot handle uncertainty | Real-time adaptation & fault tolerance |
| Static knowledge | Dynamic experience accumulation |
| Detached from environment | The environment is the teacher |

---

## 3. History

### 3.1 Timeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        具身智能发展时间线                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1950s      1970s       1990s       2010s       2020s       现在    │
│    │         │           │           │           │          │       │
│    ▼         ▼           ▼           ▼           ▼          ▼       │
│  ┌────┐   ┌────┐     ┌────┐     ┌────┐     ┌────┐     ┌────┐       │
│  │起源│   │控制│     │学习│     │深度│     │大模型│    │爆发│       │
│  └────┘   └────┘     └────┘     └────┘     └────┘     └────┘       │
│    │         │           │           │           │                  │
│  图灵测试   Shakey    强化学习    AlexNet    GPT时代    具身        │
│  思考机器   移动机器人   出现       革命      开始      智能        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

*(Timeline: 起源/图灵测试思考机器 = Origins / Turing test "thinking machines" → 控制/Shakey 移动机器人 = Control / Shakey mobile robot → 学习/强化学习出现 = Learning / RL emerges → 深度/AlexNet 革命 = Deep learning / AlexNet revolution → 大模型/GPT时代开始 = Foundation models / GPT era → 爆发/具身智能 = Explosion / Embodied AI)*

### 3.2 Development Phases

#### 🔹 Phase 1: Early Exploration (1950s-1980s)

**Milestones**:
- **1950** — Turing asks "can machines think," launching AI research
- **1966** — Shakey (SRI International): the first mobile robot capable of reasoning and acting
- **1980s** — Rise of industrial robots (KUKA, ABB, FANUC)

**Characteristics**:
- Rule-based symbolic AI
- Simple perception-reaction loops
- Restricted to structured environments

#### 🔹 Phase 2: The Behaviorist Revolution (1986-2000s)

**Milestones**:
- **1986** — Rodney Brooks proposes the **Subsumption Architecture**
- Emphasizes intelligence **without internal representations**
- Hierarchical control from insects to complex behaviors

```
┌─────────────────────────────────────────────────┐
│           包容架构（Subsumption Architecture）    │
├─────────────────────────────────────────────────┤
│                                                 │
│  Layer 4: 目标导向行为（如探索、地图构建）         │
│           ▲                                    │
│  Layer 3: 有限状态行为（如漫游、跟随）            │
│           ▲                                    │
│  Layer 2: 反应式行为（如避障、保持平衡）          │
│           ▲                                    │
│  Layer 1: 本能行为（如防撞、站立）               │
│           ▲                                    │
│      传感器输入 → 世界                            │
│                                                 │
│  特点：下层可抑制上层，无需中央控制器              │
└─────────────────────────────────────────────────┘
```

*(Layer 4: goal-directed behavior (exploration, mapping); Layer 3: finite-state behavior (wandering, following); Layer 2: reactive behavior (obstacle avoidance, balance); Layer 1: instinctive behavior (collision avoidance, standing); lower layers can suppress upper layers; no central controller needed)*

#### 🔹 Phase 3: Statistical Learning & Probabilistic Methods (2000s-2010s)

**Milestones**:
- **SLAM** (Simultaneous Localization and Mapping) matures
- **Probabilistic robotics** rises (Sebastian Thrun)
- Research platforms such as **PR2, TurtleBot** become widespread

**Technical breakthroughs**:
- Bayesian filters (particle filter, Kalman filter)
- Sampling-based motion planning (RRT, PRM)
- Early deep learning applied to vision

#### 🔹 Phase 4: The Deep Learning Revolution (2012-2020)

**Milestones**:
- **2012** — AlexNet ignites deep learning
- **2016** — AlphaGo defeats Lee Sedol
- **End-to-end learning** applied to autonomous driving and robot control

**Paradigm shift**:
```
传统方法：传感器 → 特征提取 → 状态估计 → 规划 → 控制
               ↑________ 人工设计 ________↑

深度学习方法：传感器 → [神经网络] → 控制/动作
                    ↑___ 端到端学习 ___↑
```

*(Traditional: sensor → feature extraction → state estimation → planning → control, with hand-designed pipeline; Deep learning: sensor → neural network → control/action, end-to-end)*

#### 🔹 Phase 5: The Foundation Model Era (2020-Present)

**Milestones**:
- **2020** — GPT-3 demonstrates the power of large-scale pretraining
- **2022** — ChatGPT sparks the general-AI wave
- **2023-2025** — **Embodied AI explodes**:
  - Google RT-1/RT-2/RT-3: Transformer-driven robot control
  - Stanford ALOHA: low-cost dual-arm teleoperation
  - Unitree H1 / G1: Chinese humanoid robot breakthrough
  - Tesla Optimus: mass-manufacturing vision
  - Figure 01: end-to-end neural network control breakthrough
  - NVIDIA GR00T: general humanoid robot foundation model
  - DeepMind RTX: generalization breakthrough

```
┌─────────────────────────────────────────────────────────────┐
│              大模型驱动的具身智能架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│   │  视觉语言模型 │    │   大语言模型  │    │  动作模型    │ │
│   │   (VLM)      │    │    (LLM)     │    │   (VLA)      │ │
│   │  CLIP/LLaVA │    │ GPT/Claude   │    │ RT-2/GR00T   │ │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘ │
│          │                   │                   │         │
│          └───────────────────┼───────────────────┘         │
│                              ▼                             │
│                    ┌──────────────────┐                    │
│                    │   多模态大模型    │                    │
│                    │  (PaLM-E, RTX)   │                    │
│                    └────────┬─────────┘                    │
│                             ▼                              │
│                    ┌──────────────────┐                    │
│                    │   机器人执行平台   │                    │
│                    └──────────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Architecture: VLM (CLIP/LLaVA) + LLM (GPT/Claude) + action model (RT-2/GR00T) → multimodal foundation model (PaLM-E, RTX) → robot execution platform)*

---

## 4. Core Technology Stack

### 4.1 Full Stack Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           物理AI / 具身智能技术栈                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     应用层（Applications）                           │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ 服务机器人│ │ 自动驾驶 │ │ 工业机械臂│ │ 人形机器人│ │ 医疗机器人│  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     智能层（Intelligence）                           │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │ 任务规划   │  │ 行为决策   │  │ 多模态理解 │  │ 人机交互   │    │   │
│  │  │ Task Plan  │  │ Decision   │  │ VLM/LLM   │  │ HRI        │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     算法层（Algorithms）                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │   │
│  │  │ 运动规划 │  │ 强化学习 │  │ 模仿学习 │  │ 世界模型 │  │扩散模型│ │   │
│  │  │ Planning │  │    RL    │  │   IL     │  │World Model│ │Diffusion│ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     感知层（Perception）                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ 计算机视觉│  │ 点云处理 │  │ 语义分割 │  │ 姿态估计 │            │   │
│  │  │   CV     │  │   LiDAR  │  │Segmentation│ │  Pose   │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     硬件层（Hardware）                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │   │
│  │  │ 机械结构 │  │ 传感器   │  │ 执行器   │  │ 计算平台 │  │ 电源  │ │   │
│  │  │Mechanism │  │ Sensor   │  │Actuator  │  │  CPU/GPU │  │Power │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

*(Stack layers, bottom-up: Hardware (mechanism, sensors, actuators, compute CPU/GPU, power) → Perception (CV, LiDAR point clouds, segmentation, pose estimation) → Algorithms (planning, RL, imitation learning, world models, diffusion) → Intelligence (task planning, decision-making, VLM/LLM understanding, HRI) → Applications (service robots, autonomous driving, industrial arms, humanoids, medical robots))*

### 4.2 Key Technologies

#### 4.2.1 Perception

**Multimodal sensor fusion**:

| Sensor Type | Data | Applications |
|-------------|------|--------------|
| RGB camera | 2D images | Object recognition, scene understanding |
| Depth camera (RGB-D) | 2D + depth | 3D reconstruction, grasping planning |
| LiDAR | 3D point clouds | SLAM, obstacle avoidance, navigation |
| IMU | Acceleration + gyroscope | State estimation, balance control |
| Force/torque sensor | 6-axis forces | Fine manipulation, force control |
| Tactile sensor | Pressure distribution | Grasp detection, texture recognition |

**Key techniques**:
- **Vision Transformer (ViT)**: splits images into patches for attention computation
- **3D vision (PointNet, Point Transformer)**: processes point clouds directly
- **Multimodal fusion (CLIP, ImageBind)**: unifies vision-language representations

#### 4.2.2 Decision & Control

**Comparison of mainstream methods**:

| Method | Principle | Pros | Cons |
|--------|-----------|------|------|
| **Classical control** | PID, MPC | Interpretable, stable | Requires accurate models |
| **Reinforcement learning (RL)** | Trial-and-error reward maximization | Discovers novel policies | Low sample efficiency |
| **Imitation learning (IL)** | Learn from expert demonstrations | High learning efficiency | Distribution shift |
| **Model-based RL (MBRL)** | Learn a world model to predict | Sample efficient | Model error accumulation |
| **Diffusion models** | Generate action sequences from noise | Strong expressiveness | High compute cost |
| **VLA (Vision-Language-Action)** | Multimodal foundation models output actions | Strong generalization | High inference latency |

#### 4.2.3 Motion Planning

```
┌─────────────────────────────────────────────────────────┐
│                  运动规划问题                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   给定：                                                 │
│   • 机器人构型空间（Configuration Space）                │
│   • 起始状态 q_start                                    │
│   • 目标状态 q_goal                                     │
│   • 障碍物（环境中不可穿越的区域）                        │
│                                                         │
│   求解：                                                 │
│   • 无碰撞路径 π: [0,1] → Q_free                       │
│   • 满足运动学/动力学约束                               │
│   • 优化：最短路径、最小能量、最平滑等                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

*(Given: robot configuration space, start state q_start, goal state q_goal, obstacles. Solve for: a collision-free path π: [0,1] → Q_free satisfying kinematic/dynamic constraints, optimized for shortest path / minimal energy / smoothness, etc.)*

**Classic algorithms**:
- **Sampling-based**: RRT (Rapidly-exploring Random Tree), PRM
- **Search-based**: A*, D* Lite
- **Optimization-based**: CHOMP, TrajOpt
- **Learning-augmented planning**: Neural MP, Diffusion Planner

#### 4.2.4 Learning Paradigms

```
┌─────────────────────────────────────────────────────────────────┐
│                     机器人学习的四大范式                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   │  强化学习    │  │  模仿学习    │  │  离线学习    │         │
│   │     RL       │  │      IL      │  │   Offline    │         │
│   ├──────────────┤  ├──────────────┤  ├──────────────┤         │
│   │ • 自探索      │  │ • 专家演示   │  │ • 历史数据   │         │
│   │ • 奖励驱动    │  │ • 行为克隆   │  │ • 无在线交互  │         │
│   │ • 试错学习    │  │ • 逆强化学习 │  │ • 保守估计   │         │
│   └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                     大模型驱动学习                        │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │ • 预训练视觉-语言模型 + 微调控制头                        │   │
│   │ • 语言指令理解 → 动作生成                                 │   │
│   │ • 零样本/少样本泛化能力                                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

*(Four paradigms of robot learning: RL (self-exploration, reward-driven, trial-and-error), IL (expert demos, behavior cloning, inverse RL), Offline (historical data, no online interaction, conservative estimates), Foundation-model-driven learning (pretrained vision-language models + fine-tuned control heads, language instructions → action generation, zero/few-shot generalization))*

---

## 5. Application Domains

### 5.1 Application Landscape

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           具身智能应用领域                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   🏠 家庭服务           🏭 工业制造           🚗 自动驾驶                   │
│   ┌──────────┐         ┌──────────┐         ┌──────────┐                   │
│   │扫地机器人 │         │协作机械臂 │         │自动驾驶车 │                   │
│   │护理机器人 │         │质量检测  │         │配送机器人 │                   │
│   │烹饪机器人 │         │装配作业  │         │飞行汽车  │                   │
│   └──────────┘         └──────────┘         └──────────┘                   │
│                                                                             │
│   🏥 医疗健康           🚀 太空探索           🌊 极端环境                   │
│   ┌──────────┐         ┌──────────┐         ┌──────────┐                   │
│   │手术机器人 │         │行星探测车 │         │深海探测器 │                   │
│   │康复外骨骼 │         │空间站维护 │         │灾难救援  │                   │
│   │药物配送  │         │小行星采矿 │         │核设施检修 │                   │
│   └──────────┘         └──────────┘         └──────────┘                   │
│                                                                             │
│   🎮 娱乐教育           🤖 人形机器人                                    │
│   ┌──────────┐         ┌──────────┐                                        │
│   │ 陪伴宠物 │         │ 通用人形 │                                        │
│   │ STEM教育 │         │ 家庭助手 │                                        │
│   │ 虚拟偶像 │         │ 工业员工 │                                        │
│   └──────────┘         └──────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

*(Home services (vacuum, care, cooking) · Industrial manufacturing (collaborative arms, quality inspection, assembly) · Autonomous driving (robotaxis, delivery robots, flying cars) · Healthcare (surgical robots, rehabilitation exoskeletons, drug delivery) · Space exploration (planetary rovers, station maintenance, asteroid mining) · Extreme environments (deep-sea, disaster rescue, nuclear facility inspection) · Entertainment & education (companion pets, STEM education, virtual idols) · Humanoids (general-purpose, home assistants, industrial workers))*

### 5.2 Key Applications

#### 🤖 Humanoid Robotics

**Why humanoid?**
1. **Environment compatibility**: human-world infrastructure is designed for human form
2. **Social acceptance**: anthropomorphic interaction feels more natural
3. **Generality**: one body, many tasks

**Representative products (2025-2026)**:
| Product | Company | Highlights |
|---------|---------|------------|
| Atlas | Boston Dynamics | Best locomotion capability |
| Figure 01/02 | Figure AI | End-to-end neural network control |
| Optimus Gen 3 | Tesla | Mass-manufacturing vision |
| Unitree H1/G1 | Unitree | High cost-effectiveness, leading in China |
| GR-1 | Fourier Intelligence | Leading in China |
| GR00T | NVIDIA | General humanoid foundation model |

**Industry milestones**:
- 2024: Figure 01 achieves end-to-end neural network control
- 2025: Tesla Optimus starts "working" in factories
- 2026: NVIDIA GR00T released — the era of general foundation models begins

#### 🚗 Autonomous Driving

**SAE levels (L0-L5)**:

```
Level 0 ───────▶ 无自动化（仅警告）
      │
Level 1 ───────▶ 驾驶辅助（单一功能：巡航/车道保持）
      │
Level 2 ───────▶ 部分自动化（组合功能，需人类监控）
      │
Level 3 ───────▶ 有条件自动化（特定条件下系统主导）
      │
Level 4 ───────▶ 高度自动化（限定区域/条件下完全自主）
      │
Level 5 ───────▶ 完全自动化（任何条件下无需人类干预）
```

*(L0 no automation (warnings only) → L1 driver assistance (single function: cruise/lane keeping) → L2 partial automation (combined functions, human monitoring required) → L3 conditional automation (system leads under specific conditions) → L4 high automation (fully autonomous in limited areas/conditions) → L5 full automation (no human intervention under any conditions))*

---

## 6. Suggested Learning Path

### 6.1 Knowledge Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         具身智能学习路径                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  第0阶段：基础准备                                                           │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐             │
│  │  数学基础    │  Python编程  │  机器学习    │  深度学习    │             │
│  │ 线性代数/微积分│   NumPy     │  scikit-learn│  PyTorch    │             │
│  └──────────────┴──────────────┴──────────────┴──────────────┘             │
│                              ↓                                              │
│  第1阶段：核心技能                                                           │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐             │
│  │ 计算机视觉   │  ROS/机器人  │  强化学习    │  SLAM        │             │
│  │  OpenCV     │  中间件      │  PPO/SAC    │  LIO-SAM    │             │
│  │  目标检测   │  运动学      │  离线RL     │  ORB-SLAM   │             │
│  └──────────────┴──────────────┴──────────────┴──────────────┘             │
│                              ↓                                              │
│  第2阶段：进阶专题                                                           │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐             │
│  │ 模仿学习     │  多模态大模型│  世界模型    │  扩散模型    │             │
│  │  DAgger     │  VLA/RT-X   │  Dreamer    │ Diffusion   │             │
│  │  GAIL       │  LLaVA      │  GAIRL      │  Policy     │             │
│  └──────────────┴──────────────┴──────────────┴──────────────┘             │
│                              ↓                                              │
│  第3阶段：实践项目                                                           │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐             │
│  │ 抓取任务     │  导航任务    │  操作任务    │  人形控制    │             │
│  │ 视觉伺服    │  自主探索    │  装配作业    │  行走平衡    │             │
│  └──────────────┴──────────────┴──────────────┴──────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

*(Stage 0 prep: math (linear algebra/calculus), Python (NumPy), ML (scikit-learn), DL (PyTorch). Stage 1 core skills: computer vision (OpenCV, detection), ROS/robotics middleware, RL (PPO/SAC, offline RL), SLAM (LIO-SAM, ORB-SLAM). Stage 2 advanced: imitation learning (DAgger, GAIL), multimodal foundation models (VLA/RT-X, LLaVA), world models (Dreamer), diffusion (Diffusion Policy). Stage 3 projects: grasping (visual servoing), navigation (autonomous exploration), manipulation (assembly), humanoid control (walking balance))*

### 6.2 Recommended Learning Resources

#### 📚 Books

| Book | Author | Note |
|------|--------|------|
| *Probabilistic Robotics* | Thrun, Burgard, Fox | The "bible" of probabilistic robotics |
| *Robotics, Vision and Control* | Peter Corke | Comprehensive introductory textbook |
| *Reinforcement Learning* | Sutton & Barto | RL classic |
| *Deep Learning for Robot Manipulation* | various | Deep learning for robotics |

#### 🎓 Online Courses

| Course | Platform | Note |
|--------|----------|------|
| CS223A - Robotics | Stanford | Robotics fundamentals |
| CS285 - Deep RL | UC Berkeley | Deep reinforcement learning |
| Robotic Manipulation | MIT 6.4210 | Robot manipulation |
| Visual Navigation | MIT | Visual navigation |

#### 🛠️ Open-Source Tools

| Tool | Purpose | Link |
|------|---------|------|
| **ROS 2** | Robot operating system | https://docs.ros.org |
| **Isaac Sim** | NVIDIA physics simulation | https://developer.nvidia.com/isaac-sim |
| **MuJoCo** | Physics engine | https://mujoco.org |
| **PyBullet** | Robot simulation | https://pybullet.org |
| **HuggingFace LeRobot** | Robot learning library | https://github.com/huggingface/lerobot |
| **Open X-Embodiment** | Datasets & models | https://robotics-transformer-x.github.io |
| **NVIDIA Isaac Lab** | Robot learning framework | https://github.com/isaac-sim/IsaacLab |

### 6.3 Practical Advice

```
┌─────────────────────────────────────────────────────────────────┐
│                     学习实践建议                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1️⃣  从仿真开始                                                  │
│      • 成本低、安全、可重复                                       │
│      • 推荐：Isaac Lab, MuJoCo, PyBullet                         │
│                                                                 │
│  2️⃣  掌握一个完整流程                                            │
│      • 感知 → 决策 → 控制                                        │
│      • 从一个简单任务（如：抓取方块）开始                          │
│                                                                 │
│  3️⃣  复现经典论文                                                │
│      • RT-2, ALOHA, Diffusion Policy, GR00T                     │
│      • 理解state-of-the-art方法的细节                             │
│                                                                 │
│  4️⃣  参与开源社区                                                │
│      • ROS社区、HuggingFace机器人板块                              │
│      • 贡献代码、提问、分享经验                                    │
│                                                                 │
│  5️⃣  硬件入门（可选）                                            │
│      • 低成本平台：TurtleBot, JetBot                               │
│      • 遥操作学习：ALOHA/Stretch                                   │
│      • 机械臂：xArm, Franka, UR                                    │
│      • 人形机器人：Unitree G1, Fourier GR-1                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

*(1. Start with simulation — low cost, safe, repeatable; try Isaac Lab, MuJoCo, PyBullet. 2. Master one full pipeline (perception → decision → control), starting from a simple task such as grasping a block. 3. Reproduce classic papers (RT-2, ALOHA, Diffusion Policy, GR00T). 4. Join open-source communities (ROS, HuggingFace robotics). 5. Optional hardware: low-cost platforms (TurtleBot, JetBot), teleoperation (ALOHA/Stretch), robot arms (xArm, Franka, UR), humanoids (Unitree G1, Fourier GR-1))*

---

## 7. Summary & Outlook

### 7.1 Current Challenges

| Challenge | Description | Research Directions |
|-----------|-------------|---------------------|
| **Sim-to-Real Gap** | Difficulty transferring from simulation to reality | Domain randomization, system identification, Sim2Real |
| **Sample Efficiency** | Real-world data collection is expensive | Offline RL, world models, teleoperation data |
| **Generalization** | Generalizing across tasks and environments | Foundation models, meta-learning, VLA |
| **Safety** | Safety guarantees for physical interaction | Safe RL, formal verification |
| **Long-horizon Tasks** | Complex multi-step task planning | Hierarchical RL, task planning, LLM planning |

### 7.2 Future Trends

```
┌─────────────────────────────────────────────────────────────────┐
│                     具身智能未来趋势                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🔮 2025-2026                                                    │
│     • VLA（视觉-语言-动作）模型成为标配                           │
│     • 人形机器人进入工厂/服务业实际部署                           │
│     • 数据飞轮：遥操作 → 学习 → 部署 → 更多数据                   │
│     • 机器人基础模型（Foundation Model）成熟                     │
│                                                                 │
│  🔮 2026-2028                                                    │
│     • 通用具身智能基础模型出现                                    │
│     • 低成本机器人平台普及（万元级人形机器人）                    │
│     • 服务机器人进入家庭场景                                      │
│                                                                 │
│  🔮 2028-2030                                                    │
│     • 机器人即服务（RaaS）商业模式成熟                            │
│     • 多机器人协作系统                                            │
│     • 通用人工智能（AGI）的物理载体                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

*(2025-2026: VLA models become standard; humanoids deployed in factories/services; the data flywheel (teleoperation → learning → deployment → more data); robot foundation models mature. 2026-2028: general embodied intelligence foundation models; low-cost platforms spread (humanoids in the ~¥10k class); service robots enter homes. 2028-2030: Robotics-as-a-Service (RaaS) business models mature; multi-robot collaboration; robots as the physical embodiment of AGI)*

### 7.3 Conclusion

> **Physical AI / Embodied AI represents the key step of AI moving from the digital world into the physical world.**
>
> It is not only a convergence of technologies, but a profound understanding of the nature of human intelligence — intelligence arises from the interaction between the body and the world.
>
> Whether your background is AI, robotics, control, or cognitive science, this field has vast space waiting to be explored.
>
> **2025-2026 is the best time to enter embodied intelligence.**

---

## 📖 Further Reading

- [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io)
- [NVIDIA GR00T](https://nvidianews.nvidia.com/)
- [Stanford HumanPlus](https://humanoid-ai.github.io/)
- [Figure AI](https://figure.ai/)
- [HuggingFace LeRobot](https://huggingface.co/lerobot)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

---

## 🤝 Contributing

This document is the first in the Physical-AI-Notes series. You are welcome to participate by:
- Submitting an Issue to report errors
- Opening a PR to add content
- Sharing your learning insights

---

*Last updated: February 2026*
*Author: Dabai (大白)*
*License: MIT*
