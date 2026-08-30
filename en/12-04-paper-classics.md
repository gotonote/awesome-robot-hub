# Classic Papers

Foundational papers and technical milestones in the Physical AI / Embodied AI field.

## Contents

- [1. Reinforcement Learning Basics](#1-reinforcement-learning-basics)
- [2. Imitation Learning](#2-imitation-learning)
- [3. Deep Perception & Computer Vision](#3-deep-perception--computer-vision)
- [4. Robot Transformers](#4-robot-transformers)
- [5. Diffusion Models & Generative Methods](#5-diffusion-models--generative-methods)
- [6. Embodied AI Benchmarks](#6-embodied-ai-benchmarks)

---

## 1. Reinforcement Learning Basics

### 1.1 DQN (Deep Q-Network)

| Item | Content |
|------|---------|
| **Paper** | Human-level control through deep reinforcement learning |
| **Authors** | Mnih et al. |
| **Institution** | DeepMind |
| **Year** | 2015 (Nature) |
| **arXiv** | [1312.5602](https://arxiv.org/abs/1312.5602) |

**Core contributions**:
- First combination of deep learning and RL
- Proposed the experience replay mechanism
- Introduced the target network for stable training

**Significance**: opened the era of deep RL, laying the foundation for robot RL.

---

### 1.2 PPO (Proximal Policy Optimization)

| Item | Content |
|------|---------|
| **Paper** | Proximal Policy Optimization Algorithms |
| **Authors** | Schulman et al. |
| **Institution** | OpenAI |
| **Year** | 2017 |
| **arXiv** | [1707.06347](https://arxiv.org/abs/1707.06347) |

**Core contributions**:
- A simplified version of TRPO (trust region policy optimization)
- Introduced the clipping mechanism to prevent drastic policy changes
- Simple implementation, data-efficient, strong generalization

**Significance**: one of the most widely used RL algorithms in robotics today.

---

### 1.3 SAC (Soft Actor-Critic)

| Item | Content |
|------|---------|
| **Paper** | Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor |
| **Authors** | Haarnoja et al. |
| **Institution** | Berkeley BAIR |
| **Year** | 2018 |
| **arXiv** | [1801.01290](https://arxiv.org/abs/1801.01290) |

**Core contributions**:
- Maximum-entropy framework improves exploration and stability
- Off-policy algorithm, data-efficient
- Automatic temperature parameter tuning

**Significance**: performs excellently in continuous control, widely used in robot manipulation.

---

### 1.4 QT-Opt

| Item | Content |
|------|---------|
| **Paper** | QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation |
| **Authors** | Kalashnikov et al. |
| **Institution** | Google DeepMind |
| **Year** | 2018 |
| **arXiv** | [1806.10293](https://arxiv.org/abs/1806.10293) |

**Core contributions**:
- Distributed RL training framework
- Vision-based grasping tasks
- 7 real robots collecting data cooperatively

**Significance**: the first large-scale demonstration of deep RL feasibility on real robots.

---

## 2. Imitation Learning

### 2.1 DAgger

| Item | Content |
|------|---------|
| **Paper** | A Reduction of Imitation Learning to No-Regret Online Learning |
| **Authors** | Ross et al. |
| **Institution** | CMU |
| **Year** | 2011 |
| **arXiv** | - |

**Core contributions**:
- Iterative learning to eliminate compounding errors
- Theoretical basis: no-regret online learning
- Core idea: train on a mix of expert data and the policy's own data

**Significance**: laid the theoretical foundation of imitation learning; the standard upgrade over BC.

---

### 2.2 GAIL (Generative Adversarial Imitation Learning)

| Item | Content |
|------|---------|
| **Paper** | Generative Adversarial Imitation Learning |
| **Authors** | Ho & Ermon |
| **Institution** | Stanford |
| **Year** | 2016 |
| **arXiv** | [1606.03476](https://arxiv.org/abs/1606.03476) |

**Core contributions**:
- Introduced GAN ideas into imitation learning
- A discriminator distinguishes expert trajectories from policy trajectories
- No explicit reward function needed

**Significance**: connected IRL with deep learning, advancing imitation learning.

---

### 2.3 Behavior Cloning (Early Work)

| Item | Content |
|------|---------|
| **Paper** | ALVINN: An Autonomous Land Vehicle In a Neural Network |
| **Authors** | Pomerleau |
| **Institution** | CMU |
| **Year** | 1989 |

**Core contributions**:
- The earliest neural-network imitation learning example
- End-to-end learning of a driving policy

**Significance**: the pioneering work of imitation learning, demonstrating learning from expert demonstrations.

---

## 3. Deep Perception & Computer Vision

### 3.1 ResNet

| Item | Content |
|------|---------|
| **Paper** | Deep Residual Learning for Image Recognition |
| **Authors** | He et al. |
| **Institution** | Microsoft Research |
| **Year** | 2015 |
| **arXiv** | [1512.03385](https://arxiv.org/abs/1512.03385) |

**Core contributions**:
- Residual connections solve the vanishing-gradient problem in deep networks
- 152-layer network on ImageNet
- The fundamental backbone of computer vision

**Significance**: the foundational architecture of visual perception; standard for robot vision.

---

### 3.2 Mask R-CNN

| Item | Content |
|------|---------|
| **Paper** | Mask R-CNN |
| **Authors** | He et al. |
| **Institution** | Facebook AI |
| **Year** | 2017 |
| **arXiv** | [1703.06870](https://arxiv.org/abs/1703.06870) |

**Core contributions**:
- Instance segmentation framework
- RoI Align replaces RoI Pooling
- Simultaneous detection, segmentation, and pose estimation

**Significance**: an important foundation for robot grasping perception.

---

## 4. Robot Transformers

### 4.1 RT-1 (Robotics Transformer)

| Item | Content |
|------|---------|
| **Paper** | RT-1: Robotics Transformer for Real-World Control at Scale |
| **Authors** | Brohan et al. |
| **Institution** | Google DeepMind |
| **Year** | 2022 |
| **arXiv** | [2212.06817](https://arxiv.org/abs/2212.06817) |

**Core contributions**:
- First large-scale vision-language-action model (VLA)
- Trained on 130k robot trajectories
- Demonstrated generalization and scalability

**Significance**: opened the era of robot Transformers.

---

### 4.2 RT-2

| Item | Content |
|------|---------|
| **Paper** | RT-2: Vision-Language-Action Models |
| **Authors** | Brohan et al. |
| **Institution** | Google DeepMind |
| **Year** | 2023 |
| **arXiv** | [2307.15818](https://arxiv.org/abs/2307.15818) |

**Core contributions**:
- A pretrained VLM directly outputs actions
- Symbolic understanding and reasoning transferred to robots
- Significantly improved generalization

**Significance**: demonstrated the value of internet-scale pretraining for robot learning.

---

### 4.3 RT-X

| Item | Content |
|------|---------|
| **Paper** | Open X-Embodiment: Robotic Learning Datasets and RT-X |
| **Authors** | Padalkar et al. |
| **Institution** | Google DeepMind + 30+ institutions |
| **Year** | 2023 |
| **arXiv** | [2310.08864](https://arxiv.org/abs/2310.08864) |

**Core contributions**:
- Integrated data from 30+ institutions, 1M trajectories
- Cross-embodiment generalization
- Open-source datasets + models

**Significance**: a milestone in embodied AI data standardization.

---

### 4.4 PALM-E

| Item | Content |
|------|---------|
| **Paper** | PALM-E: An Embodied Multimodal Language Model |
| **Authors** | Driess et al. |
| **Institution** | Google DeepMind |
| **Year** | 2023 |
| **arXiv** | [2303.03372](https://arxiv.org/abs/2303.03372) |

**Core contributions**:
- 562B parameter foundation model
- Vision encoder + language model + robot state
- Zero-shot reasoning and planning

**Significance**: demonstrated the potential of multimodal foundation models to understand the physical world.

---

### 4.5 VoxPoser

| Item | Content |
|------|---------|
| **Paper** | VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models |
| **Authors** | Zhou et al. |
| **Institution** | Stanford |
| **Year** | 2023 |
| **arXiv** | [2307.05973](https://arxiv.org/abs/2307.05973) |

**Core contributions**:
- LLM generates 3D value maps
- Composable task generalization
- No extra training needed

**Significance**: a new paradigm of LLM + robot planning.

---

## 5. Diffusion Models & Generative Methods

### 5.1 Diffusion Policy

| Item | Content |
|------|---------|
| **Paper** | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion |
| **Authors** | Chi et al. |
| **Institution** | Columbia, Toyota Research |
| **Year** | 2024 (RSS) |
| **arXiv** | [2303.04137](https://arxiv.org/abs/2303.04137) |

**Core contributions**:
- Models actions as a diffusion process
- Significant improvement over LSTM/Transformer policies
- Demonstrates modeling of high-dimensional action spaces

**Significance**: pioneering work applying diffusion models to robot manipulation.

---

### 5.2 ACT (Action Chunking Transformer)

| Item | Content |
|------|---------|
| **Paper** | Learning to Generate Conservative Progressively and Imitate Flexibly |
| **Authors** | Zhao et al. |
| **Institution** | Stanford |
| **Year** | 2024 |
| **GitHub** | [ACT](https://github.com/tonylzq/ALOHA) |

**Core contributions**:
- Action chunking handles temporal dependencies
- Transformer architecture
- Companion paper for ALOHA hardware

**Significance**: a low-cost imitation learning hardware + algorithm solution.

---

### 5.3 RDT (Robust Diffusion Transformer)

| Item | Content |
|------|---------|
| **Paper** | RDT: Robust Diffusion Transformer for Manipulation |
| **Authors** | Lin et al. |
| **Institution** | Tsinghua University, et al. |
| **Year** | 2024 |

**Core contributions**:
- Combines diffusion models with Transformers
- Enhanced generalization and robustness
- Billion-parameter scale model

**Significance**: a foundation-model-scale diffusion policy.

---

## 6. Embodied AI Benchmarks

### 6.1 RLBench

| Item | Content |
|------|---------|
| **Paper** | RLBench: The Robot Learning Benchmark and Learning Environment |
| **Authors** | James et al. |
| **Institution** | Imperial College London |
| **Year** | 2020 |
| **arXiv** | [2006.12983](https://arxiv.org/abs/2006.12983) |

**Core contributions**:
- Standardized robot learning benchmark
- 100+ task definitions
- Unified evaluation protocol

**Significance**: advanced the standardization of robot learning research.

---

### 6.2 Meta-World

| Item | Content |
|------|---------|
| **Paper** | Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning |
| **Authors** | Yu et al. |
| **Institution** | UC Berkeley |
| **Year** | 2019 |
| **arXiv** | [1910.10897](https://arxiv.org/abs/1910.10897) |

**Core contributions**:
- Multi-task RL benchmark
- 50 different robot manipulation tasks
- High task variability

**Significance**: advanced meta-learning and multi-task learning research.

---

### 6.3 SAPIEN

| Item | Content |
|------|---------|
| **Paper** | SAPIEN: A SimulAted Part-based Interactive ENvironment |
| **Authors** | Geng et al. |
| **Institution** | Stanford, NVIDIA |
| **Year** | 2020 |
| **arXiv** | [2003.08515](https://arxiv.org/abs/2003.08515) |

**Core contributions**:
- Physically realistic interactive environment
- Part-based object representation
- Supports large-scale data generation

**Significance**: a high-quality robot simulation environment.

---

## Reading Advice

### Getting-Started Route
1. **RL basics**: DQN → PPO → SAC
2. **Imitation learning**: BC → DAgger → GAIL
3. **Deep vision**: ResNet → Mask R-CNN
4. **Robot Transformers**: RT-1 → RT-2 → RT-X

### Advanced Route
1. **Diffusion policies**: Diffusion Policy
2. **LLM + robots**: VoxPoser, PALM-E
3. **Latest VLAs**: RT-4, OpenVLA, π0

### Must-Read Classics
- **RL**: PPO, SAC
- **Imitation Learning**: DAgger, GAIL
- **Robot Learning**: QT-Opt, RT-X
- **Diffusion**: Diffusion Policy

---

*This chapter is continuously updated — PRs adding more classic papers are welcome.*
