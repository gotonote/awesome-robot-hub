# 06 Imitation Learning

Imitation learning acquires skills by mimicking expert behavior — an important paradigm in robot learning. This chapter covers behavior cloning, inverse reinforcement learning, diffusion policies, and other core techniques.

## Contents

- [1. Behavior Cloning](01-behavior-cloning.md)
  - Supervised learning methods
  - Regularization techniques
- [2. Inverse Reinforcement Learning](02-irl.md)
  - Maximum entropy IRL
  - GAIL
- [3. Diffusion Policy](03-diffusion-policy.md)
  - Conditional diffusion models
  - Visuomotor policies
- [4. DAgger](04-dagger.md)
  - Iterative expert aggregation
  - Selective labeling

---

## Core Concepts

### Imitation Learning vs. Reinforcement Learning

| Aspect | RL | Imitation Learning |
|--------|-----|--------------------|
| Supervision | Sparse rewards | Expert demonstrations |
| Sample efficiency | Low | High |
| Exploration | Required | Not required |
| Reward design | Required | Not required |

### Method Comparison

```
┌─────────────────────────────────────────┐
│            模仿学习方法                │
├─────────────────────────────────────────┤
│  BC        │ 直接监督学习，简单        │
│            │ 分布偏移问题              │
├────────────┼──────────────────────────┤
│  IRL       │ 推断奖励函数              │
│            │ 计算复杂                 │
├────────────┼──────────────────────────┤
│  Diffusion │ 多模态策略               │
│            │ 生成式方法               │
├────────────┼──────────────────────────┤
│  DAgger    │ 纠正分布偏移              │
│            │ 需要专家在线             │
└─────────────────────────────────────────┘
```

*(模仿学习方法 = Imitation learning methods, 直接监督学习，简单 = Direct supervised learning, simple, 分布偏移问题 = Distribution shift, 推断奖励函数 = Infer reward function, 计算复杂 = Computationally expensive, 多模态策略 = Multimodal policies, 生成式方法 = Generative methods, 纠正分布偏移 = Corrects distribution shift, 需要专家在线 = Requires online expert)*

---

## Key Papers

1. **DAgger** (2011): Ross et al. - A reduction of imitation learning
2. **GAIL** (2016): Ho & Ermon - Generative Adversarial Imitation Learning
3. **Diffusion Policy** (2023): Chi et al. - Visuomotor Policy Learning via Action Diffusion

---

## Practical Frameworks

### Datasets

| Dataset | Description | Scenario |
|---------|-------------|----------|
| DAML | Robotic manipulation dataset | Grasping, placing |
| RoboNet | Multi-robot dataset | Multi-task |
| Bridge Data | Internet robot data | Cross-domain generalization |

### Training Tips

1. **Data augmentation**: add noise for robustness
2. **Curriculum learning**: from simple to complex
3. **Expert mixing**: BC + RL

---

*This chapter is continuously updated...*
