# 05 Reinforcement Learning

Reinforcement Learning (RL) is a core technology of Physical AI, enabling agents to learn optimal policies through interaction with their environment. This chapter covers RL theory, algorithm implementations, and robotic applications.

## Contents

- [1. RL Basics](05-rl-basics.md)
  - Markov Decision Processes
  - Value functions & policy gradients
  - DQN and its variants
- [2. Offline Reinforcement Learning](05-offline-rl.md)
  - Distribution shift problem
  - CQL algorithm
  - Practical frameworks
- [3. Sim-to-Real Transfer](05-sim2real.md)
  - Domain randomization
  - Domain adaptation
  - Curriculum learning
- [4. Distributed Reinforcement Learning](05-distributed-rl.md)
  - Distributed architectures
  - IMPALA / Ape-X
  - Parameter synchronization

---

## Core Concepts

### The RL Framework

```
┌─────────────────────────────────────────┐
│           强化学习交互框架              │
├─────────────────────────────────────────┤
│                                         │
│    ┌───────┐         ┌───────┐         │
│    │ 智能体 │────────▶│  环境  │         │
│    │ Agent │◀────────│Env    │         │
│    └───────┘         └───────┘         │
│       │                  │              │
│       │ Action (a)       │              │
│       │─────────────────▶│              │
│       │                  │              │
│       │◀─────────────────│              │
│       │   Reward (r)     │              │
│       │   State (s')     │              │
│                                         │
└─────────────────────────────────────────┘
```

*(智能体 = Agent, 环境 = Environment)*

### Key Algorithm Milestones

| Year | Algorithm | Contribution |
|------|-----------|--------------|
| 2013 | DQN | Breakthrough in deep RL |
| 2016 | A3C | Asynchronous distributed training |
| 2017 | PPO | Stable policy optimization |
| 2018 | IMPALA | Scalable distributed RL |
| 2019 | AlphaStar | Superhuman performance |
| 2020 | CQL | Offline reinforcement learning |

---

## Learning Path

### Beginner
1. Understand MDP basics
2. Master Q-learning and SARSA
3. Learn Deep Q-Networks (DQN)

### Intermediate
4. Policy gradient methods (A2C / PPO)
5. Offline reinforcement learning
6. Sim-to-Real transfer techniques

### Advanced
7. Distributed reinforcement learning
8. Meta-learning and multi-task learning
9. Large-scale robotic applications

---

## Practical Frameworks

### Simulation Environments

| Environment | Features | Best For |
|-------------|----------|----------|
| MuJoCo | High-fidelity physics | Continuous control |
| PyBullet | Open-source, easy | Rapid prototyping |
| Isaac Sim | NVIDIA GPU-accelerated | Large-scale training |
| Gazebo | ROS integration | Outdoor robots |

### Training Libraries

- **Stable-Baselines3**: simple, easy-to-use RL library
- **Ray RLlib**: large-scale distributed training
- **OpenAI Baselines**: classic algorithm implementations

---

## Key Resources

### Classic Papers
1. Mnih et al. (2015). Human-level control through deep RL
2. Schulman et al. (2017). Proximal Policy Optimization
3. Haarnoja et al. (2018). Soft Actor-Critic
4. Kumar et al. (2020). Conservative Q-Learning

### Open-Source Projects
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- [DeepMind Control Suite](https://github.com/deepmind/dm_control)

---

*This chapter is continuously updated...*
