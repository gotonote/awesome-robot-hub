# Classic Paper Interpretations

> This chapter compiles milestone papers in Physical AI / Embodied AI, with interpretations of core ideas and key contributions.

## 1. Reinforcement Learning

### 1.1 PPO (Proximal Policy Optimization)

**Paper**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

**Core idea**: avoid drastic policy changes by clipping the policy update magnitude.

```python
# PPO core loss
def ppo_loss(log_pi, old_log_pi, advantage, clip_eps=0.2):
    """
    PPO clipped objective
    """
    ratio = torch.exp(log_pi - old_log_pi)
    
    # Clip
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    
    # Take the minimum
    loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
    
    return loss.mean()
```

**Key contributions**:
- Clipped surrogate objective
- Simple implementation
- Stable training

---

### 1.2 SAC (Soft Actor-Critic)

**Paper**: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)

**Core idea**: maximum-entropy RL, optimizing both the policy and entropy.

$$\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi}[R(\tau) + \alpha H(\pi(\cdot|s))]$$

```python
# SAC soft value update
def sac_update(q_network, target_q, optimizer, batch, alpha=0.2):
    obs, action, reward, next_obs, done = batch
    
    # Current Q value
    q_value = q_network(obs, action)
    
    # Target Q (using the double-Q trick)
    next_action, log_prob = policy.sample(next_obs)
    target_q = target_q(next_obs, next_action) - alpha * log_prob
    
    # Loss
    q_loss = F.mse_loss(q_value, target_q.detach())
    
    return q_loss
```

**Key contributions**:
- Maximum-entropy framework
- Automatic temperature adjustment
- Stable continuous control

---

## 2. Imitation Learning

### 2.1 DAgger (Dataset Aggregation)

**Paper**: Ross et al., "A Reduction of Imitation Learning to No-Regret Online Learning" (2011)

**Core idea**: iteratively aggregate expert data to solve distribution shift.

```
┌─────────────────────────────────────────────────────────┐
│                    DAgger 算法流程                       │
│                                                         │
│   1. 收集专家演示 D = {(o_t, a_t^expert)}              │
│   2. 训练策略 π_θ(a|o) from D                          │
│   3. 运行 π_θ，收集轨迹 {(o_t, a_t)}                   │
│   4. 询问专家获取正确动作 a_t^expert                     │
│   5. 聚合数据 D = D ∪ {(o_t, a_t^expert)}              │
│   6. 重复 2-5                                           │
└─────────────────────────────────────────────────────────┘
```

*(1. Collect expert demonstrations D = {(o_t, a_t^expert)}. 2. Train policy π_θ(a|o) from D. 3. Run π_θ and collect trajectories {(o_t, a_t)}. 4. Query the expert for correct actions a_t^expert. 5. Aggregate data D = D ∪ {(o_t, a_t^expert)}. 6. Repeat steps 2-5.)*

**Key contributions**:
- Addresses causal confusion
- Theoretical guarantees
- Simple and effective

---

### 2.2 GAIL (Generative Adversarial Imitation Learning)

**Paper**: Ho & Ermon, "Generative Adversarial Imitation Learning" (2016)

**Core idea**: learn a policy with a GAN framework; the discriminator distinguishes expert data from policy-generated data.

```python
class GAIL:
    def __init__(self, policy, discriminator):
        self.policy = policy
        self.discriminator = discriminator
        
    def discriminator_loss(self, expert_obs, expert_action, policy_obs, policy_action):
        """
        Discriminator loss
        """
        expert_pairs = torch.cat([expert_obs, expert_action], dim=-1)
        policy_pairs = torch.cat([policy_obs, policy_action], dim=-1)
        
        expert_logits = self.discriminator(expert_pairs)
        policy_logits = self.discriminator(policy_pairs)
        
        loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits)
        ) + F.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros_like(policy_logits)
        )
        return loss
```

**Key contributions**:
- End-to-end learning
- No explicit reward function needed
- Learns from small amounts of expert data

---

## 3. World Models

### 3.1 World Models

**Paper**: Ha & Schmidhuber, "World Models" (2018)

**Architecture**: VAE + MDN-RNN

```
Observation image ──(VAE)──> latent vector z
                              │
                              v
              (RNN sequence processing) predict next latent state
                              │
                              v
                     latent dynamics learning
```

**Key contributions**:
- First to learn world models in a compressed latent space
- Demonstrated the "fast weights" idea
- Laid the foundation for later work such as Dreamer

---

### 3.2 Dreamer

**Paper**: Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination" (2020)

**Core innovations**:
- Variational autoencoder + RSSM
- Imagination rollout
- Actor-critic architecture

```python
# Dreamer core loop
def dreamer_update(model, batch):
    # 1. World model learning
    obs_emb = model.encoder(obs)
    post = model.posterior(obs_emb)
    prior = model.prior(action)
    dyn_loss = kl_loss(post, prior)
    rec_loss = mse_loss(model.decoder(post), obs)
    
    # 2. Imagination rollout
    imagined = model.imagine(post, actions, horizon)
    
    # 3. Policy learning
    policy_loss = -value(imagined).mean()
    
    return dyn_loss + rec_loss + policy_loss
```

---

## 4. Multimodal Foundation Models

### 4.1 RT-2 (Robotics Transformer 2)

**Paper**: Brohan et al., "RT-2: Vision-Language-Action Models" (2023)

**Core idea**: a vision-language model directly outputs robot actions.

```
┌─────────────────────────────────────────────────────────┐
│                      RT-2 架构                           │
│                                                         │
│   输入:                                                  │
│   ┌─────────┐    ┌─────────┐                          │
│   │ 图像    │ +  │ 文本指令│                          │
│   └─────────┘    └─────────┘                          │
│        │               │                               │
│        v               v                               │
│   ┌─────────────────────────────┐                       │
│   │    VLA (Vision-Language-Action)                   │
│   │    大模型微调               │                       │
│   └─────────────────────────────┘                       │
│               │                                          │
│               v                                          │
│   ┌─────────────────────────────┐                       │
│   │    动作 tokens (末端+关节)   │                       │
│   └─────────────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

*(Input: image + text instruction → VLA (vision-language-action) foundation model fine-tuning → action tokens (end-effector + joints).)*

**Key contributions**:
- Large improvement in generalization
- Semantic reasoning ability
- Zero-shot transfer

---

### 4.2 PaLM-E

**Paper**: Driess et al., "PaLM-E: An Embodied Multimodal Language Model" (2023)

**Core innovation**: multimodal embeddings enter the language model.

---

## 5. Diffusion Policies

### 5.1 Diffusion Policy

**Paper**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (2023)

**Core formula**:

$$a_0 = \text{Denoise}(o_t, \epsilon_\theta)$$

```python
# Diffusion policy sampling
@torch.no_grad()
def get_action_diffusion(policy, obs, num_steps=10):
    action = torch.randn_like(action_dim)
    
    for t in reversed(range(num_steps)):
        noise_pred = policy(obs, action, t)
        action = (action - noise_pred) / np.sqrt(1 - alpha[t])
        
    return action
```

**Key contributions**:
- High-quality action generation
- Models complex multimodal distributions
- Stable training

---

## 6. Simulation & Data Collection

### 6.1 SIMPLER

**Paper**: "SIMPLER: Single-Image Policy Learning with RGB Cameras and Manipulators" (2023)

---

### 6.2 MT-OPT

**Paper**: "MT-OPT: Multi-Task Optical Perception" (2022)

---

## 7. Paper-Reading Advice

### 7.1 Quick Reading Structure

```
1. Abstract - core contribution
2. Introduction - problem definition + method overview
3. Method - core technical details
4. Experiment - effect validation
5. Conclusion - summary
```

### 7.2 Code Reproduction Path

```python
# Recommended reproduction order
paper_reading_order = [
    "PPO",          # RL basics
    "SAC",          # continuous control
    "DAgger",      # imitation learning basics
    "GAIL",        # adversarial imitation learning
    "World Models", # world model basics
    "Dreamer",     # advanced world models
    "Diffusion Policy", # diffusion policies
    "RT-2",        # VLA
]
```

---

## 8. Summary Table

| Category | Paper | Year | Core Contribution |
|----------|-------|------|-------------------|
| RL | PPO | 2017 | Clipped policy updates |
| RL | SAC | 2018 | Maximum-entropy RL |
| Imitation | DAgger | 2011 | Dataset aggregation |
| Imitation | GAIL | 2016 | Adversarial learning |
| World models | World Models | 2018 | VAE + RNN |
| World models | Dreamer | 2020 | Latent imagination |
| VLA | RT-2 | 2023 | Vision-language-action |
| Diffusion | Diffusion Policy | 2023 | Action generation |

---

*More papers continuously updated...*
