# World Models Overview

> This chapter introduces the basic concepts of world models — a core component for building intelligent decision-making in Physical AI systems.

## 1. What Is a World Model?

### 1.1 Definition

A **World Model** is an agent's internal representation and prediction model of the external environment.

```
┌─────────────────────────────────────────────────────────────┐
│                     世界模型在智能系统中的位置                │
│                                                             │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐           │
│   │  感知   │  ->  │ 世界模型 │  ->  │   决策   │           │
│   │ Perception│     │World Model│     │  Policy  │           │
│   └─────────┘      └─────────┘      └─────────┘           │
│        │                 │                 │              │
│        v                 v                 v              │
│     观测 o           状态 s           动作 a              │
│                   预测 s'                               │
└─────────────────────────────────────────────────────────────┘
```

*(Perception → World Model → Policy. Observation o → state s (predicting s') → action a.)*

### 1.2 Core Functions of a World Model

| Function | Description | Example |
|----------|-------------|---------|
| State representation | Encode perceptual information into compact states | CNN encoding observations |
| Transition prediction | Predict the next state | $s_{t+1} = f(s_t, a_t)$ |
| Reward prediction | Predict the immediate reward | $r_t = g(s_t, a_t)$ |
| Imagination | Simulate the future in latent space | Imagined sequence rollout |

## 2. History of World Models

### 2.1 Classical World Models

- **POMDP**: Partially Observable Markov Decision Process
- **MDP**: Markov Decision Process
- **Kalman filter**: state estimation for linear systems

### 2.2 Deep World Models

| Year | Model | Contribution |
|------|-------|--------------|
| 2018 | World Models | First world model built with VAE + RNN |
| 2019 | PlaNet | Model-based reinforcement learning |
| 2020 | Dreamer | Learning in the imagination space |
| 2022 | VI-Projects | Visual prediction |

## 3. Mathematical Foundations

### 3.1 Markov Decision Process (MDP)

An MDP is defined by the 5-tuple $(S, A, P, R, \gamma)$:

- $S$: state space
- $A$: action space
- $P(s'|s, a)$: transition probability
- $R(s, a)$: reward function
- $\gamma$: discount factor

### 3.2 Probabilistic Interpretation of World Models

```
p(s_{1:T}, a_{1:T}, r_{1:T}) = p(s_1) ∏ p(s_t|s_{t-1}, a_{t-1}) p(a_t|s_{1:t-1}) p(r_t|s_t, a_t)
```

A world model learns:
- **Transition model**: $p(s_t|s_{t-1}, a_{t-1})$
- **Reward model**: $p(r_t|s_t, a_t)$

### 3.3 Latent-Space World Models

```
观测 o_t ──(编码器)──> 隐状态 z_t
                        │
           ┌────────────┼────────────┐
           │            │            │
      (转移模型)    (奖励模型)    (表征模型)
           │            │            │
           v            v            v
        z_{t+1}       r_t         重建 o_t
```

*(Observation o_t → (encoder) → latent state z_t; transition model → z_{t+1}; reward model → r_t; representation model → reconstruct o_t.)*

## 4. World Model Architectures

### 4.1 Classic Architecture: World Models (2018)

```python
import torch
import torch.nn as nn

class WorldModel(nn.Module):
    """
    Classic World Models architecture
    VAE (observation) + MDN-RNN (latent dynamics)
    """
    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.z_dim = z_dim
        
        # VAE encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * 2)  # mu, logvar
        )
        
        # VAE decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # MDN-RNN (Mixture Density Network)
        self.rnn = nn.LSTM(z_dim + action_dim, hidden_dim, num_layers=2)
        self.mdn = nn.Linear(hidden_dim, z_dim * 3)  # mixture components
        
    def forward(self, obs, action, hidden=None):
        # Encode the observation
        z_mu, z_logvar = torch.chunk(self.encoder(obs), 2, dim=-1)
        z = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_logvar)
        
        # RNN sequence processing
        rnn_input = torch.cat([z, action], dim=-1)
        rnn_out, hidden = self.rnn(rnn_input, hidden)
        
        # Predict the next latent state
        z_next_params = self.mdn(rnn_out)
        
        return z_next_params, z_mu, z_logvar, hidden
```

### 4.2 The Dreamer Architecture

```python
class DreamerWorldModel(nn.Module):
    """
    Dreamer's world model
    Includes: encoder, transition model, reward model, value model
    """
    def __init__(self, obs_channels, action_dim, latent_dim=30, hidden=200):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden),
            nn.ReLU()
        )
        
        # Latent prior/posterior
        self.latent_embed = nn.Linear(hidden, latent_dim * 2)
        
        # Transition model (RSSM)
        self.trans_deter = nn.GRU(latent_dim + action_dim, hidden)
        self.trans_stoch = nn.Linear(hidden, latent_dim * 2)
        
        # Reward model
        self.reward_head = nn.Sequential(
            nn.Linear(hidden + latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_channels * 64 * 64)
        )
        
    def imagine(self, latent, action, horizon):
        """
        Imagination rollout
        """
        imagined_trajs = []
        hidden = None
        
        for _ in range(horizon):
            deter = self.trans_deter(torch.cat([latent, action], dim=-1))
            stoch_params = self.trans_stoch(deter)
            stoch = stoch_params[:, :self.latent_dim]
            
            imagined_trajs.append(stoch)
            latent = stoch.detach()
            
        return torch.stack(imagined_trajs, dim=1)
```

## 5. Training World Models

### 5.1 Training Objective

$$\mathcal{L}_{world} = \mathcal{L}_{recon} + \mathcal{L}_{KL} + \mathcal{L}_{reward}$$

```python
def compute_world_model_loss(model, batch):
    """
    Compute the world model loss
    """
    obs, action, reward, next_obs = batch
    
    # Encode the current observation
    h = model.encoder(obs)
    z_params = model.latent_embed(h)
    z = z_params[:, :model.latent_dim]
    
    # Transition prediction
    deter = model.trans_deter(torch.cat([z, action], dim=-1))
    z_next_params = model.trans_stoch(deter)
    z_next = z_next_params[:, :model.latent_dim]
    
    # Reconstruction
    recon = model.decoder(torch.cat([z, deter], dim=-1))
    
    # Reward prediction
    pred_reward = model.reward_head(torch.cat([z, deter], dim=-1))
    
    # Losses
    recon_loss = F.mse_loss(recon, obs)
    reward_loss = F.mse_loss(pred_reward, reward)
    kl_loss = kl_divergence(z, z_next)
    
    total_loss = recon_loss + reward_loss + 0.1 * kl_loss
    
    return total_loss
```

### 5.2 Model-Based RL (MBRL)

```
┌─────────────────────────────────────────────────────────┐
│              MBRL (Model-Based RL) 流程                  │
│                                                         │
│   1. 收集数据    2. 学习世界模型   3. 想象 rollout      │
│   ─────────>   ──────────────>   ──────────────>       │
│   真实环境         模型训练          策略优化            │
│                                                         │
│   4. 执行策略    5. 评估            6. 返回步骤1         │
│   ─────────>   ──────────────>   ──────────────>       │
└─────────────────────────────────────────────────────────┘
```

*(1. collect data (real environment) → 2. learn the world model (model training) → 3. imagination rollout (policy optimization) → 4. execute the policy → 5. evaluate → 6. return to step 1)*

## 6. Applications of World Models

### 6.1 Applications in Physical AI

| Application | Description |
|-------------|-------------|
| Robot control | Learn robot-environment interaction |
| Action prediction | Predict action execution outcomes |
| Planning | Plan in latent space |
| Zero-shot generalization | Imagine unseen scenarios |

### 6.2 Example Application

```python
class RobotWorldModel:
    """
    Robot world model example
    Used to predict the outcomes of robot actions
    """
    def __init__(self, state_dim, action_dim):
        self.model = build_world_model(state_dim, action_dim)
        
    def predict_next_state(self, state, action):
        """
        Predict the next state
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_tensor = torch.FloatTensor(action)
            next_state = self.model.predict(state_tensor, action_tensor)
        return next_state.numpy()
    
    def plan(self, start_state, goal_state, max_steps=100):
        """
        Plan in latent space
        """
        # Simplified: random sampling planning
        best_plan = None
        best_cost = float('inf')
        
        for _ in range(100):
            plan = self.random_plan(start_state, max_steps)
            cost = self.estimate_cost(plan, goal_state)
            if cost < best_cost:
                best_cost = cost
                best_plan = plan
                
        return best_plan
```

## 7. Summary & Outlook

```
┌────────────────────────────────────────────────────────┐
│                    世界模型核心要点                       │
├────────────────────────────────────────────────────────┤
│  ✓ 状态表征: 将高维感知映射到低维隐空间                   │
│  ✓ 转移预测: 学习环境动态 p(s'|s,a)                     │
│  ✓ 奖励预测: 学习奖励信号 p(r|s,a)                       │
│  ✓ 想象推演: 在隐空间进行规划与决策                       │
├────────────────────────────────────────────────────────┤
│  挑战:                                                   │
│  - 组合复杂度                                            │
│  - 稀疏奖励                                              │
│  - 部分可观测                                            │
│  - 泛化能力                                              │
└────────────────────────────────────────────────────────┘
```

*(Core points: state representation (map high-dimensional perception to low-dimensional latent space); transition prediction (learn environment dynamics p(s'|s,a)); reward prediction (learn reward signal p(r|s,a)); imagination (plan and decide in latent space). Challenges: combinatorial complexity; sparse rewards; partial observability; generalization.)*

## 8. Further Reading

- Ha & Schmidhuber (2018). "World Models"
- Hafner et al. (2019). "Learning Latent Dynamics for Planning"
- Hafner et al. (2020). "Dream to Control: Learning Behaviors by Latent Imagination"

---

*The next chapter introduces advanced neural world model content and the latest progress.*
