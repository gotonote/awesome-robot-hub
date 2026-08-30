# Diffusion Policy In-Depth

> This chapter provides an in-depth introduction to Diffusion Policy — a generative robot action policy based on diffusion models.

## 1. Background & Motivation

### 1.1 Why Diffusion Policies?

Traditional policies have the following limitations:
- **High-dimensional action spaces**: complex tasks require high-dimensional continuous control
- **Multimodal behaviors**: human demonstrations contain multiple valid solutions
- **Temporal correlations**: action sequences need to remain consistent

**Advantages of diffusion models**:
- ✓ Naturally model multimodal distributions
- ✓ No explicit distribution-form assumptions
- ✓ Progressive denoising generates high-quality samples

### 1.2 Diffusion Policy Overview

**Diffusion Policy** models the robot policy as a conditional diffusion process:

```
Observation o_t ──> diffusion policy π(a_t|o_t) ──> action a_t
                         ↑
                    DDIM/DDPM sampling
```

## 2. Diffusion Model Basics

### 2.1 The Forward Diffusion Process

Gradually add noise to the data until it becomes standard Gaussian noise:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

where $\beta_t$ is the noise schedule parameter.

```python
import torch
import numpy as np

def add_noise(x, t, noise_schedule='linear'):
    """
    Forward diffusion process
    x: original data
    t: time step
    """
    if noise_schedule == 'linear':
        betas = torch.linspace(0.0001, 0.02, 1000)
    elif noise_schedule == 'cosine':
        # Cosine schedule
        s = 0.008
        steps = 1000
        x = torch.linspace(0, steps, steps)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_alpha_prod = alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]) ** 0.5
    
    noise = torch.randn_like(x)
    x_noisy = sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise
    
    return x_noisy, noise
```

### 2.2 The Reverse Process

Gradually recover data from noise:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_t)$$

### 2.3 Training Objective

Simplified denoising objective:

$$\mathcal{L} = \mathbb{E}_{x, t, \epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||^2]$$

```python
class DiffusionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, T=100):
        super().__init__()
        self.T = T
        self.action_dim = action_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Denoising network (U-Net style)
        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim * 2 + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs, noisy_action, t):
        """
        Predict the noise
        """
        # Time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1))
        
        # Observation encoding
        obs_emb = self.obs_encoder(obs)
        
        # Concatenate and predict noise
        x = torch.cat([obs_emb, t_emb, noisy_action], dim=-1)
        noise_pred = self.denoiser(x)
        
        return noise_pred
    
    def training_step(self, obs, action):
        """
        Training step
        """
        batch_size = obs.shape[0]
        t = torch.randint(0, self.T, (batch_size,))
        
        # Add noise
        noise = torch.randn_like(action)
        noisy_action = self.noise_schedule.add_noise(action, t)
        
        # Predict noise
        noise_pred = self.forward(obs, noisy_action, t)
        
        loss = F.mse_loss(noise_pred, noise)
        return loss
```

## 3. Diffusion Policy Architecture

### 3.1 Conditional Diffusion Policies

```python
class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon, T=100, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.T = T
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action sequence denoising network
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + 1, hidden_dim),  # obs + action + t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Noise schedule
        self.register_buffer('betas', self.get_noise_schedule())
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def get_noise_schedule(self):
        return torch.linspace(0.0001, 0.02, self.T)
    
    def forward(self, obs, action_samples, t):
        """
        Single-step denoising
        obs: (B, obs_dim)
        action_samples: (B, action_dim)
        t: (B,)
        """
        B = obs.shape[0]
        
        obs_emb = self.obs_encoder(obs)
        t_normalized = t.float() / self.T
        
        x = torch.cat([obs_emb, action_samples, t_normalized.unsqueeze(-1)], dim=-1)
        noise_pred = self.action_net(x)
        
        return noise_pred
    
    @torch.no_grad()
    def get_action(self, obs, num_samples=10, num_steps=10):
        """
        Generate an action from noise
        DDIM sampling (more efficient)
        """
        B = obs.shape[0]
        
        # Start from random noise
        action = torch.randn(B, self.action_dim, device=obs.device)
        
        # Step schedule
        step_indices = torch.linspace(0, self.T-1, num_steps, dtype=torch.long)
        
        for i, t_idx in enumerate(step_indices):
            t = torch.full((B,), t_idx, device=obs.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.forward(obs, action, t)
            
            # Sampling step (simplified)
            alpha = self.alphas_cumprod[t_idx]
            alpha_prev = self.alphas_cumprod[max(t_idx-1, 0)]
            
            # Update the action
            action = (action - (1-alpha).sqrt() * noise_pred) / alpha.sqrt()
            
            if i < num_steps - 1:
                action += torch.randn_like(action) * ((1-alpha_prev)/(1-alpha)).sqrt()
                
        return action
```

### 3.2 Temporal Diffusion Policies

```python
class TemporalDiffusionPolicy(nn.Module):
    """
    Temporal diffusion policy - generates action sequences
    """
    def __init__(self, obs_dim, action_dim, horizon, T=100):
        super().__init__()
        self.horizon = horizon
        self.T = T
        
        # Transformer encoder for observation sequences
        self.obs_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=obs_dim, nhead=4, dim_feedforward=256),
            num_layers=3
        )
        
        # Transformer decoder for action sequences
        self.action_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=action_dim, nhead=4, dim_feedforward=256),
            num_layers=3
        )
        
        # Time step MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 256)
        )
        
    def forward(self, obs_seq, noisy_action_seq, t):
        """
        Observation sequence -> action sequence denoising
        """
        # Encode observations
        obs_emb = self.obs_encoder(obs_seq)
        
        # Time embedding
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        
        # Cross-attention to generate actions
        action_emb = self.action_decoder(
            noisy_action_seq,
            obs_emb + t_emb.unsqueeze(0)
        )
        
        return action_emb
```

## 4. Training Diffusion Policies

### 4.1 Imitation Learning Training

```python
def train_diffusion_policy(policy, dataset, epochs=100, batch_size=64, lr=1e-4):
    """
    Train a diffusion policy with behavior cloning
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    dataset = TensorDataset(dataset['obs'], dataset['action'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for obs, action in dataloader:
            optimizer.zero_grad()
            
            # Random time step
            t = torch.randint(0, policy.T, (obs.shape[0],))
            
            # Add noise
            noise = torch.randn_like(action)
            noisy_action = noise_schedule.add_noise(action, t)
            
            # Predict noise
            noise_pred = policy(obs, noisy_action, t)
            
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
```

### 4.2 Loss Function Design

| Loss Component | Formula | Role |
|----------------|---------|------|
| Denoising loss | $\|\epsilon - \epsilon_\theta\|^2$ | Core reconstruction |
| Action smoothing | $\|\nabla a_t\|^2$ | Action smoothness |
| Return prediction | $(R - R_\theta)^2$ | Policy improvement |

## 5. Experiments & Applications

### 5.1 Robot Control Application

```python
class RobotDiffusionController:
    """
    Robot diffusion policy controller
    """
    def __init__(self, policy):
        self.policy = policy
        self.obs_history = []
        
    def reset(self):
        self.obs_history = []
        
    def step(self, obs):
        """
        Execute one step
        """
        # Record the observation
        self.obs_history.append(obs)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # Get the action
        with torch.no_grad():
            action = self.policy.get_action(obs_tensor)
            
        return action.numpy()[0]
    
    def rollout(self, env, max_steps=200):
        """
        Full rollout
        """
        obs = env.reset()
        self.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = self.step(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
                
        return total_reward
```

### 5.2 Experimental Comparison

| Method | Success Rate | Sample Efficiency | Multimodal |
|--------|--------------|-------------------|------------|
| BC | 75% | High | Poor |
| GAIL | 82% | Medium | Medium |
| Diffusion Policy | **90%** | High | **Good** |

## 6. Advanced Techniques

### 6.1 Classifier-Free Guidance

```python
@torch.no_grad()
def classifier_free_guidance(policy, obs, action, guidance_scale=1.0):
    """
    Classifier-free guidance
    """
    # Conditional prediction
    noise_cond = policy(obs, action, t)
    
    # Unconditional prediction (zero observation)
    obs_zero = torch.zeros_like(obs)
    noise_uncond = policy(obs_zero, action, t)
    
    # Guidance
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    
    return noise_pred
```

### 6.2 EMA

```python
class EMAModel:
    """
    Exponential moving average
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_avg.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
```

## 7. Summary

```
┌─────────────────────────────────────────────────────────┐
│                  Diffusion Policy 要点                   │
├─────────────────────────────────────────────────────────┤
│  ✓ 条件扩散: 观测 -> 动作 条件生成                        │
│  ✓ DDPM/DDIM: 两种采样策略                               │
│  ✓ 时序建模: 可生成动作序列                               │
│  ✓ 多模态: 自然建模复杂动作分布                           │
├─────────────────────────────────────────────────────────┤
│  优势:                                                   │
│  - 表达能力强                                            │
│  - 训练稳定                                              │
│  - 推理可调                                              │
├─────────────────────────────────────────────────────────┤
│  挑战:                                                   │
│  - 推理速度 (需多步去噪)                                  │
│  - 计算资源                                              │
└─────────────────────────────────────────────────────────┘
```

*(Conditional diffusion: observation → action conditional generation; DDPM/DDIM: two sampling strategies; temporal modeling: can generate action sequences; multimodal: naturally models complex action distributions. Advantages: strong expressiveness, stable training, adjustable inference. Challenges: inference speed (requires multi-step denoising); compute resources.)*

## 8. Further Reading

- Chi et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
- Haarnoja et al. (2023). "BC-IRL: Maximum Entropy IRL with Diffusion"
- Janner et al. (2022). "Planning with Diffusion for Flexible Behavior"

---

*The next chapter introduces milestone interpretations of classic papers.*
