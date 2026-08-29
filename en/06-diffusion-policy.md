# Diffusion Policy

Diffusion policy models robot policies as conditional diffusion processes, effectively handling multimodal action distributions with strong performance in robot control tasks.

## Contents

- [1. Diffusion Model Basics](#1-diffusion-model-basics)
- [2. Diffusion Policy Architecture](#2-diffusion-policy-architecture)
- [3. Training & Inference](#3-training--inference)
- [4. Visuomotor Diffusion Policy](#4-visuomotor-diffusion-policy)
- [5. Practical Implementation](#5-practical-implementation)

---

## 1. Diffusion Model Basics

### 1.1 The Forward Process

The forward process gradually adds noise to the data:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

After T steps, $x_T$ approximates standard Gaussian noise:

$$
x_T \approx \mathcal{N}(0, I)
$$

### 1.2 The Reverse Process

The reverse process gradually recovers data from noise:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

### 1.3 Simplified Training Objective

The training objective simplifies to noise prediction:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \| \epsilon - \epsilon_\theta(x_t, t) \|^2
$$

---

## 2. Diffusion Policy Architecture

### 2.1 Conditional Diffusion Models

```python
import torch
import torch.nn as nn
import numpy as np

class ConditionalDiffusion(nn.Module):
    """
    Conditional diffusion model
    Used to learn a conditional policy p(a|o)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_steps=100):
        super(ConditionalDiffusion, self).__init__()
        
        self.num_steps = num_steps
        self.action_dim = action_dim
        
        # Time embedding layer
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Noise prediction network
        # Input: noise, time step, condition (observation)
        self.noise_pred_net = nn.Sequential(
            nn.Linear(action_dim + state_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, noisy_action, t, observation):
        """
        Predict the noise.
        noisy_action: noised action
        t: time step
        observation: state/observation
        """
        # Time embedding
        t_embed = self.time_embed(t.float().unsqueeze(-1))
        
        # Concatenate inputs
        x = torch.cat([noisy_action, observation, t_embed], dim=-1)
        
        # Predict noise
        predicted_noise = self.noise_pred_net(x)
        
        return predicted_noise
    
    def get_noise_schedule(self):
        """Noise schedule"""
        # Linear schedule
        betas = torch.linspace(0.0001, 0.02, self.num_steps)
        return betas
```

### 2.2 Diffusion Policy

```python
class DiffusionPolicy(nn.Module):
    """
    Diffusion policy
    Uses a conditional diffusion model as the robot policy
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 num_diffusion_steps=100, horizon=1):
        super(DiffusionPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon  # prediction horizon
        
        # Observation encoder
        self.observation_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Diffusion model
        self.diffusion = ConditionalDiffusion(
            state_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_steps=num_diffusion_steps
        )
        
    def forward(self, observation, training=True):
        """
        Forward pass.
        observation: (B, state_dim)
        """
        # Encode the observation
        obs_embed = self.observation_encoder(observation)
        
        if training:
            # Training: predict noise
            batch_size = observation.shape[0]
            
            # Random time step
            t = torch.randint(0, self.diffusion.num_steps, (batch_size,))
            
            # Random noise
            noise = torch.randn(batch_size, self.action_dim).to(observation.device)
            
            # Noised action
            noisy_action = self.add_noise(observation, noise, t)
            
            # Predict noise
            predicted_noise = self.diffusion(noisy_action, t, obs_embed)
            
            return predicted_noise, noise
        else:
            # Inference: DDIM sampling
            action = self.sample(observation)
            return action
    
    def add_noise(self, observation, noise, t):
        """Add noise"""
        betas = self.diffusion.get_noise_schedule().to(observation.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Alpha at step t
        sqrt_alpha_prod = alphas_cumprod[t].sqrt().view(-1, 1)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]).sqrt().view(-1, 1)
        
        noisy_action = sqrt_alpha_prod * noise + sqrt_one_minus_alpha_prod * noise
        return noisy_action
    
    @torch.no_grad()
    def sample(self, observation, num_steps=None):
        """Sample to generate an action"""
        if num_steps is None:
            num_steps = self.diffusion.num_steps
            
        batch_size = observation.shape[0]
        device = observation.device
        
        # Encode the observation
        obs_embed = self.observation_encoder(observation)
        
        # Start from random noise
        action = torch.randn(batch_size, self.action_dim).to(device)
        
        # Noise schedule
        betas = self.diffusion.get_noise_schedule().to(device)
        
        # Gradual denoising
        for t in reversed(range(num_steps)):
            t_tensor = torch.ones(batch_size).to(device) * t
            
            # Predict noise
            predicted_noise = self.diffusion(action, t_tensor, obs_embed)
            
            # Denoising step
            if t > 0:
                noise = torch.randn_like(action)
            else:
                noise = torch.zeros_like(action)
            
            beta = betas[t]
            sqrt_one_minus_alpha = (1 - beta).sqrt()
            sqrt_alpha = beta.sqrt()
            
            action = (action - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha
            action = action + sqrt_alpha * noise
            
        return action
```

---

## 3. Training & Inference

### 3.1 Training Loop

```python
class DiffusionPolicyTrainer:
    """
    Diffusion policy trainer
    """
    def __init__(self, policy, lr=3e-4):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
    def train_step(self, batch):
        """
        Training step.
        
        batch: {
            'observations': (B, T, state_dim),
            'actions': (B, T, action_dim)
        }
        """
        observations = batch['observations']
        actions = batch['actions']
        
        # Take the last observation as the condition
        obs_cond = observations[:, -1, :]
        
        # Forward pass
        predicted_noise, target_noise = self.policy(obs_cond, training=True)
        
        # Loss
        loss = nn.MSELoss()(predicted_noise, target_noise)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, num_epochs=100):
        """Full training"""
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in dataloader:
                loss = self.train_step(batch)
                total_loss += loss
                
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
```

### 3.2 Inference Optimization

```python
class DDIMScheduler:
    """
    DDIM sampling acceleration
    Generate high-quality actions with fewer steps
    """
    def __init__(self, num_steps=50, eta=0.0):
        self.num_steps = num_steps
        self.eta = eta
        
    def sample(self, policy, observation):
        """DDIM sampling"""
        batch_size = observation.shape[0]
        device = observation.device
        
        # Observation encoding
        obs_embed = policy.observation_encoder(observation)
        
        # Initial noise
        action = torch.randn(batch_size, policy.action_dim).to(device)
        
        # Jump steps (sample every step_size steps)
        step_size = 100 // self.num_steps
        
        for i in reversed(range(0, 100, step_size)):
            t = torch.ones(batch_size).to(device) * i
            
            # Predict noise
            predicted_noise = policy.diffusion(action, t, obs_embed)
            
            # DDIM update
            if i > 0:
                # Random part
                noise = torch.randn_like(action)
            else:
                noise = torch.zeros_like(action)
            
            # Simplified DDIM update
            action = action - predicted_noise * 0.1 + noise * 0.01
            
        return action
```

---

## 4. Visuomotor Diffusion Policy

### 4.1 Visual Encoder

```python
class VisualEncoder(nn.Module):
    """
    Visual encoder
    Converts images to feature vectors
    """
    def __init__(self, embed_dim=256):
        super(VisualEncoder, self).__init__()
        
        # Use a pretrained ResNet as the backbone
        import torchvision.models as models
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection layer
        self.projector = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, images):
        """
        images: (B, C, H, W)
        """
        features = self.backbone(images)
        features = features.flatten(1)
        embedded = self.projector(features)
        return embedded
```

### 4.2 Complete Visuomotor Diffusion Policy

```python
class VisuomotorDiffusionPolicy(nn.Module):
    """
    Visuomotor diffusion policy
    Combines visual observations and action generation
    """
    def __init__(self, image_dim, state_dim, action_dim, 
                 hidden_dim=256, num_steps=100):
        super(VisuomotorDiffusionPolicy, self).__init__()
        
        # Visual encoder
        self.visual_encoder = VisualEncoder(hidden_dim)
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Observation fusion
        self.observation_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Diffusion model
        self.diffusion = ConditionalDiffusion(
            state_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps
        )
        
    def forward(self, image, state, training=True):
        """Forward pass"""
        # Encode the observation
        visual_feat = self.visual_encoder(image)
        state_feat = self.state_encoder(state)
        
        # Fuse
        observation = torch.cat([visual_feat, state_feat], dim=-1)
        observation = self.observation_fusion(observation)
        
        return self.diffusion(training, observation)
    
    @torch.no_grad()
    def act(self, image, state):
        """Inference to get an action"""
        return self.forward(image, state, training=False)
```

---

## 5. Practical Implementation

### 5.1 Data Preprocessing

```python
class DiffusionDataProcessor:
    """
    Diffusion policy data preprocessing
    """
    def __init__(self, action_scale=1.0):
        self.action_scale = action_scale
        
    def process_demonstrations(self, demonstrations):
        """
        Process demonstration data.
        
        demonstrations: [{'image': ..., 'state': ..., 'action': ...}, ...]
        """
        processed = {
            'images': [],
            'states': [],
            'actions': []
        }
        
        for demo in demonstrations:
            processed['images'].append(demo['image'])
            processed['states'].append(demo['state'])
            processed['actions'].append(demo['action'] * self.action_scale)
            
        return {
            'images': np.array(processed['images']),
            'states': np.array(processed['states']),
            'actions': np.array(processed['actions'])
        }
```

### 5.2 Evaluation

```python
def evaluate_diffusion_policy(env, policy, num_episodes=10):
    """Evaluate the diffusion policy"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get the observation
            image = state['image']
            state_vec = state['vector']
            
            # Convert to tensors
            image_tensor = torch.FloatTensor(image).unsqueeze(0) / 255.0
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
            
            # Get the action
            with torch.no_grad():
                action = policy.act(image_tensor, state_tensor)
                
            # Execute
            next_state, reward, done, _ = env.step(action.squeeze().numpy())
            
            episode_reward += reward
            state = next_state
            
        total_rewards.append(episode_reward)
        
    return np.mean(total_rewards), np.std(total_rewards)
```

---

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
2. Chi, C., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS.
3. Janner, M., et al. (2022). Planning with Diffusion for Flexible Behavior Synthesis. ICLR.
4. Margelidon, T., et al. (2023). EDMP: Ensemble of Diffusion Models for Policy Learning. arXiv.

---

*This chapter is continuously updated...*
