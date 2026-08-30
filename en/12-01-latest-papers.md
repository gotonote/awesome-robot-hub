# Latest Papers

Frontier research achievements in the Physical AI field.

## Contents

- [1. Foundation Models & Robots](#1-foundation-models--robots)
- [2. Reinforcement Learning](#2-reinforcement-learning)
- [3. Imitation Learning](#3-imitation-learning)
- [4. Perception Technology](#4-perception-technology)

---

## 1. Foundation Models & Robots

### 1.1 Important Papers 2024-2025

| Paper | Institution | Contribution |
|-------|-------------|--------------|
| RT-4 | Google DeepMind | Vision-language-action model |
| OpenVLA | Stanford | Open-source VLA |
| π0 | Physical Intelligence | Flow action model |

### 1.2 Core Progress

- **Multimodal understanding**: vision + language + tactile fusion
- **Long-horizon reasoning**: chain-of-thought applied to robots
- **Generalization**: cross-task, cross-embodiment generalization

### 1.3 Example VLA Architecture

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

class VLAModel(nn.Module):
    """Base class of a vision-language-action model"""
    
    def __init__(self, model_name="google/robot-vla"):
        super().__init__()
        self.vision_encoder = AutoModel.from_pretrained(
            "google/siglip-so-400m-patch14-224"
        )
        self.text_encoder = AutoModel.from_pretrained(
            "google/bert-base-uncased"
        )
        self.action_head = nn.Linear(768, 7)  # 7-DOF actions
        
    def forward(self, image, text, state=None):
        # Vision features
        vision_outputs = self.vision_encoder(pixel_values=image)
        vision_embeds = vision_outputs.last_hidden_state
        
        # Text features
        text_outputs = self.text_encoder(input_ids=text)
        text_embeds = text_outputs.last_hidden_state
        
        # Cross-modal fusion
        fused = vision_embeds * text_embeds  # element-wise multiplication
        
        # Action prediction
        action = self.action_head(fused.mean(dim=1))
        
        return action
    
    def predict_action(self, image, instruction, current_state):
        """Action prediction at inference"""
        with torch.no_grad():
            # Encode
            image_enc = self.vision_encoder(image)
            text_enc = self.text_encoder(instruction)
            
            # Fuse
            fused = image_enc.last_hidden_state * text_enc.last_hidden_state
            
            # Predict the action
            action = self.action_head(fused.mean(dim=1))
            
        return action
```

### 1.4 π0 Model Features

```python
class PiZeroModel(nn.Module):
    """
    π0 flow action model
    Feature: a two-stage method of pretraining + post-training
    """
    
    def __init__(self):
        super().__init__()
        # Pretraining stage: internet video understanding
        self.pretrain_encoder = SelfSupervisedEncoder()
        
        # Post-training stage: robot actions
        self.action_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 14),  # 7 joints + 7 velocities
        )
        
    def forward(self, image_sequence, text_instruction):
        # Process the image sequence
        batch_size, seq_len = image_sequence.shape[:2]
        
        # Extract spatiotemporal features
        features = self.pretrain_encoder(
            images=image_sequence.view(-1, *image_sequence.shape[2:])
        )
        features = features.view(batch_size, seq_len, -1)
        
        # Action prediction
        action = self.action_head(features[:, -1])
        
        return action
```

---

## 2. Reinforcement Learning

### 2.1 Sim-to-Real

- Domain randomization techniques mature
- Adaptive domain adaptation
- Few-shot transfer

### 2.2 Domain Randomization Example

```python
class DomainRandomization:
    """Domain randomization - Sim-to-Real transfer"""
    
    def __init__(self):
        self.param_ranges = {
            'friction': (0.3, 1.5),
            'mass': (0.5, 2.0),
            'link_damping': (0.0, 0.5),
            'visual_color': None,  # color randomization
            'light_position': None,
            'texture_randomization': True,
        }
        
    def randomize(self, env):
        """Randomize simulation environment parameters"""
        # Physics parameters
        for body in env.sim.model.body_mass:
            body_mass *= np.random.uniform(*self.param_ranges['mass'])
            
        for geom in env.sim.model.geom_friction:
            geom_friction *= np.random.uniform(*self.param_ranges['friction'])
            
        # Visual randomization
        if self.param_ranges['texture_randomization']:
            self._randomize_textures(env)
            
        return env
    
    def _randomize_textures(self, env):
        """Randomize object textures"""
        texture_pool = ['metal', 'wood', 'plastic', 'rubber']
        for mat_id in env.sim.model.mat_id:
            mat_id.texture_id = np.random.choice(texture_pool)
```

### 2.3 Offline RL

- CQL/BCQ industrial applications
- Data-efficient algorithms
- Safety-constrained learning

### 2.4 CQL Implementation

```python
class ConservativeQLearning:
    """
    CQL (Conservative Q-Learning)
    Core idea: minimize Q values to avoid over-optimism
    """
    
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.005):
        self.gamma = gamma
        self.alpha = alpha  # conservative coefficient
        
        self.q_network = QNetwork(state_dim, action_dim)
        self.q_target = QNetwork(state_dim, action_dim)
        self.q_target.load_state_dict(self.q_network.state_dict())
        
    def compute_cql_loss(self, batch):
        """Compute the CQL loss"""
        states, actions, rewards, next_states, dones = batch
        
        # Standard Q-learning loss
        current_q = self.q_network(states, actions)
        with torch.no_grad():
            next_q = self.q_target(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * (1 - dones) * next_q
            
        standard_loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Additional CQL loss: encourage conservative Q values
        # Sample actions ~ Uniform(a)
        sampled_actions = torch.rand(states.shape[0], actions.shape[1]) * 2 - 1
        sampled_q = self.q_network(states, sampled_actions)
        
        # Maximize the sampled Q, then negate (minimize)
        cql_loss = -self.alpha * (sampled_q.mean() - current_q.mean())
        
        return standard_loss + cql_loss
```

---

## 3. Imitation Learning

### 3.1 Diffusion Policies

- Diffusion Policy (RSS 2023)
- Industrial application cases
- Real-time inference optimization

### 3.2 Diffusion Policy Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionPolicy(nn.Module):
    """Diffusion policy - policy learning based on a denoising process"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_steps=100):
        super().__init__()
        self.num_steps = num_steps
        
        # Time step embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Noise prediction network
        self.noise_pred_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, state, noisy_action, t):
        """Predict the noise"""
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        x = torch.cat([state, noisy_action], dim=-1)
        noise_pred = self.noise_pred_net(x + t_emb)
        return noise_pred
    
    def sample_actions(self, state, num_samples=10):
        """Start from noise and gradually denoise to obtain actions"""
        batch_size = state.shape[0]
        
        # Initial noise
        action = torch.randn(batch_size, self.action_dim, device=state.device)
        
        # Gradual denoising
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((batch_size,), t / self.num_steps, device=state.device)
            
            with torch.no_grad():
                noise_pred = self.forward(state, action, t_tensor)
                
            # Denoising step (simplified DDPM)
            alpha = (t + 1) / self.num_steps
            action = action - (1 - alpha) * noise_pred / torch.sqrt(alpha)
            
        return action
```

### 3.3 Datasets

- Open X-Embodiment
- DROID
- Internet robot data

### 3.4 Data Loading Example

```python
import h5py
from torch.utils.data import Dataset

class RobotDataset(Dataset):
    """Robot demonstration dataset loader"""
    
    def __init__(self, dataset_path):
        self.data = h5py.File(dataset_path, 'r')
        
    def __len__(self):
        return len(self.data['observations'])
    
    def __getitem__(self, idx):
        # Observation
        obs = {
            'image': self.data['observations']['image'][idx],
            'state': self.data['observations']['state'][idx],
            'instruction': self.data['observations']['language_instruction'][idx].decode(),
        }
        
        # Action
        action = self.data['actions'][idx]
        
        return obs, action
    
    def get_episode(self, episode_id):
        """Get one complete episode"""
        ep_start = self.data['episodestarts'][episode_id]
        ep_end = self.data['episodeends'][episode_id]
        
        observations = self.data['observations'][ep_start:ep_end]
        actions = self.data['actions'][ep_start:ep_end]
        
        return observations, actions
```

---

## 4. Perception Technology

### 4.1 3D Perception

- Real-time NeRF
- Neural RGBD
- End-to-end perception

### 4.2 Neural RGBD Reconstruction

```python
class NeuralRGBD(nn.Module):
    """Neural RGBD - depth estimation combined with neural rendering"""
    
    def __init__(self):
        super().__init__()
        
        # Depth estimation network
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),
            nn.ReLU(),
            *self._make_encoder_block(64, 128),
            *self._make_encoder_block(128, 256),
            *self._make_encoder_block(256, 512),
        )
        
        self.depth_decoder = nn.Sequential(
            *self._make_decoder_block(512, 256),
            *self._make_decoder_block(256, 128),
            *self._make_decoder_block(128, 64),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),  # depth normalized to [0,1]
        )
        
        # NeRF color prediction
        self.nerf_head = nn.Sequential(
            nn.Linear(128 + 3 + 2),  # feature + position + direction
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(128, 4),  # RGB + density
        )
        
    def forward(self, image):
        # Depth estimation
        features = self.depth_encoder(image)
        depth = self.depth_decoder(features)
        
        # Neural rendering
        color, density = self.nerf_render(features, image.shape[2:])
        
        return depth, color, density
    
    def _make_encoder_block(self, in_ch, out_ch):
        return [
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        ]
    
    def _make_decoder_block(self, in_ch, out_ch):
        return [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.ReLU(),
        ]
```

---

## References

1. Brohan, A., et al. (2024). RT-2: Vision-Language-Action Models. arXiv.
2. Black, K., et al. (2024). π0: A Vision-Language-Action Flow Model. Physical Intelligence.
3. Haarnoja, T., et al. (2024). Diffusion Policies for Real-World Robot Learning. RSS.
4. Kumar, A., et al. (2021). Conservative Q-Learning for Offline RL. NeurIPS.

---

*This chapter is continuously updated...*
