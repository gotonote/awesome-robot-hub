# ALOHA / ACT

ALOHA (Affordable Low-cost Open-source Anthropomorphic) and ACT (Action Chunking Transformer) form a low-cost, open-source robot learning platform specifically designed for imitation learning.

## Contents

- [1. The ALOHA Platform](#1-the-aloha-platform)
- [2. The ACT Algorithm](#2-the-act-algorithm)
- [3. Hardware Design](#3-hardware-design)
- [4. Training Pipeline](#4-training-pipeline)
- [5. Code Implementation](#5-code-implementation)

---

## 1. The ALOHA Platform

### 1.1 Platform Overview

ALOHA is a low-cost robot learning platform with the following features:

- Open-source hardware design
- Precise position-controlled robot arms
- Tactile sensors
- Bimanual (dual-arm) operation

### 1.2 System Architecture

```
┌─────────────────────────────────────────┐
│           ALOHA 系统架构                │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐    ┌──────────┐          │
│  │  机械臂  │    │  机械臂  │          │
│  │  (Left)  │    │  (Right) │          │
│  └────┬─────┘    └────┬─────┘          │
│       │                │                │
│       └────────┬───────┘                │
│                │                         │
│         ┌─────┴─────┐                   │
│         │  控制器    │                   │
│         │  (ESP32)  │                   │
│         └─────┬─────┘                   │
│               │                         │
│         ┌─────┴─────┐                   │
│         │   PC      │                   │
│         │  (Training)│                  │
│         └───────────┘                   │
│                                         │
└─────────────────────────────────────────┘
```

*(Robot arms (left/right) → controller (ESP32) → PC (training))*

---

## 2. The ACT Algorithm

### 2.1 Action Chunking

Core idea: package a sequence of actions into a chunk and predict the whole chunk:

```
Traditional: predict frame by frame a_t
ACT: predict a chunk [a_t, a_{t+1}, ..., a_{t+T-1}]
```

### 2.2 Transformer Architecture

```python
import torch
import torch.nn as nn
import numpy as np

class ActionChunkingTransformer(nn.Module):
    """
    Action Chunking Transformer (ACT)
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=512, 
                 num_layers=6, num_heads=8, chunk_size=100):
        super(ActionChunkingTransformer, self).__init__()
        
        self.chunk_size = chunk_size
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Positional encodings
        self.action_pos_embedding = nn.Parameter(
            torch.randn(1, chunk_size, hidden_dim) * 0.02
        )
        self.query_pos_embedding = nn.Parameter(
            torch.randn(1, chunk_size, hidden_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Action decoder
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * chunk_size)
        )
        
    def forward(self, obs_history, action_history=None):
        """
        obs_history: (B, T, obs_dim) history of observations
        action_history: (B, T, action_dim) history of actions (optional)
        """
        batch_size = obs_history.shape[0]
        
        # Encode observations
        obs_feat = self.obs_encoder(obs_history)
        
        # Add query positional encoding
        obs_feat = obs_feat + self.query_pos_embedding[:, :obs_feat.shape[1], :]
        
        # Transformer encoding
        encoded = self.transformer(obs_feat)
        
        # Take the last chunk_size as queries
        query = encoded[:, -self.chunk_size:, :]
        
        # Add action positional encoding
        query = query + self.action_pos_embedding
        
        # Predict actions
        actions = self.action_head(query)
        
        # Reshape: (B, T, action_dim)
        actions = actions.view(batch_size, self.chunk_size, -1)
        
        return actions
    
    def predict(self, obs):
        """
        Inference: predict the actions of the next chunk
        """
        with torch.no_grad():
            # Expand to a history sequence
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
                
            # Repeat to the history length
            obs = obs.unsqueeze(1).repeat(1, 10, 1)
            
            # Predict
            chunk_actions = self.forward(obs)
            
            # Return the first action
            return chunk_actions[:, 0, :]
```

---

## 3. Hardware Design

### 3.1 Robot Arm Specifications

| Parameter | Value |
|-----------|-------|
| DOF | 6 DOF |
| Control frequency | 100 Hz |
| Repeatability | ±0.2 mm |
| Payload | 0.5 kg |
| Cost | ~$500/arm |

### 3.2 Sensors

```
- Joint position sensors: magnetic encoders
- End-effector force sensors: 6-axis torque sensors
- Vision: RealSense D435
- Tactile: customized GelSight
```

---

## 4. Training Pipeline

### 4.1 Data Collection

```python
def collect_demonstration(env, num_episodes=100):
    """
    Collect human demonstrations
    """
    dataset = {
        'observations': [],
        'actions': [],
        'language': []
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        
        episode_data = {
            'observations': [],
            'actions': []
        }
        
        while not done:
            # Render
            env.render()
            
            # Manual control or teleoperation
            action = teleoperation.get_action()
            
            # Execute
            next_obs, reward, done, info = env.step(action)
            
            # Record
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            
            obs = next_obs
            
        dataset['observations'].extend(episode_data['observations'])
        dataset['actions'].extend(episode_data['actions'])
        
    return dataset
```

### 4.2 Model Training

```python
def train_act(dataset, epochs=100):
    """
    Train the ACT model
    """
    model = ActionChunkingTransformer(
        obs_dim=14 + 3*224*224,  # state + images
        action_dim=14,
        chunk_size=100
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Data conversion
    observations = torch.FloatTensor(dataset['observations'])
    actions = torch.FloatTensor(dataset['actions'])
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        
        # Randomly sample sequences
        for i in range(0, len(observations) - 100, 100):
            obs_seq = observations[i:i+100]
            act_seq = actions[i:i+100]
            
            # Predict
            pred_act = model(obs_seq.unsqueeze(0))
            
            # Loss
            loss = nn.MSELoss()(pred_act.squeeze(0), act_seq)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        
    return model
```

---

## 5. Code Implementation

### 5.1 Complete Training Script

```python
import torch
import numpy as np
from pathlib import Path

class ALOHATrainer:
    def __init__(self, config):
        self.config = config
        
        # Initialize the model
        self.model = ActionChunkingTransformer(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            chunk_size=config.chunk_size
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
    def train(self, dataset_path):
        """Train"""
        # Load data
        dataset = self.load_dataset(dataset_path)
        
        for epoch in range(self.config.epochs):
            # Randomly sample a batch
            batch = self.sample_batch(dataset)
            
            # Forward
            pred_actions = self.model(batch['obs'])
            
            # Loss
            loss = self.compute_loss(pred_actions, batch['actions'])
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Logging
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
                
    def load_dataset(self, path):
        """Load the dataset"""
        # Load the npz file
        data = np.load(path)
        
        return {
            'observations': data['observations'],
            'actions': data['actions'],
            'language_instructions': data['language']
        }
```

---

## References

1. Zhao, T. Z., et al. (2023). ALBEF: Align Before Fuse. arXiv.
2. Zhou, Y., et al. (2023). Learning from Rich Human Demonstration. arXiv.
3. Fu, Z., et al. (2024). Learning Fine-grained Bimanual Manipulation. arXiv.

---

*This chapter is continuously updated...*
