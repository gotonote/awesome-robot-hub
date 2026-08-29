# RT-X / Open X-Embodiment

RT-X and Open X-Embodiment are large-scale cross-embodiment datasets and models in robotics, aiming to achieve generalization across different robot embodiments.

## Contents

- [1. Open X-Embodiment Overview](#1-open-x-embodiment-overview)
- [2. Dataset Structure](#2-dataset-structure)
- [3. Model Training](#3-model-training)
- [4. Cross-Embodiment Generalization](#4-cross-embodiment-generalization)
- [5. Applications](#5-applications)

---

## 1. Open X-Embodiment Overview

### 1.1 Background

```
Traditional problem: every robot needs to be trained separately
Solution: cross-embodiment learning, robots share data

Goal: one model controls different robots
```

### 1.2 Scale

| Statistic | Count |
|-----------|-------|
| Robot types | 30+ |
| Datasets | 100+ |
| Total samples | 1M+ |
| Tasks | 1000+ |

---

## 2. Dataset Structure

### 2.1 Standardized Format

```python
class XEmbodimentDataset:
    """
    Open X-Embodiment dataset format
    """
    def __init__(self):
        # Observation space
        self.observation = {
            'image': (3, 224, 224),      # RGB image
            'wrist_image': (3, 224, 224), # wrist camera
            'state': (14,),              # joint positions + velocities
            'language_instruction': str, # language instruction
        }
        
        # Action space
        self.action = (7,)  # end-effector position + rotation + gripper
        
    def load_dataset(self, dataset_path):
        """Load the dataset"""
        import h5py
        
        data = h5py.File(dataset_path, 'r')
        
        # Standardized processing
        for episode in data['episodes']:
            yield {
                'observation': {
                    'image': episode['observation']['image'],
                    'state': episode['observation']['state'],
                },
                'action': episode['action'],
                'language_instruction': episode['language_instruction'].decode()
            }
```

---

## 3. Model Training

### 3.1 Cross-Embodiment Policy Learning

```python
import torch
import torch.nn as nn

class CrossEmbodimentPolicy(nn.Module):
    """
    Cross-embodiment policy network
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super(CrossEmbodimentPolicy, self).__init__()
        
        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # State encoder
        self.state_encoder = nn.Linear(obs_dim, hidden_dim)
        
        # Text encoder (frozen pretrained)
        from transformers import CLIPTextModel
        self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
        for p in self.text_encoder.parameters():
            p.requires_grad = False
            
        # Fusion
        self.fusion = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, image, state, text):
        # Vision
        img_feat = self.vision_encoder(image)
        
        # State
        state_feat = self.state_encoder(state)
        
        # Text
        text_feat = self.text_encoder(text).last_hidden_state
        
        # Fusion
        fused, _ = self.fusion(
            img_feat.unsqueeze(1), 
            text_feat, 
            text_feat
        )
        
        # Action
        action = self.action_head(fused.squeeze(1))
        
        return action
```

---

## 4. Cross-Embodiment Generalization

### 4.1 Domain Randomization

```python
class DomainRandomization:
    """
    Domain randomization to improve cross-embodiment generalization
    """
    def __init__(self):
        self.robot_type = None
        
    def randomize_observation(self, obs, robot_type):
        """Randomize observations"""
        # Observations may differ across robots
        if robot_type == 'franka':
            # Add noise
            obs['state'] += torch.randn_like(obs['state']) * 0.01
            
        elif robot_type == 'xarm':
            # Adjust the observation range
            obs['state'] = obs['state'] * 0.8
            
        return obs
```

---

## 5. Applications

### 5.1 RT-X Inference

```python
def inference_with_rtx(model, observation, instruction):
    """
    Inference with an RT-X model
    """
    # Preprocess the observation
    image = preprocess_image(observation['image'])
    state = preprocess_state(observation['state'])
    text = preprocess_text(instruction)
    
    # Predict
    with torch.no_grad():
        action = model(image, state, text)
        
    return action
```

---

## References

1. Padalkar, A., et al. (2023). Open X-Embodiment: Robot Learning across Robots and Tasks. arXiv.
2. Brohan, A., et al. (2022). RT-1: Robotics Transformer. arXiv.
3. Xue, T., et al. (2023). Cross-Embodiment Learning. arXiv.

---

*This chapter is continuously updated...*
