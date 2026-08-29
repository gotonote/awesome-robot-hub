# Inverse Reinforcement Learning (IRL)

Inverse reinforcement learning infers the reward function from expert demonstrations — a key technique for understanding expert behavior and generalizing skills.

## Contents

- [1. IRL Overview](#1-irl-overview)
- [2. Maximum Entropy IRL](#2-maximum-entropy-irl)
- [3. Generative Adversarial IRL](#3-generative-adversarial-irl)
- [4. Neural IRL](#4-neural-irl)
- [5. Application Cases](#5-application-cases)

---

## 1. IRL Overview

### 1.1 Problem Definition

```
Traditional RL: given r(s,a) → find the optimal π
Inverse IRL: given demonstrations of π* → infer r(s,a)

Core assumption: expert behavior is optimal or near-optimal
```

### 1.2 Method Taxonomy

| Method | Features | Pros/Cons |
|--------|----------|-----------|
| Maximum entropy IRL | Probabilistic framework, handles multimodality | Computationally complex |
| GAIL | Adversarial learning | Unstable training |
| Neural IRL | Neural network representation | End-to-end |
| Structured IRL | Assumes reward structure | Interpretable |

---

## 2. Maximum Entropy IRL

### 2.1 Principle

The core idea of maximum entropy IRL:

$$
P(\tau | \theta) = \frac{1}{Z(\theta)} \exp(\theta \cdot f(\tau))
$$

where $f(\tau)$ is the trajectory feature and $Z(\theta)$ the partition function.

```python
import torch
import torch.nn as nn
import numpy as np

class MaxEntIRL:
    """
    Maximum entropy inverse reinforcement learning
    """
    def __init__(self, state_dim, action_dim, feature_dim, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        # Reward function parameters
        self.theta = nn.Parameter(torch.randn(feature_dim))
        
        # Optimizer
        self.optimizer = torch.optim.Adam([self.theta], lr=lr)
        
    def reward(self, states, actions):
        """Compute the reward: r(s,a) = θ · φ(s,a)"""
        features = self.extract_features(states, actions)
        return torch.sum(features * self.theta, dim=-1)
    
    def extract_features(self, states, actions):
        """Extract features"""
        # Simplified as state-action concatenation
        return torch.cat([states, actions], dim=-1)
    
    def compute_expert_feature_expectation(self, expert_trajectories):
        """Compute the feature expectation of expert demonstrations"""
        total_feature = 0
        total_steps = 0
        
        for traj in expert_trajectories:
            for step in traj:
                state = torch.FloatTensor(step['state']).unsqueeze(0)
                action = torch.FloatTensor(step['action']).unsqueeze(0)
                
                feature = self.extract_features(state, action)
                total_feature += feature
                total_steps += 1
                
        return total_feature / total_steps
    
    def compute_policy_feature_expectation(self, policy, env, num_trajectories=100):
        """Compute the feature expectation of a policy"""
        total_feature = 0
        total_steps = 0
        
        for _ in range(num_trajectories):
            state = env.reset()
            done = False
            
            while not done:
                action = policy.select_action(state)
                
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action_t = torch.FloatTensor(action).unsqueeze(0)
                
                feature = self.extract_features(state_t, action_t)
                total_feature += feature
                total_steps += 1
                
                state, _, done, _ = env.step(action)
                
        return total_feature / total_steps
    
    def train(self, expert_trajectories, env, policy, num_iterations=100):
        """Train"""
        # Compute the expert feature expectation
        expert_feature_exp = self.compute_expert_feature_expectation(expert_trajectories)
        
        for iteration in range(num_iterations):
            # Feature expectation of the current policy
            policy_feature_exp = self.compute_policy_feature_expectation(policy, env)
            
            # Loss: minimize the difference
            loss = -torch.sum(self.theta * (expert_feature_exp - policy_feature_exp))
            
            # Add regularization
            loss += 0.01 * torch.sum(self.theta ** 2)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if iteration % 10 == 0:
                print(f"Iter {iteration}, Loss: {loss.item():.4f}")
                
        return self.theta
```

---

## 3. Generative Adversarial IRL

### 3.1 The GAIL Principle

GAIL uses a GAN framework:
- Discriminator: distinguishes expert data from policy data
- Generator (policy): tries to fool the discriminator

```python
class GAIL(nn.Module):
    """
    Generative adversarial imitation learning
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(GAIL, self).__init__()
        
        # Policy network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def get_action(self, state):
        """Get an action"""
        return self.actor(state)
    
    def discriminator_loss(self, expert_states, expert_actions, policy_states, policy_actions):
        """Discriminator loss"""
        # Expert: label 1
        expert_input = torch.cat([expert_states, expert_actions], dim=-1)
        expert_prob = self.discriminator(expert_input)
        
        # Policy: label 0
        policy_input = torch.cat([policy_states, policy_actions], dim=-1)
        policy_prob = self.discriminator(policy_input)
        
        # Binary cross-entropy
        loss = -torch.log(expert_prob + 1e-8).mean() - \
               torch.log(1 - policy_prob + 1e-8).mean()
        
        return loss
    
    def actor_loss(self, states, actions):
        """Policy loss (uses the discriminator as a reward)"""
        input_feat = torch.cat([states, actions], dim=-1)
        
        # Discriminator output as reward (higher is better)
        reward = self.discriminator(input_feat)
        
        # Policy gradient
        loss = -torch.log(reward + 1e-8).mean()
        
        return loss
```

---

## 4. Neural IRL

### 4.1 Neural Reward Functions

```python
class NeuralRewardFunction(nn.Module):
    """
    Neural network reward function
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(NeuralRewardFunction, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        input_feat = torch.cat([state, action], dim=-1)
        return self.network(input_feat)


class NeuralIRL:
    """
    Neural inverse reinforcement learning
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.reward_fn = NeuralRewardFunction(state_dim, action_dim, hidden_dim)
        
        # Use PPO as the policy optimizer
        self.policy = PPO(state_dim, action_dim)
        
    def train_step(self, expert_batch, policy_batch):
        """Training step"""
        # Compute rewards
        expert_rewards = self.reward_fn(expert_batch['states'], expert_batch['actions'])
        policy_rewards = self.reward_fn(policy_batch['states'], policy_batch['actions'])
        
        # Reward loss: expert rewards should be higher
        reward_loss = torch.relu(policy_rewards - expert_rewards + 0.1).mean()
        
        # Update the reward function
        reward_optimizer = torch.optim.Adam(self.reward_fn.parameters())
        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()
        
        # Train the policy with the reward function
        self.policy.update(policy_batch['states'], policy_batch['actions'],
                          expert_rewards)
        
        return reward_loss.item()
```

---

## 5. Application Cases

### 5.1 Autonomous Driving

```python
class AutonomousDrivingIRL:
    """
    Autonomous driving inverse reinforcement learning
    Learn the reward function from human driving data
    """
    def __init__(self):
        # State: position, velocity, acceleration, distance to the car ahead, etc.
        # Action: steering angle, throttle, brake
        pass
    
    def extract_driving_features(self, state, action):
        """Extract driving features"""
        features = {
            'speed': state['velocity'],
            'acceleration': action['throttle'],
            'lateral_deviation': state['lane_offset'],
            'time_to_collision': state['ttc'],
            'comfort': abs(action['steering']),
        }
        return features
```

### 5.2 Robot Manipulation

```python
class RobotManipulationIRL:
    """
    Robot manipulation inverse reinforcement learning
    """
    def __init__(self):
        pass
    
    def define_reward_features(self):
        """Define reward features"""
        return {
            # Task-related
            'task_completion': 'whether the task is completed',
            'grasp_stability': 'grasp stability',
            
            # Efficiency-related
            'time_efficiency': 'completion time',
            'path_length': 'path length',
            
            # Safety-related
            'collision': 'whether collision occurs',
            'force_limits': 'force limits',
        }
```

---

## References

1. Ng, A. Y., & Russell, S. (2000). Algorithms for inverse reinforcement learning. ICML.
2. Ziebart, B. D., et al. (2008). Maximum entropy inverse reinforcement learning. AAAI.
3. Ho, J., & Ermon, S. (2016). Generative adversarial imitation learning. NeurIPS.
4. Finn, C., Christiano, P., & Abbeel, P. (2016). A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. NeurIPS.

---

*This chapter is continuously updated...*
