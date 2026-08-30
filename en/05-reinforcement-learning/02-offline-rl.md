# Offline Reinforcement Learning (Offline RL)

Offline reinforcement learning, also called batch RL, learns a policy from a fixed dataset without online interaction with the environment. This is highly relevant for robot learning because real-robot online interaction is expensive and carries safety risks.

## Contents

- [1. Offline RL Overview](#1-offline-rl-overview)
- [2. The Distribution Shift Problem](#2-the-distribution-shift-problem)
- [3. Constraint-Based Methods](#3-constraint-based-methods)
- [4. Model-Based Methods](#4-model-based-methods)
- [5. Experience Replay Methods](#5-experience-replay-methods)
- [6. Practical Frameworks](#6-practical-frameworks)

---

## 1. Offline RL Overview

### 1.1 Offline RL vs. Online RL

| Aspect | Online RL | Offline RL |
|--------|-----------|------------|
| Data collection | Online interaction with the environment | Pre-collected fixed dataset |
| Exploration | Required | Not required |
| Sample efficiency | Lower | Higher |
| Safety risk | Potentially risky | Risk-free |
| Distribution shift | None | Present |

### 1.2 Challenges of Offline RL

```
Core problem: Distribution Shift

Online RL: policy collects data → learn policy → collect new data → ...
Offline RL: fixed dataset → learn policy (distribution shift!)

The distribution of data collected by the learned policy ≠ the distribution in the dataset
```

---

## 2. The Distribution Shift Problem

### 2.1 The Essence of the Problem

```
Dataset: D = {(s, a, r, s')}
Policy: π(a|s)

The learned Q(s, a) is based on (s, a) pairs in the dataset
But at deployment, π(a|s) may produce actions never seen in the dataset
```

### 2.2 Error Propagation

```python
# The problem with classic DQN in the offline setting
class OfflineRLProblem:
    def demonstrate_extrapolation_error():
        """
        Extrapolation error example
        
        Assume the dataset only covers a small range of actions
        When the policy selects actions slightly outside this range,
        Q-value estimates deviate severely
        """
        pass
    
    def demonstrate_overestimation():
        """
        Overestimation problem
        
        Maximizing Q values tends to select actions not covered by the dataset,
        causing policy degradation
        """
        pass
```

---

## 3. Constraint-Based Methods

### 3.1 Conservative Q-Learning (CQL)

```python
import torch
import torch.nn as nn
import numpy as np

class ConservativeQLearning:
    """
    Conservative Q-Learning (CQL)
    Core idea: penalize the Q values of uncovered actions to avoid distribution shift
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99):
        self.gamma = gamma
        
        # Q networks
        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.Q1_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        
        self.Q2_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=3e-4
        )
        
    def update(self, batch, alpha=1.0):
        states, actions, rewards, next_states, dones = batch
        
        # ===== Critic Update (CQL) =====
        # Standard TD loss
        with torch.no_grad():
            next_actions = self.policy(next_states)
            next_q1 = self.Q1_target(torch.cat([next_states, next_actions], dim=-1))
            next_q2 = self.Q2_target(torch.cat([next_states, next_actions], dim=-1))
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Current Q values
        current_q1 = self.Q1(torch.cat([states, actions], dim=-1)).squeeze()
        current_q2 = self.Q2(torch.cat([states, actions], dim=-1)).squeeze()
        
        # TD loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + \
                      nn.MSELoss()(current_q2, target_q)
        
        # CQL regularization loss
        # Sample actions and compute their Q values
        random_actions = torch.rand_like(actions) * 2 - 1  # uniform distribution
        sampled_actions = self.policy(states).detach()
        
        # Compute Q values of uncovered actions
        q_random = self.Q1(torch.cat([states, random_actions], dim=-1)).mean()
        q_sampled = self.Q1(torch.cat([states, sampled_actions], dim=-1)).mean()
        
        # CQL loss: encourage a lower bound on Q
        cql_loss = alpha * (q_random - q_sampled)
        
        total_critic_loss = critic_loss - cql_loss
        
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()
        
        # ===== Actor Update =====
        new_actions = self.policy(states)
        q_new = self.Q1(torch.cat([states, new_actions], dim=-1)).mean()
        
        actor_loss = -q_new
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'cql_loss': cql_loss.item(),
            'actor_loss': actor_loss.item()
        }
```

### 3.2 Policy Constraint Methods

```python
class BehavioralCloningPolicy:
    """
    Behavior cloning: learn a policy that imitates the dataset
    Used as a constraint term in the RL objective
    """
    def __init__(self, state_dim, action_dim):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def behavior_clone(self, states, expert_actions):
        """Behavior cloning loss"""
        pred_actions = self.policy(states)
        loss = nn.MSELoss()(pred_actions, expert_actions)
        return loss


class BCQAgent:
    """
    BCQ (Batch Constrained Q-Learning)
    Core idea: maximize under constraints
    """
    def __init__(self, state_dim, action_dim):
        # A VAE that generates expert-like actions
        self.vae = VAE(state_dim, action_dim)
        
        # Q network
        self.Q = QNetwork(state_dim, action_dim)
        self.Q_target = QNetwork(state_dim, action_dim)
        
        # Perturbation network
        self.perturb = PerturbationNetwork(state_dim, action_dim)
        
    def select_action(self, state):
        # Generate an action
        recon_action, _, _ = self.vae(state)
        
        # Add perturbation
        perturbation = self.perturb(state, recon_action)
        action = recon_action + perturbation
        
        return torch.clamp(action, -1, 1)
```

---

## 4. Model-Based Methods

### 4.1 Model Predictive Control (MPC)

```python
class ModelBasedRL:
    """
    Model-based RL: first learn an environment model, then plan with it
    """
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Learn the environment dynamics model
        self.dynamics_model = DynamicsModel(state_dim, action_dim)
        
        # Controller
        self.controller = MPCController(self.dynamics_model)
        
    def update_model(self, dataset):
        """Learn the dynamics model from data"""
        # Supervised learning: s' = f(s, a)
        states, actions, next_states = dataset['s'], dataset['a'], dataset['s']
        
        loss = self.dynamics_model.fit(states, actions, next_states)
        return loss
    
    def plan(self, state, horizon=10):
        """Plan with the model"""
        return self.controller.solve(state, horizon)
```

### 4.2 The Dreamer Architecture

```python
class Dreamer:
    """
    Dreamer: learn a world model from images
    Core idea: imagination rollout
    """
    def __init__(self, state_dim, action_dim):
        # World model
        self.encoder = Encoder()
        self.dynamics = RSSM()  # Recurrent State-Space Model
        self.reward_model = RewardModel()
        self.decoder = Decoder()
        
        # Policy and value networks
        self.actor = Actor()
        self.critic = Critic()
        
    def imagine_rollout(self, latent, policy, horizon=50):
        """Imagination rollout"""
        trajectory = [latent]
        
        for _ in range(horizon):
            action = policy(latent)
            latent, reward = self.dynamics.step(latent, action)
            trajectory.append(latent)
            
        return trajectory
```

---

## 5. Experience Replay Methods

### 5.1 Weighted Experience Replay

```python
class WeightedReplayBuffer:
    """
    Prioritized Experience Replay
    Prioritize experiences with high TD error
    """
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # priority exponent
        self.beta = beta     # importance sampling exponent
        
        self.buffer = []
        self.priorities = []
        
    def push(self, state, action, reward, next_state, done):
        # Maximum priority
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            # Replace
            min_idx = np.argmin(self.priorities)
            self.buffer[min_idx] = experience
            self.priorities[min_idx] = max_priority
            
    def sample(self, batch_size):
        # Compute sampling probabilities
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        samples = [self.buffer[i] for i in indices]
        
        return samples, indices, weights
```

---

## 6. Practical Frameworks

### 6.1 The D4RL Datasets

```python
import gym
import d4rl  # standard offline RL datasets

def load_d4rl_dataset(env_name='maze2d-medium-v0'):
    """Load a D4RL offline dataset"""
    env = gym.make(env_name)
    dataset = env.get_dataset()
    
    return {
        'observations': dataset['observations'],
        'actions': dataset['actions'],
        'rewards': dataset['rewards'],
        'next_observations': dataset['next_observations'],
        'dones': dataset['terminals']
    }
```

### 6.2 Complete Training Pipeline

```python
def train_offline_rl():
    """Complete offline RL training pipeline"""
    # 1. Load the environment
    env = gym.make('ant-medium-v0')
    
    # 2. Load the offline dataset
    dataset = env.get_dataset()
    
    # 3. Initialize the agent (CQL as an example)
    agent = ConservativeQLearning(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    # 4. Training loop
    for step in range(1000000):
        # Sample a batch
        batch = sample_batch(dataset, batch_size=256)
        
        # Update
        losses = agent.update(batch)
        
        # Periodic evaluation
        if step % 10000 == 0:
            eval_return = evaluate(agent, env)
            print(f"Step {step}, Eval Return: {eval_return}")
```

---

## References

1. Levine, S., et al. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. arXiv.
2. Kumar, A., et al. (2020). Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS.
3. Fujita, Y., et al. (2021). Combating Selection Bias in Offline Reinforcement Learning. arXiv.

---

*This chapter is continuously updated...*
