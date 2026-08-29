# DAgger (Dataset Aggregation)

DAgger solves the distribution shift problem of behavior cloning through iterative expert aggregation — an important algorithm in imitation learning.

## Contents

- [1. DAgger Principle](#1-dagger-principle)
- [2. Algorithm Implementation](#2-algorithm-implementation)
- [3. Variants & Improvements](#3-variants--improvements)
- [4. Application Scenarios](#4-application-scenarios)

---

## 1. DAgger Principle

### 1.1 The Problem with Behavior Cloning

Behavior cloning objective:

$$
\min_\theta \mathbb{E}_{(s,a) \sim D}[L(\pi_\theta(s), a)]
$$

**Problem**: the state distribution at training time ≠ the state distribution at test time

```
Training: s ~ D (expert trajectory state distribution)
Testing:  s ~ D_π (policy-generated trajectory state distribution)

When the policy deviates from the expert, the gap between test and
training distributions grows → errors compound → policy collapse
```

### 1.2 The DAgger Solution

```
DAgger algorithm:

1. Obtain an expert policy π*
2. Collect a dataset D = {(s, a*)} with π*
3. Train a policy π_Dagger ≈ argmin_π E[L(π(s), a*)]
4. Collect states with π_Dagger: D_π = {s}
5. Expert annotation: label each state in D_π with a* = π*(s)
6. Aggregate: D ← D ∪ D_π
7. Return to step 3
```

---

## 2. Algorithm Implementation

### 2.1 Basic DAgger

```python
import numpy as np
import torch
import torch.nn as nn
from collections import deque

class DAgger:
    """
    DAgger (Dataset Aggregation) implementation
    """
    def __init__(self, policy, expert_policy, env):
        """
        policy: the learned policy to train
        expert_policy: the expert policy
        env: the environment
        """
        self.policy = policy
        self.expert_policy = expert_policy
        self.env = env
        
        # Aggregated dataset
        self.dataset = {
            'states': [],
            'actions': []
        }
        
    def collect_expert_demonstrations(self, num_episodes=10):
        """Collect demonstrations with the expert policy"""
        demonstrations = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_data = []
            
            while not done:
                # The expert chooses an action
                action = self.expert_policy.get_action(state)
                
                # Store
                episode_data.append({
                    'state': state,
                    'action': action
                })
                
                # Execute
                state, reward, done, _ = self.env.step(action)
                
            demonstrations.append(episode_data)
            
        return demonstrations
    
    def collect_policy_demonstrations(self, policy, num_episodes=10):
        """Collect states with the policy"""
        policy_states = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                # The policy chooses an action (states are what matter)
                action = policy.get_action(state)
                
                # Store the state
                policy_states.append(state.copy())
                
                # Execute (keep using the policy)
                state, _, done, _ = self.env.step(action)
                
        return policy_states
    
    def aggregate_dataset(self, demonstrations, policy_states, expert_policy):
        """Aggregate the dataset"""
        # Add expert demonstrations
        for episode in demonstrations:
            for step in episode:
                self.dataset['states'].append(step['state'])
                self.dataset['actions'].append(step['action'])
        
        # Add policy states with expert annotations
        for state in policy_states:
            # Expert annotation
            expert_action = expert_policy.get_action(state)
            
            self.dataset['states'].append(state)
            self.dataset['actions'].append(expert_action)
            
    def train_policy(self, epochs=10, batch_size=32):
        """Train the policy"""
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.dataset['states']))
        actions = torch.FloatTensor(np.array(self.dataset['actions']))
        
        dataset = torch.utils.data.TensorDataset(states, actions)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_states, batch_actions in dataloader:
                # Predict
                pred_actions = self.policy(batch_states)
                
                # Loss
                loss = nn.MSELoss()(pred_actions, batch_actions)
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
                
    def run(self, num_iterations=5, expert_demos_per_iter=5, 
            policy_demos_per_iter=5, epochs=10):
        """Run DAgger"""
        # Initial: collect expert demonstrations
        print("Collecting expert demonstrations...")
        demonstrations = self.collect_expert_demos(expert_demos_per_iter)
        
        for episode in demonstrations:
            for step in episode:
                self.dataset['states'].append(step['state'])
                self.dataset['actions'].append(step['action'])
        
        # Iterate
        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            
            # 1. Train the policy
            print("Training policy...")
            self.train_policy(epochs=epochs)
            
            # 2. Collect states with the policy
            print("Collecting policy states...")
            policy_states = self.collect_policy_demos(policy_demos_per_iter)
            
            # 3. Expert annotation
            print("Expert annotation...")
            for state in policy_states:
                expert_action = self.expert_policy.get_action(state)
                self.dataset['states'].append(state)
                self.dataset['actions'].append(expert_action)
                
            print(f"Dataset size: {len(self.dataset['states'])}")
            
        print("\nTraining complete!")
```

### 2.2 DAgger with PPO

```python
class DAggerPPO:
    """
    DAgger combined with PPO
    Uses policy gradients to improve performance
    """
    def __init__(self, policy, expert_policy, env, lr=3e-4):
        self.policy = policy
        self.expert = expert_policy
        self.env = env
        
        # PPO optimizer
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
    def compute_advantage(self, states, expert_actions):
        """Compute the advantage using expert actions as a baseline"""
        with torch.no_grad():
            expert_actions = torch.FloatTensor(expert_actions)
            
            # Predicted actions
            pred_actions = self.policy(states)
            
            # Advantage = -||π(s) - a*||^2
            advantage = -((pred_actions - expert_actions) ** 2).sum(dim=-1)
            
        return advantage
    
    def ppo_update(self, states, expert_actions):
        """PPO update"""
        advantage = self.compute_advantage(states, expert_actions)
        
        # Policy loss
        pred_actions = self.policy(states)
        
        # Simplified policy gradient
        loss = -(pred_actions * advantage.unsqueeze(-1)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## 3. Variants & Improvements

### 3.1 AggreVaTe

AggreVaTe (Aggregate Value Functions) extends DAgger:

```python
class AggreVaTe:
    """
    AggreVaTe: aggregate value functions
    Considers long-term returns rather than one-step loss
    """
    def __init__(self, policy, expert, env):
        self.policy = policy
        self.expert = expert
        self.env = env
        
        # Value function
        self.value_fn = ValueNetwork()
        
    def dagger_iteration(self):
        """DAgger iteration + value function"""
        # 1. Collect data
        expert_trajs = self.collect_expert_trajectories()
        policy_states = self.collect_policy_states()
        
        # 2. Estimate the cost of each state
        # Q(s, a*) - V(s) approximates the expert advantage
        for state in policy_states:
            expert_action = self.expert.get_action(state)
            q_value = self.value_fn(state, expert_action)
            v_value = self.value_fn(state, self.policy.get_action(state))
            cost = q_value - v_value
            
            # Weight = -cost (higher cost → larger weight)
            self.weighted_dataset.append((state, expert_action, -cost))
```

### 3.2 DAgger with Uncertainty

```python
class DAggerWithUncertainty:
    """
    DAgger with uncertainty
    Query the expert preferentially on high-uncertainty states
    """
    def __init__(self, policy, expert, env):
        self.policy = policy
        self.expert = expert
        self.env = env
        
        # Bayesian neural network for uncertainty estimation
        self.ensemble = EnsembleNetwork(n_models=5)
        
    def selective_labeling(self, states):
        """Selective labeling - only have the expert label uncertain states"""
        uncertainties = []
        
        for state in states:
            # Predicted actions
            actions = self.ensemble.get_predictions(state)
            
            # Uncertainty = variance of predictions
            uncertainty = actions.var(dim=0).mean()
            uncertainties.append(uncertainty)
            
        # Select the top-K uncertain states
        k = len(states) // 10  # 10%
        selected_indices = np.argsort(uncertainties)[-k:]
        
        return [states[i] for i in selected_indices]
```

---

## 4. Application Scenarios

### 4.1 Autonomous Driving

```python
class AutonomousDrivingDAgger:
    """
    Autonomous driving DAgger application
    """
    def __init__(self):
        self.state_dim = 20  # position, velocity, lane, etc.
        self.action_dim = 2  # steering, throttle
        
    def collect_state(self):
        """Collect the driving state"""
        return {
            'position': self.get_gps(),
            'velocity': self.get_speed(),
            'lane_offset': self.get_lane_position(),
            'distance_to_vehicle': self.get_distance(),
            'traffic_light': self.get_traffic_light(),
        }
    
    def run_dagger(self):
        """Run DAgger"""
        policy = DrivingPolicyNetwork()
        expert = HumanDriver()
        
        dagger = DAgger(policy, expert, self.env)
        dagger.run(num_iterations=20)
```

### 4.2 Robot Manipulation

```python
class RobotManipulationDAgger:
    """
    Robot manipulation DAgger application
    """
    def __init__(self):
        self.robot = Robot()
        
    def run(self):
        """Run DAgger"""
        policy = ManipulationPolicy()
        expert = TeleoperationExpert(self.robot)
        
        dagger = DAgger(policy, expert, self.env)
        dagger.run(
            num_iterations=10,
            expert_demos_per_iter=20,
            policy_demos_per_iter=20
        )
```

---

## References

1. Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning to no-regret online learning. AISTATS.
2. Abbeel, P., & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning. ICML.
3. Ross, S., & Bagnell, D. (2014). Learning Monotonic Alignments. NeurIPS.

---

*This chapter is continuously updated...*
