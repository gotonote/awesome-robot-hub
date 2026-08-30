# Reinforcement Learning Fundamentals

Reinforcement Learning (RL) is a core technology in Physical AI — the agent learns optimal policies through interaction with the environment. This chapter covers RL theory, core algorithms, and implementations.

## Contents

- [1. RL Overview](#1-rl-overview)
- [2. Markov Decision Processes](#2-markov-decision-processes)
- [3. Dynamic Programming](#3-dynamic-programming)
- [4. Monte Carlo Methods](#4-monte-carlo-methods)
- [5. Temporal Difference Learning](#5-temporal-difference-learning)
- [6. Value Function Approximation](#6-value-function-approximation)
- [7. Policy Gradient Methods](#7-policy-gradient-methods)
- [8. DQN and Its Variants](#8-dqn-and-its-variants)

---

## 1. RL Overview

### 1.1 The RL Framework

```
┌─────────────────────────────────────────┐
│           强化学习交互框架              │
├─────────────────────────────────────────┤
│                                         │
│    ┌───────┐         ┌───────┐         │
│    │ 智能体 │────────▶│  环境  │         │
│    │ Agent │◀────────│Env    │         │
│    └───────┘         └───────┘         │
│       │                  │              │
│       │ Action (a)       │              │
│       │─────────────────▶│              │
│       │                  │              │
│       │◀─────────────────│              │
│       │   Reward (r)     │              │
│       │   State (s')     │              │
│                                         │
└─────────────────────────────────────────┘
```

*(Agent ↔ Environment: action (a) sent to the environment; reward (r) and state (s') returned to the agent.)*

### 1.2 Basic RL Elements

| Element | Description | Role |
|---------|-------------|------|
| State (s) | Observation of the environment | Basis for agent decisions |
| Action (a) | Agent behavior | Affects the environment state |
| Reward (r) | Environment feedback | Guides policy optimization |
| Policy π | Mapping from states to actions | Agent's behavior rule |
| Value function V | Estimate of long-term return | Evaluates states/actions |
| Model | Environment dynamics | Predicts future states and rewards |

---

## 2. Markov Decision Processes

### 2.1 The Markov Property

**Markov property**: the future depends only on the current state, not on history.

$$
P(s_{t+1} | s_t, s_{t-1}, ..., s_0) = P(s_{t+1} | s_t)
$$

### 2.2 MDP Definition

An MDP is defined by the 5-tuple $(S, A, P, R, \gamma)$:

- **S**: State Space
- **A**: Action Space
- **P**: Transition Probability $P(s'|s,a)$
- **R**: Reward Function $R(s,a,s')$
- **$\gamma$**: Discount Factor, $\gamma \in [0, 1]$

### 2.3 Returns and Value Functions

**Return**: the cumulative discounted reward starting from time t

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

**State Value Function**

$$
V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]
$$

**Action Value Function**

$$
Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]
$$

**Bellman Equations**

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]
$$

$$
Q^\pi(s, a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]
$$

### 2.4 Optimal Policies

**Optimal value functions**

$$
V^*(s) = \max_\pi V^\pi(s)
$$

$$
Q^*(s, a) = \max_\pi Q^\pi(s, a)
$$

**Bellman optimality equations**

$$
V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]
$$

---

## 3. Dynamic Programming

### 3.1 Policy Iteration

```python
import numpy as np

class PolicyIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # Initialize
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        self.V = np.zeros(n_states)
        self.policy = np.ones([n_states, n_actions]) / n_actions  # uniform random policy
        
    def policy_evaluation(self):
        """Policy evaluation: compute the value function of a given policy"""
        while True:
            delta = 0
            
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                
                # Compute the new value
                new_v = 0
                for a in range(self.env.action_space.n):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        new_v += self.policy[s, a] * prob * (
                            reward + self.gamma * self.V[next_state]
                        )
                
                self.V[s] = new_v
                delta = max(delta, abs(v - new_v))
            
            if delta < self.theta:
                break
                
    def policy_improvement(self):
        """Policy improvement: update the policy based on the value function"""
        policy_stable = True
        
        for s in range(self.env.observation_space.n):
            old_action = np.argmax(self.policy[s])
            
            # Compute Q values for each action
            action_values = []
            for a in range(self.env.action_space.n):
                q_value = 0
                for prob, next_state, reward, done in self.env.P[s][a]:
                    q_value += prob * (reward + self.gamma * self.V[next_state])
                action_values.append(q_value)
            
            # Choose the best action
            best_action = np.argmax(action_values)
            
            if old_action != best_action:
                policy_stable = False
            
            # Update the policy
            self.policy[s] = np.zeros(self.env.action_space.n)
            self.policy[s, best_action] = 1.0
            
        return policy_stable
    
    def train(self, max_iterations=1000):
        """Policy iteration training"""
        for i in range(max_iterations):
            # Policy evaluation
            self.policy_evaluation()
            
            # Policy improvement
            if self.policy_improvement():
                print(f"Policy converged after {i+1} iterations")
                break
```

### 3.2 Value Iteration

```python
class ValueIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        self.V = np.zeros(n_states)
        self.policy = np.zeros([n_states, n_actions])
        
    def value_iteration(self):
        """Value iteration"""
        while True:
            delta = 0
            
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                
                # Compute the max
                action_values = []
                for a in range(self.env.action_space.n):
                    q_value = 0
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        q_value += prob * (reward + self.gamma * self.V[next_state])
                    action_values.append(q_value)
                
                self.V[s] = max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            if delta < self.theta:
                break
        
        # Extract the policy from the value function
        self.extract_policy()
        
    def extract_policy(self):
        """Extract the policy"""
        for s in range(self.env.observation_space.n):
            action_values = []
            for a in range(self.env.action_space.n):
                q_value = 0
                for prob, next_state, reward, done in self.env.P[s][a]:
                    q_value += prob * (reward + self.gamma * self.V[next_state])
                action_values.append(q_value)
            
            best_action = np.argmax(action_values)
            self.policy[s, best_action] = 1.0
```

---

## 4. Monte Carlo Methods

### 4.1 Monte Carlo Prediction

```python
class MonteCarloPrediction:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        
        n_states = env.observation_space.n
        self.V = np.zeros(n_states)
        self.returns = {s: [] for s in range(n_states)}  # record returns per state
        
    def run_episode(self, policy):
        """Run one episode"""
        state = self.env.reset()
        done = False
        episode = []
        
        while not done:
            action = np.random.choice(
                range(self.env.action_space.n), 
                p=policy[state]
            )
            next_state, reward, done, _ = self.env.step(action)
            
            episode.append((state, action, reward))
            state = next_state
            
        return episode
    
    def train(self, num_episodes=10000, policy=None):
        """Monte Carlo learning"""
        if policy is None:
            policy = np.ones([self.env.observation_space.n, 
                            self.env.action_space.n]) / self.env.action_space.n
        
        for _ in range(num_episodes):
            episode = self.run_episode(policy)
            
            # Compute returns
            G = 0
            for t, (state, action, reward) in enumerate(episode):
                G = self.gamma * G + reward
                
                # First-visit MC
                if state not in [s for s, _, _ in episode[:t]]:
                    self.returns[state].append(G)
                    self.V[state] = np.mean(self.returns[state])
```

---

## 5. Temporal Difference Learning

### 5.1 TD(0)

```python
class TDZero:
    def __init__(self, env, alpha=0.1, gamma=0.9):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma
        
        n_states = env.observation_space.n
        self.V = np.zeros(n_states)
        
    def train(self, num_steps=10000, policy=None):
        """TD(0) learning"""
        state = self.env.reset()
        
        for _ in range(num_steps):
            if policy is None:
                action = np.random.randint(self.env.action_space.n)
            else:
                action = np.random.choice(
                    range(self.env.action_space.n),
                    p=policy[state]
                )
            
            next_state, reward, done, _ = self.env.step(action)
            
            # TD update
            td_target = reward + self.gamma * self.V[next_state]
            td_error = td_target - self.V[state]
            self.V[state] += self.alpha * td_error
            
            if done:
                state = self.env.reset()
            else:
                state = next_state
```

### 5.2 SARSA (On-Policy TD Control)

```python
class SARSA:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        self.Q = np.zeros([n_states, n_actions])
        
    def choose_action(self, state):
        """ε-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes=500):
        """SARSA training"""
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                
                # SARSA update
                td_target = reward + self.gamma * self.Q[next_state, next_action]
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                
                state = next_state
                action = next_action
                
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
```

### 5.3 Q-Learning (Off-Policy TD Control)

```python
class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        self.Q = np.zeros([n_states, n_actions])
        
    def choose_action(self, state):
        """ε-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes=500):
        """Q-learning training"""
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Q-learning update (uses max, not the actual action)
                td_target = reward + self.gamma * np.max(self.Q[next_state])
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                
                state = next_state
                
            self.epsilon = max(0.01, self.epsilon * 0.995)
```

---

## 6. Value Function Approximation

### 6.1 Deep Q-Network (DQN)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001):
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
    def select_action(self, state, training=True):
        """ε-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.policy_net.output_dim)
        else:
            with torch.no_grad():
                q_values = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
                return q_values.argmax().item()
    
    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target(self):
        """Update the target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
```

---

## 7. Policy Gradient Methods

### 7.1 REINFORCE

```python
class REINFORCE:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.01):
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    
    def update(self, log_probs, rewards, gamma=0.99):
        """Policy gradient update"""
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Policy gradient
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### 7.2 Actor-Critic

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Actor: policy network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: value network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value

class A2CAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.model(state)
        
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute TD targets
        with torch.no_grad():
            _, next_value = self.model(next_states)
            td_target = rewards + self.gamma * next_value.squeeze() * (1 - dones)
        
        # Current value
        _, current_value = self.model(states)
        
        # Actor loss (policy gradient + advantage)
        probs, values = self.model(states)
        action_dist = torch.distributions.Categorical(probs)
        log_probs = action_dist.log_prob(actions)
        
        advantage = td_target - values.squeeze()
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = nn.MSELoss()(values.squeeze(), td_target.detach())
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

---

## 8. DQN and Its Variants

### 8.1 Double DQN

Addresses Q-value overestimation:

```python
class DoubleDQN:
    def __init__(self, state_dim, action_dim):
        self.online_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        
        # Use the online network to select actions, the target network to evaluate
        # ...
        
    def update(self):
        # Double DQN update
        with torch.no_grad():
            # Select actions with the online network
            next_action = self.online_net(next_states).argmax(1)
            # Evaluate with the target network
            next_q = self.target_net(next_states).gather(1, next_action.unsqueeze(1))
```

### 8.2 Dueling DQN

Separates the value and advantage functions:

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # State value function
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage function
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        
        # Q = V + A - mean(A)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q
```

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
3. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning.

---

*This chapter is continuously updated...*
