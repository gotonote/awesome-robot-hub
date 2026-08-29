# RL-Based Motion Control

> Updated: 2026-02-26

## 1. Reinforcement Learning Overview

### 1.1 What Is Reinforcement Learning?

Reinforcement Learning (RL) is an important branch of machine learning — the agent learns an optimal policy by interacting with the environment to maximize cumulative reward.

```
┌─────────────────────────────────────────────────────────────┐
│                    强化学习基本框架                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌─────────┐     ┌─────────┐     ┌─────────┐             │
│    │  智能体  │────▶│  环境   │────▶│  智能体  │             │
│    │ (Agent) │◀────│(Environment)◀────│(Agent) │             │
│    └─────────┘     └─────────┘     └─────────┘             │
│         │               │               │                  │
│         ▼               ▼               ▼                  │
│    ┌─────────┐     ┌─────────┐     ┌─────────┐             │
│    │ 动作(a) │     │状态(s)  │     │奖励(r)  │             │
│    └─────────┘     └─────────┘     └─────────┘             │
│                                                             │
│  核心要素：                                                  │
│  • 状态 (s)：智能体对环境的观察                               │
│  • 动作 (a)：智能体可采取的行为                              │
│  • 奖励 (r)：环境对动作的反馈                                │
│  • 策略 π(a|s)：状态到动作的映射                            │
│  • 价值函数 V(s) / Q(s,a)：长期收益估计                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Core elements: state (s) — the agent's observation; action (a) — behaviors the agent can take; reward (r) — environment feedback; policy π(a|s) — mapping from states to actions; value functions V(s) / Q(s,a) — long-term return estimates)*

### 1.2 RL vs. Supervised Learning

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|---------------------|------------------------|
| Data | Labeled data | Interaction data |
| Goal | Predict labels | Maximize cumulative reward |
| Feedback | Immediate, accurate | Delayed and sparse |
| Exploration | Not required | Required |

---

## 2. Markov Decision Processes

### 2.1 MDP Definition

A Markov Decision Process (MDP) is the mathematical framework of reinforcement learning.

**5-tuple (S, A, P, R, γ)**:
- **S**: state space
- **A**: action space
- **P(s'|s, a)**: state transition probability
- **R(s, a, s')**: reward function
- **γ**: discount factor (0 ≤ γ < 1)

### 2.2 Returns & Value Functions

**Cumulative return**:
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

**State value function**:
$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]$$

**Action value function**:
$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$$

### 2.3 Bellman Equations

**Bellman optimality equation**:
$$V^*(s) = \max_a \sum_{s'} P(s'|s, a)[R(s,a,s') + \gamma V^*(s')]$$

---

## 3. Classical RL Algorithms

### 3.1 Value Iteration & Policy Iteration

**Value iteration**:
```python
# Value iteration
for iteration in range(max_iterations):
    for s in states:
        V[s] = max_a sum_{s'} P(s'|s,a) * (R(s,a,s') + gamma * V[s'])
```

**Policy iteration**:
1. Policy evaluation: compute the value function of the current policy
2. Policy improvement: greedily choose the best action
3. Iterate until convergence

### 3.2 Q-Learning

**Q-learning is a model-free, off-policy RL algorithm**:

```python
# Q-Learning
Q = initialize_table(states, actions)

for episode in range(num_episodes):
    s = env.reset()
    done = False
    
    while not done:
        # ε-greedy exploration
        if random.random() < epsilon:
            a = random.choice(actions)
        else:
            a = argmax(Q[s])
        
        s', r, done = env.step(a)
        
        # Q-value update
        Q[s, a] = Q[s, a] + alpha * (r + gamma * max(Q[s']) - Q[s, a])
        s = s'
```

### 3.3 Deep Q-Network (DQN)

**DQN approximates the Q function with a deep neural network**:

```
┌─────────────────────────────────────────────────────────────┐
│                      DQN网络结构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入状态 s                                                │
│        │                                                    │
│        ▼                                                    │
│   ┌─────────┐                                               │
│   │ CNN/    │  提取特征                                     │
│   │ MLP     │                                               │
│   └─────────┘                                               │
│        │                                                    │
│        ▼                                                    │
│   ┌─────────┐                                               │
│   │ FC层    │  输出Q(s,a) for all actions                 │
│   └─────────┘                                               │
│                                                             │
│   关键技术：                                                 │
│   • 经验回放 (Experience Replay)                            │
│   • 目标网络 (Target Network)                               │
│   • 损失函数：MSE(Q_target, Q_current)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Key techniques: experience replay, target network, loss = MSE(Q_target, Q_current))*

---

## 4. Policy Gradient Methods

### 4.1 REINFORCE

**REINFORCE is a Monte-Carlo-sampled policy gradient method**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t]$$

```python
# REINFORCE
def REINFORCE(env, num_episodes):
    policy_net = PolicyNetwork()
    optimizer = Adam(policy_net.parameters(), lr=3e-4)
    
    for episode in range(num_episodes):
        trajectory = []
        states, actions, rewards = [], [], []
        s = env.reset()
        
        while True:
            probs = policy_net(s)
            a = Categorical(probs).sample()
            s', r, done = env.step(a)
            
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s'
            
            if done:
                break
        
        # Compute returns
        returns = compute_returns(rewards)
        
        # Policy gradient update
        for s, a, G in zip(states, actions, returns):
            probs = policy_net(s)
            log_prob = torch.log(probs[a])
            loss = -log_prob * G
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 4.2 Actor-Critic Methods

**Actor-Critic combines value function approximation with policy gradients**:

```
┌─────────────────────────────────────────────────────────────┐
│                  Actor-Critic 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────┐          │
│   │              环境                            │          │
│   └─────────────────────────────────────────────┘          │
│                         │                                   │
│                         ▼ s, r                              │
│   ┌─────────────┐    ┌─────────────┐                       │
│   │   Actor     │◀───│   Critic    │                       │
│   │ (策略网络)  │    │ (价值网络)  │                       │
│   │ π(a|s; θ)   │    │ V(s; w)     │                       │
│   └─────────────┘    └─────────────┘                       │
│        │                     │                              │
│        └──────────┬───────────┘                              │
│                   ▼ a                                       │
│                                                             │
│   Actor: 负责选择动作                                       │
│   Critic: 评估当前策略的价值                                │
│   优势：方差降低，收敛更快                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Actor: chooses actions; Critic: evaluates the value of the current policy; advantage: lower variance, faster convergence)*

---

## 5. Deep RL Algorithms

### 5.1 DDPG (Deep Deterministic Policy Gradient)

**An off-policy algorithm for continuous action spaces**:

```python
# DDPG core
class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = copy(self.actor)
        self.target_critic = copy(self.critic)
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Critic update
        target_q = rewards + gamma * self.target_critic(next_states, 
                                                         self.target_actor(next_states))
        current_q = self.critic(states, actions)
        critic_loss = MSE(current_q, target_q)
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Soft update of target networks
        soft_update(self.target_actor, self.actor)
        soft_update(self.target_critic, self.critic)
```

### 5.2 PPO (Proximal Policy Optimization)

**PPO improves stability by limiting the policy update magnitude**:

```python
# PPO loss
def ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    
    # Advantage function
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    
    return -torch.min(loss1, loss2).mean()
```

### 5.3 SAC (Soft Actor-Critic)

**SAC is a maximum-entropy RL algorithm with better exploration and stability**:

```python
# SAC value update
def sac_update(batch):
    # Automatic entropy coefficient tuning
    alpha = torch.exp(self.log_alpha)
    
    # Q function update
    q1, q2 = self.critic1(states, actions), self.critic2(states, actions)
    min_q = torch.min(q1, q2)
    
    # Entropy regularization
    policy_loss = (alpha * log_probs - min_q).mean()
    
    # Value function update
    v_target = min_q - alpha * log_probs
    critic_loss = MSE(Q(states, actions), v_target)
```

---

## 6. RL for Robot Motion Control

### 6.1 Motion Control Tasks

| Task | State Space | Action Space | Reward Design |
|------|-------------|--------------|---------------|
| Bipedal walking | Joint angles + angular velocities | Joint torques | Forward velocity + posture stability |
| Arm manipulation | End-effector position + goal | Joint velocities | Reach goal + avoid obstacles |
| Drone flight | Position + pose + velocity | Propeller thrusts | Trajectory tracking + energy |

### 6.2 Sim-to-Real Transfer

**Domain randomization**:
```python
# Randomize physics parameters during training
def randomize_domain():
    params = {
        'mass': uniform(0.8, 1.2) * nominal_mass,
        'friction': uniform(0.5, 1.5) * nominal_friction,
        'delay': uniform(0, 0.01),
        'noise': uniform(0, 0.05)
    }
    return params
```

### 6.3 Reward Design Principles

**Design notes**:
1. **Sparse rewards**: reward only at task completion → hard to learn
2. **Dense rewards**: reward every step → may lead to suboptimal behavior
3. **Shaped rewards**: combine sparse and dense → balanced

**Example: bipedal walking reward**:
```python
def compute_reward(state, action, next_state):
    r_vel = forward_velocity * 2.0  # forward velocity reward
    r_upright = posture_stability * 1.5  # posture stability reward
    r_smooth = -acceleration**2 * 0.1  # smoothness reward
    r_height = -|hip_height - target|**2  # height keeping
    r_energy = -energy_consumption * 0.01  # energy penalty
    
    return r_vel + r_upright + r_smooth + r_height + r_energy
```

---

## 7. Hands-On: Training a Robot to Walk

### 7.1 Environment Setup

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Create the bipedal robot environment
env = gym.make('Humanoid-v4')

# Create a PPO agent
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)
```

### 7.2 Training & Evaluation

```python
# Train
model.learn(total_timesteps=1_000_000)

# Save the model
model.save('humanoid_walk')

# Evaluate
eval_env = gym.make('Humanoid-v4')
episode_rewards = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Average reward: {np.mean(episode_rewards):.2f}")
```

### 7.3 Deployment to a Real Robot

```python
# Load the trained policy
policy = PPO.load('humanoid_walk')

# Real-time control loop
while True:
    observation = robot.get_state()
    action, _ = policy.predict(observation, deterministic=True)
    robot.execute_action(action)
    rate.sleep(50)  # 50Hz control frequency
```

---

## 8. Summary & Outlook

### 8.1 Advantages of RL in Motion Control

1. **End-to-end learning**: directly from perception to control
2. **Handles complex dynamics**: no precise modeling required
3. **Adapts to dynamic environments**: online learning ability

### 8.2 Current Challenges

1. **Sample efficiency**: requires large amounts of interaction data
2. **Safety**: dangerous behaviors may occur during training
3. **Interpretability**: black-box policies are hard to debug

### 8.3 Frontier Directions

- **Offline reinforcement learning**: learning from existing data
- **Multi-task learning**: one policy, many tasks
- **Embodied intelligence**: combining LLM reasoning

---

## Resources

- **Book**: *Reinforcement Learning: An Introduction* by Sutton & Barto
- **Course**: Stanford CS234 Reinforcement Learning
- **Frameworks**: Stable-Baselines3, RLlib

---

*This guide is auto-generated and updated by AI*
