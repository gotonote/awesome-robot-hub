# Hands-On Tutorial: Robot Control Algorithm Implementation

> This chapter provides complete implementation examples of robot control algorithms, from basics to advanced.

## 1. Environment Setup

### 1.1 Installing Dependencies

```bash
pip install numpy torch gymnasium pybullet matplotlib
```

### 1.2 Basic Environment Configuration

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict

# Set the random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)
```

---

## 2. A Basic Robot Simulation Environment

### 2.1 Creating a Simple Environment with PyBullet

```python
import pybullet as p
import pybullet_data
import numpy as np

class SimpleRobotEnv:
    """
    A simple 2D robot environment
    """
    def __init__(self, render=False):
        # Connect to the client
        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load the environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        
        # Joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = [i for i in range(self.num_joints) 
                            if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        
        # State space
        self.observation_dim = len(self.joint_indices) * 2  # positions + velocities
        self.action_dim = len(self.joint_indices)
        
    def reset(self) -> np.ndarray:
        """Reset the environment"""
        # Reset joint positions
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, 0)
        
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute an action"""
        # Simple PD control
        for i, idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id, idx,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=100
            )
        
        p.stepSimulation()
        
        obs = self._get_obs()
        reward = self._compute_reward()
        done = False
        info = {}
        
        return obs, reward, done, info
    
    def _get_obs(self) -> np.ndarray:
        """Get the observation"""
        obs = []
        for idx in self.joint_indices:
            state = p.getJointState(self.robot_id, idx)
            obs.extend([state[0], state[1]])  # position, velocity
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self) -> float:
        """Compute the reward"""
        # Simplified reward function
        return 0.0
    
    def close(self):
        p.disconnect(self.client)
```

---

## 3. Reinforcement Learning Implementation

### 3.1 PPO

```python
class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, obs_dim, action_dim, max_size=10000):
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        
    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size) -> Tuple:
        indices = np.random.randint(0, self.size, batch_size)
        return (
            self.obs_buf[indices],
            self.action_buf[indices],
            self.reward_buf[indices],
            self.next_obs_buf[indices],
            self.done_buf[indices]
        )


class ActorCritic(nn.Module):
    """Actor-Critic network"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared feature extraction layer
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor: policy network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # output [-1, 1]
        )
        
        # Critic: value network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs):
        features = self.feature(obs)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, value


class PPOAgent:
    """PPO agent"""
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, 
                 clip_eps=0.2, epochs=10):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        
        # Networks
        self.policy = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Buffer
        self.buffer = ReplayBuffer(obs_dim, action_dim)
        
    def select_action(self, obs, training=True):
        """Select an action"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, value = self.policy(obs_tensor)
            
        if training:
            action = action_mean + torch.randn_like(action_mean) * 0.1
        else:
            action = action_mean
            
        return action.squeeze(0).numpy(), value.item()
    
    def update(self, batch_size=256):
        """Update the policy"""
        # Collect enough data
        if self.buffer.size < batch_size:
            return
        
        # Sample data
        obs, action, reward, next_obs, done = self.buffer.sample(batch_size)
        
        obs = torch.FloatTensor(obs)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.FloatTensor(done)
        
        # Compute GAE
        with torch.no_grad():
            _, values = self.policy(obs)
            _, next_values = self.policy(next_obs)
            
            td_target = reward + self.gamma * (1 - done) * next_values
            advantage = td_target - values
            
        # PPO update
        for _ in range(self.epochs):
            action_mean, values = self.policy(obs)
            
            # Policy loss
            ratio = torch.exp(action - action_mean)  # simplified
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, td_target.detach())
            
            # Update
            self.optimizer.zero_grad()
            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            self.optimizer.step()
    
    def train(self, env, num_episodes=500):
        """Training loop"""
        for episode in range(num_episodes):
            obs = env.reset()
            total_reward = 0
            
            while True:
                action, _ = self.select_action(obs)
                next_obs, reward, done, _ = env.step(action)
                
                self.buffer.add(obs, action, reward, next_obs, done)
                
                obs = next_obs
                total_reward += reward
                
                if done:
                    break
            
            # Update
            self.update()
            
            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}")


# Usage example
if __name__ == "__main__":
    # Create the environment
    env = SimpleRobotEnv()
    
    # Create the agent
    agent = PPOAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        lr=3e-4
    )
    
    # Train
    agent.train(env, num_episodes=100)
    
    env.close()
```

---

## 4. Imitation Learning Implementation

### 4.1 Behavior Cloning

```python
class BehaviorCloning:
    """Behavior cloning"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        
    def train(self, observations, actions, epochs=100):
        """Train"""
        observations = torch.FloatTensor(observations)
        actions = torch.FloatTensor(actions)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            pred_actions = self.policy(observations)
            loss = F.mse_loss(pred_actions, actions)
            
            loss.backward()
            self.optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                
    def predict(self, obs):
        """Predict an action"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = self.policy(obs_tensor)
        return action.squeeze(0).numpy()


# Load demonstration data example
def load_demonstration(file_path):
    """Load expert demonstration data"""
    data = np.load(file_path)
    observations = data['observations']
    actions = data['actions']
    return observations, actions


# Train behavior cloning
if __name__ == "__main__":
    # Assume demonstration data exists
    # observations, actions = load_demonstration("demo.npy")
    
    # Simulated data
    observations = np.random.randn(1000, 10).astype(np.float32)
    actions = np.random.randn(1000, 3).astype(np.float32)
    
    bc = BehaviorCloning(obs_dim=10, action_dim=3)
    bc.train(observations, actions, epochs=200)
```

---

## 5. Complete Project: Robot Arm Grasping

### 5.1 Project Structure

```
robot_grasp_project/
├── env/
│   └── grasp_env.py      # grasping environment
├── models/
│   ├── actor_critic.py    # policy network
│   └── world_model.py    # world model
├── agents/
│   └── ppo_agent.py      # PPO agent
├── utils/
│   └── replay_buffer.py  # utility functions
├── train.py              # training script
└── evaluate.py           # evaluation script
```

### 5.2 Training Script

```python
#!/usr/bin/env python3
"""
Robot arm grasping training script
"""

import numpy as np
import torch
from grasp_env import GraspEnv
from ppo_agent import PPOAgent
from replay_buffer import ReplayBuffer


def train():
    # Create the environment
    env = GraspEnv(render=True)
    
    # Create the agent
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        gamma=0.99
    )
    
    # Training parameters
    num_episodes = 2000
    max_steps = 500
    
    # Training loop
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select an action
            action, _ = agent.select_action(obs)
            
            # Execute
            next_obs, reward, done, info = env.step(action)
            
            # Store
            agent.buffer.add(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            
            if done:
                break
        
        # Update the policy
        agent.update()
        
        # Logging
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}")
    
    # Save the model
    torch.save(agent.policy.state_dict(), "policy.pth")
    print("Training complete!")
    
    env.close()


if __name__ == "__main__":
    train()
```

### 5.3 Evaluation Script

```python
#!/usr/bin/env python3
"""
Evaluation script
"""

import torch
import numpy as np
from grasp_env import GraspEnv
from actor_critic import ActorCritic


def evaluate(policy_path, num_episodes=20):
    env = GraspEnv(render=True)
    
    # Load the policy
    policy = ActorCritic(env.obs_dim, env.action_dim)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()
    
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_mean, _ = policy(obs_tensor)
                action = action_mean.numpy()[0]
            
            obs, reward, done, info = env.step(action)
            
            if info.get('success', False):
                success_count += 1
                break
    
    success_rate = success_count / num_episodes
    print(f"Success Rate: {success_rate:.2%}")
    
    env.close()


if __name__ == "__main__":
    evaluate("policy.pth")
```

---

## 6. Advanced: Using MuJoCo

```python
import mujoco
import mujoco.viewer

class MuJoCoRobot:
    """MuJoCo robot environment"""
    def __init__(self, model_path):
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Launch the viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
    def step(self, ctrl):
        """Execute one step"""
        self.data.ctrl = ctrl
        mujoco.mj_step(self.model, self.data)
        
        # Get the observation
        obs = self._get_obs()
        reward = self._compute_reward()
        done = False
        
        return obs, reward, done, {}
    
    def _get_obs(self):
        """Get joint positions and velocities"""
        return np.concatenate([
            self.data.qpos,  # positions
            self.data.qvel   # velocities
        ])
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()
    
    def close(self):
        self.viewer.close()
```

---

## 7. Debugging & Visualization

### 7.1 Plotting Training Curves

```python
import matplotlib.pyplot as plt

def plot_training_curve(rewards, save_path="training_curve.png"):
    """Plot the training curve"""
    plt.figure(figsize=(10, 5))
    
    # Smoothing
    window = 50
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
```

### 7.2 Action Visualization

```python
def visualize_actions(actions, joint_names):
    """Visualize joint actions"""
    plt.figure(figsize=(12, 4))
    
    for i, name in enumerate(joint_names):
        plt.subplot(1, len(joint_names), i+1)
        plt.plot(actions[:, i])
        plt.title(name)
        plt.xlabel("Time Step")
        plt.ylabel("Position")
    
    plt.tight_layout()
    plt.savefig("actions.png")
    plt.show()
```

---

## 8. Summary

```
┌─────────────────────────────────────────────────────────┐
│                   实战项目清单                           │
├─────────────────────────────────────────────────────────┤
│  ✓ 基础: PPO 机器人控制                                  │
│  ✓ 基础: 行为克隆                                        │
│  ✓ 进阶: 世界模型 + MBRL                                 │
│  ✓ 进阶: 扩散策略                                        │
│  ✓ 工具: PyBullet / MuJoCo 仿真                         │
│  ✓ 可视化: 训练曲线、动作轨迹                             │
└─────────────────────────────────────────────────────────┘
```

*(Checklist: basics — PPO robot control, behavior cloning; advanced — world models + MBRL, diffusion policies; tools — PyBullet/MuJoCo simulation; visualization — training curves, action trajectories.)*

## 9. Next Steps

- Try more robot tasks
- Implement Dreamer / World Models
- Explore Diffusion Policy
- Incorporate visual input

---

*Happy learning!*
