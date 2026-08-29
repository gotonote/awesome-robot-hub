# Obstacle Avoidance

Obstacle avoidance is a core problem in robot navigation — ensuring the robot can safely bypass obstacles to reach its goal.

## 📋 Contents

- [1. Overview](#1-overview)
- [2. Dynamic Window Approach (DWA)](#2-dynamic-window-approach-dwa)
- [3. Artificial Potential Field](#3-artificial-potential-field)
- [4. Model Predictive Control (MPC)](#4-model-predictive-control-mpc)
- [5. Deep-Learning-Based Obstacle Avoidance](#5-deep-learning-based-obstacle-avoidance)
- [6. Algorithm Comparison](#6-algorithm-comparison)

---

## 1. Overview

### 1.1 The Obstacle Avoidance Problem

```
┌─────────────────────────────────────────────────────────────┐
│                    避障问题                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  目标：在保证安全的前提下，向目标前进                        │
│                                                             │
│  输入：                                                      │
│  • 机器人当前状态（位置、速度）                              │
│  • 传感器感知的障碍物信息                                    │
│  • 目标位置                                                 │
│                                                             │
│  输出：                                                      │
│  • 安全的速度指令（v, ω）或加速度                           │
│                                                             │
│  约束：                                                      │
│  • 运动学约束（最大速度、加速度）                            │
│  • 动力学约束                                               │
│  • 安全距离约束                                             │
│                                                             │
│  挑战：                                                      │
│  • 传感器噪声和不确定性                                      │
│  • 动态障碍物                                               │
│  • 实时性要求                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Goal: advance toward the target while ensuring safety. Input: current robot state (position, velocity), obstacle information from sensors, goal position. Output: safe velocity commands (v, ω) or accelerations. Constraints: kinematic (max velocity/acceleration), dynamic, safe distance. Challenges: sensor noise and uncertainty, dynamic obstacles, real-time requirements.)*

### 1.2 Taxonomy of Avoidance Methods

| Type | Representative | Features |
|------|----------------|----------|
| **Reactive** | Potential field, VFH | Fast, locally optimal |
| **Predictive** | DWA, MPC | Considers the motion model |
| **Learning-based** | Neural networks, RL | Strong generalization |

---

## 2. Dynamic Window Approach (DWA)

### 2.1 Algorithm Principle

DWA achieves obstacle avoidance by searching for the optimal velocity command in velocity space.

```
┌─────────────────────────────────────────────────────────────┐
│                    DWA 核心思想                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 确定搜索空间（动态窗口）：                               │
│     • 考虑最大速度/加速度约束                               │
│     • 考虑当前速度                                          │
│     • 只考虑一个周期内可达的速度                            │
│                                                             │
│  2. 对每个候选速度进行评分：                                 │
│     G(v, ω) = σ(α*heading + β*dist + γ*velocity)          │
│                                                             │
│     • heading：朝向目标的程度                               │
│     • dist：到最近障碍物的距离                              │
│     • velocity：前进速度（越快越好）                        │
│                                                             │
│  3. 选择评分最高的速度执行                                   │
│                                                             │
│  4. 循环执行                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(1. Determine the search space (dynamic window): consider max velocity/acceleration constraints and the current velocity; only consider velocities reachable within one cycle. 2. Score each candidate velocity: G(v, ω) = σ(α·heading + β·dist + γ·velocity), where heading is how much the robot heads toward the goal, dist is the distance to the nearest obstacle, and velocity rewards forward speed. 3. Execute the highest-scoring velocity. 4. Loop.)*

### 2.2 Computing the Dynamic Window

```
The dynamic window Vs ⊂ V is the set of reachable velocities:

Vs = {(v, ω) | v ∈ [v_curr - a_dec*Δt, v_curr + a_acc*Δt]
           ∧ ω ∈ [ω_curr - α_dec*Δt, ω_curr + α_acc*Δt]
           ∧ v ∈ [0, v_max] ∧ ω ∈ [-ω_max, ω_max]}

where:
• v_curr, ω_curr: current linear and angular velocities
• a_acc, a_dec: linear acceleration and deceleration limits
• α_acc, α_dec: angular acceleration limits
• Δt: time step
```

### 2.3 Trajectory Prediction

For a given velocity (v, ω), predict the future trajectory:

```
Assume uniform circular motion:

x(t+Δt) = x(t) + v * cos(θ) * Δt
y(t+Δt) = y(t) + v * sin(θ) * Δt
θ(t+Δt) = θ(t) + ω * Δt
```

### 2.4 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class DynamicWindowApproach:
    """Dynamic Window Approach obstacle avoidance"""
    
    def __init__(self, 
                 max_speed: float = 1.0,
                 min_speed: float = 0.0,
                 max_yaw_rate: float = 40.0 * np.pi / 180.0,
                 max_accel: float = 0.5,
                 max_delta_yaw_rate: float = 40.0 * np.pi / 180.0,
                 dt: float = 0.1,
                 predict_time: float = 3.0,
                 heading_cost_gain: float = 0.15,
                 speed_cost_gain: float = 1.0,
                 obstacle_cost_gain: float = 1.0,
                 robot_radius: float = 0.3):
        """
        Initialize DWA parameters.
        
        Args:
            max_speed: max linear velocity (m/s)
            min_speed: min linear velocity (m/s)
            max_yaw_rate: max angular velocity (rad/s)
            max_accel: max acceleration (m/s²)
            max_delta_yaw_rate: max angular acceleration (rad/s²)
            dt: time step (s)
            predict_time: trajectory prediction time (s)
            heading_cost_gain: heading cost weight
            speed_cost_gain: speed cost weight
            obstacle_cost_gain: obstacle cost weight
            robot_radius: robot radius (m)
        """
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_yaw_rate = max_yaw_rate
        self.max_accel = max_accel
        self.max_delta_yaw_rate = max_delta_yaw_rate
        self.dt = dt
        self.predict_time = predict_time
        self.heading_cost_gain = heading_cost_gain
        self.speed_cost_gain = speed_cost_gain
        self.obstacle_cost_gain = obstacle_cost_gain
        self.robot_radius = robot_radius
        
    def calc_dynamic_window(self, v: float, ω: float) -> Tuple[float, float, float, float]:
        """
        Compute the dynamic window.
        
        Args:
            v: current linear velocity
            ω: current angular velocity
        
        Returns:
            (v_min, v_max, ω_min, ω_max)
        """
        # Consider acceleration limits
        Vs = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        
        # Consider dynamic constraints
        Vd = [v - self.max_accel * self.dt,
              v + self.max_accel * self.dt,
              ω - self.max_delta_yaw_rate * self.dt,
              ω + self.max_delta_yaw_rate * self.dt]
        
        # Final window
        v_min = max(Vs[0], Vd[0])
        v_max = min(Vs[1], Vd[1])
        ω_min = max(Vs[2], Vd[2])
        ω_max = min(Vs[3], Vd[3])
        
        return v_min, v_max, ω_min, ω_max
    
    def predict_trajectory(self, x: float, y: float, θ: float, 
                          v: float, ω: float) -> np.ndarray:
        """
        Predict the trajectory.
        
        Args:
            x, y, θ: current pose
            v, ω: velocity command
        
        Returns:
            trajectory array, shape (N, 3) - (x, y, θ)
        """
        trajectory = np.array([[x, y, θ]])
        time = 0
        
        while time <= self.predict_time:
            # Kinematic model
            x += v * np.cos(θ) * self.dt
            y += v * np.sin(θ) * self.dt
            θ += ω * self.dt
            
            trajectory = np.vstack([trajectory, [x, y, θ]])
            time += self.dt
        
        return trajectory
    
    def calc_heading_cost(self, trajectory: np.ndarray, goal: np.ndarray) -> float:
        """Compute the heading cost (smaller angle is better)"""
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        angle_to_goal = np.arctan2(dy, dx)
        
        angle_diff = abs(angle_to_goal - trajectory[-1, 2])
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        return abs(angle_diff)
    
    def calc_obstacle_cost(self, trajectory: np.ndarray, obstacles: np.ndarray) -> float:
        """
        Compute the obstacle cost.
        
        Args:
            trajectory: predicted trajectory
            obstacles: obstacle positions, shape (N, 2)
        
        Returns:
            cost (higher when closer)
        """
        min_dist = float('inf')
        
        for point in trajectory:
            for obs in obstacles:
                dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                if dist < self.robot_radius:
                    return float('inf')  # collision
                
                if dist < min_dist:
                    min_dist = dist
        
        # Closer is costlier
        return 1.0 / min_dist if min_dist > 0 else float('inf')
    
    def calc_speed_cost(self, v: float) -> float:
        """Compute the speed cost (faster is better)"""
        return self.max_speed - v
    
    def calc_total_cost(self, trajectory: np.ndarray, v: float, 
                       goal: np.ndarray, obstacles: np.ndarray) -> float:
        """Compute the total cost"""
        heading_cost = self.calc_heading_cost(trajectory, goal)
        obstacle_cost = self.calc_obstacle_cost(trajectory, obstacles)
        speed_cost = self.calc_speed_cost(v)
        
        return (self.heading_cost_gain * heading_cost + 
                self.obstacle_cost_gain * obstacle_cost + 
                self.speed_cost_gain * speed_cost)
    
    def plan(self, state: np.ndarray, goal: np.ndarray, 
            obstacles: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Run DWA planning.
        
        Args:
            state: current state [x, y, θ, v, ω]
            goal: goal position [x, y]
            obstacles: obstacle array (N, 2)
        
        Returns:
            (v_cmd, ω_cmd, best_trajectory)
        """
        x, y, θ, v, ω = state
        
        # Compute the dynamic window
        v_min, v_max, ω_min, ω_max = self.calc_dynamic_window(v, ω)
        
        # Search for the best velocity
        min_cost = float('inf')
        best_v, best_ω = 0.0, 0.0
        best_trajectory = None
        
        # Velocity sampling
        v_samples = np.linspace(v_min, v_max, 10)
        ω_samples = np.linspace(ω_min, ω_max, 20)
        
        for v_sample in v_samples:
            for ω_sample in ω_samples:
                # Predict the trajectory
                trajectory = self.predict_trajectory(x, y, θ, v_sample, ω_sample)
                
                # Compute the cost
                cost = self.calc_total_cost(trajectory, v_sample, goal, obstacles)
                
                if cost < min_cost:
                    min_cost = cost
                    best_v = v_sample
                    best_ω = ω_sample
                    best_trajectory = trajectory
        
        return best_v, best_ω, best_trajectory


# ============ Usage example ============
def simulate_dwa():
    """DWA simulation example"""
    # Initialize
    dwa = DynamicWindowApproach(
        max_speed=1.0,
        max_yaw_rate=np.pi/3,
        robot_radius=0.3
    )
    
    # Initial state [x, y, θ, v, ω]
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Goal
    goal = np.array([8.0, 8.0])
    
    # Obstacles
    obstacles = np.array([
        [3.0, 3.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [6.0, 6.0],
        [2.0, 5.0],
        [7.0, 3.0],
    ])
    
    # Simulation
    trajectory = [state[:3].copy()]
    
    for _ in range(500):
        # Plan
        v_cmd, ω_cmd, pred_traj = dwa.plan(state, goal, obstacles)
        
        # Update state
        state[0] += v_cmd * np.cos(state[2]) * dwa.dt
        state[1] += v_cmd * np.sin(state[2]) * dwa.dt
        state[2] += ω_cmd * dwa.dt
        state[3] = v_cmd
        state[4] = ω_cmd
        
        trajectory.append(state[:3].copy())
        
        # Check whether the goal is reached
        if np.sqrt((state[0] - goal[0])**2 + (state[1] - goal[1])**2) < 0.3:
            print("Goal reached!")
            break
    
    # Visualize
    plt.figure(figsize=(10, 10))
    
    # Draw obstacles
    for obs in obstacles:
        circle = plt.Circle(obs, 0.3, color='red', alpha=0.6)
        plt.gca().add_patch(circle)
    
    # Draw the trajectory
    traj = np.array(trajectory)
    plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Robot Path')
    
    # Draw start and goal
    plt.plot(0, 0, 'go', markersize=15, label='Start')
    plt.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
    
    plt.xlim(-1, 10)
    plt.ylim(-1, 10)
    plt.legend()
    plt.grid(True)
    plt.title('DWA Obstacle Avoidance')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.savefig('dwa_simulation.png')
    plt.show()

if __name__ == "__main__":
    simulate_dwa()
```

---

## 3. Artificial Potential Field

### 3.1 Algorithm Principle (for Obstacle Avoidance)

The potential field method models the environment as a potential field:
- The goal generates an attractive field
- Obstacles generate repulsive fields

See [Path Planning - Potential Field](./路径规划.md#5-势场法) *(content in Chinese)*

### 3.2 Improvements

To address the local minimum problem:

```
┌─────────────────────────────────────────────────────────────┐
│                势场法改进策略                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 虚拟目标点                                              │
│     当检测到局部极小值时，临时添加虚拟目标                  │
│                                                             │
│  2. 随机扰动                                                │
│     在合力方向添加随机扰动跳出局部极小                      │
│                                                             │
│  3. 沿墙行走                                                │
│     当陷入时沿障碍物边界移动                                │
│                                                             │
│  4. 改进势场函数                                            │
│     使用旋转势场或其他形式的势场                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(1. Virtual goal point: temporarily add a virtual goal when a local minimum is detected. 2. Random perturbation: add random perturbation to the resultant direction to escape local minima. 3. Wall following: move along the obstacle boundary when stuck. 4. Improved potential functions: use rotational potential fields or other forms.)*

### 3.3 Potential Field with Obstacle Avoidance

```python
class ImprovedPotentialField:
    """Improved potential field obstacle avoidance"""
    
    def __init__(self, k_att=1.0, k_rep=100.0, rho_0=2.0):
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        self.stuck_counter = 0
        self.prev_position = None
        
    def detect_local_minimum(self, position: np.ndarray, force: np.ndarray) -> bool:
        """Detect whether stuck in a local minimum"""
        if self.prev_position is not None:
            # Check whether position change is small and force is also small
            pos_change = np.linalg.norm(position - self.prev_position)
            force_mag = np.linalg.norm(force)
            
            if pos_change < 0.01 and force_mag < 0.1:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        self.prev_position = position.copy()
        return self.stuck_counter > 10
    
    def escape_local_minimum(self, position: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        """Escape a local minimum - wall following"""
        # Find the nearest obstacle
        min_dist = float('inf')
        nearest_obs = None
        
        for obs in obstacles:
            dist = np.linalg.norm(position - obs)
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs
        
        if nearest_obs is not None:
            # Move along the obstacle tangent
            direction = position - nearest_obs
            tangent = np.array([-direction[1], direction[0]])
            return tangent / np.linalg.norm(tangent)
        
        return np.zeros(2)
    
    def compute_force(self, position: np.ndarray, goal: np.ndarray, 
                     obstacles: np.ndarray) -> np.ndarray:
        """Compute the improved potential field force"""
        # Attractive force
        f_att = self.k_att * (goal - position)
        
        # Repulsive force
        f_rep = np.zeros(2)
        for obs in obstacles:
            direction = position - obs
            dist = np.linalg.norm(direction)
            
            if dist < self.rho_0 and dist > 0:
                magnitude = self.k_rep * (1/dist - 1/self.rho_0) / (dist**2)
                f_rep += magnitude * direction / dist
        
        total_force = f_att + f_rep
        
        # Detect local minima
        if self.detect_local_minimum(position, total_force):
            total_force = self.escape_local_minimum(position, obstacles)
            self.stuck_counter = 0
        
        return total_force
```

---

## 4. Model Predictive Control (MPC)

### 4.1 MPC Principle

MPC achieves obstacle avoidance through receding-horizon optimization.

```
┌─────────────────────────────────────────────────────────────┐
│                    MPC 避障框架                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  优化问题：                                                  │
│                                                             │
│  min  Σ (||x_k - x_ref||²_Q + ||u_k||²_R)                  │
│  u                                                            │
│                                                             │
│  约束：                                                      │
│  • x_{k+1} = f(x_k, u_k)  (动力学模型)                      │
│  • ||p_k - p_obs|| ≥ r_safe  (安全距离)                    │
│  • u_min ≤ u_k ≤ u_max    (控制输入限制)                   │
│  • x_min ≤ x_k ≤ x_max    (状态限制)                       │
│                                                             │
│  步骤：                                                      │
│  1. 在预测时域内优化控制序列                                 │
│  2. 执行第一个控制输入                                       │
│  3. 更新状态，重复步骤1                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Optimization problem: min Σ(||x_k - x_ref||²_Q + ||u_k||²_R) subject to: dynamics model x_{k+1} = f(x_k, u_k); safety distance ||p_k - p_obs|| ≥ r_safe; control limits u_min ≤ u_k ≤ u_max; state limits x_min ≤ x_k ≤ x_max. Steps: 1. optimize the control sequence over the prediction horizon; 2. execute the first control input; 3. update the state and repeat step 1.)*

### 4.2 Python Implementation (Simplified)

```python
from scipy.optimize import minimize
import numpy as np

class MPCObstacleAvoidance:
    """MPC obstacle avoidance controller"""
    
    def __init__(self, 
                 horizon: int = 10,
                 dt: float = 0.1,
                 v_max: float = 1.0,
                 ω_max: float = np.pi/3,
                 Q: np.ndarray = None,
                 R: np.ndarray = None):
        """
        Args:
            horizon: prediction horizon
            dt: time step
            v_max: max linear velocity
            ω_max: max angular velocity
            Q: state cost matrix
            R: control cost matrix
        """
        self.horizon = horizon
        self.dt = dt
        self.v_max = v_max
        self.ω_max = ω_max
        
        # Default cost matrices
        self.Q = Q if Q is not None else np.diag([1.0, 1.0, 0.1])
        self.R = R if R is not None else np.diag([0.1, 0.1])
        
    def motion_model(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Kinematic model"""
        x, y, θ = state
        v, ω = u
        
        x_new = x + v * np.cos(θ) * self.dt
        y_new = y + v * np.sin(θ) * self.dt
        θ_new = θ + ω * self.dt
        
        return np.array([x_new, y_new, θ_new])
    
    def cost_function(self, u_seq: np.ndarray, 
                     state: np.ndarray, 
                     goal: np.ndarray,
                     obstacles: np.ndarray,
                     robot_radius: float = 0.3) -> float:
        """
        Cost function.
        
        Args:
            u_seq: control input sequence [v1, ω1, v2, ω2, ...]
            state: current state
            goal: goal state
            obstacles: obstacle list
            robot_radius: robot radius
        """
        cost = 0.0
        x = state.copy()
        
        for k in range(self.horizon):
            # Get the current control input
            u = u_seq[2*k:2*k+2]
            
            # Predict the next state
            x = self.motion_model(x, u)
            
            # State cost (track the goal)
            state_error = x[:2] - goal[:2]  # position only
            cost += state_error @ self.Q[:2, :2] @ state_error
            
            # Control cost
            cost += u @ self.R @ u
            
            # Obstacle cost
            for obs in obstacles:
                dist = np.sqrt((x[0] - obs[0])**2 + (x[1] - obs[1])**2)
                if dist < robot_radius:
                    cost += 1e6  # collision penalty
                elif dist < robot_radius + 0.5:
                    # Soft constraint
                    cost += 100.0 / (dist - robot_radius + 0.01)
        
        return cost
    
    def plan(self, 
             state: np.ndarray, 
             goal: np.ndarray, 
             obstacles: np.ndarray) -> Tuple[float, float]:
        """
        MPC planning.
        
        Returns:
            (v_cmd, ω_cmd)
        """
        # Initial guess
        u0 = np.zeros(2 * self.horizon)
        
        # Control input bounds
        bounds = []
        for _ in range(self.horizon):
            bounds.extend([
                (0, self.v_max),           # v
                (-self.ω_max, self.ω_max)  # ω
            ])
        
        # Optimize
        result = minimize(
            self.cost_function,
            u0,
            args=(state, goal, obstacles),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50}
        )
        
        # Return the first control input
        v_cmd = result.x[0]
        ω_cmd = result.x[1]
        
        return v_cmd, ω_cmd


# MPC simulation
def simulate_mpc():
    """MPC obstacle avoidance simulation"""
    mpc = MPCObstacleAvoidance(horizon=15, dt=0.1)
    
    state = np.array([0.0, 0.0, 0.0])
    goal = np.array([8.0, 8.0, 0.0])
    obstacles = np.array([[3.0, 3.0], [5.0, 4.0], [6.0, 6.0]])
    
    trajectory = [state.copy()]
    
    for _ in range(200):
        v, ω = mpc.plan(state, goal, obstacles)
        
        # Update state
        state = mpc.motion_model(state, [v, ω])
        trajectory.append(state.copy())
        
        if np.linalg.norm(state[:2] - goal[:2]) < 0.3:
            break
    
    return np.array(trajectory)

if __name__ == "__main__":
    traj = simulate_mpc()
    print(f"Trajectory length: {len(traj)}")
```

---

## 5. Deep-Learning-Based Obstacle Avoidance

### 5.1 End-to-End Learning

```
┌─────────────────────────────────────────────────────────────┐
│                深度学习避障架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入                        神经网络           输出        │
│  ┌─────────┐              ┌─────────┐       ┌─────────┐   │
│  │ 激光雷达 │──┐          │  CNN +  │       │  v_cmd  │   │
│  │  扫描   │  ├──▶ 拼接 ──▶│  LSTM   │──▶ MLP──▶│         │   │
│  │ 目标方向│──┘          │         │       │  ω_cmd  │   │
│  └─────────┘              └─────────┘       └─────────┘   │
│                                                             │
│  训练方式：                                                  │
│  1. 监督学习：专家演示                                      │
│  2. 强化学习：奖励驱动                                      │
│  3. 模仿学习：行为克隆                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Input: LiDAR scan + goal direction → concatenated → neural network (CNN + LSTM) → MLP → outputs v_cmd and ω_cmd. Training: supervised learning (expert demonstrations), RL (reward-driven), imitation learning (behavior cloning).)*

### 5.2 Simple Network Example

```python
import torch
import torch.nn as nn

class ObstacleAvoidanceNet(nn.Module):
    """LiDAR-based obstacle avoidance network"""
    
    def __init__(self, lidar_points: int = 360, hidden_size: int = 128):
        super().__init__()
        
        # LiDAR processing branch
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_points, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Goal direction processing
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 32),  # goal relative position (dx, dy)
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU()
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(64, 2),  # (v, ω)
            nn.Tanh()
        )
        
    def forward(self, lidar_scan: torch.Tensor, goal_relative: torch.Tensor):
        """
        Args:
            lidar_scan: LiDAR data [batch, 360]
            goal_relative: goal relative position [batch, 2]
        
        Returns:
            v, ω: velocity command [batch, 2]
        """
        lidar_feat = self.lidar_encoder(lidar_scan)
        goal_feat = self.goal_encoder(goal_relative)
        
        fused = torch.cat([lidar_feat, goal_feat], dim=-1)
        hidden = self.fusion(fused)
        output = self.output(hidden)
        
        # Scale to the actual velocity range
        v = (output[:, 0] + 1) / 2 * 1.0  # [0, 1] m/s
        ω = output[:, 1] * np.pi / 3       # [-π/3, π/3] rad/s
        
        return torch.stack([v, ω], dim=-1)


# Training example
def train_obstacle_avoidance():
    """Train the obstacle avoidance network (pseudocode)"""
    model = ObstacleAvoidanceNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Assume training data exists
    # lidar_data: [N, 360]
    # goal_data: [N, 2]
    # expert_actions: [N, 2]
    
    for epoch in range(100):
        # Forward pass
        pred_actions = model(lidar_data, goal_data)
        
        # Compute the loss
        loss = criterion(pred_actions, expert_actions)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 6. Algorithm Comparison

### 6.1 Performance Comparison

| Algorithm | Real-time | Optimality | Tuning | Dynamic Obstacles | Use Cases |
|-----------|-----------|------------|--------|-------------------|-----------|
| **DWA** | High | Local optimum | Medium | Fair | Indoor navigation |
| **Potential field** | Highest | No guarantee | Simple | Poor | Simple environments |
| **MPC** | Medium | Fairly good | Complex | Good | Precision control |
| **Deep learning** | High | Depends on training | Difficult | Fairly good | Complex environments |

### 6.2 Selection Guide

```
┌─────────────────────────────────────────────────────────────┐
│                  避障算法选择                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  实时性要求高？                                              │
│     ├── 是 → 势场法 / 简化DWA                              │
│     └── 否 → MPC                                           │
│                                                             │
│  有动态障碍物？                                              │
│     ├── 是 → MPC / 深度学习                                │
│     └── 否 → DWA / 势场法                                  │
│                                                             │
│  需要考虑机器人动力学？                                      │
│     ├── 是 → MPC                                           │
│     └── 否 → DWA / 势场法                                  │
│                                                             │
│  有大量训练数据？                                            │
│     ├── 是 → 深度学习                                      │
│     └── 否 → 传统方法                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(High real-time requirement? → potential field / simplified DWA, else MPC. Dynamic obstacles? → MPC / deep learning, else DWA / potential field. Need to consider robot dynamics? → MPC, else DWA / potential field. Lots of training data? → deep learning, else classical methods.)*

### 6.3 Practical Recommendations

1. **Indoor service robots**: DWA + local planner
2. **Autonomous driving**: MPC + decision layer
3. **Drones**: potential field + trajectory tracking
4. **Complex environments**: deep learning + classical methods

---

## Further Reading

### Papers
- "The Dynamic Window Approach to Collision Avoidance" — Fox et al.
- "Model Predictive Control for Autonomous and Semiautonomous Vehicles"

### Open-Source Implementations
- **ROS Navigation Stack**: DWA implementation
- **CVXGEN**: MPC code generator

---

*This chapter is continuously updated...*
