# 04 Motion Control

Motion planning is one of the core problems in robotics — finding a collision-free path for a robot from a start state to a goal state.

## Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Chapter Content](#chapter-content)
- [Learning Path](#learning-path)

---

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    运动规划问题定义                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入：                                                      │
│  • 机器人模型（运动学/动力学约束）                            │
│  • 起始状态 q_start                                         │
│  • 目标状态 q_goal                                          │
│  • 环境模型（障碍物）                                        │
│                                                             │
│  输出：                                                      │
│  • 无碰撞路径 π: [0,1] → Configuration Space               │
│  • 时间参数化的轨迹 τ: [0,T] → State Space                 │
│                                                             │
│  约束：                                                      │
│  • 避免碰撞（几何/运动学/动力学）                            │
│  • 满足运动学限制（关节限位）                                │
│  • 满足动力学限制（速度/加速度）                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Chinese labels in the diagram above: 输入 = Input, 输出 = Output, 约束 = Constraints, 避免碰撞 = Collision avoidance, 运动学/动力学约束 = Kinematic/dynamic constraints, 无碰撞路径 = Collision-free path, 时间参数化的轨迹 = Time-parameterized trajectory)*

---

## Core Concepts

### Configuration Space

The **configuration space** is the space describing all possible poses of the robot.

```
┌─────────────────────────────────────────────────────────────┐
│                 工作空间 vs 构型空间                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  工作空间（Workspace）：                                     │
│  • 末端执行器在笛卡尔空间中的位置和姿态                      │
│  • 维度：3D位置 + 3D姿态 = 6 DOF                            │
│  • 直观但计算复杂                                            │
│                                                             │
│  构型空间（Configuration Space, C-Space）：                 │
│  • 机器人所有关节角度组成的空间                              │
│  • 维度 = 关节数量                                          │
│  • 例：6-DOF机械臂 → 6维C-Space                             │
│  • 障碍物映射到C-Space → C_obs                              │
│                                                             │
│  优势：在C-Space中，机器人简化为一个点                       │
│                                                             │
│         工作空间                构型空间                     │
│    ┌──────────────┐         ┌──────────────┐               │
│    │   ╔═══╗      │         │    ******    │               │
│    │   ║机器人║   │   ──▶   │   *      *   │               │
│    │   ╚═══╝障碍  │         │  *   ·   *   │ ← 机器人是一个点│
│    │              │         │   *      *   │               │
│    └──────────────┘         │    ******    │ ← C_obs       │
│                             └──────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(工作空间 = Workspace, 构型空间 = Configuration Space, 末端执行器 = End-effector, 关节角度 = Joint angles, 机器人简化为一个点 = The robot is simplified to a point)*

### Path vs. Trajectory

| Concept | Definition | Characteristics |
|---------|------------|-----------------|
| **Path** | A curve in space, no time information | Pure geometric description |
| **Trajectory** | A time-parameterized path | Includes velocity, acceleration |

---

## Chapter Content

### [Path Planning](04-02-path-planning.md)
- A* algorithm
- RRT (Rapidly-exploring Random Tree)
- RRT* (optimal RRT)
- Potential field method

### [Kinematics](04-01-kinematics.md)
- Forward kinematics
- Inverse kinematics
- DH parameters
- Jacobian matrix

### [Obstacle Avoidance](04-03-obstacle-avoidance.md)
- Dynamic Window Approach (DWA)
- Artificial potential field
- Model Predictive Control (MPC)

### [Model Predictive Control (MPC)](04-04-mpc.md)
- MPC fundamentals
- Linear / nonlinear MPC
- Mobile robot trajectory tracking
- Robot arm end-effector control
- Bipedal robot gait planning
- Drone flight control

### [RL-Based Motion Control](04-05-rl-motion-control.md)
- RL basics (MDP, Bellman equation)
- Classic algorithms (Q-Learning, DQN)
- Policy gradient methods (REINFORCE, Actor-Critic)
- Deep RL (DDPG, PPO, SAC)
- Robotic applications & Sim-to-Real transfer

---

## Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│                    运动规划学习路径                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  第1步：理解基本概念                                         │
│  ├── Configuration Space                                    │
│  ├── Workspace vs C-Space                                  │
│  └── 路径 vs 轨迹                                           │
│                                                             │
│  第2步：掌握运动学                                           │
│  ├── 正运动学（DH参数）                                      │
│  ├── 逆运动学（解析/数值解）                                 │
│  └── 雅可比矩阵                                             │
│                                                             │
│  第3步：学习路径规划算法                                     │
│  ├── 图搜索：A*, D*                                         │
│  ├── 采样方法：RRT, PRM                                     │
│  └── 优化方法：CHOMP, TrajOpt                               │
│                                                             │
│  第4步：实践应用                                             │
│  ├── MoveIt! 框架                                           │
│  ├── OMPL 库                                                │
│  └── 实际机器人导航                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Common Tools

| Tool | Purpose | Link |
|------|---------|------|
| **OMPL** | Open-source motion planning library | https://ompl.kavrakilab.org/ |
| **MoveIt** | ROS motion planning framework | https://moveit.ros.org/ |
| **PyBullet** | Physics simulation & planning | https://pybullet.org/ |

---

## Code Example: Simple 2D Path Planning

```python
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

class AStarPlanner:
    """A* path planning algorithm implementation"""
    
    def __init__(self, obstacle_map, resolution=0.1):
        self.map = obstacle_map
        self.resolution = resolution
        self.height, self.width = obstacle_map.shape
        
    def heuristic(self, a, b):
        """Heuristic function: Euclidean distance"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node):
        """Get 8-connected neighbors"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                if self.map[nx, ny] == 0:  # free cell
                    neighbors.append((nx, ny))
        return neighbors
    
    def plan(self, start, goal):
        """Run A* planning"""
        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heappop(frontier)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for next_node in self.get_neighbors(current):
                new_cost = cost_so_far[current] + self.heuristic(current, next_node)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, goal)
                    heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        
        return None  # no path found

# Usage example
if __name__ == "__main__":
    # Create a simple obstacle map
    obstacle_map = np.zeros((50, 50))
    obstacle_map[15:35, 20:25] = 1  # obstacle
    
    planner = AStarPlanner(obstacle_map)
    path = planner.plan((5, 5), (45, 45))
    
    print(f"Path found, length: {len(path)}")
```

---

## Further Reading

- **Book**: *Planning Algorithms* by Steven M. LaValle
- **Course**: MIT 6.832 Underactuated Robotics
- **Paper**: "Sampling-based Algorithms for Optimal Motion Planning" (RRT*)

---

*This chapter is continuously updated...*
