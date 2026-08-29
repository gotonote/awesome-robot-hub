# PyBullet / Gazebo

PyBullet and Gazebo are two widely used open-source robot simulation platforms.

## Contents

- [1. PyBullet](#1-pybullet)
- [2. Gazebo](#2-gazebo)
- [3. Integration & Comparison](#3-integration--comparison)

---

## 1. PyBullet

### 1.1 Introduction

- Python-native
- Real-time simulation
- Easy to use
- VR support

### 1.2 Installation

```bash
pip install pybullet
```

### 1.3 Basic Usage

```python
import pybullet as p
import pybullet_data
import numpy as np

# Connect to the simulation
client = p.connect(p.DIRECT)  # no GUI
client = p.connect(p.GUI)     # GUI

# Add a search path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the ground
plane_id = p.loadURDF("plane.urdf")

# Load a robot
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

# Set joint control
num_joints = p.getNumJoints(robot_id)
for j in range(num_joints):
    p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0)

# Simulation loop
for _ in range(1000):
    p.stepSimulation()
```

### 1.4 Reinforcement Learning Interface

```python
import gym
import pybullet_envs

# Create the environment
env = gym.make('HalfCheetahBulletEnv-v0')

# Training loop
obs = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
```

---

## 2. Gazebo

### 2.1 Introduction

- Native ROS integration
- Indoor/outdoor simulation
- High-fidelity physics
- Widely used

### 2.2 ROS2 Integration

```python
# ROS2 + Gazebo launch
# Launch file
launch_gazebo = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([
        PathJoinSubstitution([
            FindPackageShare('gazebo_ros'),
            'launch',
            'gzserver.launch.py'
        ])
    ]),
    launch_arguments={'world': world_path}.items()
)
```

### 2.3 Robot Model (SDF)

```xml
<sdf version="1.6">
  <model name="robot">
    <static>false</static>
    <link name="base_link">
      <pose>0 0 0.1 0 0 0</pose>
      <collision>
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
      </collision>
      <visual>
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
```

---

## 3. Integration & Comparison

### 3.1 Comparison

| Feature | PyBullet | Gazebo |
|---------|----------|--------|
| Physics engine | Bullet | ODE/Simbody/DART |
| ROS integration | Basic | Deep |
| Difficulty | Simple | Medium |
| Rendering | Basic | High quality |
| Best for | Rapid prototyping | Research |

### 3.2 Selection Advice

- **Rapid experiments**: PyBullet
- **ROS projects**: Gazebo
- **High realism**: Isaac Sim
- **Continuous control**: MuJoCo

---

*This chapter is continuously updated...*
