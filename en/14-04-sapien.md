# SAPIEN

SAPIEN is a high-fidelity interactive robot simulation platform supporting realistic physical interaction and visual rendering.

## Contents

- [1. SAPIEN Overview](#1-sapien-overview)
- [2. Installation & Setup](#2-installation--setup)
- [3. Core Features](#3-core-features)
- [4. Application Scenarios](#4-application-scenarios)

---

## 1. SAPIEN Overview

### 1.1 Features

- GPU physics simulation
- RTX rendering
- Large-scale scenes
- Interactive objects

### 1.2 Comparison with Other Simulators

| Simulator | Physics | Rendering | Use Case |
|-----------|---------|-----------|----------|
| SAPIEN | High-fidelity | RTX | Dexterous manipulation |
| Isaac Sim | High-fidelity | RTX | Large scale |
| PyBullet | Real-time | Basic | Rapid prototyping |
| Gazebo | Medium | Medium | ROS integration |

---

## 2. Installation & Setup

### 2.1 Installation

```bash
# Install via pip
pip install sapien

# Install from source
git clone https://github.com/haosulab/SAPIEN.git
cd SAPIEN
pip install -e .
```

---

## 3. Core Features

### 3.1 Basic Usage

```python
import sapien
import numpy as np

# Create the engine
engine = sapien.Engine()

# Create the renderer
renderer = sapien.VulkanRenderer()

# Create the scene
scene = engine.create_scene(renderer=renderer)

# Add the ground
ground = scene.create_actor_builder().build_kinematic()

# Load a robot
loader = scene.create_robot_loader()
robot = loader.load("franka_panda.urdf")
robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

# Simulation loop
for _ in range(1000):
    scene.step()
    renderer.render()
```

---

## 4. Application Scenarios

### 4.1 Object Manipulation

```python
# Grasping task
def grasp_object(robot, object_actor):
    # Move above the object
    robot.set_target_pose(object_pose + offset)
    
    # Descend
    robot.set_target_pose(object_pose)
    
    # Close the gripper
    robot.set_qpos(gripper_closed)
```

---

## References

1. SAPIEN: A Physically Realistic Robot Interaction Simulator

---

*This chapter is continuously updated...*
