# 14 Simulation

Robot simulation is an essential tool for training and validating robot algorithms. This chapter introduces the features and usage of mainstream simulation platforms.

## Contents

- [1. NVIDIA Isaac Sim](01-isaac-sim.md)
  - GPU acceleration
  - RTX rendering
- [2. MuJoCo](02-mujoco.md)
  - High-precision physics
  - Control benchmarks
- [3. PyBullet / Gazebo](03-pybullet-gazebo.md)
  - Open-source, easy to use
  - ROS integration
- [4. SAPIEN](04-sapien.md)
  - High-fidelity interaction

---

## Platform Comparison

| Platform | Physics Accuracy | Rendering | Difficulty | Best For |
|----------|-----------------|-----------|------------|----------|
| Isaac Sim | High | RTX | Medium | Large-scale training |
| MuJoCo | High | Medium | Low | Control research |
| PyBullet | Medium | Low | Low | Rapid prototyping |
| Gazebo | Medium | Medium | Medium | ROS projects |
| SAPIEN | High | RTX | Medium | Dexterous manipulation |

---

## Selection Guide

1. **Continuous control research**: MuJoCo
2. **Large-scale training**: Isaac Sim
3. **Rapid prototyping**: PyBullet
4. **ROS integration**: Gazebo
5. **Object manipulation**: SAPIEN

---

*This chapter is continuously updated...*
