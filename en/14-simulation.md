# 14 Simulation

Robot simulation is an essential tool for training and validating robot algorithms. This chapter introduces the features and usage of mainstream simulation platforms.

## Contents

- [1. NVIDIA Isaac Sim](../14_仿真环境/NVIDIA_Isaac_Sim.md) *(content in Chinese)*
  - GPU acceleration
  - RTX rendering
- [2. MuJoCo](../14_仿真环境/MuJoCo.md) *(content in Chinese)*
  - High-precision physics
  - Control benchmarks
- [3. PyBullet / Gazebo](../14_仿真环境/PyBullet_Gazebo.md) *(content in Chinese)*
  - Open-source, easy to use
  - ROS integration
- [4. SAPIEN](../14_仿真环境/SAPIEN.md) *(content in Chinese)*
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
