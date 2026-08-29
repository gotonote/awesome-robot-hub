# End-Effectors & Sensors

## Contents

- [1. Grippers](#1-grippers)
- [2. Sensors](#2-sensors)
- [3. Selection Guide](#3-selection-guide)

---

## 1. Grippers

### 1.1 Common Types

| Type | Features | Use Case |
|------|----------|----------|
| Parallel gripper | Simple, reliable | Regular objects |
| Three-finger gripper | Multi-angle grasping | Irregular objects |
| Vacuum suction cup | Planar suction | Objects with flat surfaces |
| Soft gripper | Adaptability | Soft objects |

### 1.2 Control Example

```python
# Robotiq 2F gripper
from robotiq_2f_gripper import Robotiq2FGripper

gripper = Robotiq2FGripper()
gripper.activate()
gripper.set_position(100)  # 0-255
gripper.set_force(50)     # 0-255
```

---

## 2. Sensors

### 2.1 Vision Sensors

```python
# RealSense D435
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Get frames
frames = pipeline.wait_for_frames()
depth = frames.get_depth_frame()
color = frames.get_color_frame()
```

### 2.2 Force Sensors

```python
# ATI Force/Torque Sensor
from ati_ft import FTSensor

sensor = FTSensor(device='/dev/ttyUSB0')
ft_data = sensor.read()  # [Fx, Fy, Fz, Tx, Ty, Tz]
```

---

## 3. Selection Guide

### 3.1 Selection Principles

1. **Task requirements**: grasping/manipulation → choose the corresponding gripper
2. **Precision requirements**: high precision → force sensor + vision
3. **Environment**: dark → depth camera + infrared
4. **Cost**: budget-limited → balance functionality and cost

---

*This chapter is continuously updated...*
