# Autonomous Driving

## Contents

- [1. Autonomous Driving Overview](#1-autonomous-driving-overview)
- [2. Perception Systems](#2-perception-systems)
- [3. Localization & Mapping](#3-localization--mapping)
- [4. Planning & Control](#4-planning--control)
- [5. Simulation Testing](#5-simulation-testing)

---

## 1. Autonomous Driving Overview

### 1.1 SAE Levels

| Level | Name | Responsibility |
|-------|------|----------------|
| L0 | No automation | Human drives |
| L1 | Driver assistance | Steering or speed control |
| L2 | Partial automation | Steering + speed |
| L3 | Conditional automation | Autonomous in specific scenarios |
| L4 | High automation | No human needed in most scenarios |
| L5 | Full automation | Autonomous in all scenarios |

---

## 2. Perception Systems

### 2.1 Sensor Configuration

```
Typical L4 configuration:
- LiDAR × 3-5
- Cameras × 6-12
- Millimeter-wave radar × 3-5
- Ultrasonic × 10+
- Integrated navigation (GPS/IMU)
```

### 2.2 Object Detection

```python
# Point cloud 3D detection
import torch
from model import PointPillars

model = PointPillars(num_classes=3)
detections = model(point_cloud)
```

---

## 3. Localization & Mapping

### 3.1 HD Maps

```python
# HD Map structure
class HDMap:
    lanes = []        # lane lines
    intersections = []  # intersections
    traffic_lights = []  # traffic lights
    stop_signs = []    # stop signs
```

---

## 4. Planning & Control

### 4.1 Motion Planning

```python
# Behavior planning
class BehaviorPlanner:
    def plan(self, env_state):
        # State machine selection
        if env_state.crossing_pedestrian:
            return "STOP"
        elif env_state.green_light:
            return "GO"
        else:
            return "WAIT"
```

---

## 5. Simulation Testing

### 5.1 Simulation Platforms

| Platform | Features |
|----------|----------|
| CARLA | Open source, complex scenarios |
| AirSim | Microsoft, Unity integration |
| LGSVL | LG open source |
| PreSIL | NVIDIA |

---

*This chapter is continuously updated...*
