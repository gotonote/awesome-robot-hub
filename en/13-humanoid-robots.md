# Humanoid Robots (Unitree, Atlas)

## Contents

- [1. Unitree](#1-unitree)
- [2. Boston Dynamics Atlas](#2-boston-dynamics-atlas)

---

## 1. Unitree

### 1.1 Product Line

| Model | Features | Use Case |
|-------|----------|----------|
| Go1 | Quadruped, companion | Research, education |
| AliEN | Quadruped, industrial | Inspection |
| H1 | Humanoid, full-size | Research |

### 1.2 Go1 Control

```python
# Unitree Go1 Python SDK
import numpy as np

class Go1Controller:
    def __init__(self, ip='192.168.1.xxx'):
        from unitree_go.connection import UdpConnection
        
        self.conn = UdpConnection()
        self.conn.client.ip = ip
        self.conn.client.port = 8080
        self.conn.Init()
        
    def move(self, x, y, yaw):
        """Motion control"""
        # Send motion commands
        pass
    
    def get_state(self):
        """Get the state"""
        pass
```

---

## 2. Boston Dynamics Atlas

### 2.1 Features

- 28 DOF
- Hydraulic/electric hybrid
- High-dynamic motion
- Offline programming

### 2.2 Control

```python
# Atlas API (requires official authorization)
# Motion planning
from atlas import AtlasRobot

robot = AtlasRobot('robot_name')
robot.load_controller('WalkingController')

# Set the goal
robot.set_goal(pose=target_pose)

# Start the controller
robot.start()
```

---

*This chapter is continuously updated...*
