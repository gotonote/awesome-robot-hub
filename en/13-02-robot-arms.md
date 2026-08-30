# Robot Arms (Franka, xArm, UR)

Introduction and usage of mainstream robot arm platforms.

## Contents

- [1. Mainstream Robot Arm Comparison](#1-mainstream-robot-arm-comparison)
- [2. Franka Emika Panda](#2-franka-emika-panda)
- [3. xArm](#3-xarm)
- [4. Universal Robots (UR)](#4-universal-robots-ur)

---

## 1. Mainstream Robot Arm Comparison

| Arm | DOF | Payload | Reach | Features |
|-----|-----|---------|-------|----------|
| Franka | 7 | 3kg | 855mm | High precision, expensive |
| xArm 7 | 7 | 7kg | 698mm | Cost-effective |
| UR5e | 6 | 5kg | 850mm | Collaborative safety |
| UR10e | 6 | 10kg | 1300mm | Large workspace |

---

## 2. Franka Emika Panda

### 2.1 Features

- 7 DOF
- Torque control
- Collision detection
- Research-friendly

### 2.2 Control Interface

```python
# libfranka Python interface
import franka_interface

# Create interfaces
gripper = GripperInterface('robot')
robot = RobotInterface('robot')

# Read state
state = robot.read_once()
print(state.q)  # joint positions

# Joint control
robot.control(
    q_desired=[0, -0.785, 0, -2.356, 0, 1.571, 0.785],
    controller_mode='joint_position'
)

# End-effector control
robot.control(
    pose_desired=[0.3, 0, 0.4, 0, 0, 0, 1],
    controller_mode='cartesian_impedance'
)
```

---

## 3. xArm

### 3.1 Features

- 7 DOF
- Open-source SDK
- Multiple control modes
- Cost-effective

### 3.2 Python Control

```python
from xarm.wrapper import XArmAPI

# Connect to the arm
arm = XArmAPI('192.168.1.xxx')

# Joint angle control
arm.set_servo_angle(angle=[0, -45, 0, -135, 0, 90, 0], wait=True)

# Position control
arm.set_position(x=200, y=0, z=300, roll=0, pitch=0, yaw=0)

# End-effector control
arm.set_end_effector_gripper(enable=True)
arm.set_gripper_position(400)  # 0-850
```

---

## 4. Universal Robots (UR)

### 4.1 Features

- Collaborative safety
- Widely used
- Easy programming
- ROS support

### 4.2 ROS2 Control

```python
# ROS2 MoveIt2 control
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_py import MoveItPy

# Load the configuration
moveit_config = MoveItConfigsBuilder("ur5e", package_path="ur5e_moveit_config").to_dict()

# Create MoveIt
moveit = MoveItPy(node_name="ur5e_moveit")
moveit.set_max_velocity_scaling_factor(0.5)

# Plan
plan = moveit.plan(
    pose_goal=PoseStamped(pose=Pose(position=Point(x=0.3, y=0.0, z=0.3)))
)

# Execute
moveit.execute(plan, wait=True)
```

---

*This chapter is continuously updated...*
