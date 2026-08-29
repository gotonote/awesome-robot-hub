# ROS / ROS2

ROS (Robot Operating System) is an important framework for robot software development; ROS2 is its next-generation version.

## Contents

- [1. ROS Overview](#1-ros-overview)
- [2. Core Concepts](#2-core-concepts)
- [3. ROS2 New Features](#3-ros2-new-features)
- [4. Robot Control](#4-robot-control)
- [5. Practice](#5-practice)

---

## 1. ROS Overview

### 1.1 Architecture

```
┌─────────────────────────────────────────┐
│           ROS 架构                      │
├─────────────────────────────────────────┤
│  应用层                                 │
│  ├── 导航 (Navigation)                  │
│  ├── 感知 (Perception)                  │
│  └── 控制 (Control)                     │
├─────────────────────────────────────────┤
│  通信层                                 │
│  ├── Topic (发布/订阅)                   │
│  ├── Service (请求/响应)                 │
│  └── Action (异步目标)                   │
├─────────────────────────────────────────┤
│  硬件抽象层                             │
│  └── 驱动 (Drivers)                      │
└─────────────────────────────────────────┘
```

*(Application layer: Navigation, Perception, Control. Communication layer: Topic (publish/subscribe), Service (request/response), Action (asynchronous goals). Hardware abstraction layer: Drivers.)*

### 1.2 Installation

```bash
# ROS2 Humble (Ubuntu 22.04)
sudo apt update
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

---

## 2. Core Concepts

### 2.1 Nodes

```python
# Python node example
import rclpy
from rclpy.node import Node

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Create a publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Create a subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
    def odom_callback(self, msg):
        self.get_logger().info(f"Position: {msg.pose.pose.position}")

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### 2.2 Topics

```bash
# List topics
ros2 topic list

# Listen to a topic
ros2 topic echo /odom

# Publish a message
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}"
```

---

## 3. ROS2 New Features

### 3.1 DDS Middleware

```yaml
# config/rmw_cyclonedds.yaml
rmw_cyclonedds:
  domain_id: 0
  medium: UDP
```

### 3.2 Actions

```python
# Action server
class NavigateActionServer(Node):
    def __init__(self):
        super().__init__('navigate_action')
        
        self._action_server = ActionServer(
            self,
            Navigate,
            'navigate',
            self.execute_callback
        )
        
    def execute_callback(self, goal_handle):
        # Execute navigation
        result = Navigate.Result()
        goal_handle.succeed(result)
        return result
```

---

## 4. Robot Control

### 4.1 Motion Control

```python
# Joint control
from trajectory_msgs.msg import JointTrajectory

def send_joint_trajectory(pub, positions):
    msg = JointTrajectory()
    msg.joint_names = ['joint1', 'joint2', 'joint3']
    point = JointTrajectoryPoint()
    point.positions = positions
    point.time_from_start = Duration(sec=1)
    msg.points = [point]
    pub.publish(msg)
```

---

## 5. Practice

### 5.1 Launch Files

```python
# launch/robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_control',
            executable='controller',
            name='robot_controller'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        )
    ])
```

---

*This chapter is continuously updated...*
