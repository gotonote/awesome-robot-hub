# 03 Perception

Perception is a core component of Physical AI systems, enabling robots to understand and interact with the physical world. This chapter covers perception principles, implementation methods, and their applications in robotic systems.

## Contents

- [1. Perception Overview](#1-perception-overview)
  - [1.1 Visual Perception](#11-visual-perception)
  - [1.2 Tactile Perception](#12-tactile-perception)
  - [1.3 Force Control Perception](#13-force-control-perception)
  - [1.4 IMU Inertial Measurement](#14-imu-inertial-measurement)
  - [1.5 Other Sensors](#15-other-sensors)
- [2. Multimodal Perception Fusion](#2-multimodal-perception-fusion)
- [3. SLAM & Localization](#3-slam--localization)
- [4. Real-Time Object Detection & Tracking](#4-real-time-object-detection--tracking)
- [5. Deep Learning in Perception](#5-deep-learning-in-perception)
- [6. Sensor Fusion Algorithms](#6-sensor-fusion-algorithms)

---

## 1. Perception Overview

The perception layer of a Physical AI system collects multimodal information from the physical world to provide environment understanding for decision-making and control.

### 1.1 Visual Perception

Vision is one of the most important ways a robot perceives its environment.

#### Key Device Types

| Device | Features | Applications |
|--------|----------|--------------|
| Monocular camera | Low cost, rich information | Object detection, semantic segmentation |
| Stereo camera | Depth information | Navigation, obstacle avoidance, 3D reconstruction |
| RGB-D camera | Real-time depth map | Indoor navigation, object grasping |
| Event camera | High dynamic range, low latency | High-speed tracking, fast response |

#### Code Example: OpenCV Camera Calibration

```python
import cv2
import numpy as np

class VisualPerception:
    def __init__(self, camera_type='rgb_d'):
        self.camera_type = camera_type
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def calibrate_camera(self, calibration_images):
        object_points = []
        image_points = []
        objp = np.zeros((9*6, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                object_points.append(objp)
                image_points.append(corners)
        
        ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points, image_points, gray.shape[::-1], None, None
        )
        return self.camera_matrix, self.dist_coeffs
```

### 1.2 Tactile Perception

Tactile sensors enable robots to sense contact forces, textures, and object properties.

#### Tactile Sensor Types

| Type | Principle | Features | Typical Use |
|------|-----------|----------|-------------|
| Resistive | Resistance changes with pressure | Low cost, durable | Industrial grippers |
| Capacitive | Capacitance change detects pressure | High sensitivity | Fine manipulation |
| Fiber-optic | Light intensity changes with deformation | EMI-immune | Medical robots |
| Piezoelectric | Piezoelectric effect | Fast dynamic response | Force feedback control |

### 1.3 Force Control Perception

Force control perception enables precise force tracking and impedance control.

```python
class ImpedanceController:
    def __init__(self, M=5.0, B=50.0, K=100.0):
        self.M = M  # inertia
        self.B = B  # damping
        self.K = K  # stiffness
        
    def compute_force(self, desired_pos, actual_pos, desired_vel, actual_vel, ext_force=0):
        position_error = desired_pos - actual_pos
        velocity_error = desired_vel - actual_vel
        return self.K * position_error + self.B * velocity_error - ext_force
```

### 1.4 IMU Inertial Measurement

IMUs provide precise measurement of robot orientation and motion state.

| Component | Measures |
|-----------|----------|
| Accelerometer | Linear acceleration |
| Gyroscope | Angular velocity |
| Magnetometer | Heading (optional) |

### 1.5 Other Sensors

| Type | Principle | Range | Features |
|------|-----------|-------|----------|
| Ultrasonic | Time of flight | 0.1-10m | Low cost |
| Infrared | Triangulation | 0.1-4m | High precision |
| LiDAR | Time of flight | 0.1-200m | 3D point cloud |
| Millimeter-wave radar | FMCW | 0.1-300m | All-weather |

---

## 2. Multimodal Perception Fusion

Multimodal fusion integrates information from different sensors for more robust and accurate perception.

### Fusion Architectures

- **Early fusion**: raw data / feature-level fusion
- **Late fusion**: decision-level fusion
- **Attention fusion**: attention-based dynamic weighted fusion

---

## 3. SLAM & Localization

SLAM enables a robot to localize and build a map simultaneously in unknown environments.

### 3.1 Visual SLAM

- Frontend: feature extraction, feature matching, odometry
- Backend: bundle adjustment, loop closure, global optimization

### 3.2 LiDAR SLAM

- Point cloud registration (ICP, NDT)
- Factor graph optimization

### 3.3 Tightly-Coupled SLAM

- Visual-inertial fusion (VIO)
- LiDAR-inertial fusion (LIO)

---

## 4. Real-Time Object Detection & Tracking

### 4.1 2D Object Detection

- YOLO family
- Faster R-CNN

### 4.2 3D Object Detection

- PointPillars
- CenterPoint

### 4.3 Multi-Object Tracking

- SORT
- DeepSORT

---

## 5. Deep Learning in Perception

### 5.1 CNN Image Processing

- ResNet, VGG backbone networks

### 5.2 Transformer Vision Models

- ViT, DETR

### 5.3 End-to-End Perception Models

- BEVFormer
- TransFusion

---

## 6. Sensor Fusion Algorithms

### 6.1 Kalman Filter

```python
class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F = F  # state transition matrix
        self.H = H  # observation matrix
        self.Q = Q  # process noise covariance
        self.R = R  # observation noise covariance
        self.x = None  # state
        self.P = None  # covariance
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(K)) - K @ self.H) @ self.P
        return self.x
```

### 6.2 Particle Filter

### 6.3 Graph Optimization

- G2O, iSAM2

---

*This chapter is continuously updated...*
