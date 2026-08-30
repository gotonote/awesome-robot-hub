# Sensors & State Estimation Basics

> This chapter introduces the sensor types, principles, and state estimation methods commonly used in Physical AI systems. These are the foundation for robots to perceive the environment and achieve precise control.

## 1. Sensor Overview

### 1.1 Sensor Taxonomy

```
┌─────────────────────────────────────────────────────────┐
│                      传感器类型                           │
├───────────────────────┬─────────────────────────────────┤
│  内部传感器           │  外部传感器                      │
│  - 编码器             │  - 视觉相机                      │
│  - IMU                │  - 深度相机                      │
│  - 力矩传感器         │  - 激光雷达                      │
│  - 温度传感器         │  - 超声波                        │
└───────────────────────┴─────────────────────────────────┘
```

*(Internal sensors: encoders, IMU, torque sensors, temperature sensors. External sensors: vision cameras, depth cameras, LiDAR, ultrasonic.)*

### 1.2 Sensor Selection Principles

| Principle | Description |
|-----------|-------------|
| Accuracy | Measurement precision meeting task requirements |
| Update rate | Higher than the system control frequency |
| Reliability | Stable, robust to interference |
| Cost | Within project budget |
| Power consumption | Especially important for mobile robots |

## 2. Internal Sensors

### 2.1 Position Encoders

**Incremental encoders**:
- Output pulse signals
- Require pulse counting
- Lose position information on power loss

```python
class IncrementalEncoder:
    def __init__(self, pulses_per_revolution=4096):
        self.ppr = pulses_per_revolution
        self.count = 0
        self.prev_state = 0
    
    def update(self, encoder_state):
        """Update encoder count"""
        # Compute state change
        delta = encoder_state - self.prev_state
        if delta < 0:
            delta += 4  # handle wraparound
        self.count += delta
        self.prev_state = encoder_state
    
    def get_position_rad(self):
        """Get position (radians)"""
        return (self.count / self.ppr) * 2 * np.pi
    
    def get_velocity_rad_s(self, dt):
        """Get velocity (rad/s)"""
        return self.get_position_rad() / dt
```

**Absolute encoders**:
- Output absolute position values
- Retain position after power loss
- Often use Gray code encoding

### 2.2 IMU (Inertial Measurement Unit)

An IMU consists of a 3-axis accelerometer and a 3-axis gyroscope:

```
┌────────────────────────────────────────┐
│                 IMU 结构                 │
│  ┌─────────────┐    ┌─────────────┐    │
│  │ 加速度计    │    │  陀螺仪     │    │
│  │ ax, ay, az  │    │  wx, wy, wz │    │
│  └─────────────┘    └─────────────┘    │
└────────────────────────────────────────┘
```

*(IMU structure: accelerometer (ax, ay, az) and gyroscope (wx, wy, wz))*

**Accelerometer principle**:
- Measures specific force (acceleration + gravity)
- $a_{meas} = a_{true} + g + b + noise$

**Gyroscope principle**:
- Measures angular velocity
- Integrate to obtain angle

```python
class IMU:
    def __init__(self):
        self.gravity = np.array([0, 0, 9.81])
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
    
    def read(self):
        """Read IMU data"""
        # In practice, calibration and filtering are needed
        accel = self.accel_bias + np.random.randn(3) * 0.01
        gyro = self.gyro_bias + np.random.randn(3) * 0.001
        return accel, gyro
    
    def compute_orientation(self, gyro_data, dt):
        """Compute orientation from gyroscope data (simplified)"""
        # In practice, use quaternions or an EKF
        return np.eye(3)  # placeholder
```

### 2.3 Force/Torque Sensors

**Types**:
- Joint torque sensors
- End-effector force/torque (F/T) sensors
- Tactile sensors

```python
class ForceSensor:
    def __init__(self, calibration_matrix):
        """
        Calibration matrix: converts raw voltage to force/torque
        """
        self.calib = calibration_matrix
    
    def read_force(self, raw_voltage):
        """
        Read force data
        raw_voltage: raw sensor voltage
        """
        # Apply calibration matrix
        force = self.calib @ raw_voltage
        return force
```

## 3. External Sensors

### 3.1 Cameras

**Types**:
| Type | Features | Applications |
|------|----------|--------------|
| Monocular | 2D info, no depth | Classification, detection |
| Stereo | Depth via disparity | 3D reconstruction, navigation |
| RGB-D | Real-time depth | Indoor navigation, obstacle avoidance |
| Event camera | High frame rate, low latency | High-speed tracking |

```python
class Camera:
    def __init__(self, fx, fy, cx, cy):
        """
        Camera intrinsics
        """
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def pixel_to_ray(self, u, v):
        """
        Pixel coordinates to ray direction (camera frame)
        """
        x = (u - self.K[0,2]) / self.K[0,0]
        y = (v - self.K[1,2]) / self.K[1,1]
        return np.array([x, y, 1.0])
    
    def triangulate(self, p1, p2, T):
        """
        Stereo triangulation for depth
        T: relative pose between cameras
        """
        # Simplified implementation
        # In practice, solve with SVD
        return np.linalg.norm(T[:3,3])  # placeholder
```

### 3.2 LiDAR

**Types**:
- 2D LiDAR: single-plane scanning
- 3D LiDAR: multi-line scanning

```python
class Lidar2D:
    def __init__(self, angle_min, angle_max, num_beams, max_range):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.num_beams = num_beams
        self.max_range = max_range
        self.angles = np.linspace(angle_min, angle_max, num_beams)
    
    def scan_to_points(self, ranges):
        """
        Convert laser scan data to a point cloud
        """
        points = []
        for angle, r in zip(self.angles, ranges):
            if 0 < r < self.max_range:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append([x, y, 0])
        return np.array(points)
    
    def detect_obstacles(self, ranges, threshold=0.3):
        """
        Obstacle detection
        """
        obstacles = []
        for angle, r in zip(self.angles, ranges):
            if r < threshold:
                obstacles.append({
                    'angle': angle,
                    'distance': r,
                    'position': [r*np.cos(angle), r*np.sin(angle)]
                })
        return obstacles
```

### 3.3 Depth Cameras

**Technology routes**:
- **Structured light**: Intel RealSense, Apple Face ID
- **ToF (Time of Flight)**: Microsoft Kinect v2
- **Stereo vision**: Stereo D435

```python
class DepthCamera:
    def __init__(self, depth_scale=0.001):
        self.depth_scale = depth_scale
    
    def depth_to_point_cloud(self, depth_image, intrinsic):
        """
        Convert depth image to a point cloud
        """
        h, w = depth_image.shape
        points = []
        
        for v in range(h):
            for u in range(w):
                z = depth_image[v, u] * self.depth_scale
                if z > 0:
                    x = (u - intrinsic.cx) * z / intrinsic.fx
                    y = (v - intrinsic.cy) * z / intrinsic.fy
                    points.append([x, y, z])
        
        return np.array(points)
```

### 3.4 Sensor Fusion

```
┌─────────────────────────────────────────────────────────┐
│                   传感器融合架构                          │
│                                                         │
│    相机      激光雷达      IMU       轮式编码器          │
│      │          │          │            │              │
│      v          v          v            v              │
│  ┌──────────────────────────────────────────────┐      │
│  │              融合中心 (EKF / UKF)              │      │
│  └──────────────────────────────────────────────┘      │
│                         │                               │
│                         v                               │
│              机器人状态估计 (位姿+速度)                  │
└─────────────────────────────────────────────────────────┘
```

*(Fusion architecture: camera, LiDAR, IMU, wheel encoders → fusion center (EKF/UKF) → robot state estimation (pose + velocity))*

## 4. State Estimation Basics

### 4.1 Probabilistic Robot Model

**State**:
$$x_t = (p_t, v_t, \theta_t)$$

- $p_t$: position
- $v_t$: velocity
- $t$: time step

**Observation**:
$$z_t = \text{sensor readings}$$

### 4.2 Kalman Filter

**Linear Gaussian systems**:
$$x_t = A x_{t-1} + B u_t + w_t$$
$$z_t = H x_t + v_t$$

```python
class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x    # state dimension
        self.dim_z = dim_z    # observation dimension
        
        # State and covariance
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)  # state covariance
        
        # Transition and observation matrices
        self.F = np.eye(dim_x)  # state transition
        self.H = np.zeros((dim_z, dim_x))  # observation
        
        # Noise
        self.Q = np.eye(dim_x) * 0.01  # process noise
        self.R = np.eye(dim_z) * 0.1   # observation noise
    
    def predict(self, u=None):
        """Prediction step"""
        # State prediction
        if u is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self, z):
        """Update step"""
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form, numerically stable)
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x
```

### 4.3 Extended Kalman Filter (EKF)

Handles nonlinear systems:
$$x_t = f(x_{t-1}, u_t) + w_t$$
$$z_t = h(x_t) + v_t$$

```python
class ExtendedKalmanFilter:
    def __init__(self, state_dim, obs_dim):
        self.kf = KalmanFilter(state_dim, obs_dim)
    
    def predict(self, u=None):
        """Nonlinear prediction"""
        # Linearize: compute Jacobian
        F = self.compute_jacobian_f(self.kf.x, u)
        self.kf.F = F
        return self.kf.predict(u)
    
    def update(self, z):
        """Nonlinear update"""
        # Linearize: compute Jacobian
        H = self.compute_jacobian_h(self.kf.x)
        self.kf.H = H
        return self.kf.update(z)
    
    def compute_jacobian_f(self, x, u):
        """Jacobian of the state transition function"""
        # Depends on the system model
        return np.eye(self.kf.dim_x)
    
    def compute_jacobian_h(self, x):
        """Jacobian of the observation function"""
        # Depends on the observation model
        return np.eye(self.kf.dim_z)
```

### 4.4 Unscented Kalman Filter (UKF)

Uses sigma points to approximate nonlinear distributions:

```python
class UnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z, kappa=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kappa = kappa
        
        # Compute weights
        n = dim_x
        self.lambda_ = n + kappa - n
        self.weights_mean = np.zeros(2*n + 1)
        self.weights_cov = np.zeros(2*n + 1)
        self.weights_mean[0] = self.lambda_ / (n + self.lambda_)
        self.weights_cov[0] = self.lambda_ / (n + self.lambda_)
        for i in range(1, 2*n + 1):
            self.weights_mean[i] = 1 / (2 * (n + self.lambda_))
            self.weights_cov[i] = 1 / (2 * (n + self.lambda_))
    
    def sigma_points(self, x, P):
        """Generate sigma points"""
        n = self.dim_x
        sigma = np.zeros((2*n + 1, n))
        sigma[0] = x
        
        try:
            P_sqrt = np.linalg.cholesky((n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            P_sqrt = np.linalg.sqrtm((n + self.lambda_) * P)
        
        for i in range(n):
            sigma[i + 1] = x + P_sqrt[:, i]
            sigma[i + 1 + n] = x - P_sqrt[:, i]
        
        return sigma
    
    def predict(self, f, Q):
        """Prediction step"""
        # Generate sigma points
        sigma = self.sigma_points(self.x, self.P)
        
        # Propagate sigma points
        sigma_pred = np.array([f(s) for s in sigma])
        
        # Predicted mean
        self.x = sum(w * s for w, s in zip(self.weights_mean, sigma_pred))
        
        # Predicted covariance
        self.P = Q.copy()
        for w, s in zip(self.weights_cov, sigma_pred):
            diff = s - self.x
            self.P += w * np.outer(diff, diff)
```

### 4.5 Particle Filter

Suitable for nonlinear, non-Gaussian systems:

```python
class ParticleFilter:
    def __init__(self, num_particles, state_dim):
        self.num_particles = num_particles
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles
    
    def predict(self, motion_model, u):
        """Prediction step"""
        for i in range(self.num_particles):
            self.particles[i] = motion_model(
                self.particles[i], u
            ) + np.random.randn(self.dim) * 0.1
    
    def update(self, observation_model, z):
        """Update step"""
        for i in range(self.num_particles):
            # Compute likelihood
            pred_z = observation_model(self.particles[i])
            likelihood = self.compute_likelihood(z, pred_z)
            self.weights[i] *= likelihood
        
        # Normalize
        self.weights /= np.sum(self.weights)
        
        # Resample
        self.resample()
    
    def resample(self):
        """Systematic resampling"""
        indices = np.random.choice(
            self.num_particles,
            self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_estimate(self):
        """Get state estimate"""
        return np.average(self.particles, weights=self.weights, axis=0)
```

## 5. Sensor Calibration

### 5.1 Camera Calibration

```python
class CameraCalibrator:
    def __init__(self):
        self.obj_points = []  # 3D world coordinates
        self.img_points = []  # 2D image coordinates
    
    def add_calibration_image(self, image, grid_size=(9,6)):
        """Add a calibration image"""
        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(
            image, grid_size, None
        )
        if ret:
            self.obj_points.append(self.compute_object_points(grid_size))
            self.img_points.append(corners)
    
    def compute_object_points(self, grid_size):
        """Generate chessboard 3D coordinates"""
        objp = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
        return objp
    
    def calibrate(self):
        """Run calibration"""
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, 
            image_size, None, None
        )
        return mtx, dist
```

### 5.2 IMU Calibration

```python
class IMUCalibrator:
    def __init__(self):
        self.gyro_biases = []
        self.accel_biases = []
    
    def calibrate_gyro(self, samples):
        """Gyroscope bias calibration"""
        # Assumes the device is stationary
        self.gyro_bias = np.mean(samples, axis=0)
        return self.gyro_bias
    
    def calibrate_accel(self, samples):
        """Accelerometer calibration (multi-position)"""
        # Identify the gravity direction at each position
        # Calibrate scale factors and biases
        # Requires at least 6 different orientations
        pass
```

## 6. Summary

```
┌────────────────────────────────────────────────────────┐
│                  传感器 vs 状态估计                      │
├────────────────────────────────────────────────────────┤
│  传感器: 原始数据采集                                   │
│  - 内部: 编码器、IMU、力传感器                          │
│  - 外部: 相机、激光雷达、深度相机                       │
├────────────────────────────────────────────────────────┤
│  状态估计: 从噪声数据中恢复真实状态                      │
│  - KF: 线性高斯                                        │
│  - EKF: 非线性（线性化近似）                            │
│  - UKF: 非线性（Sigma点近似）                          │
│  - PF: 非线性非高斯（粒子近似）                         │
└────────────────────────────────────────────────────────┘
```

*(Sensors: raw data acquisition — internal: encoders, IMU, force sensors; external: cameras, LiDAR, depth cameras. State estimation: recover true state from noisy data — KF: linear Gaussian; EKF: nonlinear (linearization approximation); UKF: nonlinear (sigma-point approximation); PF: nonlinear non-Gaussian (particle approximation))*

## 7. Further Reading

- *Probabilistic Robotics* — Thrun, Burgard, Fox
- *State Estimation for Robotics* — Barfoot
- OpenCV camera calibration documentation
- ROS sensor_msgs documentation

---

*The next chapter introduces perception technology — computer vision and depth perception in Physical AI.*
