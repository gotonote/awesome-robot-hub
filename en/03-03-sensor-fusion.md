# Sensor Fusion (LiDAR, Camera, Radar)

Sensor fusion integrates information from different sensors to obtain more accurate and robust environment perception. This chapter details the principles of each sensor type and fusion methods.

## Contents

- [1. Sensor Overview](#1-sensor-overview)
- [2. Camera-Radar Fusion](#2-camera-radar-fusion)
- [3. LiDAR-Camera Fusion](#3-lidar-camera-fusion)
- [4. Multi-Sensor Time Synchronization](#4-multi-sensor-time-synchronization)
- [5. Spatial Alignment](#5-spatial-alignment)
- [6. Fusion Algorithms](#6-fusion-algorithms)

---

## 1. Sensor Overview

### 1.1 Sensor Comparison

| Sensor | Pros | Cons | Typical Use |
|--------|------|------|-------------|
| Camera | Rich semantic info, low cost | Affected by lighting, no depth | Object classification, lane detection |
| LiDAR | Precise depth, 3D structure | Expensive, no semantics | 3D detection, SLAM |
| Radar | Strong velocity measurement, all-weather | Low resolution, noisy | Object tracking, forward collision warning |
| Ultrasonic | Very cheap, simple | Short range, low accuracy | Parking assistance |

### 1.2 Sensor Selection

```
Application scenario → sensor combination

Low-speed indoor navigation: RGB-D camera + ultrasonic
Autonomous driving: LiDAR + camera + millimeter-wave radar
Industrial inspection: line laser + industrial camera
```

---

## 2. Camera-Radar Fusion

### 2.1 Fusion Architecture

```
┌──────────────┐     ┌──────────────┐
│   相机检测   │     │   毫米波雷达  │
│  (2D bbox)   │     │   (目标列表)  │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌────────────────────────────────┐
│         融合模块              │
│   - 空间投影                  │
│   - 目标关联                  │
│   - 状态融合                  │
└──────────────┬───────────────┘
               │
               ▼
        ┌──────────────┐
        │   融合结果   │
        │  (3D目标)    │
        └──────────────┘
```

*(Camera detection (2D bbox) + millimeter-wave radar (target list) → fusion module (spatial projection, target association, state fusion) → fused result (3D objects))*

### 2.2 Camera-Radar Fusion Code

```python
import numpy as np
import cv2

class CameraRadarFusion:
    def __init__(self, camera_params, radar_params):
        # Camera parameters
        self.K = camera_params['K']           # intrinsic matrix
        self.R = camera_params['R']            # extrinsic rotation
        self.t = camera_params['t']            # extrinsic translation
        self.width = camera_params['width']
        self.height = camera_params['height']
        
        # Radar parameters
        self.radar_extrinsics = radar_params['extrinsics']
        
    def radar_to_camera(self, radar_objects):
        """
        Project radar targets into the camera frame
        radar_objects: [[x, y, z, vx, vy], ...] (radar frame)
        """
        projected = []
        
        for obj in radar_objects:
            # Radar coordinates (x, y, z)
            radar_coord = np.array([obj[0], obj[1], obj[2], 1])
            
            # Convert to camera coordinates
            # First to the vehicle frame, then to the camera frame
            camera_coord = self.radar_extrinsics @ radar_coord
            
            # Project to image
            if camera_coord[2] > 0:  # in front of the camera
                uvw = self.K @ camera_coord[:3]
                u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
                
                if 0 <= u < self.width and 0 <= v < self.height:
                    projected.append({
                        'u': u,
                        'v': v,
                        'depth': camera_coord[2],
                        'vx': obj[3],
                        'vy': obj[4],
                        'range': np.sqrt(obj[0]**2 + obj[1]**2),
                        'doppler': obj[3]  # Doppler velocity
                    })
                    
        return projected
    
    def fuse(self, camera_detections, radar_objects, iou_threshold=0.3):
        """
        Fuse camera and radar detections
        camera_detections: [{'bbox': [x1,y1,x2,y2], 'class': 'car', 'score': 0.9}, ...]
        radar_objects: list of radar targets
        """
        # Project radar targets to the image
        radar_projected = self.radar_to_camera(radar_objects)
        
        # Target association
        fused_objects = []
        
        for cam_det in camera_detections:
            cam_bbox = cam_det['bbox']
            cam_center = [(cam_bbox[0] + cam_bbox[2]) / 2, 
                          (cam_bbox[1] + cam_bbox[3]) / 2]
            
            # Find the nearest radar target
            best_match = None
            best_dist = float('inf')
            
            for radar_obj in radar_projected:
                dist = np.sqrt((radar_obj['u'] - cam_center[0])**2 + 
                              (radar_obj['v'] - cam_center[1])**2)
                
                if dist < best_dist and dist < 50:  # pixel distance threshold
                    best_dist = dist
                    best_match = radar_obj
            
            if best_match:
                # Fuse: camera classification + radar depth and velocity
                fused_objects.append({
                    'bbox': cam_bbox,
                    'class': cam_det['class'],
                    'score': cam_det['score'],
                    'depth': best_match['depth'],
                    'velocity': np.sqrt(best_match['vx']**2 + best_match['vy']**2),
                    'source': 'fused'
                })
            else:
                # No match: use pure camera detection
                fused_objects.append({
                    'bbox': cam_bbox,
                    'class': cam_det['class'],
                    'score': cam_det['score'],
                    'depth': None,
                    'velocity': None,
                    'source': 'camera'
                })
        
        return fused_objects
```

---

## 3. LiDAR-Camera Fusion

### 3.1 Deep Fusion Approaches

#### Approach 1: Late Fusion

Each sensor detects independently; fusion at the decision level:

```python
class LateFusionLidarCamera:
    def __init__(self):
        self.detector_2d = TwoDDetector()  # 2D object detector
        self.detector_3d = ThreeDDetector()  # 3D object detector
        
    def fuse(self, image, point_cloud):
        # Independent detection
        dets_2d = self.detector_2d.detect(image)  # 2D bbox
        dets_3d = self.detector_3d.detect(point_cloud)  # 3D bbox
        
        # Match and associate
        fused = self.match_and_fuse(dets_2d, dets_3d)
        
        return fused
    
    def match_and_fuse(self, dets_2d, dets_3d):
        """IoU-based matching fusion"""
        fused = []
        
        for det_2d in dets_2d:
            best_iou = 0
            best_3d = None
            
            for det_3d in dets_3d:
                # Project 3D to 2D and compute IoU
                iou = self.compute_iou_2d(det_2d.bbox, det_3d.bbox_2d)
                
                if iou > best_iou:
                    best_iou = iou
                    best_3d = det_3d
            
            if best_iou > 0.3:
                # Fuse
                fused_obj = self.merge_detection(det_2d, best_3d)
                fused.append(fused_obj)
            else:
                fused.append(det_2d)
        
        return fused
```

#### Approach 2: Early Fusion

Feature-level fusion:

```python
class EarlyFusionLidarCamera(nn.Module):
    def __init__(self):
        # Vision branch
        self.visual_encoder = VisualEncoder()
        
        # LiDAR branch (BEV features)
        self.lidar_encoder = LiDAREncoderBEV()
        
        # Fusion module
        self.fusion_transformer = FusionTransformer(d_model=256)
        
        # Detection head
        self.detection_head = DetectionHead()
        
    def forward(self, image, point_cloud):
        # Visual features
        visual_feat = self.visual_encoder(image)  # (B, C, H, W)
        
        # LiDAR BEV features
        lidar_feat = self.lidar_encoder(point_cloud)  # (B, C, H', W')
        
        # Align sizes
        if visual_feat.shape[2:] != lidar_feat.shape[2:]:
            visual_feat = nn.functional.interpolate(
                visual_feat, size=lidar_feat.shape[2:]
            )
        
        # Feature fusion
        fused_feat = self.fusion_transformer(visual_feat, lidar_feat)
        
        # Detection
        detections = self.detection_head(fused_feat)
        
        return detections
```

### 3.2 Point Cloud-to-Image Projection

```python
def project_lidar_to_image(points_3d, K, R, t, image_shape):
    """
    Project 3D point cloud to a 2D image
    
    points_3d: (N, 3) - points in the LiDAR frame
    K: (3, 3) - camera intrinsics
    R, t: camera extrinsics
    """
    # Convert to camera frame
    points_cam = (R @ points_3d.T + t.reshape(3, 1)).T  # (N, 3)
    
    # Filter points behind the camera
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]
    
    # Project to the image plane
    points_img = (K @ points_cam[:, :3].T).T  # (N, 3)
    uvw = points_img / points_img[:, 2:3]
    
    # Pixel coordinates
    u = uvw[:, 0]
    v = uvw[:, 1]
    
    # Filter points within the image
    h, w = image_shape
    in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    
    # Return results
    return {
        'u': u[in_image],
        'v': v[in_image],
        'depth': points_cam[in_image, 2],
        'intensity': points_3d[valid][in_image, 3] if points_3d.shape[1] > 3 else None
    }
```

---

## 4. Multi-Sensor Time Synchronization

### 4.1 Hardware Synchronization

```
Sensor time synchronization schemes:
1. GPS/PPS clock synchronization
2. Hardware trigger synchronization
3. Software timestamp alignment
```

### 4.2 Software Time Synchronization

```python
import time
from collections import deque

class TimeSynchronizer:
    def __init__(self, tolerance=0.05):  # 50ms tolerance
        self.tolerance = tolerance
        self.buffers = {
            'camera': deque(),
            'lidar': deque(),
            'radar': deque()
        }
        
    def register(self, sensor_name, data, timestamp):
        """Register sensor data"""
        self.buffers[sensor_name].append({
            'data': data,
            'timestamp': timestamp
        })
        
    def get_synchronized(self, timestamp):
        """Get synchronized data"""
        synced = {}
        
        for sensor, buffer in self.buffers.items():
            if not buffer:
                continue
            
            # Find data closest to the timestamp
            best_idx = 0
            best_diff = float('inf')
            
            for i, item in enumerate(buffer):
                diff = abs(item['timestamp'] - timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            
            if best_diff <= self.tolerance:
                synced[sensor] = buffer[best_idx]['data']
                
                # Clean up old data
                while best_idx > 0:
                    buffer.popleft()
                    best_idx -= 1
        
        return synced
```

---

## 5. Spatial Alignment

### 5.1 Extrinsic Calibration

```python
class ExtrinsicCalibration:
    def __init__(self):
        pass
    
    def calibrate_lidar_camera(self, lidar_points, camera_image, K):
        """
        LiDAR-camera extrinsic calibration
        Approach: calibration board
        """
        # Detect checkerboard corners in the image
        corners_2d = self.detect_checkerboard(camera_image)
        
        # Detect checkerboard corners in the point cloud
        corners_3d = self.detect_lidar_corners(lidar_points)
        
        # Solve extrinsics with PnP
        success, R, t = cv2.solvePnP(
            corners_3d, corners_2d, K, distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        return R, t
    
    def calibrate_radar_camera(self, radar_data, camera_image):
        """
        Radar-camera extrinsic calibration
        Approach: corner reflectors
        """
        # Extract static targets (corner reflectors) from radar
        static_targets = self.extract_static_targets(radar_data)
        
        # Corner reflector positions in the image
        corner_reflectors = self.detect_corner_reflectors(camera_image)
        
        # Solve the transformation matrix
        # Use EPnP or another PnP variant
        pass
```

### 5.2 Coordinate Transformations

```python
import tf
import numpy as np

class CoordinateTransformer:
    def __init__(self):
        self.tf_listener = tf.TransformListener()
        
    def transform_point(self, point, from_frame, to_frame):
        """Coordinate transformation"""
        # Create a point message
        pt = PointStamped()
        pt.header.frame_id = from_frame
        pt.header.stamp = self.tf_listener.getLatestCommonTime(to_frame, from_frame)
        pt.point.x, pt.point.y, pt.point.z = point
        
        # Transform
        transformed = self.tf_listener.transformPoint(to_frame, pt)
        return [transformed.point.x, transformed.point.y, transformed.point.z]
    
    def lidar_to_camera_frame(self, lidar_point, T_lidar_to_camera):
        """Convert a LiDAR point to camera coordinates"""
        point_h = np.append(lidar_point, 1)
        camera_point = T_lidar_to_camera @ point_h
        return camera_point[:3]
```

---

## 6. Fusion Algorithms

### 6.1 Kalman Filter Fusion

```python
class KalmanFusion:
    def __init__(self, state_dim=7, meas_dim=4):
        # State: [x, y, z, vx, vy, vz, w]
        self.kf = KalmanFilter(state_dim, meas_dim)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 1, 0],  # z = z + vz
            [0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 1, 0],  # vz = vz
            [0, 0, 0, 0, 0, 0, 1]   # w = w
        ])
        
        # Observation matrix (camera observes position only)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
    def update_with_camera(self, bbox_2d, depth):
        """Update with camera detection"""
        measurement = np.array([bbox_2d[0], bbox_2d[1], depth, bbox_2d[2]])
        self.kf.update(measurement)
        
    def update_with_radar(self, range, angle, doppler):
        """Update with radar"""
        # Radar coordinate conversion
        x = range * np.cos(angle)
        y = range * np.sin(angle)
        
        measurement = np.array([x, y, doppler, 0])
        self.kf.update(measurement)
```

### 6.2 Extended Kalman Filter (EKF)

```python
class ExtendedKalmanFilter:
    def __init__(self):
        self.x = None  # state
        self.P = None  # covariance
        
    def predict(self, dt):
        """Prediction step"""
        # Nonlinear state transition
        self.x[3] += self.x[6] * dt  # vx
        self.x[4] += self.x[6] * dt  # vy
        
        # Jacobian matrix
        F = np.eye(7)
        F[3, 6] = dt
        F[4, 6] = dt
        
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z, sensor_type):
        """Update step"""
        # Observation function
        if sensor_type == 'camera':
            h = np.array([self.x[0], self.x[1], self.x[2], self.x[6]])
        elif sensor_type == 'radar':
            h = np.array([
                np.sqrt(self.x[0]**2 + self.x[1]**2),
                np.arctan2(self.x[1], self.x[0]),
                self.x[3]
            ])
        
        # Jacobian matrix
        H = self.compute_jacobian(sensor_type)
        
        # Update
        y = z - h
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ H) @ self.P
```

---

## References

1. Chen, X., et al. (2017). Multi-View 3D Object Detection Network for Autonomous Driving. CVPR.
2. Nobis, F., et al. (2019). A Deep Learning-based Radar and Camera Sensor Fusion Architecture for Object Detection. IEEE.
3. Qi, C. R., et al. (2018). Frustum PointNets for 3D Object Detection from RGB-D Data. CVPR.

---

*This chapter is continuously updated...*
