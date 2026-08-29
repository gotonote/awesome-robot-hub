# 3D Perception & Depth Estimation

3D perception enables robots to understand 3D spatial structure — the foundation for tasks such as navigation, grasping, and environment reconstruction. This chapter covers depth estimation, 3D object detection, and point cloud processing.

## Contents

- [1. Depth Estimation](#1-depth-estimation)
- [2. 3D Object Detection](#2-3d-object-detection)
- [3. Point Cloud Processing](#3-point-cloud-processing)
- [4. Stereo Vision](#4-stereo-vision)
- [5. NeRF & 3D Reconstruction](#5-nerf--3d-reconstruction)

---

## 1. Depth Estimation

### 1.1 Classical Methods

#### Triangulation

```
Depth Z = (f × baseline B) / disparity d
```

```python
import numpy as np

def depth_from_disparity(disparity, baseline=0.5, focal_length=500):
    """Compute depth from disparity"""
    # Avoid division by zero
    disparity = np.where(disparity > 0, disparity, 0.001)
    depth = (baseline * focal_length) / disparity
    return depth
```

#### Stereo Matching

```python
import cv2

def stereo_matching(left_img, right_img, method='sgbm'):
    """Stereo matching"""
    if method == 'sgbm':
        # SGBM (Semi-Global Block Matching)
        window_size = 3
        min_disp = 0
        max_disp = 64
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=max_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        disparity = stereo.compute(left_img, right_img)
        return disparity.astype(np.float32) / 16.0
    
    elif method == 'bm':
        # Block Matching
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        disparity = stereo.compute(left_img, right_img)
        return disparity.astype(np.float32)
```

### 1.2 Deep Learning Methods

#### Monodepth (Monocular Depth Estimation)

```python
import torch
import torch.nn as nn

class Monodepth(nn.Module):
    def __init__(self, num_layers=18, pretrained=True):
        super(Monodepth, self).__init__()
        
        # Use ResNet as the encoder
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # outputs 0-1 depth probability
        )
        
    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth
```

#### MiDaS (Multi-Task Depth Estimation)

```python
def estimate_depth_midas(image):
    """Depth estimation with MiDaS"""
    # MiDaS supports multiple input types
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
    model.eval()
    
    # Preprocessing
    input_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        depth = model(input_tensor)
    
    return depth
```

### 1.3 Depth Estimation Losses

```python
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
        
    def forward(self, pred_depth, gt_depth):
        # Depth smoothness loss
        smooth_loss = self.compute_smooth_loss(pred_depth)
        
        # Disparity consistency loss (if two frames)
        # ssim_loss = self.compute_ssim(pred_depth, gt_depth)
        
        # L1 loss
        l1_loss = torch.abs(pred_depth - gt_depth).mean()
        
        # Edge-aware smoothness
        edge_aware_smooth = self.edge_aware_smooth(pred_depth, gt_depth)
        
        return l1_loss + 0.1 * smooth_loss
    
    def compute_smooth_loss(self, depth):
        """Depth smoothness loss"""
        grad_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        grad_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        return grad_x.mean() + grad_y.mean()
    
    def edge_aware_smooth(self, depth, image):
        """Edge-aware smoothness loss"""
        # Reduce the smoothness constraint near image edges
        pass
```

---

## 2. 3D Object Detection

### 2.1 Camera-Based 3D Detection

#### 3D Bounding Box Representation

```
3D bounding box = (x, y, z, l, w, h, yaw)
where:
- (x, y, z): center position
- (l, w, h): length, width, height
- yaw: yaw angle
```

#### F-PointNet

```python
class FPointNet(nn.Module):
    def __init__(self, num_classes=3):
        super(FPointNet, self).__init__()
        
        # 2D detector
        self.rcnn = RCNN2D()
        
        # Frustum proposal
        self.frustum_proposal = FrustumProposal()
        
        # Feature extraction
        self.pointnet = PointNetSetAbstraction(
            npoint=1024,
            radius=0.2,
            nsample=32,
            in_channel=3,
            mlp=[32, 32, 64],
            group_all=False
        )
        
        # Classification and regression heads
        self.cls_head = nn.Linear(64, num_classes)
        self.reg_head = nn.Linear(64, 3 + 2)  # center offset + size
    
    def forward(self, image, point_cloud):
        # 2D detection
        box2d = self.rcnn(image)
        
        # Generate frustum
        frustum = self.frustum_proposal(box2d, image.shape)
        
        # Crop point cloud
        roi_points = self.crop_point_cloud(point_cloud, frustum)
        
        # Feature extraction
        features, _ = self.pointnet(roi_points)
        
        # Prediction
        cls_score = self.cls_head(features)
        bbox3d = self.reg_head(features)
        
        return cls_score, bbox3d
```

### 2.2 LiDAR 3D Detection

#### PointPillars

```python
class PointPillars(nn.Module):
    def __init__(self, num_classes=3):
        super(PointPillars, self).__init__()
        
        # Point cloud pillar network
        self.pillar_net = PillarNet(
            max_points=100,
            max_pillars=12000,
            grid_size=[0.16, 0.16, 4]
        )
        
        # Feature encoding
        self.scn = SparseConv3d(64, 64)
        
        # RPN
        self.rpn = RPN()
        
        # Detection head
        self.detection_head = DetectionHead(num_classes)
        
    def forward(self, point_cloud):
        # Pillarization
        pillars, coords = self.pillar_net(point_cloud)
        
        # Feature extraction
        features = self.scn(pillars)
        
        # RPN
        rpn_out = self.rpn(features)
        
        # Detection
        detections = self.detection_head(rpn_out)
        
        return detections
```

### 2.3 Multimodal 3D Detection

```python
class MultiModal3DDetection(nn.Module):
    def __init__(self):
        # Vision branch
        self.visual_branch = Visual3DBranch()
        
        # LiDAR branch
        self.lidar_branch = LiDAR3DBranch()
        
        # Fusion module
        self.fusion = AttentionFusion(dim=256)
        
        # Detection head
        self.head = DetectionHead(num_classes=3)
        
    def forward(self, image, point_cloud):
        # Visual 3D detection
        visual_boxes = self.visual_branch(image)
        
        # LiDAR 3D detection
        lidar_boxes = self.lidar_branch(point_cloud)
        
        # Feature-level fusion
        fused_features = self.fusion(
            visual_boxes.features,
            lidar_boxes.features
        )
        
        # Final detection
        final_boxes = self.head(fused_features)
        
        return final_boxes
```

---

## 3. Point Cloud Processing

### 3.1 Basic Point Cloud Operations

```python
import open3d as o3d
import numpy as np

class PointCloudProcessor:
    def __init__(self):
        pass
        
    def load_point_cloud(self, path):
        """Load a point cloud"""
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    
    def downsample(self, points, voxel_size=0.05):
        """Voxel downsampling"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downpcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(downpcd.points)
    
    def remove_outliers(self, points, nb_neighbors=20, std_ratio=2.0):
        """Remove outliers"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return np.asarray(pcd.points)[ind]
    
    def estimate_normals(self, points, radius=0.1):
        """Estimate normals"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=30
            )
        )
        return np.asarray(pcd.normals)
```

### 3.2 Point Cloud Segmentation

```python
def point_cloud_segmentation(points, model="ransac", distance_threshold=0.2):
    """Point cloud segmentation"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if model == "ransac":
        # RANSAC plane segmentation
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        # Inlier points (on the plane)
        inlier_points = np.asarray(pcd.points)[inliers]
        
        # Remaining points
        outlier_points = np.asarray(pcd.points)[np.delete(np.arange(len(points)), inliers)]
        
        return plane_model, inlier_points, outlier_points
    
    elif model == "dbscan":
        # DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=10))
        return labels
```

### 3.3 Point Cloud Registration

```python
def point_cloud_registration(source, target, method="icp"):
    """Point cloud registration"""
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)
    
    if method == "icp":
        # ICP registration
        threshold = 0.02
        trans_init = np.eye(4)
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        return reg_p2p.transformation
    
    elif method == "feature":
        # Feature-based registration
        # Extract FPFH features
        source_fpfh = compute_fpfh(source_pcd)
        target_fpfh = compute_fpfh(target_pcd)
        
        # RANSAC registration
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd, target_pcd, source_fpfh, target_fpfh, True, 0.3
        )
        return result.transformation
```

---

## 4. Stereo Vision

### 4.1 Stereo Camera Calibration

```python
class StereoCalibration:
    def __init__(self):
        self.K1, self.D1 = None, None  # left camera intrinsics
        self.K2, self.D2 = None, None  # right camera intrinsics
        self.R, self.T = None, None    # relative extrinsics
        self.E, self.F = None, None    # essential and fundamental matrices
        
    def calibrate(self, left_images, right_images):
        """Stereo camera calibration"""
        object_points = []
        left_image_points = []
        right_image_points = []
        
        # Extract corners
        for left_img, right_img in zip(left_images, right_images):
            # Calibrate left and right cameras separately
            ret1, mtx1, dist1, _, _ = cv2.calibrateCamera(...)
            ret2, mtx2, dist2, _, _ = cv2.calibrateCamera(...)
            
            # Stereo calibration
            ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
                object_points, left_image_points, right_image_points,
                mtx1, dist1, mtx2, dist2, gray.shape[::-1]
            )
            
        return K1, D1, K2, D2, R, T
```

### 4.2 Optimized Disparity Computation

```python
def compute_disparity_optimized(left_img, right_img):
    """Optimized disparity computation"""
    # Preprocessing
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Left-right consistency check
    # ... implement left-right consistency check ...
    
    # Disparity post-processing
    # 1. Sub-pixel fitting
    # 2. Speckle filtering
    # 3. Occlusion handling
    
    return disparity
```

---

## 5. NeRF & 3D Reconstruction

### 5.1 NeRF Basics

Neural Radiance Fields (NeRF) represent 3D scenes with a neural network:

```python
class NeRF(nn.Module):
    def __init__(self, D=8, W=256):
        super(NeRF, self).__init__()
        
        # Positional encoding
        self.pos_encoding = PosEncoding(L=10)
        
        # Network structure
        layers = []
        for i in range(D):
            if i == 0:
                layers.append(nn.Linear(60, W))  # 3*10*2 = 60
            else:
                layers.append(nn.Linear(W, W))
            layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
        # Output heads
        self.sigma_head = nn.Linear(W, 1)  # density
        self.color_head = nn.Linear(W, 3)  # color
        
    def forward(self, x, d):
        """
        x: 3D position (N, 3)
        d: view direction (N, 3)
        """
        # Positional encoding
        x_enc = self.pos_encoding(x)
        d_enc = self.pos_encoding(d)
        
        # MLP
        h = self.mlp(x_enc)
        
        # Outputs
        sigma = self.sigma_head(h)
        color = torch.sigmoid(self.color_head(h))
        
        return color, sigma
```

### 5.2 3D Gaussian Splatting

```python
class Gaussian3D:
    """3D Gaussian Splatting representation"""
    def __init__(self):
        self.positions = None      # positions
        self.scales = None         # scales
        self.rotations = None      # rotations
        self.opacities = None      # opacities
        self.colors = None         # colors
        self.sh_coeffs = None      # spherical harmonic coefficients
        
    def render(self, camera):
        """Rasterization rendering"""
        # Project 3D Gaussians to 2D
        # Compute the color of each pixel
        pass
```

---

## References

1. Godard, C., et al. (2019). Digging Into Self-Supervised Monocular Depth Estimation. ICCV.
2. Qi, C. R., et al. (2018). Frustum PointNets for 3D Object Detection from RGB-D Data. CVPR.
3. Shi, S., et al. (2019). PointPillars: Fast Encoders for Object Detection from Point Clouds. CVPR.
4. Mildenhall, B., et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. ECCV.

---

*This chapter is continuously updated...*
