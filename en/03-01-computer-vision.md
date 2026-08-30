# Computer Vision Fundamentals

This chapter introduces the fundamentals of computer vision, including image processing, feature extraction, and classic algorithms — laying the groundwork for the perception techniques covered later.

## Contents

- [1. Image Basics](#1-image-basics)
- [2. Image Preprocessing](#2-image-preprocessing)
- [3. Feature Extraction & Description](#3-feature-extraction--description)
- [4. Image Classification & Detection](#4-image-classification--detection)
- [5. Semantic Segmentation](#5-semantic-segmentation)
- [6. Object Tracking](#6-object-tracking)

---

## 1. Image Basics

### 1.1 Digital Image Representation

A digital image is a 2D matrix of pixels, usually represented as an array:

```
Image I(x, y) denotes the pixel value at coordinates (x, y)
```

- **Grayscale image**: single channel, pixel values 0-255 (8-bit)
- **RGB image**: three channels, red / green / blue components
- **Depth image**: each pixel represents the distance to an object

### 1.2 Camera Model

#### Pinhole Camera Model

A camera projects 3D world coordinates onto a 2D image plane:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z} \begin{bmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
$$

where:
- $(X, Y, Z)$: 3D world coordinates
- $(u, v)$: 2D image coordinates
- $f$: focal length
- $(c_x, c_y)$: principal point

#### Intrinsic Matrix

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

### 1.3 Coordinate Frame Transformations

| Frame | Description |
|-------|-------------|
| World frame | Absolute reference frame in 3D space |
| Camera frame | Frame centered at the camera |
| Image frame | 2D image plane frame |
| Pixel frame | Discrete pixel coordinates of the image |

---

## 2. Image Preprocessing

### 2.1 Grayscale Conversion

```python
import cv2
import numpy as np

def rgb_to_gray(image):
    """RGB to grayscale - weighted average method"""
    # Standard weights: Y = 0.299*R + 0.587*G + 0.114*B
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return gray.astype(np.uint8)

# Using OpenCV
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### 2.2 Image Filtering

#### Gaussian Filter

For noise removal and smoothing:

```python
def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """Gaussian filter"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
```

#### Median Filter

Effective against salt-and-pepper noise:

```python
def median_filter(image, kernel_size=5):
    """Median filter"""
    return cv2.medianBlur(image, kernel_size)
```

#### Bilateral Filter

Removes noise while preserving edges:

```python
def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Bilateral filter"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
```

### 2.3 Image Enhancement

#### Histogram Equalization

Enhances image contrast:

```python
def histogram_equalization(gray_image):
    """Histogram equalization"""
    # Global equalization
    equ = cv2.equalizeHist(gray_image)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray_image)
    return cl1
```

---

## 3. Feature Extraction & Description

### 3.1 Corner Detection

#### Harris Corner Detection

```python
def harris_corner_detection(gray_image, threshold=0.01):
    """Harris corner detection"""
    # Convert to float32
    gray = np.float32(gray_image)
    
    # Compute Harris response
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    
    # Thresholding
    dst = cv2.dilate(dst, None)
    corners = dst > threshold * dst.max()
    
    return corners
```

#### Shi-Tomasi Corner Detection

```python
def shi_tomasi_detection(gray_image, max_corners=100, quality_level=0.01):
    """Shi-Tomasi corner detection"""
    corners = cv2.goodFeaturesToTrack(gray_image, 
                                       max_corners, 
                                       quality_level, 
                                       minDistance=10)
    return corners
```

### 3.2 SIFT Features

Scale-Invariant Feature Transform:

```python
def sift_features(image):
    """SIFT feature extraction"""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors
```

**SIFT characteristics**:
- Scale invariance
- Rotation invariance
- Robust to illumination changes

### 3.3 ORB Features

Oriented FAST and Rotated BRIEF — ideal for real-time applications:

```python
def orb_features(image, n_features=1000):
    """ORB feature extraction"""
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors
```

**ORB characteristics**:
- Fast computation
- Low memory footprint
- Rotation invariance

### 3.4 Feature Matching

```python
def feature_matching(image1, image2, method='bf'):
    """Feature matching"""
    # Extract features
    kp1, des1 = orb_features(image1)
    kp2, des2 = orb_features(image2)
    
    # BFMatcher (Brute Force)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, matches
```

---

## 4. Image Classification & Detection

### 4.1 Classical Methods

#### HOG Features + SVM

```python
def hog_svm_classification(image):
    """HOG features + SVM classification"""
    # Extract HOG features
    win_size = (64, 64)
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, 
                           cell_size, num_bins)
    features = hog.compute(image)
    
    # Classification (requires a pre-trained SVM)
    # prediction = svm.predict(features)
    return features
```

### 4.2 Deep Learning Methods

#### CNN Image Classification

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

#### YOLO Object Detection

```python
# Object detection with YOLOv5
def yolo_detection(image_path, model_path='yolov5s.pt'):
    """YOLO object detection"""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    # Detect
    results = model(image_path)
    
    # Parse results
    detections = results.pandas().xyxy[0]
    return detections
```

---

## 5. Semantic Segmentation

### 5.1 FCN (Fully Convolutional Network)

```python
class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()
        # Use VGG as the backbone
        vgg = models.vgg16(pretrained=True)
        
        # Encoder
        self.pool3 = vgg.features[:17]  # 1/8
        self.pool4 = vgg.features[17:24]  # 1/16
        self.pool5 = vgg.features[24:]  # 1/32
        
        # Decoder
        self.conv6 = nn.Conv2d(512, 4096, 1)
        self.conv7 = nn.Conv2d(4096, 4096, 1)
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8)
```

### 5.2 DeepLab Family

- **DeepLabv3+**: uses ASPP (Atrous Spatial Pyramid Pooling) with an encoder-decoder structure
- **Features**: multi-scale atrous convolutions, strong contextual information

---

## 6. Object Tracking

### 6.1 Kalman Filter Tracking

```python
class KalmanTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])  # state transition matrix
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])  # observation matrix
        self.kf.P *= 10
        self.kf.R *= 0.1
        
    def predict(self):
        return self.kf.predict()
    
    def update(self, measurement):
        return self.kf.update(measurement)
```

### 6.2 SORT Tracking

Simple Online and Realtime Tracking:

```python
def sort_tracking(detections, iou_threshold=0.3):
    """SORT object tracking"""
    # 1. Predict: predict positions of all tracked boxes with Kalman filter
    # 2. Match: match detections to tracks with the Hungarian algorithm
    # 3. Update: update matched track states
    # 4. Manage: add new tracks, remove lost tracks
    pass
```

---

## References

1. Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. IJCV.
2. Rublee, E., et al. (2011). ORB: An efficient alternative to SIFT or SURF. ICCV.
3. Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR.
4. Long, J., et al. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.

---

*This section is continuously updated...*
