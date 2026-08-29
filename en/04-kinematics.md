# Kinematics

Kinematics is the study of robot motion without considering forces and mass. Understanding kinematics is the foundation for robot control and planning.

## 📋 Contents

- [1. Basic Concepts](#1-basic-concepts)
- [2. DH Parameters](#2-dh-parameters)
- [3. Forward Kinematics](#3-forward-kinematics)
- [4. Inverse Kinematics](#4-inverse-kinematics)
- [5. Jacobian Matrix](#5-jacobian-matrix)
- [6. Python Implementation](#6-python-implementation)

---

## 1. Basic Concepts

### 1.1 Rigid Body Motion

```
┌─────────────────────────────────────────────────────────────┐
│                    刚体位姿描述                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  位姿 = 位置 + 姿态（Orientation）                         │
│                                                             │
│  位置：在三维空间中的坐标 (x, y, z)                         │
│                                                             │
│  姿态表示方法：                                              │
│  1. 旋转矩阵 R (3x3)                                        │
│  2. 欧拉角 (roll, pitch, yaw)                              │
│  3. 四元数 q (qw, qx, qy, qz)                             │
│  4. 轴角表示 (axis, angle)                                 │
│                                                             │
│  齐次变换矩阵 T (4x4)：                                     │
│       ┌           ┐                                        │
│       │ R   p     │                                        │
│       │           │                                        │
│       │ 0   1     │                                        │
│       └           ┘                                        │
│  其中 R 是旋转矩阵，p 是位置向量                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Pose = position + orientation. Position: coordinates (x, y, z) in 3D space. Orientation representations: rotation matrix R (3x3), Euler angles (roll, pitch, yaw), quaternion q (qw, qx, qy, qz), axis-angle (axis, angle). Homogeneous transformation matrix T (4x4) with R the rotation matrix and p the position vector.)*

### 1.2 Rotation Representations

#### Euler Angles

```
ZYX Euler angles (Roll-Pitch-Yaw):

1. Rotate about Z axis: yaw (ψ)
2. Rotate about Y axis: pitch (θ)
3. Rotate about X axis: roll (φ)

Rotation matrix:
R = Rz(ψ) * Ry(θ) * Rx(φ)

Problem: Gimbal Lock
When pitch = ±90°, one degree of freedom is lost
```

#### Quaternions

```
Quaternion q = [qw, qx, qy, qz] = qw + qx*i + qy*j + qz*k

Constraint: ||q|| = 1 (unit quaternion)

Advantages:
• No gimbal lock
• Smooth interpolation (SLERP)
• High computational efficiency
```

---

## 2. DH Parameters

### 2.1 DH Parameter Definition

DH parameters (Denavit-Hartenberg) are a systematic way to describe robot joints and links.

```
┌─────────────────────────────────────────────────────────────┐
│                    DH参数定义                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  四个参数描述相邻连杆关系：                                  │
│                                                             │
│  • θ (theta)：关节角 - 绕zi轴旋转角度（旋转关节变量）        │
│  • d：连杆偏距 - 沿zi轴的位移（移动关节变量）                │
│  • a：连杆长度 - 沿xi轴的距离                               │
│  • α (alpha)：连杆扭转角 - 绕xi轴旋转角度                   │
│                                                             │
│  连杆变换矩阵：                                              │
│                                                             │
│       i-1        ┌                        ┐                │
│      T    =      │ cθ  -sθcα   sθsα   a*cθ│                │
│       i          │ sθ   cθcα  -cθsα   a*sθ│                │
│                    │ 0    sα     cα      d │                │
│                    │ 0     0      0      1 │                │
│                    └                        ┘                │
│  其中 cθ = cos(θ), sθ = sin(θ)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Four parameters describe the relationship between adjacent links: θ — joint angle (rotation about the z_i axis, the variable for revolute joints); d — link offset (translation along z_i, the variable for prismatic joints); a — link length (distance along x_i); α — link twist (rotation about the x_i axis).)*

### 2.2 Example DH Table

Taking a common 6-axis robot arm as an example:

| Joint | θ | d | a | α |
|-------|---|---|---|---|
| 1 | θ1* | d1 | 0 | -90° |
| 2 | θ2* | 0 | a2 | 0° |
| 3 | θ3* | 0 | a3 | 0° |
| 4 | θ4* | d4 | 0 | -90° |
| 5 | θ5* | 0 | 0 | 90° |
| 6 | θ6* | d6 | 0 | 0° |

\* joint variable (variable)

---

## 3. Forward Kinematics

### 3.1 Definition

**Forward Kinematics**: given joint angles, compute the position and orientation of the end-effector.

```
Input: joint angles q = [θ1, θ2, ..., θn]
Output: end-effector pose T = [position p, rotation R]
```

### 3.2 Computation

```
┌─────────────────────────────────────────────────────────────┐
│                   正运动学计算                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  末端位姿 = 所有连杆变换矩阵的乘积                           │
│                                                             │
│       0        0   1   2        n-1  n                     │
│      T   =  T   * T   * ... * T                            │
│       n        1   2   3        n    n+1                   │
│                                                             │
│  或表示为：                                                  │
│       n                                                       │
│      T   = Π  i-1 T                                          │
│       0   i=1    i                                           │
│                                                             │
│  例：2R机械臂（两个旋转关节）                                │
│                                                             │
│      x = L1*cos(θ1) + L2*cos(θ1+θ2)                        │
│      y = L1*sin(θ1) + L2*sin(θ1+θ2)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(End-effector pose = product of all link transformation matrices. Example (2R arm): x = L1·cos(θ1) + L2·cos(θ1+θ2); y = L1·sin(θ1) + L2·sin(θ1+θ2))*

### 3.3 2D Illustration

```
2R arm forward kinematics:

        End-effector
            ●
           /|
          / | L2
         /  |
        /θ2 |
       ●────┘
       |   
   L1  |   
       |   
       ●─────┐
        θ1   │
        Base │
```

---

## 4. Inverse Kinematics

### 4.1 Definition

**Inverse Kinematics**: given the target pose of the end-effector, compute the required joint angles.

```
Input: target pose T_target = [position p, rotation R]
Output: joint angles q = [θ1, θ2, ..., θn]
```

### 4.2 Solution Methods

```
┌─────────────────────────────────────────────────────────────┐
│                   逆运动学求解方法                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 解析法（Analytical）                                    │
│     • 几何关系推导                                          │
│     • 计算快，精度高                                        │
│     • 仅适用于简单结构                                      │
│     • 可能有闭式解或无解                                    │
│                                                             │
│  2. 数值法（Numerical）                                     │
│     • 雅可比迭代法                                          │
│     • 牛顿-拉夫森法                                         │
│     • 适用于任意结构                                        │
│     • 可能陷入局部最优                                      │
│                                                             │
│  3. 优化法                                                  │
│     • 将IK转化为优化问题                                    │
│     • 可加入约束（关节限位、避障）                          │
│     • 计算较慢                                              │
│                                                             │
│  多解性：                                                    │
│  同一位姿可能对应多组关节角度                                │
│  选择准则：最接近当前姿态、无碰撞、最小能耗                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(1. Analytical: derived from geometric relations — fast and precise, but only for simple structures; may have closed-form solutions or none. 2. Numerical: Jacobian iteration, Newton-Raphson — works for any structure, may fall into local optima. 3. Optimization: cast IK as an optimization problem, can add constraints (joint limits, obstacle avoidance), slower. Multiple solutions: the same pose may correspond to multiple joint-angle sets; selection criteria: closest to the current pose, collision-free, minimal energy.)*

### 4.3 Analytical IK for a 2R Arm

```
Target: (x, y)
Link lengths: L1, L2

Steps:
1. Compute the distance from the target to the base
   r = √(x² + y²)
   
2. Check reachability
   If |L1 - L2| > r > L1 + L2, unreachable

3. Law of cosines for θ2
   cos(θ2) = (x² + y² - L1² - L2²) / (2*L1*L2)
   θ2 = ±acos(...)  // two solutions: elbow up / elbow down

4. Solve for θ1
   β = atan2(y, x)
   ψ = atan2(L2*sin(θ2), L1 + L2*cos(θ2))
   θ1 = β - ψ
```

---

## 5. Jacobian Matrix

### 5.1 Definition

The Jacobian matrix describes the linear relationship between end-effector velocity and joint velocity.

```
┌─────────────────────────────────────────────────────────────┐
│                    雅可比矩阵                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  速度关系：                                                  │
│  ┌    ┐       ┌    ┐                                       │
│  │ ẋ  │       │     │                                      │
│  │    │  = J *│  q̇  │                                     │
│  │ ω  │       │     │                                      │
│  └    ┘       └    ┘                                       │
│                                                             │
│  其中：                                                      │
│  • ẋ：末端线速度 (3x1)                                      │
│  • ω：末端角速度 (3x1)                                      │
│  • q̇：关节速度 (nx1)                                       │
│  • J：雅可比矩阵 (6xn)                                      │
│                                                             │
│  雅可比矩阵：                                                │
│       ┌      ┐                                              │
│       │ J_v  │   (线速度部分)                               │
│  J =  │      │                                              │
│       │ J_ω  │   (角速度部分)                               │
│       └      ┘                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Velocity relationship: [ẋ; ω] = J · q̇, where ẋ is the end-effector linear velocity (3x1), ω the angular velocity (3x1), q̇ the joint velocity (nx1), and J the Jacobian (6xn) composed of a linear velocity part J_v and an angular velocity part J_ω.)*

### 5.2 Computing the Jacobian

For revolute joint i:

```
Linear velocity component: J_vi = zi × (p_e - pi)

Angular velocity component: J_ωi = zi

where:
• zi: axis direction of joint i
• pi: origin position of joint i
• p_e: end-effector position
```

### 5.3 Singularities

```
When det(J) = 0, the robot is in a singular configuration:

Types:
1. Boundary singularity: end-effector at the workspace boundary
2. Interior singularity: joint axes are collinear

Effects:
• Loss of mobility in some directions
• Joint velocities may tend toward infinity
• Need to be avoided or specially handled
```

---

## 6. Python Implementation

### 6.1 Base Class

```python
import numpy as np
from typing import Tuple, List, Optional

class RobotKinematics:
    """Robot kinematics computation class"""
    
    def __init__(self, dh_params: np.ndarray):
        """
        Initialize robot kinematics.
        
        Args:
            dh_params: DH parameter table, shape (n_joints, 4)
                       each row is [a, alpha, d, theta_offset]
        """
        self.dh_params = dh_params
        self.n_joints = len(dh_params)
        
    def dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """
        Compute a single DH transformation matrix.
        
        Args:
            a: link length
            alpha: link twist
            d: link offset
            theta: joint angle
        
        Returns:
            4x4 homogeneous transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct,   -st*ca,   st*sa,   a*ct],
            [st,    ct*ca,  -ct*sa,   a*st],
            [0,     sa,      ca,      d   ],
            [0,     0,       0,       1   ]
        ])
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward kinematics.
        
        Args:
            joint_angles: joint angle array, shape (n_joints,)
        
        Returns:
            position: end-effector position [x, y, z]
            rotation: end-effector rotation matrix 3x3
        """
        T = np.eye(4)
        
        for i in range(self.n_joints):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i
        
        position = T[:3, 3]
        rotation = T[:3, :3]
        
        return position, rotation
    
    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix.
        
        Args:
            joint_angles: joint angles
        
        Returns:
            6 x n_joints Jacobian matrix
        """
        J = np.zeros((6, self.n_joints))
        
        # Compute the transformation of each joint to the base
        T = [np.eye(4)]
        for i in range(self.n_joints):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T.append(T[-1] @ T_i)
        
        # End-effector position
        p_e = T[-1][:3, 3]
        
        # Compute each column of the Jacobian
        for i in range(self.n_joints):
            z_i = T[i][:3, 2]  # z axis of joint i
            p_i = T[i][:3, 3]  # position of joint i
            
            # Linear velocity part
            J[:3, i] = np.cross(z_i, p_e - p_i)
            # Angular velocity part
            J[3:, i] = z_i
        
        return J


class TwoRManipulator:
    """2R planar arm (simplified model)"""
    
    def __init__(self, L1: float = 1.0, L2: float = 1.0):
        """
        Args:
            L1: length of the first link
            L2: length of the second link
        """
        self.L1 = L1
        self.L2 = L2
    
    def forward_kinematics(self, theta1: float, theta2: float) -> Tuple[float, float, float]:
        """
        Forward kinematics.
        
        Args:
            theta1: first joint angle (radians)
            theta2: second joint angle (radians)
        
        Returns:
            x, y, phi: end-effector position and orientation angle
        """
        x = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        y = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta1 + theta2)
        phi = theta1 + theta2  # end-effector orientation angle
        
        return x, y, phi
    
    def inverse_kinematics(self, x: float, y: float, 
                          elbow_up: bool = True) -> Optional[Tuple[float, float]]:
        """
        Inverse kinematics (analytical solution).
        
        Args:
            x, y: target position
            elbow_up: True for elbow-up solution, False for elbow-down
        
        Returns:
            (theta1, theta2) or None (unreachable)
        """
        # Distance to the target
        r_sq = x**2 + y**2
        r = np.sqrt(r_sq)
        
        # Check reachability
        if r > self.L1 + self.L2 or r < abs(self.L1 - self.L2):
            print(f"Target ({x}, {y}) is outside the workspace")
            return None
        
        # Law of cosines for theta2
        cos_theta2 = (r_sq - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # numerical stability
        
        if elbow_up:
            theta2 = -np.arccos(cos_theta2)  # elbow up
        else:
            theta2 = np.arccos(cos_theta2)   # elbow down
        
        # Compute theta1
        beta = np.arctan2(y, x)
        psi = np.arctan2(self.L2 * np.sin(theta2), 
                        self.L1 + self.L2 * np.cos(theta2))
        theta1 = beta - psi
        
        return theta1, theta2
    
    def jacobian(self, theta1: float, theta2: float) -> np.ndarray:
        """
        Compute the Jacobian matrix.
        
        Returns:
            2x2 Jacobian matrix (position only)
        """
        J = np.array([
            [-self.L1*np.sin(theta1) - self.L2*np.sin(theta1+theta2), 
             -self.L2*np.sin(theta1+theta2)],
            [self.L1*np.cos(theta1) + self.L2*np.cos(theta1+theta2),  
             self.L2*np.cos(theta1+theta2)]
        ])
        return J
    
    def jacobian_inverse_ik(self, x_target: float, y_target: float,
                           theta_init: Tuple[float, float],
                           max_iter: int = 100,
                           tol: float = 1e-4) -> Optional[Tuple[float, float]]:
        """
        Solve inverse kinematics by Jacobian iteration.
        
        Args:
            x_target, y_target: target position
            theta_init: initial joint angles
            max_iter: maximum iterations
            tol: convergence tolerance
        
        Returns:
            (theta1, theta2) or None
        """
        theta1, theta2 = theta_init
        alpha = 0.5  # step size
        
        for i in range(max_iter):
            # Current end-effector position
            x, y, _ = self.forward_kinematics(theta1, theta2)
            
            # Position error
            error = np.array([x_target - x, y_target - y])
            
            if np.linalg.norm(error) < tol:
                return theta1, theta2
            
            # Jacobian matrix
            J = self.jacobian(theta1, theta2)
            
            # Jacobian pseudoinverse
            try:
                J_pinv = np.linalg.pinv(J)
            except np.linalg.LinAlgError:
                print("Jacobian is singular!")
                return None
            
            # Update joint angles
            dtheta = alpha * J_pinv @ error
            theta1 += dtheta[0]
            theta2 += dtheta[1]
        
        print("Did not converge!")
        return None


# ============ Usage example ============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a 2R arm
    robot = TwoRManipulator(L1=1.0, L2=0.8)
    
    # Forward kinematics test
    theta1_test = np.pi/4
    theta2_test = np.pi/6
    x, y, phi = robot.forward_kinematics(theta1_test, theta2_test)
    print(f"Forward kinematics: θ1={np.degrees(theta1_test):.1f}°, θ2={np.degrees(theta2_test):.1f}°")
    print(f"End-effector: ({x:.3f}, {y:.3f}), orientation: {np.degrees(phi):.1f}°")
    
    # Inverse kinematics test
    x_target, y_target = 1.2, 0.8
    result = robot.inverse_kinematics(x_target, y_target, elbow_up=True)
    if result:
        theta1_ik, theta2_ik = result
        print(f"\nInverse kinematics: target ({x_target}, {y_target})")
        print(f"Solution: θ1={np.degrees(theta1_ik):.1f}°, θ2={np.degrees(theta2_ik):.1f}°")
        
        # Verify
        x_verify, y_verify, _ = robot.forward_kinematics(theta1_ik, theta2_ik)
        print(f"Verification: ({x_verify:.3f}, {y_verify:.3f})")
    
    # Visualize the workspace
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw workspace boundaries
    theta1_range = np.linspace(0, 2*np.pi, 100)
    for t2 in [0, np.pi]:
        r = np.sqrt(robot.L1**2 + robot.L2**2 + 2*robot.L1*robot.L2*np.cos(t2))
        x_boundary = r * np.cos(theta1_range)
        y_boundary = r * np.sin(theta1_range)
        ax.plot(x_boundary, y_boundary, 'b--', alpha=0.3)
    
    # Draw the arm pose
    def draw_robot(theta1, theta2, color='blue'):
        # First joint
        x1 = robot.L1 * np.cos(theta1)
        y1 = robot.L1 * np.sin(theta1)
        # End-effector
        x2, y2, _ = robot.forward_kinematics(theta1, theta2)
        
        ax.plot([0, x1], [0, y1], 'o-', color=color, linewidth=3, markersize=10)
        ax.plot([x1, x2], [y1, y2], 'o-', color=color, linewidth=3, markersize=10)
        ax.plot(x2, y2, 'r*', markersize=15)
    
    # Draw both IK solutions
    draw_robot(theta1_test, theta2_test, 'blue')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2R Manipulator Kinematics')
    plt.savefig('2r_kinematics.png')
    plt.show()
```

### 6.2 UR5 Example

```python
class UR5Kinematics:
    """UR5 robot arm kinematics"""
    
    def __init__(self):
        # UR5 DH parameters (meters and radians)
        self.dh_params = np.array([
            # a,      alpha,      d,         theta_offset
            [0.0,     np.pi/2,    0.089159,  0.0],
            [-0.425,  0.0,        0.0,       0.0],
            [-0.39225,0.0,        0.0,       0.0],
            [0.0,     np.pi/2,    0.10915,   0.0],
            [0.0,    -np.pi/2,    0.09465,   0.0],
            [0.0,     0.0,        0.0823,    0.0]
        ])
        self.robot = RobotKinematics(self.dh_params)
    
    def get_end_effector_pose(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get the end-effector pose"""
        return self.robot.forward_kinematics(joint_angles)
    
    def get_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Get the Jacobian matrix"""
        return self.robot.compute_jacobian(joint_angles)


# UR5 test
if __name__ == "__main__":
    ur5 = UR5Kinematics()
    
    # Zero pose
    joints_zero = np.array([0, 0, 0, 0, 0, 0])
    pos, rot = ur5.get_end_effector_pose(joints_zero)
    print(f"UR5 zero-pose end-effector position: {pos}")
    
    # Jacobian matrix
    J = ur5.get_jacobian(joints_zero)
    print(f"Jacobian shape: {J.shape}")
    print(f"Jacobian determinant: {np.linalg.det(J[:3, :3])}")
```

---

## Further Reading

### Textbooks
- *Robot Modeling and Control* — Spong, Hutchinson, Vidyasagar
- *Modern Robotics* — Lynch & Park

### Online Resources
- [Modern Robotics open textbook](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)
- [Peter Corke Robotics Toolbox](https://petercorke.com/toolboxes/robotics-toolbox/)

---

*This chapter is continuously updated...*
