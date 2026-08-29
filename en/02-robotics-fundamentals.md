# Robotics Fundamentals

> This chapter introduces the most fundamental concepts in robotics: kinematics and dynamics. These are the foundation for understanding robot motion control.

## 1. Robotics Overview

### 1.1 What Is Robotics?

Robotics is an interdisciplinary field that studies the design, manufacture, operation, and application of robots.

```
┌─────────────────────────────────────────────────────────┐
│                      机器人系统                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐ │
│  │  感知   │ -> │  决策   │ -> │  控制   │ -> │  执行  │ │
│  │ Sensors │    │  Plan   │    │ Control │    │  Act   │ │
│  └─────────┘    └─────────┘    └─────────┘    └────────┘ │
└─────────────────────────────────────────────────────────┘
```

*(Sensors → Planning → Control → Actuation)*

### 1.2 Robot Joint Types

| Joint Type | Description | DOF |
|------------|-------------|-----|
| Revolute | Rotates about an axis | 1 DOF |
| Prismatic | Translates along an axis | 1 DOF |
| Spherical | 3D rotation | 3 DOF |

## 2. Kinematics Basics

### 2.1 Rigid Body Transformations

**Homogeneous transformation matrices** describe the pose of a rigid body in 3D space:

$$
T = \begin{bmatrix}
R & p \\
0 & 1
\end{bmatrix}
$$

where:
- $R$ is a $3 \times 3$ rotation matrix
- $p$ is a $3 \times 1$ position vector

### 2.2 Rotation Representations

#### Euler Angles
Rotation sequences about three coordinate axes:
- Roll: about the x axis
- Pitch: about the y axis
- Yaw: about the z axis

#### Quaternions
Solve the gimbal-lock problem:

$$
q = w + xi + yj + zk = [w, x, y, z]
$$

**Advantages**:
- Compact (4 parameters)
- No singularities
- Easy to interpolate

### 2.3 Forward Kinematics

**Definition**: given joint angles, compute the end-effector pose

```
Joint angles (θ1, θ2, θ3) → End-effector pose (position + orientation)
```

**DH parameter notation**:
| Parameter | Description |
|-----------|-------------|
| $a_i$ | Link length |
| $\alpha_i$ | Link twist |
| $d_i$ | Link offset |
| $\theta_i$ | Joint angle |

**Code example**:
```python
import numpy as np

def forward_kinematics_dh(theta, dh_params):
    """
    Compute forward kinematics using DH parameters.
    theta: list of joint angles
    dh_params: DH parameters (a, alpha, d)
    """
    T = np.eye(4)
    for i, (t, (a, alpha, d)) in enumerate(zip(theta, dh_params)):
        ct = np.cos(t)
        st = np.sin(t)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        Ti = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,     d   ],
            [0,   0,      0,      1   ]
        ])
        T = T @ Ti
    return T

# 2-joint arm example
dh_params = [(1, 0, 0), (1, 0, 0)]  # a, alpha, d
theta = [np.pi/4, np.pi/4]
T = forward_kinematics_dh(theta, dh_params)
print("End-effector pose:\n", T)
```

### 2.4 Inverse Kinematics

**Definition**: given the end-effector pose, compute joint angles

```
End-effector pose (position + orientation) → Joint angles (θ1, θ2, θ3)
```

**Analytical method** (closed-form solution):
```python
def inverse_kinematics_2link(x, y, l1, l2):
    """
    Analytical inverse kinematics for a 2R arm.
    """
    r = np.sqrt(x**2 + y**2)
    
    # Check reachability
    if r > l1 + l2 or r < abs(l1 - l2):
        return None
    
    # Law of cosines for joint 2 angle
    cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = np.arccos(np.clip(cos_theta2, -1, 1))
    
    # Joint 1 angle
    phi = np.arctan2(y, x)
    psi = np.arccos((r**2 + l1**2 - l2**2) / (2 * r * l1))
    theta1 = phi - psi
    
    return [theta1, theta2]
```

## 3. Dynamics Basics

### 3.1 Lagrangian Mechanics

**Lagrangian**:
$$L(q, \dot{q}) = T(q, \dot{q}) - V(q)$$

- $T$: kinetic energy
- $V$: potential energy

**Euler-Lagrange equation**:
$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = \tau_i$$

### 3.2 Robot Dynamics Equation

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau$$

| Symbol | Meaning |
|--------|---------|
| $M(q)$ | Inertia matrix |
| $C(q, \dot{q})$ | Coriolis/centrifugal matrix |
| $g(q)$ | Gravity vector |
| $\tau$ | Joint torques |

### 3.3 Inertia Matrix

For an n-joint robot:
$$M(q) = \sum_{i=1}^{n} J_i^T I_i J_i$$

### 3.4 Dynamics Code Example

```python
def compute_dynamics(q, q_dot, q_ddot, M, C, g):
    """
    Compute robot dynamics.
    q: joint positions
    q_dot: joint velocities
    q_ddot: joint accelerations
    """
    # Inertial force
    inertia_force = M @ q_ddot
    
    # Coriolis force
    coriolis_force = C @ q_dot
    
    # Gravity
    gravity_force = g
    
    # Total torque
    tau = inertia_force + coriolis_force + gravity_force
    return tau

# Example: 2-joint dynamics
M = np.array([[2, 0.5], [0.5, 1]])  # inertia matrix
C = np.array([[0, -0.5], [0.5, 0]])  # Coriolis matrix
g = np.array([0, 9.8])              # gravity

tau = compute_dynamics(
    q=np.array([0.5, 0.5]),
    q_dot=np.array([0.1, 0.1]),
    q_ddot=np.array([0, 0]),
    M=M, C=C, g=g
)
print("Required torque:", tau)
```

## 4. Jacobian Matrix

### 4.1 Definition

The **Jacobian matrix** represents the linear mapping between joint velocities and end-effector velocities:

$$
\dot{x} = J(q)\dot{q}
$$

### 4.2 Velocity Jacobian

```python
def compute_jacobian(robot, q):
    """
    Compute the robot Jacobian matrix.
    """
    n = len(q)
    J = np.zeros((6, n))
    
    # Simplified computation - the real version depends on the specific robot
    # This demonstrates the concept
    for i in range(n):
        # Contribution of each joint
        zi = robot.get_axis(i, q)  # joint axis direction
        pi = robot.get_link_origin(i, q)  # joint position
        pe = robot.get_end_effector(q)   # end-effector position
        
        # Linear velocity Jacobian
        J[:3, i] = np.cross(zi, pe - pi)
        # Angular velocity Jacobian
        J[3:, i] = zi
    
    return J
```

### 4.3 Singularities

When $\det(J) = 0$, the robot is in a **singular configuration**:
- Loses mobility in some direction
- Joint velocities tend toward infinity

## 5. Summary

```
┌────────────────────────────────────────────────────────┐
│                    运动学 vs 动力学                       │
├────────────────────────────────────────────────────────┤
│  运动学 (Kinematics)                                    │
│  - 描述运动关系，不考虑力                                │
│  - 正运动学: 关节 -> 末端                               │
│  - 逆运动学: 末端 -> 关节                               │
├────────────────────────────────────────────────────────┤
│  动力学 (Dynamics)                                      │
│  - 描述力和运动的关系                                    │
│  - 正向: 力矩 -> 运动                                   │
│  - 逆向: 运动 -> 所需力矩                               │
└────────────────────────────────────────────────────────┘
```

*(Kinematics: describes motion relationships without forces — forward: joints → end-effector, inverse: end-effector → joints. Dynamics: describes the relationship between forces and motion — forward: torque → motion, inverse: motion → required torque)*

## 6. Further Reading

- *Introduction to Robotics* — Craig
- *Modern Robotics* — Lynch & Park
- ROS MoveIt! documentation

---

*The next chapter introduces world model concepts — how agents build internal representations of the world.*
