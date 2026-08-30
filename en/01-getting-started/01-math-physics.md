# Mathematical & Physical Foundations for Physical AI

> **Mathematical and Physical Foundations for Physical AI**
>
> This document provides an in-depth introduction to the core mathematical tools and physical principles in the Physical AI field, laying a solid theoretical foundation for later study of perception, planning, and control algorithms.

---

## 📋 Contents

1. [Linear Algebra Basics](#1-linear-algebra-basics)
2. [Probability & Statistics](#2-probability--statistics)
3. [Optimization Theory](#3-optimization-theory)
4. [Rigid-Body Kinematics](#4-rigid-body-kinematics)
5. [Rigid-Body Dynamics](#5-rigid-body-dynamics)
6. [Control Theory Basics](#6-control-theory-basics)
7. [Core Formula Summary](#7-core-formula-summary)

---

## 1. Linear Algebra Basics

### 1.1 Vectors and Matrices

In robotics, **vectors** and **matrices** are the basic mathematical tools for describing states, transformations, and systems.

#### Vector Definition
$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n
$$

#### Common Operations

| Operation | Formula | Note |
|-----------|---------|------|
| **Inner product** | $\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b} = \sum_{i=1}^n a_i b_i$ | Projection, angle computation |
| **Cross product** | $\mathbf{a} \times \mathbf{b} = [\mathbf{a}]_\times \mathbf{b}$ | Cross product, torque |
| **Norm** | $\|\mathbf{v}\| = \sqrt{\mathbf{v}^T \mathbf{v}}$ | Length, magnitude |

#### Cross-Product Matrix (Skew-Symmetric Matrix)
$$
[\mathbf{a}]_\times = \begin{bmatrix}
0 & -a_z & a_y \\
a_z & 0 & -a_x \\
-a_y & a_x & 0
\end{bmatrix}
$$

> 💡 **Use**: converts the cross product $\mathbf{a} \times \mathbf{b}$ into matrix multiplication $[\mathbf{a}]_\times \mathbf{b}$

### 1.2 Coordinate Transformations

Coordinate transformations are the most central mathematical tool in robotics, describing the position and orientation relationships between reference frames.

#### Homogeneous Transformation Matrix (4×4)
$$
^{A}T_{B} = \begin{bmatrix}
^{A}R_{B} & ^{A}\mathbf{p}_{B/O} \\
\mathbf{0}_{1 \times 3} & 1
\end{bmatrix}
$$

where:
- $^{A}R_{B} \in SO(3)$: rotation matrix from frame B to frame A
- $^{A}\mathbf{p}_{B/O}$: position of the origin of frame B expressed in frame A

#### Properties of Rotation Matrices

1. **Orthogonality**: $R^T R = I$, $R^{-1} = R^T$
2. **Determinant**: $\det(R) = +1$
3. **Group property**: forms the $SO(3)$ group

#### Coordinate Transformation Formulas
$$
^{A}\mathbf{p} = ^{A}T_{B} \cdot ^{B}\mathbf{p}
$$
$$
^{A}\mathbf{v} = ^{A}R_{B} \cdot ^{B}\mathbf{v}
$$

### 1.3 Rotation Vectors

#### Axis-Angle Representation
$$
\mathbf{\omega} = \theta \hat{\mathbf{k}}
$$

where $\theta$ is the rotation angle and $\hat{\mathbf{k}}$ the unit rotation axis.

#### Rodrigues' Formula
$$
R = I + \sin\theta [\hat{\mathbf{k}}]_\times + (1 - \cos\theta) [\hat{\mathbf{k}}]_\times^2
$$

or equivalently:
$$
R = \exp([\boldsymbol{\omega}]_\times) = I + \frac{\sin\theta}{\theta}[\boldsymbol{\omega}]_\times + \frac{1-\cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times^2
$$

> 🎯 **Example**: rotation by angle $\theta$ about the z axis
> $$
> R_z(\theta) = \begin{bmatrix}
> \cos\theta & -\sin\theta & 0 \\
> \sin\theta & \cos\theta & 0 \\
> 0 & 0 & 1
> \end{bmatrix}
> $$

### 1.4 Quaternions

Quaternions are another representation of rotation vectors that avoid the singularity problem of Euler angles.

#### Definition
$$
q = w + xi + yj + zk = \begin{bmatrix} w \\ x \\ y \\ z \end{bmatrix}, \quad \|q\| = 1
$$

or equivalently:
$$
q = \begin{bmatrix} \cos(\theta/2) \\ \hat{\mathbf{k}} \sin(\theta/2) \end{bmatrix}
$$

#### Quaternion Multiplication
$$
q_1 \otimes q_2 = \begin{bmatrix}
w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\
w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\
w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\
w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2
\end{bmatrix}
$$

#### Rotation Matrix ↔ Quaternion Conversion
$$
R = \begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}
$$

---

## 2. Probability & Statistics

### 2.1 Probability Basics

#### Probability Distributions

| Distribution | Formula | Use Case |
|--------------|---------|----------|
| **Gaussian (Normal)** | $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | Measurement noise, state estimation |
| **Bernoulli** | $P(X=1)=p, P(X=0)=1-p$ | Binary decisions |
| **Binomial** | $P(X=k) = \frac{n!}{k!(n-k)!} p^k (1-p)^{n-k}$ | Classification problems |

#### Properties of Gaussian Distributions

- **Linear combination**: if $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$, then
  $$aX + bY \sim \mathcal{N}(a\mu_1 + b\mu_2, a^2\sigma_1^2 + b^2\sigma_2^2)$$

- **Marginalization**: the marginal of a Gaussian is still Gaussian
- **Conditioning**: the conditional of a Gaussian is still Gaussian (this is the theoretical foundation of the Kalman filter)

### 2.2 Bayesian Inference

Bayes' theorem is the theoretical core of state estimation:
$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

In robot state estimation:
$$
\underbrace{P(\mathbf{x}|\mathbf{z})}_{\text{posterior}} = \frac{\underbrace{P(\mathbf{z}|\mathbf{x})}_{\text{likelihood}} \cdot \underbrace{P(\mathbf{x})}_{\text{prior}}}{\underbrace{P(\mathbf{z})}_{\text{normalization constant}}}
$$

where:
- $\mathbf{x}$: robot state (position, orientation, etc.)
- $\mathbf{z}$: sensor observation

### 2.3 Kalman Filter

#### Core Kalman Filter Equations

**Prediction step**:
$$
\hat{\mathbf{x}}_{k|k-1} = F_k \hat{\mathbf{x}}_{k-1|k-1} + B_k \mathbf{u}_k
$$
$$
P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k
$$

**Update step**:
$$
K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1}
$$
$$
\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_k (\mathbf{z}_k - H_k \hat{\mathbf{x}}_{k|k-1})
$$
$$
P_{k|k} = (I - K_k H_k) P_{k|k-1}
$$

where:
- $\hat{\mathbf{x}}$: state estimate
- $P$: covariance matrix
- $F$: state transition matrix
- $B$: input matrix
- $H$: observation matrix
- $Q$: process noise covariance
- $R$: observation noise covariance
- $K$: Kalman gain

#### Kalman Filter Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    卡尔曼滤波流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐      ┌─────────────┐      ┌──────────────┐   │
│   │ k-1时刻  │ ──▶ │   预测步骤   │ ──▶ │   更新步骤    │   │
│   │ 状态估计 │      │ (状态传播)    │      │ (观测融合)    │   │
│   └─────────┘      └─────────────┘      └──────────────┘   │
│        ▲                                        │           │
│        │                                        ▼           │
│        │         ┌─────────────────────────────────────┐    │
│        └─────────┤           k时刻状态估计              │    │
│                  └─────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(State estimate at k-1 → prediction step (state propagation) → update step (observation fusion) → state estimate at k → loop)*

### 2.4 Particle Filter

When the system is nonlinear or non-Gaussian, use the particle filter (Sequential Monte Carlo):

```
┌─────────────────────────────────────────────────────────────┐
│                    粒子滤波算法                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 初始化：生成 N 个粒子 {x_0^i, w_0^i}                     │
│                                                             │
│  2. 预测：对每个粒子应用运动模型                               │
│     x_k^i ~ p(x_k | x_{k-1}^i, u_k)                        │
│                                                             │
│  3. 更新：根据观测计算权重                                    │
│     w_k^i ∝ w_{k-1}^i · p(z_k | x_k^i)                     │
│                                                             │
│  4. 重采样：去除低权重粒子，复制高权重粒子                     │
│                                                             │
│  5. 状态估计：x_k = Σ w_k^i · x_k^i                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(1. Initialize: generate N particles {x_0^i, w_0^i}. 2. Predict: apply the motion model to each particle x_k^i ~ p(x_k | x_{k-1}^i, u_k). 3. Update: compute weights from the observation w_k^i ∝ w_{k-1}^i · p(z_k | x_k^i). 4. Resample: remove low-weight particles, duplicate high-weight ones. 5. State estimate: x_k = Σ w_k^i · x_k^i.)*

---

## 3. Optimization Theory

### 3.1 Optimization Problem Definition

#### Standard Form
$$
\min_{\mathbf{x}} \quad f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) = 0, \quad h_j(\mathbf{x}) \leq 0
$$

#### Lagrange Multipliers

For equality-constrained optimization:
$$
\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x})
$$

KKT (Karush-Kuhn-Tucker) conditions:
$$
\begin{cases}
\nabla_{\mathbf{x}} \mathcal{L} = 0 \\
g_i(\mathbf{x}) = 0 \\
h_j(\mathbf{x}) \leq 0 \\
\lambda_j \geq 0 \\
\lambda_j h_j(\mathbf{x}) = 0
\end{cases}
$$

### 3.2 Gradient Descent

#### Batch Gradient Descent
$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)
$$

#### Stochastic Gradient Descent (SGD)
$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f_i(\mathbf{x}_k)
$$

where $i$ is a randomly selected sample.

#### Learning Rate Schedules

| Method | Formula | Feature |
|--------|---------|---------|
| Fixed | $\alpha_k = \alpha_0$ | Simple |
| Exponential decay | $\alpha_k = \alpha_0 \cdot \gamma^k$ | Smooth decay |
| Cosine annealing | $\alpha_k = \alpha_{min} + \frac{1}{2}(\alpha_{max}-\alpha_{min})(1+\cos\frac{k}{K}\pi)$ | Adaptive |
| Adam | Adaptive learning rate | Most commonly used today |

### 3.3 Quadratic Programming (QP)

A common optimization form in robot control:
$$
\min_{\mathbf{x}} \quad \frac{1}{2}\mathbf{x}^T H \mathbf{x} + \mathbf{c}^T \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \leq \mathbf{b}
$$

#### Application: Model Predictive Control (MPC)

```
┌─────────────────────────────────────────────────────────────┐
│                    模型预测控制（MPC）                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  预测时域 N                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  x(t) → x(t+1|t) → ... → x(t+N|t)                  │   │
│  │  u(t) → u(t+1|t) → ... → u(t+N-1|t)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  优化目标：                                                 │
│  min Σ [x^T Q x + u^T R u] + x_N^T P x_N                  │
│                                                             │
│  约束：                                                     │
│  • 动力学约束：x_{k+1} = f(x_k, u_k)                       │
│  • 控制约束：u_min ≤ u_k ≤ u_max                           │
│  • 状态约束：x_min ≤ x_k ≤ x_max                           │
│                                                             │
│  只执行第一步 u(t)，滚动优化                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Prediction horizon N. Objective: min Σ[x^T Q x + u^T R u] + x_N^T P x_N. Constraints: dynamics x_{k+1} = f(x_k, u_k); control limits u_min ≤ u_k ≤ u_max; state limits x_min ≤ x_k ≤ x_max. Only the first step u(t) is executed; receding-horizon optimization.)*

---

## 4. Rigid-Body Kinematics

### 4.1 Robot Kinematic Models

#### DH Parameters (Denavit-Hartenberg)

Transformation of link frame $i$ relative to link frame $i-1$:
$$
^{i-1}T_i = \text{Rot}_z(\theta_i) \cdot \text{Trans}_z(d_i) \cdot \text{Trans}_x(a_i) \cdot \text{Rot}_x(\alpha_i)
$$

$$
= \begin{bmatrix}
\cos\theta_i & -\sin\theta_i \cos\alpha_i & \sin\theta_i \sin\alpha_i & a_i \cos\theta_i \\
\sin\theta_i & \cos\theta_i \cos\alpha_i & -\cos\theta_i \sin\alpha_i & a_i \sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

#### Forward Kinematics

End-effector pose:
$$
T_{EE} = T_0^1 \cdot T_1^2 \cdot \ldots \cdot T_{n-1}^n
$$

#### Inverse Kinematics

Given the end-effector pose $T_{EE}$, solve for joint angles $\mathbf{q}$.

**Analytical method**: suitable for 6-DOF arms
**Numerical method**: iterative solutions (e.g., Jacobian pseudoinverse)

### 4.2 Jacobian Matrix

#### Definition
$$
\dot{\mathbf{x}} = J(\mathbf{q}) \dot{\mathbf{q}}
$$

where:
- $\mathbf{x}$: end-effector pose (position + orientation)
- $\mathbf{q}$: joint angles
- $J$: Jacobian matrix

#### Velocity Mapping
$$
\mathbf{v}_{EE} = J_v(\mathbf{q}) \dot{\mathbf{q}}
$$
$$
\boldsymbol{\omega}_{EE} = J_\omega(\mathbf{q}) \dot{\mathbf{q}}
$$

#### Singularity Detection
When $\det(J) = 0$, the robot is in a singular configuration, where:
- Mobility is lost in some directions
- Joint velocities tend toward infinity

---

## 5. Rigid-Body Dynamics

### 5.1 Newton-Euler Equations

#### Newton's Equation (Translation)
$$
\mathbf{F} = m\mathbf{a}
$$

#### Euler's Equation (Rotation)
$$
\boldsymbol{\tau} = I\boldsymbol{\dot{\omega}} + \boldsymbol{\omega} \times (I\boldsymbol{\omega})
$$

where:
- $\mathbf{F}$: resultant force
- $m$: mass
- $\mathbf{a}$: acceleration
- $\boldsymbol{\tau}$: torque
- $I$: inertia tensor (3×3 symmetric matrix)
- $\boldsymbol{\omega}$: angular velocity

### 5.2 Lagrangian Mechanics

#### Lagrangian
$$
L(q, \dot{q}) = T(q, \dot{q}) - V(q)
$$

where:
- $T$: kinetic energy of the system
- $V$: potential energy of the system

#### Euler-Lagrange Equation
$$
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = \tau_i
$$

### 5.3 Robot Dynamics Equation

#### Form 1: Lagrangian Form
$$
M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau
$$

where:
- $M(q)$: inertia matrix ($n \times n$)
- $C(q, \dot{q})$: Coriolis and centrifugal matrix
- $g(q)$: gravity vector
- $\tau$: joint torques

#### Form 2: Newton-Euler Recursion

```
┌─────────────────────────────────────────────────────────────┐
│              牛顿-欧拉递推算法                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入：q, dq, ddq（位置、速度、加速度）                       │
│                                                             │
│  步骤1：正向递推（计算各连杆速度、加速度）                     │
│    for i = 1 to n:                                         │
│      ω_i = R_{i-1}^i^T (ω_{i-1} + dq_i * z_{i-1})          │
│      α_i = R_{i-1}^i^T α_{i-1} + ddq_i * z_{i-1}           │
│              + ω_{i-1} × (dq_i * z_{i-1})                  │
│      a_i = R_{i-1}^i^T a_{i-1} + α_i × p_{i-1}^i           │
│              + ω_i × (ω_i × p_{i-1}^i)                    │
│                                                             │
│  步骤2：逆向递推（计算各连杆受力）                            │
│    for i = n to 1:                                         │
│      f_i = m_i * a_i + R_{i}^i+1^T * f_{i+1}               │
│      τ_i = p_{i-1}^i × f_i + I_i * α_i                     │
│              + ω_i × (I_i * ω_i)                          │
│                                                             │
│  输出：τ（关节力矩）                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Input: q, dq, ddq (position, velocity, acceleration). Step 1 forward recursion (compute link velocities and accelerations). Step 2 backward recursion (compute link forces). Output: τ (joint torques).)*

### 5.4 Inertia Tensor

#### Definition
$$
I = \begin{bmatrix}
I_{xx} & -I_{xy} & -I_{xz} \\
-I_{xy} & I_{yy} & -I_{yz} \\
-I_{xz} & -I_{yz} & I_{zz}
\end{bmatrix}
$$

#### Principal Axes of Inertia
Find the principal axes via eigendecomposition:
$$
I = R \begin{bmatrix}
I_1 & 0 & 0 \\
0 & I_2 & 0 \\
0 & 0 & I_3
\end{bmatrix} R^T
$$

---

## 6. Control Theory Basics

### 6.1 PID Control

#### Control Law
$$
u(t) = K_p e(t) + K_i \int_0^t e(\tau)d\tau + K_d \frac{de(t)}{dt}
$$

or in transfer-function form:
$$
C(s) = K_p + \frac{K_i}{s} + K_d s
$$

#### Parameter Tuning (Ziegler-Nichols Method)

| Parameter | P | PI | PID |
|-----------|-----|-----|------|
| $K_p$ | $0.5K_{cr}$ | $0.45K_{cr}$ | $0.6K_{cr}$ |
| $K_i$ | 0 | $0.54K_{cr}/T_u$ | $1.2K_{cr}/T_u$ |
| $K_d$ | 0 | 0 | $0.075K_{cr}T_u$ |

where $K_{cr}$ is the critical gain and $T_u$ the critical oscillation period.

### 6.2 State-Space Control

#### State Equations
$$
\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}
$$
$$
\mathbf{y} = C\mathbf{x} + D\mathbf{u}
$$

#### Pole Placement
Closed-loop pole placement via state feedback $\mathbf{u} = -K\mathbf{x}$:
$$
\dot{\mathbf{x}} = (A - BK)\mathbf{x}
$$

#### Observer Design
$$
\hat{\dot{\mathbf{x}}} = A\hat{\mathbf{x}} + B\mathbf{u} + L(\mathbf{y} - C\hat{\mathbf{x}})
$$

### 6.3 Linear Quadratic Regulator (LQR)

#### Optimal Control Problem
$$
\min_{\mathbf{u}} \int_0^\infty (\mathbf{x}^T Q \mathbf{x} + \mathbf{u}^T R \mathbf{u}) dt
$$

#### Solution: Riccati Equation
$$
P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
$$
$$
K = (R + B^T P B)^{-1} B^T P A
$$

> 💡 **Application**: LQR is a foundation of robot control, e.g., arm joint control, balance control.

### 6.4 Impedance Control

#### Objective
Control the interaction force between the robot and the environment while maintaining some position-tracking capability.

#### Impedance Relationship
$$
M_d \ddot{e} + D_d \dot{e} + K_d e = F_{ext}
$$

where:
- $e = x - x_d$: position error
- $M_d, D_d, K_d$: desired inertia, damping, stiffness
- $F_{ext}$: external/contact force

---

## 7. Core Formula Summary

### 7.1 Kinematics Formulas

| Formula | Use |
|---------|-----|
| $^{A}T_{B} = \begin{bmatrix} ^{A}R_{B} & ^{A}p_{B/O} \\ 0 & 1 \end{bmatrix}$ | Homogeneous transformation |
| $R = \exp([\omega]_\times)$ | Rotation vector → matrix |
| $v = J(q)\dot{q}$ | Velocity mapping |
| $T_{EE} = T_0^1 T_1^2 \cdots T_{n-1}^n$ | Forward kinematics |

### 7.2 Dynamics Formulas

| Formula | Use |
|---------|-----|
| $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau$ | Robot dynamics |
| $F = ma$ | Newton's second law |
| $\tau = I\dot{\omega} + \omega \times (I\omega)$ | Euler's equation |
| $L = T - V$ | Lagrangian |

### 7.3 Control Formulas

| Formula | Use |
|---------|-----|
| $u = K_p e + K_i \int e + K_d \dot{e}$ | PID control |
| $u = -Kx$ | State feedback |
| $M_d\ddot{e} + D_d\dot{e} + K_d e = F_{ext}$ | Impedance control |
| $\min \int (x^T Q x + u^T R u) dt$ | LQR optimal control |

### 7.4 Estimation Formulas

| Formula | Use |
|---------|-----|
| $P(x\|z) = \frac{P(z\|x)P(x)}{P(z)}$ | Bayes' theorem |
| $\hat{x}_{k\|k-1} = F\hat{x}_{k-1} + Bu$ | Prediction |
| $K = P_{k\|k-1}H^T(HP_{k\|k-1}H^T + R)^{-1}$ | Kalman gain |
| $x_{k\|k} = x_{k\|k-1} + K(z - Hx_{k\|k-1})$ | Update |

---

## 📚 References

1. Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control*. Pearson.
2. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
3. Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). *Robot Modeling and Control*. Wiley.
4. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
5. Bertsekas, D. P. (2019). *Reinforcement Learning and Optimal Control*. Athena Scientific.

---

## 🔗 Related Chapters

- [03 Perception](../../03_感知技术/README.md) — perception (vision, depth, sensors)
- [04 Motion Control](../../04_运动控制/README.md) — motion planning & control
- [05 Reinforcement Learning](../../05_强化学习/README.md) — RL basics
- [06 Imitation Learning](../../06_模仿学习/README.md) — imitation learning basics

---

*This document is advanced foundation content of the Physical-AI-Notes series*

*Last updated: February 2026*
*Author: Dabai (大白)*
