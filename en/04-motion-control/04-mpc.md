# Model Predictive Control (MPC) for Robot Motion Control

Model Predictive Control (MPC) is an advanced control strategy widely used in robot motion control. This document details the fundamental principles of MPC, its types, and its applications in robotic systems.

## Contents

- [MPC Overview](#mpc-overview)
- [MPC Fundamentals](#mpc-fundamentals)
- [MPC Types](#mpc-types)
- [Applications in Robot Control](#applications-in-robot-control)
- [Code Implementation](#code-implementation)
- [References](#references)

---

## MPC Overview

### What Is Model Predictive Control?

```
┌─────────────────────────────────────────────────────────────┐
│                    MPC核心思想                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   当前时刻 t                                              │
│       │                                                    │
│       ▼                                                    │
│   ┌─────────────────────────────────────────┐              │
│   │  1. 使用系统模型预测未来N步状态            │              │
│   │  2. 优化控制序列使目标函数最小化           │              │
│   │  3. 只执行第一个控制动作                   │              │
│   │  4. 等待下一个时刻，重复上述过程           │              │
│   └─────────────────────────────────────────┘              │
│       │                                                    │
│       ▼                                                    │
│   执行 u(t) → 移动到 t+1 → 重新计算...                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(At the current time t: 1. use the system model to predict the next N steps; 2. optimize the control sequence to minimize the objective; 3. execute only the first control action; 4. wait for the next time step and repeat. Execute u(t) → move to t+1 → recompute...)*

### MPC vs. Traditional PID

| Property | MPC | PID |
|----------|-----|-----|
| **Strategy** | Model-based feedforward + feedback | Pure feedback |
| **Constraint handling** | Naturally handles input/state constraints | Difficult to handle constraints |
| **Multivariable control** | Naturally handles MIMO systems | Requires decoupling |
| **Computational cost** | High (online optimization) | Low |
| **Prediction ability** | Yes | No |
| **Use cases** | Complex constrained systems | Simple systems |

---

## MPC Fundamentals

### 1. System Model

MPC relies on a predictive model of the system, usually in state-space form:

```
┌─────────────────────────────────────────────────────────────┐
│                    离散状态空间模型                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   x(k+1) = A·x(k) + B·u(k) + w(k)    状态方程               │
│   y(k)   = C·x(k) + v(k)              输出方程              │
│                                                             │
│   其中：                                                    │
│   • x(k): 状态向量（位置、速度、加速度等）                   │
│   • u(k): 控制输入（力、力矩、电压等）                       │
│   • y(k): 输出向量（传感器测量）                            │
│   • w(k), v(k): 过程噪声和测量噪声                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Discrete state-space model: x(k+1) = A·x(k) + B·u(k) + w(k) (state equation); y(k) = C·x(k) + v(k) (output equation). Where x(k) is the state vector (position, velocity, acceleration...), u(k) the control input (force, torque, voltage...), y(k) the output vector (sensor measurements), and w(k), v(k) process and measurement noise.)*

### 2. Receding-Horizon Optimization

The core of MPC is receding-horizon optimization:

```python
# MPC optimization problem
# minimize: J = Σ(i=1 to N) [x(i)ᵀQx(i) + u(i)ᵀRu(i)] + x(N)ᵀPNx(N)
# subject to: x(i+1) = Ax(i) + Bu(i)
#             x_min ≤ x(i) ≤ x_max
#             u_min ≤ u(i) ≤ u_max
```

### 3. Feedback Correction

```
┌─────────────────────────────────────────────────────────────┐
│                    MPC反馈校正机制                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   预测                                                    │
│     │                                                     │
│     ▼                                                     │
│   ┌──────────────┐    误差    ┌──────────────┐           │
│   │ 预测状态      │ ───────▶  │ 误差校正      │           │
│   │ x_p(k+i|k)   │            │ x_c(k+i|k)   │           │
│   └──────────────┘            └──────────────┘           │
│        │                           │                       │
│        ▼                           ▼                       │
│   优化控制序列 ──────────▶ 执行第一个控制                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Predicted state → error → error correction → optimized control sequence → execute the first control.)*

---

## MPC Types

### 1. Linear MPC vs. Nonlinear MPC

| Type | Features | Complexity | Use Cases |
|------|----------|------------|-----------|
| **Linear MPC** | Linear model + quadratic objective | Medium | Approximately linear systems |
| **Nonlinear MPC** | Nonlinear model | High | Strongly nonlinear systems |
| **Adaptive MPC** | Online model parameter estimation | High | Systems with uncertain parameters |
| **Robust MPC** | Considers model uncertainty | Very high | Safety-critical systems |

### 2. Continuous vs. Discrete Control Sets

- **Continuous MPC**: control inputs optimized in continuous space (e.g., forces, torques)
- **Discrete MPC**: control inputs selected from a finite set (e.g., switching control)
- **Mixed-integer MPC**: contains both continuous and discrete variables

---

## Applications in Robot Control

### 1. Mobile Robot Trajectory Tracking

```
┌─────────────────────────────────────────────────────────────┐
│              移动机器人MPC控制框架                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   参考轨迹                     MPC控制器                     │
│   ─────────▶─────────▶────────────────────▶ 车轮输入      │
│      │                      │                               │
│      │              ┌──────┴──────┐                        │
│      │              │  预测模型    │                        │
│      │              │ (运动学)     │                        │
│      │              │  代价函数    │                        │
│      │              │  约束条件    │                        │
│      │              └─────────────┘                        │
│      │                      │                               │
│      └──────────┬───────────┘                               │
│                 ▼                                           │
│            状态反馈                                          │
│        (里程计/激光雷达)                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Reference trajectory → MPC controller (prediction model (kinematics), cost function, constraints) → wheel inputs; state feedback (odometry/LiDAR) closes the loop.)*

### 2. Robot Arm End-Effector Trajectory Control

```
┌─────────────────────────────────────────────────────────────┐
│              机械臂末端位置MPC控制                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   目标末端位置 ──▶ ┌──────────────┐ ──▶ 关节力矩            │
│                   │   MPC控制器   │                         │
│   末端位置反馈 ◀── └──────────────┘ ◀── 关节角度/速度       │
│   (视觉/力传感器)                                            │
│                                                             │
│   优化目标：                                                  │
│   • 末端位置跟踪误差最小                                     │
│   • 关节力矩平滑                                             │
│   • 关节限位/速度约束                                        │
│   • 碰撞避免（可选）                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Goal end-effector position → MPC controller → joint torques; end-effector feedback (vision/force sensors) ← joint angles/velocities. Objectives: minimize end-effector tracking error; smooth joint torques; joint limits/velocity constraints; collision avoidance (optional).)*

### 3. Bipedal Robot Gait Planning

```
┌─────────────────────────────────────────────────────────────┐
│              双足机器人MPC步态控制                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────┐           │
│   │           MPC预测 horizon                    │           │
│   │  ───────────────────────────────────────────│           │
│   │  │支撑相│  │摆动相│  │支撑相│  │摆动相│    │           │
│   │  ───────────────────────────────────────────│           │
│   │        ◀── N 步预测 ──▶                     │           │
│   └─────────────────────────────────────────────┘           │
│                         │                                    │
│   状态:                                           │           │
│   • CoM位置/速度                                  │           │
│   • ZMP位置                                       ▼           │
│   • 支撑多边形              ┌──────────────────┐            │
│   • 关节角度/角速度         │   CoM/ZMP跟踪    │            │
│                            └──────────────────┘            │
│                                                             │
│   输出:                                                      │
│   • 关节目标位置 ──▶ 低级PD控制器                           │
│   • 步态时序                                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(State: CoM position/velocity, ZMP position, support polygon, joint angles/velocities → CoM/ZMP tracking → outputs: joint target positions → low-level PD controllers; gait timing.)*

### 4. Drone Flight Control

```
┌─────────────────────────────────────────────────────────────┐
│              四旋翼无人机MPC控制                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   位置控制环（外环MPC）                                      │
│   ┌──────────────────────────────────────────┐             │
│   │ 输入: 目标位置, 当前位置                   │             │
│   │ 输出: 目标姿态角 + 推力                    │             │
│   │ 约束: 状态边界, 输入边界                   │             │
│   └──────────────────────────────────────────┘             │
│                         │                                    │
│                         ▼                                    │
│   姿态控制环（内环）                                         │
│   ┌──────────────────────────────────────────┐             │
│   │ 输入: 目标姿态, 当前姿态                   │             │
│   │ 输出: 电机PWM                             │             │
│   └──────────────────────────────────────────┘             │
│                                                             │
│   MPC优势:                                                   │
│   • 软着陆约束                                               │
│   • 避障约束                                                 │
│   • 飞行走廊约束                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Position control loop (outer MPC): inputs target position and current position, outputs target attitude angles + thrust, with state/input bounds. Attitude control loop (inner): inputs target attitude and current attitude, outputs motor PWM. MPC advantages: soft-landing constraints, obstacle avoidance constraints, flight-corridor constraints.)*

---

## Code Implementation

### 1. Simple 1D MPC

```python
import numpy as np
from scipy.linalg import solve

class SimpleMPC:
    """1D position-tracking MPC controller"""
    
    def __init__(self, horizon=10, dt=0.1):
        self.N = horizon    # prediction horizon
        self.dt = dt        # time step
        
        # System parameters (2nd-order: position-velocity)
        self.A = np.array([[1, dt], [0, 1]])
        self.B = np.array([[0.5*dt**2], [dt]])
        
        # Weight matrices
        self.Q = np.diag([1.0, 0.1])   # state weights
        self.R = np.array([[0.1]])     # control weight
        
    def solve(self, x0, x_ref):
        """
        Solve the MPC optimization problem.
        
        Args:
            x0: initial state [position, velocity]
            x_ref: reference state
            
        Returns:
            u: optimal control input
        """
        # Build the augmented matrix
        N = self.N
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        
        # Build the Hessian matrix
        H = np.zeros((2*N, 2*N))
        for i in range(N):
            H[2*i:2*i+2, 2*i:2*i+2] = Q
        H[2*N-2:2*N, 2*N-2:2*N] = Q  # terminal cost
        
        # Control weights
        for i in range(N):
            H[2*N+i, 2*N+i] = R[0, 0]
        
        # Simplified linear solve (in practice use OSQP and other solvers)
        # Here we use an analytical solution
        x = x0
        u_seq = []
        
        for i in range(N):
            # Predict the state
            if i == 0:
                x_pred = A @ x + B * 0
            else:
                x_pred = A @ x_pred + B * u_seq[-1]
            
            # Compute the control gain
            error = x_pred - x_ref
            u = -0.1 * error[0] - 0.05 * error[1]  # simplified PD control
            u_seq.append(u)
            x = x_pred
            
        return u_seq[0]

# Usage example
if __name__ == "__main__":
    mpc = SimpleMPC(horizon=10, dt=0.1)
    
    # Initial state
    x0 = np.array([0.0, 0.0])  # position=0, velocity=0
    
    # Reference state
    x_ref = np.array([1.0, 0.0])  # target position=1
    
    # Simulate control
    for t in range(50):
        u = mpc.solve(x0, x_ref)
        
        # Apply control
        x0 = mpc.A @ x0 + mpc.B.flatten() * u
        
        print(f"t={t*0.1:.1f}, position={x0[0]:.3f}, velocity={x0[1]:.3f}, u={u:.3f}")
        
        if abs(x0[0] - 1.0) < 0.01 and abs(x0[1]) < 0.01:
            print("Goal reached!")
            break
```

### 2. MPC with OSQP

```python
import numpy as np
import osqp
from scipy import sparse

class QPMPC:
    """MPC controller using the OSQP solver"""
    
    def __init__(self, A, B, Q, R, N, x_min, x_max, u_min, u_max):
        """
        Initialize MPC.
        
        Args:
            A, B: system matrices
            Q, R: weight matrices
            N: prediction horizon
            x_min, x_max: state constraints
            u_min, u_max: control constraints
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        self.nx = A.shape[0]  # state dimension
        self.nu = B.shape[1]   # control dimension
        
        # Constraints
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        
        # Build the OSQP problem
        self._build_problem()
        
    def _build_problem(self):
        """Build the QP problem"""
        N = self.N
        nx = self.nx
        nu = self.nu
        
        # State dimension (includes all prediction steps)
        nx_total = N * nx
        nu_total = (N + 1) * nu
        
        # Build the Hessian of the cost
        # J = x^T Q x + u^T R u
        P = sparse.block_diag([
            sparse.kron(sparse.eye(N), self.Q),
            self.Q,  # terminal cost
            sparse.kron(sparse.eye(N + 1), self.R)
        ]).tocsc()
        
        # Equality constraints (state equations)
        # x(k+1) = Ax(k) + Bu(k)
        A_eq = sparse.lil_matrix(((N + 1) * nx, nx_total + nu_total))
        b_eq = np.zeros((N + 1) * nx)
        
        for i in range(N):
            # x(i+1) constraint
            A_eq[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = -np.eye(nx)
            A_eq[i*nx:(i+1)*nx, N*nx:(N+1)*nx] = self.A
            A_eq[i*nx:(i+1)*nx, nx_total + i*nu:nx_total + (i+1)*nu] = self.B
            
        A_eq = A_eq.tocsc()
        
        # Inequality constraints (box constraints)
        A_ineq = sparse.eye(nx_total + nu_total)
        lb = np.hstack([
            np.full(nx_total, -np.inf),
            np.full(nu_total, self.u_min)
        ])
        ub = np.hstack([
            np.full(nx_total, np.inf),
            np.full(nu_total, self.u_max)
        ])
        
        # Create the OSQP problem
        self.problem = osqp.OSQP()
        self.problem.setup(P, q=np.zeros(nx_total + nu_total),
                          A_eq=A_eq, b_eq=b_eq,
                          A_ineq=A_ineq, l=lb, u=ub,
                          warm_start=True)
        
    def solve(self, x0, x_ref):
        """
        Solve MPC.
        
        Args:
            x0: initial state
            x_ref: reference state sequence (N+1, nx)
            
        Returns:
            u_opt: optimal control sequence
        """
        N = self.N
        
        # Update the reference in the cost
        q = np.zeros(self.nx * (self.N + 1) + self.nu * (self.N + 1))
        for i in range(N + 1):
            q[i*self.nx:(i+1)*self.nx] = -2 * self.Q @ x_ref[i]
        
        self.problem.update(q=q)
        
        # Set the initial state constraint
        self.problem.update(A_eq=self.problem.data.A_eq,
                          b_eq=np.hstack([x0, np.zeros(self.nx * self.N)]))
        
        # Solve
        results = self.problem.solve()
        
        if results.info.status != 'solved':
            print(f"Warning: {results.info.status}")
            return None
            
        # Extract control inputs
        u_opt = results.x[self.nx * (self.N + 1):]
        
        return u_opt[:self.nu]
```

### 3. Bipedal ZMP-MPC Example

```python
import numpy as np

class ZMPMPC:
    """Bipedal robot ZMP-MPC gait controller"""
    
    def __init__(self, dt=0.1, N=20):
        """
        Initialize.
        
        Args:
            dt: time step
            N: prediction horizon steps
        """
        self.dt = dt
        self.N = N
        
        # Robot parameters
        self.g = 9.81        # gravity
        self.z_c = 0.8       # CoM height
        self.com_height = 0.8
        
        # LIPM (Linear Inverted Pendulum Model)
        # x(k+1) = A*x(k) + b*u(k)
        w = np.sqrt(self.g / self.z_c)
        self.A_lipm = np.array([
            [1, dt],
            [w*dt, 1]
        ])
        self.B_lipm = np.array([
            [dt],
            [w*dt]
        ])
        
    def compute_zmp_trajectory(self, com_state, foot_steps, support_type):
        """
        Compute the ZMP reference trajectory.
        
        Args:
            com_state: current CoM state [x, vx, y, vy]
            foot_steps: footstep sequence
            support_type: support type list ['left', 'right', 'double']
            
        Returns:
            zmp_ref: ZMP reference trajectory
        """
        N = self.N
        zmp_ref = np.zeros((N, 2))
        
        for i in range(N):
            if i < len(foot_steps):
                step = foot_steps[i]
                if support_type[i] == 'double':
                    # Double support: ZMP at the midpoint of both feet
                    zmp_ref[i] = (step['left'] + step['right']) / 2
                else:
                    # Single support: ZMP at the support foot
                    foot = step[support_type[i]]
                    zmp_ref[i] = foot[:2]
            else:
                # Repeat the last step
                zmp_ref[i] = zmp_ref[i-1]
                
        return zmp_ref
    
    def solve(self, com0, zmp_ref, support_type):
        """
        Solve MPC to obtain the CoM trajectory.
        
        Args:
            com0: initial CoM state
            zmp_ref: ZMP reference trajectory
            support_type: support type
            
        Returns:
            com_ref: CoM reference trajectory
        """
        N = self.N
        com_ref = np.zeros((N + 1, 4))
        com_ref[0] = com0
        
        # Simplified LIPM tracking
        for i in range(N):
            # Use a linear quadratic regulator
            zmp_error = com_ref[i, :2] - zmp_ref[i]
            
            # Compute the correction force
            Kp = 2.0
            
            # Predict the next step
            com_ref[i+1, 0] = com_ref[i, 1] * self.dt + com_ref[i, 0]
            com_ref[i+1, 2] = com_ref[i, 3] * self.dt + com_ref[i, 2]
            
            # Velocity correction (drive ZMP toward the reference)
            com_ref[i+1, 1] = com_ref[i, 1] - Kp * zmp_error[0]
            com_ref[i+1, 3] = com_ref[i, 3] - Kp * zmp_error[1]
            
        return com_ref

# Usage example
if __name__ == "__main__":
    controller = ZMPMPC(dt=0.1, N=20)
    
    # Initial state
    com0 = np.array([0.0, 0.0, 0.0, 0.0])  # x, vx, y, vy
    
    # Footstep sequence
    foot_steps = [
        {'left': np.array([0.0, 0.1, 0.0]), 'right': np.array([0.0, -0.1, 0.0])},
        {'left': np.array([0.3, 0.1, 0.0]), 'right': np.array([0.3, -0.1, 0.0])},
        {'left': np.array([0.6, 0.1, 0.0]), 'right': np.array([0.6, -0.1, 0.0])},
    ]
    
    support_type = ['double', 'left', 'right', 'left', 'right']
    
    # Compute the ZMP trajectory
    zmp_ref = controller.compute_zmp_trajectory(com0, foot_steps, support_type)
    
    # Solve MPC
    com_ref = controller.solve(com0, zmp_ref, support_type)
    
    print("CoM reference trajectory:")
    print(com_ref)
```

---

## Practical Tips

### 1. Real-Time Optimization

```
┌─────────────────────────────────────────────────────────────┐
│                    MPC实时性优化技巧                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 减少预测时域 N                                          │
│     • N↓ → 计算量↓ → 响应速度↑                              │
│     • 代价：稳定性可能下降                                  │
│                                                             │
│  2. 降低模型精度                                            │
│     • 使用线性模型代替非线性                                 │
│     • 降维状态空间                                          │
│                                                             │
│  3. 使用高效求解器                                          │
│     • OSQP, HPIPM, qpOASES                                 │
│     • 预先计算矩阵分解                                      │
│                                                             │
│  4. 增量求解                                                │
│     • Warm start: 使用上一时刻解初始化                       │
│     • 主动集更新                                            │
│                                                             │
│  5. 近似求解                                               │
│     • 近似QP (无约束控制 + 投影)                            │
│     • 启发式控制分配                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(1. Reduce the prediction horizon N: lower N → less computation → faster response; at the cost of possibly reduced stability. 2. Reduce model accuracy: use linear models instead of nonlinear; reduce state-space dimensionality. 3. Use efficient solvers: OSQP, HPIPM, qpOASES; precompute matrix factorizations. 4. Incremental solving: warm start with the previous solution; active-set updates. 5. Approximate solving: approximate QP (unconstrained control + projection); heuristic control allocation.)*

### 2. Constraint Handling

```python
# Soft vs hard constraints
class SoftConstraintMPC:
    """Soft-constraint MPC - allows violations but adds penalties"""
    
    def __init__(self):
        # Softening factor weight
        self.epsilon = 1000  # large, so constraints are satisfied when possible
        
    def add_soft_constraints(self, H, A_ineq, lb, ub):
        """
        Add soft constraints.
        
        Original: A*x ≤ b
        Softened: A*x ≤ b + ε,  min ε²
        """
        # Add slack variables
        n = A_ineq.shape[1]
        m = A_ineq.shape[0]
        
        # Extend the matrices
        A_soft = np.hstack([A_ineq, np.eye(m)])
        lb_soft = lb
        ub_soft = ub
        
        # Extend the Hessian
        H_soft = np.zeros((n + m, n + m))
        H_soft[:n, :n] = H
        H_soft[n:, n:] = self.epsilon * np.eye(m)
        
        return H_soft, A_soft, lb_soft, ub_soft
```

### 3. Stability Guarantees

| Method | Principle |
|--------|-----------|
| **Terminal constraint** | Force x(N) = 0 or inside a terminal set |
| **Terminal cost** | Use a terminal cost P → x(N)'Px(N) |
| **Contractive constraint** | Gradually tighten the feasible region |
| **Infinite horizon** | N → ∞ (theoretical analysis) |

---

## References

### Papers

1. **Rawlings, J. B., & Mayne, D. Q. (2009)** — "Model Predictive Control: Theory and Design"
2. **Kwon, W. H., & Han, S. (2005)** — "Receding Horizon Control"
3. **Diehl, M., et al. (2005)** — "Fast nonlinear MPC for industrial robots"

### Books

- *Model Predictive Control: Theory, Algorithms, and Applications*
- *Predictive Control for Linear and Hybrid Systems*

### Open-Source Libraries

| Library | Features |
|---------|----------|
| **OSQP** | Efficient QP solver |
| **HPIPM** | Hierarchical predictive control |
| **ACADOS** | Integrated NLP solver |
| **CasADi** | Automatic differentiation + NLP |

---

## Summary

MPC is a powerful and versatile control method, especially suited for:
- Constrained control systems
- Multivariable coupled systems
- Control tasks requiring prediction and planning

In robotics, MPC has been successfully applied to:
- Mobile robot navigation
- Arm manipulation control
- Bipedal gait generation
- Drone flight control

Key challenges:
1. **Computational efficiency**: online optimization is expensive
2. **Model accuracy**: model errors affect control performance
3. **Robustness**: model uncertainty

---

*This section is continuously updated...*
