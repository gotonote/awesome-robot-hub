# 模型预测控制（MPC）在机器人运动控制中的应用

模型预测控制（Model Predictive Control, MPC）是一种先进的控制策略，在机器人运动控制领域有着广泛的应用。本文详细介绍MPC的基本原理、类型及其在机器人系统中的具体应用。

## 目录

- [MPC概述](#mpc概述)
- [MPC基本原理](#mpc基本原理)
- [MPC类型](#mpc类型)
- [在机器人控制中的应用](#在机器人控制中的应用)
- [代码实现](#代码实现)
- [参考资料](#参考资料)

---

## MPC概述

### 什么是模型预测控制？

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

### MPC与传统PID控制的对比

| 特性 | MPC | PID控制 |
|------|-----|---------|
| **控制策略** | 基于模型的前馈+反馈 | 纯反馈 |
| **约束处理** | 自然处理输入/状态约束 | 难以处理约束 |
| **多变量控制** | 自然处理多变量系统 | 需要解耦 |
| **计算复杂度** | 高（需要在线优化） | 低 |
| **预测能力** | 有 | 无 |
| **适用场景** | 复杂约束系统 | 简单系统 |

---

## MPC基本原理

### 1. 系统模型

MPC基于系统的预测模型，通常表示为状态空间形式：

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

### 2. 滚动优化

MPC的核心是滚动优化（Receding Horizon Optimization）：

```python
# MPC优化问题形式
# minimize: J = Σ(i=1 to N) [x(i)ᵀQx(i) + u(i)ᵀRu(i)] + x(N)ᵀPNx(N)
# subject to: x(i+1) = Ax(i) + Bu(i)
#             x_min ≤ x(i) ≤ x_max
#             u_min ≤ u(i) ≤ u_max
```

### 3. 反馈校正

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

---

## MPC类型

### 1. 线性MPC vs 非线性MPC

| 类型 | 特点 | 计算复杂度 | 应用场景 |
|------|------|-----------|---------|
| **线性MPC** | 线性模型 + 二次目标函数 | 中等 | 近似线性系统 |
| **非线性MPC** | 非线性模型 | 高 | 强非线性系统 |
| **自适应MPC** | 在线估计模型参数 | 高 | 参数不确定系统 |
| **鲁棒MPC** | 考虑模型不确定性 | 很高 | 安全性关键系统 |

### 2. 连续控制集 vs 离散控制集

- **连续MPC**: 控制输入在连续空间中优化（如力、力矩）
- **离散MPC**: 控制输入在有限集合中选择（如开关控制）
- **混合整数MPC**: 包含连续和离散变量

---

## 在机器人控制中的应用

### 1. 移动机器人轨迹跟踪

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

### 2. 机械臂末端轨迹控制

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

### 3. 双足机器人步态规划

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

### 4. 无人机飞行控制

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

---

## 代码实现

### 1. 简单的一维MPC实现

```python
import numpy as np
from scipy.linalg import solve

class SimpleMPC:
    """一维位置跟踪MPC控制器"""
    
    def __init__(self, horizon=10, dt=0.1):
        self.N = horizon    # 预测时域
        self.dt = dt        # 时间步长
        
        # 系统参数 (二阶系统: 位置-速度)
        self.A = np.array([[1, dt], [0, 1]])
        self.B = np.array([[0.5*dt**2], [dt]])
        
        # 权重矩阵
        self.Q = np.diag([1.0, 0.1])   # 状态权重
        self.R = np.array([[0.1]])     # 控制权重
        
    def solve(self, x0, x_ref):
        """
        求解MPC优化问题
        
        参数:
            x0: 初始状态 [位置, 速度]
            x_ref: 参考状态
            
        返回:
            u: 最优控制输入
        """
        # 构建增广矩阵
        N = self.N
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        
        # 构建Hessian矩阵
        H = np.zeros((2*N, 2*N))
        for i in range(N):
            H[2*i:2*i+2, 2*i:2*i+2] = Q
        H[2*N-2:2*N, 2*N-2:2*N] = Q  # 终端代价
        
        # 构建控制权重
        for i in range(N):
            H[2*N+i, 2*N+i] = R[0, 0]
        
        # 简化的线性求解（实际使用OSQP等求解器）
        # 这里用解析解
        x = x0
        u_seq = []
        
        for i in range(N):
            # 预测状态
            if i == 0:
                x_pred = A @ x + B * 0
            else:
                x_pred = A @ x_pred + B * u_seq[-1]
            
            # 计算控制增益
            error = x_pred - x_ref
            u = -0.1 * error[0] - 0.05 * error[1]  # 简化的PD控制
            u_seq.append(u)
            x = x_pred
            
        return u_seq[0]

# 使用示例
if __name__ == "__main__":
    mpc = SimpleMPC(horizon=10, dt=0.1)
    
    # 初始状态
    x0 = np.array([0.0, 0.0])  # 位置=0, 速度=0
    
    # 参考状态
    x_ref = np.array([1.0, 0.0])  # 目标位置=1
    
    # 模拟控制
    for t in range(50):
        u = mpc.solve(x0, x_ref)
        
        # 应用控制
        x0 = mpc.A @ x0 + mpc.B.flatten() * u
        
        print(f"t={t*0.1:.1f}, position={x0[0]:.3f}, velocity={x0[1]:.3f}, u={u:.3f}")
        
        if abs(x0[0] - 1.0) < 0.01 and abs(x0[1]) < 0.01:
            print("到达目标!")
            break
```

### 2. 使用OSQP的MPC实现

```python
import numpy as np
import osqp
from scipy import sparse

class QPMPC:
    """使用OSQP求解器的MPC控制器"""
    
    def __init__(self, A, B, Q, R, N, x_min, x_max, u_min, u_max):
        """
        初始化MPC
        
        参数:
            A, B: 系统矩阵
            Q, R: 权重矩阵
            N: 预测时域
            x_min, x_max: 状态约束
            u_min, u_max: 控制约束
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        self.nx = A.shape[0]  # 状态维度
        self.nu = B.shape[1]   # 控制维度
        
        # 约束
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        
        # 构建OSQP问题
        self._build_problem()
        
    def _build_problem(self):
        """构建QP问题"""
        N = self.N
        nx = self.nx
        nu = self.nu
        
        # 状态维度 (包含所有预测步骤)
        nx_total = N * nx
        nu_total = (N + 1) * nu
        
        # 构建代价函数 Hessian
        # J = x^T Q x + u^T R u
        P = sparse.block_diag([
            sparse.kron(sparse.eye(N), self.Q),
            self.Q,  # 终端代价
            sparse.kron(sparse.eye(N + 1), self.R)
        ]).tocsc()
        
        # 构建等式约束 (状态方程)
        # x(k+1) = Ax(k) + Bu(k)
        A_eq = sparse.lil_matrix(((N + 1) * nx, nx_total + nu_total))
        b_eq = np.zeros((N + 1) * nx)
        
        for i in range(N):
            # x(i+1) 约束
            A_eq[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = -np.eye(nx)
            A_eq[i*nx:(i+1)*nx, N*nx:(N+1)*nx] = self.A
            A_eq[i*nx:(i+1)*nx, nx_total + i*nu:nx_total + (i+1)*nu] = self.B
            
        A_eq = A_eq.tocsc()
        
        # 不等式约束 (箱约束)
        A_ineq = sparse.eye(nx_total + nu_total)
        lb = np.hstack([
            np.full(nx_total, -np.inf),
            np.full(nu_total, self.u_min)
        ])
        ub = np.hstack([
            np.full(nx_total, np.inf),
            np.full(nu_total, self.u_max)
        ])
        
        # 创建OSQP问题
        self.problem = osqp.OSQP()
        self.problem.setup(P, q=np.zeros(nx_total + nu_total),
                          A_eq=A_eq, b_eq=b_eq,
                          A_ineq=A_ineq, l=lb, u=ub,
                          warm_start=True)
        
    def solve(self, x0, x_ref):
        """
        求解MPC
        
        参数:
            x0: 初始状态
            x_ref: 参考状态序列 (N+1, nx)
            
        返回:
            u_opt: 最优控制序列
        """
        N = self.N
        
        # 更新代价函数中的参考状态
        q = np.zeros(self.nx * (self.N + 1) + self.nu * (self.N + 1))
        for i in range(N + 1):
            q[i*self.nx:(i+1)*self.nx] = -2 * self.Q @ x_ref[i]
        
        self.problem.update(q=q)
        
        # 设置初始状态约束
        self.problem.update(A_eq=self.problem.data.A_eq,
                          b_eq=np.hstack([x0, np.zeros(self.nx * self.N)]))
        
        # 求解
        results = self.problem.solve()
        
        if results.info.status != 'solved':
            print(f"Warning: {results.info.status}")
            return None
            
        # 提取控制输入
        u_opt = results.x[self.nx * (self.N + 1):]
        
        return u_opt[:self.nu]
```

### 3. 双足机器人ZMP-MPC示例

```python
import numpy as np

class ZMPMPC:
    """双足机器人ZMP-MPC步态控制器"""
    
    def __init__(self, dt=0.1, N=20):
        """
        初始化
        
        参数:
            dt: 时间步长
            N: 预测时域步        self.dt数
        """
 = dt
        self.N = N
        
        # 机器人参数
        self.g = 9.81        # 重力加速度
        self.z_c = 0.8       # 质心高度
        self.com_height = 0.8
        
        # LIPM (线性倒立摆) 模型
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
        计算ZMP参考轨迹
        
        参数:
            com_state: 当前质心状态 [x, vx, y, vy]
            foot_steps: 脚掌序列
            support_type: 支撑类型列表 ['left', 'right', 'double']
            
        返回:
            zmp_ref: ZMP参考轨迹
        """
        N = self.N
        zmp_ref = np.zeros((N, 2))
        
        for i in range(N):
            if i < len(foot_steps):
                step = foot_steps[i]
                if support_type[i] == 'double':
                    # 双脚支撑期，ZMP在两脚中心
                    zmp_ref[i] = (step['left'] + step['right']) / 2
                else:
                    # 单脚支撑期，ZMP在支撑脚位置
                    foot = step[support_type[i]]
                    zmp_ref[i] = foot[:2]
            else:
                # 复制最后一步
                zmp_ref[i] = zmp_ref[i-1]
                
        return zmp_ref
    
    def solve(self, com0, zmp_ref, support_type):
        """
        求解MPC获得CoM轨迹
        
        参数:
            com0: 初始质心状态
            zmp_ref: ZMP参考轨迹
            support_type: 支撑类型
            
        返回:
            com_ref: 质心参考轨迹
        """
        N = self.N
        com_ref = np.zeros((N + 1, 4))
        com_ref[0] = com0
        
        # 简化的LIPM跟踪
        for i in range(N):
            # 使用线性二次调节器
            zmp_error = com_ref[i, :2] - zmp_ref[i]
            
            # 计算修正力
            Kp = 2.0
            
            # 预测下一步
            com_ref[i+1, 0] = com_ref[i, 1] * self.dt + com_ref[i, 0]
            com_ref[i+1, 2] = com_ref[i, 3] * self.dt + com_ref[i, 2]
            
            # 速度修正（使ZMP趋向参考）
            com_ref[i+1, 1] = com_ref[i, 1] - Kp * zmp_error[0]
            com_ref[i+1, 3] = com_ref[i, 3] - Kp * zmp_error[1]
            
        return com_ref

# 使用示例
if __name__ == "__main__":
    controller = ZMPMPC(dt=0.1, N=20)
    
    # 初始状态
    com0 = np.array([0.0, 0.0, 0.0, 0.0])  # x, vx, y, vy
    
    # 脚掌序列
    foot_steps = [
        {'left': np.array([0.0, 0.1, 0.0]), 'right': np.array([0.0, -0.1, 0.0])},
        {'left': np.array([0.3, 0.1, 0.0]), 'right': np.array([0.3, -0.1, 0.0])},
        {'left': np.array([0.6, 0.1, 0.0]), 'right': np.array([0.6, -0.1, 0.0])},
    ]
    
    support_type = ['double', 'left', 'right', 'left', 'right']
    
    # 计算ZMP轨迹
    zmp_ref = controller.compute_zmp_trajectory(com0, foot_steps, support_type)
    
    # 求解MPC
    com_ref = controller.solve(com0, zmp_ref, support_type)
    
    print("CoM参考轨迹:")
    print(com_ref)
```

---

## 实践技巧

### 1. 实时性优化

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

### 2. 约束处理

```python
# 软约束 vs 硬约束
class SoftConstraintMPC:
    """软约束MPC - 允许约束违反但增加惩罚"""
    
    def __init__(self):
        # 软化因子权重
        self.epsilon = 1000  # 很大，使约束尽量满足
        
    def add_soft_constraints(self, H, A_ineq, lb, ub):
        """
        添加软约束
        
        原始: A*x ≤ b
        软化: A*x ≤ b + ε,  min ε²
        """
        # 添加松弛变量
        n = A_ineq.shape[1]
        m = A_ineq.shape[0]
        
        # 扩展矩阵
        A_soft = np.hstack([A_ineq, np.eye(m)])
        lb_soft = lb
        ub_soft = ub
        
        # 扩展Hessian
        H_soft = np.zeros((n + m, n + m))
        H_soft[:n, :n] = H
        H_soft[n:, n:] = self.epsilon * np.eye(m)
        
        return H_soft, A_soft, lb_soft, ub_soft
```

### 3. 稳定性保证

| 方法 | 原理 |
|------|------|
| **终端约束** | 强制 x(N) = 0 或在目标集合内 |
| **终端代价** | 使用终端代价函数 P → x(N)'Px(N) |
| **收缩约束** | 逐步收紧可行域 |
| **无穷时域** | N → ∞ (理论分析用) |

---

## 参考资料

### 论文

1. **Rawlings, J. B., & Mayne, D. Q. (2009)** - "Model Predictive Control: Theory and Design"
2. **Kwon, W. H., & Han, S. (2005)** - "Receding Horizon Control"
3. **Diehl, M., et al. (2005)** - "Fast nonlinear MPC for industrial robots"

### 书籍

- 《Model Predictive Control: Theory, Algorithms, and Applications》
- 《Predictive Control for Linear and Hybrid Systems》

### 开源库

| 库 | 特点 |
|----|------|
| **OSQP** | 高效QP求解器 |
| **HPIPM** | 层次化预测控制 |
| **ACADOS** | 集成NLP求解器 |
| **CasADi** | 自动微分 + NLP |

---

## 总结

MPC是一种强大且通用的控制方法，特别适合：
- 有约束的控制系统
- 多变量耦合系统
- 需要预测和规划的控制任务

在机器人领域，MPC已成功应用于：
- 移动机器人导航
- 机械臂操作控制
- 双足机器人步态生成
- 无人机飞行控制

关键挑战：
1. **计算效率**：在线优化计算量大
2. **模型精度**：模型误差影响控制性能
3. **鲁棒性**：模型不确定性

---

*本节持续更新中...*
