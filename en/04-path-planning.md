# Path Planning

Path planning is the fundamental problem of motion planning — finding a collision-free path for a robot from a start point to a goal point.

## 📋 Contents

- [1. Problem Definition](#1-problem-definition)
- [2. A* Algorithm](#2-a-algorithm)
- [3. RRT Algorithm](#3-rrt-algorithm)
- [4. RRT* Algorithm](#4-rrt-algorithm)
- [5. Potential Field Method](#5-potential-field-method)
- [6. Algorithm Comparison](#6-algorithm-comparison)

---

## 1. Problem Definition

### 1.1 The Path Planning Problem

```
┌─────────────────────────────────────────────────────────────┐
│                    路径规划问题                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  给定：                                                      │
│  • 构型空间 Q                                               │
│  • 自由空间 Q_free = Q \ Q_obs                             │
│  • 起点 q_start ∈ Q_free                                   │
│  • 终点 q_goal ∈ Q_free                                    │
│                                                             │
│  求解：                                                      │
│  • 路径 π: [0,1] → Q_free                                  │
│  • 满足 π(0) = q_start, π(1) = q_goal                      │
│                                                             │
│  优化目标（可选）：                                           │
│  • 最短路径：min ∫||π'(t)||dt                              │
│  • 最平滑路径：min ∫||π''(t)||dt                           │
│  • 最安全路径：max 距离障碍物                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Given: configuration space Q, free space Q_free = Q \ Q_obs, start q_start ∈ Q_free, goal q_goal ∈ Q_free. Find: a path π: [0,1] → Q_free with π(0) = q_start, π(1) = q_goal. Optional objectives: shortest path min ∫||π'(t)||dt; smoothest path min ∫||π''(t)||dt; safest path max distance from obstacles.)*

### 1.2 Problem Taxonomy

| Category | Features | Typical Algorithms |
|----------|----------|-------------------|
| **Discrete space** | Graph/grid representation | A*, Dijkstra, D* |
| **Continuous space** | No discretization | RRT, PRM, potential field |
| **Completeness** | Always finds a solution (if one exists) | A*, RRT (probabilistically complete) |
| **Optimality** | Finds the optimal solution | A*, RRT* |

---

## 2. A* Algorithm

### 2.1 Algorithm Principle

A* is the most classic graph search algorithm, combining Dijkstra's optimality with the efficiency of heuristic search.

```
┌─────────────────────────────────────────────────────────────┐
│                    A* 算法核心                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  评估函数：f(n) = g(n) + h(n)                               │
│                                                             │
│  • g(n)：从起点到节点n的实际代价                             │
│  • h(n)：从节点n到终点的启发式估计                           │
│  • f(n)：经过节点n的总估计代价                               │
│                                                             │
│  启发式函数选择：                                            │
│  • 曼哈顿距离：h = |x1-x2| + |y1-y2|   （4连通）            │
│  • 欧几里得距离：h = √((x1-x2)² + (y1-y2)²)  （8连通）       │
│  • 切比雪夫距离：h = max(|x1-x2|, |y1-y2|)                  │
│                                                             │
│  可接纳性（Admissibility）：                                 │
│  • h(n) ≤ h*(n) （不超过真实代价）                          │
│  • 保证找到最优路径                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Evaluation function f(n) = g(n) + h(n): g(n) is the actual cost from the start to node n; h(n) is the heuristic estimate from node n to the goal; f(n) is the total estimated cost through n. Heuristics: Manhattan distance (4-connected), Euclidean distance (8-connected), Chebyshev distance. Admissibility: h(n) ≤ h*(n) guarantees the optimal path.)*

### 2.2 Algorithm Flow

```
┌─────────────────────────────────────────┐
│           A* 算法流程                    │
├─────────────────────────────────────────┤
│                                         │
│  1. 初始化                              │
│     Open = {start}                      │
│     Closed = {}                         │
│     g[start] = 0                        │
│                                         │
│  2. 循环直到 Open 为空                  │
│     ┌──────────────────────────────┐   │
│     │ a. 取出 f 值最小的节点 n      │   │
│     │    从 Open 移到 Closed        │   │
│     │                              │   │
│     │ b. 若 n == goal，返回路径     │   │
│     │                              │   │
│     │ c. 对每个邻居 m：             │   │
│     │    若 m ∈ Closed，跳过       │   │
│     │    计算 tentative_g = g[n]+d │   │
│     │    若 m ∉ Open 或更优：       │   │
│     │       更新 g[m], f[m]        │   │
│     │       设置 parent[m] = n     │   │
│     │       将 m 加入 Open         │   │
│     └──────────────────────────────┘   │
│                                         │
│  3. 返回失败（无路径）                  │
│                                         │
└─────────────────────────────────────────┘
```

*(1. Initialize: Open = {start}, Closed = {}, g[start] = 0. 2. Loop until Open is empty: (a) pop the node n with the smallest f and move it to Closed; (b) if n == goal, return the path; (c) for each neighbor m: skip if m ∈ Closed; compute tentative_g = g[n] + d; if m ∉ Open or the cost improves: update g[m], f[m], set parent[m] = n, add m to Open. 3. Return failure (no path).)*

### 2.3 Python Implementation

```python
import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple, Optional

class AStar:
    """A* path planning algorithm"""
    
    def __init__(self, grid: np.ndarray):
        """
        Initialize the A* planner.
        
        Args:
            grid: 2D grid map, 0 = free, 1 = obstacle
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        
        # 8 directions: 4-neighborhood + 4 diagonals
        self.directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # 4-connected
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonals
        ]
        
        # Diagonal moves cost √2, straight moves cost 1
        self.move_cost = [1, 1, 1, 1, 
                          np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check whether the position is valid"""
        x, y = pos
        return (0 <= x < self.rows and 
                0 <= y < self.cols and 
                self.grid[x, y] == 0)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        """Get valid neighbors and their move costs"""
        neighbors = []
        x, y = pos
        
        for (dx, dy), cost in zip(self.directions, self.move_cost):
            nx, ny = x + dx, y + dy
            if self.is_valid((nx, ny)):
                # For diagonal moves, check that corners are not cut
                if abs(dx) + abs(dy) == 2:  # diagonal move
                    if self.grid[x + dx, y] == 0 and self.grid[x, y + dy] == 0:
                        neighbors.append(((nx, ny), cost))
                else:
                    neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Run A* planning.
        
        Args:
            start: start position (row, col)
            goal: goal position (row, col)
            
        Returns:
            The path list, or None if unreachable
        """
        if not self.is_valid(start) or not self.is_valid(goal):
            return None
        
        # Priority queue: (f value, counter, position)
        open_set = []
        heappush(open_set, (0, 0, start))
        
        # Bookkeeping
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        closed_set = set()
        counter = 0
        
        while open_set:
            _, _, current = heappop(open_set)
            
            if current in closed_set:
                continue
            
            if current == goal:
                # Reconstruct the path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            closed_set.add(current)
            
            for (neighbor, cost) in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    counter += 1
                    heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        return None  # no path found
    
    def smooth_path(self, path: List[Tuple[int, int]], iterations: int = 100) -> List[Tuple[int, int]]:
        """Path smoothing (gradient descent)"""
        if len(path) <= 2:
            return path
        
        # Convert to continuous coordinates
        path = np.array(path, dtype=float)
        smoothed = path.copy()
        
        # Weights
        weight_data = 0.5  # stay close to the original path
        weight_smooth = 0.3  # smoothness
        
        for _ in range(iterations):
            for i in range(1, len(path) - 1):
                for j in range(2):  # x and y
                    smoothed[i, j] += weight_data * (path[i, j] - smoothed[i, j])
                    smoothed[i, j] += weight_smooth * (smoothed[i-1, j] + smoothed[i+1, j] - 2 * smoothed[i, j])
        
        return smoothed.tolist()


# ============ Usage example ============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a test map
    grid_size = 50
    grid = np.zeros((grid_size, grid_size))
    
    # Add obstacles
    grid[10:30, 20:25] = 1  # vertical wall
    grid[15:20, 25:40] = 1  # horizontal wall
    
    # Create the planner
    planner = AStar(grid)
    
    # Plan a path
    start = (5, 5)
    goal = (45, 45)
    path = planner.plan(start, goal)
    
    if path:
        print(f"Path found, length: {len(path)}")
        
        # Visualize
        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='binary')
        
        path_array = np.array(path)
        plt.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2, label='A* Path')
        plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
        plt.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal')
        
        plt.legend()
        plt.title('A* Path Planning')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.savefig('astar_path.png')
        plt.show()
    else:
        print("No path found!")
```

---

## 3. RRT Algorithm

### 3.1 Algorithm Principle

RRT (Rapidly-exploring Random Tree) is a sampling-based path planning algorithm.

```
┌─────────────────────────────────────────────────────────────┐
│                    RRT 算法核心                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  核心思想：                                                  │
│  通过随机采样，在构型空间中构建一棵快速探索的树               │
│                                                             │
│  优势：                                                      │
│  • 可处理高维空间                                            │
│  • 无需显式构建障碍物边界                                    │
│  • 概率完备（采样足够多必定找到解）                          │
│                                                             │
│  劣势：                                                      │
│  • 路径通常不是最优                                          │
│  • 在狭窄通道中效率较低                                      │
│                                                             │
│  算法步骤：                                                  │
│  1. 从起点初始化树 T = {q_start}                            │
│  2. 在自由空间随机采样 q_rand                               │
│  3. 找到树中最近节点 q_near                                  │
│  4. 从 q_near 向 q_rand 扩展步长 step_size 得到 q_new       │
│  5. 若无碰撞，将 q_new 加入树                                │
│  6. 重复直到 q_new 接近 q_goal                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Core idea: build a rapidly exploring tree in configuration space via random sampling. Advantages: handles high-dimensional spaces; no need to explicitly build obstacle boundaries; probabilistically complete (finds a solution with enough samples). Disadvantages: paths are usually not optimal; inefficient in narrow passages. Steps: 1. initialize the tree T = {q_start} from the start; 2. randomly sample q_rand in free space; 3. find the nearest node q_near; 4. extend from q_near toward q_rand by step_size to obtain q_new; 5. if collision-free, add q_new to the tree; 6. repeat until q_new is close to q_goal.)*

### 3.2 Visualization

```
         q_goal
            ★
           /|\
          / | \
         /  |  \
        /   |   \
       /    |    \
      /     |     \
     /    q_new    \
    /       |       \
   /        |        \
  /      q_near      \
 ●--------●-----------●--------●
q_start   \         / 
           \       /
            \     /
             \   /
              \ /
               ● q_rand
               
RRT通过随机采样逐步扩展树，直到接近目标
```

*(RRT gradually expands the tree by random sampling until it approaches the goal.)*

### 3.3 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class RRT:
    """RRT path planning algorithm"""
    
    def __init__(self, 
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 obstacle_list: List[Tuple[float, float, float]],
                 bounds: Tuple[float, float, float, float],
                 step_size: float = 0.5,
                 goal_sample_rate: float = 0.1,
                 max_iter: int = 5000):
        """
        Initialize the RRT planner.
        
        Args:
            start: start point (x, y)
            goal: goal point (x, y)
            obstacle_list: obstacle list [(x, y, radius), ...]
            bounds: boundaries (x_min, x_max, y_min, y_max)
            step_size: extension step size
            goal_sample_rate: probability of sampling the goal directly
            max_iter: maximum iterations
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacle_list = obstacle_list
        self.bounds = bounds
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        
        # Tree structure: node list and edge list
        self.nodes = [self.start]
        self.parent = {0: -1}  # the root has no parent
        
    def get_random_point(self) -> np.ndarray:
        """Sample a random point within the boundaries"""
        if np.random.random() < self.goal_sample_rate:
            return self.goal.copy()
        
        x = np.random.uniform(self.bounds[0], self.bounds[1])
        y = np.random.uniform(self.bounds[2], self.bounds[3])
        return np.array([x, y])
    
    def get_nearest_node(self, point: np.ndarray) -> int:
        """Find the index of the nearest node to a given point"""
        distances = [np.linalg.norm(node - point) for node in self.nodes]
        return np.argmin(distances)
    
    def steer(self, from_node: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """Extend one step from from_node toward to_point"""
        direction = to_point - from_node
        dist = np.linalg.norm(direction)
        
        if dist <= self.step_size:
            return to_point.copy()
        else:
            return from_node + self.step_size * direction / dist
    
    def is_collision_free(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        """Check whether the path from point1 to point2 is collision-free"""
        # Check that the point is within the boundaries
        if not (self.bounds[0] <= point2[0] <= self.bounds[1] and
                self.bounds[2] <= point2[1] <= self.bounds[3]):
            return False
        
        # Check collision with every obstacle
        for (ox, oy, radius) in self.obstacle_list:
            obstacle_center = np.array([ox, oy])
            
            # Compute the shortest distance from the segment to the center
            line_vec = point2 - point1
            point_vec = obstacle_center - point1
            line_len = np.linalg.norm(line_vec)
            
            if line_len < 1e-6:
                # The two points coincide
                dist = np.linalg.norm(point_vec)
            else:
                # Project onto the segment
                line_unitvec = line_vec / line_len
                proj_length = np.dot(point_vec, line_unitvec)
                proj_length = max(0, min(line_len, proj_length))
                closest_point = point1 + proj_length * line_unitvec
                dist = np.linalg.norm(obstacle_center - closest_point)
            
            if dist <= radius:
                return False
        
        return True
    
    def plan(self) -> Optional[List[np.ndarray]]:
        """Run RRT planning"""
        for i in range(self.max_iter):
            # Random sampling
            q_rand = self.get_random_point()
            
            # Find the nearest node
            nearest_idx = self.get_nearest_node(q_rand)
            q_near = self.nodes[nearest_idx]
            
            # Extend toward the sample
            q_new = self.steer(q_near, q_rand)
            
            # Collision detection
            if self.is_collision_free(q_near, q_new):
                # Add the new node
                self.nodes.append(q_new)
                self.parent[len(self.nodes) - 1] = nearest_idx
                
                # Check whether the goal is reached
                if np.linalg.norm(q_new - self.goal) <= self.step_size:
                    if self.is_collision_free(q_new, self.goal):
                        # Add the goal node
                        self.nodes.append(self.goal)
                        self.parent[len(self.nodes) - 1] = len(self.nodes) - 2
                        
                        # Reconstruct the path
                        return self.extract_path()
        
        return None
    
    def extract_path(self) -> List[np.ndarray]:
        """Extract the path from the tree"""
        path = []
        node_idx = len(self.nodes) - 1
        
        while node_idx != -1:
            path.append(self.nodes[node_idx])
            node_idx = self.parent[node_idx]
        
        return path[::-1]


# ============ Usage example ============
if __name__ == "__main__":
    # Parameters
    start = (5, 5)
    goal = (45, 45)
    bounds = (0, 50, 0, 50)
    
    # Obstacles (circles)
    obstacles = [
        (20, 20, 5),
        (30, 30, 5),
        (25, 35, 4),
        (15, 25, 3),
    ]
    
    # Create the planner and plan
    rrt = RRT(start, goal, obstacles, bounds, step_size=2.0)
    path = rrt.plan()
    
    if path:
        print(f"Path found, nodes: {len(path)}")
        
        # Visualize
        plt.figure(figsize=(10, 10))
        
        # Draw obstacles
        for (ox, oy, radius) in obstacles:
            circle = plt.Circle((ox, oy), radius, color='red', alpha=0.6)
            plt.gca().add_patch(circle)
        
        # Draw the tree
        for i, node in enumerate(rrt.nodes):
            parent_idx = rrt.parent[i]
            if parent_idx >= 0:
                parent = rrt.nodes[parent_idx]
                plt.plot([parent[0], node[0]], [parent[1], node[1]], 'g-', alpha=0.3)
        
        # Draw the path
        path_array = np.array(path)
        plt.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='RRT Path')
        
        # Draw start and goal
        plt.plot(start[0], start[1], 'go', markersize=15, label='Start')
        plt.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
        
        plt.xlim(bounds[0], bounds[1])
        plt.ylim(bounds[2], bounds[3])
        plt.legend()
        plt.title('RRT Path Planning')
        plt.grid(True)
        plt.savefig('rrt_path.png')
        plt.show()
    else:
        print("No path found!")
```

---

## 4. RRT* Algorithm

### 4.1 Algorithm Principle

RRT* improves on RRT with a **rewiring** mechanism, converging asymptotically to the optimal path.

```
┌─────────────────────────────────────────────────────────────┐
│                    RRT* 改进                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  相比RRT的改进：                                             │
│  1. 选择父节点：在半径r内选择代价最小的节点作为父节点         │
│  2. 重新连接：检查新节点能否降低邻居节点的代价               │
│                                                             │
│  伪代码：                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ NearNeighbors = FindNearNodes(q_new, radius)        │   │
│  │                                                      │   │
│  │ // 选择最优父节点                                    │   │
│  │ for q_near in NearNeighbors:                        │   │
│  │     if Cost(q_near) + d(q_near, q_new) < min_cost:  │   │
│  │         q_parent = q_near                           │   │
│  │         min_cost = Cost(q_near) + d(...)           │   │
│  │                                                      │   │
│  │ // 重新连接邻居                                      │   │
│  │ for q_near in NearNeighbors:                        │   │
│  │     if Cost(q_new) + d(q_new, q_near) < Cost(q_near):│   │
│  │         Reparent(q_near, q_new)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  渐进最优性：                                                │
│  随着采样点数量增加，路径代价收敛到最优                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Improvements over RRT: 1. parent selection — choose the lowest-cost node within radius r as the parent; 2. rewiring — check whether the new node can reduce the cost of neighboring nodes. Asymptotic optimality: the path cost converges to the optimum as the number of samples grows.)*

### 4.2 Cost Function

```
Cost(q) = Cost(parent(q)) + distance(parent(q), q)

Parent-selection objective: minimize Cost(q_new)
```

### 4.3 Python Implementation

```python
class RRTStar(RRT):
    """RRT* path planning algorithm - inherits from RRT"""
    
    def __init__(self, 
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 obstacle_list: List[Tuple[float, float, float]],
                 bounds: Tuple[float, float, float, float],
                 step_size: float = 0.5,
                 goal_sample_rate: float = 0.1,
                 max_iter: int = 5000,
                 search_radius: float = 10.0):
        """
        Args:
            search_radius: radius for searching neighbors
        """
        super().__init__(start, goal, obstacle_list, bounds, 
                        step_size, goal_sample_rate, max_iter)
        self.search_radius = search_radius
        self.cost = [0.0]  # cost of each node
        
    def get_near_nodes(self, point: np.ndarray) -> List[int]:
        """Find all nodes within the search radius"""
        distances = [np.linalg.norm(node - point) for node in self.nodes]
        return [i for i, d in enumerate(distances) if d <= self.search_radius]
    
    def compute_cost(self, node_idx: int) -> float:
        """Compute the cost from the start to a node"""
        cost = 0.0
        idx = node_idx
        
        while self.parent[idx] != -1:
            cost += np.linalg.norm(self.nodes[idx] - self.nodes[self.parent[idx]])
            idx = self.parent[idx]
        
        return cost
    
    def choose_parent(self, q_new: np.ndarray, near_nodes: List[int]) -> Tuple[int, float]:
        """Choose the best parent"""
        min_cost = float('inf')
        best_parent = -1
        
        for near_idx in near_nodes:
            q_near = self.nodes[near_idx]
            if self.is_collision_free(q_near, q_new):
                cost = self.compute_cost(near_idx) + np.linalg.norm(q_near - q_new)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = near_idx
        
        return best_parent, min_cost
    
    def rewire(self, q_new_idx: int, near_nodes: List[int]):
        """Rewire: check whether the new node reduces neighbor costs"""
        q_new = self.nodes[q_new_idx]
        cost_to_new = self.compute_cost(q_new_idx)
        
        for near_idx in near_nodes:
            if near_idx == q_new_idx:
                continue
            
            q_near = self.nodes[near_idx]
            potential_cost = cost_to_new + np.linalg.norm(q_new - q_near)
            
            if potential_cost < self.compute_cost(near_idx):
                if self.is_collision_free(q_new, q_near):
                    # Reset the parent
                    self.parent[near_idx] = q_new_idx
    
    def plan(self) -> Optional[List[np.ndarray]]:
        """Run RRT* planning"""
        for i in range(self.max_iter):
            # Random sampling
            q_rand = self.get_random_point()
            
            # Find the nearest node
            nearest_idx = self.get_nearest_node(q_rand)
            q_near = self.nodes[nearest_idx]
            
            # Extend toward the sample
            q_new = self.steer(q_near, q_rand)
            
            # Find nearby nodes
            near_nodes = self.get_near_nodes(q_new)
            
            # Choose the best parent
            if near_nodes:
                best_parent, min_cost = self.choose_parent(q_new, near_nodes)
                if best_parent >= 0 and self.is_collision_free(self.nodes[best_parent], q_new):
                    # Add the new node
                    self.nodes.append(q_new)
                    new_idx = len(self.nodes) - 1
                    self.parent[new_idx] = best_parent
                    self.cost.append(min_cost)
                    
                    # Rewire
                    self.rewire(new_idx, near_nodes)
                    
                    # Check whether the goal is reached
                    if np.linalg.norm(q_new - self.goal) <= self.step_size:
                        if self.is_collision_free(q_new, self.goal):
                            self.nodes.append(self.goal)
                            goal_idx = len(self.nodes) - 1
                            self.parent[goal_idx] = new_idx
                            return self.extract_path()
        
        return None


# Usage example
if __name__ == "__main__":
    obstacles = [(20, 20, 5), (30, 30, 5), (25, 35, 4)]
    
    rrt_star = RRTStar(
        start=(5, 5), 
        goal=(45, 45),
        obstacle_list=obstacles,
        bounds=(0, 50, 0, 50),
        step_size=2.0,
        search_radius=15.0
    )
    
    path = rrt_star.plan()
    if path:
        print(f"RRT* found a path, length: {len(path)}")
```

---

## 5. Potential Field Method

### 5.1 Algorithm Principle

The potential field method models the environment as a potential field — the goal attracts, obstacles repel.

```
┌─────────────────────────────────────────────────────────────┐
│                    势场法原理                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  总势场 = 引力场 + 斥力场                                   │
│  U(q) = U_att(q) + U_rep(q)                                │
│                                                             │
│  引力场（目标点）：                                          │
│  U_att(q) = 1/2 * k_att * ||q - q_goal||²                  │
│  F_att(q) = -∇U_att = k_att * (q_goal - q)                 │
│                                                             │
│  斥力场（障碍物）：                                          │
│  U_rep(q) = { 1/2 * k_rep * (1/ρ - 1/ρ₀)²  if ρ ≤ ρ₀      │
│             { 0                              otherwise      │
│  F_rep(q) = -∇U_rep                                         │
│                                                             │
│  合力：F_total = F_att + ΣF_rep                            │
│  运动方向：q_new = q + step * F_total / ||F_total||        │
│                                                             │
│  问题：                                                      │
│  • 局部极小值：可能陷入非目标的极小势能点                    │
│  • 振荡：在狭窄通道中来回震荡                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Total field = attractive + repulsive. Attractive (goal): U_att(q) = ½·k_att·||q - q_goal||², F_att(q) = -∇U_att = k_att·(q_goal - q). Repulsive (obstacles): U_rep(q) = ½·k_rep·(1/ρ - 1/ρ₀)² if ρ ≤ ρ₀ else 0. Resultant force F_total = F_att + ΣF_rep; motion direction q_new = q + step·F_total/||F_total||. Problems: local minima (may get stuck at non-goal minima); oscillation in narrow passages.)*

### 5.2 Visualization

```
            目标点（引力中心）
                ★
               ↗ ↑ ↖
              /  |  \
             /   |   \
            /    |    \
           ←     |     →
          /      |      \
         /       |       \
        /      障碍物     \
       /      ╔═══╗      \
      /       ║   ║       \
     ←        ║   ║        →
              ╚═══╝
                ↑
               斥力

机器人从起点出发，被目标吸引，被障碍物排斥
```

*(The robot starts from the start point, attracted by the goal and repelled by obstacles.)*

### 5.3 Python Implementation

```python
import numpy as np
from typing import List, Tuple

class PotentialFieldPlanner:
    """Potential field path planning"""
    
    def __init__(self, 
                 goal: Tuple[float, float],
                 obstacles: List[Tuple[float, float, float]],
                 k_att: float = 1.0,
                 k_rep: float = 100.0,
                 rho_0: float = 5.0,
                 step_size: float = 0.5):
        """
        Args:
            goal: goal position
            obstacles: obstacle list [(x, y, radius), ...]
            k_att: attractive gain
            k_rep: repulsive gain
            rho_0: obstacle influence range
            step_size: step size
        """
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        self.step_size = step_size
        
    def attractive_force(self, q: np.ndarray) -> np.ndarray:
        """Compute the attractive force"""
        direction = self.goal - q
        dist = np.linalg.norm(direction)
        
        if dist < 1e-6:
            return np.zeros(2)
        
        return self.k_att * direction
    
    def repulsive_force(self, q: np.ndarray) -> np.ndarray:
        """Compute the sum of repulsive forces from all obstacles"""
        f_rep = np.zeros(2)
        
        for (ox, oy, radius) in self.obstacles:
            obstacle = np.array([ox, oy])
            direction = q - obstacle
            dist = np.linalg.norm(direction)
            
            # Actual distance minus the obstacle radius
            rho = dist - radius
            
            if rho <= self.rho_0 and rho > 1e-6:
                # Repulsive magnitude
                magnitude = self.k_rep * (1/rho - 1/self.rho_0) * (1/rho**2)
                # Repulsive direction
                f_rep += magnitude * direction / dist
        
        return f_rep
    
    def total_force(self, q: np.ndarray) -> np.ndarray:
        """Compute the resultant force"""
        return self.attractive_force(q) + self.repulsive_force(q)
    
    def plan(self, start: Tuple[float, float], 
             max_iter: int = 1000,
             goal_threshold: float = 1.0) -> List[np.ndarray]:
        """Run potential field planning"""
        path = [np.array(start)]
        current = np.array(start)
        
        for _ in range(max_iter):
            # Check whether the goal is reached
            if np.linalg.norm(current - self.goal) < goal_threshold:
                path.append(self.goal.copy())
                break
            
            # Compute the resultant force
            force = self.total_force(current)
            force_mag = np.linalg.norm(force)
            
            if force_mag < 1e-6:
                # Stuck in a local minimum
                print("Warning: trapped in a local minimum!")
                break
            
            # Move along the force direction
            direction = force / force_mag
            current = current + self.step_size * direction
            path.append(current.copy())
        
        return path
    
    def get_potential_field(self, x_range, y_range, resolution=0.5):
        """Generate potential field data for visualization"""
        x = np.arange(x_range[0], x_range[1], resolution)
        y = np.arange(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        U = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                q = np.array([X[i, j], Y[i, j]])
                # Total potential energy
                # Attractive potential
                u_att = 0.5 * self.k_att * np.linalg.norm(q - self.goal)**2
                
                # Repulsive potential
                u_rep = 0
                for (ox, oy, radius) in self.obstacles:
                    dist = np.linalg.norm(q - np.array([ox, oy])) - radius
                    if dist < self.rho_0 and dist > 0:
                        u_rep += 0.5 * self.k_rep * (1/dist - 1/self.rho_0)**2
                
                U[i, j] = u_att + u_rep
        
        return X, Y, U


# Usage example
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Setup
    goal = (45, 45)
    obstacles = [(20, 20, 5), (30, 30, 5), (25, 35, 4)]
    
    planner = PotentialFieldPlanner(goal, obstacles, k_att=1.0, k_rep=200.0)
    path = planner.plan((5, 5))
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: path
    for (ox, oy, radius) in obstacles:
        circle = plt.Circle((ox, oy), radius, color='red', alpha=0.6)
        ax1.add_patch(circle)
    
    path_array = np.array(path)
    ax1.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Path')
    ax1.plot(5, 5, 'go', markersize=15, label='Start')
    ax1.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 50)
    ax1.legend()
    ax1.set_title('Potential Field Path')
    ax1.grid(True)
    
    # Right: potential field
    X, Y, U = planner.get_potential_field((0, 50), (0, 50))
    contour = ax2.contourf(X, Y, U, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label='Potential')
    ax2.set_title('Potential Field Visualization')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('potential_field.png')
    plt.show()
```

---

## 6. Algorithm Comparison

### 6.1 Performance Comparison

| Algorithm | Completeness | Optimality | Dimensionality | Complexity | Pros | Cons |
|-----------|--------------|------------|----------------|------------|------|------|
| **A*** | Complete | Optimal | Low (grid) | O(n log n) | Optimal solution | Requires discretization |
| **RRT** | Probabilistically complete | Not optimal | High | O(n log n) | Simple, efficient | Winding paths |
| **RRT*** | Probabilistically complete | Asymptotically optimal | High | O(n log n) | Converges to optimum | High compute |
| **Potential field** | Incomplete | Not optimal | Any | O(1) | Real-time | Local minima |

### 6.2 Selection Guide

```
┌─────────────────────────────────────────────────────────────┐
│                  路径规划算法选择                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  空间维度 ≤ 3？                                             │
│     ├── 是：需要最优解？                                    │
│     │      ├── 是 → A*                                     │
│     │      └── 否 → RRT                                    │
│     │                                                       │
│     └── 否（高维）：                                        │
│            需要最优？                                        │
│            ├── 是 → RRT*                                   │
│            └── 否 → RRT / PRM                              │
│                                                             │
│  需要实时规划？                                              │
│     ├── 是 → 势场法 / 改进RRT                              │
│     └── 否 → RRT* / 优化后处理                             │
│                                                             │
│  动态环境？                                                  │
│     ├── 是 → D* Lite / 势场法                              │
│     └── 否 → 静态规划算法                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Dimension ≤ 3? — if yes, need optimal? → A* / else RRT; if no (high-dimensional), need optimal? → RRT* / else RRT or PRM. Need real-time planning? → potential field / improved RRT, else RRT* / post-optimization. Dynamic environment? → D* Lite / potential field, else static planning algorithms.)*

### 6.3 Practical Recommendations

| Application | Recommended | Reason |
|-------------|-------------|--------|
| Mobile robot navigation | A* + path smoothing | Low-dimensional, needs optimal |
| Arm motion planning | RRT* + MoveIt | High-dimensional configuration space |
| Drone path planning | RRT / A* | 3D space considerations |
| Autonomous driving | Hybrid A* | Vehicle kinematics considerations |
| Real-time obstacle avoidance | Potential field / MPC | Low-latency requirement |

---

## Further Reading

### Papers
- "Rapidly-Exploring Random Trees" — LaValle (1998)
- "Sampling-based Algorithms for Optimal Motion Planning" — Karaman & Frazzoli (2011)
- Original A* paper on optimal path planning

### Open-Source Libraries
- **OMPL**: Open Motion Planning Library
- **MoveIt**: ROS motion planning framework
- **PYROBOCOP**: Python robot control and optimization

---

*This chapter is continuously updated...*
