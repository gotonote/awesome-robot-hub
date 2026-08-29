# Distributed Reinforcement Learning

Distributed reinforcement learning accelerates training by collecting experience in parallel — a key technology for training large-scale robot policies. This chapter covers the architectures, algorithms, and implementations of distributed RL.

## Contents

- [1. Distributed RL Overview](#1-distributed-rl-overview)
- [2. Distributed Architectures](#2-distributed-architectures)
- [3. Experience Collectors](#3-experience-collectors)
- [4. Distributed Training Algorithms](#4-distributed-training-algorithms)
- [5. Communication Mechanisms](#5-communication-mechanisms)
- [6. Implementation Frameworks](#6-implementation-frameworks)

---

## 1. Distributed RL Overview

### 1.1 Why Distributed RL?

```
Problems of single-agent RL:
- Low sample efficiency
- Long training time
- Limited exploration

Advantages of distributed RL:
- Parallel experience collection
- Faster training
- Better exploration
- Distributed compute resources
```

### 1.2 Distributed RL Taxonomy

| Type | Description | Representative Algorithms |
|------|-------------|---------------------------|
| A3C | Asynchronous Actor-Critic | A3C, PAAC |
| PPO | Parallel PPO | PPO-S, MPPI |
| Experience replay | Distributed experience collection | Ape-X |
| Model-based | Distributed planning | DreamerV3 |

---

## 2. Distributed Architectures

### 2.1 Classic Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    分布式RL架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                  │
│    │Worker│  │Worker│  │Worker│  │Worker│  ← 环境交互     │
│    │  1   │  │  2   │  │  3   │  │  N   │                  │
│    └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                  │
│       │        │        │        │                        │
│       └────────┴────────┴────────┘                        │
│                    │                                       │
│                    ▼                                       │
│    ┌───────────────────────────────┐                      │
│    │      经验缓冲 (Replay Buffer) │                      │
│    └───────────────┬───────────────┘                      │
│                    │                                       │
│                    ▼                                       │
│    ┌───────────────────────────────┐                      │
│    │         训练服务器             │  ← 参数更新        │
│    │      (Learner / Trainer)      │                      │
│    └───────────────┬───────────────┘                      │
│                    │                                       │
│                    ▼                                       │
│            参数同步/广播                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*(Workers (environment interaction) → experience replay buffer → training server (parameter updates) → parameter synchronization/broadcast)*

### 2.2 Communication Patterns

```python
import numpy as np
from abc import ABC, abstractmethod

class CommunicationProtocol(ABC):
    """Base communication protocol class"""
    
    @abstractmethod
    def send(self, data, destination):
        """Send data"""
        pass
    
    @abstractmethod
    def receive(self, source):
        """Receive data"""
        pass


class ParameterServer(CommunicationProtocol):
    """
    Parameter-server architecture
    Centralized parameter management
    """
    def __init__(self, server_address="localhost:8000"):
        self.server_address = server_address
        self.parameters = {}  # model parameters
        self.gradients = {}  # gradient cache
        
    def receive_gradients(self, worker_id, gradients):
        """Receive gradients from a Worker"""
        self.gradients[worker_id] = gradients
        
    def update_parameters(self, optimizer='adam'):
        """Update parameters"""
        # Aggregate gradients
        avg_gradients = self.average_gradients()
        
        # Update parameters
        self.parameters = self.apply_gradients(avg_gradients)
        
        return self.parameters
    
    def broadcast_parameters(self):
        """Broadcast parameters to all Workers"""
        return self.parameters


class DecentralizedCommunication(CommunicationProtocol):
    """
    Decentralized communication
    Workers communicate directly with each other
    """
    def __init__(self, worker_id, peer_ids):
        self.worker_id = worker_id
        self.peer_ids = peer_ids
        self.local_parameters = {}
        
    def send_to_peer(self, peer_id, data):
        """Send data to a peer"""
        pass
    
    def receive_from_peers(self):
        """Receive from peers"""
        pass
```

---

## 3. Experience Collectors

### 3.1 Parallel Environments

```python
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np

class ParallelEnvironmentCollector:
    """
    Parallel environment collector
    Collect experience in parallel with multiple processes
    """
    def __init__(self, env_fn, num_workers=8):
        self.num_workers = num_workers
        self.env_fn = env_fn
        self.queues = [Queue() for _ in range(num_workers)]
        self.processes = []
        
    def start(self):
        """Start the collection processes"""
        for i in range(self.num_workers):
            p = Process(
                target=self.worker_loop,
                args=(i, self.env_fn, self.queues[i])
            )
            p.start()
            self.processes.append(p)
            
    def worker_loop(self, worker_id, env_fn, queue):
        """Worker process loop"""
        env = env_fn()
        state = env.reset()
        
        while True:
            # Receive the policy parameter
            action = queue.get()
            
            # Execute the action
            next_state, reward, done, info = env.step(action)
            
            # Send the experience
            experience = (state, action, reward, next_state, done)
            queue.put(experience)
            
            if done:
                state = env.reset()
            else:
                state = next_state
                
    def collect(self, num_steps):
        """Collect a specified number of steps"""
        experiences = []
        
        for _ in range(num_steps):
            # Broadcast actions
            # ... simplified implementation
            pass
            
        return experiences
    
    def stop(self):
        """Stop all processes"""
        for p in self.processes:
            p.terminate()
```

### 3.2 Distributed Actors

```python
import torch
import torch.multiprocessing as mp

class DistributedActor(mp.Process):
    """
    Distributed Actor
    Each Actor independently interacts with the environment and collects experience
    """
    def __init__(self, worker_id, env_fn, policy, replay_buffer, 
                 batch_size=32, update_freq=100):
        super().__init__()
        self.worker_id = worker_id
        self.env_fn = env_fn
        self.policy = policy  # local policy copy
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.update_freq = update_freq
        
    def run(self):
        """Run the Actor"""
        env = self.env_fn()
        state = env.reset()
        
        step_count = 0
        
        while True:
            # Select an action
            action = self.policy.select_action(state)
            
            # Execute
            next_state, reward, done, _ = env.step(action)
            
            # Store the experience
            self.replay_buffer.push(
                state, action, reward, next_state, done
            )
            
            # Periodically sync parameters
            if step_count % self.update_freq == 0:
                self.sync_parameters()
                
            if done:
                state = env.reset()
            else:
                state = next_state
                
            step_count += 1
            
    def sync_parameters(self):
        """Sync parameters from the central server"""
        # Pull the latest parameters from the parameter server
        # simplified implementation
        pass
```

---

## 4. Distributed Training Algorithms

### 4.1 IMPALA Architecture

```python
import torch
import torch.nn as nn

class IMPALA:
    """
    IMPALA (Importance Weighted Actor-Learner Architecture)
    Asynchronous experience collection, batched learning
    """
    def __init__(self, state_dim, action_dim, num_actors=16):
        self.num_actors = num_actors
        
        # Learner
        self.learner = Learner(state_dim, action_dim)
        
        # Actors
        self.actors = [
            Actor(i, self.learner.model)
            for i in range(num_actors)
        ]
        
        # Trajectory buffer
        self.trajectory_queue = mp.Queue()
        
    def train(self, num_steps):
        """Training loop"""
        # Start Actors
        for actor in self.actors:
            actor.start()
            
        # Learner training
        while True:
            # Collect trajectories from Actors
            trajectories = self.collect_trajectories()
            
            # Compute returns and advantages
            for traj in trajectories:
                returns, advantages = self.compute_returns(traj)
                
            # Update the policy
            self.learner.update(trajectories)
            
    def compute_returns(self, trajectory, gamma=0.99, lambda_=0.95):
        """Compute returns and advantages (V-trace)"""
        returns = []
        advantages = []
        
        # Simplified GAE computation
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(trajectory))):
            value = trajectory[t]['value']
            reward = trajectory[t]['reward']
            
            delta = reward + gamma * next_value - value
            gae = delta + gamma * lambda_ * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + value)
            
            next_value = value
            
        return returns, advantages
```

### 4.2 Ape-X Architecture

```python
class ApeX:
    """
    Ape-X: distributed experience replay
    Multiple Actors collect in parallel, centralized prioritized replay
    """
    def __init__(self, state_dim, action_dim, num_actors=8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_actors = num_actors
        
        # Prioritized experience replay
        self.replay_buffer = PrioritizedReplayBuffer(capacity=1000000)
        
        # Multiple Learners
        self.learners = [Learner(state_dim, action_dim) for _ in range(4)]
        
        # Multiple Actors
        self.actors = [
            Actor(i, self.learners[i % len(self.learners)], self.replay_buffer)
            for i in range(num_actors)
        ]
        
    def train(self):
        """Ape-X training"""
        # Start Actors
        for actor in self.actors:
            actor.start()
            
        # Distributed learning
        for learner in self.learners:
            learner.start()
            
        # Main loop
        while True:
            # Learners keep learning
            pass
```

### 4.3 Seed RL

```python
class SeedRL:
    """
    Seed RL: a unified distributed architecture
    Core idea: the Learner stays central; Actors only collect data
    """
    def __init__(self, state_dim, action_dim, num_actors=16):
        # Shared model
        self.model = ActorCritic(state_dim, action_dim)
        
        # Distributed Actors
        self.actors = [
            SeedActor(i, self.model, self.get_inference_server())
            for i in range(num_actors)
        ]
        
    def get_inference_server(self):
        """Inference server"""
        # Use Ray or gRPC for inference serving
        pass
```

---

## 5. Communication Mechanisms

### 5.1 Parameter Synchronization

```python
class ParameterSynchronizer:
    """
    Parameter synchronizer
    Manages parameter synchronization in distributed training
    """
    def __init__(self, sync_mode='async'):
        self.sync_mode = sync_mode
        self.local_version = 0
        self.server_version = 0
        
    def sync_from_server(self):
        """Sync parameters from the server"""
        if self.sync_mode == 'sync':
            # Synchronous: wait for all Workers
            self.wait_for_workers()
            return self.pull_parameters()
            
        elif self.sync_mode == 'async':
            # Asynchronous: pull directly
            return self.pull_parameters()
            
        elif self.sync_mode == 'lag':
            # Synchronization with lag
            if self.server_version - self.local_version > 10:
                return self.pull_parameters()
                
    def push_to_server(self, gradients):
        """Push gradients to the server"""
        pass
```

### 5.2 Efficient Communication

```python
class EfficientCommunicator:
    """
    Efficient communication optimizations
    - Gradient compression
    - Quantization
    - Sparsification
    """
    def __init__(self, compress=True, quantize=True):
        self.compress = compress
        self.quantize = quantize
        
    def compress_gradients(self, gradients, compression=0.01):
        """Gradient compression"""
        # Top-K sparsification
        flat_grad = gradients.flatten()
        
        # Keep the largest K elements
        k = int(len(flat_grad) * compression)
        threshold = np.sort(np.abs(flat_grad))[-k]
        
        mask = np.abs(flat_grad) >= threshold
        sparse_grad = flat_grad * mask
        
        return sparse_grad, mask
    
    def quantize_parameters(self, params, bits=8):
        """Parameter quantization"""
        # Simple uniform quantization
        min_val = params.min()
        max_val = params.max()
        
        num_bins = 2 ** bits
        
        # Quantize
        quantized = ((params - min_val) / (max_val - min_val) * num_bins).round()
        
        # Dequantize
        dequantized = quantized / num_bins * (max_val - min_val) + min_val
        
        return dequantized
```

---

## 6. Implementation Frameworks

### 6.1 Ray RLlib

```python
import ray
import ray.rllib as rllib
from ray.rllib.agents import ppo

def train_with_rllib():
    """Distributed training with Ray RLlib"""
    # Initialize Ray
    ray.init(num_cpus=64)
    
    # Configure PPO
    config = {
        'env': 'HalfCheetah-v2',
        'num_workers': 32,  # number of parallel Workers
        'num_gpus': 4,      # number of GPUs
        'lr': 0.001,
        'train_batch_size': 32000,
        'rollout_fragment_length': 200,
        
        # Distributed-specific configuration
        'num_envs_per_worker': 5,
        'remote_worker_envs': True,
    }
    
    # Create the Trainer
    trainer = ppo.PPOTrainer(config=config)
    
    # Train
    for i in range(1000):
        result = trainer.train()
        
        if i % 100 == 0:
            print(result)
            
    ray.shutdown()
```

### 6.2 Distributed PPO

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedPPO:
    """
    Distributed PPO implementation
    Using PyTorch Distributed
    """
    def __init__(self, state_dim, action_dim, num_processes):
        self.num_processes = num_processes
        
        # Model
        self.policy = ActorCritic(state_dim, action_dim)
        self.policy = DDP(self.policy)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        
    def reduce_gradients(self):
        """Gradient synchronization"""
        for param in self.policy.parameters():
            dist.all_reduce(param.grad.data)
            param.grad.data /= self.num_processes
            
    def broadcast_parameters(self):
        """Parameter broadcast"""
        for param in self.policy.parameters():
            dist.broadcast(param.data, src=0)
```

---

## References

1. Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. ICML.
2. Espeholt, L., et al. (2018). IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures. ICML.
3. Horgan, D., et al. (2018). ApeX: A Scalable Architecture for Data-Efficient Parallel Reinforcement Learning. arXiv.

---

*This chapter is continuously updated...*
