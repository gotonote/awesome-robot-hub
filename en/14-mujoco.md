# MuJoCo (Multi-Joint Dynamics with Contact)

MuJoCo is a physics engine for robot simulation, especially suited for continuous control tasks.

## Contents

- [1. MuJoCo Overview](#1-mujoco-overview)
- [2. Model Definition](#2-model-definition)
- [3. Python Interface](#3-python-interface)
- [4. Robot Simulation](#4-robot-simulation)
- [5. Training Example](#5-training-example)

---

## 1. MuJoCo Overview

### 1.1 Features

- Accurate physics simulation
- Contact dynamics
- Open source and free
- Widely used (DeepMind Control Suite)

### 1.2 Installation

```bash
pip install mujoco
pip install mujoco-py  # Python 2 interface (deprecated)
pip install mujoco     # new-version Python interface
```

---

## 2. Model Definition

### 2.1 XML Model Format

```xml
<mujoco model="robot_arm">
  <!-- Compiler settings -->
  <compiler angle="radian" meshdir="meshes"/>
  
  <!-- Global options -->
  <option timestep="0.002" iterations="50" solver="Newton"/>
  
  <!-- World -->
  <worldbody>
    <!-- Ground -->
    <geom type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"/>
    
    <!-- Light -->
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    
    <!-- Robot base -->
    <body name="base_link" pos="0 0 0">
      <joint type="free"/>
      <geom type="mesh" mesh="base"/>
    </body>
  </worldbody>
  
  <!-- Actuators -->
  <actuator>
    <motor joint="joint1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
```

---

## 3. Python Interface

### 3.1 Basic Usage

```python
import mujoco
import numpy as np

# Load the model
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# Simulation loop
for _ in range(1000):
    # Step the simulation
    mujoco.mj_step(model, data)
    
    # Get the state
    qpos = data.qpos  # joint positions
    qvel = data.qvel  # joint velocities
    
    # Set control
    data.ctrl[:] = [0.0] * model.nu
    
print(f"Final joint positions: {data.qpos}")
```

### 3.2 Rendering

```python
import mujoco.viewer

# Create a renderer
viewer = mujoco.viewer.launch_passive(model, data)

# Render loop
while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()

viewer.close()
```

---

## 4. Robot Simulation

### 4.1 Loading Predefined Models

```python
# Use a MuJoCo built-in model
from mujoco import viewer

# Humanoid-style robot
model = mujoco.MjModel.from_xml_string("""
<mujoco model="humanoid">
  <compiler angle="degree" meshdir="."/>
  <option timestep="0.005"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
    <body name="torso" pos="0 0 1.4">
      <freejoint/>
      <geom type="capsule" size="0.07" fromto="0 0 -.3 0 0 .2" rgba="0.7 0.7 0.7 1"/>
      <joint type="free"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="freejoint" gear="200"/>
  </actuator>
</mujoco>
""")

data = mujoco.MjData(model)
viewer = viewer.launch_passive(model, data)
```

---

## 5. Training Example

### 5.1 Simple Policy Gradient

```python
import mujoco
import numpy as np
import torch
import torch.nn as nn

class SimplePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

# Initialize
model = mujoco.MjModel.from_xml_path("swimmer.xml")
data = mujoco.MjData(model)
policy = SimplePolicy(model.nq + model.nv, model.nu)

optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

# Training
for episode in range(100):
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    episode_data = []
    
    for step in range(200):
        # Get the observation
        obs = np.concatenate([data.qpos, data.qvel])
        
        # Policy
        with torch.no_grad():
            action = policy(torch.FloatTensor(obs)).numpy()
            
        # Execute
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        
        # Reward
        reward = -np.sum(action**2) * 0.01  # simple penalty
        
        episode_data.append((obs, action, reward))
        
    # Compute returns
    G = 0
    for obs, action, reward in reversed(episode_data):
        G = reward + 0.99 * G
        
    print(f"Episode {episode}, Return: {G:.2f}")
```

---

## References

1. Todorov, E., et al. (2012). MuJoCo: A physics engine for model-based control.
2. DeepMind Control Suite Documentation

---

*This chapter is continuously updated...*
