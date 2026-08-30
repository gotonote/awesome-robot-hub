# Sim-to-Real Transfer

Sim-to-Real transfer is a key technology in robot learning — it addresses how policies trained in simulation can be effectively transferred to real robots.

## Contents

- [1. Sim-to-Real Overview](#1-sim-to-real-overview)
- [2. Domain Randomization](#2-domain-randomization)
- [3. Domain Adaptation](#3-domain-adaptation)
- [4. Curriculum Learning](#4-curriculum-learning)
- [5. System Identification](#5-system-identification)
- [6. Practical Strategies](#6-practical-strategies)

---

## 1. Sim-to-Real Overview

### 1.1 Why Sim-to-Real

| Aspect | Simulation | Real |
|--------|-----------|------|
| Sample collection | Fast, parallel | Slow, expensive |
| Safety | Risk-free | Risky |
| Repeatability | High | Low |
| Physical fidelity | Imperfect | Real |
| Sensor noise | Simplified | Complex |

### 1.2 The Sim-to-Real Gap

```
Sim-to-Real Gap = |performance(real) - performance(sim)|

Main sources:
1. Visual differences (texture, lighting, noise)
2. Dynamics differences (friction, mass, latency)
3. Sensor differences (noise, resolution)
4. Actuation differences (latency, precision)
```

---

## 2. Domain Randomization

### 2.1 Visual Randomization

```python
import numpy as np
import cv2

class VisualDomainRandomization:
    """
    Visual domain randomization
    Randomize visual parameters in simulation to improve generalization
    """
    def __init__(self):
        pass
    
    def randomize_textures(self, image, texture_type='random'):
        """Texture randomization"""
        if texture_type == 'random':
            # Random color jitter
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            
            # Random brightness change
            hsv[:, :, 2] *= np.random.uniform(0.5, 1.5)
            
            # Random hue change
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-20, 20)) % 180
            
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        elif texture_type == 'noise':
            # Add noise
            noise = np.random.normal(0, 10, image.shape)
            return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    def randomize_camera(self, image):
        """Camera parameter randomization"""
        # Random blur
        if np.random.random() < 0.3:
            kernel_size = np.random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Random resolution change (simulated)
        # Random distortion
        if np.random.random() < 0.3:
            h, w = image.shape[:2]
            K = np.array([[w/2, 0, w/2], [0, h/2, h/2], [0, 0, 1]], dtype=np.float32)
            
            # Random distortion coefficients
            k1 = np.random.uniform(-0.1, 0.1)
            k2 = np.random.uniform(-0.1, 0.1)
            
            # Undistortion
            new_K = K.copy()
            new_K[0, 0] *= np.random.uniform(0.9, 1.1)
            new_K[1, 1] *= np.random.uniform(0.9, 1.1)
            
        return image
    
    def apply_randomization(self, image):
        """Apply all randomizations"""
        image = self.randomize_textures(image, np.random.choice(['random', 'noise']))
        image = self.randomize_camera(image)
        return image
```

### 2.2 Physics Randomization

```python
class PhysicsDomainRandomization:
    """
    Physics domain randomization
    Randomize physics parameters for robustness to unmodeled dynamics
    """
    def __init__(self):
        # Randomizable physics parameters
        self.params = {
            'friction': [0.1, 2.0],           # friction coefficient
            'mass': [0.5, 2.0],                # mass ratio
            'restitution': [0.0, 0.5],         # restitution coefficient
            'gravity': [9.6, 10.0],            # gravity
            'motor_torque': [0.8, 1.2],        # motor torque
            'latency': [0.0, 0.05],           # latency (seconds)
            'joint_damping': [0.0, 1.0],      # joint damping
        }
        
    def sample_params(self):
        """Sample physics parameters"""
        sampled = {}
        for param, (low, high) in self.params.items():
            sampled[param] = np.random.uniform(low, high)
        return sampled
    
    def apply_to_sim(self, sim, params):
        """Apply randomized parameters to the simulation"""
        # Set friction
        sim.set_body_friction('robot_body', params['friction'])
        
        # Set mass
        sim.set_body_mass('robot_body', params['mass'])
        
        # Set gravity
        sim.set_gravity(params['gravity'])
        
        # Set latency
        sim.set_action_latency(params['latency'])
        
        # ... more parameters
```

### 2.3 Automatic Domain Randomization

```python
class AdaptiveDomainRandomization:
    """
    Automatic Domain Randomization (ADR)
    Automatically adjust the randomization level based on real-world performance
    """
    def __init__(self, base_ranges):
        self.base_ranges = base_ranges  # base randomization ranges
        self.current_ranges = base_ranges.copy()
        self.performance_history = []
        
    def update_ranges(self, real_performance):
        """Update the randomization ranges based on real performance"""
        self.performance_history.append(real_performance)
        
        if len(self.performance_history) < 10:
            return
            
        # Check the performance trend
        recent_perf = np.mean(self.performance_history[-5:])
        
        if recent_perf > threshold:
            # Good performance: increase randomization difficulty
            self.increase_difficulty()
        else:
            # Poor performance: decrease difficulty
            self.decrease_difficulty()
            
    def increase_difficulty(self):
        """Increase the randomization difficulty"""
        for param, (low, high) in self.current_ranges.items():
            # Widen the range
            center = (low + high) / 2
            width = (high - low) * 1.2
            self.current_ranges[param] = (
                center - width / 2,
                center + width / 2
            )
```

---

## 3. Domain Adaptation

### 3.1 Pixel-Level Domain Adaptation

```python
import torch
import torch.nn as nn

class PixelLevelDomainAdaptation:
    """
    Pixel-level domain adaptation
    Convert real image style to simulation image style
    """
    def __init__(self):
        # Generator
        self.generator = ResNetGenerator()
        
        # Discriminator
        self.discriminator = PatchGANDiscriminator()
        
    def forward(self, real_image):
        # Generate a simulation-style image
        fake_image = self.generator(real_image)
        return fake_image
    
    def train_step(self, real_images, sim_images):
        # Reconstruction loss
        reconstruction_loss = nn.L1Loss()(real_images, sim_images)
        
        # Adversarial loss
        fake_pred = self.discriminator(real_images)
        real_loss = nn.BCEWithLogitsLoss()(fake_pred, torch.ones_like(fake_pred))
        
        return reconstruction_loss + 0.1 * real_loss
```

### 3.2 Feature-Level Domain Adaptation

```python
class FeatureLevelDomainAdaptation:
    """
    Feature-level domain adaptation
    Eliminate domain differences in feature space
    """
    def __init__(self, feature_dim=256):
        # Feature extractor
        self.encoder = Encoder(feature_dim)
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # source/target domain classification
        )
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer()
        
    def forward(self, source_features, target_features):
        # Source domain features
        source_pred = self.domain_classifier(self.grl(source_features))
        
        # Target domain features
        target_pred = self.domain_classifier(self.grl(target_features))
        
        return source_pred, target_pred
    
    def domain_loss(self, source_pred, target_pred):
        """Domain adaptation loss"""
        source_loss = nn.CrossEntropyLoss()(source_pred, torch.zeros(len(source_pred)))
        target_loss = nn.CrossEntropyLoss()(target_pred, torch.ones(len(target_pred)))
        return source_loss + target_loss


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer"""
    def forward(self, x):
        return x
    
    def backward(self, grad):
        return -grad  # reverse the gradient
```

---

## 4. Curriculum Learning

### 4.1 Progressive Curriculum Learning

```python
class ProgressiveCurriculum:
    """
    Progressive curriculum learning
    Gradually increase task difficulty from simple to complex
    """
    def __init__(self, task_generator):
        self.task_generator = task_generator
        self.current_difficulty = 0.0
        self.difficulty_schedule = 'linear'  # linear, exponential
        
    def get_current_task(self):
        """Get the task at the current difficulty"""
        return self.task_generator.generate(difficulty=self.current_difficulty)
    
    def update_difficulty(self, success_rate):
        """Update the difficulty based on the success rate"""
        if success_rate > 0.8:
            # Too easy: increase the difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + 0.05)
        elif success_rate < 0.3:
            # Too hard: decrease the difficulty
            self.current_difficulty = max(0.0, self.current_difficulty - 0.1)
            
        return self.current_difficulty
    
    def train_with_curriculum(self, agent, num_iterations):
        """Curriculum learning training"""
        for iteration in range(num_iterations):
            task = self.get_current_task()
            
            # Train
            performance = agent.train_on_task(task)
            
            # Update the difficulty
            self.update_difficulty(performance['success_rate'])
            
            # Periodically test on the real environment
            if iteration % 1000 == 0:
                real_performance = agent.evaluate_on_real()
                print(f"Iteration {iteration}, Real perf: {real_performance}")
```

---

## 5. System Identification

### 5.1 Online System Identification

```python
class OnlineSystemIdentification:
    """
    Online system identification
    Estimate physics parameters online in the real environment
    """
    def __init__(self, param_bounds):
        self.param_bounds = param_bounds
        
        # Parameter estimator
        self.param_estimator = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, len(param_bounds))
        )
        
    def estimate_params(self, states, actions, next_states):
        """
        Estimate system parameters from trajectory data
        Use a neural network to predict residuals
        """
        # Build the input
        x = torch.cat([states, actions], dim=-1)
        
        # Predict the change in next state
        delta_s = next_states - states
        
        # Estimate physics parameters
        params = self.param_estimator(x)
        
        return params
    
    def update_sim_with_estimated_params(self, sim, estimated_params):
        """Update the simulation with estimated parameters"""
        for param_name, param_value in zip(self.param_bounds.keys(), estimated_params):
            sim.set_param(param_name, param_value)
```

### 5.2 Bayesian System Identification

```python
class BayesianSystemID:
    """
    Bayesian system identification
    Estimate the parameter distribution with Bayesian methods
    """
    def __init__(self, param_names, prior_means, prior_stds):
        self.param_names = param_names
        
        # Initial priors
        self.params = {
            name: {'mean': mean, 'std': std}
            for name, mean, std in zip(param_names, prior_means, prior_stds)
        }
        
    def update(self, observation, action, next_observation):
        """Update the parameter posterior with an observation"""
        # Simplified Bayesian update
        # In practice, use particle filters or UKF
        
        predicted_next = self.predict(observation, action)
        
        # Compute the prediction error
        error = next_observation - predicted_next
        
        # Update parameter means and variances
        for name in self.param_names:
            learning_rate = 0.01
            self.params[name]['mean'] += learning_rate * error
            self.params[name]['std'] *= 0.99  # variance decay
```

---

## 6. Practical Strategies

### 6.1 Sim-to-Real Checklist

```python
def sim_to_real_checklist():
    """
    Sim-to-Real practical checklist
    """
    checklist = {
        # 1. Simulation fidelity
        'physics_accuracy': [
            "Check the friction model",
            "Check the mass distribution",
            "Check joint limits",
            "Check sensor latency",
            "Check actuator dynamics"
        ],
        
        # 2. Visual fidelity
        'visual_accuracy': [
            "Check texture quality",
            "Check the lighting model",
            "Check camera parameters",
            "Check the noise model"
        ],
        
        # 3. Domain randomization
        'domain_randomization': [
            "Visual randomization",
            "Physics randomization",
            "Sensor noise randomization"
        ],
        
        # 4. Training strategy
        'training_strategy': [
            "Curriculum learning",
            "Progressive transfer",
            "Fine-tuning on the real environment"
        ],
        
        # 5. Evaluation
        'evaluation': [
            "Simulation performance baseline",
            "Real-world performance evaluation",
            "Failure mode analysis"
        ]
    }
    
    return checklist
```

### 6.2 Success Case: Robotic Grasping

```python
class Sim2RealGrasping:
    """
    Sim-to-Real grasping case
    Successful transfer to a real robot using domain randomization
    """
    def __init__(self):
        # 1. Simulation environment configuration
        self.sim_config = {
            'physics': {
                'friction_range': (0.1, 1.5),
                'mass_range': (0.5, 2.0),
            },
            'vision': {
                'texture_variations': 1000,
                'lighting_variations': 50,
                'camera_noise': True
            }
        }
        
        # 2. Training strategy
        self.training_strategy = {
            'total_steps': 1000000,
            'domain_randomization': True,
            'curriculum': True
        }
        
    def train(self):
        """Training pipeline"""
        # Phase 1: simulation training (with domain randomization)
        # Phase 2: real-environment evaluation
        # Phase 3: failure case analysis
        # Phase 4: adjust randomization parameters
        # Phase 5: continue training
        # Phase 6: final deployment
        pass
```

---

## References

1. Tobin, J., et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS.
2. Peng, X. B., et al. (2018). Sim-to-Real Transfer with Domain Adaptation. arXiv.
3. Sadeghi, F., & Levine, S. (2017). CAD2RL: Real Single-Flight Flight Learning. RSS.

---

*This chapter is continuously updated...*
