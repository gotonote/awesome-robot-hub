# Company Updates

Company developments and product releases in the Physical AI field.

## Contents

- [1. Tech Giants](#1-tech-giants)
- [2. Robot Companies](#2-robot-companies)
- [3. Autonomous Driving](#3-autonomous-driving)
- [4. Startups](#4-startups)
- [5. Latest Product Comparison](#5-latest-product-comparison)

---

## 1. Tech Giants

### 1.1 Google DeepMind

- **RT series**: RT-1, RT-2, RT-3, RT-4
- **Gemini Robotics**: multimodal foundation models
- **SIMA**: general agents

### 1.2 NVIDIA

- **Isaac Sim**: simulation platform
- **GR00T**: humanoid robot foundation model
- **Cosmos**: world model

### 1.3 Isaac Sim Code Example

```python
import omni.usd
from pxr import Usd, UsdGeom, Gf

class IsaacSimSetup:
    """NVIDIA Isaac Sim environment configuration"""
    
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        
    def create_robot(self, robot_usd_path, prim_path="/World/Robot"):
        """Load a robot USD"""
        # Load the robot USD asset
        UsdGeom.ImportPayloadOrReference(
            self.stage, 
            robot_usd_path, 
            prim_path
        )
        
        # Get the robot prim
        self.robot_prim = self.stage.GetPrimAtPath(prim_path)
        return self.robot_prim
    
    def setup_dof_properties(self, robot_prim):
        """Configure joint properties"""
        from omni.isaac.core.robots import Robot
        
        # Set joint control modes
        robot = Robot(robot_prim)
        
        # Configure joint properties
        robot.set_joint_stiffness({
            'joint1': 1000.0,
            'joint2': 800.0,
            'joint3': 600.0,
        })
        
        robot.set_joint_damping({
            'joint1': 50.0,
            'joint2': 40.0,
            'joint3': 30.0,
        })
        
        return robot
    
    def add_camera(self, camera_path, position, rotation):
        """Add a camera sensor"""
        camera_prim = UsdGeom.Camera.Define(self.stage, camera_path)
        
        # Set camera properties
        camera_prim.CreateFocalLengthAttr(50.0)
        camera_prim.CreateFocusDistanceAttr(1000.0)
        camera_prim.CreateApertureAttr(5.6)
        
        # Set position
        xform = UsdGeom.Xformable(camera_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(*rotation))
        
        return camera_prim
    
    def run_simulation(self, num_steps=1000):
        """Run the simulation"""
        from omni.isaac.core import SimulationContext
        
        sim = SimulationContext()
        sim.play()
        
        for _ in range(num_steps):
            sim.step()
            
        return sim.get_physics_dt()
```

### 1.4 OpenAI

- **Robotics research team**: dexterous hands
- **Video understanding**: Sora applications

---

## 2. Robot Companies

### 2.1 Boston Dynamics

- Atlas electric version
- Spot continuous iteration

### 2.2 Figure AI

- Figure 01/02
- Partnership with OpenAI

### 2.3 Unitree

- G1 humanoid robot
- Affordable pricing

### 2.4 Unitree G1 Control Example

```python
import numpy as np
import sdk.units as units

class UnitreeG1Controller:
    """Unitree G1 humanoid robot controller"""
    
    def __init__(self):
        # Joint configuration
        self.joint_names = [
            'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
            'left_knee', 'right_hip_pitch', 'right_hip_roll',
            'right_hip_yaw', 'right_knee',
            'torso', 'left_shoulder_pitch', 'left_shoulder_roll',
            'left_elbow', 'right_shoulder_pitch', 'right_shoulder_roll',
            'right_elbow'
        ]
        
        # Default joint angles (radians)
        self.default_pose = {
            'left_hip_pitch': -0.1,
            'left_hip_roll': 0.0,
            'left_hip_yaw': 0.0,
            'left_knee': 0.2,
            'right_hip_pitch': -0.1,
            'right_hip_roll': 0.0,
            'right_hip_yaw': 0.0,
            'right_knee': 0.2,
            'torso': 0.0,
            'left_shoulder_pitch': 0.3,
            'left_shoulder_roll': 0.0,
            'left_elbow': -0.5,
            'right_shoulder_pitch': 0.3,
            'right_shoulder_roll': 0.0,
            'right_elbow': -0.5,
        }
        
    def forward_kinematics(self, joint_angles):
        """Forward kinematics - compute the foot position"""
        # Simplified kinematics
        # A complete DH-parameter model is needed in practice
        hip_height = 0.65  # hip height
        leg_length = 0.65  # leg length
        
        # Compute the foot position
        hip_pitch = joint_angles['left_hip_pitch']
        knee = joint_angles['left_knee']
        
        # Simplified model
        x = 0
        y = 0
        z = hip_height - leg_length * np.cos(hip_pitch + knee)
        
        return np.array([x, y, z])
    
    def inverse_kinematics(self, target_pos, leg='left'):
        """Inverse kinematics - compute joint angles"""
        hip_height = 0.65
        thigh_len = 0.4
        calf_len = 0.4
        
        x, y, z = target_pos
        
        # Compute hip and knee angles
        d = np.sqrt(x**2 + (z - hip_height)**2)
        
        # Law of cosines
        cos_knee = (thigh_len**2 + calf_len**2 - d**2) / (2 * thigh_len * calf_len)
        knee_angle = np.arccos(np.clip(cos_knee, -1, 1))
        
        # Hip angle
        alpha = np.arctan2(z - hip_height, x)
        beta = np.arccos((thigh_len**2 + d**2 - calf_len**2) / (2 * thigh_len * d))
        
        hip_pitch = alpha + beta
        
        return {
            f'{leg}_hip_pitch': -hip_pitch,
            f'{leg}_knee': np.pi - knee_angle,
        }
    
    def balance_control(self, imu_data, contact_states):
        """Balance control"""
        roll, pitch, yaw = imu_data['rpy']
        
        # PD control
        kp = 50.0
        kd = 5.0
        
        # Target angles
        target_roll = 0.0
        target_pitch = 0.0
        
        # Compute compensation
        compensation = {
            'left_hip_roll': kp * (roll - target_roll),
            'right_hip_roll': -kp * (roll - target_roll),
            'left_hip_pitch': kp * (pitch - target_pitch),
            'right_hip_pitch': kp * (pitch - target_pitch),
        }
        
        return compensation
```

---

## 3. Autonomous Driving

| Company | Progress | Level |
|---------|----------|-------|
| Waymo | Commercial operation | L4 |
| Cruise | Operation paused | L4 |
| Tesla | FSD v12 | L2+ |
| Chinese OEMs | Pilots in multiple cities | L3/L4 |

### 3.1 Autonomous Driving Architecture Example

```python
class AutonomousDrivingSystem:
    """Autonomous driving system architecture"""
    
    def __init__(self):
        # Perception module
        self.perception = PerceptionModule()
        
        # Localization module
        self.localization = LocalizationModule()
        
        # Prediction module
        self.prediction = PredictionModule()
        
        # Planning module
        self.planning = PlanningModule()
        
        # Control module
        self.control = ControlModule()
        
    def perception_step(self, sensor_data):
        """Perception step"""
        # Camera detection
        objects = self.perception.detect_objects(sensor_data['cameras'])
        
        # Radar detection
        radar_objects = self.perception.detect_radar(sensor_data['radar'])
        
        # LiDAR processing
        lidar_objects = self.perception.process_lidar(sensor_data['lidar'])
        
        # Fusion
        fused_objects = self.perception.fuse_detections(
            [objects, radar_objects, lidar_objects]
        )
        
        return fused_objects
    
    def prediction_step(self, objects, hd_map):
        """Prediction step - predict trajectories of other traffic participants"""
        trajectories = {}
        
        for obj in objects:
            # Short-term prediction (3 seconds)
            short_term = self.prediction.predict_short_term(
                obj, 
                horizon=3.0
            )
            
            # Long-term prediction
            long_term = self.prediction.predict_long_term(
                obj,
                hd_map,
                horizon=8.0
            )
            
            trajectories[obj.id] = {
                'short_term': short_term,
                'long_term': long_term,
            }
            
        return trajectories
    
    def planning_step(self, ego_state, objects, trajectories, hd_map):
        """Planning step"""
        # Behavior planning
        behavior = self.planning.behavior_planning(
            ego_state, 
            objects,
            trajectories
        )
        
        # Motion planning
        trajectory = self.planning.motion_planning(
            ego_state,
            behavior,
            hd_map
        )
        
        # Speed planning
        speed_profile = self.planning.speed_planning(trajectory)
        
        return {
            'behavior': behavior,
            'trajectory': trajectory,
            'speed_profile': speed_profile,
        }
    
    def control_step(self, plan, ego_state):
        """Control step"""
        # Lateral control (LQR)
        lateral_cmd = self.control.lateral_control(
            plan['trajectory'],
            ego_state
        )
        
        # Longitudinal control (PID)
        longitudinal_cmd = self.control.longitudinal_control(
            plan['speed_profile'],
            ego_state.velocity
        )
        
        return {
            'steering': lateral_cmd,
            'throttle': longitudinal_cmd['throttle'],
            'brake': longitudinal_cmd['brake'],
        }
```

---

## 4. Startups

### 4.1 Physical Intelligence

- π0 model
- Flow actions

### 4.2 Covariant

- RFUniverse
- Multimodal AI

### 4.3 π0 Inference Example

```python
class PiZeroInference:
    """π0 model inference interface"""
    
    def __init__(self, model_path="pi0_base.pt"):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # Image preprocessor
        self.image_processor = AutoImageProcessor.from_pretrained(
            "google/siglip-so-400m-patch14-224"
        )
        
    @torch.no_grad()
    def predict(self, images, instruction, state=None):
        """
        π0 action prediction
        
        Args:
            images: robot-view images (B, T, C, H, W)
            instruction: language instruction string
            state: joint state (B, 14)
        """
        # Preprocess images
        processed_images = []
        for img in images:
            processed = self.image_processor(img)
            processed_images.append(processed)
        
        image_tensor = torch.stack(processed_images)
        
        # Encode the instruction
        instruction_ids = self.tokenizer(instruction)
        
        # Inference
        action = self.model(
            image_tensor=image_tensor,
            instruction_ids=instruction_ids,
            state=state,
        )
        
        return action
    
    def stream_action(self, image, instruction, state):
        """Streaming inference - real-time control"""
        image_buffer = []
        
        while True:
            # Collect the recent T frames
            image_buffer.append(image)
            if len(image_buffer) > self.window_size:
                image_buffer.pop(0)
                
            # Predict the action
            action = self.predict(
                images=image_buffer,
                instruction=instruction,
                state=state
            )
            
            yield action
```

---

## 5. Latest Product Comparison

### 5.1 Humanoid Robot Comparison

| Model | DOF | Height | Weight | Battery | Price |
|-------|-----|--------|--------|---------|-------|
| Figure 02 | 42 | 170cm | 70kg | 5h | $ |
| Unitree G1 | 43 | 167cm | 35kg | 2h | $$ |
| Atlas | 28 | 150cm | 89kg | 1h | $$$$ |
| Tesla Optimus | 40+ | 172cm | 57kg | TBD | TBD |

### 5.2 Simulation Platform Comparison

| Platform | Rendering | Physics | Real-time | Cost |
|----------|-----------|---------|-----------|------|
| Isaac Sim | RTX | PhysX | ★★★★★ | $ |
| Gazebo | Ogre | ODE/Bullet | ★★★★ | Free |
| PyBullet | OpenGL | Bullet | ★★★ | Free |
| SAPIEN | CUDA | PhysX | ★★★★ | $$ |

---

*This chapter is continuously updated...*
