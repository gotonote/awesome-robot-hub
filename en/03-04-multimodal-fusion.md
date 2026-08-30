# Multimodal Fusion (Vision + Language + Tactile)

Multimodal perception is a core technology of Physical AI — integrating vision, language, tactile, and other modalities so robots can understand the environment more comprehensively and execute tasks.

## Contents

- [1. Multimodal Fusion Overview](#1-multimodal-fusion-overview)
- [2. Vision-Language Fusion](#2-vision-language-fusion)
- [3. Vision-Tactile Fusion](#3-vision-tactile-fusion)
- [4. Tri-Modal Fusion](#4-tri-modal-fusion)
- [5. Fusion Architectures](#5-fusion-architectures)
- [6. Application Scenarios](#6-application-scenarios)

---

## 1. Multimodal Fusion Overview

### 1.1 Why Multimodal Fusion?

| Single Modality | Limitation | Multimodal Advantage |
|-----------------|------------|---------------------|
| Vision | Illumination changes, occlusion | Complementary information |
| Tactile | Limited sensing range | Precise contact information |
| Language | Semantic ambiguity | Task instructions |

### 1.2 Fusion Levels

```
Data-level → Feature-level → Decision-level → Attention-based
    ↓            ↓               ↓               ↓
 Early fusion  Mid fusion    Late fusion    Hybrid fusion
```

---

## 2. Vision-Language Fusion

### 2.1 Visual Language Models (VLM)

```python
import torch
from transformers import AutoModel, AutoProcessor

class VisualLanguageModel:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def process(self, image, text):
        """Process image and text inputs"""
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs
```

### 2.2 CLIP

Contrastive Language-Image Pre-Training:

```python
import torch
import clip

class CLIPModel:
    def __init__(self, device="cuda"):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        
    def encode_image(self, image):
        """Encode an image"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        return image_features / image_features.norm(dim=-1, keepdim=True)
    
    def encode_text(self, text):
        """Encode text"""
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        return text_features / text_features.norm(dim=-1, keepdim=True)
    
    def zero_shot_classify(self, image, class_names):
        """Zero-shot classification"""
        image_features = self.encode_image(image)
        text_features = self.encode_text(class_names)
        
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        return similarity
```

### 2.3 Visual Question Answering (VQA)

```python
def visual_question_answering(image, question, vlm_model):
    """Visual question answering"""
    # Build the input
    prompt = f"Question: {question} Answer:"
    
    # Get the answer
    outputs = vlm_model.process(image, prompt)
    answer = vlm_model.generate(outputs)
    
    return answer
```

---

## 3. Vision-Tactile Fusion

### 3.1 Tactile Sensor Types

| Type | Principle | Output |
|------|-----------|--------|
| GelSight | Visual elastomer | Depth image |
| Takktile | Resistive | Force distribution |
| BioTac | Multimodal | Force + temperature + texture |

### 3.2 Vision-Tactile Alignment

```python
import numpy as np

class VisualTactileFusion:
    def __init__(self):
        self.tactile_to_visual_T = None  # tactile-to-visual transformation
        
    def calibrate(self, tactile_data, visual_data):
        """Calibrate tactile-visual alignment"""
        # Use a calibration block for alignment
        # Compute the transformation matrix
        pass
    
    def fuse(self, rgb_image, depth_image, tactile_image):
        """Fuse visual and tactile information"""
        # Tactile image preprocessing
        tactile_features = self.extract_tactile_features(tactile_image)
        
        # Visual feature extraction
        visual_features = self.extract_visual_features(rgb_image, depth_image)
        
        # Early fusion
        fused = np.concatenate([visual_features, tactile_features], axis=-1)
        
        return fused
    
    def extract_tactile_features(self, tactile_image):
        """Extract tactile features"""
        # Force distribution, contact area, center position, etc.
        features = {
            'force': np.sum(tactile_image),
            'contact_area': np.count_nonzero(tactile_image > 0.01),
            'center_of_mass': self.compute_center(tactile_image)
        }
        return features
```

### 3.3 Tactile-Guided Grasping

```python
class TactileGuidedGrasping:
    def __init__(self):
        self.visual_encoder = VisualEncoder()
        self.tactile_encoder = TactileEncoder()
        self.grasp_planner = GraspPlanner()
        
    def plan_grasp(self, rgb, depth, tactile):
        """Plan a grasp based on vision and tactile"""
        # Visual features
        visual_feat = self.visual_encoder(rgb, depth)
        
        # Tactile features
        tactile_feat = self.tactile_encoder(tactile)
        
        # Fuse
        fused = torch.cat([visual_feat, tactile_feat], dim=-1)
        
        # Plan the grasp
        grasp = self.grasp_planner(fused)
        
        return grasp
```

---

## 4. Tri-Modal Fusion

### 4.1 Architecture Design

```
┌─────────────┐
│   视觉     │ → Visual Encoder
└─────────────┘
       ↓
┌─────────────┐
│   语言     │ → Language Encoder
└─────────────┘
       ↓
┌─────────────┐
│   触觉     │ → Tactile Encoder
└─────────────┘
       ↓
┌─────────────────────────────┐
│     多模态 Transformer      │
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│      融合特征输出           │
└─────────────────────────────┘
```

*(Vision → Visual Encoder, Language → Language Encoder, Tactile → Tactile Encoder → Multimodal Transformer → Fused feature output)*

### 4.2 Implementation

```python
import torch
import torch.nn as nn
from transformers import BertModel

class TriModalFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(TriModalFusion, self).__init__()
        
        # Per-modality encoders
        self.visual_encoder = VisualEncoder(embed_dim)
        self.language_encoder = LanguageEncoder(embed_dim)
        self.tactile_encoder = TactileEncoder(embed_dim)
        
        # Modality alignment layers
        self.visual_projection = nn.Linear(512, embed_dim)
        self.language_projection = nn.Linear(768, embed_dim)
        self.tactile_projection = nn.Linear(64, embed_dim)
        
        # Multimodal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # regression / classification output
        )
        
    def forward(self, visual, language, tactile):
        # Encode each modality
        visual_feat = self.visual_encoder(visual)
        language_feat = self.language_encoder(language)
        tactile_feat = self.tactile_encoder(tactile)
        
        # Project to a unified space
        visual_proj = self.visual_projection(visual_feat)
        language_proj = self.language_projection(language_feat)
        tactile_proj = self.tactile_projection(tactile_feat)
        
        # Stack
        fused = torch.stack([visual_proj, language_proj, tactile_proj], dim=1)
        
        # Transformer fusion
        fused = self.transformer(fused)
        
        # Output
        output = self.output_head(fused.mean(dim=1))
        
        return output
```

---

## 5. Fusion Architectures

### 5.1 Early Fusion

Fusion at the raw data level:

```python
class EarlyFusion(nn.Module):
    def __init__(self):
        super(EarlyFusion, self).__init__()
        # Concatenate raw data from each modality
        self.conv = nn.Conv2d(3 + 1 + 1, 64, 3, padding=1)  # RGB + Depth + Tactile
        
    def forward(self, rgb, depth, tactile):
        # Concatenate raw data
        fused = torch.cat([rgb, depth, tactile], dim=1)
        out = self.conv(fused)
        return out
```

### 5.2 Late Fusion

Each modality is processed independently, then fused:

```python
class LateFusion(nn.Module):
    def __init__(self):
        super(LateFusion, self).__init__()
        self.rgb_branch = Branch(3)
        self.depth_branch = Branch(1)
        self.tactile_branch = Branch(1)
        
        # Fusion layer
        self.fusion = nn.Linear(512 * 3, 512)
        
    def forward(self, rgb, depth, tactile):
        rgb_feat = self.rgb_branch(rgb)
        depth_feat = self.depth_branch(depth)
        tactile_feat = self.tactile_branch(tactile)
        
        fused = torch.cat([rgb_feat, depth_feat, tactile_feat], dim=-1)
        return self.fusion(fused)
```

### 5.3 Attention Fusion

```python
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim=512):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        
    def forward(self, features_list):
        # features_list: [visual_feat, language_feat, tactile_feat]
        # Dynamically weight modalities with attention
        features = torch.stack(features_list, dim=0)  # (3, B, D)
        
        attended, weights = self.attention(
            features, features, features
        )
        
        return attended.mean(dim=0), weights
```

---

## 6. Application Scenarios

### 6.1 Robotic Grasping

```
Task: grasp an unknown object
Input: RGB image + tactile sensor data
Output: grasp point, grasp force
```

### 6.2 Human-Robot Interaction

```
Task: understand human instructions
Input: speech + gestures + image
Output: intent understanding + action planning
```

### 6.3 Environment Perception

```
Task: navigation in complex environments
Input: vision + tactile (contact sensing) + language (instructions)
Output: environment understanding + behavior decisions
```

---

## References

1. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.
2. Li, Y., et al. (2023). Visual-Tactile Fusion for Robotic Manipulation. arXiv.
3. Huang, Y., et al. (2022). CLIP-Adapter: Better CLIP than CLIP. arXiv.

---

*This chapter is continuously updated...*
