# 09 Diffusion Models

Diffusion models have moved from image generation to robot policy learning in recent years. **Diffusion Policy** has become a representative generative approach for robotic action policies.

## Contents

- [1. Diffusion Policy In-Depth](../09_扩散模型/Diffusion_Policy_详解.md) *(content in Chinese)*
  - Background & motivation
  - Diffusion model fundamentals
  - Action generation pipeline
  - Robotic applications & experiments

---

## Core Concepts

### Why Use Diffusion Models for Policies?

- **Multimodal behavior**: naturally models the diverse valid solutions in human demonstrations
- **High-dimensional action spaces**: supports high-dimensional continuous control
- **Temporal consistency**: models action sequences, reducing jitter

### Key Techniques

| Concept | Description |
|---------|-------------|
| Forward diffusion | Gradually add noise until pure noise |
| Reverse denoising | Recover action samples from noise |
| Conditional generation | Generate actions conditioned on observations |
| Training & inference | Noise prediction objective + iterative sampling |

---

*This chapter is continuously updated...*
