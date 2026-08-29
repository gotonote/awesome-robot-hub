# 07 World Models

A **World Model** is an agent's internal representation and predictive model of the external environment — considered a key direction toward general embodied intelligence.

## Contents

- [1. World Models Overview](../07_世界模型/世界模型概论.md) *(content in Chinese)*
  - Definition & core ideas
  - Where world models fit in intelligent systems
  - Classic frameworks & implementations
- [2. Frontier Updates](../07_世界模型/2026-02-22.md) *(content in Chinese)*
  - Latest research progress

---

## Core Concepts

### Why Do We Need World Models?

- **Predict the future**: reason about action consequences in an imagined space, reducing costly real-world trial and error
- **Efficient planning**: make decisions based on an internal model instead of relying purely on environment feedback
- **Imagination-based learning**: train policies in "dreams" to improve sample efficiency

### Classic Approaches

| Representative Work | Core Idea |
|--------------------|-----------|
| Ha & Schmidhuber (2018) | Compress history with an RNN; imagine and train in latent space |
| Dreamer series | Learn a latent dynamics model + learn policies in imagination |
| Video prediction models | Pixel-level world models for embodied manipulation |

---

*This chapter is continuously updated...*
