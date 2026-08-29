<div align="center">

# 🤖 Awesome Robot Hub

### Full-Stack Learning Notes for Physical AI / Embodied AI

[🇨🇳 中文](README.md) · [🇬🇧 English](README.en.md)

> A curated, beginner-friendly learning path for Physical AI / Embodied AI — from fundamentals (perception, control, RL, imitation learning) to frontier topics (world models, VLA models, diffusion policies) and industry applications. Continuously updated.

**中文版**: 一套系统、由浅入深的物理 AI 学习路线：覆盖 **感知 → 控制 → 强化学习 → 模仿学习 → 世界模型 → 多模态大模型 → 产业应用**，持续更新中。

[![GitHub stars](https://img.shields.io/github/stars/gotonote/awesome-robot-hub?style=for-the-badge&logo=github&color=orange)](https://github.com/gotonote/awesome-robot-hub/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/gotonote/awesome-robot-hub?style=for-the-badge&logo=github&color=blue)](https://github.com/gotonote/awesome-robot-hub/network)
[![Chapters](https://img.shields.io/badge/📚-15%20Chapters-22c55e?style=for-the-badge)]()
[![Docs](https://img.shields.io/badge/📄-63%20Docs-3b82f6?style=for-the-badge)]()
[![Daily Updates](https://img.shields.io/badge/🤖-Daily%20Updates-f59e0b?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/📜-MIT%20License-8b5cf6?style=for-the-badge)](./LICENSE)
[![PRs Welcome](https://img.shields.io/badge/🙏-PRs%20Welcome-ff6600?style=for-the-badge)](./CONTRIBUTING.md)

<a href="https://gotonote.github.io/awesome-robot-hub/" style="display:inline-block;background:linear-gradient(90deg,#4ade80 0%,#16a34a 55%,#0ea5e9 100%);color:#ffffff;font-size:1.25em;font-weight:800;padding:14px 44px;border-radius:999px;text-decoration:none;box-shadow:0 6px 20px rgba(22,163,74,.45);">
  <svg width="34" height="20" viewBox="0 0 40 22" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;margin-right:6px;"><g><animateTransform attributeName="transform" type="translate" values="0 0;0 8;0 0" dur="1.2s" repeatCount="indefinite"/><polygon points="20 22, 2 4, 38 4" fill="#ffffff"/></g></svg>🚀 Explore the Online Docs
</a>

<p style="margin-top:10px;color:#64748b;font-size:.9em;">🌐 Full-stack learning path for Physical AI / Embodied AI · 15 chapters · 60+ docs · 100% free</p>

</div>

---

## 🎯 What Is This

**Physical AI** (a.k.a. **Embodied AI**) refers to intelligent systems with physical bodies that can perceive, understand, and interact with the world in real or simulated environments.

This repo organizes fragmented domain knowledge into a complete learning system: **15 chapters, 60+ docs, ~20k lines of content** — covering the full stack from "robotics fundamentals" to "Diffusion Policy / VLA models". **Suitable for beginners to advanced researchers.**

### Physical AI vs. Traditional AI

| Dimension | Traditional AI | Physical AI / Embodied AI |
|-----------|----------------|---------------------------|
| 🗃️ Medium | Pure software / data | Physical entities (robots) |
| 🔄 Interaction | Passively consumes data | Actively interacts with the environment |
| 📚 Learning | Static datasets | Online interaction / trial-and-error |
| 🎯 Goal | Understand the world | **Understand + change** the world |

---

## ✨ Highlights

| | Feature | Description |
|---|---------|-------------|
| 🧩 | **Structured** | 15 chapters from easy to hard; the table of contents *is* the learning path |
| 🔄 | **Continuously updated** | Frontier news and paper reviews updated weekly, tracking NVIDIA GR00T, Unitree, Figure, and more |
| 💻 | **Runnable code** | Hands-on chapters include complete Python implementations (PyBullet / PyTorch / Gymnasium) |
| 🎨 | **Rich visuals** | Diagrams, formulas, and comparison tables for a friendly reading experience |
| 🌍 | **Beginner-friendly** | Explained in plain language; no steep English-only barrier |
| 🤝 | **Open collaboration** | Everyone is welcome to add content, fix errors, and share |

---

## 📖 Table of Contents (15 Chapters)

> The learning roadmap of this repo: 15 chapters from easy to hard — reading in order is recommended.

| Chapter | Topic | What's Inside |
|---------|-------|---------------|
| [01 Getting Started 🚪](01_入门指引/README.md) | Getting Started | What is Physical AI, history, learning path |
| [02 Fundamentals 🧮](02_基础概念/README.md) | Fundamentals | Robotics, kinematics, dynamics, state estimation |
| [03 Perception 👁️](03_感知技术/README.md) | Perception | Computer vision, depth estimation, sensor fusion, event cameras |
| [04 Motion Control 🎮](04_运动控制/README.md) | Control | Path planning, MPC, obstacle avoidance, RL-based control |
| [05 Reinforcement Learning 🧠](05_强化学习/README.md) | Learning | RL basics, PPO, SAC, DQN, Sim-to-Real |
| [06 Imitation Learning 📋](06_模仿学习/README.md) | Learning | Behavior cloning, DAgger, GAIL, IRL |
| [07 World Models 🌍](07_世界模型/README.md) | Frontier | World models overview, frontier progress |
| [08 Multimodal LLMs 🗣️](08_多模态大模型/README.md) | Frontier | VLM, VLA, RT series, PaLM-E, ALOHA/ACT |
| [09 Diffusion Models 🎨](09_扩散模型/README.md) | Frontier | Diffusion Policy, generative actions |
| [10 Classic Papers 📄](10_经典论文/README.md) | Classics | Milestone paper deep-dives (PPO, DQN, RT series…) |
| [11 Industry Applications 🏭](11_产业应用/README.md) | Applications | Autonomous driving, humanoids, healthcare, industry |
| [12 Frontier News 📰](12_前沿动态/README.md) | News | Latest papers, company updates, tech trends |
| [13 Open-Source Projects 🔧](13_开源项目/README.md) | Ecosystem | ROS/ROS2, robot arms, humanoids, sensors |
| [14 Simulation 🖥️](14_仿真环境/README.md) | Practice | MuJoCo, Isaac Sim, PyBullet, SAPIEN |
| [15 Hands-On Tutorials 💻](15_实战教程/README.md) | Practice | Runnable code examples & demos |

---

## 🚀 Quick Start

Zero setup: no dependencies to install — pure Markdown, readable directly on GitHub.

```bash
git clone https://github.com/gotonote/awesome-robot-hub.git
cd awesome-robot-hub
# Open any chapter's README.md to read (or use VS Code / Typora / any Markdown editor)
```

---

## 🤝 Contributing

All contributions are welcome: **add content, fix errors, improve formatting, translate & share**.

- 📋 See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full process
- 🐛 Suggestions / bug reports → [Issues](https://github.com/gotonote/awesome-robot-hub/issues)
- 🔀 Submit changes → [Pull Requests](https://github.com/gotonote/awesome-robot-hub/pulls)

```bash
git clone https://github.com/gotonote/awesome-robot-hub.git
git checkout -b feature/xxx
# ... edit content ...
git commit -m "feat: add xxx content"
git push origin feature/xxx
# Then open a Pull Request
```

---

## 📈 Star History

**Star growth** (auto-updated daily):

<div align="center">

[![Star History](https://raw.githubusercontent.com/gotonote/awesome-robot-hub/main/docs/star-chart.svg)](https://github.com/gotonote/awesome-robot-hub/stargazers)

</div>

> 💡 If the image fails to load, click the link above to visit the [stargazers list](https://github.com/gotonote/awesome-robot-hub/stargazers) or [star-history.com](https://star-history.com/#gotonote/awesome-robot-hub&Date) for the full growth curve.

---

## 📄 License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Released under the **MIT License**. Free to learn, use, and share — please keep attribution.

---

## 📮 Contact

- 💬 Content suggestions / corrections → [GitHub Issues](https://github.com/gotonote/awesome-robot-hub/issues)
- 📨 Join us via Pull Requests to build this open-source learning repo together

---

<div align="center">

## ⭐ If This Repo Helps You

Please star ⭐ the repo in the top-right corner to support our continuous updates!

[![Star this repo](https://img.shields.io/badge/⭐-Star%20this%20Repo-6c8cff?style=for-the-badge)](https://github.com/gotonote/awesome-robot-hub/stargazers)
[![Report Issue](https://img.shields.io/badge/🐛-Report%20an%20Issue-e74c3c?style=for-the-badge)](https://github.com/gotonote/awesome-robot-hub/issues/new)
[![Suggest Content](https://img.shields.io/badge/📚-Suggest%20Content-22c55e?style=for-the-badge)](https://github.com/gotonote/awesome-robot-hub/issues/new)

Star chart auto-updated daily by GitHub Actions · always fresh 🌱

</div>
