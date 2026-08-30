<div align="center">

[🇨🇳 中文](README.md) · [🇬🇧 English](README.en.md)

# 🤖 Awesome Robot Hub

### Full-Stack Learning Notes for Physical AI / Embodied AI

> A curated, beginner-friendly learning path for Physical AI / Embodied AI — from fundamentals (perception, control, RL, imitation learning) to frontier topics (world models, VLA models, diffusion policies) and industry applications. Continuously updated.

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

> The learning roadmap of this repo: 15 chapters from easy to hard — reading in order is recommended. 📖 Each chapter has an [English introduction page](en/README.md); the in-depth documents inside chapters are being translated progressively.

| Chapter | Topic | What's Inside |
|---------|-------|---------------|
| [01 Getting Started 🚪](en/01-getting-started/README.md) | Getting Started | What is Physical AI, history, learning path |
| [02 Fundamentals 🧮](en/02-fundamentals/README.md) | Fundamentals | Robotics, kinematics, dynamics, state estimation |
| [03 Perception 👁️](en/03-perception/README.md) | Perception | Computer vision, depth estimation, sensor fusion, event cameras |
| [04 Motion Control 🎮](en/04-motion-control/README.md) | Control | Path planning, MPC, obstacle avoidance, RL-based control |
| [05 Reinforcement Learning 🧠](en/05-reinforcement-learning/README.md) | Learning | RL basics, PPO, SAC, DQN, Sim-to-Real |
| [06 Imitation Learning 📋](en/06-imitation-learning/README.md) | Learning | Behavior cloning, DAgger, GAIL, IRL |
| [07 World Models 🌍](en/07-world-models/README.md) | Frontier | World models overview, frontier progress |
| [08 Multimodal LLMs 🗣️](en/08-multimodal-llms/README.md) | Frontier | VLM, VLA, RT series, PaLM-E, ALOHA/ACT |
| [09 Diffusion Models 🎨](en/09-diffusion-models/README.md) | Frontier | Diffusion Policy, generative actions |
| [10 Classic Papers 📄](en/10-classic-papers/README.md) | Classics | Milestone paper deep-dives (PPO, DQN, RT series…) |
| [11 Industry Applications 🏭](en/11-industry-applications/README.md) | Applications | Autonomous driving, humanoids, healthcare, industry |
| [12 Frontier News 📰](en/12-frontier-news/README.md) | News | Latest papers, company updates, tech trends |
| [13 Open-Source Projects 🔧](en/13-open-source-projects/README.md) | Ecosystem | ROS/ROS2, robot arms, humanoids, sensors |
| [14 Simulation 🖥️](en/14-simulation/README.md) | Practice | MuJoCo, Isaac Sim, PyBullet, SAPIEN |
| [15 Hands-On Tutorials 💻](en/15-hands-on-tutorials/README.md) | Practice | Runnable code examples & demos |

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
