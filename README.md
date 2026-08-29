<div align="center">

# 🤖 Awesome Robot Hub

### 物理 AI / 具身智能（Embodied AI）全栈学习笔记

> 一套系统、由浅入深的物理AI学习路线：覆盖 **感知 → 控制 → 强化学习 → 模仿学习 → 世界模型 → 多模态大模型 → 产业应用**，持续更新中。

**English**: A curated, beginner-friendly learning path for Physical AI / Embodied AI — from fundamentals (perception, control, RL, imitation learning) to frontier topics (world models, VLA models, diffusion policies) and industry applications. Continuously updated.

[![Stars](https://img.shields.io/github/stars/gotonote/awesome-robot-hub?style=flat-square&logo=github&color=orange)](https://github.com/gotonote/awesome-robot-hub/stargazers)
[![Forks](https://img.shields.io/github/forks/gotonote/awesome-robot-hub?style=flat-square&logo=github&color=blue)](https://github.com/gotonote/awesome-robot-hub/network)
[![License](https://img.shields.io/github/license/gotonote/awesome-robot-hub?style=flat-square&color=green)](./LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/gotonote/awesome-robot-hub?style=flat-square&color=purple)](https://github.com/gotonote/awesome-robot-hub/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/gotonote/awesome-robot-hub/pulls)
[![Made with Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg?style=flat-square)](https://www.markdownguide.org/)

**🚀 Star ⭐ 一下，让更多人看到这份持续更新的学习资料！**

</div>

---

## 📖 目录

- [🎯 这是什么](#-这是什么)
- [✨ 特性亮点](#-特性亮点)
- [🗺️ 内容地图（15 章）](#️-内容地图15-章)
- [🧭 学习路径](#-学习路径)
- [🚀 快速开始](#-快速开始)
- [📅 每日前沿](#-每日前沿)
- [🤝 参与贡献](#-参与贡献)
- [🛣️ Roadmap](#️-roadmap)
- [📈 Star History](#-star-history)
- [📜 更新日志](#-更新日志)
- [📄 License](#-license)
- [📮 联系方式](#-联系方式)

---

## 🎯 这是什么

**物理 AI（Physical AI）** 又称 **具身智能（Embodied AI）**，指具有物理实体、能够在真实或模拟环境中感知、理解并与世界交互的智能系统。

本仓库把碎片化的领域知识整理成 **15 个章节、60+ 篇文档、约 2 万行内容** 的完整学习体系，覆盖从「机器人学基础」到「Diffusion Policy / VLA 大模型」的完整技术栈，**适合初学者到进阶研究者**。

### 与传统 AI 的区别

| 维度 | 传统 AI | 物理 AI / 具身智能 |
|------|---------|-------------------|
| 载体 | 纯软件 / 数据 | 物理实体（机器人） |
| 交互 | 被动接收数据 | 与环境主动交互 |
| 学习 | 静态数据集 | 在线交互 / 试错 |
| 目标 | 理解世界 | **理解 + 改变**世界 |

---

## ✨ 特性亮点

- ✅ **体系化**：15 章由浅入深，目录即学习路线，无需自己拼凑资料
- ✅ **持续更新**：前沿动态、论文解读每周更新，紧跟 NVIDIA GR00T、Unitree、Figure 等行业进展
- ✅ **代码可跑**：实战章节提供完整 Python 实现（PyBullet / PyTorch / Gymnasium）
- ✅ **图文并茂**：含原理图解、算法公式、对比表格，阅读体验友好
- ✅ **中文友好**：全中文讲解，降低非英语母语者学习门槛
- ✅ **开放协作**：欢迎任何人补充内容、修正错误、翻译分享

---

## 🗺️ 内容地图（15 章）

```
awesome-robot-hub/
├── 01_入门指引/      🚪 什么是物理AI、发展历程、学习路径
├── 02_基础概念/      🧮 机器人学、运动学、动力学、状态估计
├── 03_感知技术/      👁️ 计算机视觉、深度估计、传感器融合、事件相机
├── 04_运动控制/      🎮 路径规划、MPC、避障、强化学习运动控制
├── 05_强化学习/      🧠 RL基础、PPO、SAC、DQN、Sim-to-Real
├── 06_模仿学习/      📋 行为克隆、DAgger、GAIL、IRL
├── 07_世界模型/      🌍 世界模型概论、前沿进展
├── 08_多模态大模型/  🗣️ VLM、VLA、RT系列、PaLM-E、ALOHA/ACT
├── 09_扩散模型/      🎨 Diffusion Policy、生成式动作
├── 10_经典论文/      📄 里程碑论文解读（PPO、DQN、RT系列…）
├── 11_产业应用/      🏭 自动驾驶、人形机器人、医疗、工业
├── 12_前沿动态/      📰 最新论文、公司动态、技术趋势
├── 13_开源项目/      🔧 ROS/ROS2、机械臂、人形机器人、传感器
├── 14_仿真环境/      🖥️ MuJoCo、Isaac Sim、PyBullet、SAPIEN
└── 15_实战教程/      💻 可运行代码案例、Demo
```

**内容规模**：`15` 章节 · `60+` 篇文档 · `~20000` 行 Markdown

---

## 🧭 学习路径

| 阶段 | 章节 | 内容 |
|------|------|------|
| 🚪 入门 | 01_入门指引 | 什么是物理AI、发展历程 |
| 🧮 基础 | 02_基础概念 | 机器人学、运动学、动力学 |
| 👁️ 感知 | 03_感知技术 | 视觉、深度感知、传感器融合 |
| 🎮 控制 | 04_运动控制 | 规划、轨迹、MPC、控制算法 |
| 🧠 学习 | 05_强化学习 / 06_模仿学习 | PPO、SAC、DAgger、GAIL |
| 🌍 前沿 | 07_世界模型 / 08_多模态大模型 / 09_扩散模型 | World Model、VLA、Diffusion Policy |
| 📄 经典 | 10_经典论文 | 里程碑论文解读 |
| 🏭 应用 | 11_产业应用 / 12_前沿动态 / 13_开源项目 | 产业落地、最新进展、开源生态 |
| 💻 实践 | 14_仿真环境 / 15_实战教程 | MuJoCo、Isaac Sim、代码实战 |

> 💡 建议从 `01_入门指引` 开始，按章节顺序学习；有基础的同学可直接跳到感兴趣的章节。

---

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/gotonote/awesome-robot-hub.git
cd awesome-robot-hub

# 直接浏览（推荐）—— 在浏览器中打开任意章节
# 或使用支持 Markdown 预览的编辑器（VS Code / Typora 等）
```

> 📌 无需安装任何依赖，纯 Markdown 内容，GitHub 网页端即可直接阅读。

---

## 📅 每日前沿

每日更新无人驾驶 / 物理AI领域最新动态、公司进展与技术突破：

👉 **[进入每日前沿 →](./12_前沿动态/)**

---

## 🤝 参与贡献

欢迎一切形式的贡献：**补充内容、修正错误、优化排版、翻译推广**。

- 详细流程见 [CONTRIBUTING.md](./CONTRIBUTING.md)
- 提内容建议/报告问题 → [Issues](https://github.com/gotonote/awesome-robot-hub/issues)
- 提交修改 → [Pull Requests](https://github.com/gotonote/awesome-robot-hub/pulls)

```bash
git clone https://github.com/gotonote/awesome-robot-hub.git
git checkout -b feature/xxx
# ... 编辑内容 ...
git commit -m "feat: 添加 xxx 内容"
git push origin feature/xxx
# 然后提交 Pull Request
```

---

## 🛣️ Roadmap

- [x] 15 章节基础框架搭建
- [x] 感知 / 控制 / 强化学习 / 模仿学习核心内容
- [x] 每日前沿动态持续更新
- [ ] 更多实战代码与可复现 Demo
- [ ] 英文版内容（国际化推广）
- [ ] 知识图谱 / 可视化学习路线
- [ ] 配套视频讲解

> 欢迎在 [Issues](https://github.com/gotonote/awesome-robot-hub/issues) 中提出你的建议！

---

## 📈 Star History

**Star 增长趋势**（每日自动更新）：

<div align="center">

[![Star 增长趋势](https://cdn.jsdelivr.net/gh/gotonote/awesome-robot-hub@main/docs/star-chart.svg)](https://github.com/gotonote/awesome-robot-hub/stargazers)

</div>

> 💡 若图片未加载，请点击上方链接直接访问 [Star 列表](https://github.com/gotonote/awesome-robot-hub/stargazers) 或 [star-history.com](https://star-history.com/#gotonote/awesome-robot-hub&Date) 查看完整增长曲线。
>
> 📌 图片由 jsDelivr CDN 托管（push 后自动刷新）；若在仓库 Settings → Pages 开启 GitHub Actions 部署，可改用 `https://gotonote.github.io/awesome-robot-hub/star-chart.svg`。

---

## 📜 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-02-26 | 运动控制：强化学习运动控制详解（MDP、贝尔曼方程、Q-Learning、DQN、策略梯度、Actor-Critic、DDPG、PPO、SAC） |
| 2026-02-25 | 运动控制：模型预测控制 MPC 详解（含 Python 实现） |
| 2026-02-25 | 感知技术：事件相机技术详解（动态视觉传感器） |
| 2026-02-24 | 基础概念：传感器与状态估计基础（Kalman 滤波、粒子滤波） |
| 2026-02-23 | 入门指引：2025-2026 最新发展（NVIDIA GR00T、Figure 02、Unitree G1） |
| 2026-02-22 | 世界模型：前沿进展更新 |

*完整历史见 [Git Commits](https://github.com/gotonote/awesome-robot-hub/commits/main)*

---

## 📄 License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本仓库基于 **MIT License** 开源，可自由学习、使用、转载，请保留署名。

---

## 📮 联系方式

- 💬 内容建议 / 错误指正 → [GitHub Issues](https://github.com/gotonote/awesome-robot-hub/issues)
- ⭐ 觉得有用请 Star，帮助更多人看到
- 📨 欢迎通过 PR 一起共建这份开源学习笔记

---

*本仓库持续更新中，如果对你有帮助，请点个 ⭐ 支持一下！*
