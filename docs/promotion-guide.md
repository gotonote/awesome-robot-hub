# 🚀 Awesome Robot Hub 流量增长执行手册

> 本手册是把仓库推广落地的操作清单。按顺序执行，每一步都有具体动作和现成文案。

---

## 一、GitHub 仓库设置（5 分钟，一次性，需在网页操作）

### 1. 添加 Topics（带来 GitHub 搜索流量）

打开仓库主页 → 右侧 **About** 区域 → ⚙️ 设置 → **Topics** 依次添加：

```
physical-ai  embodied-ai  robotics  robot-learning
reinforcement-learning  imitation-learning  world-models
vision-language-action  diffusion-policy  sim2real
mujoco  isaac-sim  具身智能
```

### 2. 开启 Discussions（激活社区感）

仓库 **Settings → General → Features** → 勾选 **Discussions**。
建议建两个分类：
- `🎓 学习交流`：读者提问、答疑
- `📰 每日前沿`：把 12_前沿动态 的每日更新同步发在这里（比 README 曝光更高，订阅者能收到通知）

### 3. 完善 About 描述

仓库 **About** 区域填写：

> 物理 AI / 具身智能（Embodied AI）全栈学习笔记：感知 → 控制 → RL → 模仿学习 → 世界模型 → VLA 大模型 → 产业应用。15 章 60+ 文档，中文友好，持续更新。

Website 填 `https://gotonote.github.io/awesome-robot-hub/`（已有 Pages 部署则直接填）。

---

## 二、加入 awesome 列表（涨 star 最快路径，需提 PR）

按优先级逐个给以下列表提交 PR，把自己的仓库加进对应类目。**PR 文案模板**（中英双语，可直接复制）：

> **Add: [Awesome Robot Hub](https://github.com/gotonote/awesome-robot-hub)**
> A beginner-friendly, Chinese-first learning path for Physical AI / Embodied AI — 15 chapters, 60+ docs covering perception → control → RL → imitation learning → world models → VLA → diffusion policies → industry applications. Continuously updated.
> （中文学习笔记：具身智能 / 物理 AI 全栈学习路线，15 章 60+ 文档，持续更新中。）

### 优先级 A（最可能收录学习资源）

| 列表 | ⭐ | 收录理由 | 加在哪个类目 |
|------|-----|---------|-------------|
| [wadeKeith/Awesome-Embodied-AI](https://github.com/wadeKeith/Awesome-Embodied-AI) | 240 | 明确含 toolkits / learning 资源类目，10 大 track | Learning / Resources 类目 |
| [jonyzhang2023/awesome-embodied-vla-va-vln](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln) | 3.5k | embodied AI / VLA 头部列表 | Embodied Learning / Learning 类目 |
| [YanjieZe/awesome-humanoid-robot-learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning) | 2.7k | 人形机器人学习，含教程/资源 | Learning Resources 类目 |
| [zchoi/Awesome-Embodied-Robotics-and-Agent](https://github.com/zchoi/Awesome-Embodied-Robotics-and-Agent) | 1.9k | embodied robotics + LLM | Interactive Embodied Learning 类目 |

### 优先级 B（相关度高）

| 列表 | ⭐ | 收录理由 |
|------|-----|---------|
| [jonyzhang2023/awesome-humanoid-learning](https://github.com/jonyzhang2023/awesome-humanoid-learning) | 942 | humanoid 资源型列表 |
| [iLearn-Lab/VLA-Diffusion-Policy-Robotics](https://github.com/iLearn-Lab/VLA-Diffusion-Policy-Robotics) | 825 | VLA / Diffusion Policy（09 章契合） |
| [haoranD/Awesome-Embodied-AI](https://github.com/haoranD/Awesome-Embodied-AI) | 528 | embodied AI 列表 |
| [ugurkanates/awesome-real-world-rl](https://github.com/ugurkanates/awesome-real-world-rl) | 458 | real-world RL / Sim-to-Real（05 章契合） |

### 操作步骤

```bash
# 以 wadeKeith/Awesome-Embodied-AI 为例
gh repo fork wadeKeith/Awesome-Embodied-AI --clone
cd Awesome-Embodied-AI
git checkout -b add/awesome-robot-hub
# 在 README 对应类目下追加一行：
# - [Awesome Robot Hub](https://github.com/gotonote/awesome-robot-hub) - 具身智能全栈中文学习笔记（15 章 60+ 文档）
git add README.md
git commit -m "Add Awesome Robot Hub (Embodied AI learning path)"
git push origin add/awesome-robot-hub
# 然后到原仓库发起 Pull Request
```

> 💡 没有 `gh` CLI 也可以直接在 GitHub 网页操作：打开列表仓库 → 编辑 README.md → 直接发起 PR。

---

## 三、平台推广文案（复制即用）

### 1. 知乎回答《如何系统入门具身智能 / 物理 AI？》

> 具身智能这两年火到什么程度？NVIDIA 发布 GR00T、Figure 机器人进厂打工、VLA 大模型刷屏——但真正想入门的人往往被碎片化资料劝退：论文看不懂、代码跑不通、资料全是英文。
>
> 我整理了一份**开源的中文全栈学习路线**（GitHub 15 章 60+ 文档，持续更新）：
>
> 1️⃣ **入门**：什么是物理 AI、发展历程
> 2️⃣ **基础**：机器人学、运动学、动力学
> 3️⃣ **感知**：计算机视觉、深度估计、传感器融合
> 4️⃣ **控制**：路径规划、MPC、强化学习控制
> 5️⃣ **学习**：RL（PPO/SAC/DQN）、模仿学习（BC/GAIL）、Sim-to-Real
> 6️⃣ **前沿**：世界模型、VLA 大模型（RT 系列/PaLM-E/ACT）、Diffusion Policy
> 7️⃣ **实战**：PyBullet/PyTorch 可运行代码 + MuJoCo/Isaac Sim 仿真环境
>
> 地址：https://github.com/gotonote/awesome-robot-hub
> 觉得有用欢迎 Star ⭐，让更多人看到～

### 2. 公众号/掘金文章《具身智能学习路线：从零到 Diffusion Policy 只差这份地图》

- 开头：行业热点切入（Figure/宇树/NVIDIA GR00T）
- 正文：按 5 个阶段展开（基础→感知→控制→学习→前沿），每阶段引用仓库对应章节
- 插入 3-4 张仓库截图（目录表格、实战代码）
- 结尾：附仓库链接 + "已整理 15 章 60+ 文档，持续更新中，欢迎共建"

### 3. X / Twitter 英文短推

> Just open-sourced a **Chinese-first learning path for Embodied AI** 🦾
>
> 15 chapters, 60+ docs:
> perception → control → RL → imitation learning → world models → VLA → diffusion policies → real-world applications
>
> Beginner-friendly, with runnable code (PyBullet / PyTorch / MuJoCo).
> ⭐ https://github.com/gotonote/awesome-robot-hub
>
> #EmbodiedAI #PhysicalAI #Robotics #VLA

### 4. 即刻/朋友圈

> 花了几个月整理的具身智能全栈学习笔记，15 章 60+ 文档全中文，从机器人学基础到 VLA 大模型 / Diffusion Policy，还带可跑代码。持续更新中，求 Star ⭐ https://github.com/gotonote/awesome-robot-hub

---

## 四、内容策略（保持增长势头）

- **保持每日/每周更新**：GitHub Trending 偏好近期活跃仓库；每次新增文档都会触发徽章自动更新（已配置 update-badges workflow），commit 记录就是活跃度证明
- **图解化**：为每章补充 1 张原理图（draw.io / excalidraw），图文并茂的仓库转发率高得多
- **每月一次"冲刺"**：挑一周集中发布 3-5 篇干货（前沿论文解读同步发知乎/公众号/掘金），配合多平台分发冲击 Trending

---

## 五、已实现的自动化（本仓库自带）

| 自动化 | 触发 | 作用 |
|--------|------|------|
| `update-stars.yml` | 每日 16:00 UTC | 自动更新 Star 增长图 |
| `update-badges.yml` | 每日 02:00 UTC + push | 自动校准 README 章节/文档徽章 |
| `markdown-link-check.yml` | push / PR | 自动检查内部链接有效性 |
| `deploy-pages.yml` | push | 自动部署 GitHub Pages |

> 每次新增/删除文档，徽章数字会在下一次 push 或每日定时任务中自动校准，永远真实。
