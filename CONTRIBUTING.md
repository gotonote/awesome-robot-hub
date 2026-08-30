# 🤝 贡献指南

感谢你对 **Awesome Robot Hub** 感兴趣！本仓库是社区共建的开源学习笔记，欢迎一切形式的贡献：

- 📝 补充 / 修正内容
- 🐛 报告失效链接、错别字、格式问题
- 🌐 翻译成英文或其他语言
- 💡 提出学习路线、Roadmap 建议

## 贡献方式

### 1. 报告问题 / 提出建议

前往 [Issues](https://github.com/gotonote/awesome-robot-hub/issues) 创建 Issue，请尽量：

- 使用清晰的标题，如 `[内容补充] 04_运动控制 缺少 xxx`、`[链接失效] 01_入门指引/01-数学物理基础.md`
- 说明具体位置（文件路径 / 章节标题）
- 附上修改建议或参考资料链接

### 2. 提交内容（Pull Request）

```bash
# 1. Fork 本仓库
# 2. 克隆你的 Fork 并创建分支
git clone https://github.com/<你的用户名>/awesome-robot-hub.git
cd awesome-robot-hub
git checkout -b feature/添加xxx内容

# 3. 编辑内容（Markdown）
# 4. 提交并推送
git add .
git commit -m "feat: 添加 xxx 内容"
git push origin feature/添加xxx内容

# 5. 在 GitHub 上发起 Pull Request
```

## 内容规范

为保持仓库一致性，贡献内容请遵循以下规范：

### 目录 / 编号

- 章节目录固定为 `NN_章节名` 格式，NN 为两位数字编号
- 每个章节目录下应有 `README.md` 作为章节入口，列出该章节的文档索引
- 新增文档请放入对应章节目录，并在该章节 `README.md` 中登记

### 文档格式

- 使用标准 Markdown，文件名建议用中文或语义化英文（如 `04-模型预测控制MPC.md`）
- 文档开头用 `# 标题` + 简短摘要（`>` 引用块）
- 正文用 `##`/`###` 分级，善用表格、代码块、引用
- 涉及算法建议附伪代码或可运行的 Python 示例
- 引用外部资源请附上可点击链接

### 链接规范

- 仓库内部链接使用相对路径，例如 `[03_感知技术](./03_感知技术/README.md)`
- 提交前请检查链接有效性（仓库 CI 会自动检查内部链接）

### 更新日志

- 重要内容更新请在根目录 `README.md` 的「更新日志」表格中追加一行
- 格式：`| 日期 | 更新内容 |`，日期格式 `YYYY-MM-DD`

## Commit 规范

推荐使用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/) 风格：

| 类型 | 示例 |
|------|------|
| `feat:` | `feat: 添加模型预测控制MPC详解` |
| `docs:` | `docs: 修正章节编号错误` |
| `fix:` | `fix: 修复失效链接` |
| `refactor:` | `refactor: 整理目录结构` |

## Review 流程

- 维护者会尽快 review PR，请耐心等待
- 修改意见会以评论形式提出，解决后即可合并
- 合并后你的名字会出现在贡献者列表中 🎉

## 感谢参与！

任何形式的贡献都值得被尊重，哪怕只是一个错别字的修正。让我们一起把这份学习笔记越做越好！
