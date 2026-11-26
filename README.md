# Portrait Spine Studio

将一张透明立绘制作成可交付的 Spine 待机动画：**一次 Canvas 粗拆 → 净化与原图补底 → 立即出 8 秒预览 → 局部微调、人工签收导出**。身体和武器保形，软饰件轻摆；持续遮住的内容不强补。

Cindy 对话负责调用 Canvas、看图和规划图层；本地 Python 工具负责对齐、净化、骨架、渲染和导出。不是全自动语义拆分服务，也不需要安装专用 Cindy 连接插件。项目 Skill 随仓库分发：[四步 SOP](.agents/skills/portrait-spine-idle/SKILL.md)。

## 本机使用

依赖现有 Python 3.9+、Pillow、numpy；视频需要 ffmpeg。前端无 Node 构建或公网 CDN。

```bash
python3 bin/start-studio.py --port 8765 --no-open
```

在运行服务的电脑打开 http://127.0.0.1:8765/。上传原图后，可接入已对齐的 Canvas parts 和骨架规划，再调图层前后、显隐和单模块摆幅，完整循环看过后签收下载。

已有 parts 时可直接离线渲染，所有输入与新输出放在共同证据根下：

```bash
python3 scripts/canvas_to_spine.py --parts-dir <parts目录> --plan <plan.json> --character <角色标识> --out-dir <新输出目录> --evidence-root <任务根> --cleanup --fps 30
```

默认一次顶层粗拆，先分开人物、独立道具、武器和特效；人物仍为干净整层时，在本地切成完整大块后与 Canvas 层合并，不再递归拆子层。仅动态真穿帮且本地处理不足时，才在有效授权内补一次局部 Canvas；作品库原件不可读时先停生成。

## 已验证与交付

逐星暗影第 2 版、爆破新星完整手臂且无火焰残丝版均已获用户视觉签收。正式交付为 Spine 4.2 ZIP、深浅底 30fps MP4 和单文件 HTML。时空幻象仍是待读取恢复后的回归样本，不把两图通过写成全库通过。两个签收版实际绑定均为 airborne、无 IK；grounded 有合成测试支持，但真实站立脚锚还待回归，不能只凭逐星的站立外观宣称已验证。

- [使用指南](docs/usage.md)：原件交接、parts 物化、预览、Studio API、签收导出。
- [验收状态](docs/acceptance-summary.md)：当前产物、真实限制和历史记录。
- [局部返修](.agents/skills/portrait-spine-idle/references/rerun-and-troubleshooting.md)：残块、特效、缺件、接缝与读取故障。

`canvas_repair_workflow.py` 和 `connection_diagnostics.py` 已冻结，不作为当前主链门禁。旧诊断命令保留维护；未传人工签收参数的历史严格导出路径没有被删除。
