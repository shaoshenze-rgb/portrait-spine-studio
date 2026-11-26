# Portrait Spine Studio

将一张立绘制作成可交付的 Spine 待机动画：**本机直通（按需去底 → 边缘贴合分层与后层桥接补全 → 100%/150% 压力体检 → 立即出 Studio 实时预览网页）→ 人工微调签收导出**。新会话第 1 轮零停顿直出，仅复杂遮挡武器或糊特效时按需局部补一次 Canvas。身体和武器保形，软饰件轻摆；持续遮住的内容不强补。

Cindy 对话负责调用 Canvas、看图和规划图层；本地 Python 工具负责对齐、净化、骨架、渲染和导出。不是全自动语义拆分服务，也不需要安装专用 Cindy 连接插件。项目 Skill 随仓库分发：[四步 SOP](.agents/skills/portrait-spine-idle/SKILL.md)。

## 本机使用

依赖现有 Python 3.9+、Pillow、numpy；视频需要 ffmpeg。前端无 Node 构建或公网 CDN。

```bash
python3 bin/start-studio.py --port 8765 --no-open
```

默认工作流：先启动本机 Studio，再用 `portrait_task.py prepare` 与 `apply-canvas` 加载任务，打开 `http://127.0.0.1:8765/?job=<job_id>`。工作台左右分栏：左侧实时播放干净待机动画（默认关闭骨骼线），右侧按中文部件名与缩略图实时微调动效幅度（0%–150%）、前后遮挡和显隐，查看自动接缝特写与 100%/150% 全身露底体检后一键烘焙签收导出。

已有 parts 时可直接离线渲染，所有输入与新输出放在共同证据根下：

```bash
python3 scripts/canvas_to_spine.py --parts-dir <parts目录> --plan <plan.json> --character <角色标识> --out-dir <新输出目录> --evidence-root <任务根> --cleanup --fps 30
```

默认本机直通：带背景原图先经 `scripts/local_matte.py` 去底并看三联对照图，透明原图直接经 `scripts/split_local_body.py`（底层 `src/pipeline/local_body_split.py`）按真实边缘贴合、前后景分层、后层遮挡桥接补全与深本色垫边切分，以 `local-direct-split` 来源直进 Studio 实时工作台。公共诊断自动检测同源像素串层、软层大断岛与 100%/150% 满滑块全身露底，红线禁止靠冻住幅度混过检查。仅复杂背景、大面积被挡武器或糊特效且本地补全不足时，才按需补一次 Canvas。

## 文件夹批量处理与真 4K 统一导出

面向 20–30 张立绘的批量生产，提供 `scripts/batch_portrait.py` 与批量验收工作台 `http://127.0.0.1:8765/?batch=<batch_id>`：支持直接扫描本地文件夹并行去底并输出 `matte-overview.jpg` 总览图，输入上限放开至 `4096px` 保留真 4K 清晰度；每个角色支持网页内一键「确认锁定」（服务端 `409 JOB_LOCKED` 写保护 + 重启持久化，重扫目录与返修绝不覆盖已确认项）或填备注「打回待修」集中返修；全部过目后可按统一导出规格（默认勾选 **4K 竖屏 2160×3840 + 底边居中对齐 + 透明网页视频**，并支持 4K 横屏/长边 4096/2K/1080p/自定义及多选 Spine 4.2 ZIP / MP4 / HTML）一键批量导出并生成 `batch-export-manifest.json`。

## 已验证与交付

逐星暗影第 2 版、爆破新星完整手臂且无火焰残丝版均已获用户视觉签收并导出四件套（Spine 4.2 ZIP、深浅底 30fps MP4、单文件 HTML）。时空幻象（遮挡武器 + 站立 `grounded` 双足锚链）已打通作品库读取、单次 Canvas 三层粗拆、17 层后层桥接补全、自然待机幅度及 100%/150% 满滑块全身压力门禁，Studio 实时预览就绪，待人工签收导出正式包。

- [使用指南](docs/usage.md)：原件交接、parts 物化、预览、Studio API、签收导出。
- [验收状态](docs/acceptance-summary.md)：当前产物、真实限制和历史记录。
- [局部返修](.agents/skills/portrait-spine-idle/references/rerun-and-troubleshooting.md)：残块、特效、缺件、接缝与读取故障。

`canvas_repair_workflow.py` 和 `connection_diagnostics.py` 已冻结，不作为当前主链门禁。旧诊断命令保留维护；未传人工签收参数的历史严格导出路径没有被删除。
