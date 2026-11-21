# Portrait Spine Studio 使用与整包分发指南

Portrait Spine Studio 是基于 Python 3.9+ 标准库与零外部前端依赖的本机离线立绘 Spine 4.2 待机动画工作台。无需安装 Node.js 依赖、无需公网 CDN，适合在本机与团队同事之间整包分发运行。

## 1. 运行环境要求

- **Python 3.9+**，已安装 `Pillow`（PIL）与 `numpy`
- **可选：`ffmpeg`**（仅在需要导出深/浅底预览 MP4 或透明 Web 视频包时需要；导出默认 Spine 4.2 `.zip` 工程包不依赖 `ffmpeg`）
- **Cindy + Mivo Canvas 环境说明**：
  - 在 Cindy 工作环境中，Mivo Canvas 作为图像抠图与遮挡补全能力层通过 `src/mivo/adapter.py` 接入（支持基于目录交接的 `FileBackedMivoAdapter` 与离线确定性回退 `DeterministicFallbackMock`）。
  - 任意阶段若发生 Mivo Canvas 异常或硬门禁失败，服务严格执行 **Fail-Stop**：自动写入 `<work_dir>/<job_id>/stage-error.json`，完整保留此前阶段的中间产物，绝不静默放行，用户可在页面选择「重试当前阶段」或「放弃任务」。

## 2. 一键启动（干净目录即开即用）

在仓库根目录下运行：

```bash
python3 bin/start-studio.py
```

或使用 Shell 包装脚本：

```bash
bash bin/start-studio.sh
```

常用参数：

- `--port 8765`：指定本地端口（默认 `8765`）
- `--host 127.0.0.1`：默认仅绑定本机回环地址
- `--work-dir <路径>`：指定任务输出与缓存目录（默认 `<仓库根目录>/.studio-work`）
- `--no-open`：仅启动服务，不自动打开浏览器

启动后访问 `http://127.0.0.1:8765/` 即可进入三步工作台。

## 3. 三步使用流程

### Step 1：上传立绘与选择预设
- 拖入或选择带透明通道的 `PNG` / `WebP` 角色立绘。长边超过 `2400px` 时会在预处理阶段自动按 LANCZOS 等比缩放，并在页面明确提示。
- 选择动作预设：
  - `agile_full`（轻盈全身）：8s 轻盈重心摆动 + 4s 呼吸 + 灵动发丝跟动
  - `heavy_weapon_full`（重武器全身）：沉稳重心与大武器联动、反向脊柱缓冲
  - `bust_half`（半身立绘）：稳固下身锚定、突出胸腔呼吸与头丝微动
- 可选勾选 **「分件后暂停确认」**，在分件完成后先人工预览 `part_map.png` 再继续构建骨骼与动画。
- 系统采用 **单任务串行执行锁**：已有任务运行时按钮自动置灰，并发请求将返回 `409 Conflict` 中文提示。

### Step 2：流水线进度、分件暂停与 Fail-Stop 恢复
- 依次展示 5 个阶段：输入预处理 → Mivo 抠图 → 分层分件与遮挡补全 → 骨骼/权重/动画构建 → 4 项硬门禁验收。
- 若开启了分件暂停，会展示 `part_map.png` 与保留的中间产物路径，点击 **「确认继续生成」** 即可续跑。
- 若任一阶段失败，页面显示错误码、失败阶段与已保留的中间产物，支持 **「重试当前阶段」**（复用中间产物）或 **「放弃任务」**。

### Step 3：Spine 4.2 实时预览、自然语言微调与导出
- **实时预览**：由 `/player/spine42-runtime.js` + `/player/canvas-renderer.js` + `/player/player-widget.js` 驱动，支持深底 / 浅底 / 棋盘格切换、播放/暂停、`0.5x`/`1.0x`、8.0s 时间轴拖动、第 0 帧对照、分件叠加与骨骼/IK 线框。
- **实时健康面板**：展示 4 项硬门禁（`rest_diff`、`setup_identity`、`no_holes`、`anchor_stability`）与 4 项软位移提示（`head`、`chest`、`hips`、`hair_tip`）。
- **自然语言微调（最小层级重跑）**：
  - **Layer A（动画层）**：如「右边长发摆动加大」「整体呼吸更柔和」或调节 `gain`/预设 → 跳过预处理、Mivo 与分件，仅重跑 `build_rig -> audit`。
  - **Layer B（绑定层）**：如「左手固定别动」「膝盖反折修正」 → 跳过预处理、Mivo 与分件，仅重跑 `build_rig -> audit`。
  - **Layer C（分件/补全层）**：如「重新拆分披风并扩大遮挡补全」 → 跳过预处理与 Mivo，重跑 `split_parts -> build_rig -> audit`（未改动分件命中 SHA-256 缓存）。
- **三类导出**：
  1. **导出 Spine 4.2 工程包（默认 `.zip`）**：包含 `<character>.spine.json`、`images/*.png`、`rig-build-report.json`、`health-report.json`。
  2. **导出深 / 浅底预览视频（`.mp4`）**：需本机可用 `ffmpeg`。
  3. **按需导出透明 Web 视频包**：生成 Side-by-Side H.264 / AV1 MP4、VP9 Alpha WebM 与首帧 `poster.webp`（需 `ffmpeg`）。

## 4. 整包分发给同事与验收命令

将整个 `portrait-spine-studio/` 目录（排除 `.studio-work/` 与 `__pycache__/`）打包发给同事，解压至任意干净目录后直接执行 `python3 bin/start-studio.py` 即可启动。

分发前可运行以下自动化验收命令：

```bash
python3 -m unittest discover -s tests/studio -v
python3 -m unittest discover -s tests -v
node tests/player/test_spine42_runtime.mjs
```
