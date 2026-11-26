---
name: portrait-spine-idle
description: 触发词「spine动画」（用户说“spine动画”或“Spine动画”时直接触发）。在对话中接收立绘素材，默认本机直通完成去底、边缘贴合分层、后层桥接补全与 100%/150% 全身压力体检，首轮直出可交互 Studio 实时预览网页；仅复杂遮挡或糊特效时按需局部补一次 Canvas。
---

# Portrait Spine Idle

先出 8 秒实时可调预览，只修动起来真正穿帮的地方。每轮恢复先读本目标。
身体、武器保形，头发、衣摆、飘带轻摆；全循环始终被遮住的内容不用强补。
默认**本机直通优先**（新会话第 1 轮不依赖 Mivo Canvas，零停顿、零付费直出 Studio 网页）；复杂背景、大面积被挡武器或糊特效才按需补一次 Canvas，不递归拆子层。
四项基础检查加 100%/150% 压力体检后由用户看图签收；代码测试不等于角色验收。
当前事实见[验收状态](../../../docs/acceptance-summary.md)，具体命令见[使用指南](../../../docs/usage.md)。

## 1. 本机直通准备原图（第 1 轮直接启动，不先调 Canvas）

1. 启动或确认本机 Studio 在线：`python3 bin/start-studio.py --port 8765`（先查 `http://127.0.0.1:8765/api/health`）。
2. 直接读取用户本轮对话附件原图：
   - **已是透明 PNG**：直接进入第 3 步。
   - **带不透明背景（JPG/PNG）**：用 `.venv/bin/python3 scripts/local_matte.py --input <附件原图> --out <透明原图.png> --check-out <去底对照.jpg>` 本机去底，目检 `<去底对照.jpg>` 确认人物轮廓干净、未误抠内部高光。
3. 创建 Studio 任务并拿到 `job_id` 与规范工作图 `matted.png`：

```bash
python3 scripts/portrait_task.py prepare --image <透明原图.png> --character <角色标识>
```

## 2. 本机语义切分、后层桥接补全，立即出实时预览

1. 读图规划语义模块（附中文 `label`，如头部、躯干、武器、悬浮道具、前后衣摆、领口围巾、围巾尾、左右臂腿与站立双脚）。前后衣摆、领巾与臂后围巾尾必须按前后深度拆开。
2. 给出语义种子掩码或轮廓引导线，运行本机统一分层入口：

```bash
.venv/bin/python3 scripts/split_local_body.py --body <当前任务的matted.png> --spec <分层规格.json> --out-dir <新分层目录>
```

   自动完成真实边缘贴合、高 `draw_order` 前景优先（零外溢）、被前景分隔的连续后层遮挡内桥接补全，以及默认 56 px 平滑本色深垫边（外沿 3 px 羽化，不改首帧可见像素）。复查 `split-stats.json`、全层总览和静止重组。
3. 组装 `material.json`（每层以 `provenance.operation = "local-direct-split"` 如实记录 `source_asset_sha256`、`result_asset_sha256`、单位 `transform` 和 `local_derivation`，不伪造 Canvas 字段）与 `plan.json`（武器随握持手、悬浮道具独立、站立图提供真实双足锚链），调用 `materialize_canvas_layer_parts` 物化标准 parts，再灌入 Studio：

```bash
python3 scripts/portrait_task.py apply-canvas --job <job_id> --parts-dir <parts目录> --plan <plan.json> --cleanup
```

   `--cleanup` 自动串联轮廓裁切、去白边、糊特效 `omit` 清理、≥200 px 缺块原图恢复、`local_layer_pixel_bleed` 串层检测、`separated_soft_islands` 软层断岛检测，以及 `strict_placement=False` 编译（6 px 静态距离仅入提示，不锁预览）。
4. 完成后实际验证并提供可点开的实时工作台链接 `http://127.0.0.1:8765/?job=<job_id>` 与单文件 HTML，确认播放器和微调、签收控件可见。深浅底 MP4、关键帧拼图、`seam-worst-crops.jpg`、`all-layers-sheet.jpg` 和诊断 JSON 作辅助，不能以视频替代 Studio。仅无界面批处理时备用 `scripts/canvas_to_spine.py`。

## 3. 看完整循环，微调与按需局部 Canvas 精修

先在 Studio 复核 `all-layers-sheet.jpg` 和自动接缝特写（原图、起始帧、局部最大变化帧），包括肩臂、头臂及自动选出的软层/本地切分交界。先消除 `local_layer_pixel_bleed` 与 `separated_soft_islands`，不靠调幅掩盖。16 个抽样时刻不能替代原速完整循环。

在 Studio 左侧舞台直接看动画，右侧实时调百分比、显隐与前后层级。100% 是角色规划的默认动作；单件与整体乘积封顶 150%，不按最近一次保存反复放大。拖滑块不发保存请求，确认后再一键烘焙并复看。

**按需局部 Canvas（可选，默认不触发）**：仅当用户明确要求使用 Mivo Canvas，或存在本机去底/桥接补全确实无法解决的复杂背景、大面积被挡武器或糊特效时，才在有效授权内最多补一次 Canvas（`remove-bg` 或顶层 `split-layers`）。调用顺序严格遵守：
- 先调 `canvas_selection` 唤醒并确认面板就绪；若 `canvas_place_image` 返回 `REFERENCE_LIBRARY_CHANGED`，先 `canvas_selection` 确认后重试一次 `canvas_place_image`，**严禁跳过放图直接读画布上旧选中节点**。
- 调 `read_images(original)` 后优先用宿主注入的只读根路径实际打开（不要只查 shell 环境变量 `$LIBRARY_READ_ROOT`）。若新会话首轮已返回 `available` 但宿主尚未挂载只读目录，保留已入板节点、不报 `fail`，附一键三反引号 `cindy-ui` 按钮：`{"version":1,"seq":1,"op":"create","id":"resume-library-mount","kind":"button","props":{"label":"继续并开始拆层","send":"继续"}}`，第 2 轮复用原节点继续；仅返回 `unavailable` / `unauthorized` / `unauth_cache` 或第 2 轮仍无映射时按读取故障停止并报告。详情见[局部返修](references/rerun-and-troubleshooting.md)。

## 4. 四项基础检查，100%/150% 压力体检与人工签收导出

确认：①骨架和贴图完整可读；②8 秒首尾无缝；③静止渲染对齐当前分层合成，原图重要结构无大错位、主体无大块漏件；④刚体无明显拉伸。其余静态间距、边缘微像素保留为提示。

全身另做 100% / 150% 压力扫描，检查主体内部封闭洞和接触带开口露底。复核 `all-layers-sheet.jpg`、150% 特写与原速循环；目标零新露底，存在可见穿帮就返修，超阈值不得签收。Studio 压力与基础检查通过后由用户签收导出（ZIP、深浅底 MP4、单文件 HTML；透明网页视频按需用 `export_transparent_web`）。脚本导出同样先核压力证据并写真实 `user-signoff.json`。

## 红线

- 默认第 1 轮走本机直通（`local_matte.py` + `split_local_body.py` + Studio），不因 Mivo Canvas 冷启动或未打开而阻断首轮交付。
- 不把软布摆动或呼吸压到近乎静止来过压力检查；保持自然可见的默认动作，用后层桥接和深垫边修露底。
- 不做二、三级递归拆子层；绝不伪造 Canvas 回执字段，纯本地直分必须如实标记 `local-direct-split`。
- 不用静态 6 px 距离或零星半透明边缘锁死预览与已签收导出。
- 不重新启用冻结的 `canvas_repair_workflow.py` / `connection_diagnostics.py` 作为主链门禁。
- 不写角色名、文件名或哈希特判；代表图只是验证样本。
