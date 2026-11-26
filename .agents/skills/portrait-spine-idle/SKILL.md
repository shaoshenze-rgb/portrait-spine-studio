---
name: portrait-spine-idle
description: 触发词「spine动画」（用户说“spine动画”或“Spine动画”时直接触发）。在对话中接收立绘素材，经一次 Canvas 粗拆、净化与原图补底，产出 8 秒 Spine 待机预览，支持本地网页交互微调或对话反馈后签收导出。
---

# Portrait Spine Idle

先出 8 秒预览，只修动起来真正穿帮的地方。每轮恢复先读本目标。
身体、武器保形，头发、衣摆、飘带轻摆；全循环始终被遮住的内容不用强补。
默认一次 Canvas 顶层粗拆，不递归拆子层；保留自然呼吸和软摆节奏。
四项基础检查后由用户看图签收；代码测试不等于角色验收。
当前事实见[验收状态](../../../docs/acceptance-summary.md)，具体命令见[使用指南](../../../docs/usage.md)。

## 1. 准备原图，一次粗拆并对齐

默认必须先启动 `python3 bin/start-studio.py --port 8765`，再用 `python3 scripts/portrait_task.py prepare --image <透明原图.png>` 创建任务，保留返回的 job_id 和 `matted.png`。在当前 Cindy 对话调用 Mivo Canvas，脚本本身不提交生成。

提交前先读通作品库原件：`read_images` 请求 original，并通过当前任务的原生读取映射实际打开。`unavailable`、`unauthorized`、`unauth_cache` 或没有映射时停止生成并报告，保留调用额度；不能猜缓存路径或重复收费。

在本任务生成授权内，对原图只做一次顶层 split-layers：优先分离悬浮道具、完整武器、独立特效和人物本体。保存真实 scene/node/task、原件与 SHA，用 `scripts/canvas_handoff.py import-layers --working-image <当前任务的matted.png>` 对齐。看原图对照，确认对齐可信、主要结构未丢。若人物本体干净但仍是整层，直接在第 2 步本地切分，不重提 Canvas。

## 2. 净化、补底，立即出预览

仅对干净人物整层在本地拆分：给出语义种子掩码或轮廓引导线，运行 `python3 scripts/split_local_body.py --body <对齐人物.png> --spec <分层规格.json> --out-dir <新目录>`。前后衣摆、领巾与臂后围巾尾分层。对被前景分隔的连续后层做严格遮挡内桥接，默认 56 px 本色深垫边、外沿 3 px 羽化，不改变可见像素。复查 `split-stats.json`、全层总览和静止重组；已独立的 Canvas 层不重切。

先看分件对照，确认归属和遮挡关系；把本地人体层与原独立道具、武器和特效合并成 `canvas_layers` RGBA 清单。本地层保留 Canvas 父层的 `provenance`，另记本地派生；`image/sha256` 指向派生图，详细字段见使用指南。按真实结构绑骨，武器随握持手，悬浮道具独立。用 `materialize_canvas_layer_parts` 生成标准 parts，再运行：

```bash
python3 scripts/portrait_task.py apply-canvas --job <job_id> --parts-dir <parts目录> --plan <plan.json> --cleanup
```

输入和输出均放在工作目录内。`--cleanup` 串联：

- 按原图前景加 1px 缓冲裁掉外部白块和透明洞残留，去半透明白边，保留轮廓内隐藏材质。
- 糊特效盖住主体时，将对应 `effect_group` 设为 `effect_mode="omit"`。停止绘制该层，补底排除外围和凸包内同色残丝；复查主体未被误删。
- 对 ≥200px 主体缺块，从原图恢复到邻接最长或中心最近的合法人体/服装层，带 2px 羽化重叠边；记录 `recovered_regions`，复查 `missing_original_regions`。覆盖恢复不自动证明归属正确。
- 以 `strict_placement=False` 编译，6px 静态距离进入 `placementWarnings`，不锁死预览；支持 grounded 与 airborne，站立图仍须提供真实足锚链。

完成后实际打开并提供 `http://127.0.0.1:8765/?job=<job_id>` 与单文件 HTML，确认播放器和微调、签收控件可见。深浅底 MP4、关键帧拼图、`seam-worst-crops.jpg` 和诊断 JSON 作辅助，不能以视频替代 Studio。仅无界面批处理时备用 `scripts/canvas_to_spine.py`。

## 3. 看完整循环，只做局部快修

在 Studio 复核自动接缝特写（原图、起始帧、局部最大变化帧），包括肩臂、头臂及自动选出的软层/本地切分交界。先消除 `local_layer_pixel_bleed`：用下层本色重做垫边，不靠调幅掩盖。遇 `separated_soft_islands` 核对深度与遮挡，必要时拆独立软层。16 个抽样时刻不能替代原速完整循环。

先在 Studio 看左侧舞台，实时调百分比、显隐与前后层级。100% 是角色规划的默认动作；单件与整体乘积封顶 150%，不按最近一次保存反复放大。武器随持握手。拖滑块不提交，确认后再保存并复看。

若仍有真实外露空洞且本地补底、垫边无法解决，才在有效授权内对该单一局部最多补一次 Canvas；不拆子层、不重做整套。结果未知先找回原任务，不重提。处理方法见[局部返修](references/rerun-and-troubleshooting.md)。

## 4. 四项基础检查，人工签收导出

确认：①骨架和贴图完整可读；②8 秒首尾无缝；③静止渲染对齐当前分层合成，原图重要结构无大错位、主体无大块漏件；④刚体无明显拉伸。其余静态间距、边缘微像素保留为提示。

全身另做 100% / 150% 压力扫描，检查主体内部封闭洞和接触带开口露底。复核 `all-layers-sheet.jpg`、150% 特写与原速循环；目标零新露底，存在可见穿帮就返修，超阈值不得签收。不要靠调低默认幅度冒充补全通过。Studio 压力与基础检查通过后，由用户签收。脚本导出也先核压力证据，再写真正的 `user-signoff.json` 并调用 `export_default_package`；底层导出器不替代这一步。

## 红线

- 不把软布摆动或呼吸压到近乎静止来过压力检查；保持自然可见的默认动作，用后层桥接和深垫边修露底。
- 不做二、三级递归拆子层；新图一次粗拆，真穿帮才局部补一次。
- 不用静态 6px 距离或零星半透明边缘锁死预览与已签收导出。
- 不重新启用冻结的 `canvas_repair_workflow.py` / `connection_diagnostics.py` 作为主链门禁。
- 不写角色名、文件名或哈希特判；代表图只是验证样本。
