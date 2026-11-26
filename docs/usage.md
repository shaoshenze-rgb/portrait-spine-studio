# Portrait Spine Studio 使用指南

默认主链是“启动 Studio → 本机读取/按需去底并准备任务 → `scripts/split_local_body.py` 边缘贴合与后层桥接补全 → 净化与 100%/150% 全身压力体检 → Studio 实时网页微调与人工签收”（首轮零 Canvas 依赖、零停顿）；仅复杂背景、大面积被挡武器或糊特效时按需局部补一次 Canvas。交付必须包含可打开的任务页和单文件 HTML，不只给视频。[验收状态](acceptance-summary.md)区分代表图通过与待验证内容。

## 启动和准备

在项目根执行，使用已有 Python 3.9+、Pillow、numpy 和 ffmpeg；不自动安装依赖。

```bash
python3 bin/start-studio.py --port 8765
# 带不透明背景的 JPG/PNG 先本机去底并看三联对照图：
.venv/bin/python3 scripts/local_matte.py --input <原始立绘> --out <透明原图.png> --check-out <去底对照.jpg>
python3 scripts/portrait_task.py prepare --image <透明原图.png> --preset agile_full
```

服务命令保持运行，prepare 在另一终端执行并返回 job_id。浏览器地址为 http://127.0.0.1:8765/，指运行浏览器那台电脑。已有透明图直接复用；带背景图先取得获授权的真实去底原件。重启不会恢复旧内存任务，保留旧文件后重新 prepare，不伪造恢复。

## 按需 Canvas 交接（可选精修）与本地直分来源

默认本地直分走 `scripts/split_local_body.py` 产出各层后，在 `material.json` 中使用一等本地来源 `provenance.operation = "local-direct-split"`（含 `source_asset_sha256`、`result_asset_sha256`、单位 `transform` 和 `local_derivation`，不伪造 Canvas 字段），经 `materialize_canvas_layer_parts` 直接接入 Studio `apply-canvas`。

仅当需要 Canvas 补拆大面积遮挡武器、复杂背景或糊特效时，再走下方 Canvas 交接：调用 `canvas_place_image` 前先 `canvas_selection` 唤醒面板，遇 `REFERENCE_LIBRARY_CHANGED` 重试一次放图，绝不读取画布旧选中节点。

### Canvas交接细节

由当前 Cindy 对话调用 Canvas，脚本不代为提交付费任务。先通过 `read_images(original)` 与宿主注入的只读根路径实际打开原件（不要只查 shell 环境变量 `$LIBRARY_READ_ROOT`）。若全新会话第 1 轮已返回 `available` 但只读映射要到第 2 轮才生效，保留已入板节点、不报故障，附一键 `cindy-ui`「继续并开始拆层」按钮触发下一轮（日常复用已跑通的同一会话不会再停）；若为 `unauth_cache` / `unavailable` / `unauthorized` 或第 2 轮仍无映射，则停止生成并记录精确字段。刷新面板或重启 Cindy 不是读取已恢复的证明。

每张新图默认只对原图做一次顶层 split-layers，优先分开人物本体、悬浮道具、完整武器和独立特效；禁止二、三级递归拆子层。下载层原件、核 SHA，保存真实回执，再对齐：

```bash
python3 scripts/canvas_handoff.py import-layers --receipt <回执.json> --source <本次Canvas输入.png> --working-image <当前任务的matted.png> --assets-dir <原件目录> --out-dir <新对齐目录>
```

成功产生 aligned PNG、变换、对照图和 import-report；附带 cleaned 候选及清理指标，原 aligned 层不覆盖。`needs_canvas_registration` 表示没有可靠位置，不猜坐标。保留主原图/工作图关系；主链不使用历史 `--parent-report` 递归拆层入口。

### 对齐层变成标准 parts

当前对话看图后准备两个 JSON：

- 材质清单：`version=1`、`material_mode=canvas_layers`、原图路径/SHA、画布尺寸、pieces。每 piece 提供 id、module_id、kind（rigid/soft）、parent、像素 pivot、draw_order、全画布 aligned RGBA 路径/SHA 与真实 provenance（scene/node/task、源/结果资产 SHA、相似变换、operation）。
- 骨架规划：pose_type、anatomy、modules、connections、occlusions、uncertainties。模块的 parent/pivot/type 与材质对应；连接用真实关节点。grounded 全身需两段刚体腿链及独立静态 foot 锚，airborne 采用浮空动作。规划不是根据文件名自动猜人体。

字段以[材质适配器](../src/pipeline/canvas_layer_parts.py)和[绑定编译器](../src/pipeline/canvas_layer_rig.py)为准。若只拿到干净人物整层，先标注语义种子掩码或左右轮廓，再运行：

```bash
python3 scripts/split_local_body.py --body <对齐人物.png> --spec <分层规格.json> --out-dir <新目录>
```

规格为 `version: 1`、`layers` 数组；每层声明 `id`、`kind`（rigid/soft）、`draw_order`，再二选一：`seed_mask`（相对规格文件路径，同画布二值 PNG，可设 `band_px`）；或 `contour`（`left_guides`、`right_guides`，各为等高、按 y 递增的像素点，可设 `radius`）。[分层模块](../src/pipeline/local_body_split.py) 用 GrabCut 或无 cv2 时的色差梯度贴边，轮廓分支用感知色差沿行寻边。命令输出 RGBA、可见归属掩码及 `split-stats.json`（含摘要、重叠与告警）；不自动决定语义。

前后衣摆、领巾与臂后围巾尾分别提供深度和软链。低层遇到被前景切开的连续布料，先在不透明遮挡内沿横纵跨度桥接，再以欧氏距离做默认 56 px 本色垫边（常用 40–64 px，外沿 3 px 羽化）。从本层向内轻蚀后取色，不拖暗描边，也不采样遮挡者；原可见像素不改。复核归属和完整性后再合并；已经独立的 Canvas 层不重切。

本地分件的 `provenance` 复用已验证 Canvas 父层的 `scene_id`、`node_id`、`task_id`、`operation`、`source_asset_sha256`、`result_asset_sha256`、`transform`，不将本地切分冒充新操作；附 `provenance.local_derivation` 记录父层对齐图 SHA、方法和掩码。分件自身的 `image` 和 `sha256` 指向本地派生 RGBA。已有标准 parts 可跳过物化；首次从清单物化：

```python
import json
from src.pipeline.canvas_layer_parts import materialize_canvas_layer_parts
manifest = json.load(open(".studio-work/task/material.json", encoding="utf-8"))
materialize_canvas_layer_parts(manifest, ".studio-work/task/parts", evidence_root=".studio-work")
```

## 净化、补底和离线批处理备选

默认走下节 Studio 接入。仅在无界面批处理时用下列离线命令；输入和新输出目录必须在 evidence-root 内，不覆盖旧结果。

```bash
python3 scripts/canvas_to_spine.py --parts-dir <parts目录> --plan <plan.json> --character <角色标识> --out-dir <新预览目录> --evidence-root <共同证据根> --cleanup --fps 30 --tweaks <微调.json>
```

`--tweaks` 可省略。`--cleanup` 启用原图轮廓裁切（前景加 1px 缓冲）、透明洞清理和半透明去白边；再从原图恢复 ≥200px 主体缺块，合入邻接最长或中心最近的合法人体/服装层并加 2px 羽化重叠边。

糊特效用 `effect_mode="omit"`：停止该槽绘制，恢复时排除外围及凸包内同色残丝，避免“关掉又补回来”。`keep` 保留，`original_extract` 从原图提取效果候选。对主体同色结构须看图复核；算法不保证任意图都不会误判。

微调示例（把示例 id 换成真实 piece/module id）：

```json
{"layers":{"effect_piece":{"effect_mode":"omit"},"arm_piece":{"drawOrder":3,"enabled":true}},"modules":{"arm":{"motion_scale":0.6}}}
```

单模块 `motion_scale` 为 0–1.5，0 可只停一个零件；保留全局呼吸和软摆。绑定采用 `strict_placement=False`，6px 静态距离写入 placementWarnings 而不阻断预览；文件、拓扑与真实站立链仍须有效。

产物：深浅底 8 秒 MP4、单文件 HTML、关键帧拼图、seam-worst-crops.jpg、preview-diagnostics.json。核对：

- `frame_count=240`、fps=30，首尾差在基础阈值内。
- `rest_composite_diff` 对照当前 setup_composite，不是证明原图所有像素未被改画。
- `cleanup` 记录净化、recovered_regions、主体 missing_original_regions；raw 缺失可能包含有意去掉的特效。
- `cleanup.warnings` 扫描实际贴图，即使跳过本地拆分 CLI 也会检查。发现 `local_layer_pixel_bleed` 必须重做下层垫边直到告警清零；发现 `separated_soft_islands` 要看图确认遮挡或按前后/左右分件。
- `dynamic_checks` 抽样 16 时刻，保留头臂、躯干臂相对摆角，并按局部边缘变化选最多四处软层或本地切分交界。Studio 直接显示告警与 `seam-worst-crops.jpg`；复核原图、起始帧和最大变化帧，再看原速完整循环。开放裂缝可能不形成洞。
- `exposure_stress` 使用默认动作的 100% 和 150%，在两档分别记录各采样帧的总露底、最大连通区域、位置和邻层。它检查静止主体内核的封闭洞及材质接触带的开口露底，排除独立道具、武器和特效。默认单处 >120 px 阻断 Studio 签收；低于阈值也要把任何肉眼可见的新露底修掉。必看 `all-layers-sheet.jpg`、`stress-150-frame.jpg` 和 150% 接缝对照。采样不能穷尽所有时间和滑块组合。

## Studio 接入与微调

用 `prepare` 得到的原图任务直接接入 parts 和规划 JSON：

```bash
python3 scripts/portrait_task.py apply-canvas --job <job_id> --parts-dir <parts目录> --plan <plan.json> --cleanup
```

成功后打开并交付返回的 `page_url`（`http://127.0.0.1:8765/?job=<job_id>`）与 `canvas_preview_html` 单文件 HTML。确认交互播放器、图层顺序/显隐、单模块滑块、应用和人工签收按钮直接可用。只把视频作为辅助。

本机 API：

| POST 路径 | 请求内容 | 结果 |
| --- | --- | --- |
| `/api/jobs/<id>/apply-canvas-layers` | `parts_dir`、`plan` 为本机路径，`cleanup: true`，可带 tweaks | 新八秒预览；支持 grounded/airborne |
| `/api/jobs/<id>/layer-tweaks` | layers/modules（`motion_ratio` 为默认动作的 0–1.5 倍）、`global_motion_scale`（0–1.8），可带当前 `expected_rig_version` | 相乘后每部件封顶 1.5 倍默认动作；实时和烘焙同口径，不累积放大 |
| `/api/jobs/<id>/user-signoff` | `{"decision":"pass","reviewer":"实际验收人"}` | 四项基础检查通过后，写签收并提供 ZIP/MP4/HTML 下载 |

工作台左侧播放、右侧调节。动效滑块、显隐、层级实时改变舞台，不发保存请求；武器与持握手臂联动，避免脱手。只有点击“保存实时微调”才烘焙。如果旧动作幅度为零，先保存以激活，再看完整循环并勾选验收。骨骼线默认关闭，原始诊断移入“开发者与技术详情”。需要区分同类部位时，可在分件清单或规划的模块上写可选中文 `label`。

## 四项基础检查与导出

签收前检查文件完整、八秒首尾无缝、静止对齐且主体无大块漏件、刚体无明显拉伸。原图的头身/四肢/武器完整性和实际前后关系由人眼复核，边缘微像素和静态间距仅提示。

Studio 自动检查文件、循环、主体覆盖与刚体形变，写 health 的静止合成结果；人眼看完整循环后点“人工验收通过并导出”。直接 Python 导出时，调用者先核 preview-diagnostics 和原图对照，再写实际签收 JSON：

```json
{"decision":"pass","user_quote":"用户对当前版本的实际确认原话","reviewer":"实际验收人"}
```

```python
from src.export.exporter import export_default_package
export_default_package(".studio-work/task/preview/rig", ".studio-work/task/final",
                       preview_fps=30, user_signoff_path=".studio-work/task/user-signoff.json")
```

底层签收出口核文件、显式签收及存在时的 rest_diff/刚体检查；不重新渲染验证全部四项，也不改写旧机器 pending 为 pass。未传签收参数时仍走旧严格出口。改材质、层序或动作后，必须对应新视频重新签收。

交付 ZIP（骨架、贴图、构建报告和签收）、深浅底 MP4、自包含 HTML。核 ZIP 完整与所有启用贴图齐备，解码视频为 240 帧／8 秒。ZIP 可能也保留禁用层的图片文件；绘制以骨架 slots 为准。透明网页视频仅按需调用 `export_transparent_web(..., user_signoff_path=...)`。

## 文件夹批量处理与真 4K 统一导出

支持直接传入包含 20–30 张立绘的本地文件夹路径，默认输入保留 `4096px` 4K 原图清晰度（`portrait_task.py prepare` 与 `batch_portrait.py init` 均默认 `--max-long-edge 4096`），`build_rig` 与 `split_local_body.py` 同步按 4K 分辨率等比缩放动作幅度和垫边宽度。

```bash
# 1. 扫描文件夹、并行本机去底、生成去底总览图并创建 Studio 任务（重扫时自动跳过已锁定角色）
.venv/bin/python3 scripts/batch_portrait.py init --input-dir <立绘文件夹> --workers 3

# 2. 各角色分层灌入 Studio（带 --wait-if-busy 排队防 409 busy）
python3 scripts/portrait_task.py apply-canvas --job <job_id> --parts-dir <parts目录> --plan <plan.json> --cleanup --wait-if-busy

# 3. 仅拉取用户在批量网页标记「打回待修」的角色与修改备注（绝不含已锁定角色）
python3 scripts/batch_portrait.py fix-list --batch <batch_id>

# 4. 统一规格批量导出全部已锁定角色（默认官网同款 60帧双档：PC+手机预乘 H.264 + poster.webp + media.json）
python3 scripts/batch_portrait.py export --batch <batch_id> --output-dir <输出目录>
```

打开 `http://127.0.0.1:8765/?batch=<batch_id>` 即可进入批量验收台：左侧按红/黄/绿体检灯浏览全部角色，满意点「确认锁定」（服务端返回 `409 JOB_LOCKED` 拒绝任何覆盖修改，重启 Studio 后自动恢复锁定状态），有问题填备注点「打回待修」，全部过目后在顶部面板调整统一导出尺寸（4K 竖/横/长边 4096、2K、1080p、自定义、对齐方式）与格式（默认勾选 `官网同款 60帧双档`，可多选 `VP9 透明 WebM`、`Spine 4.2 ZIP`、`深浅底 MP4`、`单文件 HTML`）并一键批量导出。

## 连接诊断与返修编排（历史遗留／当前主链停用）

`canvas_repair_workflow.py`、`connection_diagnostics.py` 冻结保留；`canvas_handoff.py workflow`、SAM 分割、旧 apply-analysis/逐模块审计路径仅供历史维护，不作为当前 Canvas 预览与签收的前置条件。不要为旧机器 pending 重启多轮审查。

历史合同见[完整模块合同](contracts/complete-modules.md)，仅按实际维护需要读取；它不覆盖当前[四步 SOP](../.agents/skills/portrait-spine-idle/SKILL.md)。现状和旧证据见[验收状态](acceptance-summary.md)，缺陷处理见[局部返修](../.agents/skills/portrait-spine-idle/references/rerun-and-troubleshooting.md)。
