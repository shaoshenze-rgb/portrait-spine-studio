---
name: portrait-spine-idle
description: 触发词「spine动画」（用户说“spine动画”/“Spine动画”，或在任意对话发单张立绘、发多张图或给一个本地文件夹路径说“批量做 Spine 动画 / 做个待机动画 / 把立绘做成动态待机 / 让立绘动起来 / 导出 4K 透明网页视频”时都会自动触发）。支持单张与 20–30 张文件夹批量处理，默认保留 4K 清晰度本机直通去底、边缘贴合分层、后层桥接补全与 100%/150% 压力体检，直出 Studio 实时预览与批量锁定验收台，默认统一导出 4K + 透明网页视频；仅复杂遮挡或糊特效时按需局部补一次 Canvas。不用于普通静态修图或与 Spine 动画无关的问答。
---

# Portrait Spine Idle

先出 8 秒实时可调预览，只修动起来真正穿帮的地方。每轮恢复先读本目标。
身体、武器保形，头发、衣摆、飘带轻摆；全循环始终被遮住的内容不用强补。
默认**本机直通优先**（新会话第 1 轮不依赖 Mivo Canvas，零停顿、零付费直出 Studio 网页）；复杂背景、大面积被挡武器或糊特效才按需补一次 Canvas，不递归拆子层。
四项基础检查加 100%/150% 压力体检后由用户看图签收；代码测试不等于角色验收。
当前事实见[验收状态](../../../docs/acceptance-summary.md)，具体命令见[使用指南](../../../docs/usage.md)。

## 0. 零残次品基础规范（永久生效 · 出品必可用）

无论单张还是 20–30 张批量任务，**以下 4 条为不可突破的基础底线，宁慢勿烂，严禁产出不可用的残次预览**：

1. **原画级透明底优先（PSD 优先 + 禁留封闭死灰洞）**：
   - 不透明底立绘只要同目录存在 `.psd` 源文件，**必须优先从 `.psd` 隐藏背景/渐变填充层直接提取原画级无损透明底**（绝不允许只对报错图用 PSD、其余不透明图偷懒跑普通抠图）；
   - 无 `.psd` 走 `local_matte.py` 时，必须清除封闭洞内纯色底（如漩涡/臂弯内的 `#808080` 死灰块），并逐张目检 `matte-overview.jpg` / 去底对照图确认**零死灰洞、零缺块、零截断火焰/特效**后，才允许进入分层。
2. **PSD 多层原画优先直拆 + 禁止粗多边形盲切与大面积漏认硬粘**：
   - 同目录存在多图层 `.psd` 源文件（≥3 个可见非背景结构层）时，**必须优先使用通用工具 `scripts/psd_split_parts.py inspect` + `split` 直接基于原画分层拆件**（自动保护跨部件高光/混色层不粘错零件、优先保留画师遮挡下原绘像素并自动补齐剩余遮挡垫边、自动 4K 去白/灰毛边），严禁每张手写一次性脚本或拍扁后用粗多边形硬切；
   - 仅无可用多图层 `.psd` 时再走 `scripts/split_local_body.py` 精密贴边拆件（复杂装备/领口/发丝/特效结合 SAM 2.1 精切），严禁在小缩略图上一口气盲写几十份粗略多边形后扔进后台队列一锅端；
   - **高良率精简大模块原则（含「PSD 剥离附件 + 孤立主体补拆头/足」混合标准）**：官网 8 秒微动待机默认按 **6–9 个清晰大模块** 拆件（躯干主体、头/发、完整手臂+武器、独立披风/飘带/悬浮特效），不把单角色硬拆成 12–14 块互相交叉的碎件反复返工；若 PSD 已将披风/漩涡/武器/护手分层、但把人物「头+躯干+腿脚」合画在单层（如「人」「组 3」），先用 `scripts/psd_split_parts.py` 剥离全部原画附件层，再**仅对该干净孤立身体层**跑一次 `split_local_body.py` 补拆绑定必需的 `head`（及站立角色所需腿/静态 `foot` 足锚），无需停下请示；对于**无原画分层底图的扁平大面积交叉遮挡区**（例如法杖穿过厚重衣摆背后、前襟与后摆大面积重叠），默认与主体合并随呼吸整体微动，只让无遮挡焦点部位（头部、悬浮书/火焰/特效、自由外垂衣角/发尾）灵动摆动，绝不硬切死遮挡制造假裂缝；
   - 无论哪种拆件路径，**漏认像素占比 `unclaimed_ratio` 必须 `< 1.5%`**（绝不允许留数万至数百万像素给自动补缺把刀刃、衣摆、火焰碎渣胡乱粘到头/手臂等错位部件上）；半透明火焰/光效/漩涡必须整组完整归属，不得切碎；后层垫边不得在摆动空隙中露出纯色涂抹糊块。
3. **单张绿灯闸门（红灯必停，不绿灯不准开下一张）**：
   - 每完成 1 张，**必须亲自目检 `all-layers-sheet.jpg`、`keyframes`、`seam-worst-crops.jpg`、`stress-150-frame.jpg` 及实际循环播放**；
   - 必须同时满足：**四项基础检查全过 + 100%/150% 露底压测绿灯（`exposure_stress.status == "pass"`）+ 自动补缺无大块错粘（`recovered_pixel_count <= 12000` 且 `recovered_region_count <= 12`）+ 肉眼零明显锯齿/死灰块/飘碎渣/色块糊边**，才算“交付可用”；
   - **红灯必停**：只要当前张亮红灯或目检穿帮，**必须当场停下来精修当前张直到绿灯**，严禁带着红灯继续盲跑下一张把残次品滚雪球。
4. **批量「3 张打样立标杆 → 3–5 张小批精做」节奏**：
   - 超过 5 张的批量任务，**必须先精做出前 3 张全绿灯样板交给用户过目确认品质达标**，再按每批 3–5 张小批精做稳步推进，绝不允许跳过样板确认直接通宵盲跑全批。
5. **动态幅度底线（拒绝“伪静态”交付）**：
   - 严禁为了保险或偷懒压测，把健康部件的 `motion_scale` 普遍压低到 `0.55–0.79` 导致成品在手机/网页上看几乎和静态立绘无异；
   - 默认所有健康可动部件保持 **`motion_scale = 1.0`**（独立悬浮道具 `prop`/`pet`、独立特效 `effect_group`、无遮挡衣角/发尾/飘带可按观感设 `1.0–1.25`），仅在 150% 压测确认**某单一关节真实露底且补垫边后仍超阈值**时，才允许仅针对该单个关节微调幅度；
   - PSD 原画已分层的前/后景特效（如前后旋涡、独立光环、火焰）、悬浮道具（如悬浮魔典/法球）、独立衣衬/飘带与无遮挡手臂/护手，**必须保留为独立可动模块（前/后景特效拆成两个独立 `effect_group` 自动反向视差漂浮）**，禁止为图省事全并回静态主体导致全图只有 2–3 根骨头微动。

## 1. 本机直通准备原图（第 1 轮直接启动，不先调 Canvas）

0. **全局调用仓库定位**：无论当前对话在哪个目录，所有命令先切到本技能软链指向的真实仓库根目录（`~/Documents/claude_code_ssz/projects/portrait-spine-studio`，可由 `dirname "$(realpath <SKILL.md>)"/../../..` 解析），再执行下文命令。
1. 启动或确认本机 Studio 在线：`python3 bin/start-studio.py --port 8765 --no-open`（先查 `http://127.0.0.1:8765/api/health`；若已在跑但版本过旧不识别 `local-direct-split`，原位重启 `8765`，保留 `.studio-work/` 已有任务不删除）。
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

## 5. 文件夹批量处理（20–30 张 · 4K · 锁定防改 · 集中返修）

当用户提供一个本地文件夹路径（或一次发多张立绘）要求批量处理时：
1. **批量初始化与并行去底**（默认保留 `4096px` 4K 分辨率，输出 `matte-overview.jpg` 总览图，重扫同一目录绝不触碰已锁定角色）：

```bash
.venv/bin/python3 scripts/batch_portrait.py init --input-dir <本地文件夹路径> --max-long-edge 4096 --workers 3
```

2. **先 3 张打样立标杆 → 再 3–5 张小批精做（严格执行第 0 节绿灯闸门）**：严禁一口气盲写全批粗多边形扔后台队列盲跑，也严禁碰 `locked: true` 角色。先完成全批原画级透明底（有 `.psd` 一律优先隐藏背景层直出）并目检 `matte-overview.jpg`；随后**先精做前 3 张到全绿灯（`unclaimed_ratio < 1.5%`、`exposure_stress=pass`、目检零死灰块/碎渣/色块糊边）**并 `batch_portrait.py sync --batch <batch_id>` 交给用户确认品质标杆，确认后再按每批 3–5 张精做推进；单张亮红灯必须当场精修过绿，不带病进入下一张。
3. **交付唯一批量验收链接**：提供 `http://127.0.0.1:8765/?batch=<batch_id>` 与 `matte-overview.jpg`。用户在左侧列表按红/黄/绿灯逐张过目，满意即点「确认锁定」（后端 409 硬锁保护），不满意则点「打回待修」并写一句修改备注。
4. **集中返修打回项**：用户回对话说「修这一批里打回的」时，先运行 `python3 scripts/batch_portrait.py fix-list --batch <batch_id>`，**只针对返回的 `needs_fix` 角色按备注返修**，绝对禁止修改或解锁任何已锁定角色。
5. **统一规格一键批量导出**：默认已勾选与火炬官网现有角色 100% 对齐的 **`官网同款 60帧双档`（60fps / 8 秒 / 480 帧多核并行渲染、自动裁透明边且拼接宽控在 ≤3840 显卡硬解线内、预乘 RGB|Alpha 左右拼接 H.264 + BT.709，一次输出 `idle-high-packed.mp4` + `idle-mobile-packed.mp4` + 无损 `poster.webp` + `media.json`）**；也可按需切换 4K 竖/横/长边 4096/2K/1080p/自定义尺寸或加勾 VP9 WebM / Spine 4.2 ZIP / MP4 / HTML，点击「一键批量导出已锁定角色」或跑 `python3 scripts/batch_portrait.py export --batch <batch_id>` 统一落盘并产出 `batch-export-manifest.json`。

## 红线

- 默认第 1 轮走本机直通（`local_matte.py` + `split_local_body.py` + Studio），不因 Mivo Canvas 冷启动或未打开而阻断首轮交付。
- 不把软布摆动或呼吸压到近乎静止来过压力检查；保持自然可见的默认动作，用后层桥接和深垫边修露底。
- 不做二、三级递归拆子层；绝不伪造 Canvas 回执字段，纯本地直分必须如实标记 `local-direct-split`。
- 不用静态 6 px 距离或零星半透明边缘锁死预览与已签收导出。
- 不重新启用冻结的 `canvas_repair_workflow.py` / `connection_diagnostics.py` 作为主链门禁。
- 批量返修或重扫文件夹绝不修改、覆盖或绕过任何 `locked: true`（已确认锁定）的角色。
- 不写角色名、文件名或哈希特判；代表图只是验证样本。
