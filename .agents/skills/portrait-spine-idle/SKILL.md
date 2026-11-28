---
name: portrait-spine-idle
description: 触发词「spine动画」（用户说“spine动画”/“Spine动画”，或在任意对话发单张立绘、发多张图或给一个本地文件夹路径说“批量做 Spine 动画 / 做个待机动画 / 把立绘做成动态待机 / 让立绘动起来 / 导出 4K 透明网页视频”时都会自动触发）。支持单张与 20–30 张文件夹批量处理：默认品质优先——优先保留 4K/8K 原画 PSD 独立无损分层，对扁平图、合并主体层及复杂遮挡默认优先调用 Mivo Canvas（拆层与 AI 补全），仅在 Canvas 不可用、报错或对齐失败时自动降级本机兜底；经 100%/150% 压力体检后直出 Studio 实时预览与批量锁定验收台，默认统一导出官网同款 60 帧双档透明网页视频。不用于普通静态修图或与 Spine 动画无关的问答。
---

# Portrait Spine Idle

先出 8 秒实时可调预览，只修动起来真正穿帮的地方。每轮恢复先读本目标。
身体、武器保形，头发、衣摆、飘带轻摆；全循环始终被遮住的内容不用强补。
默认**品质优先（Mivo Canvas 拆层与补全优先 + 原画无损 PSD 直取 + 本机自动兜底）**：扁平图、无分层立绘及 PSD 中的合并主体层默认优先调用 Mivo Canvas（`xd-mivo-canvas` 的 `canvas_split_layers` / `canvas_edit_image` / `canvas_remove_background`）获取更高质量的拆层与遮挡补全（单图一次顶层拆层，不递归）；画师 PSD 已独立无损画好的 4K/8K 附件/特效层直接无损提取；**仅当 Mivo Canvas 不可用、面板未开、缺 Key、报错或对齐失败时，才自动降级走本机兜底**。
四项基础检查加 100%/150% 压力体检后由用户看图签收；代码测试不等于角色验收。
当前事实见[验收状态](../../../docs/acceptance-summary.md)，具体命令见[使用指南](../../../docs/usage.md)。

## 0. 零残次品基础规范（永久生效 · 出品必可用）

无论单张还是 20–30 张批量任务，**以下 4 条为不可突破的基础底线，宁慢勿烂，严禁产出不可用的残次预览**：

1. **原画级透明底优先（PSD 优先 + 禁留封闭死灰洞）**：
   - 不透明底立绘只要同目录存在 `.psd` 源文件，**必须优先从 `.psd` 隐藏背景/渐变填充层直接提取原画级无损透明底**（绝不允许只对报错图用 PSD、其余不透明图偷懒跑普通抠图）；
   - 无 `.psd` 走 `local_matte.py` 时，必须清除封闭洞内纯色底（如漩涡/臂弯内的 `#808080` 死灰块），并逐张目检 `matte-overview.jpg` / 去底对照图确认**零死灰洞、零缺块、零截断火焰/特效**后，才允许进入分层。
2. **Mivo Canvas 默认优先拆补 + PSD 4K 无损层直取 + 本机兜底（禁止粗多边形盲切）**：
   - **无多层原画的扁平图（含单层 PNG/JPG 与扁平 PSD）**：默认**必须优先调用 Mivo Canvas**（`ghost_id="xd-mivo-canvas", namespace="xd"`：`canvas_selection` 验面板就绪 → 经 `ghost_call.attachments` 送入工作目录 `matted.png` 调一次顶层 `canvas_split_layers` 做 AI 拆层与遮挡补全 → `read_images(original)` 读回层原件 → `scripts/canvas_handoff.py import-layers` 真实回执与 SIFT 对齐入库；若关键大面积被挡武器/披风/手臂仍缺真实纹理，按需对目标节点补一次 `canvas_edit_image` 局部遮挡补全）；
   - **多层原画 `.psd`（≥3 个可见非背景结构层）**：画师已无损画好的 4K/8K 独立结构层（前后景特效、披风后摆、发束、独立手臂/武器）优先用 `scripts/psd_split_parts.py inspect` + `split` 无损提取（保留画师遮挡下原绘像素，不降清到 2K）；**若 PSD 把人物「头+躯干+手臂+腿脚」合画在单个合并身体层，先剥离原画附件层，再优先把该合并身体层送 Mivo Canvas `canvas_split_layers` 做 AI 拆层与遮挡补全**（若只需轻量切头/足且无复杂遮挡，也可直接无损补拆）；
   - **Canvas 拆层入库硬门槛（拆乱即拒，绝不带病采用）**：走 Mivo Canvas 拆回的图层**绝不是无条件直接采用**，必须先过 `scripts/canvas_handoff.py import-layers` 的自动质检门（`import-report.json["quality_gate"]`）+ 人工目检 `canvas-composite-comparison.jpg` 与各层 `*-comparison.jpg`：
     - **机判拒用红线（基于实测 2K/4K 平衡点，不误杀好拆层也不放过拆乱）**：
       - **硬拒（`verdict == "reject_fallback_local"`）**：有效主体层（面积 `≥ 0.5%`）`< 2`；SIFT 失败率 `> 50%`（或 `> 35%` 且前景漏件 `> 6%`）；总漏件 `> 12%` 或单块连通缺件 `> 3.5%`（防整把武器/头/手被吞）；外观失真（`min_rgb_mae > 18` 或高差异 `> 10%` 或单块连通改绘失真 `> 3.0%`，防 AI 变脸/改甲）；重影克隆层（两层前景 `IoU >= 0.72` 且重叠区颜色雷同 `MAE <= 22 / 相似 >= 65%`，或 `IoU >= 0.88`；正常不同色前后遮挡补全如暗色披风在亮甲后不误杀）；拆成 `> 24` 层或 `≥ 5` 个且超 `60%` 为 `< 0.2%` 碎渣；
       - **可清理软告警（`warnings`，不掀翻整套好拆层）**：仅个别虚化光晕 SIFT 未对齐但主体覆盖 `≥ 94%`、或 2–4 个可丢弃的小火星碎点，直接清理或本机补光效后正常进入 Studio 验收；
     - **目检拒用红线**：脸部/五官变形、盔甲武器被 AI 改绘糊件、边缘脏边溢色、主体和部件错切粘连；
     - **拒用后动作（严禁整图拆漏就一刀切放弃 Canvas）**：整图 `canvas_split_layers` 因漏件/失真被拒收时，**必须先改用 `canvas_edit_image` 定点补遮挡底图**（保留 95% 原画 4K 像素，只让 Canvas 补全被披风/裙布/长辫/道具遮挡的后层底图；**`canvas_edit_image` 生成模型天生返回不透明 RGB（常带纯色或伪棋盘格底）与轻微全局色偏，严禁因此直接拒收整张补图**——必须用 `scripts/canvas_handoff.py apply-edit-patch` 对齐到 4K 原图、自动剥离外围棋盘格/纯色底（半透明光效用 `--alpha-mode inpaint_alpha`，或先对节点调一次 `canvas_remove_background`）并做邻域色调对齐后，**仅把补底像素贴入前景遮挡区内，保持可见原画 100% 像素级无损**）；只有整图拆层与 `canvas_edit_image` 局部补底均报错或被质检拒收（且在 `detail` 中如实记录两次回执）时，才允许降级走本机/PSD 兜底；即便入库放行，后续也**一视同仁**过全套 Studio 硬门（含新增的 `canvas_original_rgb_drift` / `duplicate_layer_silhouette` 红灯拦截、四项基础检查、100%/150% 露底压测与满幅动感验收），不达标绝不亮绿灯；
   - **代码硬门禁锁死「禁止无回执绕过 Canvas，且画布只读库未授权绝不算降级理由」**：`src/pipeline/canvas_layer_cleanup.py` 与 `src/studio/server.py` 已在代码层强制核验拆层来源——凡非 Mivo Canvas（`split-layers`/`edit-image`/`remove-background`）且非 PSD 画师独立无损层（`local_derivation.method == "psd_artist_layer_split"`）的本机兜底，**合法 `reason` 仅限 5 项：`canvas_panel_closed` / `canvas_panel_not_ready` / `canvas_key_required` / `canvas_tool_error` / `canvas_gate_rejected`**（附 ≥8 字真实回执 `detail`）。**`canvas_library_unmounted`、`canvas_read_unavailable`、`canvas_unauthorized` 已被代码硬禁（直接触发 `unjustified_canvas_bypass` 打红灯）**——画布开着却读不到原件只是缺用户授权，**必须立刻停下并给用户一键授权/继续按钮，严禁静默降级跳过 Canvas**；
   - **严禁跨会话自动派工跑 Canvas 拆补任务**：Cindy 宿主对跨窗口 `send_to_session` 自动唤起的回合不会挂载 Canvas 只读图库；凡需调用 Mivo Canvas 拆层或补底的任务，必须由用户在目标执行窗口亲自发消息触发授权，不得由大脑会话跨窗口自动代发。仅当确认 `PANEL_CLOSED`、`PANEL_NOT_READY`、`KEY_REQUIRED`、`MIVO_AUTH_INVALID`、超时报错，或拆层与 `canvas_edit_image` 局部补底均未通过入库硬门槛时，才允许降级本机兜底并如实写明 `local-direct-split` 来源（绝不伪造 Canvas 回执）；
   - **高良率精简大模块原则**：官网 8 秒微动待机默认按 **6–9 个清晰大模块** 组织（躯干主体、头/发、完整手臂+武器、独立披风/飘带/悬浮特效），不硬拆 12–14 块互相交叉的碎件；走本机兜底遇到无底图的扁平死交叉遮挡区时，并回主体随呼吸整体微动，绝不硬切死遮挡制造假裂缝；
   - 无论哪种拆件路径，**漏认像素占比 `unclaimed_ratio` 必须 `< 1.5%`**（绝不允许留数万至数百万像素给自动补缺把刀刃、衣摆、火焰碎渣胡乱粘到头/手臂等错位部件上）；半透明火焰/光效/漩涡必须整组完整归属，不得切碎；后层垫边不得在摆动空隙中露出纯色涂抹糊块。
3. **单张绿灯闸门 + 3 轮阶梯止损熔断（不降门槛、不堵整批）**：
   - 每完成 1 张，**必须亲自目检 `all-layers-sheet.jpg`、`keyframes`、`seam-worst-crops.jpg`、`stress-150-frame.jpg` 及实际循环播放**；
   - **质量门槛绝不放松**：必须同时满足 **四项基础检查全过 + 100%/150% 露底压测绿灯（`exposure_stress.status == "pass"`）+ 自动补缺无大块错粘（`recovered_pixel_count <= 12000` 且 `recovered_region_count <= 12`）+ 无 `canvas_original_rgb_drift`/`duplicate_layer_silhouette` + 肉眼零明显锯齿/死灰块/飘碎渣/色块糊边**，才算“交付可用”；
   - **3 轮阶梯止损法（严禁在同一缝上无效磨十几轮）**：
     - **第 1 轮（先定性）**：先看 `seam-worst-crops.jpg` 与实渲染错帧，明确区分「缺垫边」还是「骨架/软布中心线折角撕网格」；骨架问题改骨架（软布 `centerline` 保持顺弧、相邻段折角 `≤ 45°`，超限会触发 `module_rig.py` 自动平滑），绝不对着自撕裂网格盲修垫边；
     - **第 2 轮（换解法降维）**：大弧度/复杂披风若用 `soft_bend` 易自折，直接改挂**单骨 `rigid sway`** 于真实连接点（自然飘动且 0 网格自裂）；纯接触露底则补深垫边；
     - **第 3 轮（单关节局部收口）**：结构与垫边均干净、唯独某单一复杂交叉件在 150% 压测仍略超线时，**仅针对该单一问题关节**微调 `motion_scale`（如 `0.80–0.85`）或将死交叉碎件并回主体（其余健康部位仍保持 `1.0–1.1` 满幅动感）；
     - **超 3 轮仍未过 → 熔断挂起，绝不拖垮整批**：不得在 1 张硬骨头上卡死 1–2 小时把整批堵住；将该张明确标为「专项待定」并留存错帧裁剪与根因记录，**立即继续完成同批其余角色至全绿**，整批交样时把这 1 张单独拎出交用户/大脑拍板（专项深修 / 简化部件 / 美术补底图）。
4. **批量「3 张打样立标杆 → 3–5 张小批精做」节奏**：
   - 超过 5 张的批量任务，**必须先精做出前 3 张全绿灯样板交给用户过目确认品质达标**，再按每批 3–5 张小批精做稳步推进，绝不允许跳过样板确认直接通宵盲跑全批。
5. **动态幅度底线（拒绝“伪静态”交付）**：
   - 严禁为了保险或偷懒压测，把健康部件的 `motion_scale` 普遍压低到 `0.55–0.79` 导致成品在手机/网页上看几乎和静态立绘无异；
   - 默认所有健康可动部件保持 **`motion_scale = 1.0`**（独立悬浮道具 `prop`/`pet`、独立特效 `effect_group`、无遮挡衣角/发尾/飘带可按观感设 `1.0–1.25`），仅在 150% 压测确认**某单一关节真实露底且补垫边后仍超阈值**时，才允许仅针对该单个关节微调幅度；
   - PSD 原画已分层的前/后景特效（如前后旋涡、独立光环、火焰）、悬浮道具（如悬浮魔典/法球）、独立衣衬/飘带与无遮挡手臂/护手，**必须保留为独立可动模块（前/后景特效拆成两个独立 `effect_group` 自动反向视差漂浮）**，禁止为图省事全并回静态主体导致全图只有 2–3 根骨头微动。

6. **下盘稳固铁律（站姿角色严禁套“全身浮空飘动”）**：
   - **先判落地姿态**：只要立绘双脚或单脚着地站立（如绝大多数英雄 `02/03/05/06/07/08`），**必须声明 `pose_type: "grounded"`**，严禁为省事不拆脚而写成 `airborne` 让整个人在画面里左右乱漂、前后歪斜、双脚离地；
   - **站姿角色的两种标准稳固绑骨（脚底 0 漂移）**：
     - **方案 A（腿脚清晰可拆时优先，如 `02/03/06`）**：`pose_type: "grounded"` + `thigh -> shin` + 静态 `foot_l / foot_r` + `foot_anchor` 2-bone IK，膝盖随呼吸微屈，脚底完全钉死在地；
     - **方案 B（扁平图或双腿紧贴难拆 4 关节时首选，如 `05`）**：`pose_type: "grounded"` + **腰胯一刀切**：下半身（胯+双腿+双脚）设为 `lower_body`（`parent: null, motion_type: "static", motion: "static", static_reason: "站姿下盘贴地固定"`）纹丝不动钉死在地，上半身 `torso`（`parent: "lower_body", motion: "breath"`, 支点在腰部）带动胸肩呼吸与头/手/发/尾/衣摆灵动摇曳（仅 1 条腰部接缝，下盘极稳且高良率）；
   - **真浮空角色（如跳跃半空 `01` 或双脚明确离地悬空的法师）**：才允许 `pose_type: "airborne"`；躯干已由代码收紧为稳重悬浮（`AIRBORNE_BREATH_TX=(3.2,1.4)`、`TY=(10.5,2.8)`、整身倾角 ≤ 1.6°），动态张力由披风、发丝、飘带、手臂、悬浮道具与前后景特效承担，绝不出现“整个人像船一样大晃”。

7. **头部/头饰/耳部完整性与极帧重影防呆铁律（禁切半耳、禁双影残留）**：
   - 凡角色带猫耳、兽耳、角、头冠、发箍、耳机等头部刚性附着物，**整副双耳 + 头饰 + 完整头部轮廓必须 100% 归属 `head`（或挂在 `head` 下的 `hair` 子层）**，严禁把半只耳朵、耳壳外沿或头顶发束切到 `torso` 上；
   - `torso`（及其他躯干底图）在颈部/衣领以上**禁止残留任何耳廓/头顶像素（含 `padding_px` 垫边外溢）**，避免头部转动时从背后露出“双耳/重影”；
   - 每张绿灯出图合入验收页前，**必须抽查 1 帧头部与四肢最大转角帧的局部放大图**，肉眼确认无双耳、无重影、无错位切片后再交付。

8. **高良率零碎边拆补铁律（吃净 PSD 原层 + Canvas 小缺口原画底补 + 降级禁硬切死贴头颈）**：
   - **多层 PSD 先吃净画师原层**：凡 PSD 含 ≥10 个可见结构层（如独立头/发/披风/围巾/武器/前后景特效/地面），**必须先按画师原层无损提取可动部件与前后景特效**，严禁把几十层先强行揉成一整张扁平 `body` 再拿剪刀反切；
   - **Canvas 高还原拆层小缺口直接原画垫底补齐**：若 Mivo Canvas 拆层已通过 `canvas_handoff.py` 入库质检（如对齐正常、`MAE <= 10`，仅在腰/前臂/衣角漏掉 `< 6%` 局部区域），**直接用 4K 原图对应像素补齐底层躯干缺口**，保留 Canvas 拆出的干净头/手/衣摆层，禁止因局部小缺口把整套合格 Canvas 拆层丢弃重回本机硬切；
   - **降级本机兜底严禁硬切“死贴头颈/领甲/法环”**：仅当 Canvas 确因报错或漏件超限未过质量门、且已按 `b4a17dd` 在 `local_derivation.canvas_fallback` 如实写入合法 `reason` 与真实 `detail` 后才允许降级本机；降级后若合绘身体里头部与高领甲、机甲座、背后法环或长发死贴无底图，**禁止用直线多边形硬切头颈制造碎片与切口**——在 `plan.json` 显式声明 `"plan_mode": "unified_upper_body"`，改为保留「站姿静态下盘（0 漂移）+ 完整上半身（头肩一体沉稳呼吸）+ 全部独立 PSD 特效/围巾/披风/武器/光环满幅摆动（`motion_scale 1.0–1.25`）」，优先保证每张立绘零穿帮基础版一次过检。

## 1. 准备原图（PSD 无损透明优先，否则 Mivo Canvas 抠图优先，本机兜底）

0. **全局调用仓库定位**：无论当前对话在哪个目录，所有命令先切到本技能软链指向的真实仓库根目录（`~/Documents/claude_code_ssz/projects/portrait-spine-studio`，可由 `dirname "$(realpath <SKILL.md>)"/../../..` 解析），再执行下文命令。
1. 启动或确认本机 Studio 在线：`python3 bin/start-studio.py --port 8765 --no-open`（先查 `http://127.0.0.1:8765/api/health`；若已在跑但版本过旧不识别 `local-direct-split`，原位重启 `8765`，保留 `.studio-work/` 已有任务不删除）。
2. 直接读取用户本轮对话附件原图：
   - **已是透明 PNG**：直接进入第 3 步。
   - **带不透明背景（JPG/PNG 且无 `.psd` 可直接隐藏背景层）**：优先调用 Mivo Canvas `canvas_remove_background`（AI 去背景）；仅当 Canvas 不可用/报错/缺 Key 时自动降级走 `.venv/bin/python3 scripts/local_matte.py --input <附件原图> --out <透明原图.png> --check-out <去底对照.jpg>` 本机兜底，目检 `<去底对照.jpg>` 确认人物轮廓干净、无死灰洞、未误抠内部高光。
3. 创建 Studio 任务并拿到 `job_id` 与规范工作图 `matted.png`：

```bash
python3 scripts/portrait_task.py prepare --image <透明原图.png> --character <角色标识>
```

## 2. Mivo Canvas 优先拆层补全（PSD 无损层直取，失败自动降级本机）并出实时预览

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

## 3. 看完整循环、Canvas 局部补全与实时微调

先在 Studio 复核 `all-layers-sheet.jpg` 和自动接缝特写（原图、起始帧、局部最大变化帧），包括肩臂、头臂及自动选出的软层/本地切分交界。先消除 `local_layer_pixel_bleed` 与 `separated_soft_islands`，不靠调幅掩盖。16 个抽样时刻不能替代原速完整循环。

在 Studio 左侧舞台直接看动画，右侧实时调百分比、显隐与前后层级。100% 是角色规划的默认动作；单件与整体乘积封顶 150%，不按最近一次保存反复放大。拖滑块不发保存请求，确认后再一键烘焙并复看。

**Mivo Canvas 标准调用、0 扣费活体预检与防卡死协议（默认优先通道）**：
- **先验“画布真能调通且读得回原件”，再发收费任务（绝不盲发卡死）**：每次批量或单张调 Canvas 前，先做 0 扣费双检：① 无参调 `ghost_call({ghost_id:"xd-mivo-canvas", namespace:"xd", tool:"canvas_selection", args:{}})` 确认面板已打开且载入完成；② 若画布已有节点，用 `canvas_snapshot(libraryDiagnostics=true, mediaNodeIds=[1个节点])` 或 `read_images` 零扣费核验 `authorizedReadonly == true` 且真能拿到 `available` 原件（可用 `assess_canvas_readiness` 判定）：
  - 若 `authorizedReadonly == true` 但旧节点报 `unavailable`（`stale_node_retry_fresh`）：说明画布是通的、只是旧节点资产过期，**立即丢弃旧 nodeId、重新 `canvas_place_image` 跑新节点**，绝不在旧节点上卡死；
  - 若面板开着却返回 `authorizedReadonly == false` / `unauthorized`（`auth_blocked_stop`）：**零扣费暂停收费拆层**，立即如实汇报精确诊断，让用户选择「① 在画布所在的本机（如 MacBook Pro）Cindy 窗口亲自发消息（非手机/另一台电脑远程转发，也非跨窗口派工）即可立刻获得 `authorizedReadonly: true` 并在同一轮连续跑完」或「② 用户明示批准防卡死逃生阀（写入 `user_approved_fallback: true` + `user_approval_note`）继续推进」，绝不静默绕过、也绝不全线卡死；
- 确认通道就绪后，批量任务将工作目录内图片绝对路径传入顶层 `attachments`（单次顶层 `canvas_split_layers` / `canvas_remove_background` / 局部补全 `canvas_edit_image`，禁止二、三级递归拆子层）；遇 `REFERENCE_LIBRARY_CHANGED` 先 `canvas_selection` 确认后重试一次，**严禁跳过放图直接读画布上旧选中节点**；
- 若返回 `PANEL_CLOSED`、`PANEL_NOT_READY`、`KEY_REQUIRED`、`MIVO_AUTH_INVALID`、超时，或整图 `canvas_split_layers` 与定点 `canvas_edit_image` 补底均报错 / SIFT 对齐失败（`needs_canvas_registration`），**记录精确失败原因后才允许降级回本机 `local_matte.py` / `split_local_body.py` 兜底**，绝不伪造 Canvas 回执；**严禁将 `library_unmounted` / `unauthorized` / `unavailable` 当作降级理由跳过 Canvas**（必须按下一条立刻停止并引导用户在当前窗口授权）。
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

- 默认品质优先 + 拆乱即拒：扁平图与合并主体层优先调 Mivo Canvas（`canvas_split_layers` / `canvas_edit_image` / `canvas_remove_background`），画师 4K/8K 原画 PSD 已独立分好的层无损直取；**Canvas 拆回的每一套层都必须先过 `import-report.json["quality_gate"]` + `canvas-composite-comparison.jpg` 入库硬门，再过全套 Studio 绿灯验收门**，拆乱、变脸、漏件、重影一律拒用，重拆 1 次仍不达标或 Canvas 不可用/报错时自动降级本机兜底，绝不带病采用。
- 不把软布摆动或呼吸压到近乎静止来过压力检查；保持自然可见的默认动作，用后层桥接和深垫边修露底。
- 不做二、三级递归拆子层；绝不伪造 Canvas 回执字段，纯本地直分必须如实标记 `local-direct-split`。
- 不用静态 6 px 距离或零星半透明边缘锁死预览与已签收导出。
- 不重新启用冻结的 `canvas_repair_workflow.py` / `connection_diagnostics.py` 作为主链门禁。
- 批量返修或重扫文件夹绝不修改、覆盖或绕过任何 `locked: true`（已确认锁定）的角色。
- 不写角色名、文件名或哈希特判；代表图只是验证样本。
