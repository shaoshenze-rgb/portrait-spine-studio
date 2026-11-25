# Portrait Spine Studio 使用与验收指南

当前入口是“Cindy 对话看图 → 项目脚本交接完整模块 V2 → 工具页绑定和动画 → 分项验收”。应用不直接调用模型，来源为 conversation_handoff；无需安装连接插件。

## 启动和发起

需要 Python 3.9+、Pillow、numpy；前端无 Node 构建、无公网 CDN。视频导出使用已有 ffmpeg，不自动安装依赖。

    python3 bin/start-studio.py --port 8765 --no-open

浏览器访问 http://127.0.0.1:8765/。该地址属于运行浏览器的电脑；远程使用时应打开运行服务那台电脑的页面。工作目录默认项目内 .studio-work。服务重启不会恢复旧的内存任务；旧文件证据仍保留，需要时公开 prepare 新任务、核验同图摘要后再明确重绑分析，不伪造“恢复旧任务”。

在 Cindy 打开本项目，发图并说：

> 用 $portrait-spine-idle 把这张立绘做成完整模块轻摆动画。身体和武器保形，软配件只轻微弯曲，先给我看模块和绑定。

项目 Skill 在 .agents/skills/portrait-spine-idle，随仓库走，不需要全局安装。旧会话未发现时可以直接提供该项目内路径。

## Canvas交接主链

每轮先读项目Skill的“主目标”。由当前Cindy任务调用Canvas实际拆层/补全，保存返回scene/source/node/task身份，再用read_images(original)取得可读原件；原件校验后进入工作目录，不能以截图或画布整页导出替代层素材。

现有工具不满足分层粒度时，将明确不合格的返回层再交Canvas局部细化，保留父子来源链；不默认要求用户手工准备分层，也不继续按角色描点。

    .venv/bin/python scripts/canvas_handoff.py import-layers --receipt <真实回执清单.json> --source <Canvas本次输入原件.png> --working-image <job/matted.png> --assets-dir <原件目录> --out-dir <job/canvas-import>

子层追加 --parent-report <父层import-report.json>。成功输出对齐PNG/变换/对照图，但状态仍needs_layer_review。置信不足返回needs_canvas_registration，不能用猜测坐标放行。

    .venv/bin/python scripts/canvas_handoff.py attach-completion --analysis <当前分析.json> --import-report <job/canvas-import/import-report.json> --node <返回层nodeId> --front <前层pieceId> --back <后层pieceId> --out <新分析.json>

若明确采用Canvas完整层轮廓作为候选补全域，可加 --domain-from-layer；这不等于语义批准。新分析仍走原inspect、apply-analysis、绑定、动画和五项视觉验收。这里的命令处理已取得的真实结果，不会自行联网或提交生成；真正的Canvas调用由当前Cindy Agent编排。主链尚未全部实测时，不宣称闭环完成。

## 连接诊断与返修编排

独立Canvas RGBA层使用同一 `scripts/canvas_handoff.py workflow` 入口；旧source-sliced V2路径仍使用后面的apply-analysis。这里的workflow.json是图片制作状态，不是项目开发授权或Host权限证明，也不会直接发起模型请求。当前已接通标准parts、连接诊断、返修回执、新输入复查和绑定候选；完整循环与正式入口接入的当前实测状态见验收摘要，不能把awaiting_rig_confirm当作成品。

    python3 scripts/canvas_handoff.py workflow --work-dir <任务根>/workflow --evidence-root <任务根> --action start --payload <开始输入.json>

开始输入包含 `material_manifest_path`（Canvas完整层材质清单）和 `plan_path`（模块/连接规划）。已有标准parts可用action=inspect，payload为parts_dir、plan_path。路径必须在任务根内，可用绝对路径或任务根相对路径。脚本校验原图、图层、裁片和静止合成后，一次检查所有连接，保存原图/父层/子层/合成图的同坐标证据；不改点、不改父级、不生成素材。

极淡边缘按本模块自身透明度衡量，均匀半透明材质不会仅因alpha低就被拒。锚点旁有父/子完整支持时作为观察提示；有支持间隔或弱孤立根点时进入规划复核。`parent_support_gap`不等于已确认缺材质，必须先区分错规划、真实遮挡和遗漏结构。已声明hidden_by只进入后续核验，诊断不授予例外。

用action=status读取状态。后续动作都带本次input_revision作为expected_revision：

- decide：按request_id记录replan，或在实际看过原图/图层且确认missing_material/visible_material_mismatch后选择canvas_material。keep_current仅能处理需视觉判断的提示，不能签收跳过数值/结构错误。
- record_layer_issue：记录具体模块的实际可见素材问题，避免修好连接后遗忘火焰等独立问题。
- reserve_submission：执行Canvas调用前预留一次提交；提供source_node_id和最近一分钟的原生Canvas快照诊断回执路径readiness_receipt_path，须同时确认只读握手获准、指定节点available及规范library引用。回执仅为观察记录，当前任务原生读取映射和真实授权仍由Host决定。
- record_submission / record_result：保存实际任务ID与回执文件；结果未知进入blocked_result，不重置次数。输入已变时结果只保留为stale_result_not_applied，不应用到新版本。
- apply_input：提交修复后的material_manifest_path、plan_path及改动理由，生成新parts并重新检查；保留全部原必需模块，旧绑定和观察失效。未改材质上的未解决视觉问题会继续保留。
- review_layers / bind：全部连接问题处理完且逐模块观察与当前版本一致后，才建立绑定候选并停在awaiting_rig_confirm。不能由命令exit=0或图片已经生成推断角色通过。

默认每个材质返修请求只预留一次提交；脚本不自动追加重试授权。当前任务的单次火焰重做授权与作品库读取就绪分开：未读通不得提交，未知结果不得重复计费。

## 从图片到绑定

    python3 scripts/portrait_task.py prepare --image <透明立绘路径> --preset agile_full

prepare 校验已有透明输入并预处理，停在 awaiting_analysis。analysis_request 返回本次 job_id、image_path、input_sha256、image_size、required_contract_version=2 和 expected_result_path。已有透明底只是透传；带背景图需先取得已授权的真实去底结果并保留来源。

Agent 必须实际打开本次 image_path，按[完整模块合同](contracts/complete-modules.md)准备 V2 JSON，额外带同一个 job_id。所有人体、器具、软配件和有意效果组的可见区域、运动类型、父级/锚点、接触及遮挡均需明确。不能把整个角色设成一个大主体域，再靠颜色/近邻猜肢体。

精确边缘优先走独立脚本（使用已具备 SAM 依赖的项目 .venv，不进入 src/ 服务）：

    python3 scripts/segment_modules_sam.py emit-grid --image <本次image_path> --out <任务目录>/coord-grid.jpg --step 50
    python3 scripts/segment_modules_sam.py segment --image <本次image_path> --spec <sam_prompts.json> --out-dir <任务目录>/sam

成功时产出 ownership-color-overlay.jpg、module-crops-sheet.jpg、带 mask_png_base64/mask_sha256 的 analysis.json。失败时 analysis.json 隔离，不能恢复旧成功文件冒充本次成功。V2 自动从实际 mask 推导关节/软链，应用前仍须核对落点及完整物件边界。preprocess 会把 alpha==0 的脏 RGB 清零。piece 可用精确掩码代替手工多边形。

    python3 scripts/portrait_task.py apply-analysis --job <job_id> --analysis <完整模块JSON>
    python3 scripts/portrait_task.py status --job <job_id>

合法提交后生成分件并停在 awaiting_rig_confirm。绑定页展示 rigid/soft/static、多个可见piece的共同归属、接触与遮挡。V2不使用旧的任意骨级混权编辑；需要修改时，在当前对话更新模块规划并对同图重新 apply-analysis。直接旧 rebind 会明确要求重规划。

未决、低置信、未归属边缘、冲突或不支持约束不能默退几何。输入自带白边/残留需作为 input_issues 记录，当前明确停在 INPUT_REVIEW_REQUIRED，不删除alpha或把残留伪装成有意粒子。

apply-analysis自动生成当前原图/逐模块孤立图。需要单独重建对照时运行：

    python3 scripts/inspect_module_plan.py --image <本次image_path> --analysis <完整模块JSON> --out-dir <任务目录>/review-modules

每个模块的多个piece聚合在同一图，与相同位置的原图对照，默认unreviewed。先核完整硬物/装饰不跨driver，再按自然关节划边；不能让颈点横切硬领圈，或把整把剑的残边挂到衣服。生成后出现新裂线先回规划，不靠填色补宽掩盖。

局部点选可用页面 refine-mask 或脚本 refine-one；仅重算目标的 SAM 提示，最终归属仍整体重新验证。成功结果经原 apply-analysis 流程绑定，失败标记或缺成功报告时禁止应用。输入/掩码修改后重新生成叠加图和模块观察，不能沿用旧通过。

## 动画与质量状态

隐藏部分不是必须补齐才生成。若刚体身体关节的父侧被围巾等遮住，可在现有 connection 上用 hidden_by 精确引用既有 occlusions；普通距离容差、子关节点、抓握和脚锚要求不变。绑定阶段只确认明确的不透明连通遮挡证据，动画后还必须由 hiddenJointCoverage 检查完整输出周期（每帧时刻＋循环终点）。指定前层始终有效遮住未补区域就无需补全；移开后实际露底或关节漂移仍失败。详细引用限制见模块合同。

确认绑定后执行；当前任务已明确授权 Agent 批量验收时，可以由 Agent 实际检查绑定后继续：

    python3 scripts/portrait_task.py generate --job <job_id> --module-review <实际模块图审记录.json>

模块图审记录与现有绑定确认一次提交，不增加第二次用户审批。记录格式为source（current_cindy_conversation或user_ui）、reviewer、original_and_modules_viewed=true、binding（观察前status返回的module_review_binding）、modules（以每个模块ID为键的status/notes对象）。每个模块必须有实际理由并为pass；缺模块、待审/失败、原图/分析/对照文件变化均不能生成。UI可整表查看后每行填写“模块ID: 实际观察依据”，再使用原有确认按钮；已授权Agent可通过CLI提交同样记录，不要求用户逐模块点击。数值覆盖通过不替代这份语义观察，也不是Host看图认证。

动作使用统一模块类别和几何左右规则，含左右反向错峰、头身反相与加快的呼吸节奏，具体参数以 module_rig.py 为准。硬模块只统一平移/旋转；软模块只受限轻弯；完整模块可有多个绘制块但共同运动。脚锚不通过挪关节、拉长硬模块或冻结身体凑通过。复杂双端约束、不可达动作或无法可靠补全的区域须待审。

health-report.json 的 quality 是唯一分项结论：

| 项目 | 含义 |
| --- | --- |
| inputOwnership | 当前输入与规划/掩码绑定、可见归属、原纹理与声明覆盖；不等于已证明语义完整 |
| bindingMotion | 实际最终保形、软形变、影响归属、可见/相对运动、接触锚点 |
| machineQC | 材质稳定、遮挡声明完整性、实际隐藏补全覆盖、负空间提示和接触扫掠 |
| visual | 实际对照原图/分件图及原速循环的明确观察；默认pending |
| overall | 所有适用项目均通过才pass，exportAllowed才为true |

status可为pass、fail、pending、not_applicable；未覆盖项目不能作为pass。negativeSpace/no_holes 和软体数值应变以 advisory 记录原始 measuredStatus；真实翻面、保形、已登记后层外露等仍阻断。真实裂口和原图负空间不得混为一谈，提示项不免除视觉观察。

机器全部通过时，任务进入 awaiting_visual_review；存在待审项目为 needs_review；明确失败仍failed。两种未通过状态均可查看已生成的诊断动画，导出保持禁用。诊断播放/截图是检查手段，不等于逐帧完整目视质量通过。

## 记录同版本视觉观察与导出

页面提供五项观察：完整语义与边缘、硬体保形、软模块轻弯、接触/遮挡/补全、原速循环观感。通过前必须实际看原图和分件图，并观察完整循环；每项要有具体依据。特别检查前层衣物移开时的胯部、腿根、肩/颈等后层完整性，按各区域自己的最大暴露时刻取证，不用呼吸峰值一帧代替。检查步骤见[遮挡露底检查](../.agents/skills/portrait-spine-idle/references/rerun-and-troubleshooting.md#遮挡移开后的完整性)。无软模块等真正不适用项才可not_applicable。

从对话提交时，先在观察前保存目标版本：

    python3 scripts/portrait_task.py status --job <job_id> > <review-target.json>

观察完成后使用五项状态与说明的 observations JSON、总体说明文本，显式提交：

    python3 scripts/portrait_task.py review --job <job_id> --decision pass --reviewer <复核者> --notes-file <观察说明.txt> --binding-file <review-target.json> --observations-file <五项观察.json> --full-loop-observed --original-and-modules-viewed

需要修改用decision=fail，尚未完成用pending。不能把对话中的未经核对结论、计时器运行或少量截图写成实际观察通过。该证据是明确复核声明，不是Host或模型身份认证。

review_binding来自实际输入PNG、完整分析JSON、骨架JSON、纹理和构建报告的文件摘要。任一变化都会使旧确认失效；不对health-report自引用。机器失败/待审不能被视觉确认覆盖。只有ready且quality.overall.exportAllowed为true才可通过页面导出；底层exporter也复核文件与分项状态，旧CLI或伪造单个布尔值不能绕过。

遮挡证据不例外：同一任务内被构建报告引用的可见/隐藏/目标/剩余掩码也纳入artifact摘要，导出前重新读取并校验合同一致性。旧审核后换文件、改内容或搬路径都会使批准失效；不接受跨任务绝对路径、父目录跳转或符号链接逃逸。

原图、分析文件和验收报告也须在同一任务内。诊断脚本若引用了另一任务的原图，只能作为诊断证据，不能直接借用旧批准导出；通过正常prepare建立任务内原件和当前分析后再复核。原图/分析即使内容相同，换到另一文件位置仍会使旧绑定失效；不要仅替换health中的paths。

rig/images等生成目录必须使用实际文件，不用文件或目录符号链接替代；骨架中的贴图引用也不得指向rig外。即使外链最初指向同字节PNG，也不能获得正式绑定或沿用旧批准。

## 局部修改与批量

V2的全局gain/统一预设可以在现有规则范围内重算，随后重新QC和视觉复核；当前最大gain为1.5。局部归属、接触、静止或补全变更先回对话重规划，不按旧关键词自动冻结手、放大补全深度或混权。

批量使用同版代码/规则，每图保存原件摘要、实际V2规划、模块图、骨架、机器QC与视觉分项。先验证清晰代表例，再扩到复杂输入；不能把31份旧V1全部拒绝写成全库回归成功。run_e2e.py是已有规格的开发诊断CLI，不负责对话识别和绑定确认；即使旧机器门禁通过，没有V2同版本视觉批准也不会导出合格包，失败可保留诊断HTML。

## 当前验证与历史边界

2026-11-24 起现役结果正本为[验收状态](acceptance-summary.md)。已核 31 张原件可复用、不必再去底；清单 `.studio-work/module-rules-20261123/inventory31.json`。原确认快照 plan.md 不修改。

代码已支持精确掩码合同与独立 SAM 脚本；合成夹具正向不能当成真实立绘质量通过。三张真实代表尚无最终合格交付。《逐星暗影》ISSUE-08 的胯部/腿根问题已进入 ISSUE-09 重要结构修复；当前 final-r12 的重要结构修复及代码/记录已独立审核通过，仍有少量软后层覆盖pending，不能引用旧机器通过作为整体成品批准。其他两图及31张同版全库回归尚未完成。不要把「只支持多边形、精确 mask 未实现」当成现役能力描述，也不要把代码审核通过写成视觉通过。

Carino现有Mivo输入的四周有alpha1–4残留，已作为实际输入坏例停止，0分件/0骨架；候选模块未冒充被接受。证据为同目录 carino-input-negative-report.json。不创建残留兜底模块、不删alpha。

以前的52项基线、15张对话交接批次、旧几何批次和未采用的连接插件都保留为历史证据，不代表新V2已经验收。plugins/cindy-vision-bridge不属于当前使用或分发必需内容，不安装、不删除。

通过验收后除 Spine 4.2 zip 与预览视频外，可导出自包含单文件 HTML 预览（内联脚本与 data:image/png;base64 贴图，零外部请求，适配 320px 手机与桌面）。工具页「导出单文件 HTML 预览」或 export kind=standalone_html。

验证命令（无需安装新依赖）：

    TMPDIR="$PWD/.studio-work" python3 -m unittest tests.studio.test_complete_module_flow tests.studio.test_conversation_handoff -v
    TMPDIR="$PWD/.studio-work" python3 -m unittest discover -s tests -v
    node tests/player/test_spine42_runtime.mjs

运行前项目 .studio-work 应存在。Skill局部重跑规则见[故障参考](../.agents/skills/portrait-spine-idle/references/rerun-and-troubleshooting.md)。
