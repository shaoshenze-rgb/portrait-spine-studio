# 当前验收状态

更新：2026-11-26。当前主链以[项目 Skill](../.agents/skills/portrait-spine-idle/SKILL.md)和[使用指南](usage.md)为准：启动本机 Studio → 一次顶层粗拆 → 必要时 `scripts/split_local_body.py` 边缘贴合与后层桥接补全 → 净化与 100%/150% 全身压力体检 → 实时网页工作台微调并人工签收导出。下方历史折叠块仅作追溯，不是现行规则。

| 代表样本 | 最新验收事实 | 正式交付 |
| --- | --- | --- |
| 逐星暗影（用户归类为站立外观） | 用户对 starshadow-r2 明确签收“逐星没问题了”；肩甲层序、左右手/头部局部幅度与垫边已验证 | [starshadow-final](../.studio-work/deliverables-20261125/starshadow-final/) 四件套已导出；[签收](../.studio-work/deliverables-20261125/starshadow-final/user-signoff-r2.json) |
| 爆破新星（浮空、去糊特效、补臂型） | 用户对 blastnova-noflame 明确签收“没问题，落实到通用规则和skill了吗”；完整手臂与无残丝版通过 | [blastnova-final](../.studio-work/deliverables-20261125/blastnova-final/) 四件套已导出；ZIP 完整，深浅底各 240 帧／8.0 秒；[签收](../.studio-work/deliverables-20261125/blastnova-final/user-signoff-noflame.json) |
| 时空幻象（遮挡武器 + 站立 `grounded`） | 已恢复作品库读取并完成单次三层粗拆（独立魔法书、完整法杖、人物整层）+ 17 层后层桥接补全；自然待机摆动下 100%/150% 全身压力门禁通过，已验证新版实时交互工作台 | Studio 实时预览已就绪，待人工签收导出正式四件套 |

## 已落地的通用升级与硬边界

- **通用本地分层与后层补全（`src/pipeline/local_body_split.py` + `scripts/split_local_body.py`）**：内置 GrabCut / 色彩梯度动态规划贴边、高 `draw_order` 优先归属（前景零外溢）、被遮挡连续后层全桥接补全与 40–64 px 平滑本色深垫边（静止首帧零可见改动）。
- **防复发自动检测（`src/pipeline/canvas_layer_cleanup.py` + `src/gates/preview_diagnostics.py`）**：自动检测同源本地分层像素串层（`local_layer_pixel_bleed`）、软层大断岛（`separated_soft_islands`）、100%/150% 满滑块全身封闭与开口露底压力扫描（`exposure_stress`），自动生成全单层棋盘底总览 `all-layers-sheet.jpg` 与高风险接缝特写 `seam-worst-crops.jpg`；红线禁止压低默认布料/呼吸幅度绕过压力检查。
- **实时交互 Studio 工作台（`src/studio/web/canvas-workbench.{js,css}` + `src/player/`）**：左右分栏、默认关闭骨骼线与 IK 调试字块、中文部件名 + 真实部位缩略图 + 分组折叠、原生 JS Spine 播放器支持 0 延迟实时调节单部位幅度/整体动感/显隐/前后顺序与悬停高亮，支持工作台内一键烘焙并人工签收导出。
- **冻结与边界**：`canvas_repair_workflow.py` 与 `connection_diagnostics.py` 保持冻结；本轮无角色名/文件名/哈希特判；三张代表样本不代表全图免目检。

## 历史记录（仅供追溯，不作为现行流程）

<details>
<summary>展开 M1–M3 收口前的阶段记录、旧门禁与诊断证据</summary>

# Portrait Spine Studio 当前验收状态

## 当前主链检查点：Canvas材质回接（2026-11-25）

主目标为立绘→按需Canvas拆层/补全→自动对齐回接→骨骼与动画→完整循环关节检查→局部返修。每轮先读项目Skill主目标，角色只作回归样本。下方“历史阶段记录”中的版本、测试、运行态和下一步均为当时事实，不代表当前状态。

| 项目 | 当前核实事实 |
|---|---|
| 源码规则 | complete-modules-5 / module-qc-10；不等于8765运行进程已更新 |
| 真实Canvas返回 | 初次5层、双手细化2层、下半身细化3层已保留；新增下半身剩余混合层细化5层、躯干细化4层已完成并入库，回执已保存，不重复生成 |
| 对齐 | 顶层5层已自动对齐；子层采用其Canvas父层再组合到原图。直接对齐失败的旧报告保留，不能冒充通过；所有层仍需语义图审 |
| 真实材质回接 | 机械手15,031个隐藏像素进入materialize；未到真实split/rig/动画 |
| 当前阻塞 | 旧未批准计划的rib→torso连接距离206.7124px，被正常门禁拒绝。下一步依据Canvas返回层建立新的ownership/parent/connection，不移动旧点绕过检查 |
| 合成专项 | 原独立审核系统/venv各13项通过，仅证明静态与合成路径，不证明真实角色通过 |
| attach隔离修复 | 系统/venv各19项通过，含旧输出失败隔离、原子发布异常、失败后恢复、并发拒绝、归档失败，以及锁创建/写入失败和无法落盘的CLI报告；Orca Worker独立复审各19项通过，限定APPROVED |
| 最近全量 | full-tests-current.log为294项，293通过、1个既有V1 Erika EXPORT_APPROVAL_REQUIRED失败；后续attach隔离修复尚需专项复核，不称全量绿 |
| 审核 | 静态合成专项、attach失败隔离分别获Orca Worker限定APPROVED；真实Canvas主链尚未通过，不能套用代码批准 |
| 新结果读取 | 当前任务原件读取映射已恢复，新增9层及2张底图已核SHA并保存到项目；9/9层自动对齐成功，中位匹配误差0.19–0.36px，Lead已实际查看9张同位置对照图。尚不是语义批准 |
| 新模块规划 | 下半身腿/腰甲、躯干衣装/肩甲已细分；深色胸衣与飘带混层、头部层携带肩侧装饰等仍需结构复核。仅建立Canvas层级候选清单，不手工挪旧关节点或借旧mask过关 |
| 两处混层返修 | 头部返回3层、胸衣/后飘带返回2层；7份原件（含2底图）已核SHA、解码并复制到项目，5层自动对齐且Lead已看图。头/肩饰、胸衣/后飘带已在返回层中分开，待独立确认和新ownership/geometry；未进split/rig/动画 |
| 新归属诊断 | 两处混层获Orca Worker限定APPROVED，可进入新规划，见HEAD-SASH-APPROVAL.json。11模块/17层候选未借旧mask；细分层直接覆盖漏94,352px，合计已有父层仍漏30,148px。Canvas自动提示→现有SAM原图分割虽零漏归属/零重叠，实际图审仍错归；保留Canvas源对应区域的诊断仍因机械臂碎件失败。候选未绑定/动画，未改生产代码或放宽阈值 |
| 完整层直达标准parts（最新开发检查点） | 新增src/pipeline/canvas_layer_parts.py上游适配器，完整RGBA按层序物化标准parts，不运行ownership/SAM/nearest，不改已有Studio入口或门禁。14张真实候选层已转换，静止合成相对上一份候选maxRGBA=0、材质不改写。原画对照仍pending，火焰仍未解决；几何仅临时规划，productionIntakeReady/exportAllowed=false |
| 渲染兼容证据 | 上游parts适配器已获Orca Worker限定APPROVED，独立双环境组合各26/26。重叠/半透明材质保留、原图与setup参考区分，合成夹具经build_rig和SpineRenderer静止误差≤2；正式导出仍拒绝。不是实际角色rig或全循环通过 |
| 骨骼候选编译（最新待审） | 新增canvas_layer_rig.py：复用V2语义/拓扑/落点/typed motion及软链检查，只编译浮空候选，不假装接通站立足锚、隐藏关节、Studio或完整QC。支持域标明不是互斥source ownership，build报告单列canvas_layers_binding_candidate，正式导出显式拒绝。双环境相关99/99通过，独立代码复审pending |
| 真实绑定预检 | 首次自动软pivot/centerline根点不同源的问题已通用修正，仅作用于自动未确认几何，手动pivot保留；新候选保留旧版。第二次预检因attach_back_ribbon_left距torso材质159.1540px>6失败，未挪点/降阈值，真实rig/动画未运行 |
| 隐藏软附件增量（最新待审） | Canvas独立层候选现已保留显式occlusions/attachment.hidden_by，核验身份/层序/区域/图层SHA后，仅豁免父侧可见支持距离。原source-sliced默认不变；普通关节/抓握/足锚不套用。相关双环境各132/132，独立代码复审pending；未生成新Canvas素材或重启服务 |
| 局部逐帧证据与完整角色阻断 | 未改原连接点/父级/材质。局部诊断仅验证躯干—飘带在指定右腿+腰甲后的连接，241时刻含终点，647登记像素maxUncovered=0、maxLowAlpha=7，结果pending而非整体pass。局部探针明确不含尚不合格前臂/火焰；完整候选未删它们，仍因前臂到torso距离89.1067px>6被拒绝，未生成完整角色动画或导出 |
| 通用连接诊断/返修编排（本轮待审） | connection_diagnostics一次核验所有连接并输出原图/父/子/合成证据；极淡锚点结合实际支持距离分为提示或复核，不把距离直接等同补画。canvas_handoff workflow已接start/inspect→返修决定/提交预留/回执→新输入复查→逐模块观察→绑定候选；不调用模型、不授予权限，未知结果不重提，输入改变会作废旧观察且不允许删模块凑完整。双环境相关45/45，独立代码复审pending |
| 本轮真实工作流 | connection-repair-mainline/workflow/workflow.json已运行：唯一连接返修单是前臂fringe_anchor+parent_support_gap，queued_for_replan；火焰可见材质问题单独保留，queued_for_canvas、attempts=0。用户已允许仅火焰重做一次，但当前回执仍unauthorized，未提交生成/未消耗该次授权。未声称完整角色循环验收完成，当前新workflow入口只到绑定确认 |
| 运行与发布 | 本轮未重启8765；旧health版本仅为历史记录，当前服务版本未重新核验。未提交、推送、正式导出 |

证据根：`.studio-work/canvas-mainline-20261125/`，见CHECKPOINT.json、aligned/import-report.json、hands/aligned-parent-chain/import-report.json、lower/aligned/import-report.json、real-handoff-job及测试日志。attach修复范围独立保存在attach-fix/；原始baseline-sha保持原值，不把修复后摘要回写成起始基线。README仅有本次attach-fix起始快照，不能据此补造更早基线。

回接失败处理：attach开始即隔离旧输出，保留prev-success副本；验证后临时文件原子发布。失败记录CANVAS_ATTACH_FAILED，进行中保留CANVAS_ATTACH_PENDING；异常后不消费旧成功名称。输出不能覆盖输入，也不并发覆盖同一结果。若文件系统拒绝隔离，错误明确记录isolation_complete=false，不宣称旧文件已移走。

锁建立前发生文件系统错误时尚不能隔离旧输出：尽可能记录FAILED、analysis=null、isolation_complete=false，并立即停止；失败记录也不可写时，CLI非零退出并报告persisted=false/status_error，禁止消费旧文件。已有pending锁属于忙碌/中断状态，拒绝调用且不覆盖它的标记或输出。锁写入中断保留pending，不能当作成功结果。该增量证据在attach-fix-r2/，未重跑Canvas任务或真实角色生成。

attach复审通过回执见attach-fix-r2/APPROVAL-RECEIPT.json，绑定复审时SHA；随后这里只追加当前事实，不改变批准的代码。继续主链时真实Canvas局部细化新任务为cindy-edit-13-mwvphmgz、cindy-edit-14-mwvpjgpy；结果、引用、旧层图审发现和下一步保存在LAYER-REFINEMENT-CHECKPOINT.json。恢复授权原件读取后继续对齐、模块规划和绑定，不沿用旧角色手调计划。

图片读取最小排查：同一批头部/胸衣节点无重建，read_images已由unavailable/unknown恢复为available；canvas_snapshot诊断显示handshake authorizedReadonly=true、canonical resolved、confirmed confirmed。当前任务映射有效，真实文件7/7摘要及PNG解码通过，已留项目副本，见read-path-diagnostics.json、READ-PATH-RECOVERY.json。历史unknown无底层错误记录，根因未复现，不能表述为已修复Canvas代码。此轮没有重生成、修改插件或Spine源码。主链候选见HEAD-SASH-CURRENT-REVIEW.json。

新ownership/geometry的诊断证据在canvas-derived-plan/RESULT.json及其逐模块图：现有单一源像素归属流程会把完整Canvas机械手再次切碎、把白护甲分进火焰；不将此候选进入rig。下一步限定为审清完整Canvas材质与现役归属/分件合同的通用交接，不继续角色点位调参、收费重生成或追小碎片。所有新mask均为诊断候选，不是正式module-review pass。

## 历史阶段记录（以下不是当前主链验收）

更新：2026-11-25。用户本轮限定“只清理躯干、四肢、头部/头发、武器和关节的重要错位，不加新功能”。当前候选为 .studio-work/v2-integration-20261123/wave2/starshadow-v2-issue09/final-r12/；本轮代码/规则、纠正记录、用户限定的重要结构视觉均经 Orca Worker reviewer 独立终审 APPROVED。**用户于2026-11-25回复“验收通过”，本版重要结构与动画效果已签收；机器残余pending保留，未正式导出。**

## 本轮修正

- 重要归属：连续发束从披风/躯干回到 head；飘带钳头从腿回到对应飘带；8,433px硬裤料保留在刚体腿部，310px真实腰布归回腰布。按结构复核保留左臂及手背金指节，不凭颜色搬到胸前。
- 前后关系：腰甲在腰布前面，腰布在后补全。静止掩码互斥/零差不能证明 draw_order 正确。
- 补底：保留明确遮挡域和现有 local 补全。局部声明不再抹掉声明区外的合法自动补边；已声明深度不再被分件阶段截回默认值。未放宽保形、翻面、接触或源像素硬检查。
- 防漏：未声明的同级软前层/刚体肢段边界 pending；所有显式遮挡均登记完整隐藏目标；实际贴图alpha和实际软前层网格参与覆盖检查。规则版本 complete-modules-3 / module-qc-7。
- 可追溯：输入6笔掩码纠正均有转移PNG、计数、前后SHA和回放核对；腰甲后置恢复未漏记。层级/关节元数据另有记录。

## 已验证与限制

| 项目 | 当前事实 |
|---|---|
| 原图保真 | rest_diff maxRGBA=0；输入前景无丢失、无重复归属 |
| 动作保留 | 与final-r4对比，241时刻全部骨骼矩阵最大差0；未降频率、幅值或挪关节隐藏裂缝 |
| 固定腰甲ROI | [1180,1170,1360,1340]，241时刻alpha<128像素最大值0 |
| 按实际部件运动的材料跟踪 | 胯部/腿部原不透明材质241时刻0丢失；新增发束完全透明样点0，35个边缘抽样低于128如实记录 |
| 隐藏覆盖 | 腰布/腰甲/双腿5个登记区域全周期pass；披风后层7个原始alpha253–254像素因软后层坐标未支持仍pending |
| 静态重要结构图审 | 12个模块已按用户范围逐项记录在module-review.observations.json；小碎边/装配邻域微块保留说明，不等于所有像素语义完美 |
| 用户视频验收 | pass；原话“验收通过”，对应final-r12，视频SHA已复核；记录user-acceptance-20261125.json |
| 正式视觉和导出 | visual/overall pending、exportAllowed=false，未提交正式视觉pass、未export |
| 其他角色/全库 | 两图已启动本版输入验证，但均未通过：爆破新星手/火焰分层待可信输入；时空幻象有限分件修正后被遮挡颈部可见mask门禁拦住。31张全库not_run |

owner-following-regression.json 按每个piece实际三角形/软链映射。旧 full-loop-regression.json 的“全部随torso运输”探针保留作对照，不是同一个坐标系，不能拿其中外轮廓位置的非零值与固定ROI的0混写。原 reviewer 的4475/585完整域包含原本半透明像素，原不透明数量分别4407/306，原始分账和全部数据均保留。材质不丢失本身也不替代隐藏覆盖和看图。

## 测试与审核

当前核心两组系统/venv已通过83项，相关9组174项通过。最终代码全量discover：251项，250通过、1失败；失败为旧V1 Erika导出缺少V2验收，错误EXPORT_APPROVAL_REQUIRED。不是全量绿，不用旧no_holes原因代替当前错误。

日志在本轮证据根：local-auto-depth-tests.log、core-tests-system-rule3.log、affected-tests-rule3-final.log、full-tests-final-code.log。手工纠正回放正反例4项通过。Skill validator与文档链接检查通过。2026-11-25 Orca Worker reviewer 已对当前 final-r12 的代码/规则、纠正记录和用户限定的重要结构视觉分别签发 APPROVED；不含整体成品、正式视觉或导出批准。核对过的11文件SHA保存在 reviewed-sha-20261125.json；本次签发未套用旧SHA。

## 使用入口与运行态

项目Skill：.agents/skills/portrait-spine-idle/SKILL.md；合同：docs/contracts/complete-modules.md；操作：docs/usage.md。规则已包含重要性优先、连续轮廓判定、前后层级、完整纠正记录、按真实部件/软链取证及各局部最大暴露时刻检查。

2026-11-25两图通用性验证阶段已使用当前代码启动8765，health=module-qc-7；新任务爆破新星59757c69f140、时空幻象862355996df3均awaiting_analysis、0 split/rig、module_review=null。未复活旧内存任务；逐星暗影签收仍只绑定原final-r12诊断文件。

未commit/push/deploy，未修改官网、全局规则、平台记忆或依赖。旧产物和已有脏树保留；本轮专用diff/SHA见证据根。本轮重要错位修正版可供用户查看；正式导出仍须满足全部现役门禁，不以本轮范围签收绕过pending。


## 两图同版验证未通过（2026-11-25）

证据目录：.studio-work/generalization-20261125/。生产/src/scripts/tests/Skill与本轮起始摘要一致，未为新图写特判。原始RGBA复用、无付费生成或重新去底。

Orca reviewer独立输入复核：两图FAIL/NEEDS_INPUT。爆破新星局部SAM和一次轮廓纠正都不能可靠分开机械手与半透明火焰，需可信局部分层；已停调点。时空幻象book/staff身份及主要物件改善，按原图做有限杖杆/前襟锚点修正后，仍遇到被围巾遮住的颈部在现有可见mask连接规则下不可通过；未改grounded、未挪关节过门禁。

两图尚未apply-analysis、bind、generate或动态验收；不能因为coverage=0缺口就说通用能力成立。用户授权“两图验收后统一本地提交”的条件未满足，因此未commit/push。下一步需决定遮挡关节校验的修复边界，以及爆破新星可靠分层来源；不扩大成新的生成能力。


## 持续有效遮挡即可省略隐藏补全（2026-11-25，当前实现待独立复审）

用户明确“能够有效遮挡，就不用非得补全，只要展示的效果显示正常即可”，并授权按此修复。当前开发代码为complete-modules-4/module-qc-9，源码、合同、Skill与使用指南已同步，尚未套用旧批准或提交。

刚体身体父子joint可通过connection.hidden_by精确引用已有遮挡pair；父侧允许不落可见mask，但子pivot、普通容差、抓握、脚锚与漂移约束不变。源隐藏域必须不透明、连通、同源同摘要、与合同一致；登记只授权验证，不生成不存在的背面材质。hiddenJointCoverage检查实际指定前景/软网格在每个输出帧时刻及循环终点是否持续遮住缺失域，露出仍失败。

真实验证目录 .studio-work/occluded-joints-20261125/timespace-r6/：颈点[1073,723]保留、grounded/脚锚保留，静止重组差0；attach_head登记4293像素，其中1484未补，241时刻仍覆盖，maxUncoveredPixels=0，所有contact检查pass。围巾用同一cloth模块的前后piece保留真实绕颈层级，局部领口实际露出的窄缝才使用已有local补全。

整图没有通过：remainingHiddenCoverage仍记录前襟/腿和外围区域的实际露底或待审，visual/overall/export未批准；不能以颈部通过代替整张角色通过。爆破新星仍待可靠手/火焰分层，不在这轮规则修改内。

测试：遮挡/证据19项正反例通过，系统相关113项通过，受影响组合136项通过；最近全量270项中269通过、1项既有V1 Erika EXPORT_APPROVAL_REQUIRED失败，日志review-fix-evidence/full-tests.log。最后的路径空值/运行期失败断言增强后，专项19项重跑通过。当前review pending。8765旧进程未在本轮重启，仍不能消费新hidden_by字段；工具页切换前须公开重启/prepare，不伪造恢复旧任务。未commit/push/export/安装/付费生成。


### 遮挡证据锁定复修（2026-11-25，待独立复审）

第一次独立复审发现：外部sourceHiddenTarget/remainingHidden不在旧artifact范围内，审核后改写可能不使旧批准失效。已用统一受控路径读取与最终外部掩码绑定修复，未复制或改写旧mask/rig/health，也未改变持续遮挡标准。

当前复核/导出摘要包含同一任务根内的visible/hidden/target/remaining全部掩码相对路径与字节SHA；导出前重新读PNG并核尺寸/二值格式/原合同target与remaining，再聚合保存的质量状态。绝对路径越界、父目录跳转和symlink escape均拒绝。旧qc版本凭据不能直接导出。

真实timespace-r6引用的20份target/remaining当前全部被绑定（全部外部依赖46份），见review-fix-evidence/real-r6-binding-proof.json；这只是新的证据完整性校验，不是重新签发动态/整体通过。r6旧health仍fail，未改图或发布；本次待审不改变其状态。


### 原图/分析/验收报告路径锁定复修（2026-11-25，待独立复审）

第二次复审指出原图source_image和分析analysis_file未统一限制在任务根，外部同字节文件可能沿用旧批准。当前修复仅扩充证据读取边界，不改变隐藏关节规则或素材：原图、分析、可选health报告在读取前使用实际rig.parent校验，build报告和骨架限制于rig内；生成前模块确认入口也拒绝任务外源图/分析/manifest/对照图。

当前binding同时包含原图/分析的规范任务相对路径与字节SHA；搬到同任务另一个文件位置也使旧批准失效，同一文件的绝对/相对表示则等价。最终视觉校验复用同一份已验证分析快照，不二次读取报告自报路径。

专项26项通过，系统/venv受影响组合各60项通过；全量277项，276通过、1既有V1 Erika EXPORT_APPROVAL_REQUIRED失败。日志review-fix-input-paths/下。真实旧r6引用另一任务原图，因此新绑定明确拒绝它（real-r6-boundary-proof.json）；没有迁移/重写旧素材或伪造新批准，旧r6继续只作诊断，整体fail不变。

当前修复待独立复审；未提交、推送、导出或重启服务。此前20/20目标/剩余掩码摘要的只读证据仍说明掩码被纳入，但不再被描述为旧r6跨任务原图也满足当前完整绑定条件。


### rig目录与渲染贴图依赖复修（2026-11-25，证据/路径范围APPROVED）

第三次复审发现旧rglob会跳过目录符号链接，可能遗漏渲染器实际读取的外部PNG。本轮改为逐层预检：进入子目录或is_file筛选之前拒绝rig内任何文件/目录/断链符号链接，并复核每个条目的真实位置；不把跳过的目录当作空目录通过。骨架实际images目录及slot attachment路径也限制在rig内，直接绝对/父目录跳转引用外部贴图同样拒绝。

专项30项通过，系统/venv受影响组合各64项通过；全量281项，280通过、1项既有V1 Erika EXPORT_APPROVAL_REQUIRED失败，日志review-fix-rig-links/。新回归覆盖首次绑定前images外链、外部PNG变更后旧批准拒绝且不读外部字节、根内链接/断链同样拒绝，以及骨架直接外部图片/附件路径不能绕过扫描。

只改验收绑定、测试与说明；动画、掩码、r6既有health及用户的持续遮挡标准均未改变。Orca reviewer已签发限定APPROVED，独立系统/venv专项各30项、相关各118项通过；仅证据绑定/rig路径范围，不构成整图/正式导出批准。回执及当时SHA见review-fix-rig-links/APPROVAL-RECEIPT.json；仍未提交或重启服务。


当前下一步：回到时空幻象timespace-r6真实动态中的前襟/腿边。先区分实际可见裂缝与仍被遮住的缺失，不因隐藏材质不完整或机器pending就扩大补全；只做已有输入/层级/local能力内的必要修正。爆破新星仍等可靠手/火焰分层。两图整体通过前，统一提交条件未满足。


### 法杖实际可见断裂的有限修正结果（2026-11-25）

独立视觉复核确认时空幻象唯一同等级重要残留是staff杆身中段断裂。按现有多piece同骨骼、正确前后层级和有界local补边做了一次候选，证据在 .studio-work/staff-occlusion-20261125/。可见原像素与握持/动作不变、rest_diff=0，但关键帧仍可见断杆，正常动作最大3220未补像素外露（压力3953）。FAIL/NEEDS_INPUT，不继续盲扩补色；需要可信杆身材质/原始分层才能继续可靠验证。

这不推翻“持续有效遮挡即可不补”的标准：当前杖杆缺段确实会露出，故不符合该通过条件。代码/证据锁定限定APPROVED保留；两个新代表图整体未通过，统一提交前置不满足，未提交或导出。

</details>
