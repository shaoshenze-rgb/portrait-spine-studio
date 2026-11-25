---
name: portrait-spine-idle
description: >
  在 Cindy 项目对话中实际读取角色立绘，规划完整语义模块 V2，通过本地脚本与工具页
  检查保形轻摆、软模块轻弯、接触遮挡和 Spine 待机动画。用于立绘制作、修改和批量验收；
  无需安装连接插件，已有图像服务结果显式交接，不把旧几何诊断当成品。
---

# Portrait Spine Idle

## 主目标（每轮开工先读）

用户目标是可复用的 Spine 动画制作主链：**立绘 → 分层与遮挡规划 → 按需由 Canvas 拆层/补全 → 分层材质按原位置回接 → 骨骼绑定与动画 → 全循环检查关节割裂 → 缺陷局部返修。**角色是验证样本，不是手工定制项目。

每次新一轮工作、恢复上下文或派工前，实际重读本节，并用一句话说明本轮在推进主链哪一环、会交付什么证据。无法对应主链的工作先停止；不能用某一角色修好了、测试变多或代码审查通过替代端到端闭环通过。

统一要求：身体/武器保形，模块完整；重要关节不能可见割裂。整个循环持续遮住的材料无需补，实际会露出的缺失内容通过 Canvas 补全并验证。Canvas 产物须核对源输入、图层位置、原可见内容及返修范围，再接入生产；生成成功不自动等于验收通过。遇到分层/材质不足，优先修这条自动交接，不默认要求用户另做素材，不继续逐角色盲描点或写角色ID补丁。

当前 Cindy 对话统一编排 Canvas 与项目脚本，工具页服务不直连模型/凭证；conversation_handoff不等于Host身份认证。主目标与当前已实现能力分开记录，真实状态见验收摘要。

当前能力边界与真实验收状态见[验收状态](../../../docs/acceptance-summary.md)。对话手工多边形只是待核草稿，不能宣称已具备可靠自动语义分割。多轮完整图审仍有跨物错归时，保留失败并回到 Canvas 分层/补全交接；能力或授权不足时明确实际阻塞，不改用持续手工描点，不重复提交状态未知的生成任务。

## 准备当前输入

在项目根运行已有本地服务，不重复启动已正确运行的实例：

先核对 `/api/health` 的 `module_qc_rules_version` 与当前 `src/gates/module_quality.py` 中的 `RULES_VERSION`。不一致时不能把旧进程的检查当作新规则已生效；按服务重启边界处理并重新 prepare，不沿用旧内存任务或旧视觉凭据。

    python3 bin/start-studio.py --port 8765 --no-open
    python3 scripts/portrait_task.py prepare --image <透明立绘路径> --preset agile_full
    python3 scripts/segment_modules_sam.py emit-grid --image <本次image_path> --out <任务目录>/coord-grid.jpg --step 50
    python3 scripts/segment_modules_sam.py segment --image <本次image_path> --spec <sam_prompts.json> --out-dir <任务目录>/sam

启动命令保持运行，prepare在另一终端/Agent调用执行。使用prepare返回的image_path、job_id、input_sha256、image_size，实际原生看图；必要时对照灰底透明合成与局部近景。带背景图先取得已授权的真实去底原件并记录来源，不用mock、截图或旧缓存代替。已有透明底是透传，不是去底。

## 规划完整模块

读取[完整模块合同](../../../docs/contracts/complete-modules.md)并按源码当前字段产出V2 JSON。旧V1的主体大mask和关键点不能自动变成完整模块。头身、手、自然肢段和硬物保形；软模块完整轻弯；宠物、附件、武器、主鱼尾与有意效果组必须有真实归属。

按自然结构和相对运动需要划分，避免顺着图案/高光/三角形切碎。一个物件被遮挡成多个可见块仍属于同模块，所有残边都要归属。piece 可提供 polygons 或经 SHA 校验的原分辨率 mask_png_base64/mask_sha256。挂在手/武器下的半透明特效用 locked_root 软链（根 0.25°、尾端轻弯）。显式写parent/pivot、接触与前后遮挡；无法可靠表达的关系停待审。输入自身残留用input_issues，不能伪装有意效果、删alpha或塞人体。不同实物已预合成到同一RGBA像素、无法保持原纹理独立拆层时，用precomposited_overlap定位并说明，等待源层或可靠补全，不强制同动掩盖。

完整模块还需真实图审；coverage=0缺口和锁定像素无变化不是语义正确证据。不使用角色名、文件名或图片摘要例外调整代码、阈值或曲线。

## Canvas结果交接

提交新的Canvas生成前，先确认本轮有有效的作品库原生读取映射，并读通已有参考原件。若映射为空或library诊断为unauthorized/authorizedReadonly=false，暂停新的生成，保留已有任务回执；这不是Mivo账号失效，不改账户/全局权限，也不复用上一轮被撤销的路径。授权恢复后先读取原结果，核SHA并存入项目再继续。read_images的not_image还需核对节点是否有asset引用，不能把空的失败槽当作原件；没有找到原件时不得以“重试生成”代替恢复。

需要分层或补全时，先发现当前Canvas工具并使用真实源节点/回执，读取original原件；通过[使用指南](../../../docs/usage.md#canvas交接主链)的canvas_handoff导入、自动对齐、回接provided材质。必须保留source/node/task/像素摘要和子层来源链。层数或坐标未合格时只返修该层，不手工改角色点、不重复已完成的收费调用。

对齐报告needs_layer_review不是语义通过；有一层needs_canvas_registration时不借旧成功结果冒充新层。新Canvas材质会使旧模块/视觉凭据失效，继续走下面的同一套图审与生成步骤。未闭合环节在验收摘要中明确记录，不以主目标文字或接口存在宣称功能完成。

独立Canvas完整层进入绑定前，使用[连接诊断与返修编排](../../../docs/usage.md#连接诊断与返修编排)检查所有连接。极淡锚点伴父侧支持间隔时先对照原图确认肢段和关节，不按距离直接补画或按最近像素移点；组合图有alpha不代表父子材质连接成立。已声明遮挡仍须编译和全循环验证。

返修决定、提交预留、实际任务回执、结果与新输入复查都进入同一任务workflow记录。脚本不调用模型，也不授予权限；真正Canvas操作仍由当前对话执行。结果未知时保留请求，不自动重提或重置次数。输入变化后重新诊断并使旧绑定/观察失效；未改材质上的已知视觉问题继续保留，不通过删模块或局部探针替代完整角色。

## 生成前的逐模块图审

验收先看正常尺寸下躯干、四肢、头部/头发、武器和关节的重要错位；小碎边按影响记录，不无限追求像素级清零。重要错归先修输入和绑定，不扩展功能；细则与人工纠正审计见[故障参考](references/rerun-and-troubleshooting.md)。

遮挡关系与运动父级分别确认：腰布和腿同挂躯干，仍可能互相遮挡。将后层预期延续到哪里写入 `occlusions`，不能把空列表或自动接缝垫边当作完整补全。移开前层检查后层的完整轮廓；若同一护甲、裤腿被分给多个驱动，先纠正归属，再讨论补底。具体排查与验收见[遮挡露底检查](references/rerun-and-troubleshooting.md#遮挡移开后的完整性)。

先逐完整硬物与装饰核对所有可见区域是否共用同一模块／驱动，再按真实身体关节定边。颈点、髋点不能直接切过闭合金属领圈、腰甲、硬图案或武器；被遮挡分开的可见块仍可属于同模块。不要用同色补全掩盖原本切坏的硬面。

    python3 scripts/inspect_module_plan.py --image <本次image_path> --analysis <V2模块JSON> --out-dir <本次任务目录>/review-modules

工具输出每个module聚合全部pieces的原生孤立图、同位置原图对照及module-review.json，状态固定unreviewed。Agent必须实际逐张查看，对照硬物连续面、所有边缘和接触；连通块数量、piece数量及mask锁定都不替代这个检查。任务页只展示摘要与当前合同匹配的对照链接。生成后出现新裂线应先回模块规划，不仅扩大填色或关掉动作。

## 应用与生成

    python3 scripts/portrait_task.py apply-analysis --job <job_id> --analysis <V2模块JSON>
    python3 scripts/portrait_task.py status --job <job_id>

默认停awaiting_rig_confirm并自动输出当前原图/逐模块孤立图。实际逐张核对完整硬面、所有边缘、遮挡分离块、形状类型、父级和连接，将每个模块的观察写入使用指南所述module-review记录；记录绑定观察前的module_review_binding，在原绑定确认中一次提交。缺证或错切先回规划，不以零漏覆盖当完整。V2不通过旧骨级混权编辑修改合同，需在对话更新完整规划后重新应用同输入。

用户确认绑定后，或本次已明确授权Agent批验，才继续：

    python3 scripts/portrait_task.py generate --job <job_id> --module-review <实际模块图审记录.json>

不靠整图僵摆、归零、拉长硬模块或移动已确认关节换通过。脚锚和握持在不伸长的可达范围内成立；不能表达或补全不可靠时保留待审。局部反馈按[故障参考](references/rerun-and-troubleshooting.md)决定重规划或最低重跑层。

## 分项验收与交付

最终显示为准：隐藏部位在整个实际动画周期内始终有效遮挡时，无需强制补全不可见材质；实际露出的缺失区域才补。刚体父子自然关节可按[模块合同](../../../docs/contracts/complete-modules.md)引用明确的 hidden_by 遮挡证据，通过完整周期 hiddenJointCoverage 后保留未补后层。仍检查关节漂移/刚体保形，不能只凭单帧被遮住或合成画面alpha通过；抓握、足锚、软端点不套这个例外。

先核输入/归属、绑定/运动、机器QC，再在正常全图以原速观察完整循环并看关键接缝。明确区分真实连续播放+抽样和逐帧终验，不以计时器推进或曲线存在自证画面正确。

对每处遮挡观察其自身暴露最大的时刻，保存同坐标原图／静止／动态局部对照。躯干呼吸最高点不等于腰布、胯部或四肢的最大相对位移。`contactSweepCoverage` 只证明连接点邻域；`remainingHiddenCoverage=not_applicable` 只表示无登记目标，都不能替代完整肢体检查。发现真实断肢、护甲错层或露底时 `contactsOcclusion`／`semanticCompleteness` 应判失败；负空间提示项通过汇总不能覆盖视觉失败。

五项视觉观察必须逐项记录status与实际依据：完整语义和边缘、硬体保形、软模块轻弯、接触遮挡补全、循环观感；实际看原图与模块图是必要条件。通过声明绑定观察前保存的status版本，命令和证据格式见[使用指南](../../../docs/usage.md)。机器失败/待审不可通过视觉确认覆盖。只有quality.overall.exportAllowed=true且文件摘要仍匹配才交付工程、视频或自包含单文件 HTML 预览。

先用代表例跑通完整合同，再扩同版全库；每张保留真实V2规划、模块图和分项证据，不把旧V1全拒算全库成功。旧诊断产物和未采用的plugins/cindy-vision-bridge保留历史，不安装、不删除、不作为本合同通过凭据。服务重启需公开prepare新任务并明确同图重绑，不能伪造恢复旧内存job。
