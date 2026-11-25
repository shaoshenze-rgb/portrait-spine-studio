# 完整语义模块 V2 合同

实现正本是 src/studio/module_contract.py，运动编译为 src/pipeline/module_rig.py，QC为 src/gates/module_quality.py。此合同描述当前可执行能力，未支持的关系不得用空壳字段冒充。

## 输入与归属

顶层使用 version=2、job_id、input_sha256、image_size=[width,height]、coord_mode=normalized、pose_type、anatomy、modules、connections、occlusions、uncertainties。输入绑定当前去底后PNG；不能按文件名、角色ID或旧哈希查表套坐标。

anatomy可声明humanoid_full、humanoid_bust、merfolk；other当前待审。鱼尾是主下身，不伪造膝踝或通过bust_half删尾。未知/低置信在uncertainties说明，不能给不确定内容虚构高置信。

每个module包含id、role、motion_type、parent、pivot、motion、confidence以及pieces。motion_type为rigid/soft/static。每个piece包含唯一id、唯一draw_order，以及polygons或原分辨率二值掩码（mask_png_base64 + mask_sha256）。掩码必须是标准PNG（魔数与IHDR宽高匹配image_size）、mode为L或1、像素严格{0,255}；base64≤16MB、解码后≤12MB、w*h≤64,000,000。掩码不得覆盖原图alpha==0区域（否则VisionReviewRequired，不会静默裁掉）。若两者同时提供以精确掩码为准。同module多个可见块共用运动归属，不因遮挡或粒子不连通而自动拆成独立骨。完整归属要求全部可见前景被明确规划，不按RGB梯度/近邻重派。

头脸、躯干、自然肢段、手、硬质武器/器具使用保形模块。soft需要从pivot出发的2–5点centerline，保持完整受限轻弯；static必须有static_reason，不能把人体冻结来换通过。挂在活动宿主下的半透明特效（effect_group）选用soft_bend时必须声明locked_root=true，且父链上须有非static/follow的活动宿主：根节锁在0.25°（正常循环局部旋转峰峰值≤0.6°，1.5x压力≤0.9°），尾端链节再渐进轻弯。locked_root不得用于其他role。独立顶层effect_group仍可static。护甲必须rigid且motion仅为sway或follow；武器必须rigid且follow。模块名是身份，不是动作调参表；motion只引用breath/sway/follow/soft_bend/static统一规则，不提供逐角色曲线。

input_issues为明确的输入缺陷区域列表：id、kind(background_residue/fringe/source_damage/precomposited_overlap)、polygons、notes。非空时当前入口INPUT_REVIEW_REQUIRED并保留输入，不能当effect_group、删alpha或兜给身体。precomposited_overlap表示实际看见不同物体材质已在同一RGBA像素预合成，保持原纹理无法恢复独立层；记录区域与依据，等待源分层或可靠补全输入，不以强制共动或新增收费生成替代。

## 接触与遮挡

每个有parent的模块要有显式连接：id、type、a、b、point。连接点与子模块pivot一致；joint表示自然关节，attachment表示真实附件连接，grip用于共同握持运动。武器子模块必须rigid/follow，并声明grip。

foot_anchor用chain指定两个保形肢段和独立静态foot，真实初始关节不挪位、无stretch；不可达时待审。双端链/双手闭环等未支持关系须明确停下，不用一个parent假装表达双端连接。

遮挡使用piece标识front/back、明确polygons与completion。补全方式为none、显式窄带local或provided。local只在既定窄带内取本模块纹理；provided接受当前Canvas拆层/改图返回、已经对齐当前画布的RGBA材质，不再以本地涂抹冒充生成结果。两者都只作用于声明遮挡域，不改已可见原图像素。

local可用completion_polygons直接限定补全窄带；也可显式选择derive_band=true，让统一规则从后层piece原有可见材质向声明遮挡区域内推导窄带，两者不能同时使用。推导深度按真实身体模块可见范围的高度（包含主鱼尾，不计外围特效或画布留白）折算为每1080像素32像素，occlusion也可选band_depth_px_at_1080在4.0..48.0微调。精确掩码piece不得再用claims[back]&visible[front]（互斥后为空），改为zone&visible[front]&~visible[back]且不腐蚀内部交界，避免2px透明沟槽。路径不穿过别件、透明孔或已有补全；只使用原图alpha=255的不透明前景区域；半透明前景下不新增会改变原图RGBA的后层材质。报告bandDerivation记录eligiblePixels、materializedPixels、unfilledEligiblePixels、unsupportedTranslucentPixels，completionStatus固定partial_pending_visual_review，部分补全不代表完整接缝或动画已通过。原local默认仍要求完整声明区域可填，超深时拒绝。

## 生成前的完整硬物审阅

先核一个完整硬物/装饰的全部可见区域（例如闭合金属领圈、腰甲、纹样边框或整把武器）是否同一module共同运动，再按自然身体关节定边。关节点坐标不能一刀切穿连续硬表面；遮挡分离块可以是同module的多个piece，不按数量或连通性判断碎件。

使用scripts/inspect_module_plan.py输出逐module的原图/孤立图对照，固定unreviewed，必须由实际看图复核后才形成观察结论。若规划将同一硬物错命名成两个module，机器不能仅凭RGB证明语义，不能声称已自动识别保证完整。动画新裂线须先查规划，不用同色local补全掩盖切坏的硬面。

已确认语义边界存在少量描点误差时，可选声明boundary_refinements：每项包含id、既有piece标识a/b、axis(x或y)、窄走廊polygons及说明实际边界的notes。此项只允许在指定两件的既有边界附近贴边，最大偏移按前景高度折算为每1080像素3像素，不发现新模块、不扩大前景覆盖；走廊多边界、边缘不清、纹理竞争、偏移不连续或连通/镂空拓扑变化时待审。超过小范围的错切必须先重新规划。

贴边产物在module_projection及rig.moduleContract.boundaryRefinements保留前后mask、变化mask、文件摘要和变化像素数，visualReview仍为pending。生成前须查看贴边后的完整模块孤立图，动画后仍检查原图语义和接缝；贴边数值通过不构成视觉通过。

## 质量与复核

coverage与掩码锁定只证明声明没有被程序改派，不证明声明本身正确。正常图审必须核对原图语义、全部边缘、遮挡分离块与有意效果组，再检查动画。

视觉报告observations使用下列五个键，每项是status和具体notes：semanticCompleteness、rigidShape、softDeformation、contactsOcclusion、loopAppearance。状态为pass/fail/pending/not_applicable；语义、保形和循环必需，soft或contacts确实不存在时才可不适用。还需明确original_and_modules_viewed和full_loop_observed。

通过声明必须绑定当前source、analysis、rig与纹理文件的实际摘要。输入、规划或动画变化后旧确认失效。整体结论由统一quality聚合，机器失败/待审不能被视觉布尔值覆盖。

机器检查区分 hiddenFill（生成材质是否合法）与 remainingHiddenCoverage（预计存在的后层材质是否仍被遮住或已实际补全）。具体范围和限制见下节，二者均不自动证明分件语义正确。

## 遮挡覆盖与自动接缝的区别

运动拓扑与遮挡是两种关系。同级衣物和腿即使没有直接 connection，也须按实际前后关系声明 occlusions。QC 的 occlusionDeclarations 会把相接的“后层刚体肢段／前层软体、共享父级”而未声明遮挡的边界列为 pending，阻止把空遮挡列表当成完整覆盖；它只提出待核边界，不自动猜补底形状或授权填充。自动接缝只针对同模块多 piece、直接父子或直接连接的窄缝；它不推断完整腿根，不对无连接交叉任意扩画。显式 local 只接管声明区，声明区外保留原有合法自动窄带；显式 none 仍关闭该 pair 的自动补全，不能被自动逻辑覆盖。

声明的局部深度须一致传到 materialize、split_parts 和构建报告，不能在分件阶段又截回默认值；实际填充始终受已绑定 hidden mask 限制，不扩大可见归属。

所有显式遮挡均登记完整的 sourceHiddenTarget 与 remainingHidden 摘要，包括 completion=none、仅补 completion_polygons 子区域的情况。对于精确可见掩码，目标为声明区域内的前层可见像素；对于多边形，仍受后层原声明形状限制。自动窄带不代替这些语义声明。

remainingHiddenCoverage 读取实际后层贴图 alpha；声明填充 mask 非空不等于该处已经填实。刚体后层使用自身真实变换，前层可为刚体或实际软网格；只检查声明的前层材质，不借整幅合成图 alpha 自证。尚未支持的软后层物质坐标仍 pending。检查范围以登记目标和实际采样时刻为限；无目标的 not_applicable 不证明语义完整。

negativeSpace/no_holes 和 softDeformation 的数值应变为提示项，实测失败值保留；翻面、退化、保形、接触与已登记区域真实外露继续阻断。可见的断肢、缺底和硬物错切必须判视觉失败，不能以提示项不阻断为理由导出。

通用动作由 src/pipeline/module_rig.py 的 role/profile/几何位置驱动，当前包含左右反向错峰、头身反相和多谐波呼吸。具体数值从代码读取，不在 Skill 中维护第二份参数表；换频率、幅值、掩码或遮挡关系后须重新绑定对应证据。

## 重要结构验收范围

### 遮挡成立时不强制生成后层

被遮挡的不可见材料允许不存在；通过条件是指定前层在实际完整动画中持续有效遮住登记的缺失区域，且关节关系无漂移。可见像素和普通落点容差不变，不用扩大补底深度、吞并围巾或挪关节解决遮挡关节。

connections 可选 hidden_by=[{front: piece_id, back: piece_id}]，仅精确引用已有 occlusions，不另带路径、mask或polygon。当前只允许刚体身体父子 joint：b.parent=a，point仍等于b的pivot，b自身的可见落点要求保留；例外只作用于父侧可见距离。back须属于a并提供精确可见掩码，避免多边形重叠语义歧义；front只能为b，或同一a下有明确连接的身体/衣物模块；weapon、effect_group、pet、prop、foot不作为该例外的遮挡物。grip、foot_anchor、foot角色和软端点仍按原规则处理。

编译必须复核遮挡pair、源摘要、mask摘要、实际前后层级和登记区域与父模块可见材料连通性；point须处在精确登记的不透明域内。该域只证明被声明的覆盖关系，不生成新纹理，不是自动推断人体结构。解剖位置仍须实际图审。

hiddenJointCoverage 在每个输出帧时刻以及循环终点检查未补区域随父刚体运动后的覆盖，只渲染被引用的实际前层slot/软网格，不借整图或任意第三者alpha。实际后层已有材质的像素无需重复遮挡；缺少后层但指定前景有效覆盖则通过，遮挡打开或透明度不足时失败。仅有稀疏样本仍pending；正常接触漂移、保形、足锚等检查不被覆盖结论替代。检查证明的是报告列出的采样范围，并仍需正常速度完整循环视觉观察。

验收凭据同时绑定实际 rig 文件和其全部外部掩码依赖：visibleMasks、hiddenMask、sourceHiddenTargetMask、remainingHiddenMask。构建期引用限定在本次物化掩码目录；运行期/最终复核的可信任务根取自实际 rig 目录的父目录，不接受报告自行扩大根。绝对路径须在根内；相对证据路径以任务根为基准；任何父目录跳转和 resolve 后越界的符号链接都拒绝，先校验路径再读取。

artifact 摘要包含证据相对路径与实际字节SHA。导出前重新核对当前PNG的摘要、二值格式、尺寸，以及按原合同重算的target和remaining内容，再检查保存的质量状态与原审核绑定。改文件、搬路径或更新单个SHA都不能复用旧批准；未通过则重新审计/复核，不修改既有结果伪通过。

原图source_image、分析analysis_file和可选health_report_path同样受实际rig父目录的任务根约束，先resolve并拒绝越界/父目录跳转/符号链接逃逸，再读字节。原图与分析的任务相对位置和内容SHA一并绑定；同字节搬到另一位置也须重新复核。分析只读取一次，最终视觉证据校验使用这一份已验证快照，不再根据报告路径重新打开文件。生成前模块确认也校验其原图、分析、manifest和对照图的任务内路径。

rig产物树禁止任何文件、目录或断链符号链接（包括指向根内的链接）。扫描在进入子目录和is_file筛选前检查，不能把目录外链静默跳过。骨架中的images目录和slot attachment实际贴图路径也必须解析到rig内的现有文件，防止直接外部引用绕过目录扫描；批准前发现外链即拒绝，不读取外部贴图后再补摘要。

默认先纠正正常播放中躯干、四肢、头部/头发、武器、关节的明显错归和错位。零星碎边和同一 grip 刚性装配附近不影响轮廓/动作的小残片写入观察说明，不按它们无限扩大修复；数值硬检查仍按现役规则执行。具体轮廓判断、人工调整追溯和探针坐标系要求见项目 Skill 的故障参考。

重要结构范围的观察通过，不代表所有像素语义或所有补底已被机器证明。必须保留真实 pending/fail，尤其原图半透明区域、未支持的软后层坐标不能写成已验证；此范围不授权跳过 overall/exportAllowed 的正式导出条件。


## Canvas材质回接（当前开发实现，待独立审核）

生产采用occlusion.completion=provided，并携带provided对象：png_base64/sha256为已对齐工作画布的RGBA PNG；input_sha256绑定本次预处理输入；canvas_scene_id/canvas_node_id/canvas_task_id/operation及source_asset_sha256/result_asset_sha256记录真实画布交接。operation仅split-layers或edit-image。不得把本地均值补色或mock伪装为Canvas回执；这些字段是当前对话的交接声明，不是Host身份认证。

后层预期范围仍由occlusion.polygons声明，可用completion_polygons收窄本次回填。provided的alpha与该区域、前层可见掩码、原图alpha==255取交集后落入任务内材质文件。原本半透明区域不偷偷增加后层改变原图；无有效覆盖时返回可执行的Canvas返修阻塞，不退回本地凑图。提供的纹理不受local深度限制，但只能写入声明域；可见像素仍逐字节锁定。同一后层多个结果重叠冲突时拒绝。

未携带结果的provided请求返回CANVAS_COMPLETION_REQUIRED，含当前input/front/back，不自动重复提交生成。当前Cindy Agent负责通过已安装Canvas工具提交、读取原件、调用scripts/canvas_handoff.py导入/对齐和attach-completion，然后仍走inspect→apply-analysis→绑定复核→generate→完整循环验收。Python服务不保存平台凭证、不直接调用模型；不新增专用Cindy连接插件。

Canvas材质逐片写入实际parts，运行QC比较实际RGBA与已登记Canvas材质，不用原图色域强夹生成材质；本地补边仍执行原色域约束。提供材质的路径/摘要/尺寸/合同内容也参与最终artifact绑定。原图对照与重要结构图审、动态补全外观、完整循环始终是独立验收，生成/对齐成功不能当成品通过。

图层原件尺寸、裁切位置可能不同，不能只按文件大小猜坐标。交接工具按共同源图特征求相似变换，低置信不输出位置；对子层，先匹配其真实Canvas父层再组合已验证父变换，避免让新补出的像素去匹配原图里不存在的材质。层级仍需按实际结构判定，模型标题不当真值。失败/中断标记阻止消费旧报告；所有原件和回执保留，状态未知不重提交收费任务。

### Canvas骨骼候选的隐藏软附件（开发增量，待独立复审）

独立Canvas层候选可用现有occlusions和attachment.hidden_by，登记缺失父侧材质被哪个身体/衣物层遮住。仅由验证过的Canvas parts编译器显式调用内部material_mode=canvas_layers验证；普通source-sliced入口不变，JSON里的同名字段不能开启例外。父侧须是刚体身体，子侧为非特效软模块；引用必须是同输入的既有pair、背层属于父模块、前层是子件或有真实连接的同父身体/衣物、绘制顺序正确。当前只接受completion=none，不静默处理其他补全或输入缺陷声明。

登记区域不得裁掉静止未遮像素；目标与剩余区域保持相同，绑定实际图层SHA、区域PNG SHA及静止参考SHA。锚点须在登记域内，区域须连到父材质。只有这种已核验声明可以豁免父侧可见mask距离；子侧pivot/软链、普通joint/grip/foot_anchor及漂移检查不变，不移动关节。

hiddenAttachmentCoverage只渲染声明的前层slot/实际网格，按父侧刚体变换跟踪区域，检查每个输出帧及循环终点；不借其他层或合成frame的alpha。任何alpha=0暴露为fail；仅0<alpha<250为lowAlpha/pending，不能称为大断肢，也不能自动签视觉通过。指标仅证明登记区域与相应采样，完整角色语义/外观仍需复核；Canvas候选export block继续保留。
