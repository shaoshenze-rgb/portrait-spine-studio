# 局部返修与读取故障

本页服务[四步主流程](../SKILL.md)。以原速动画中的真实缺陷决定返修，不按机器告警数量扩任务。

| 观察到的问题 | 最小处理 | 完成条件 |
| --- | --- | --- |
| 白块、透明洞残留或白边 | 同一原件重跑 `--cleanup`；查看轮廓裁切和 defringe 指标 | 深浅底都无明显残块，主体轮廓保留 |
| 特效糊且盖住手/脸 | 将该 `effect_group` 设为 `effect_mode="omit"` 并重跑净化补底 | 特效停止绘制，外围和身体凸包内均无残丝；手/脸未误删 |
| Canvas 漏掉手肘、上臂或躯干 | 自动原图补底；查看恢复块的 target_layer 和对照图 | 主体静止完整，运动中恢复材质随正确父层，不再开裂 |
| 手压肩甲、衣物盖错 | 先调 `drawOrder`，相关子武器一起核对 | 最大摆角下前后关系正确；静止不双绘 |
| 头手或关节摆开裂缝 | 单模块 `motion_scale` 先试 0.5–0.7，再检查已有 local 垫边 | 目标接缝完整，躯干呼吸与软摆仍自然 |
| 站立脚漂或 IK 不可达 | 核对 grounded 的真实两段刚体链与静态 foot 锚 | 脚稳且硬物不拉长；不要改浮空掩盖问题 |
| 仅静态距离大或少数边缘像素低 alpha | 保留提示，直接生成完整预览观察 | 以实际露底/穿模判定，不用告警单独阻断 |
| 整图 `canvas_edit_image` 把腿/躯干画偏 | 先调 `scripts/canvas_handoff.py prepare-edit-crop` 自动裁局部带锚点框再上板，回传用 `apply-edit-patch --crop-meta` | SIFT 高内点对齐，不改动裁框外任何部位 |
| 遮挡区底层已有旧像素/半透明残影导致补底写不进 | `apply-edit-patch` 加 `--replace-occluded`（半透明光效同时加 `--alpha-mode inpaint_alpha`） | 仅替换遮挡区内旧像素，可见区 `max_outside_diff == 0` |
| 补底回执报 `open_air_no_backing`（背后本是透明空气） | 调 `scripts/canvas_handoff.py clean-open-air-halo` 清主层紧贴前景的软光晕残边 | 实心主体 `max_solid_core_diff == 0`，无旧光晕残边 |
| 无底图死交叉层反复压测露底 | 调 `scripts/canvas_handoff.py rebind-dead-crossings` 按解剖层级自动改绑次级交叉件 | 原 `draw_order` 不变、站姿脚底 `0 px`；有任意层对被拦截即退出码 `2` 挂起待审 |
| 运动后层从静态靴子/腿甲/地面背后探出双边重影 | 调 `scripts/canvas_handoff.py trim-hidden-pokeout` 无损裁掉静态不透明前景背后临近空域的隐藏像素 | 静止可见区 `rest_visible_diff == 0`，正常出露区保留 48 px 垫边，无双边重影 |
| 前景道具移开后，后层半透明光效露出硬切轮廓边 | 调 `scripts/canvas_handoff.py fill-translucent-underlap` 仅在道具遮挡区内平滑延展半透明光效 | 遮挡区外 `max_outside_diff == 0`，道具摆开后光效连续无硬切边 |
| 局部裁图面积过大（>55% 整图）或未命中压测露底区 | `prepare-edit-crop` 自动返回 `ok:false`（退出码 `2`）并标注 `warnings` | 不直接发整图冒充局部补底，交用户审核或缩小目标层对 |
| 同角色和谐版/原版、国内/国际版变体 | 调 `scripts/canvas_handoff.py transfer-sibling-variant` 迁移已绿灯同构兄弟切分 | `diff_ratio <= 15%`，目标变体静止合成与原图一致 |
| 通用流程 3 轮止损后仍未过门 | **立即挂起标为 `needs_human_review` 并汇报用户**；未获用户明确同意严禁擅自单张手搓特调 | 用户审批后方可在 `per_character_override` 记录批准并定向处理 |

## 遮挡移开后的完整性

围巾后脖子、衣物下躯干等在整个循环始终被挡住时，不要求补全。真正露出的大缺口才修。`seam-worst-crops.jpg` 给出新封闭透明洞与头—手、躯干—手的最大相对摆角截图；开放接缝可能没有洞计数，仍须看完整视频。

恢复覆盖为零缺失，只证明像素被补回，不证明自动选择的父层或解剖正确。尤其检查完整手臂、武器杆身、硬甲与腰腿。补底有歧义时保留待审；已有素材确实不足且当前授权允许，才针对一个可见缺口补一次 Canvas。不要用任意同色块代替真实材质。

特效清理从被关闭效果的颜色及范围识别残丝；与主体同色的结构可能受影响，不能宣称对所有新图都无误删。检查清理前后和 `missing_original_regions_raw`：有意移除的效果不等于人物缺件。

## 默认优先使用 Mivo Canvas 与作品库读取（不可用/报错时降级本机兜底）

默认主链为**品质优先（Mivo Canvas 拆层与补全优先 + 原画多层 PSD 独立层无损直取 + Canvas 不可用/报错/对齐失败时自动降级本机兜底）**。调用 `xd-mivo-canvas` 时严格遵守：

0. **放图顺序防错**：调用 `canvas_place_image` 前先调 `canvas_selection` 唤醒并确认面板就绪；若返回 `REFERENCE_LIBRARY_CHANGED`，先 `canvas_selection` 确认再重试一次 `canvas_place_image`，严禁跳过放图直接读取画布上以前选中的旧节点。

随后区分三种读取状态，优先读宿主注入到会话的只读目录根，不要只凭 shell 环境变量 `$LIBRARY_READ_ROOT` 为空误判：

1. **新会话首轮冷启动（`read_images(original)` 已返回 `available`，仅首轮未挂载只读映射）**：不是权限故障。保留已入板节点、暂不提交付费任务，不报 `fail`，给出一键 `cindy-ui` 按钮（`label="继续并开始拆层"`、`send="继续"`）进第 2 轮即可自动生效；后续在同一会话继续做新图不会再停。
2. **真实不可读（`unauth_cache` / `unavailable` / `unauthorized` 或第 2 轮仍无映射）**：停止新的生成，保留 scene/node/task 与精确诊断，不改账号/全局权限，不猜旧缓存路径。
3. **恢复后续跑**：先取已有结果，核 SHA 和 PNG 再继续；刷新面板或重启本身不是已修复证据。

## 导出与版本

签收对应实际看过的版本。素材、层级或动作变更后重新看视频，不沿用旧签收。Studio 检查基础条件并提供签收按钮；底层 `user_signoff_path` 放行保留旧机器提示，不修改旧 health 为全绿。未传签收参数的历史严格路径仍可能拒绝导出，不能据此重新启动旧审计流程。

具体参数与可运行示例见[使用指南](../../../../docs/usage.md)。旧 `workflow`、SAM 与逐模块审计工具留作历史维护，本主链不用它们调度返修。
