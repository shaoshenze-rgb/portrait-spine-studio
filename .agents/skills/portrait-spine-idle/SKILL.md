---
name: portrait-spine-idle
description: 将角色立绘经一次 Canvas 粗拆、净化与补底制作成 Spine 待机动画，先预览再微调、签收导出。用于立绘制作与返修。
---

# Portrait Spine Idle

先出 8 秒预览，只修动起来真正穿帮的地方。每轮恢复先读本目标。
身体、武器保形，头发、衣摆、飘带轻摆；全循环始终被遮住的内容不用强补。
默认一次 Canvas 顶层粗拆，不递归拆子层；保留自然呼吸和软摆节奏。
四项基础检查后由用户看图签收；代码测试不等于角色验收。
当前事实见[验收状态](../../../docs/acceptance-summary.md)，具体命令见[使用指南](../../../docs/usage.md)。

## 1. 准备原图，一次粗拆并对齐

使用本地透明 PNG；也可在 Studio 启动后运行 `python3 scripts/portrait_task.py prepare --image <原图.png>`。在当前 Cindy 对话调用 Mivo Canvas，脚本本身不提交生成。

提交前先读通作品库原件：`read_images` 请求 original，并通过当前任务的原生读取映射实际打开。`unavailable`、`unauthorized`、`unauth_cache` 或没有映射时停止生成并报告，保留调用额度；不能猜缓存路径或重复收费。

在本任务生成授权内，对原图只做一次顶层 split-layers，目标 5–10 个完整大块：头、躯干、完整四肢、武器、软饰件、独立特效。保存真实 scene/node/task、原件与 SHA，用 `scripts/canvas_handoff.py import-layers` 自动对齐。确认原件可读、对齐可靠且主要结构未丢；失败不递归。

## 2. 净化、补底，立即出预览

按真实层结构写材质清单与骨架规划，保留 Canvas 独立 RGBA 与层序；不要再送回 SAM 切成互斥小碎块。用 `materialize_canvas_layer_parts` 生成标准 parts，再运行：

```bash
python3 scripts/canvas_to_spine.py --parts-dir <parts目录> --plan <plan.json> --character <角色标识> --out-dir <新输出目录> --evidence-root <共同证据根> --cleanup --fps 30 --tweaks <微调.json>
```

没有微调时省略 `--tweaks`；输入与输出均放在证据根内。`--cleanup` 串联：

- 按原图前景加 1px 缓冲裁掉外部白块和透明洞残留，去半透明白边，保留轮廓内隐藏材质。
- 糊特效盖住主体时，将对应 `effect_group` 设为 `effect_mode="omit"`。停止绘制该层，补底排除外围和凸包内同色残丝；复查主体未被误删。
- 对 ≥200px 主体缺块，从原图恢复到邻接最长或中心最近的合法人体/服装层，带 2px 羽化重叠边；记录 `recovered_regions`，复查 `missing_original_regions`。覆盖恢复不自动证明归属正确。
- 以 `strict_placement=False` 编译，6px 静态距离进入 `placementWarnings`，不锁死预览；支持 grounded 与 airborne，站立图仍须提供真实足锚链。

完成后应有 8 秒、30fps 深浅底 MP4、单文件 HTML、关键帧拼图、`seam-worst-crops.jpg` 和诊断 JSON。Studio 的 `apply-canvas-layers` 走同一流程；语义规划仍由当前对话完成。

## 3. 看完整循环，只做局部快修

重点看手与肩甲的前后层级、头与手最大相对摆角处、衣物移开后的真实缺口；不要只看静帧。接缝图抽取 16 个时刻，辅助定位，不能替代原速完整循环。

优先用 Studio `layer-tweaks` 调整 `drawOrder`、单层显隐和单模块 `motion_scale`（0.0–1.5）。易穿帮手臂可先试 0.5–0.7；单模块可设 0，武器随握持父级。保留呼吸，改后复看。

若仍有真实外露空洞且本地补底、垫边无法解决，才在有效授权内对该单一局部最多补一次 Canvas；不拆子层、不重做整套。结果未知先找回原任务，不重提。处理方法见[局部返修](references/rerun-and-troubleshooting.md)。

## 4. 四项基础检查，人工签收导出

确认：①骨架和贴图完整可读；②8 秒首尾无缝；③静止渲染对齐当前分层合成，原图重要结构无大错位、主体无大块漏件；④刚体无明显拉伸。其余静态间距、边缘微像素保留为提示。

Studio 展示基础检查，完整循环看过后点“人工验收通过并导出”。脚本交付先核诊断和原图对照，再写 `user-signoff.json`：`decision="pass"` 加验收原话或验收人；调用 `export_default_package(..., preview_fps=30, user_signoff_path=...)`。底层导出器仅核文件、签收及已有基础报告，不重跑四项。交付 ZIP、深浅底 MP4、单文件 HTML；透明网页视频按需用 `export_transparent_web`。

## 红线

- 不做二、三级递归拆子层；新图一次粗拆，真穿帮才局部补一次。
- 不用静态 6px 距离或零星半透明边缘锁死预览与已签收导出。
- 不重新启用冻结的 `canvas_repair_workflow.py` / `connection_diagnostics.py` 作为主链门禁。
- 不写角色名、文件名或哈希特判；代表图只是验证样本。
