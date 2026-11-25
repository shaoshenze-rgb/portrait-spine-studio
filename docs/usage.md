# Portrait Spine Studio 使用指南

当前主链是“一次 Canvas 顶层粗拆 → 自动对齐、净化与原图补底 → 立即预览 → 局部微调 → 人工签收导出”。[项目 Skill](../.agents/skills/portrait-spine-idle/SKILL.md)规定四步 SOP，[验收状态](acceptance-summary.md)区分代表图通过与待验证内容。

## 启动和准备

在项目根执行，使用已有 Python 3.9+、Pillow、numpy 和 ffmpeg；不自动安装依赖。

```bash
python3 bin/start-studio.py --port 8765 --no-open
python3 scripts/portrait_task.py prepare --image <透明原图.png> --preset agile_full
```

服务命令保持运行，prepare 在另一终端执行并返回 job_id。浏览器地址为 http://127.0.0.1:8765/，指运行浏览器那台电脑。已有透明图直接复用；带背景图先取得获授权的真实去底原件。重启不会恢复旧内存任务，保留旧文件后重新 prepare，不伪造恢复。

## Canvas交接主链

由当前 Cindy 对话调用 Canvas，脚本不代为提交付费任务。先实际读通作品库 original 原件及当前任务原生映射；unauth_cache / unavailable 时停止生成，记录精确字段，待读取恢复。刷新面板或重启 Cindy 不是读取已恢复的证明。

每张新图默认只对原图做一次顶层 split-layers，目标 5–10 个完整大块。头、完整四肢、躯干、武器与软饰件按结构分开；特效独立，武器杆身保持完整。禁止二、三级递归拆子层。下载层原件、核 SHA，保存真实回执，再对齐：

```bash
python3 scripts/canvas_handoff.py import-layers --receipt <回执.json> --source <本次Canvas输入.png> --working-image <工作原图.png> --assets-dir <原件目录> --out-dir <新对齐目录>
```

成功产生 aligned PNG、变换、对照图和 import-report；附带 cleaned 候选及清理指标，原 aligned 层不覆盖。`needs_canvas_registration` 表示没有可靠位置，不猜坐标。保留主原图/工作图关系；主链不使用历史 `--parent-report` 递归拆层入口。

### 对齐层变成标准 parts

当前对话看图后准备两个 JSON：

- 材质清单：`version=1`、`material_mode=canvas_layers`、原图路径/SHA、画布尺寸、pieces。每 piece 提供 id、module_id、kind（rigid/soft）、parent、像素 pivot、draw_order、全画布 aligned RGBA 路径/SHA 与真实 provenance（scene/node/task、源/结果资产 SHA、相似变换、operation）。
- 骨架规划：pose_type、anatomy、modules、connections、occlusions、uncertainties。模块的 parent/pivot/type 与材质对应；连接用真实关节点。grounded 全身需两段刚体腿链及独立静态 foot 锚，airborne 采用浮空动作。规划不是根据文件名自动猜人体。

字段以[材质适配器](../src/pipeline/canvas_layer_parts.py)和[绑定编译器](../src/pipeline/canvas_layer_rig.py)为准。保持完整独立 RGBA，不重新用 SAM 将它们切成互斥碎片。已有标准 parts 可跳过物化；首次从清单物化：

```python
import json
from src.pipeline.canvas_layer_parts import materialize_canvas_layer_parts
manifest = json.load(open(".studio-work/task/material.json", encoding="utf-8"))
materialize_canvas_layer_parts(manifest, ".studio-work/task/parts", evidence_root=".studio-work")
```

## 净化、补底和预览

输入 paths 与新输出目录必须都在 evidence-root 内；输出已存在时换新目录，不覆盖旧视频。

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
- `dynamic_checks` 抽样 16 时刻查新封闭透明洞及头—手、躯干—手最大相对摆角。开放裂缝可能不形成洞，必须看原速完整循环。

## Studio 接入与微调

上传同一原图后，在页面“接入已有 Canvas 图层”填写 parts 目录和规划 JSON，净化后直接预览；也可用 CLI：

```bash
python3 scripts/portrait_task.py apply-canvas --job <job_id> --parts-dir <parts目录> --plan <plan.json> --cleanup
```

本机 API：

| POST 路径 | 请求内容 | 结果 |
| --- | --- | --- |
| `/api/jobs/<id>/apply-canvas-layers` | `parts_dir`、`plan` 为本机路径，`cleanup: true`，可带 tweaks | 新八秒预览；支持 grounded/airborne |
| `/api/jobs/<id>/layer-tweaks` | 与上述微调 JSON 相同的 layers/modules | 从基础输入重建新版本，不累积破坏原件 |
| `/api/jobs/<id>/user-signoff` | `{"decision":"pass","reviewer":"实际验收人"}` | 四项基础检查通过后，写签收并提供 ZIP/MP4/HTML 下载 |

页面支持前后移层、单层显隐、单模块滑块；调整后点击应用更新。优先修层序和目标摆幅，再处理真实露缝。始终被挡住的脖子等不用补。只有动态真穿帮且本地补底/垫边不足，才在有效授权内对单一局部最多补一次 Canvas；结果未知不重提。

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

## 连接诊断与返修编排（历史遗留／当前主链停用）

`canvas_repair_workflow.py`、`connection_diagnostics.py` 冻结保留；`canvas_handoff.py workflow`、SAM 分割、旧 apply-analysis/逐模块审计路径仅供历史维护，不作为当前 Canvas 预览与签收的前置条件。不要为旧机器 pending 重启多轮审查。

历史合同见[完整模块合同](contracts/complete-modules.md)，仅按实际维护需要读取；它不覆盖当前[四步 SOP](../.agents/skills/portrait-spine-idle/SKILL.md)。现状和旧证据见[验收状态](acceptance-summary.md)，缺陷处理见[局部返修](../.agents/skills/portrait-spine-idle/references/rerun-and-troubleshooting.md)。
