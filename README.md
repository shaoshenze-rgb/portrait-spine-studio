# Portrait Spine Studio

每轮开工、恢复或派工前，先读[项目 Skill 的主目标](.agents/skills/portrait-spine-idle/SKILL.md#主目标每轮开工先读)，确认当前动作推进 Canvas→Spine 主链。角色仅作回归样本，不能滑向逐角色手工修图。

在 Cindy 项目对话中实际看图，使用项目内 Skill 和脚本交接完整模块规划，再在本机工具页检查绑定、动画和质量。无需安装连接插件。

    python3 bin/start-studio.py --port 8765 --no-open

启动后访问 http://127.0.0.1:8765/；该地址指浏览器所在电脑。依赖 Python 3.9+、Pillow、numpy，视频导出另需已有 ffmpeg，没有 Node 构建步骤。

当前生产入口要求完整模块 V2：头身、自然肢段、手和硬质器具保形；软模块保持完整归属并有限轻弯；接触、遮挡和输入缺陷显式声明。旧 V1 关键点或大主体掩码不能自动转成合格模块。

    python3 scripts/portrait_task.py prepare --image <透明立绘路径>

默认依次停在等待对话分析、绑定确认。生成后还需机器检查和同版本的五项视觉观察，全部通过才可导出。机器通过、模块mask没有改派、播放过8秒都不能单独证明成品合格。

当前验收状态和证据统一见[验收状态](docs/acceptance-summary.md)。2026-11-25正在接通Canvas拆层/补全主链：真实返回材质已完成物化，新的模块规划、实际分件/骨骼/动画和完整循环验收尚未完成，不能以历史单角色签收替代通用流程通过。操作见[使用指南](docs/usage.md)，检查方法见[项目Skill](.agents/skills/portrait-spine-idle/SKILL.md)。本轮未提交、推送或发布。
