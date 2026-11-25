# 立绘视觉连接

**未采用的实验方案。** 2026-11-22 用户已改用项目 Skill、脚本和工具页的对话交接流程，不安装本插件。本目录源码与旧包仅保留作为实验记录；当前入口以 [项目使用指南](../../docs/usage.md) 为准，下面的安装与接口说明不是当前执行步骤。

本地 Cindy 插件源码，面向 Portrait Spine Studio。保留 Python 的确定性分件与预览，通过 Cindy 可见任务分析本次附带图片，返回结构化关节与实体区域。

## 启动

由 Cindy Forge 校验、安装后，在当前已授权任务中调用 `start_bridge`，参数 `projectRoot` 为本机 Portrait Spine Studio 仓库绝对路径。宿主须支持普通任务图片附件、从当前工具调用继承模型和显式 `ask` 权限。插件不更改账号、全局或任务模型设置，不使用文本 oneshot 冒充视觉。

启动会创建或复用一条独立可见的分析任务。工作目录下 `.studio-work/vision-bridge/connection.json` 保存本机随机端口与认证令牌，权限为 0600；只供 Python 读取，禁止回传前端或记录令牌。插件停用、Host 退出或 Worker 结束时服务随之结束。

## 本机接口

- `POST /v1/jobs`：Bearer 认证；JSON 包含 `jobId`、`inputSha256`、`width`、`height`、`imageBase64`，可选 `preset`。仅接受预处理 PNG，最大 10 MiB。相同作业 ID 与输入幂等，输入变化返回 409。
- `GET /v1/jobs/<jobId>`：读取本次作业状态、绑定信息与最终结果。
- `GET /v1/health`：认证后的连接状态。

只监听 `127.0.0.1`，拒绝浏览器 Origin 和不匹配 Host。令牌不属于 Cindy 账号凭证；插件不读取宿主凭证、内部数据库或私有推理接口。

结果只取本次 `runId` 的终态 `outputMessageId`，并绑定输入摘要与尺寸。超时、派发响应未知或重启中的未完成作业标为 `outcome_unknown`，不自动换 ID 或再次派发。重试只回读原结果；低置信度或不完整语义由 Python 校验后进入人工检查。

## 验证边界

```bash
node --test plugins/cindy-vision-bridge/tests/bridge.test.cjs
```

上述测试从仓库根目录执行，覆盖本机协议与模拟公开 tasks 接口，不是实际模型验收。安装后还必须核对真实 Host capabilities、任务模型与执行来源，并用真实图片跑通附件到最终 JSON。宿主安装版记录为 0.1.143；其精确源码 commit 当前无法取得，本实现依随包公开契约并对未知回执结构停止。

实体边界是视觉模型建议，不保证像素级正确。宠物、尾巴、武器与人体归属须由输入绑定、确定性 mask 校验和真实预览共同验收。
