# vrchat-osc-mcp

一个 **VRChat 官方 OSC** 的 **MCP Server（Python）**，目标是“更工程化、更友好”：

- 统一的 MCP 工具接口（FastMCP）
- 可靠的安全阀：输入强制复位、轴限幅、Chatbox 裁剪、全局节流/限流
- 可测试：不依赖真实 VRChat（用本地 UDP dummy server 验证 OSC 序列）

> 约束是硬的（见 `PLAN.md` / `AGENTS.md`）：`/input/*` 必须自动复位；axis clamp 到 [-1,1]；Chatbox 144 字 / 9 行。

## 快速开始（本地 stdio）

- 安装依赖：使用 uv（推荐）
- 运行：
  - `uv run python main.py`

默认会把 OSC 消息发往 `127.0.0.1:9000`（VRChat 默认 incoming 端口）。

## 配置

支持 YAML：默认探测 `./config.yaml` 或 `./config/config.yaml`。

示例见仓库根目录 `config.yaml`。

CLI 覆盖 YAML：

- `--transport {stdio,sse,http}`
- `--osc-send-ip` / `--osc-send-port`
- `--sse-host` / `--sse-port`
- `--http-host` / `--http-port` / `--http-path`
- `--log-level {DEBUG,INFO,WARNING,ERROR}`
- `--vrchat-osc-root`（覆盖 Windows 默认 LocalLow\VRChat\VRChat\OSC）
- `--avatar-config`（显式指定 avtr_*.json，用于 schema 严格校验；也可用仓库内 sample.json 做开发调试）

## 运行方式（transport）

- `stdio`：默认，适合 Claude Desktop / 本地工具。
- `http`：**推荐**（FastMCP 的 Streamable HTTP），便于多客户端/服务化部署。
- `sse`：兼容旧 SSE 客户端（不推荐作为新项目默认）。

## 暴露的 MCP 工具（MVP-0）

- `vrc_status`
- `vrc_list_inputs`（VRChat 官方 /input 轴/按钮清单与作用说明；`vrc_input_*` 会按此清单校验）
- `vrc_list_parameters`（若已加载 Avatar schema，输出参数能力表）
- `vrc_set_parameter`（MVP-0 默认 allowlist：allowed_parameters 为空时会拒绝所有参数写入）
- `vrc_input_tap`（强制复位）
- `vrc_input_axis`（强制复位 + clamp + duration cap）
- `vrc_chat_send`（144/9 裁剪 + 限流）
- `vrc_chat_typing`

> 命名说明：为兼容 OpenAI 的 tool/function 命名约束，本项目工具名只使用 `[A-Za-z0-9_-]`，不使用点号分组（例如 `vrc.chat.send`）。

> 提示：当 `safety.parameter_policy: strict` 时，`vrc_set_parameter` 会依赖本地 schema（LocalLow 或 `--avatar-config`）来校验参数存在性、类型与可写性，并使用 schema 中的 `input.address` 作为实际发送地址。

## 测试

- `uv run pytest`

这些测试会启动一个本地 UDP dummy server 来断言 OSC 消息序列：

- `tap(Jump)`：必须观测到 `1` 后跟 `0`
- `axis(Vertical, 2.0)`：必须 clamp 到 `1.0`，并在 duration 后发送 `0.0`
