# AGENTS.md

## Dev environment tips

- Read `PLAN.md` before changing code:
	- This is a **VRChat Official OSC** **MCP Server (Python)** project.
	- Non-negotiable constraints:
		- Input reset semantics: `/input/*` button/axis must auto-reset (`finally` + timeout).
		- Axis values must be clamped to $[-1, 1]$; `duration_ms` is required and capped.
		- Chatbox must be trimmed to **144 chars / 9 lines**.
		- Default transport is `stdio` (SSE is optional).
		- No OSCQuery.

- The “context triple” (avoid guess-fixes):
	- Find the relevant modules/functions: CLI entry, MCP tool registration, OSC send/receive, domain adapter, safety valves, test fixtures.
	- Write down expected behavior and edges: success path, invalid params, timeouts/rate limits, VRChat OSC disabled.
	- Confirm assumptions by reading existing code and/or writing a test first.

- Use `uv` only (no pip/poetry):
	- Python version: see `pyproject.toml` (currently `>=3.12`).
	- Common ops:
		- Install/sync deps: `uv sync`
		- Run locally: `uv run python main.py`
		- Add dependency: `uv add <package>`
		- Add dev dependency: `uv add --dev <package>`

- Default “safe posture” for domain logic:
	- Put safety valves in the domain adapter (not as “please be careful” in MCP tools).
	- Anything that can persist must have auto-revert/auto-reset semantics.
	- Prefer structured logging: one tool call = one trace_id; include osc_address/value, duration_ms, and a result/error code.

## Testing instructions

- Primary goal: verify **reset semantics / clamping / rate limiting / trimming** without requiring a live VRChat instance.
- Recommended layers:
	- Unit: in-memory MCP session (validate parameters, error codes, safety rejects).
	- Integration: local UDP dummy server (assert OSC message sequences), VRChat not required.

- Minimum cases to add when shipping new behavior:
	- `input.tap(Jump)`: must observe `1` then `0`.
	- `input.axis(Vertical, value, duration_ms=...)`: must reset to `0` after duration; value must be clamped.
	- `chat.send(text)`: trimmed to 144 chars / 9 lines.
	- When rate-limited: return an actionable error (include suggested wait time).

- Run tests:
	- `uv run pytest`

## PR instructions

- Commit messages must follow Angular convention (type/scope/subject):
	- Suggested scopes: `server` / `mcp` / `osc` / `domain` / `config` / `tests` / `docs` / `chore`
	- Examples:
		- `feat(mcp): add vrc_chatbox_send tool`
		- `fix(domain): enforce axis auto-reset timeout`
		- `test(osc): assert tap sends 1 then 0`

- Keep PRs small and readable:
	- One intent per PR; split aggressively when needed.
	- Description should answer: what changed, why, how verified (command + key scenarios).
	- If touching Windows paths / VRChat LocalLow, document path assumptions and troubleshooting notes.
