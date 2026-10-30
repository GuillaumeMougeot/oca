# Model switching

**Status:** OPEN. Sessions survive server restarts, but resuming one needs its last model to be up.

## Question
Can a task continue on another model, as in Claude Code, when the server it started on stops?

## Finding (2026-10-30)
- OpenCode stores sessions on the user's machine, so they outlive any server, whoever started it.
- It resumes a session with the session's last model. When that server is down, oca's config no longer lists the model, and OpenCode fails with `ProviderModelNotFoundError: Model not found: oca-cluster/qwen3.8-27b-fp8`.
- Naming another model works. `opencode run --continue -m oca-local/qwen3.6-35b-a3b` resumed a cluster session on the local CPU model, which recalled the fix it had not made itself. In the TUI, `/models` does the same.

## Idea
One stable local endpoint and model name for OpenCode, routed by oca to the best live server. Sessions would then resume, and a task would carry on mid-way, when a server stops. Context limits differ (262k on the cluster, 64k locally), so a long session would need compaction on fallback.
