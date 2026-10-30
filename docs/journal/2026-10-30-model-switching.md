# Model switching

**Status:** RESOLVED. OpenCode sees one model, which oca's router serves from the best live server. One session moved from the cluster to the local model and back with no user action.

## Question
Can a task continue on another model, as in Claude Code, when the server it started on stops?

## Finding (2026-10-30)
- OpenCode stores sessions on the user's machine, so they outlive any server, whoever started it.
- It resumes a session with the session's last model. When that server is down, oca's config no longer lists the model, and OpenCode fails with `ProviderModelNotFoundError: Model not found: oca-cluster/qwen3.8-27b-fp8`.
- Naming another model works. `opencode run --continue -m oca-local/qwen3.6-35b-a3b` resumed a cluster session on the local CPU model, which recalled the fix it had not made itself. In the TUI, `/models` does the same.

## Design (built 2026-10-30)
- OpenCode's config has one provider and model, `oca/auto`, at `127.0.0.1:<port>`. A session never names a server, so it always resumes.
- `oca.router` listens there. Each request goes to the UCloud server while its health check passes (cached 5 s), else to llama-server, now on `<port>+1` and started on demand. The `model` field is rewritten for the backend.
- A refused connection to the cluster falls back to local within the same request. A stream cut mid-way ends that response; OpenCode retries it, and the retry goes local.
- Every 60 s, while a job is known but not answering, the router runs `cluster.check`: it opens the tunnel once a queued job runs, and forgets ended jobs.
- The context limit is the local model's, so a session can always fall back. The CPU model now runs its full 262k context, like both cluster models: only 10 of its 41 layers keep a KV cache, and llama-server's RSS grew from 36.1 to 40.5 GB.
- Rejected: LiteLLM's proxy. It has fallbacks, but not starting llama-server or re-opening tunnels, and it is a large dependency for about 100 lines of routing.

## Test
- Offline: two fake servers; the "cluster" stops between two requests of one session, and the second request is served locally (`tests/test_oca.py`).
- Local stack: OpenCode through the router fixed the bug-fix task on the CPU model (144 s cold).

**Hypothesis for DeepSeek** (written before testing): the switch is invisible to OpenCode. The first request after a switch re-reads the session on the new server: about 1 s on DeepSeek, 1 to 2 min on the local CPU for a 10k-token session.

**Live test** (Qwen3.8-Flash-Next on 1×B200; a member state with its own tunnel and router):
1. A session fixed the bug on Flash-Next.
2. Its tunnel was killed. The session's next request went to the local CPU model, which described the fix it had not made, in 20 s.
3. The router re-opened the tunnel 41 s later.
4. The session's next request went back to Flash-Next: it added a test, and pytest passed, in 2 s.
