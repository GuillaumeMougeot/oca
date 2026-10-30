# Shared server

**Status:** OPEN. Feasible without new access setup: UCloud already lets project members SSH into each other's SSH-enabled jobs. The design below is proposed, not built.

## Question
Several members of a UCloud project want DeepSeek at the same time. Can one server serve them all, and is it worth it?

## Findings (2026-10-30, on the DeepSeek pull job)
- The job's `authorized_keys` held 12 keys, but this account has 4 registered. UCloud installs other members' keys too, so any member with a registered key can open the same tunnel.
- oca's tunnel code opened in 1.1 s and reached a test server inside the job.
- vLLM batches concurrent requests. Its prefix cache is shared, so OpenCode's identical 11k-token system prompt is computed once for everyone.
- In an agent session, the GPU sits idle while the person reads and tools run locally, so one server covers several people.

## Design (proposed)
1. Whoever starts the server writes `<drive>/oca/server.json` with the job id, model and SSH endpoint.
2. Another member's `oca cluster up` reads that file and only opens a tunnel; no second job.
3. `oca cluster down` from a non-starter closes only their own tunnel.
4. The idle stop already reads the server's counters, so it waits for every user.
5. `--max-num-seqs 8` caps concurrent requests; raise it if requests queue.

## Caveats
- The trust boundary is the project: SSH access gives a shell in the job, not just the API.
- A long prompt from one user slows the others' generation briefly; chunked prefill (4,096-token steps) bounds that.
