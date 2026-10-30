# Journal

One file per question: `YYYY-MM-DD-topic.md`, dated by when it was opened. Each file has a `**Status:**` line set to OPEN or RESOLVED.

| Entry | Status | Summary |
|---|---|---|
| [2026-10-29-stack-survey](2026-10-29-stack-survey.md) | RESOLVED | Which open tools and models; the plan |
| [2026-10-29-local-mvp](2026-10-29-local-mvp.md) | RESOLVED | Phase 1 working on CPU: 12 tok/s, a bug fixed in 1.5 to 3.5 min |
| [2026-10-29-cluster-model](2026-10-29-cluster-model.md) | RESOLVED | DeepSeek-V4.1-Flash on 2×B200 (vLLM): 256 to 420 tok/s, bug-fix task in under 3 s |
| [2026-10-29-speed](2026-10-29-speed.md) | RESOLVED | The model is the bottleneck, not Python. Bug-fix task on CPU: 201 s down to 38 s (cache, MTP, no title agent) |
| [2026-10-29-cluster-smoke-test](2026-10-29-cluster-smoke-test.md) | RESOLVED | Serves on a B200 at 111 tok/s, ready 4.5 min after start; tunnel opens in 1.1 s |
| [2026-10-30-shared-server](2026-10-30-shared-server.md) | OPEN | One server per project: works with a simulated second member; 8 requests at once keep 70 to 90% of solo speed |
| [2026-10-30-model-switching](2026-10-30-model-switching.md) | OPEN | Sessions outlive servers, but resuming needs the session's model up or chosen by hand |
