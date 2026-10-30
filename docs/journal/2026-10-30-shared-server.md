# Shared server

**Status:** OPEN. Works with a simulated second member; still to try with a real second UCloud account.

## Question
Several members of a UCloud project want DeepSeek at the same time. Can one server serve them all, and is it worth it?

## Findings (2026-10-30, on the DeepSeek pull job)
- The job's `authorized_keys` held 12 keys, but this account has 4 registered. UCloud installs other members' keys too, so any member with a registered key can open the same tunnel.
- oca's tunnel code opened in 1.1 s and reached a test server inside the job.
- vLLM batches concurrent requests. Its prefix cache is shared, so the part of OpenCode's 11k-token prompt that is identical for everyone is computed once.
- In an agent session, the GPU sits idle while the person reads and tools run locally, so one server covers several people.

## Design
1. Whoever starts the server lists it at `<drive>/oca/servers/<model>.json` (job id, tag, SSH endpoint), and the job removes the listing when it exits.
2. Another member's `oca cluster up` reads that file and only opens a tunnel; no second job.
3. `oca cluster down` from a non-starter closes only their own tunnel. Only the starter can extend the job.
4. The idle stop already reads the server's counters, so it waits for every user.
5. `--max-num-seqs 8` caps concurrent requests; raise it if requests queue.

## Caveats
- The trust boundary is the project: SSH access gives a shell in the job, not just the API.
- A long prompt from one user slows the others' generation briefly; chunked prefill (4,096-token steps) bounds that.
- A key registered after the job started is not in it: that member waits for the next server.
- Two members starting the same model while B200s are queued get two jobs: a job is listed only once it runs.

## Test (2026-10-30)
Qwen3.8-27B on 1×B200. A second oca state on the same machine played the member, on the same UCloud account.
- The starter's oca listed the server as soon as the job ran.
- The member's `oca cluster up` joined in seconds with its own tunnel. `status` shows "joined", and `extend` is refused.
- The member's `down` left the server running (health 200). The starter's `down` stopped the job and removed the listing.

| Requests at once | Qwen 1×B200, per request | Qwen total | DeepSeek 2×B200, per request | DeepSeek total |
|---|---|---|---|---|
| 1 | 108 tok/s | 108 | 331 tok/s | 331 |
| 4 | 104 | 416 | 260 | 1,040 |
| 8 | 98 | 787 | 227 | 1,815 |

Not yet tried: another UCloud account reading the listing and the job.
