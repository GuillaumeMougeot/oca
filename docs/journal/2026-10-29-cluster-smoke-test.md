# Cluster smoke test

**Status:** OPEN. Serving works on a B200 at 111 tok/s, and the job is ready 4.5 min after it starts. Only the tunnel is still untested: it needs this machine's SSH key registered on UCloud, which the owner must do.

## Question
Does the whole cluster path work end to end, and how fast is it? The path is:
1. A `terminal-almalinux` batch job on 1×B200.
2. vLLM installed with uv onto the drive, and weights on `/12383018/hf`.
3. An SSH tunnel to `127.0.0.1:8001`.
4. OpenCode.

Model: Qwen3.8-27B-FP8, 31 GB, Apache-2.0. The cap is `cluster_hours = 1`.

## Hypothesis (written before the job ran)
- The app sees the GPU, and UCloud's sshd allows `-L` forwarding.
- The first start is ready within 15 min. Later starts are faster thanks to the caches on the drive.
- 80 to 120 tok/s single-stream; the bug-fix task in under 20 s.

## Job 12438889
**Queue:** 65 min IN_QUEUE ("no machines available"); about 7 GPU-minutes used.

**Confirmed working:**
- The B200 is visible.
- Internet access from the job works.
- uv installed vLLM 0.32.0 (torch 2.13+cu132, flashinfer 0.7.0.post1) onto the drive; together with the 31 GB weight download, that took under 4 min.
- Loading was fast: weights in 6 s from the drive, model in 20 s (27.6 GiB), `torch.compile` in 49 s, now cached.

**Failed:**
1. **FlashInfer kernels:** FlashInfer JIT-compiles a decode kernel and needs `nvcc`; the app has no CUDA toolkit.
   - Fix: install the matching prebuilt `flashinfer-cubin` and `flashinfer-jit-cache` wheels from flashinfer.ai.
2. **SSH tunnel:** `Permission denied (publickey)`. This workstation's ed25519 key is not among the three keys registered on UCloud (all RSA, from other machines).
   - Fix: the owner runs `oca ucloud ssh-keys add ~/.ssh/id_ed25519.pub --title <name>`. The agent was not allowed to add account access.
   - `oca cluster up` now checks the key before submitting.
3. **Misleading status:** UCloud reported the job as SUCCESS even though vLLM had failed, confirming "read logs, not status".

## Job 12438944
**Queue:** 1.5 min IN_QUEUE; about 5 GPU-minutes used.

**The FlashInfer fix works.** From job start to `oca: ready` took 4.5 min:
- prebuilt FlashInfer wheels installed,
- weights loaded in 3 s,
- `torch.compile` in 0.5 s (cached),
- warm-up and CUDA graphs in 44 s.

**Self-test:** 512 tokens in 4.6 s, about 111 tok/s single-stream, with no speculative decoding. That is within the 80 to 120 tok/s hypothesis and about 5 times the local CPU (20 tok/s with MTP).

**Not yet measured:**
- the bug-fix task through the tunnel,
- Qwen3.8's MTP head in vLLM (the recipe reports 85% acceptance at FP8).
