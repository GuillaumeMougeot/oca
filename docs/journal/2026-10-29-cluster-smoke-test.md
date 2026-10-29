# Cluster smoke test

**Status:** OPEN. Job 12438889 was queued at 16:36: "no machines available".

## Question
Does the whole cluster path work end to end, and how fast is it? The path is:
1. A `terminal-almalinux` batch job on 1×B200.
2. vLLM installed with uv onto the drive, and weights on `/12383018/hf`.
3. An SSH tunnel to `127.0.0.1:8001`.
4. OpenCode.

Model: Qwen3.8-27B-FP8, 31 GB, Apache-2.0. The cap is `cluster_hours = 1`, so at most 1 GPU-hour.

## Hypothesis (written before the job ran)
- The app sees the GPU (`nvidia-smi`), and UCloud's sshd allows `-L` forwarding.
- The first start is ready within 15 min: uv plus the vLLM venv on the drive in about 5 min, the 31 GB download in 2 to 5 min, then load and compile in 3 to 5 min.
- Later starts are faster, because the venv and compile caches persist on the drive.
- Speed: 80 to 120 tok/s single-stream generation without MTP, under 2 s to prefill the 11k-token OpenCode prompt, and the bug-fix task in under 20 s (70 s on local CPU).

## Finding so far
- Full B200s can be fully booked. A queued job costs nothing, but `oca cluster up` blocks until it runs (2 h timeout).
