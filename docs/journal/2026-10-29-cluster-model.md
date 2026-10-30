# Cluster model

**Status:** RESOLVED. DeepSeek-V4.1-Flash-1017 works on 2×B200 with vLLM: 256 to 420 tok/s single-stream, and the bug-fix task in under 3 s.

## Question
What is the strongest open-weight coder we can serve reliably and fast on UCloud B200s?

## Evidence (vendor model cards, 2026-10-29)
DeepSWE v1.1 is the only benchmark that appears on most of the cards. The harnesses differ, so treat this as a rough ranking only.

| Model | Total / active params | DeepSWE | License |
|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 | 552B backbone plus Engram tables / 8–16B active | 77.0 (Opus 5: 74.0 in the same table) | MIT |
| GLM-5.3-Flash | 321B / 18B | 63.4 | MIT |
| Qwen3.8-Flash-Next | 125B / 6B | 58.7 | Qwen Community |
| Qwen3.8-27B | 27B dense | 42.2 | Apache-2.0 |

## Options for DeepSeek-V4.1-Flash-1017

| | ISTA IQ2_XXS GGUF | Original weights (FP8 dense, MXFP4 experts) |
|---|---|---|
| Size | 249 GB | 510 GB |
| GPUs | 1×B200, llama.cpp | 2×B200, vLLM TP2, Engram tables in host RAM |
| Maturity | llama.cpp support since 2026-10-25 (#30817); tool-call parsing for V4.1's DSML format unverified | vLLM ≥0.31; recipe marks B200 TP2 as verified for single-user agentic use, with `deepseek_v41` tool and reasoning parsers |
| Quality | Card shows 100% recovery on AIME, GPQA and LCB only; nothing agentic | Reference |
| Speed | Unknown; llama.cpp has no sparse-MLA kernels | FlashInfer sparse MLA plus the DSpark drafter (5 tokens per round) |

## Decision rationale
- The priority is speed; coding effort does not matter.
- The original weights on vLLM win on both counts. There is also no quantization doubt and no 4-day-old code path.
- The 2-bit GGUF stays a fallback if GPU-hours become the constraint.

## Pull (2026-10-30)
- 510 GB in 14 min on 2 vCPUs (about 600 MB/s, peaks of 800): under 0.5 core-hours.
- `hf` used all of its 2 vCPUs. More vCPUs would finish sooner for about the same core-hours.
- The last two files are 101.5 GB each, so the end runs on two streams.

## Validation on 2×B200
**Hypothesis** (written before the job ran):
- The first start is ready within 30 min: loading 510 GB from the drive, compiling, capturing CUDA graphs.
- 100 to 250 tok/s single-stream with DSpark drafting, against 111 tok/s for Qwen3.8-27B on 1×B200 without drafting.
- The bug-fix task through the tunnel takes 15 to 40 s, against 38 s on the local CPU.
- 4 concurrent requests give at least 2.5× the throughput of one.

**Result** (2026-10-30):
- The first try failed after loading: DeepGEMM compiles kernels at startup and found no CUDA toolkit. pip had put one in the vLLM venv, so `CUDA_HOME` now points at it, with `nvcc` pinned to the runtime's 13.2 (13.4 fails CCCL's version check). The fix was checked on a 2-vCPU job, then on 1×B200, before the next 2×B200 run.
- Ready 26 min after the job started: weights in 8 min, then 12 min of engine start-up and kernel warm-up. The compile caches stay on the drive for later starts.
- Single-stream: 256 tok/s in the self-test, 330 to 420 tok/s once warm. DSpark accepts 3.5 tokens per step.
- Bug-fix task through the tunnel: 2.6 and 2.8 s, both correct. Qwen3.8-27B on 1×B200 took 5.7 to 7.1 s; the local CPU took 38 s.
- 4 concurrent requests: 3.1× the throughput of one. 8 concurrent: 1,815 tok/s in total, 227 per request.
- Prefill: 120k tokens in 2.1 s.
- Queue: 2×B200 jobs waited 1.5 and 2 h today; a 1×B200 job started within minutes.
- Every hypothesis held, and speed beat it 2 to 10×.
