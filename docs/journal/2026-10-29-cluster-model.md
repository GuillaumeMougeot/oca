# Cluster model

**Status:** OPEN. Recommended: DeepSeek-V4.1-Flash-1017 with its original weights on 2×B200 with vLLM. Needs the owner's go-ahead: 510 GB of the roughly 1 TB left, and 2 GPU-hours per hour.

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
- The owner's priority is speed, and coding effort was declared not to matter.
- The original weights on vLLM win on both counts. There is also no quantization doubt and no 4-day-old code path.
- The 2-bit GGUF stays a fallback if GPU-hours become the constraint.

## Next
1. Pull the original weights from a 2-vCPU job: 16 CPU core-hours worst case.
2. Validate on 2×B200 against the Qwen3.8-27B smoke-test baseline.
