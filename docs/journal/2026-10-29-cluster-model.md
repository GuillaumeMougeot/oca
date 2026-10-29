# Cluster model

**Status:** OPEN. Proposed: DeepSeek-V4.1-Flash-1017 at IQ2_XXS on 1×B200 with llama.cpp. Waiting on the owner and on phase 3.

## Question
What is the strongest open-weight coder that runs on 1×B200 (about 180 GB)?

## Evidence (vendor model cards, 2026-10-29)
DeepSWE v1.1 is the only benchmark that appears on most of the cards. The harnesses differ, so treat this as a rough ranking only.

| Model | Total / active params | DeepSWE | License | Fits 1×B200? |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 | 763B / about 13B (estimated from config) | 77.0 (Opus 5: 74.0 in the same table) | MIT | only as a 2-bit GGUF |
| GLM-5.3-Flash | 321B / 18B | 63.4 | MIT | no (FP8 needs 2×) |
| Qwen3.8-Flash-Next | 125B / 6B | 58.7 | Qwen Community | yes (NVFP4, 133 GB) |
| Qwen3.8-27B | 27B dense | 42.2 | Apache-2.0 | yes |

Kimi-K3 (2.8T) and K2.7-Code (1T, INT4) need 4× or more B200s. K3 is behind on DeepSWE in DeepSeek's table.

## ISTA-DASLab GSQ-RCO GGUF
- IQ2_XXS is 249 GB: a 138 GB backbone plus a 110 GB Engram table. The Engram table is memory-mapped and read one row per token.
- On 1×B200 the backbone sits in VRAM and the Engram table in the node's 288 GB RAM.
- llama.cpp has supported the format since b11870; oca pins b11982.
- The card claims 100% recovery, but only on AIME25 (saturated), GPQA and LCB. Nothing agentic or long-context, and those are where 2-bit quantization hurts most.
- The "12 GB GPU" claim needs a third-party engine (Strata) and about 140 GB of system RAM. The card's own memory section still says "TBD". This PC has 125 GB, so it can't run here.

## Plan
1. Run the phase-3 eval: IQ2_XXS on 1×B200 against the Qwen3.8-Flash-Next fallback.
2. If tool calling degrades, spend a few GPU-hours on the official FP8/FP4 weights on 4×B200 with vLLM to measure what the quantization costs.
