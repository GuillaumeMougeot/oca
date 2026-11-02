# New models, late October 2026

**Status:** RESOLVED. Neither beats Qwen3.6-35B-A3B on the 5090, and that model also beats our GPU default, Qwen3.8-27B.

## Question
Do the models released in late October 2026 beat our picks on 2×B200, 1×B200, a DGX Spark (128 GB unified memory, 273 GB/s) or an RTX 5090 (32 GB)?

## Findings (2026-11-01)
Scores are the makers' own; none had been reproduced independently.

| Model | What | Verdict |
|---|---|---|
| Nanbeige4.3-33B | Dense 33B (Llama architecture), Apache-2.0, 512k context. DeepSWE 65.5 against 42.5 for Qwen3.8-27B; the makers of the others report 58.7 for Flash-Next and 77.0 for DeepSeek | Test. 5090: Q4 is 20.8 GB, leaving about 80k tokens of 8-bit cache. 1×B200: BF16 is 66 GB. Spark: dense, so about 10 tok/s |
| LFM3 | Liquid AI, six ~2B on-device models, LFM Open License | Too small for an agent |
| Apex-35B-V1.0 | A new startup's fine-tune of Qwen3.6-35B-A3B, our CPU model; Apache-2.0 | Drop-in, so cheap to test, but the claims look inflated: SWE-bench Pro 92.4 against 72.4 for its base. Keeps our CPU speed only with MTP layers |
| Groot | Presumably NVIDIA's GR00T, a robot-control model | Not a language model |
| Clover-35B-A3B-Preview | Block-diffusion version of Qwen3.6-35B-A3B; Apache-2.0, 128k context; claims 5 to 10× faster | Wait: preview, its own SGLang and llama.cpp code, tool calls unproven |
| Spark-X2.6-1.7B | iFLYTEK, on-device (unrelated to the DGX Spark) | Too small |
| SenseNova-Loop-9B-Preview | SenseTime, looped 9B (14.6B of compute per token), text and image | 9B class; stock llama.cpp cannot load it |

On a Spark, decode speed is roughly 273 GB/s divided by the bytes read per token, so it needs few active parameters. gpt-oss-120b (5.1B active) measures about 50 tok/s; a 35B-A3B at 8 bits should be similar, and dense 27 to 33B models about 10. Flash-Next (183 GB) does not fit.

## Harder tasks
Flash-Next solved all 9 history tasks, so the suite is too easy at the top; 30B-class models had never run it. Five multi-file commits from 2026-10-30 join it (14 tasks), each failing before its commit and passing after (`--check`). The largest, f157bb8, puts keys on every server across 8 files.

## Eval (5090, 2026-11-01)
The 14 history tasks at high effort, 30 min each. The GPU was shared with another job (the user agreed, if 2 GiB or more stayed free), so every model ran at 40k context with an 8-bit cache, and without MTP drafting so that speeds compare. Expectation: Nanbeige solves more than Qwen3.8-27B and runs about 15% slower; Apex is no better than its base.

| Model | Solved | Median per task | Decode | Tokens generated |
|---|---|---|---|---|
| Qwen3.6-35B-A3B (Q4_K_XL) | **12/14** | 89 s | 112 tok/s | 0.39 M |
| Apex-35B (Q4_K_M) | 9/14 | 529 s | 122 tok/s | 1.22 M |
| Qwen3.8-27B (Q4_K_XL) | 9/14 | 1738 s | 41 tok/s | 0.75 M |
| Nanbeige4.3-33B (IQ4_XS) | 5/14 | 1800 s | 33 tok/s | 0.65 M |

- Apex: as expected, 3 fewer than its base with 3× the tokens. The card's scores do not hold here.
- Nanbeige: wrong on quality, close on speed (20% slower than Qwen3.8-27B). 10 of its 14 tasks hit the time limit.
- 40k was tight: OpenCode compacted the conversation in every task of the three slower models, and in 6 of Qwen3.6's. Nanbeige's three early failures all stopped right after a compaction, replying with a summary instead of working on. Its score says it is unusable on a 5090, not how it does at full context on a B200.
- All four failed 24f112d and f157bb8 (keys on every server, 8 files).
- One run of 14 tasks: a gap of one or two tasks may be noise. The speed gaps are not.

## Decisions
- No catalog entry for Nanbeige or Apex.
- Proposed: Qwen3.6-35B-A3B as the GPU default too, with MTP; one model for both devices. Its full 262k context needs about 30 GB, so the GPU needs a smaller default context.
- Nanbeige on a 1×B200: not worth GPU-hours for now. It would decode slower than Flash-Next, and nothing here suggests it solves more.

Sources: [Nanbeige 4.3](https://www.eesel.ai/blog/nanbeige-4-3), [LFM3](https://www.marktechpost.com/2026/10/27/liquid-ai-lfm3-open-weight-2b-multimodal-models/), [Apex-35B](https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0), [Clover](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview), [Clover claims](https://eu.36kr.com/en/p/4045119600399241), [Spark-X2.6](https://aiweekly.co/alerts/sparkllm-ships-spark-x26-on-device-models-in-17b-and-4b), [SenseNova-Loop](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview), [gpt-oss on a Spark](https://lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark).
