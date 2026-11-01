# New models, late October 2026

**Status:** OPEN. Nanbeige4.3-33B and Apex-35B wait for our eval.

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

## Next
- 5090, when free: Nanbeige4.3-33B and Apex-35B against Qwen3.8-27B on the history suite. Expectation: Nanbeige solves more and runs about 15% slower; Apex is no better than its base.
- 1×B200, after asking: Nanbeige4.3-33B against Flash-Next.

Sources: [Nanbeige 4.3](https://www.eesel.ai/blog/nanbeige-4-3), [LFM3](https://www.marktechpost.com/2026/10/27/liquid-ai-lfm3-open-weight-2b-multimodal-models/), [Apex-35B](https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0), [Clover](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview), [Clover claims](https://eu.36kr.com/en/p/4045119600399241), [Spark-X2.6](https://aiweekly.co/alerts/sparkllm-ships-spark-x26-on-device-models-in-17b-and-4b), [SenseNova-Loop](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview), [gpt-oss on a Spark](https://lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark).
