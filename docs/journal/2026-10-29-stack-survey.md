# Stack survey

**Status:** RESOLVED. OpenCode with llama.cpp locally and vLLM on UCloud; choices are in ROADMAP.

## Question
Which open-source pieces give a Claude Code equivalent? Which of the names the owner had heard are needed?

## Findings (2026-10-29)

**Host and cluster**
- Host: Ultra 9 285K, 125 GB RAM, RTX 5090 32 GB, Ubuntu 24.04. Docker and Node are not installed.
- The GPU is shared: another user has about 6.4 GB in use at 99% utilization.
- UCloud: 9.1k B200 GPU-hours, about 1.3 TB storage left.
- The `/models` drive (`/12383018`) is empty.
- SSH-capable apps: `terminal-almalinux`, `ubuntu-xfce`, `coder`. `pytorch-te` has no SSH. There is no vLLM or Ollama app.

**The names**
- Kept: OpenCode (agent), vLLM (serving), Qwen3.8 (a real model family, Aug 2026).
- Not needed:
  - LiteLLM: OpenCode is already multi-provider.
  - n8n: workflow automation, and fair-code rather than open source.
  - OpenClaw and Hermes: personal assistants, not coding-first.
  - Pi: minimal; keep as a fallback.

**Models.** Checked against the Hugging Face API, because search results disagreed with each other.

| Model | Params | Size | License |
|---|---|---|---|
| Qwen3.8-27B | 27.8B | – | Apache-2.0 |
| Qwen3.8-Flash-Next | 180B | FP8 186 GB, NVFP4 133 GB | Qwen Community License |
| GLM-5.3-Flash | 321B | – | MIT |
| DeepSeek-V4.1-Flash | – | 510 GB | MIT |
| Kimi-K3 | 2.8T | – | custom (too big) |

**Engine.** llama.cpp now ships prebuilt Ubuntu CUDA and CPU tarballs, so nothing needs to be compiled.

## Owner decisions
- Use the `oca` repo; the old `code-agent` repo is abandoned.
- The cluster model must fit on 1×B200. That rules out GLM-5.3-Flash and selects Qwen3.8-Flash-Next NVFP4.
- CPU option for local inference while the GPU is busy.
