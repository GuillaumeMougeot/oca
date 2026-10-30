# Changelog

## v0.1.0 (2026-10-30)

First release.
- **Install:** one line on Linux or macOS (WSL works): uv, oca, OpenCode 1.19.5 and llama.cpp b11982.
- **Local model:** llama.cpp on CPU or GPU. Qwen3.6-35B-A3B with MTP drafting at its full 262k context, or Qwen3.8-27B.
- **UCloud B200 servers** with vLLM 0.32:
  - Qwen3.8-Flash-Next on one B200 (default): ready in about 5 min, 350 to 390 tok/s.
  - DeepSeek-V4.1-Flash on two B200s: ready in about 25 min, 256 to 420 tok/s.
  - Project members share servers, which stop themselves when idle.
- **`oca/auto`:** one model for OpenCode, served by the best live server, so a session carries on when the model behind it changes.
- **Effort:** `ctrl+t` cycles `none`, `low`, `medium`, `high` and `max`, translated for each model.
- **Phone:** `oca web` serves OpenCode's web interface on localhost, with a password, for an SSH tunnel.
- **Guide:** [docs/guide.md](docs/guide.md).
