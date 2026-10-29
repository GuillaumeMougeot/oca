# oca: Open Coding Agent

A Claude Code–like coding agent built only from open-source software and open-weight models. oca installs and wires up:
- [OpenCode](https://opencode.ai) as the agent,
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for local models,
- [vLLM](https://github.com/vllm-project/vllm) on [UCloud](https://cloud.sdu.dk) GPUs (planned).

```bash
curl -LsSf https://raw.githubusercontent.com/GuillaumeMougeot/oca/main/install.sh | sh
oca            # first run downloads the model, starts it, opens OpenCode
```

| Command | Does |
|---|---|
| `oca` / `oca run -- <opencode args>` | ensure everything is ready, then open OpenCode |
| `oca config device cpu\|gpu\|auto` | where the local model runs (also `model`, `threads`, `context`, `port`) |
| `oca models`, `oca pull [id]` | list and download models |
| `oca start`, `oca stop`, `oca status`, `oca logs -f` | local server |

Data (models, binaries, logs) goes in `$OCA_DATA`. The default is `/data/$USER/oca` if that disk exists, otherwise `~/.local/share/oca`.

Status: [ROADMAP.md](ROADMAP.md). Why things are the way they are: [docs/journal](docs/journal/README.md).

Development: `uv sync`, then `uv run pytest`, `uv run ruff check .`, `uv run mypy`.
