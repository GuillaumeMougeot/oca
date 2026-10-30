# oca: Open Coding Agent

A Claude Code–like coding agent built only from open-source software and open-weight models. OpenCode alone needs a hosted model API; oca adds the model servers and wires them in:
- [OpenCode](https://opencode.ai) as the agent,
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for a local model,
- [vLLM](https://github.com/vllm-project/vllm) on [UCloud](https://cloud.sdu.dk) B200s, through [ucloud-api](https://github.com/GuillaumeMougeot/ucloud-api), shared by a project's members,
- a router that serves OpenCode one model, `oca/auto`, from the best live server, so a session carries on when that changes.

```bash
curl -LsSf https://raw.githubusercontent.com/GuillaumeMougeot/oca/main/install.sh | sh
oca            # first run downloads the local model, starts it, opens OpenCode
oca cluster up # optional: a UCloud B200 server; `oca` switches to it once it answers
oca web        # OpenCode in a browser, e.g. on a phone
```

How to use it: [docs/guide.md](docs/guide.md). Status: [ROADMAP.md](ROADMAP.md). Why things are the way they are: [docs/journal](docs/journal/README.md).

Development: `uv sync`, then `uv run pytest`, `uv run ruff check .`, `uv run mypy`.
