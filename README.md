# oca: Open Coding Agent

A Claude Code–like coding agent built only from open-source software and open-weight models. OpenCode alone needs a hosted model API; oca adds the model servers and wires them in:
- [OpenCode](https://opencode.ai) as the agent,
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for a local model,
- [vLLM](https://github.com/vllm-project/vllm) on [UCloud](https://cloud.sdu.dk) B200s, through [ucloud-api](https://github.com/GuillaumeMougeot/ucloud-api), shared by a project's members,
- any machine you reach over SSH, such as a DGX Spark, set up for you or for all its users,
- a router that gives OpenCode one model per server and, when one stops, asks whether to restart it or carry on with another.

```bash
curl -LsSf https://raw.githubusercontent.com/GuillaumeMougeot/oca/main/install.sh | sh
oca                          # first run asks which servers to use, then opens OpenCode
oca remote add spark you@spark   # optional: a machine over SSH; or `ucloud`, or a URL
oca -r spark                 # a session on that server; `oca remote default spark` for all
oca web                      # OpenCode in a browser, e.g. on a phone
```

How to use it: [docs/guide.md](docs/guide.md), also at [guillaumemougeot.github.io/oca](https://guillaumemougeot.github.io/oca/). Status: [ROADMAP.md](ROADMAP.md). Why things are the way they are: [docs/journal](docs/journal/README.md).

**Why oca:** tools such as `ollama launch` connect a coding agent to a model on your own computer. oca also runs open models on the GPUs a team can reach, such as a UCloud allocation or a DGX Spark, shares those servers between colleagues, and, when one stops, offers to restart it or carry the session over. Your code and prompts stay on machines you control.

Development: `uv sync`, then `uv run pytest`, `uv run ruff check .`, `uv run mypy`.
