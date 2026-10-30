# oca: Open Coding Agent

A Claude Code–like coding agent built only from open-source software and open-weight models. OpenCode alone needs a hosted model API; oca adds the model servers and wires them in:
- [OpenCode](https://opencode.ai) as the agent,
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for local models,
- [vLLM](https://github.com/vllm-project/vllm) on [UCloud](https://cloud.sdu.dk) B200s, through [ucloud-api](https://github.com/GuillaumeMougeot/ucloud-api).

```bash
curl -LsSf https://raw.githubusercontent.com/GuillaumeMougeot/oca/main/install.sh | sh
oca            # first run downloads the model, starts it, opens OpenCode
```

| Command | Does |
|---|---|
| `oca` / `oca run -- <opencode args>` | ensure everything is ready, then open OpenCode |
| `oca config device cpu\|gpu\|auto` | where the local model runs (also `model`, `threads`, `context`, `port`) |
| `oca models`, `oca pull [id]` | list and download models |
| `oca start`, `oca stop`, `oca status`, `oca logs -f` | local server and oca's endpoint |
| `oca uninstall [--keep-models] [--purge]` | remove everything oca installed, including the `oca` command |

## UCloud (optional)

```bash
oca ucloud login                        # once: paste the refresh token (see ucloud-api)
oca ucloud files drives                 # pick a project drive for weights and job files
oca config cluster_drive /<drive id>
oca cluster pull deepseek-v4.1-flash    # big weights: cheap CPU job, no GPU-hours
oca cluster up                          # B200 job + SSH tunnel; `oca` now uses it
oca cluster status | logs -f | extend 2 | down
```

Jobs stop on their own after `cluster_idle` minutes without requests (default 60).

OpenCode sees one model, `oca/auto`, at `http://127.0.0.1:8080/v1`: the cluster model while it answers, else the local one. A session carries on across the switch, mid-task or when resumed.

Members of one UCloud project share servers. With the same `cluster_drive`, `oca cluster up` joins a running server instead of starting a second one, and `oca cluster down` then only leaves it. To join, a member needs:
1. `oca ucloud login --project <project id>`.
2. An SSH key registered before the server started: `oca ucloud ssh-keys add ~/.ssh/id_ed25519.pub --title $(hostname)`.
3. `oca config cluster_drive /<drive id>`, then `oca cluster up`.

Data (models, binaries, logs) goes in `$OCA_DATA`. The default is `/data/$USER/oca` if that disk exists, otherwise `~/.local/share/oca`.

Status: [ROADMAP.md](ROADMAP.md). Why things are the way they are: [docs/journal](docs/journal/README.md).

Development: `uv sync`, then `uv run pytest`, `uv run ruff check .`, `uv run mypy`.
