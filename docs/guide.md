# oca guide

## Install

```bash
curl -LsSf https://raw.githubusercontent.com/GuillaumeMougeot/oca/main/install.sh | sh
```

- **Platform:** Linux or macOS, x64 or arm64; WSL works. On Apple Silicon the local model runs on the GPU.
- **Requirement:** git.
- **What it installs:** uv if missing, the `oca` command, OpenCode and llama.cpp (about 1.3 GB).
- **Where heavy files go:** `$OCA_DATA`, else `/data/$USER/oca` if that exists, else `~/.local/share/oca`.

Run the same line again to update.

## Daily use

- **`oca`**, in a project, opens OpenCode. The first run offers to download the local model, 23 GB for CPU.
- **`oca run -- run "fix the failing test"`** runs one task without the terminal UI.
- **`oca web`** opens OpenCode in a browser; see [Phone](#phone-and-other-machines).

OpenCode sees one model, `oca/auto`: the UCloud model while a server answers, else the local model. A session carries on when the model behind it changes, mid-task or when you resume it later.

**Resume:** `oca -c` reopens the last session in this directory, and `oca -s <id>` reopens a given one. Inside OpenCode, `ctrl+x l` (or `/sessions`) lists sessions.

**Effort:** `ctrl+t` in OpenCode cycles `oca/auto`'s variants: `none`, `low`, `medium`, `high`, `max`. Without a variant, each model uses its default. For one task: `oca run -- run --variant low "…"`.
- `none` turns thinking off: fastest, but weakest.
- Qwen3.8-Flash-Next maps the others to `low`, `medium` and `xhigh` (its default).
- DeepSeek maps them to a budget of 50, 62, 75 (its default) and 100.
- The local model only has thinking on or off.

**Project instructions:** the agent reads `AGENTS.md` from the project and its parent directories, or `CLAUDE.md` where there is none, plus your global `~/.claude/CLAUDE.md`. OpenCode does this itself; checked through oca with a codeword the model answered without opening a file.

**Permissions:** like Claude Code's auto mode.
- Allowed: everything else.
- Asks first: deleting (`rm`), rewriting git history or pushing, reaching other machines (`curl`, `ssh`, web fetch and search), installing outside the project, and touching paths outside the project. Answer *always* to stop being asked for the rest of the session.
- Never allowed: administration, such as `sudo`, `su`, `systemctl` or `--break-system-packages`.

## Local model

| Command | Does |
|---|---|
| `oca config device cpu`, `gpu` or `auto` | where the local model runs |
| `oca models` | list local and UCloud models |
| `oca pull <id>`, then `oca config model <id>` | use another local model |
| `oca status`, `oca logs -f`, `oca stop` | check, follow or stop the local server |

## UCloud

### Once per person

1. **Log in.** Open cloud.sdu.dk in a browser, then DevTools → Application → Cookies, and copy `refreshToken`. Run `oca ucloud login --project <id>`; `oca ucloud projects` lists your projects.
2. **Register an SSH key.** Run `oca ucloud ssh-keys add ~/.ssh/id_ed25519.pub --title $(hostname)`, after `ssh-keygen -t ed25519` if you have no key. A server only accepts keys registered before it started.
3. **Pick the drive.** Run `oca ucloud files drives`, then `oca config cluster_drive /<drive id>`.

Once per project, download the weights: `oca cluster pull [model]`. This runs a CPU job that takes a few minutes and uses no GPU-hours.

### Servers

| Command | Does |
|---|---|
| `oca cluster up [model]` | join the project's running server of that model, or start one after showing its GPU-hour cap |
| `oca cluster down` | stop your server, or leave one you joined |
| `oca cluster status`, `oca cluster logs -f` | state and time left; the job's log |
| `oca cluster extend 2` | add 2 h; starter only |

| Model | B200s | Start | Note |
|---|---|---|---|
| `qwen3.8-flash-next` (default) | 1 | about 5 min | single B200s are usually free |
| `deepseek-v4.1-flash` | 2 | about 25 min | stronger; pairs can queue for hours |
| `qwen3.8-27b-fp8` | 1 | about 5 min | small |

**Switching model:** run `oca cluster up deepseek-v4.1-flash`. It offers to stop or leave the current server. OpenCode works on the local model meanwhile, and moves to the new one once it answers.

### Cost and timeouts

- `cluster_idle`, default 60, is how many minutes without requests a server waits before stopping itself.
- `cluster_hours`, default 8, caps how long a job can run.
- Set either with, for example, `oca config cluster_idle 30`. The change applies to the next server you start.
- Queue time is free. A server shared by several people costs the same as one.

### Sharing

Everyone in the project with the same `cluster_drive` shares servers: `oca cluster up` joins a running server instead of starting another. Project members' SSH keys also give them a shell in the job, so share servers only within a trusted project.

## Phone and other machines

Run `oca web` in a project. It serves OpenCode's web interface on this machine, on localhost only, and prints a password. The user name is `opencode`.

From a phone, forward the port with any SSH app, over the university network or VPN:

```bash
ssh -N -L 4096:127.0.0.1:4096 you@workstation
```

Then open `http://127.0.0.1:4096` on the phone.

- To keep the web server running after you log out: `tmux new -d -s oca-web oca web`.
- To use the terminal UI over SSH instead: `tmux new -As oca oca`.

## Settings

`oca config` shows all settings; `oca config <key> <value>` sets one.

| Key | Default | Meaning |
|---|---|---|
| `device` | auto | cpu, gpu or auto (gpu if `nvidia-smi` exists) |
| `model` | auto | local model id |
| `port` | 8080 | oca's endpoint for OpenCode and other tools; llama-server uses the next port |
| `threads`, `context` | 0 | CPU threads and local context; 0 lets llama.cpp and the model decide |
| `cluster_drive` | none | UCloud drive for weights and jobs |
| `cluster_model` | qwen3.8-flash-next | default for `oca cluster up` and `oca cluster pull` |
| `cluster_hours`, `cluster_idle` | 8, 60 | job cap in hours; idle stop in minutes |
| `cluster_port` | 8001 | local end of the SSH tunnel |

## Troubleshooting

- **"UCloud rejected this machine's SSH key":** register the key (see above). The server then has to be restarted.
- **Job "in queue":** the B200s are busy. Waiting costs nothing, and `oca cluster down` cancels.
- **UCloud says SUCCESS but nothing works:** UCloud reports success for failed jobs, so read `oca cluster logs`.
- **Logs:** `oca logs -f` for the local server, `oca cluster logs` for the job, and `logs/router.log` in the data directory for which model served each request.

## Uninstall

`oca uninstall` removes everything oca installed, including the `oca` command. `--keep-models` keeps the downloaded models; `--purge` also deletes OpenCode's sessions and logs.
