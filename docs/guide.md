# oca guide

## Install

```bash
curl -LsSf https://raw.githubusercontent.com/GuillaumeMougeot/oca/main/install.sh | sh
```

- **Platform:** Linux or macOS, x64 or arm64; WSL works. On Apple Silicon the local model runs on the GPU.
- **Requirement:** git.
- **What it installs:** uv if missing, the `oca` command, OpenCode and llama.cpp (about 1.3 GB).
- **Where heavy files go:** `$OCA_DATA`, else `/data/$USER/oca` if that exists, else `~/.local/share/oca`.

`oca update` installs the newest release. For the latest unreleased code: `OCA_SOURCE=git+https://github.com/GuillaumeMougeot/oca sh install.sh`.

**Tab completion:** run `oca --install-completion` once, then open a new terminal: Tab completes commands, ids (`oca fl` Tab) and server names.

## Daily use

- **`oca`**, in a project, opens OpenCode on the default server. The first run asks which servers to use besides this machine, and offers to download the local model, 23 GB for CPU.
- **`oca ls`** lists every model on every server, each with an id such as `flash-next@spark` or `qwen3.6@local`; **`oca <id>`** opens one. Part of an id will do (`oca deep`); a word that fits several, such as `oca flash`, gets a numbered menu. See [Servers](#servers).
- **`oca run "fix the failing test"`** runs one task without the terminal UI; `-m <id>` picks the model.
- **`oca web`** opens OpenCode in a browser; see [Phone](#phone-and-other-machines).

OpenCode lists one model per server: `oca/local` and one per remote; `ctrl+x m` switches mid-session.

**When a server is down,** oca never switches by itself. It asks in a menu, in OpenCode (↑↓ and Enter), saying why the server is down: restart it, use another server, or stop. If the restart fails, it says why and asks again. While a server starts, the reply shows its progress. A server used instead serves until the session's own server answers again, and says so in one line. If its context window is smaller, it sees only the latest messages; the session keeps its whole history. Launched while its server is down, `oca` asks the same in the terminal.

**Sessions:** `oca sessions` lists this folder's sessions; `oca sessions rm <id>` deletes one, and `oca sessions clear` all those started in this folder or below. `oca -c` reopens the last session, and `oca -r <id>` a given one, as in Claude Code. Inside OpenCode, `ctrl+x l` (or `/sessions`) lists them. Sessions get titles when a remote answers at launch; the local CPU model skips them, since a title costs 40 to 80 s there.

**Effort:** `ctrl+t` in OpenCode cycles the model's variants: `none`, `low`, `medium`, `high`, `max`. Without a variant, each model uses its default. For one task: `oca run --variant low "…"`.
- `none` turns thinking off: fastest, but weakest.
- Qwen3.8-Flash-Next maps the others to `low`, `medium` and `xhigh` (its default).
- DeepSeek maps them to a budget of 50, 62, 75 (its default) and 100.
- The local model only has thinking on or off.

**Project instructions:** the agent reads `AGENTS.md` from the project and its parent directories, or `CLAUDE.md` where there is none, plus your global `~/.claude/CLAUDE.md`. OpenCode does this itself; checked through oca with a codeword the model answered without opening a file.

**Permissions:** like Claude Code's auto mode.
- Allowed: everything else.
- Asks first: deleting (`rm`), rewriting git history or pushing, reaching other machines (`curl`, `ssh`, web fetch and search), installing outside the project, and touching paths outside the project. Answer *always* to stop being asked for the rest of the session.
- Never allowed: administration, such as `sudo`, `su`, `systemctl` or `--break-system-packages`.
- Unattended runs (`oca run …`) cannot ask, so what would ask is refused instead, and the agent carries on another way.

**Keys:** oca's servers listen on localhost, which every user of the machine can reach, so each one needs a key.
- The router and the local llama-server take your key, in `~/.config/oca/api-key`; OpenCode sends it for you.
- Each UCloud server has its own key, kept in its listing on the project drive, where members find it.
- A server on an SSH machine uses the key of the oca installed there; your oca reads it over SSH.
- Files that hold keys are readable only by you, or by the group of an install for all users.

## Local model

| Command | Does |
|---|---|
| `oca config device cpu`, `gpu` or `auto` | where the local model runs |
| `oca qwen3.8-27b@local` | open another local model; `oca ls` lists those that fit this machine |
| `oca pull <id>` | download one ahead of time; otherwise `oca` asks first |
| `oca status`, `oca logs local -f`, `oca stop local` | check, follow or stop the local server |

## Servers

Besides this machine (`local`), oca uses other servers: UCloud, a machine you reach over SSH such as a DGX Spark, or any OpenAI-compatible server. A server serves one model at a time; each model it can serve is a pair with its own id, such as `deepseek@ucloud`.

| Command | Does |
|---|---|
| `oca ls` | every server, and under it each model with its id and context; `*` marks the default. The model a server serves, or would start with, shows the server's state (ready, loading, down, stopped and why, or started by a project member); the others say whether they are downloaded there, or nothing where that is unknown |
| `oca <id>` | open OpenCode on that model, starting its server; another model on a running server restarts it, after asking |
| `oca stop [id or server]` | stop the server you started, or leave one you joined; `--all` stops every one, and oca's endpoint |
| `oca logs [id or server] -f`, `oca pull [id]` | its log; download its model ahead of time |
| `oca config default <id>` | what plain `oca` opens |
| `oca server add <name> <target>` | add one: `ucloud`, `you@host`, or a URL |
| `oca server set <name> idle 30` | change a setting: `idle`, `hours`, `model`, `port`, `context` |
| `oca server update <name>` | reinstall oca on a machine over SSH; `OCA_SOURCE` picks the version, as for install.sh |
| `oca server rm <name>` | forget it |

Without an id, a command acts on the default. When its server is down, plain `oca` offers to restart it, or another server.

### A machine over SSH

`oca server add spark you@spark`, for a DGX Spark for example, needs SSH without a password (`ssh-copy-id you@spark`). It asks whether to install oca there for you only, or for all its users:
- **For you:** in your home there.
- **For all users:** in `/opt/oca`, which an admin of that machine creates once, for a group `oca` holding every user who shares it (they log in again to join it):
  ```bash
  sudo groupadd -f oca
  sudo usermod -aG oca <user>
  sudo install -d -g oca -m 2775 /opt/oca
  ```
  Whoever adds the machine next finds that install and shares its server.

It then asks which model to serve first, the biggest that fits the machine's GPU on top: `qwen3.8-flash-next` (94 GB) on a DGX Spark's 128 GB, `qwen3.6-35b-a3b` on a 32 GB GPU. `oca ls` lists the others that fit. The server runs llama.cpp, stops after `idle` minutes without requests, and only its starter can stop it or change its model.

### UCloud

Once per person:
1. **Log in.** Open cloud.sdu.dk in a browser, then DevTools → Application → Cookies, and copy `refreshToken`. Run `oca ucloud login --project <id>`; `oca ucloud projects` lists your projects.
2. **Register an SSH key.** Run `oca ucloud ssh-keys add ~/.ssh/id_ed25519.pub --title $(hostname)`, after `ssh-keygen -t ed25519` if you have no key. A server only accepts keys registered before it started.
3. **Add it.** `oca server add ucloud ucloud` lists your drives and asks which one holds the weights and job files.

Once per project, download the weights: `oca pull flash-next@ucloud`. This runs a CPU job that takes a few minutes and uses no GPU-hours. `oca flash-next@ucloud` shows a server's GPU-hour cap before starting it; `oca server extend 2` adds 2 h (starter only).

| Id | B200s | Start | Note |
|---|---|---|---|
| `flash-next@ucloud` (default) | 1 | about 5 min | single B200s are usually free |
| `deepseek@ucloud` | 2 | about 25 min | stronger; pairs can queue for hours |
| `qwen3.8-27b@ucloud` | 1 | about 5 min | small |

**Switching model:** run `oca deepseek@ucloud`. It offers to stop or leave the current server. Replies wait for the new one meanwhile, showing its progress.

**Cost and timeouts:** a server stops after `idle` minutes without requests (default 60), and a job runs at most `hours` (default 8): `oca server set ucloud idle 30`. Queue time is free. A server shared by several people costs the same as one.

**Sharing:** everyone in the project using the same drive shares servers: `oca ls` shows a server a member started, and `oca <id>` joins it instead of starting another. Project members' SSH keys also give them a shell in the job, so share servers only within a trusted project.

## Phone and other machines

`oca web` serves OpenCode's web interface for the current folder, in the background. It listens on localhost only and needs a password: the user is `opencode`, and `oca web` prints the password. It keeps running after you log out; `oca web stop` ends it.

From a phone, one SSH command starts it if needed and opens the tunnel, with no tmux:

```bash
ssh -L 4096:127.0.0.1:4096 you@workstation 'cd ~/codes/PROJECT && ~/.local/bin/oca web && sleep infinity'
```

Then open `http://127.0.0.1:4096` on the phone. `Ctrl+C` closes the tunnel, and the server keeps running for next time.

- **Android:** Termux works (`pkg install openssh`). JuiceSSH's port forwarding needs its paid Pro pack.
- **Through a jump host,** such as a VPS that reaches the workstation: add `-J you@vps`.
- **Another project:** the web interface opens the folder `oca web` started in. Run it in the other folder to switch; this replaces the running server. In OpenCode's own folder picker, clicking a folder selects it, so reach a subfolder by typing its path.
- **The terminal UI over SSH** also works: `tmux new -As oca oca`.

## Settings

`oca config` shows all settings; `oca config <key> <value>` sets one.

| Key | Default | Meaning |
|---|---|---|
| `device` | auto | cpu, gpu or auto (gpu if `nvidia-smi` exists) |
| `model` | auto | local model id |
| `port` | 8080 | oca's endpoint for OpenCode and other tools; llama-server uses the next port |
| `threads`, `context` | 0 | CPU threads and local context; 0 lets llama.cpp and the model decide |
| `default` | local | the model plain `oca` opens: an id from `oca ls`, or part of one |

Each server has its own settings: `oca server set <name> <key> <value>`.

## Troubleshooting

- **"this endpoint needs oca's key":** an OpenCode session started before the update; restart it with `oca`.

- **"UCloud rejected this machine's SSH key":** register the key (see above). The server then has to be restarted.
- **Job "in queue":** the B200s are busy. Waiting costs nothing, and `oca stop ucloud` cancels.
- **UCloud says SUCCESS but nothing works:** UCloud reports success for failed jobs, so read `oca logs ucloud`.
- **"cannot reach … over SSH":** `ssh you@host` must work without a password; `ssh-copy-id you@host` sets that up.
- **Logs:** `oca logs <id or server> -f` for a server's, and `logs/router.log` in the data directory for which server answered each request.

## Uninstall

`oca uninstall` removes everything oca installed, including the `oca` command. `--keep-models` keeps the downloaded models; `--purge` also deletes OpenCode's sessions and logs.
