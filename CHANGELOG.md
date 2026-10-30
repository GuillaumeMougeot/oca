# Changelog

## Unreleased

- **`oca cluster up` returns at once.** `oca` works locally and switches once the server answers; `--wait` blocks as before.
- **`oca web stop`** replaces `oca web --stop`.
- **`oca sessions --clear`** deletes the sessions started in this folder or below.
- **Evaluation:** runs keep their sessions in their own store, out of the user's list.
- **CI:** a clean install runs `oca update`.

## v0.2.0 (2026-10-30)

- **Keys on every server.** Another user of a shared machine could use your endpoints, including your UCloud server through its tunnel. Now the router and llama-server need your key, and each UCloud server has its own, shared with project members through its listing.
- **Private files:** oca's configs and state, which hold keys, are readable only by you.
- **`oca update`** installs the newest release, and `install.sh` now installs the newest release rather than `main`.
- **`oca sessions`** lists or deletes this folder's sessions. Sessions get titles while a UCloud server answers.
- **`oca web`** runs in the background; one SSH command from a phone starts it and opens the tunnel.
- **Permissions** follow Claude Code's auto mode. Unattended runs refuse instead of asking, so a disallowed step no longer ends the run.
- **Effort** variants (`ctrl+t`): `none` to `max`, translated for each model.
- **Housekeeping:** logs rotate at 10 MB, `oca uninstall` removes eval runs, and `oca cluster up` starts afresh after a job ended.
- **Evaluation:** `eval/run.py`. Qwen3.8-Flash-Next solves 34/34 polyglot exercises and 9/9 tasks from oca's history.
- **Docs site:** [guillaumemougeot.github.io/oca](https://guillaumemougeot.github.io/oca/).

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
