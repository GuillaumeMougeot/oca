# CLAUDE.md

Router for agents working on oca. It holds only what the code can't tell you and what prevents mistakes. A fact lives in one place; other places link to it.

## What

oca (Open Coding Agent) installs and runs a coding agent built only from open-source software and open-weight models:
- OpenCode as the agent,
- llama.cpp for local inference,
- vLLM on UCloud B200s, driven through [ucloud-api](https://github.com/GuillaumeMougeot/ucloud-api).

Goal: installing it is as easy as uv or Claude Code, plus one model download. oca is glue: configure upstream tools, don't rebuild them.

## Read first

1. `ROADMAP.md`: status and next step.
2. `docs/journal/README.md`: why things are the way they are.
3. Module docstrings in `src/oca/`. Pinned versions and models are in `config.py`.

## Invariants

- Heavy files (models, binaries, logs) go under `data_dir()` (here `/data/$USER/oca`), never in the repo or `$HOME`.
- Pin every external version (llama.cpp tag, OpenCode version, model file). Bump deliberately and journal why.
- Open-source software and open-weight models only. Record each license in the ROADMAP table.
- Never touch the user's global OpenCode config. oca passes its own through `OPENCODE_CONFIG`.
- Keep the agent permission rules in `agent.py`. Without them, a local model ran `pip install --break-system-packages` on the host ([journal](docs/journal/2026-10-29-local-mvp.md)).
- The local GPU is shared with another user. Check `nvidia-smi` before using `device=gpu`. This host uses `cpu` for now.
- UCloud:
  - Ask the owner before spending GPU-hours.
  - The `/models` drive (`/12383018`) is tight: about 1 TB left for the whole project.
  - Read job logs, not job status.
  - ucloud-api has its own traps; see its README.

## How to work

- Smallest thing that works. Delete rather than add. No speculative options.
- Measure before claiming (tok/s, wall time, pass rate). Write the expectation in the journal before the number lands.
- Tests run offline: mock network, processes and GPU.
- Fewer, shorter docs. Word count matters, in docs and in messages.

## Owner preferences

- Journal everything: `docs/journal/YYYY-MM-DD-topic.md`, indexed in `docs/journal/README.md`.
- Plan and discuss before sizeable changes. Give a recommendation, not a survey.
- Commit and push freely to `main` in this repo.
- Use uv for everything.

## Before finishing

- Run `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`.
- Update ROADMAP status and the journal.
- If you learned something that would have changed how you started, add it here.
