# Permissions

**Status:** RESOLVED. The rules follow Claude Code's auto mode: work freely, ask before damage, never administer.

## Question
In build mode, the agent asked for permission too often. Claude Code's auto mode rarely asks: only for things like deleting or reading outside the repository. It never runs `sudo`.

## Before
Shell commands asked by default, except read-only and test commands. This followed the incident in [local-mvp](2026-10-29-local-mvp.md): a local model ran `pip install --break-system-packages`.

## Now (`agent.PERMISSION`; the last matching rule wins)
- **allow:** every shell command and edit not listed below.
- **ask:**
  - deleting (`rm`, `shred`, `dd`, …);
  - rewriting git history or pushing (`git push`, `reset --hard`, `clean`, `rebase`, …);
  - reaching other machines (`curl`, `wget`, `ssh`, `scp`, `rsync`, web fetch and search);
  - installing outside the project (`pip install`, `npm install -g`, `uv tool install`);
  - `kill`, `crontab`, `docker`;
  - any path outside the project.
- **deny:** `sudo`, `su`, `doas`, `pkexec`, `systemctl`, `service`, `mount`, `reboot`, `shutdown`, `passwd`, user management, `--break-system-packages`, and `rm -rf` of `/` or `~`.

OpenCode's prompt offers *always*, which lasts for the rest of the session.

## Check (2026-10-30, Flash-Next, non-interactive)
- `python3 -c 'print(6*7)'` ran without a prompt.
- `rm junk.txt` asked; non-interactive runs reject asks, and the file stayed.
- `sudo true` was refused by rule, with no prompt.
