"""Generate oca's OpenCode config and launch OpenCode with it."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, NoReturn

from . import install
from .config import EFFORTS, ClusterModel, OcaError, Settings, config_dir

PROVIDER = "oca"
MODEL = "auto"  # oca's router serves it with the best live server

# Like Claude Code's auto mode: work freely in the project; ask before deleting, rewriting
# git history, reaching other machines or installing outside the project; never administer
# the system (a local model once ran `pip install --break-system-packages`). OpenCode's
# prompts offer "always" for the rest of the session. The last matching rule wins.
PERMISSION: dict[str, Any] = {
    "edit": "allow",
    "external_directory": "ask",  # any path outside the project
    "bash": {
        "*": "allow",
        **dict.fromkeys(
            [
                *("rm *", "rmdir *", "shred *", "truncate *", "dd *"),
                *("git push*", "git reset --hard*", "git clean*", "git rebase*"),
                *("git checkout -- *", "git restore *", "git branch -D *", "git stash drop*"),
                *("curl *", "wget *", "ssh *", "scp *", "rsync *"),
                *("pip install*", "pip3 install*", "python -m pip install*"),
                *("python3 -m pip install*", "npm install -g*", "uv tool install*"),
                *("kill *", "pkill *", "killall *", "crontab *", "docker *"),
            ],
            "ask",
        ),
        **dict.fromkeys(
            [
                *("sudo *", "su", "su *", "doas *", "pkexec *"),
                *("systemctl *", "service *", "mount *", "umount *"),
                *("reboot*", "shutdown*", "poweroff*", "passwd*"),
                *("useradd *", "usermod *", "userdel *", "*--break-system-packages*"),
                *("rm -rf /", "rm -rf ~", "rm -rf ~/"),
            ],
            "deny",
        ),
    },
    "webfetch": "ask",
    "websearch": "ask",
}


def config_path() -> Path:
    return config_dir() / "opencode.json"


def _provider(name: str, port: int, model_id: str, label: str, context: int) -> dict[str, Any]:
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": name,
        # CPU prefill of a long prompt can exceed the 5 min default.
        "options": {"baseURL": f"http://127.0.0.1:{port}/v1", "headerTimeout": 1_800_000},
        "models": {
            model_id: {
                "name": label,
                "tool_call": True,
                "reasoning": True,
                "limit": {"context": context, "output": min(32768, context // 2)},
                # ctrl+t in OpenCode cycles these; oca's router translates them per model.
                "variants": {e: {"reasoningEffort": e} for e in EFFORTS},
            }
        },
    }


def opencode_config(settings: Settings, remote: ClusterModel | None = None) -> dict[str, Any]:
    """One model, ``oca/auto``; its label names ``remote`` when the cluster answers now.

    The context limit is the local model's, so a session can always fall back to it.
    """
    local = settings.resolved_model
    now = f"{remote.id} on UCloud" if remote else f"{local.id} ({settings.resolved_device})"
    providers = {
        PROVIDER: _provider(
            "oca", settings.port, MODEL, f"auto, now {now}", settings.resolved_context
        )
    }
    ref = f"{PROVIDER}/{MODEL}"
    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,  # versions are pinned by oca
        "share": "disabled",
        "enabled_providers": list(providers),  # open models only, no hosted fallbacks
        "permission": PERMISSION,
        "provider": providers,
        "model": ref,
        "small_model": ref,
        # A reasoning model thinks for 500-1,400 tokens to write a session title, and with
        # one server slot (MTP) the first turn waits behind it: 29-84 s per session on CPU.
        "agent": {"title": {"disable": True}},
    }


def write_config(settings: Settings, remote: ClusterModel | None = None) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(opencode_config(settings, remote), indent=2) + "\n")
    return path


def web_password() -> str:
    """The password of `oca web`, made once and readable only by this user."""
    path = config_dir() / "web-password"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        path.write_text(secrets.token_urlsafe(18))
    return path.read_text().strip()


def launch(args: list[str], **env: str) -> NoReturn:
    """Replace this process with OpenCode; the user's global config is left untouched."""
    exe = install.opencode()
    if exe is None:
        raise OcaError("OpenCode is not installed; run `oca setup`")
    # OpenCode takes its project from $PWD, which a caller may not have updated.
    full = dict(os.environ, OPENCODE_CONFIG=str(config_path()), PWD=os.getcwd(), **env)
    os.execve(exe, [str(exe), *args], full)
