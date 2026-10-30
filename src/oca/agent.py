"""Generate oca's OpenCode config and launch OpenCode with it."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, NoReturn

from . import install
from .config import EFFORTS, ClusterModel, OcaError, Settings, config_dir, secret, write_private

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


def config_path(unattended: bool = False) -> Path:
    """oca's OpenCode config; the unattended one is for `opencode run`, which cannot ask."""
    return config_dir() / ("opencode-run.json" if unattended else "opencode.json")


def _refuse_asks(rules: Any) -> Any:
    """``rules`` with every "ask" turned into "deny".

    Unattended, OpenCode rejects what would ask, and a rejection ends the run. A denial is
    an error the model reads and works around: 2 of 9 history tasks and most polyglot
    failures ended on such a rejection.
    """
    if isinstance(rules, dict):
        return {k: _refuse_asks(v) for k, v in rules.items()}
    return "deny" if rules == "ask" else rules


def _provider(name: str, port: int, model_id: str, label: str, context: int) -> dict[str, Any]:
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": name,
        "options": {
            "baseURL": f"http://127.0.0.1:{port}/v1",
            # Other users of this machine reach localhost too: the router wants this key.
            "apiKey": api_key(),
            "headerTimeout": 1_800_000,  # CPU prefill of a long prompt can exceed the 5 min default
        },
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
        # Session titles, without thinking: free on the cluster (3.6 s per task with, 3.7
        # without), but on the local CPU server's single slot (MTP) a title request evicts the
        # session's cached prompt: 77-121 s per task against 39 s. So only when it is up.
        "agent": {"title": {"model": ref, "variant": "none"} if remote else {"disable": True}},
    }


def write_config(settings: Settings, remote: ClusterModel | None = None) -> Path:
    cfg = opencode_config(settings, remote)
    write_private(config_path(), json.dumps(cfg, indent=2) + "\n")
    unattended = {**cfg, "permission": _refuse_asks(cfg["permission"])}
    write_private(config_path(unattended=True), json.dumps(unattended, indent=2) + "\n")
    return config_path()


def web_password() -> str:
    """The password of `oca web`."""
    return secret("web-password")


def api_key() -> str:
    """The key of oca's endpoint (router), which also guards the local llama-server."""
    return secret("api-key")


def opencode(args: list[str]) -> str:
    """Run OpenCode with oca's config here; returns its output."""
    exe = install.opencode()
    if exe is None:
        raise OcaError("OpenCode is not installed; run `oca setup`")
    env = dict(os.environ, OPENCODE_CONFIG=str(config_path()), PWD=os.getcwd())
    return subprocess.run(
        [str(exe), *args], env=env, capture_output=True, text=True, check=True
    ).stdout


def sessions_in(folder: Path) -> list[str]:
    """Ids of the sessions started in ``folder`` or below.

    OpenCode lists sessions per project, and files every folder outside git under one
    "global" project, so its list from such a folder also holds other folders' sessions.
    """
    found = json.loads(opencode(["session", "list", "--format", "json", "-n", "1000000"]) or "[]")
    root = str(folder)
    return [
        s["id"]
        for s in found
        if (d := s.get("directory") or "") == root or d.startswith(root.rstrip(os.sep) + os.sep)
    ]


def launch(args: list[str], **env: str) -> NoReturn:
    """Replace this process with OpenCode; the user's global config is left untouched."""
    exe = install.opencode()
    if exe is None:
        raise OcaError("OpenCode is not installed; run `oca setup`")
    # OpenCode takes its project from $PWD, which a caller may not have updated.
    cfg = config_path(unattended=args[:1] == ["run"])
    full = dict(os.environ, OPENCODE_CONFIG=str(cfg), PWD=os.getcwd(), **env)
    os.execve(exe, [str(exe), *args], full)
