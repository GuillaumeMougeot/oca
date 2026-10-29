"""Generate oca's OpenCode config and launch OpenCode with it."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NoReturn

from . import install
from .config import ClusterModel, OcaError, Settings, config_dir

PROVIDER = "oca-local"
CLUSTER = "oca-cluster"

# OpenCode allows every tool by default; a local model once ran
# `pip install --break-system-packages`. Ask for shell commands, except read-only
# and test ones. The last matching rule wins.
PERMISSION: dict[str, Any] = {
    "edit": "allow",
    "external_directory": "ask",
    "bash": {
        "*": "ask",
        **dict.fromkeys(
            [
                *("ls*", "pwd", "cat *", "head *", "tail *", "wc *", "grep *", "rg *", "find *"),
                *("git status*", "git diff*", "git log*", "git show*"),
                *("pytest*", "python -m pytest*", "python3 -m pytest*", "uv run *"),
            ],
            "allow",
        ),
        **dict.fromkeys(["sudo *", "*--break-system-packages*", "rm -rf /*"], "deny"),
    },
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
            }
        },
    }


def opencode_config(settings: Settings, remote: ClusterModel | None = None) -> dict[str, Any]:
    """The local model; plus ``remote``, as the default, when the cluster answers."""
    model = settings.resolved_model
    providers = {
        PROVIDER: _provider(
            "oca local (llama.cpp)",
            settings.port,
            model.id,
            f"{model.id} ({settings.resolved_device})",
            settings.resolved_context,
        )
    }
    ref = f"{PROVIDER}/{model.id}"
    if remote:
        providers[CLUSTER] = _provider(
            "oca cluster (vLLM on UCloud)",
            settings.cluster_port,
            remote.id,
            f"{remote.id} ({remote.gpus}x B200)",
            remote.context,
        )
        ref = f"{CLUSTER}/{remote.id}"
    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,  # versions are pinned by oca
        "share": "disabled",
        "enabled_providers": list(providers),  # open models only, no hosted fallbacks
        "permission": PERMISSION,
        "provider": providers,
        "model": ref,
        "small_model": ref,
    }


def write_config(settings: Settings, remote: ClusterModel | None = None) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(opencode_config(settings, remote), indent=2) + "\n")
    return path


def launch(args: list[str]) -> NoReturn:
    """Replace this process with OpenCode; the user's global config is left untouched."""
    exe = install.opencode()
    if exe is None:
        raise OcaError("OpenCode is not installed; run `oca setup`")
    env = dict(os.environ, OPENCODE_CONFIG=str(config_path()))
    os.execve(exe, [str(exe), *args], env)
