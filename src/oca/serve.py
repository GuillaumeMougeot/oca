"""`oca serve`: this machine's model for other machines' oca, which reach it over SSH.

`oca serve` starts a background supervisor and answers with one line of JSON: model, port,
key and state (installing, downloading, loading, ready). The supervisor downloads the model
if needed, runs llama-server, and stops it after ``idle`` minutes without requests; the
listing keeps why it stopped, for the users who find it stopped. On an
install for all users of a machine (OCA_SHARED=1) everyone's oca joins this one server;
only its starter can stop it or change its model.
"""

from __future__ import annotations

import getpass
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from . import install, server
from .config import (
    OcaError,
    Settings,
    data_dir,
    model_path,
    open_log,
    process_alive,
    secret,
    write_private,
)

# The llama.cpp counters that move with any request; unchanged for `idle` minutes is idle.
ACTIVITY = (
    "llamacpp:prompt_tokens_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:requests_processing",
)


def listing_path() -> Path:
    return data_dir() / "run" / "serve.json"


def log_path() -> Path:
    return data_dir() / "logs" / "serve.log"


def _read() -> dict[str, Any]:
    try:
        entry: dict[str, Any] = json.loads(listing_path().read_text())
    except (FileNotFoundError, ValueError):
        return {}
    return entry


def _set(**fields: Any) -> None:
    write_private(listing_path(), json.dumps({**_read(), **fields}))


def status() -> dict[str, Any]:
    entry = _read()
    if not process_alive(int(entry.get("pid", 0)), "oca.serve"):
        return {"state": "stopped", "why": entry.get("why", "")}
    return {
        **{k: entry[k] for k in ("state", "model", "context", "started_by")},
        "port": server.port(Settings.load()),
        "key": secret("api-key"),  # llama-server's; readable by this install's users
        "mine": entry["started_by"] == getpass.getuser(),
    }


def start(model: str | None, idle: int) -> dict[str, Any]:
    """Serve ``model`` (default: the configured one), or join the server that runs it."""
    now = status()
    if now["state"] != "stopped":
        if model in (None, now["model"]):
            return now
        if not now["mine"]:
            raise OcaError(
                f"this machine serves {now['model']} for {now['started_by']}; "
                "use that model, or wait until it stops"
            )
        stop()
    settings = Settings.load()
    if model:
        settings.set("model", model)
        settings.save()
    entry = {"model": settings.resolved_model.id, "context": settings.resolved_context}
    write_private(listing_path(), json.dumps({**entry, "started_by": getpass.getuser()}))
    with open_log(log_path()) as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "oca.serve", str(idle)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # outlives the SSH session that started it
        )
    for _ in range(100):  # until it listed itself; from then on only it writes the listing
        if _read().get("pid") == proc.pid or proc.poll() is not None:
            break
        time.sleep(0.1)
    return status()


def stop() -> bool:
    entry = _read()
    pid = int(entry.get("pid", 0))
    if not process_alive(pid, "oca.serve"):
        return False
    if entry.get("started_by") != getpass.getuser():
        raise OcaError(f"only {entry.get('started_by')}, who started it, can stop it")
    os.kill(pid, signal.SIGTERM)
    for _ in range(300):
        if not process_alive(pid, "oca.serve"):
            return True
        time.sleep(0.1)
    raise OcaError(f"the server did not stop; see {log_path()}")


def _idle(settings: Settings, minutes: int) -> str:
    """Return once llama-server is gone, or after ``minutes`` without requests (0: never);
    says which."""
    url = f"http://127.0.0.1:{server.port(settings)}/metrics"
    headers = {"Authorization": f"Bearer {secret('api-key')}"}
    quiet, last = 0, ""
    while server.read_state() is not None:
        time.sleep(60)
        try:
            text = httpx.get(url, headers=headers, timeout=10).text
        except httpx.HTTPError:
            continue  # no metrics is no proof of idleness
        now = "\n".join(line for line in text.splitlines() if line.startswith(ACTIVITY))
        quiet = quiet + 1 if now == last else 0
        last = now
        if minutes and quiet >= minutes:
            print(f"idle for {minutes} min, stopping", flush=True)
            return f"idle for {minutes} min"
    return "llama-server exited; `oca remote logs` shows why"


def supervise(idle: int) -> None:
    """The background process: install, download, serve, stop when idle, say why."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # runs the cleanup below
    _set(pid=os.getpid(), state="starting")
    why = f"stopped by {getpass.getuser()}"  # SIGTERM, from `oca remote down`
    try:
        settings = Settings.load()
        if not install.is_installed():
            _set(state="installing")
            install.setup()
        if not model_path(settings.resolved_model).exists():
            _set(state="downloading")
            install.pull(settings.resolved_model)
        _set(state="loading")
        server.start(settings)
        _set(state="ready")
        why = _idle(settings, idle)
    except Exception as e:
        why = f"failed: {e}"
        raise
    finally:
        server.stop()
        if _read().get("pid") == os.getpid():
            _set(state="stopped", why=f"{why}, at {time.strftime('%H:%M')}")


if __name__ == "__main__":
    supervise(int(sys.argv[1]))
