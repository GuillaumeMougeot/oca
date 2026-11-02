"""OpenCode's web interface as a background server, for phones and other machines.

It listens on localhost behind a password, so the way in is an SSH tunnel; or, with
``--vpn``, on this machine's WireGuard address only, for devices that joined that VPN. It
keeps running after the terminal that started it closes.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from . import agent, install
from .config import OcaError, data_dir, open_log, process_alive


def state_path() -> Path:
    return data_dir() / "run" / "web.json"


def log_path() -> Path:
    return data_dir() / "logs" / "web.log"


def running() -> dict[str, Any] | None:
    """The server's pid, host, port and project, if it runs."""
    try:
        state: dict[str, Any] = json.loads(state_path().read_text())
    except (FileNotFoundError, ValueError):
        return None
    return state if process_alive(int(state.get("pid", 0)), "opencode") else None


def _listening(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def vpn_address() -> str:
    """This machine's address on its WireGuard VPN, where devices that joined it reach it."""

    def ip(*args: str) -> list[list[str]]:
        try:
            out = subprocess.run(["ip", "-o", *args], capture_output=True, text=True).stdout
        except OSError:  # no iproute2, e.g. macOS
            return []
        return [line.split() for line in out.splitlines()]

    links = [
        words[1].rstrip(":").split("@")[0] for words in ip("link", "show", "type", "wireguard")
    ]
    found = [w[3].split("/")[0] for link in links for w in ip("-4", "addr", "show", "dev", link)]
    if len(found) != 1:
        raise OcaError(
            f"this machine has {len(found)} WireGuard addresses ({', '.join(found) or 'none'});"
            " `oca web --vpn` needs one"
        )
    return found[0]


def start(port: int, project: Path, host: str = "127.0.0.1") -> dict[str, Any]:
    """Serve ``project`` on ``host``:``port``, unless that server already runs."""
    wanted = (host, port, str(project))
    if (state := running()) and (
        state.get("host", "127.0.0.1"),
        state["port"],
        state["project"],
    ) == wanted:
        return state
    stop()
    exe = install.opencode()
    if exe is None:
        raise OcaError("OpenCode is not installed; run `oca setup`")
    env = dict(
        os.environ,
        OPENCODE_CONFIG=str(agent.config_path()),
        OPENCODE_SERVER_PASSWORD=agent.web_password(),
        PWD=str(project),  # OpenCode takes its project from $PWD
        BROWSER="true",  # `opencode web` opens a browser; there is none to open here
    )
    with open_log(log_path()) as log:
        proc = subprocess.Popen(
            [str(exe), "web", "--hostname", host, "--port", str(port)],
            cwd=project,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # outlives the terminal, or SSH session, that started it
        )
    state = {"pid": proc.pid, "host": host, "port": port, "project": str(project)}
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(state))
    for _ in range(300):
        if _listening(host, port):
            return state
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    stop()
    raise OcaError(f"OpenCode's web server did not start; see {log_path()}")


def stop() -> bool:
    state = running()
    if state:
        os.kill(int(state["pid"]), signal.SIGTERM)
    state_path().unlink(missing_ok=True)
    return state is not None
