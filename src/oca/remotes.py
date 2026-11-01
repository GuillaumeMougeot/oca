"""Servers on other machines ("remotes"), by kind:

- ucloud: vLLM in a UCloud B200 job (cluster.py), shared by a project's members.
- ssh: a machine you reach over SSH, such as a DGX Spark. oca installs itself there, for
  you or for all its users, and runs `oca serve` (serve.py): llama.cpp, which stops when
  idle. Your oca reaches it through an SSH tunnel.
- url: an OpenAI-compatible server run by someone else.

How to reach a running ssh server is kept in run/remote-<name>.json; ucloud keeps its
job in run/cluster.json.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from . import cluster
from .config import (
    CLUSTER_MODELS,
    MODELS,
    Down,
    OcaError,
    Remote,
    Settings,
    data_dir,
    process_alive,
    write_private,
)

SHARED_ROOT = "/opt/oca"  # the install for all users of a machine, like Anaconda's
SHARED_PORT = 8180  # its endpoint; llama-server listens one above, off users' own oca ports
INSTALLER = "https://raw.githubusercontent.com/GuillaumeMougeot/oca/main/install.sh"
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10")

# Installs oca in a folder all users of the machine can read; exit 3 when it is not writable.
SHARED_INSTALL = """set -eu
R={root}
{{ [ -d "$R" ] && [ -w "$R" ]; }} || exit 3
umask 002
export PATH="$HOME/.local/bin:$PATH" UV_TOOL_DIR="$R/tools" UV_TOOL_BIN_DIR="$R/bin"
export UV_PYTHON_INSTALL_DIR="$R/python" UV_CACHE_DIR="$R/cache/uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
tag=$(git ls-remote --tags --refs --sort=-v:refname https://github.com/GuillaumeMougeot/oca 'v*' \\
  | head -n 1 | sed 's|.*refs/tags/||')
uv tool install --force --python 3.12 \\
  "${{OCA_SOURCE:-git+https://github.com/GuillaumeMougeot/oca@${{tag:-main}}}}"
export OCA_SHARED=1 OCA_DATA="$R/data" OCA_CONFIG_DIR="$R/config"
"$R/bin/oca" config port {port} >/dev/null
"$R/bin/oca" setup
echo "installed_by = \\"$(whoami)@$(hostname)\\"" > "$R/shared.toml"
"""


@dataclass(frozen=True)
class Backend:
    """Where requests for a server go."""

    name: str  # the server: "local" or a remote's name
    url: str
    model: str  # the id the server knows its model by
    key: str  # its bearer token; "" for none
    context: int
    efforts: tuple[str | int, ...] = ()


@dataclass
class Conn:
    """A running ssh server: its port on that machine, key and model."""

    model: str
    port: int
    key: str
    context: int
    mine: bool  # started by this user, so `oca remote down` stops it
    tunnel_pid: int = 0


def conn_path(name: str) -> Path:
    return data_dir() / "run" / f"remote-{name}.json"


def read_conn(name: str) -> Conn | None:
    try:
        return Conn(**json.loads(conn_path(name).read_text()))
    except (FileNotFoundError, ValueError, TypeError):
        return None


def _save(name: str, conn: Conn | None) -> None:
    if conn is None:
        conn_path(name).unlink(missing_ok=True)
    else:
        write_private(conn_path(name), json.dumps(asdict(conn)))


# -- what OpenCode and the router see ------------------------------------------------------


def context(settings: Settings, name: str) -> int:
    if name == "local":
        return settings.resolved_context
    r = settings.remotes[name]
    if r.kind == "ucloud":
        return CLUSTER_MODELS[r.model].context
    if r.kind == "ssh":
        conn = read_conn(name)
        return conn.context if conn else r.context or MODELS[r.model].context
    return r.context


def label(settings: Settings, name: str) -> str:
    if name == "local":
        return f"{settings.resolved_model.id} on this machine ({settings.resolved_device})"
    r = settings.remotes[name]
    return f"{r.model} on " + {"ucloud": "UCloud", "ssh": r.host, "url": r.url}[r.kind]


def backend(settings: Settings, name: str) -> Backend | None:
    """Where requests for remote ``name`` go, if oca is connected to it (health unchecked)."""
    r = settings.remotes[name]
    here = f"http://127.0.0.1:{r.port}"
    if r.kind == "ucloud":
        job = cluster.read("serve")
        if job is None:
            return None
        m = CLUSTER_MODELS.get(job.model)
        size, efforts = (m.context, m.efforts) if m else (0, ())
        return Backend(name, here, job.model, job.key, size, efforts)
    if r.kind == "ssh":
        conn = read_conn(name)
        return None if conn is None else Backend(name, here, conn.model, conn.key, conn.context)
    return Backend(name, r.url, r.model, r.key, r.context)


def healthy(b: Backend) -> bool:
    """llama-server and vLLM answer /health once loaded; other servers list their models."""
    headers = {"Authorization": f"Bearer {b.key}"} if b.key else {}
    try:
        r = httpx.get(f"{b.url}/health", headers=headers, timeout=5)
        if r.status_code in (401, 404):
            r = httpx.get(f"{b.url}/v1/models", headers=headers, timeout=5)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


# -- connecting ----------------------------------------------------------------------------


def check(settings: Settings, name: str) -> str | None:
    """One connection attempt: None when ``name`` answers, else what it is doing.

    Re-opens a tunnel if needed. Raises Down when nothing serves there.
    """
    r = settings.remotes[name]
    if r.kind == "ucloud":
        job = cluster.read("serve")
        if job is None:
            raise Down(f"{name}: no server started")
        return cluster.check(r, job)
    if r.kind == "url":
        b = backend(settings, name)
        if b and healthy(b):
            return None
        raise Down(f"{name}: {r.url} does not answer")
    conn = read_conn(name)
    if conn is None:
        raise Down(f"{name}: no server started")
    state = serve_call(r, "--status")
    if state["state"] == "stopped":
        _forget(name, conn)
        raise Down(f"{name}: its server stopped")
    conn.port, conn.key, conn.context = state["port"], state["key"], state["context"]
    _tunnel(name, r, conn)
    here = Backend(name, f"http://127.0.0.1:{r.port}", conn.model, conn.key, conn.context)
    if state["state"] == "ready" and healthy(here):
        return None
    return str(state["state"])


def up(settings: Settings, name: str) -> str | None:
    """Start the ssh server's model there, or join its running server; None once it answers."""
    r = settings.remotes[name]
    state = serve_call(r, f"--model {shlex.quote(r.model)} --idle {r.idle}")
    old = read_conn(name)
    conn = Conn(state["model"], state["port"], state["key"], state["context"], state["mine"])
    conn.tunnel_pid = old.tunnel_pid if old else 0
    _tunnel(name, r, conn)
    return None if state["state"] == "ready" else str(state["state"])


def down(settings: Settings, name: str) -> bool | None:
    """Close the ssh tunnel and stop the server if this user started it.

    Returns None when not connected, True when stopped, False when left for others.
    """
    conn = read_conn(name)
    if conn is None:
        return None
    _forget(name, conn)
    if conn.mine:
        serve_call(settings.remotes[name], "--stop")
    return conn.mine


def _forget(name: str, conn: Conn) -> None:
    if process_alive(conn.tunnel_pid, "ssh"):
        os.kill(conn.tunnel_pid, signal.SIGTERM)
    _save(name, None)


def wait_ready(
    settings: Settings, name: str, on_progress: Callable[[str], None], timeout: float = 7200
) -> None:
    r = settings.remotes[name]
    if r.kind == "ucloud":
        job = cluster.read("serve")
        if job is None:
            raise Down(f"{name}: no server started")
        cluster.wait_ready(r, job, on_progress, timeout)
        return
    deadline = time.monotonic() + timeout
    while (doing := check(settings, name)) is not None:
        on_progress(doing)
        if time.monotonic() > deadline:
            raise OcaError(f"not ready after {timeout / 60:.0f} min; see `oca remote logs {name}`")
        time.sleep(10)


def _tunnel(name: str, r: Remote, conn: Conn) -> None:
    """Forward the remote's local port ``r.port`` to the server's port there."""
    saved = read_conn(name)  # the router's background check may have opened one meanwhile
    pid = saved.tunnel_pid if saved and saved.port == conn.port else conn.tunnel_pid
    if process_alive(pid, "ssh") and cluster.port_open(r.port):
        conn.tunnel_pid = pid
        _save(name, conn)
        return
    for old in {conn.tunnel_pid, saved.tunnel_pid if saved else 0}:  # leads to a moved server
        if process_alive(old, "ssh"):
            os.kill(old, signal.SIGTERM)
    for _ in range(20):  # until its port is free again
        if not cluster.port_open(r.port):
            break
        time.sleep(0.1)
    cmd = [
        *SSH,
        *("-N", "-o", "ExitOnForwardFailure=yes"),
        *("-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"),
        *("-L", f"127.0.0.1:{r.port}:127.0.0.1:{conn.port}"),
        r.host,
    ]
    conn.tunnel_pid = cluster.open_tunnel(cmd, r.port, data_dir() / "logs" / f"tunnel-{name}.log")
    _save(name, conn)


# -- the other machine ---------------------------------------------------------------------


def ssh(host: str, command: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*SSH, host, command],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def oca_command(r: Remote) -> str:
    """The oca command there: the install for all its users, or the user's own."""
    if not r.shared:
        return '"$HOME/.local/bin/oca"'
    root = shlex.quote(r.shared)
    env = f"OCA_SHARED=1 OCA_DATA={root}/data OCA_CONFIG_DIR={root}/config"
    return f"umask 007; {env} {root}/bin/oca"


def serve_call(r: Remote, args: str) -> dict[str, Any]:
    """Run ``oca serve <args>`` there; it answers with one line of JSON."""
    done = ssh(r.host, f"{oca_command(r)} serve {args}")
    lines = done.stdout.strip().splitlines()
    try:
        answer: dict[str, Any] = json.loads(lines[-1])
    except (IndexError, ValueError):
        said = done.stderr.strip() or done.stdout.strip() or f"exit {done.returncode}"
        raise OcaError(f"`oca serve` on {r.host} failed: {said.splitlines()[-1]}") from None
    if "error" in answer:
        raise OcaError(f"{r.host}: {answer['error']}")
    return answer


@dataclass
class Probe:
    arch: str
    gpu: str  # name and memory of its first GPU, "" without one
    shared: bool  # oca is installed there for all users
    personal: bool  # this user has their own oca there


def probe(host: str, root: str = SHARED_ROOT) -> Probe:
    q = shlex.quote(root)
    done = ssh(
        host,
        f"uname -m; test -f {q}/shared.toml && echo @shared; "
        'test -x "$HOME/.local/bin/oca" && echo @personal; '
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1;"
        " true",
    )
    if done.returncode != 0:
        said = done.stderr.strip().splitlines()
        raise OcaError(
            f"cannot reach {host} over SSH ({said[-1] if said else 'no answer'}). It must work "
            f"without a password: `ssh {host}` once, and `ssh-copy-id {host}` if it asks for one."
        )
    lines = done.stdout.strip().splitlines()
    gpu = [line for line in lines[1:] if not line.startswith("@")]
    arch = lines[0] if lines else ""
    return Probe(arch, gpu[0] if gpu else "", "@shared" in lines, "@personal" in lines)


def install(host: str, shared: str = "") -> None:
    """Install oca there: for this user, or for all users in folder ``shared``.

    OCA_SOURCE, if set here, picks what to install, as for install.sh.
    """
    source = os.environ.get("OCA_SOURCE", "")
    env = f"OCA_SOURCE={shlex.quote(source)} " if source else ""
    if shared:
        steps = SHARED_INSTALL.format(root=shlex.quote(shared), port=SHARED_PORT)
        script = f"{env}sh -c {shlex.quote(steps)}"
    else:
        script = f"curl -LsSf {INSTALLER} | {env}sh"
    done = ssh(host, script, timeout=3600)
    if done.returncode == 3:
        raise OcaError(
            f"{shared} on {host} must exist and be writable by the users who share it. An admin "
            f"of {host} runs once, with a group all of them are in:\n"
            f"  sudo mkdir -p {shared} && sudo chgrp <group> {shared} && sudo chmod 2775 {shared}"
        )
    if done.returncode != 0:
        said = (done.stderr.strip() or done.stdout.strip() or "no output").splitlines()
        raise OcaError(f"installing oca on {host} failed: {said[-1]}")
