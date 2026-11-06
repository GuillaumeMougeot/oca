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
from collections.abc import Callable, Collection
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
    catalog,
    data_dir,
    memory_gb,
    model_path,
    process_alive,
    write_private,
)

SHARED_ROOT = "/opt/oca"  # the install for all users of a machine, like Anaconda's
SHARED_PORT = 8180  # its endpoint; llama-server listens one above, off users' own oca ports
INSTALLER = "https://raw.githubusercontent.com/GuillaumeMougeot/oca/main/install.sh"
SSH = ("ssh", "-o", "BatchMode=yes")

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
    mine: bool  # started by this user, so `oca stop <name>` stops it
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


# -- pairs: a model on a server, what `oca ls` lists and `oca <id>` opens -------------------


@dataclass(frozen=True)
class Pair:
    id: str  # the model's short name @ the server, such as flash-next@spark2
    server: str
    model: str  # its catalog id; for a url server, the server's own name for it
    context: int


def pair_id(settings: Settings, server: str, model: str | None = None) -> str:
    """The id of ``model`` (default: the one it serves) on ``server``."""
    model = model or settings.model_of(server)
    r = settings.remotes.get(server)
    if r is not None and r.kind == "url":
        return f"{model}@{server}"
    found = CLUSTER_MODELS.get(model) if r is not None and r.kind == "ucloud" else MODELS.get(model)
    return f"{found.short if found else model}@{server}"


def pairs(settings: Settings) -> list[Pair]:
    """Every model each server can serve, the one it serves first; this machine and SSH
    machines list what fits their memory."""
    found = []
    for name in settings.servers():
        r = settings.remotes.get(name)
        kind = "local" if r is None else r.kind
        if r is not None and kind == "url":
            ids = [r.model]
        else:
            memory = memory_gb(settings.resolved_device == "gpu") if r is None else r.memory
            ids = [
                m
                for m in catalog(kind).values()
                if kind == "ucloud" or not memory or MODELS[m].size_gb * 1.2 <= memory
            ]
        now = settings.model_of(name)
        for m in sorted(ids, key=lambda m: m != now):
            if m == now:
                size = context(settings, name)
            elif kind == "ucloud":
                size = CLUSTER_MODELS[m].context
            else:
                size = (settings.context if r is None else 0) or MODELS[m].context
            found.append(Pair(pair_id(settings, name, m), name, m, size))
    return found


def find(settings: Settings, word: str) -> list[Pair]:
    """The pairs ``word`` names: the one whose id it is, else those whose id contains it, as
    `oca deep` names deepseek@ucloud, else those whose model@server does."""
    every = pairs(settings)
    full = {p.id: f"{p.model}@{p.server}" for p in every}
    for found in (
        [p for p in every if word in (p.id, full[p.id])],
        [p for p in every if word in p.id],
        [p for p in every if word in full[p.id]],
    ):
        if found:
            return found
    return []


def default_pair(settings: Settings) -> Pair:
    """What plain `oca` opens."""
    _, at, server = settings.default.rpartition("@")
    wanted = settings.default if at else pair_id(settings, server)
    return next(p for p in pairs(settings) if p.id == wanted)


def where(settings: Settings, name: str) -> str:
    """Where server ``name`` runs: this machine, UCloud, an SSH host or a URL."""
    if name == "local":
        return f"this machine ({settings.resolved_device})"
    r = settings.remotes[name]
    return {"ucloud": "UCloud", "ssh": r.host, "url": r.url}[r.kind]


def label(settings: Settings, name: str) -> str:
    model = settings.resolved_model.id if name == "local" else settings.remotes[name].model
    return f"{model} on {where(settings, name)}"


@dataclass(frozen=True)
class Choice:
    """An option while a server is down, such as "Restart spark" or "Use local"."""

    label: str
    description: str


def choices(
    settings: Settings, down: str, live: Callable[[str], bool], failed: Collection[str] = ()
) -> list[Choice]:
    """What to offer while server ``down`` is down: restarting it, then the other servers,
    running ones and the largest first. What just ``failed`` is only offered again if it is
    ``down`` itself, last."""
    size = context(settings, down)
    running = {n: live(n) for n in settings.servers() if n != down}

    def startable(n: str) -> bool:
        if n == "local":
            return model_path(settings.resolved_model).exists()
        return settings.remotes[n].kind != "url"

    def about(n: str) -> str:
        ctx = context(settings, n)
        text = f"{label(settings, n)}, {ctx // 1000}k context"
        if ctx < size:
            text += f"; sees only the latest {ctx // 1000}k tokens"
        r = settings.remotes.get(n)
        if r and r.kind == "ucloud" and not running.get(n):
            gpus = CLUSTER_MODELS[r.model].gpus
            text += f"; a {gpus}x B200 job, up to {gpus * r.hours} GPU-hours"
        return text

    first = [Choice(f"Restart {down}", about(down))]
    others = [n for n, up in running.items() if n not in failed and (up or startable(n))]
    others.sort(key=lambda n: (not running[n], n != "local", -context(settings, n)))
    again = [Choice(f"Try {down} again", about(down))]
    use = [
        Choice(f"Use {n}", ("running: " if running[n] else "starts ") + about(n)) for n in others
    ]
    return [
        *(first if down not in failed and startable(down) else []),
        *use,
        *(again if down in failed or not startable(down) else []),
    ]


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
        if conn is None:
            return None
        efforts = MODELS[conn.model].efforts if conn.model in MODELS else ()
        return Backend(name, here, conn.model, conn.key, conn.context, efforts)
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

    Re-opens a tunnel if needed, and joins a server someone else started over SSH. Raises
    Down, saying why, when nothing serves there.
    """
    r = settings.remotes[name]
    if r.kind == "ucloud":
        job = cluster.read("serve")
        if job is None:  # a project member may run it: join that
            shared = cluster.join(r, CLUSTER_MODELS[r.model])
            if shared is None:
                raise Down("no server started")
            job = shared[0]
        return cluster.check(r, job)
    if r.kind == "url":
        b = backend(settings, name)
        if b and healthy(b):
            return None
        raise Down(f"{r.url} does not answer")
    conn = read_conn(name)
    state = serve_call(r, "--status")
    if state["state"] == "stopped":
        if conn:
            _forget(name, conn)
        raise Down(f"its server stopped: {state['why']}" if state.get("why") else "not started")
    if conn is None:
        conn = Conn(state["model"], state["port"], state["key"], state["context"], state["mine"])
    conn.port, conn.key, conn.context = state["port"], state["key"], state["context"]
    _tunnel(name, r, conn)
    here = Backend(name, f"http://127.0.0.1:{r.port}", conn.model, conn.key, conn.context)
    if state["state"] == "ready" and healthy(here):
        return None
    return str(state["state"])


def start(
    settings: Settings,
    name: str,
    confirm: Callable[[str], None] = lambda question: None,
    say: Callable[[str], None] = lambda text: None,
) -> str | None:
    """Start remote ``name``'s server, or join a running one; None once it answers, else
    what it is doing. ``confirm`` raises to refuse what spends GPU-hours or stops a server.
    """
    r = settings.remotes[name]
    if r.kind == "url":
        return check(settings, name)
    if r.kind == "ssh":
        return up(settings, name)
    m = CLUSTER_MODELS[r.model]
    job = cluster.read("serve")
    if job and cluster.ended(r, job):
        job = None  # it finished since: start or join a new one
    if job and job.model != m.id:
        confirm(f"{'Stop' if job.owner else 'Leave'} {job.model} and switch to {m.id}?")
        cluster.terminate(r, "serve")
        job = None
    if job is None:
        if key := cluster.unregistered_key():
            raise OcaError(
                "UCloud does not know this machine's SSH key, so the tunnel would fail. "
                f"Register it once: `oca ucloud ssh-keys add {key} --title $(hostname)`"
            )
        if shared := cluster.join(r, m):
            job, starter = shared
            say(f"Joining {m.id}, started by {starter} (job {job.job_id}).")
    if job is None:
        if m.size_gb > 100 and not cluster.weights_ready(r, m):
            raise OcaError(
                f"{m.id} ({m.size_gb:g} GB) is not on the drive yet. Run `oca pull "
                f"{pair_id(settings, name, m.id)}` first, so the download spends no GPU-hours."
            )
        confirm(
            f"Serve {m.id} on {m.gpus}x B200 for up to {r.hours} h ({m.gpus * r.hours} "
            f"GPU-hours max, stops after {r.idle} idle min)?"
        )
        job = cluster.submit_serve(r, m)
        say(f"Job {job.job_id} submitted.")
    return cluster.check(r, job)  # one attempt: it may answer already


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
        *("-N", "-o", "ConnectTimeout=10", "-o", "ExitOnForwardFailure=yes"),
        *("-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"),
        *("-L", f"127.0.0.1:{r.port}:127.0.0.1:{conn.port}"),
        r.host,
    ]
    conn.tunnel_pid = cluster.open_tunnel(cmd, r.port, data_dir() / "logs" / f"tunnel-{name}.log")
    _save(name, conn)


# -- the other machine ---------------------------------------------------------------------


def ssh(
    host: str, command: str, timeout: float = 120, connect: int = 10
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` on ``host``: ``connect`` seconds to connect, ``timeout`` in all."""
    try:
        return subprocess.run(
            [*SSH, "-o", f"ConnectTimeout={connect}", host, command],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise OcaError(f"{host} did not answer within {timeout:.0f} s") from None


def oca_command(r: Remote) -> str:
    """The oca command there: the install for all its users, or the user's own."""
    if not r.shared:
        return '"$HOME/.local/bin/oca"'
    root = shlex.quote(r.shared)
    env = f"OCA_SHARED=1 OCA_DATA={root}/data OCA_CONFIG_DIR={root}/config"
    return f"umask 007; {env} {root}/bin/oca"


def serve_call(r: Remote, args: str, timeout: float = 120, connect: int = 10) -> dict[str, Any]:
    """Run ``oca serve <args>`` there; it answers with one line of JSON."""
    done = ssh(r.host, f"{oca_command(r)} serve {args}", timeout, connect)
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
    gpu: str  # its first NVIDIA GPU, "" without one
    memory: float  # GB that GPU has: its RAM when they share it, as on a DGX Spark
    shared: bool  # oca is installed there for all users
    personal: bool  # this user has their own oca there


def probe(host: str, root: str = SHARED_ROOT) -> Probe:
    q = shlex.quote(root)
    done = ssh(
        host,
        f"uname -m; test -f {q}/shared.toml && echo @shared; "
        'test -x "$HOME/.local/bin/oca" && echo @personal; '
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1;"
        " awk '/^MemTotal:/ {print \"@mem\", $2}' /proc/meminfo; true",
    )
    if done.returncode != 0:
        said = done.stderr.strip().splitlines()
        raise OcaError(
            f"cannot reach {host} over SSH ({said[-1] if said else 'no answer'}). It must work "
            f"without a password: `ssh {host}` once, and `ssh-copy-id {host}` if it asks for one."
        )
    lines = done.stdout.strip().splitlines()
    kib = next((line.split()[1] for line in lines if line.startswith("@mem ")), "0")
    gpus = [line for line in lines[1:] if not line.startswith("@")]
    gpu, _, size = (gpus[0] if gpus else "").partition(", ")
    vram = float(size.split()[0]) * 1.048576 / 1000 if size.endswith(" MiB") else None
    memory = (int(kib) * 1024 / 1e9 if vram is None else vram) if gpu else 0.0  # None: unified
    arch = lines[0] if lines else ""
    return Probe(arch, gpu, memory, "@shared" in lines, "@personal" in lines)


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
            f"of {host} runs once, adding each of them (you too) to the group oca:\n"
            f"  sudo groupadd -f oca && sudo usermod -aG oca <user> && "
            f"sudo install -d -g oca -m 2775 {shared}"
        )
    if done.returncode != 0:
        said = (done.stderr.strip() or done.stdout.strip() or "no output").splitlines()
        raise OcaError(f"installing oca on {host} failed: {said[-1]}")
