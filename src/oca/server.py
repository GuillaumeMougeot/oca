"""Run llama-server in the background, one instance, tracked by a state file."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from . import install
from .config import (
    OcaError,
    Settings,
    data_dir,
    model_path,
    open_log,
    process_alive,
    secret,
    secret_path,
)


@dataclass
class State:
    pid: int
    port: int
    cmd: list[str]


def state_path() -> Path:
    return data_dir() / "run" / "llama-server.json"


def log_path() -> Path:
    return data_dir() / "logs" / "llama-server.log"


def port(settings: Settings) -> int:
    """llama-server listens one port above oca's router, which OpenCode talks to."""
    return settings.port + 1


def command(settings: Settings, exe: Path) -> list[str]:
    model = settings.resolved_model
    gpu = settings.resolved_device == "gpu"
    cmd = [
        str(exe),
        "--model", str(model_path(model)),
        "--alias", model.id,
        "--host", "127.0.0.1",
        "--port", str(port(settings)),
        "--api-key-file", str(secret_path("api-key")),  # a file: argv is visible in `ps`
        "--ctx-size", str(settings.resolved_context),
        "--n-gpu-layers", "999" if gpu else "0",
        "--jinja",
        *model.args,
    ]  # fmt: skip
    if settings.threads:
        cmd += ["--threads", str(settings.threads)]
    return cmd


def environment(settings: Settings, exe: Path) -> dict[str, str]:
    env = dict(os.environ)
    # The release ships its shared libs next to the binary, the CUDA runtime in a sibling dir.
    lib_dirs = [exe.parent, *sorted({p.parent for p in install.llama_dir().rglob("*.so*")})]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        dict.fromkeys(p for p in (*map(str, lib_dirs), env.get("LD_LIBRARY_PATH", "")) if p)
    )
    if settings.resolved_device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # keep the CUDA build off a shared GPU
    return env


def read_state() -> State | None:
    try:
        state = State(**json.loads(state_path().read_text()))
    except (FileNotFoundError, ValueError, TypeError):
        return None
    return state if _alive(state.pid) else None


def _alive(pid: int) -> bool:
    return process_alive(pid, "llama-server")


def healthy(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def log_tail(lines: int = 20) -> str:
    try:
        return "\n".join(log_path().read_text(errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return ""


def start(settings: Settings, timeout: float = 900) -> State:
    """Start (or restart, if settings changed) the server and wait until it serves."""
    exe = install.llama_server()
    if exe is None:
        raise OcaError("llama.cpp is not installed; run `oca setup`")
    if not model_path(settings.resolved_model).exists():
        raise OcaError(f"model missing; run `oca pull {settings.resolved_model.id}`")
    secret("api-key")  # make it before llama-server reads it
    cmd = command(settings, exe)
    state = read_state()
    if state and state.cmd == cmd:
        return _wait(state, None, timeout)
    if state:
        stop()

    with open_log(log_path()) as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment(settings, exe),
            start_new_session=True,  # survives the terminal that started it
        )
    state = State(proc.pid, port(settings), cmd)
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(asdict(state)))
    return _wait(state, proc, timeout)


def _wait(state: State, proc: subprocess.Popen[bytes] | None, timeout: float) -> State:
    deadline = time.monotonic() + timeout
    while not healthy(state.port):
        if (proc and proc.poll() is not None) or not _alive(state.pid):
            raise OcaError(f"llama-server exited; log tail:\n{log_tail()}")
        if time.monotonic() > deadline:
            raise OcaError(f"llama-server not ready after {timeout:.0f}s; see `oca logs`")
        time.sleep(1)
    return state


def stop() -> bool:
    state = read_state()
    if state is None:
        return False
    os.kill(state.pid, signal.SIGTERM)
    for _ in range(100):
        if not _alive(state.pid):
            break
        time.sleep(0.1)
    else:
        os.kill(state.pid, signal.SIGKILL)
    state_path().unlink(missing_ok=True)
    return True
