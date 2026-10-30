"""oca's endpoint: one stable model for OpenCode, served by the best live server.

OpenCode talks to ``127.0.0.1:<port>`` and sees a single model, ``oca/auto``. Each
request goes to the UCloud server while it answers, else to the local llama-server,
which starts on demand. So a session resumes, and a task carries on, whichever server
is up. A background check re-opens the tunnel once a queued or restarted job runs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path

import httpx

from . import agent, cluster, server
from .config import (
    CLUSTER_MODELS,
    EFFORTS,
    OcaError,
    Settings,
    data_dir,
    open_log,
    process_alive,
)

CHECK_S = 5  # a cluster health check stays valid this long
SUPERVISE_S = 60  # how often a down cluster is checked again (job state, tunnel)
HOP = {"connection", "keep-alive", "transfer-encoding"}  # not forwarded to OpenCode


@dataclass(frozen=True)
class Backend:
    url: str
    model: str
    remote: bool  # the UCloud server, not the local one
    key: str  # what the server wants as a bearer token; "" for none
    efforts: tuple[str | int, ...] = ()


class Router:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.key = agent.api_key()  # what OpenCode must send
        self._start = threading.Lock()  # one local server start at a time
        self._seen: float = -CHECK_S
        self._remote: cluster.Job | None = None

    def remote(self) -> cluster.Job | None:
        """The cluster job while its server answers."""
        if time.monotonic() - self._seen > CHECK_S:
            job = cluster.read("serve")
            self._remote = job if job and cluster.healthy(self.settings) else None
            self._seen = time.monotonic()
        return self._remote

    def forget_remote(self) -> None:
        self._remote, self._seen = None, time.monotonic()

    def backend(self) -> Backend:
        """Where the next request goes: the cluster while it answers, else the local server."""
        if job := self.remote():
            efforts = CLUSTER_MODELS[job.model].efforts if job.model in CLUSTER_MODELS else ()
            url = f"http://127.0.0.1:{self.settings.cluster_port}"
            return Backend(url, job.model, True, job.key, efforts)
        with self._start:
            server.start(self.settings)
        url = f"http://127.0.0.1:{server.port(self.settings)}"
        return Backend(url, self.settings.resolved_model.id, False, self.key)


def _retarget(body: bytes, model: str, efforts: tuple[str | int, ...]) -> bytes:
    """The request body for the backend: its model id, and oca's effort in its own terms."""
    with suppress(ValueError):
        payload = json.loads(body)
        if isinstance(payload, dict) and "model" in payload:
            payload["model"] = model
            level = payload.pop("reasoning_effort", None)
            kwargs = payload.get("chat_template_kwargs") or {}
            if level == "none":
                kwargs["enable_thinking"] = False  # Qwen templates and DeepSeek's encoder
            elif efforts and level in EFFORTS:
                kwargs["reasoning_effort"] = efforts[EFFORTS.index(level) - 1]
            if kwargs:
                payload["chat_template_kwargs"] = kwargs
            return json.dumps(payload).encode()
    return body


def handler(router: Router) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._forward(b"")

        def do_POST(self) -> None:
            self._forward(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

        def _forward(self, body: bytes) -> None:
            given = self.headers.get("Authorization", "").encode()
            if not hmac.compare_digest(given, f"Bearer {router.key}".encode()):
                return self._error(401, "this endpoint needs oca's key, as in its OpenCode config")
            for retry in (False, True):
                try:
                    to = router.backend()
                except OcaError as e:
                    return self._error(503, f"no model is up: {e}")
                headers = {
                    "Content-Type": self.headers.get("Content-Type", "application/json"),
                    "Accept": self.headers.get("Accept", "*/*"),
                    "Accept-Encoding": "identity",
                    **({"Authorization": f"Bearer {to.key}"} if to.key else {}),
                }
                url, timeout = to.url + self.path, httpx.Timeout(None, connect=3)
                try:
                    stream = httpx.stream(
                        self.command, url, content=_retarget(body, to.model, to.efforts),
                        headers=headers, timeout=timeout,
                    )  # fmt: skip
                    with stream as r:
                        sys.stderr.write(f"{time.strftime('%X')} {self.path} -> {to.model}\n")
                        self.send_response(r.status_code)
                        for k, v in r.headers.items():
                            if k.lower() not in HOP:
                                self.send_header(k, v)
                        self.end_headers()
                        with suppress(BrokenPipeError, ConnectionResetError, httpx.ReadError):
                            for chunk in r.iter_raw():  # tokens stream through as they come
                                self.wfile.write(chunk)
                                self.wfile.flush()
                    return
                except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                    if not to.remote or retry:
                        return self._error(503, f"{to.model} unreachable: {e}")
                    router.forget_remote()  # the cluster went away mid-session: go local

        def _error(self, status: int, message: str) -> None:
            data = json.dumps({"error": {"message": f"oca: {message}"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            pass  # _forward logs one line per request

    return Handler


def supervise(router: Router) -> None:
    """Re-open the tunnel when a queued or restarted job runs; forget jobs that ended."""
    while True:
        time.sleep(SUPERVISE_S)
        job = cluster.read("serve")
        if job and router.remote() is None:
            with suppress(Exception):  # UCloud or SSH hiccups: try again next round
                cluster.check(router.settings, job)


def serve(settings: Settings) -> None:
    router = Router(settings)
    threading.Thread(target=supervise, args=(router,), daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", settings.port), handler(router))
    httpd.daemon_threads = True
    httpd.serve_forever()


# -- the background process ----------------------------------------------------------------


def state_path() -> Path:
    return data_dir() / "run" / "router.json"


def log_path() -> Path:
    return data_dir() / "logs" / "router.log"


def _key(settings: Settings) -> str:
    """Changes when the router must restart: new settings or new oca code."""
    code = b"".join(f.read_bytes() for f in sorted(Path(__file__).parent.glob("*.py")))
    raw = json.dumps([asdict(settings), version("oca")], sort_keys=True).encode() + code
    return hashlib.sha256(raw).hexdigest()[:16]


def _alive(pid: int) -> bool:
    return process_alive(pid, "oca.router")


def _listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def running() -> int | None:
    """The router's pid, if it runs."""
    try:
        pid = int(json.loads(state_path().read_text())["pid"])
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None
    return pid if _alive(pid) else None


def ensure(settings: Settings) -> None:
    """Start the router, or restart it if settings changed, and wait until it listens."""
    with suppress(FileNotFoundError, ValueError, KeyError, TypeError):
        if running() and json.loads(state_path().read_text())["key"] == _key(settings):
            return
    stop()
    if (old := server.read_state()) and old.port != server.port(settings):
        server.stop()  # an older oca ran llama-server on the router's port
    with open_log(log_path()) as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "oca.router"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # outlives the terminal that started it
        )
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({"pid": proc.pid, "key": _key(settings)}))
    for _ in range(100):
        if _listening(settings.port):
            return
        if proc.poll() is not None:
            raise OcaError(f"oca's router exited; see {log_path()}")
        time.sleep(0.1)
    raise OcaError(f"oca's router is not listening on port {settings.port}; see {log_path()}")


def stop() -> bool:
    pid = running()
    if pid:
        os.kill(pid, signal.SIGTERM)
    state_path().unlink(missing_ok=True)
    return pid is not None


if __name__ == "__main__":
    serve(Settings.load())
