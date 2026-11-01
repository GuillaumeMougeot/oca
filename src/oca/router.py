"""oca's endpoint: one OpenCode model per server, each served by the best live server.

OpenCode talks to ``127.0.0.1:<port>`` and sees a model per server, ``local`` and each
remote. A request goes to its server while that answers, else to the next that does: the
default server, the other remotes by context window, then this machine's llama-server,
which starts on demand. The reply then says so in one line, and a smaller server sees
only the latest messages that fit. So a session carries on whichever server is up. A
background check re-opens tunnels once a queued or restarted server runs.
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
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx

from . import agent, remotes, server
from .config import EFFORTS, OcaError, Settings, data_dir, model_path, open_log, process_alive
from .remotes import Backend

CHECK_S = 5  # a remote's health check stays valid this long
SUPERVISE_S = 60  # how often a down remote is checked again (job state, tunnel)
HOP = {"connection", "keep-alive", "transfer-encoding"}  # not forwarded to OpenCode
CHARS_PER_TOKEN = 3  # low for code and JSON, so token counts come out high: a safe margin


class Router:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.key = agent.api_key()  # what OpenCode must send
        self._start = threading.Lock()  # one local server start at a time
        self._seen: dict[str, tuple[float, Backend | None]] = {}
        self._told: dict[str, tuple[str, bool]] = {}  # per requested server: who replied, trimmed

    def order(self, requested: str) -> list[str]:
        """Servers to try for ``requested``: it, the default, other remotes by context, local."""
        s = self.settings
        others = sorted(s.remotes, key=lambda name: -remotes.context(s, name))
        return list(dict.fromkeys([requested, s.default, *others, "local"]))

    def remote(self, name: str) -> Backend | None:
        """Remote ``name`` while it answers."""
        at, found = self._seen.get(name, (-CHECK_S, None))
        if time.monotonic() - at > CHECK_S:
            found = remotes.backend(self.settings, name)
            found = found if found and remotes.healthy(found) else None
            self._seen[name] = (time.monotonic(), found)
        return found

    def forget(self, name: str) -> None:
        self._seen[name] = (time.monotonic(), None)

    def local(self) -> Backend | None:
        """This machine's llama-server, started on demand; None without a downloaded model."""
        model = self.settings.resolved_model
        if not model_path(model).exists():
            return None
        with self._start:
            server.start(self.settings)
        url = f"http://127.0.0.1:{server.port(self.settings)}"
        return Backend("local", url, model.id, self.key, self.settings.resolved_context)

    def backend(self, requested: str, skip: set[str]) -> Backend:
        """The first server in ``order(requested)`` that answers."""
        for name in self.order(requested):
            found = None if name in skip else self.local() if name == "local" else self.remote(name)
            if found:
                return found
        raise OcaError("no server answers; start one with `oca remote up` or `oca start`")

    def note(self, requested: str, to: Backend, dropped: int) -> str:
        """A line for the reply when who replies to ``requested`` changed since the last one."""
        now, before = (to.name, dropped > 0), self._told.get(requested, (requested, False))
        self._told[requested] = now
        if now == before:
            return ""
        if to.name == requested:
            return f"↪ oca: {requested} replies again."
        said = f"↪ oca: {requested} is not answering, so {to.name} replies"
        return said + (f", seeing the latest {to.context // 1000}k tokens." if dropped else ".")


def fit(messages: list[Any], budget: int) -> tuple[list[Any], int]:
    """The latest ``messages`` that fit ``budget`` tokens; how many were left out.

    The system messages stay, and the kept part starts at a user message, so no tool result
    loses its call. Its first message says that earlier ones were left out.
    """

    def size(m: Any) -> int:
        return len(json.dumps(m)) // CHARS_PER_TOKEN

    head = 0
    while head < len(messages) and isinstance(messages[head], dict):
        if messages[head].get("role") != "system":
            break
        head += 1
    rest = messages[head:]
    room = budget - sum(size(m) for m in messages[:head])
    users = [i for i, m in enumerate(rest) if isinstance(m, dict) and m.get("role") == "user"]
    total, cut = sum(size(m) for m in rest), 0
    while total > room and users and cut < users[-1]:
        total -= size(rest[cut])
        cut += 1
    while cut and cut < len(rest) and rest[cut].get("role") != "user":
        total -= size(rest[cut])
        cut += 1
    if not cut:
        return messages, 0
    first = dict(rest[cut])
    note = f"[oca: the {cut} earlier messages of this session were left out to fit this model.]"
    content = first.get("content")
    if isinstance(content, list):
        first["content"] = [{"type": "text", "text": note}, *content]
    else:
        first["content"] = f"{note}\n\n{content or ''}"
    return [*messages[:head], first, *rest[cut + 1 :]], cut


def _retarget(payload: dict[str, Any], to: Backend, room: int = 0) -> tuple[bytes, int]:
    """The request for ``to``: its model id, and oca's effort in its own terms. When OpenCode
    sized the conversation for a server holding ``room`` tokens and ``to`` holds fewer, only
    the latest messages that fit. Returns the body and how many messages were left out."""
    payload = {**payload, "model": to.model}
    level = payload.pop("reasoning_effort", None)
    kwargs = payload.get("chat_template_kwargs") or {}
    if level == "none":
        kwargs["enable_thinking"] = False  # Qwen templates and DeepSeek's encoder
    elif to.efforts and level in EFFORTS:
        kwargs["reasoning_effort"] = to.efforts[EFFORTS.index(level) - 1]
    if kwargs:
        payload["chat_template_kwargs"] = kwargs
    dropped = 0
    if 0 < to.context < room and isinstance(payload.get("messages"), list):
        reply = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 0)
        budget = to.context - (reply or to.context // 4)
        payload["messages"], dropped = fit(payload["messages"], budget)
    return json.dumps(payload).encode(), dropped


def _chunk(model: str, text: str) -> bytes:
    """``text`` as the first event of a streamed reply."""
    delta = {"role": "assistant", "content": f"{text}\n\n"}
    event = {
        "id": "oca-note",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(event)}\n\n".encode()


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
            payload: Any = None
            with suppress(ValueError):
                payload = json.loads(body) if body else None
            if not isinstance(payload, dict):
                payload = None
            asked = payload.get("model") if payload else None
            servers = router.settings.servers()
            requested = asked if asked in servers else router.settings.default  # "auto": v0.2
            skip: set[str] = set()
            while True:
                try:
                    to = router.backend(requested, skip)
                except OcaError as e:
                    return self._error(503, f"no model is up: {e}")
                # OpenCode compacts for the requested server; a smaller stand-in sees less.
                room = 0 if to.name == requested else remotes.context(router.settings, requested)
                content, dropped = _retarget(payload, to, room) if payload else (body, 0)
                # Agent steps carry tools; title and summary requests get no note.
                note = router.note(requested, to, dropped) if payload and "tools" in payload else ""
                headers = {
                    "Content-Type": self.headers.get("Content-Type", "application/json"),
                    "Accept": self.headers.get("Accept", "*/*"),
                    "Accept-Encoding": "identity",
                    **({"Authorization": f"Bearer {to.key}"} if to.key else {}),
                }
                url, timeout = to.url + self.path, httpx.Timeout(None, connect=3)
                try:
                    stream = httpx.stream(
                        self.command, url, content=content, headers=headers, timeout=timeout
                    )
                    with stream as r:
                        line = f"{time.strftime('%X')} {self.path} {requested} -> {to.model}\n"
                        sys.stderr.write(line)
                        self.send_response(r.status_code)
                        for k, v in r.headers.items():
                            if k.lower() not in HOP:
                                self.send_header(k, v)
                        self.end_headers()
                        with suppress(BrokenPipeError, ConnectionResetError, httpx.ReadError):
                            if note and r.headers.get("content-type", "").startswith(
                                "text/event-stream"
                            ):
                                self.wfile.write(_chunk(to.model, note))
                            for chunk in r.iter_raw():  # tokens stream through as they come
                                self.wfile.write(chunk)
                                self.wfile.flush()
                    return
                except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                    if to.name == "local":
                        return self._error(503, f"{to.model} unreachable: {e}")
                    router.forget(to.name)  # it went away mid-session: try the next server
                    skip.add(to.name)

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
    """Re-open tunnels when a queued or restarted server runs; forget servers that ended."""
    while True:
        time.sleep(SUPERVISE_S)
        for name, r in router.settings.remotes.items():
            if r.kind != "url" and router.remote(name) is None:
                with suppress(Exception):  # UCloud or SSH hiccups: try again next round
                    remotes.check(router.settings, name)


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
