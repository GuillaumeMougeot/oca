"""oca's endpoint: one OpenCode model per server; a session stays on the server it asked for.

OpenCode talks to ``127.0.0.1:<port>`` and sees a model per server: ``local`` and each
remote. A request goes to its server; this machine's llama-server starts on demand. When a
remote is down, oca does not switch by itself. It replies with OpenCode's question menu, as
a model asking would: restart that server, use another, or stop. While a server starts, the
reply shows its progress, then carries on with the model's answer. A server picked instead
serves until the requested one answers again, and if it holds less, it sees only the latest
messages that fit. Each change of server is one line in the reply. Models never see oca's
lines, questions and answers. A background check re-opens tunnels once a queued or
restarted server runs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx

from . import agent, remotes, server
from .config import EFFORTS, OcaError, Settings, data_dir, model_path, open_log, process_alive
from .remotes import Backend, Choice

CHECK_S = 5  # a remote's health check stays valid this long
SUPERVISE_S = 60  # how often a down remote is checked again (job state, tunnel)
WAIT_S = 20 * 60  # how long a reply waits for a server to start, before asking again
POLL_S = 10  # how often a starting server is checked; each check keeps the reply alive
HOP = {"connection", "keep-alive", "transfer-encoding"}  # not forwarded to OpenCode
CHARS_PER_TOKEN = 3  # low for code and JSON, so token counts come out high: a safe margin
NOTE = "↪ oca: "  # starts oca's own lines in replies
ASK = "oca: "  # starts the header of oca's own questions


class Router:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.key = agent.api_key()  # what OpenCode must send
        self._lock = threading.Lock()
        self._starts: dict[str, threading.Lock] = {}  # one start per server at a time
        self._seen: dict[str, tuple[float, Backend | None]] = {}
        self.chosen: dict[str, str] = {}  # per requested server that is down: the one picked
        self._told: dict[str, str] = {}  # per requested server: who replied last

    def starting(self, name: str) -> threading.Lock:
        with self._lock:
            return self._starts.setdefault(name, threading.Lock())

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

    def drop(self, name: str) -> None:
        """Check ``name`` again on the next request."""
        self._seen.pop(name, None)

    def local(self) -> Backend:
        """This machine's llama-server, started on demand; raises OcaError when it cannot."""
        model = self.settings.resolved_model
        if not model_path(model).exists():
            raise OcaError(f"no local model; `oca pull {model.id}` downloads it")
        with self.starting("local"):
            server.start(self.settings)
        url = f"http://127.0.0.1:{server.port(self.settings)}"
        return Backend("local", url, model.id, self.key, self.settings.resolved_context)

    def live(self, name: str) -> bool:
        """Whether ``name`` answers now, without starting it."""
        if name == "local":
            state = server.read_state()
            return state is not None and server.healthy(state.port)
        return self.remote(name) is not None

    def serving(self, requested: str) -> Backend | None:
        """Who serves ``requested`` now: it while it answers, else the server picked instead.

        None means ask. Raises OcaError when this machine's server cannot start.
        """
        if requested == "local":
            return self.local()
        if found := self.remote(requested):
            self.chosen.pop(requested, None)
            return found
        picked = self.chosen.get(requested)
        if picked == "local":
            with suppress(OcaError):
                return self.local()
        elif picked and (found := self.remote(picked)):
            return found
        self.chosen.pop(requested, None)
        return None

    def bring_up(self, name: str, progress: Callable[[str], None]) -> Backend:
        """Start server ``name``, or join it, and wait until it answers; raises why not."""
        if name == "local":
            return self.local()
        with self.starting(name):
            doing = remotes.start(self.settings, name)
            deadline = time.monotonic() + WAIT_S
            while doing is not None:
                progress(doing)
                if time.monotonic() > deadline:
                    raise OcaError(f"not ready after {WAIT_S // 60} min ({doing})")
                time.sleep(POLL_S)
                doing = remotes.check(self.settings, name)
        self.drop(name)
        if found := self.remote(name):
            return found
        raise OcaError("it started but does not answer")

    def note(self, requested: str, to: Backend) -> str:
        """A line for the reply when who replies to ``requested`` changed since the last one."""
        before = self._told.get(requested, requested)
        self._told[requested] = to.name
        if to.name == before:
            return ""
        if to.name == requested:
            return f"{requested} answers again."
        said = f"{to.name} replies while {requested} is down"
        if to.context < remotes.context(self.settings, requested):
            return f"{said}; it sees only the latest {to.context // 1000}k tokens."
        return f"{said}."

    def later(self, name: str) -> str:
        """What the user can do about server ``name`` themselves."""
        if name == "local":
            return "`oca start` restarts it; ctrl+x m picks another server."
        if self.settings.remotes[name].kind == "url":
            return "ctrl+x m picks another server."  # someone else's to restart
        return f"`oca remote up {name}` restarts it; ctrl+x m picks another server."

    def question(self, down: str, title: str, failed: tuple[str, ...] = ()) -> dict[str, Any]:
        """oca's question while ``down`` is down, as the arguments of OpenCode's tool."""
        stop = Choice("Stop", f"Nothing now. {self.later(down)}")
        options = [*remotes.choices(self.settings, down, self.live, failed), stop]
        text = f"{title}. What should oca do?".replace('"', "'")  # the answer quotes it
        return {
            "questions": [
                {
                    "question": text,
                    "header": f"{ASK}{down} is down"[:30],
                    "options": [asdict(c) for c in options],
                }
            ]
        }


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
    sized the conversation for a server holding ``room`` tokens and ``to`` holds fewer, the
    reply as long as OpenCode allows ``to`` (a quarter), and only the latest messages that
    fit beside it and the tools. Returns the body and how many messages were left out."""
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
        asked = payload.get("max_tokens") or payload.get("max_completion_tokens") or to.context
        reply = min(int(asked), agent.output_limit(to.context))
        for k in ("max_tokens", "max_completion_tokens"):
            if k in payload:
                payload[k] = reply
        tools = len(json.dumps(payload.get("tools") or [])) // CHARS_PER_TOKEN
        payload["messages"], dropped = fit(payload["messages"], to.context - reply - tools)
    return json.dumps(payload).encode(), dropped


def _why(e: BaseException) -> str:
    """The first line of an error, which says what happened."""
    text = str(e).strip()
    return text.splitlines()[0].rstrip(".") if text else type(e).__name__


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(p.get("text") or "" for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ""


def _ours(call: Any) -> bool:
    """Whether a tool call is one of oca's questions."""
    try:
        fn = call["function"]
        if fn["name"] != "question":
            return False
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
        return bool(args["questions"][0]["header"].startswith(ASK))
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return False


def _our_calls(messages: list[Any]) -> set[str]:
    return {
        str(c.get("id"))
        for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant"
        for c in m.get("tool_calls") or []
        if isinstance(c, dict) and _ours(c)
    }


def picked(messages: list[Any]) -> str | None:
    """The label the user picked in oca's question, when the conversation ends with it ("" for
    an answer that is no option)."""
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or last.get("role") != "tool":
        return None
    if str(last.get("tool_call_id")) not in _our_calls(messages):
        return None
    # OpenCode's answer: User has answered your questions: "<question>"="<label>". ...
    found = re.search(r'"="([^"]*)"', _text(last.get("content")))
    return found.group(1) if found else ""


def _unsay(content: Any) -> Any:
    """``content`` without oca's lines."""

    def clean(text: str) -> str:
        lines = text.splitlines(keepends=True)
        return "".join(line for line in lines if not line.startswith(NOTE)).lstrip("\n")

    if isinstance(content, str):
        return clean(content)
    if isinstance(content, list):
        return [
            {**p, "text": clean(p["text"])}
            if isinstance(p, dict) and isinstance(p.get("text"), str)
            else p
            for p in content
        ]
    return content


def strip(messages: list[Any]) -> list[Any]:
    """``messages`` without oca's questions, their answers and oca's lines in replies."""
    ours, out = _our_calls(messages), []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool" and str(m.get("tool_call_id")) in ours:
            continue
        if isinstance(m, dict) and m.get("role") == "assistant":
            calls = [c for c in m.get("tool_calls") or [] if not _ours(c)]
            content = _unsay(m.get("content"))
            emptied = not calls and not _text(content).strip() and not m.get("reasoning_content")
            if emptied and (m.get("tool_calls") or content != m.get("content")):
                continue  # it held only oca's question or lines
            m = {**m, "content": content}
            if "tool_calls" in m:
                m.pop("tool_calls")
                if calls:
                    m["tool_calls"] = calls
        out.append(m)
    return out


def _tools(payload: Any) -> set[str]:
    """The names of the tools a request offers the model."""
    return {
        str(t["function"].get("name"))
        for t in (payload or {}).get("tools") or []
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }


def _event(model: str, delta: dict[str, Any], finish: str | None = None) -> bytes:
    """One event of a streamed reply."""
    event = {
        "id": "oca",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(event)}\n\n".encode()


def _say(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": f"{NOTE}{text}\n\n"}


class Reply:
    """A streamed reply oca writes itself: its lines, then a question, an end, or a model's
    reply. Writing raises once OpenCode has gone, e.g. after Esc."""

    def __init__(self, out: BaseHTTPRequestHandler, model: str) -> None:
        self.out, self.model = out, model
        out.send_response(200)
        out.send_header("Content-Type", "text/event-stream")
        out.send_header("Cache-Control", "no-cache")
        out.end_headers()

    def send(self, data: bytes) -> None:
        self.out.wfile.write(data)
        self.out.wfile.flush()

    def note(self, text: str) -> None:
        self.send(_event(self.model, _say(text)))

    def ask(self, question: dict[str, Any]) -> None:
        call = {
            "index": 0,
            "id": f"oca_{os.urandom(6).hex()}",
            "type": "function",
            "function": {"name": "question", "arguments": json.dumps(question)},
        }
        self.send(_event(self.model, {"role": "assistant", "tool_calls": [call]}))
        self.send(_event(self.model, {}, "tool_calls") + b"data: [DONE]\n\n")

    def end(self) -> None:
        self.send(_event(self.model, {}, "stop") + b"data: [DONE]\n\n")

    def say(self, text: str) -> None:
        """``text`` as the whole reply."""
        self.note(text)
        self.end()

    def wait(self, name: str, start: Callable[[Callable[[str], None]], Backend]) -> Backend:
        """``start(progress)`` in the background, showing what ``name`` is doing meanwhile."""
        doing: list[str] = []
        done: list[Backend | BaseException] = []

        def run() -> None:
            try:
                done.append(start(doing.append))
            except Exception as e:
                done.append(e)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        shown = ""
        while worker.is_alive():
            worker.join(POLL_S)
            if doing and doing[-1] != shown:
                shown = doing[-1]
                self.note(f"{name}: {shown}")
            else:
                self.send(b": oca\n\n")  # a comment: OpenCode gives up after 5 silent minutes
        if isinstance(done[0], BaseException):
            raise done[0]
        return done[0]


def handler(router: Router) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._handle(b"")

        def do_POST(self) -> None:
            self._handle(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

        def _handle(self, body: bytes) -> None:
            given = self.headers.get("Authorization", "").encode()
            if not hmac.compare_digest(given, f"Bearer {router.key}".encode()):
                return self._error(401, "this endpoint needs oca's key, as in its OpenCode config")
            payload: Any = None
            with suppress(ValueError):
                payload = json.loads(body) if body else None
            if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
                payload = None
            asked = payload.get("model") if payload else None
            servers = router.settings.servers()
            requested = asked if asked in servers else router.settings.default  # "auto": v0.2
            pick = picked(payload["messages"]) if payload else None
            if payload:
                payload["messages"] = strip(payload["messages"])
            why = ""
            if pick is None:
                try:
                    to = router.serving(requested)
                except OcaError as e:
                    to, why = None, _why(e)
                if to and self._forward(requested, to, payload, body):
                    return
                if to:
                    router.chosen.pop(requested, None)  # the server picked went away too
            with suppress(BrokenPipeError, ConnectionResetError):  # OpenCode left, e.g. Esc
                self._recover(requested, pick, payload, body, why)

        def _recover(
            self, requested: str, pick: str | None, payload: Any, body: bytes, why: str
        ) -> None:
            """``requested`` is down, or the user just picked what to do about it."""
            streamed = payload is not None and payload.get("stream") is True
            can_ask = streamed and "question" in _tools(payload)
            later = router.later(requested)
            target, doing = requested, None
            if pick is None and not why:
                try:
                    if requested == "local":
                        raise OcaError("its llama-server does not answer")
                    doing = remotes.check(router.settings, requested)  # re-opens tunnels
                except Exception as e:
                    why = _why(e)
                if not why and doing is None:  # it answers again
                    router.drop(requested)
                    to = router.remote(requested)
                    if to and self._forward(requested, to, payload, body):
                        return
                    why = "it does not answer"
            if pick is None and why:
                title = f"{requested} is not answering: {why}"
                if can_ask:
                    return Reply(self, requested).ask(router.question(requested, title))
                if not streamed:
                    return self._error(424, f"{title}. {later}")  # OpenCode retries 5xx
                return Reply(self, requested).say(f"{title}. {later}")
            if not streamed:
                return self._error(424, f"{requested} is starting ({doing}); try again soon")
            reply = Reply(self, requested)

            def stuck(title: str, failed: str) -> None:
                if can_ask:
                    return reply.ask(router.question(requested, title, (failed,)))
                return reply.say(f"{title}. {later}")

            if pick is not None:
                verb, _, rest = pick.partition(" ")
                target = rest.split(" ")[0]
                if verb not in ("Restart", "Try", "Use") or target not in router.settings.servers():
                    return reply.say(f"{requested} stays down. {later}")
            to = None if target == "local" else router.remote(target)
            if to is None:
                if doing:
                    reply.note(f"{target} is starting ({doing}); this reply waits for it.")
                elif not router.live(target):
                    reply.note(f"starting {target}…")
                try:
                    to = reply.wait(target, lambda progress: router.bring_up(target, progress))
                except (BrokenPipeError, ConnectionResetError):
                    raise
                except Exception as e:
                    return stuck(f"{target} did not start: {_why(e)}", target)
            if target != requested:
                router.chosen[requested] = target
            if not self._forward(requested, to, payload, body, reply):
                router.chosen.pop(requested, None)
                stuck(f"{to.name} does not answer", to.name)

        def _forward(
            self, requested: str, to: Backend, payload: Any, body: bytes, reply: Reply | None = None
        ) -> bool:
            """Send the request to ``to`` and its reply back; False if ``to`` does not answer.

            Without a ``reply`` begun by oca, the server's status and headers pass through.
            """
            # OpenCode compacts for the requested server; a smaller stand-in sees less.
            room = 0 if to.name == requested else remotes.context(router.settings, requested)
            content, _ = _retarget(payload, to, room) if payload else (body, 0)
            # Agent steps carry tools; title and summary requests get no note.
            note = router.note(requested, to) if payload and "tools" in payload else ""
            headers = {
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Accept": self.headers.get("Accept", "*/*"),
                "Accept-Encoding": "identity",
                **({"Authorization": f"Bearer {to.key}"} if to.key else {}),
            }
            url, timeout = to.url + self.path, httpx.Timeout(None, connect=3)
            try:
                with httpx.stream(
                    self.command, url, content=content, headers=headers, timeout=timeout
                ) as r:
                    line = f"{time.strftime('%X')} {self.path} {requested} -> {to.model}\n"
                    sys.stderr.write(line)
                    sse = r.headers.get("content-type", "").startswith("text/event-stream")
                    with suppress(BrokenPipeError, ConnectionResetError, httpx.ReadError):
                        if reply is None:
                            self.send_response(r.status_code)
                            for k, v in r.headers.items():
                                if k.lower() not in HOP:
                                    self.send_header(k, v)
                            self.end_headers()
                        elif r.status_code != 200 or not sse:
                            said = r.read().decode(errors="replace")[:300]
                            reply.note(f"{to.name} answered {r.status_code}: {said}")
                            reply.end()
                            return True
                        if note and sse:
                            self.wfile.write(_event(to.model, _say(note)))
                        for chunk in r.iter_raw():  # tokens stream through as they come
                            self.wfile.write(chunk)
                            self.wfile.flush()
                return True
            except (httpx.ConnectError, httpx.ConnectTimeout):
                router.forget(to.name)  # it went away
                return False

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
    """Re-open tunnels when a queued or restarted server runs; forget servers that ended.

    A down server another one stands in for is watched too, so its sessions return to it.
    """
    while True:
        time.sleep(SUPERVISE_S)
        s = router.settings
        for name, r in s.remotes.items():
            watched = name in router.chosen or remotes.backend(s, name) is not None
            if r.kind == "url" or not watched or router.remote(name):
                continue
            lock = router.starting(name)
            if lock.acquire(blocking=False):  # not while a reply starts it
                try:
                    with suppress(Exception):  # UCloud or SSH hiccups: try again next round
                        remotes.check(s, name)
                finally:
                    lock.release()


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
