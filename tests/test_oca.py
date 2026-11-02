"""Offline tests: no network, no GPU, no real server."""

from __future__ import annotations

import io
import json
import os
import re
import socket
import subprocess
import tarfile
import threading
import zipfile
from contextlib import nullcontext
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from oca import agent, config, install, server
from oca.config import Down, OcaError, Remote, Settings
from oca.remotes import Backend

UCLOUD = Remote("ucloud", model="deepseek-v4.1-flash", port=8001, drive="/9")
SPARK = Remote("ssh", model="qwen3.6-35b-a3b", port=8002, host="spark", shared="/opt/oca")


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCA_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("OCA_CONFIG_DIR", str(tmp_path / "config"))


@pytest.fixture
def gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "has_nvidia_gpu", lambda: True)


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "has_nvidia_gpu", lambda: False)


def test_auto_device_and_model_follow_hardware(gpu: None) -> None:
    s = Settings()
    assert s.resolved_device == "gpu"
    assert s.resolved_model.id == config.DEFAULT_MODEL["gpu"]
    s.set("device", "cpu")
    assert s.resolved_model.id == config.DEFAULT_MODEL["cpu"]


def test_settings_roundtrip_and_validation() -> None:
    s = Settings()
    s.set("port", "9000")
    s.set("model", "qwen3.8-27b")
    s.save()
    loaded = Settings.load()
    assert (loaded.port, loaded.model) == (9000, "qwen3.8-27b")
    for key, value in [("device", "tpu"), ("model", "nope"), ("port", "x"), ("colour", "red")]:
        with pytest.raises(OcaError):
            Settings().set(key, value)


def test_cpu_server_hides_gpu_and_offloads_nothing(gpu: None) -> None:
    s = Settings(device="cpu", threads=8)
    exe = Path("/opt/llama/llama-server")
    cmd = server.command(s, exe)
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "0"
    assert cmd[cmd.index("--threads") + 1] == "8"
    assert cmd[cmd.index("--alias") + 1] == s.resolved_model.id
    env = server.environment(s, exe)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["LD_LIBRARY_PATH"].startswith("/opt/llama")


def test_gpu_server_offloads_all(gpu: None) -> None:
    s = Settings(device="gpu")
    cmd = server.command(s, Path("llama-server"))
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "999"
    env = server.environment(s, Path("llama-server"))
    assert env.get("CUDA_VISIBLE_DEVICES") == os.environ.get("CUDA_VISIBLE_DEVICES")


def test_opencode_sees_one_model_per_server(no_gpu: None) -> None:
    lab = Remote("url", model="m", url="http://lab:8000", context=32768)
    s = Settings(port=8123, default="ucloud", remotes={"ucloud": UCLOUD, "lab": lab})
    cfg = agent.opencode_config(s, titles=True)
    provider = cfg["provider"][agent.PROVIDER]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8123/v1"
    assert cfg["model"] == cfg["small_model"] == f"{agent.PROVIDER}/ucloud"  # the default
    assert cfg["enabled_providers"] == [agent.PROVIDER]
    models = provider["models"]
    assert list(models) == ["local", "ucloud", "lab"]
    assert models["ucloud"]["name"] == "deepseek@ucloud"  # what each serves, as `oca ls` says
    assert models["local"]["name"] == "qwen3.6@local" and models["lab"]["name"] == "m@lab"
    # Each server's own window, so OpenCode compacts a session for the one it uses.
    assert models["lab"]["limit"] == {"context": 32768, "output": 8192}
    assert models["ucloud"]["limit"]["context"] == config.CLUSTER_MODELS[UCLOUD.model].context
    assert cfg["permission"]["webfetch"] == "ask"
    assert models["local"]["variants"]["max"] == {"reasoningEffort": "max"}
    assert cfg["agent"]["title"] == {"model": "oca/ucloud", "variant": "none"}  # no thinking
    assert agent.opencode_config(s)["agent"]["title"] == {"disable": True}  # not on CPU


def test_old_cluster_settings_become_the_default_ucloud_remote() -> None:
    Settings.path().parent.mkdir(parents=True)
    Settings.path().write_text(
        'device = "cpu"\ncluster_drive = "/9"\ncluster_model = "deepseek-v4.1-flash"\n'
        "cluster_idle = 30\n"
    )
    s = Settings.load()
    assert s.default == "ucloud" and s.device == "cpu"
    assert s.remotes == {"ucloud": Remote("ucloud", "deepseek-v4.1-flash", 8001, 30, "/9", 8)}
    s.save()
    assert "cluster_" not in Settings.path().read_text() and Settings.load() == s
    Settings.path().write_text('cluster_model = "qwen3.8-27b-fp8"\n')  # no drive: no remote
    assert Settings.load() == Settings()


def test_remote_names_kinds_and_models_are_checked() -> None:
    bad = [
        Settings(remotes={"Spark!": SPARK}),
        Settings(remotes={"local": SPARK}),
        Settings(remotes={"x": Remote("ssh", model="nope", host="h")}),
        Settings(remotes={"x": Remote("ftp")}),
        Settings(default="nope"),
    ]
    for s in bad:
        with pytest.raises(OcaError):
            s.validate()
    Settings(default="spark", remotes={"spark": SPARK}).validate()


def test_unattended_runs_refuse_instead_of_asking() -> None:
    def values(rules: Any) -> list[str]:
        if isinstance(rules, dict):
            return [x for v in rules.values() for x in values(v)]
        return [rules]

    agent.write_config(Settings())
    interactive = json.loads(agent.config_path().read_text())["permission"]
    unattended = json.loads(agent.config_path(unattended=True).read_text())["permission"]
    assert "ask" in values(interactive) and "ask" not in values(unattended)
    assert unattended["bash"]["rm *"] == "deny" and unattended["bash"]["*"] == "allow"
    assert unattended["question"] == "deny" and "question" not in interactive


def test_router_translates_effort_for_each_model() -> None:
    from oca import router

    def send(level: str, model: str, efforts: tuple[str | int, ...]) -> dict[str, Any]:
        to = Backend("x", "http://x", model, "", 0, efforts)
        body, dropped = router._retarget({"model": "ucloud", "reasoning_effort": level}, to)
        out: dict[str, Any] = json.loads(body)
        assert out["model"] == model and "reasoning_effort" not in out and dropped == 0
        return out.get("chat_template_kwargs", {})

    flash = config.CLUSTER_MODELS["qwen3.8-flash-next"].efforts
    deepseek = config.CLUSTER_MODELS["deepseek-v4.1-flash"].efforts
    assert send("max", "qwen3.8-flash-next", flash) == {"reasoning_effort": "xhigh"}
    assert send("medium", "deepseek-v4.1-flash", deepseek) == {"reasoning_effort": 62}
    assert send("none", "deepseek-v4.1-flash", deepseek) == {"enable_thinking": False}
    assert send("high", "qwen3.6-35b-a3b", ()) == {}  # local: thinking stays on


def test_llama_assets() -> None:
    tag = config.LLAMA_CPP_TAG
    assert install.llama_assets("linux", "x64", False) == [f"llama-{tag}-bin-ubuntu-x64.tar.gz"]
    cuda = install.llama_assets("linux", "x64", True)
    assert len(cuda) == 2 and all("cuda-12.8-x64" in a for a in cuda)
    assert install.llama_assets("macos", "arm64", False) == [f"llama-{tag}-bin-macos-arm64.tar.gz"]
    assert install.opencode_asset("macos", "arm64") == "opencode-darwin-arm64.zip"
    assert install.opencode_asset("linux", "x64") == "opencode-linux-x64.tar.gz"


def test_zip_install_keeps_executables_and_stays_inside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("opencode")
        info.external_attr = 0o755 << 16
        zf.writestr(info, b"x")
        zf.writestr("../escape", b"x")
    monkeypatch.setattr(install, "_download", lambda url, dest: dest.write_bytes(buf.getvalue()))
    dest = tmp_path / "bin" / "opencode"
    install.install_release(["opencode-darwin-arm64.zip"], dest)
    assert os.access(dest / "opencode", os.X_OK)
    assert not (tmp_path / "bin" / "escape").exists()


def _tarball(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_install_release_is_atomic_and_rejects_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "good": _tarball({"pkg/bin/llama-server": b"x"}),
        "evil": _tarball({"../escape": b"x"}),
    }
    monkeypatch.setattr(install, "_download", lambda url, dest: dest.write_bytes(payloads[url]))
    dest = tmp_path / "bin" / "llama"
    install.install_release(["good"], dest)
    assert install._find(dest, "llama-server") is not None

    with pytest.raises(tarfile.TarError):
        install.install_release(["evil"], tmp_path / "bin" / "evil")
    assert not (tmp_path / "bin" / "evil").exists()
    assert not (tmp_path / "bin" / "escape").exists()


def test_files_with_keys_are_private_and_logs_rotate(monkeypatch: pytest.MonkeyPatch) -> None:
    from oca import cluster, remotes

    agent.write_config(Settings())
    cluster._write(cluster.Job("serve", "1", "m", "t", key="k"), "serve")
    remotes._save("spark", remotes.Conn("m", 8181, "k", 1, True))
    Settings(remotes={"lab": Remote("url", model="m", url="http://x", key="k")}).save()
    for path in (
        *(agent.config_path(), agent.config_path(unattended=True), cluster.state_path()),
        *(remotes.conn_path("spark"), Settings.path()),
    ):
        assert path.stat().st_mode & 0o077 == 0, path
    assert (config.config_dir() / "api-key").stat().st_mode & 0o077 == 0
    monkeypatch.setattr(config, "LOG_LIMIT", 10)
    log = config.data_dir() / "logs" / "x.log"
    for _ in range(2):
        with config.open_log(log) as fh:
            fh.write(b"more than ten bytes")
    assert log.with_name("x.log.1").exists() and log.stat().st_size == 19


def test_an_ended_job_is_forgotten_so_up_starts_anew(ucloud: dict[str, Any]) -> None:
    from oca import cluster

    job = cluster.Job("serve", "42", "m", "t")
    cluster._write(job, "serve")
    assert not cluster.ended(UCLOUD, job) and cluster.read("serve")
    ucloud["state"] = "SUCCESS"
    assert cluster.ended(UCLOUD, job) and cluster.read("serve") is None


def test_sessions_in_a_folder_exclude_other_folders(monkeypatch: pytest.MonkeyPatch) -> None:
    listed = [
        {"id": "a", "directory": "/w/p", "updated": 1},
        {"id": "b", "directory": "/w/p/sub", "updated": 2},
        {"id": "c", "directory": "/w/project2", "updated": 3},  # shares the prefix only
        {"id": "d", "directory": "/elsewhere", "updated": 4},
    ]
    monkeypatch.setattr(agent, "opencode", lambda args: json.dumps(listed))
    assert [s["id"] for s in agent.sessions(Path("/w/p"))] == ["b", "a"]  # the latest first


def test_stale_web_state_is_ignored() -> None:
    from oca import web

    web.state_path().parent.mkdir(parents=True)
    web.state_path().write_text('{"pid": 999999999, "port": 4096, "project": "/x"}')
    assert web.running() is None and web.stop() is False


def test_stale_state_is_ignored() -> None:
    server.state_path().parent.mkdir(parents=True)
    server.state_path().write_text('{"pid": 999999999, "port": 8080, "cmd": []}')
    assert server.read_state() is None
    assert server.stop() is False


def test_permissions_allow_work_ask_before_damage_deny_administration() -> None:
    cfg = agent.opencode_config(Settings())
    bash = cfg["permission"]["bash"]
    assert next(iter(bash.items())) == ("*", "allow")  # first, so later rules override it
    assert bash["rm *"] == bash["git push*"] == bash["curl *"] == "ask"
    assert bash["sudo *"] == bash["*--break-system-packages*"] == "deny"
    assert list(bash)[-1] in {k for k, v in bash.items() if v == "deny"}  # denies win
    assert cfg["permission"]["external_directory"] == cfg["permission"]["webfetch"] == "ask"


def test_serve_script_is_valid_bash_and_stops_when_idle(tmp_path: Path) -> None:
    from oca import cluster

    r = Remote("ucloud", idle=45)
    for model in config.CLUSTER_MODELS.values():
        script = cluster.serve_script(model, r, "serve-x")
        path = tmp_path / "serve.sh"
        path.write_text(script)
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
        assert "-ge 45 ]" in script and "/work/oca/logs/serve-x.log" in script
        assert f"--served-model-name {model.id}" in script
        assert "--host 127.0.0.1" in script  # reachable only through the SSH tunnel
        assert f"vllm=={config.VLLM_VERSION}" in script
        assert f"rm -f /work/oca/servers/{model.id}.json" in script  # unlisted on exit
        assert "trap unlist EXIT" in script and "export CUDA_HOME=" in script
        keyed = cluster.serve_script(model, r, "serve-x", "k3y")
        assert "--api-key k3y" in keyed and "Bearer k3y" in keyed and "--api-key" not in script


def _fake_server(
    seen: list[str], bodies: list[Any] | None = None, port: int = 0
) -> ThreadingHTTPServer:
    """An OpenAI-like server: /health answers, a completion streams three events."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(f"{body['model']} {self.headers.get('Authorization')}")
            if bodies is not None:
                bodies.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: {i}\n\n".encode())
                self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


ASK_TOOL = {"type": "function", "function": {"name": "question", "parameters": {}}}


def _events(text: str) -> list[Any]:
    """The JSON events of a streamed reply."""
    return [json.loads(e[6:]) for e in text.split("\n\n") if e.startswith("data: {")]


def _question(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The tool call that ends a reply of oca's, and the question in it."""
    call = _events(text)[-2]["choices"][0]["delta"]["tool_calls"][0]
    return call, json.loads(call["function"]["arguments"])["questions"][0]


def _answered(messages: list[Any], call: dict[str, Any], label: str) -> list[Any]:
    """``messages`` after the user picked ``label``, as OpenCode sends them."""
    said = f'User has answered your questions: "q"="{label}". You can now continue.'
    return [
        *messages,
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": said},
    ]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_router_asks_before_another_server_replies_and_returns_when_back(
    monkeypatch: pytest.MonkeyPatch, no_gpu: None
) -> None:
    import httpx

    from oca import cluster, remotes, router

    remote_seen: list[str] = []
    local_seen: list[str] = []
    local_bodies: list[Any] = []
    remote, local = _fake_server(remote_seen), _fake_server(local_seen, local_bodies)
    port = _free_port()
    ucloud = Remote("ucloud", model="deepseek-v4.1-flash", port=remote.server_address[1])
    s = Settings(port=port, default="ucloud", context=65536, remotes={"ucloud": ucloud})
    config.model_path(s.resolved_model).parent.mkdir(parents=True)
    config.model_path(s.resolved_model).touch()
    job = cluster.Job("serve", "1", "deepseek-v4.1-flash", "t", key="job-key")
    monkeypatch.setattr(cluster, "read", lambda kind: job)
    monkeypatch.setattr(server, "start", lambda settings: None)
    monkeypatch.setattr(server, "port", lambda settings: local.server_address[1])
    monkeypatch.setattr(router, "CHECK_S", -1)  # every request checks health again
    rt = router.Router(s)
    endpoint = ThreadingHTTPServer(("127.0.0.1", port), router.handler(rt))
    threading.Thread(target=endpoint.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    ask = [{"role": "user", "content": "fix it"}]
    step = {"model": "ucloud", "stream": True, "messages": ask, "tools": [ASK_TOOL]}
    assert httpx.post(url, json=step).status_code == 401  # another user of this machine
    key = {"Authorization": f"Bearer {agent.api_key()}"}
    assert httpx.post(url, json=step, headers=key).text == "data: 0\n\ndata: 1\n\ndata: 2\n\n"
    old = {"model": "auto", "stream": True, "messages": []}  # a session from before remotes
    assert httpx.post(url, json=old, headers=key).text.startswith("data: 0")

    remote.shutdown()
    remote.server_close()  # the job ends mid-session: oca asks, it does not switch
    gone = Down("job 1 ended (idle for 60 min). Log tail:\n...")
    monkeypatch.setattr(remotes, "check", lambda settings, name: (_ for _ in ()).throw(gone))
    asked = httpx.post(url, json=step, headers=key).text
    call, question = _question(asked)
    assert question["header"] == "oca: ucloud is down" and call["id"].startswith("oca_")
    assert question["question"].startswith("ucloud is not answering: job 1 ended (idle for 60 min)")
    labels = [o["label"] for o in question["options"]]
    assert labels == ["Restart ucloud", "Use local", "Stop"] and not local_seen
    assert "16 GPU-hours" in question["options"][0]["description"]
    assert _events(asked)[-1]["choices"][0]["finish_reason"] == "tool_calls"

    picked = {**step, "messages": _answered(ask, call, "Use local")}
    moved = httpx.post(url, json=picked, headers=key).text
    notes = [e["choices"][0]["delta"]["content"] for e in _events(moved)]
    assert notes[0] == "↪ oca: starting local…\n\n"
    assert notes[1] == (
        "↪ oca: local replies while ucloud is down; it sees only the latest 65k tokens.\n\n"
    )
    assert moved.endswith("data: 0\n\ndata: 1\n\ndata: 2\n\n")
    assert local_bodies[-1]["messages"] == ask  # the model never sees oca's question
    assert httpx.post(url, json=step, headers=key).text == "data: 0\n\ndata: 1\n\ndata: 2\n\n"

    back = _fake_server(remote_seen, port=ucloud.port)  # someone restarted it
    returned = httpx.post(url, json=step, headers=key).text
    assert _events(returned)[0]["choices"][0]["delta"]["content"] == (
        "↪ oca: ucloud answers again.\n\n"
    )
    assert returned.endswith("data: 0\n\ndata: 1\n\ndata: 2\n\n") and not rt.chosen
    assert remote_seen == ["deepseek-v4.1-flash Bearer job-key"] * 3  # each server, its key
    assert local_seen == [f"{s.resolved_model.id} Bearer {agent.api_key()}"] * 2
    for httpd in (endpoint, local, back):
        httpd.shutdown()


def test_router_says_why_a_restart_failed_and_offers_the_rest(
    monkeypatch: pytest.MonkeyPatch, no_gpu: None
) -> None:
    import httpx

    from oca import remotes, router

    port = _free_port()
    s = Settings(port=port, default="spark", remotes={"spark": SPARK})  # no local model
    stopped = Down("its server stopped: idle for 60 min, at 14:02")
    unreachable = OcaError("cannot reach spark over SSH (No route to host).\nMore")
    monkeypatch.setattr(remotes, "check", lambda settings, name: (_ for _ in ()).throw(stopped))
    monkeypatch.setattr(remotes, "start", lambda settings, name: (_ for _ in ()).throw(unreachable))
    endpoint = ThreadingHTTPServer(("127.0.0.1", port), router.handler(router.Router(s)))
    threading.Thread(target=endpoint.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    key = {"Authorization": f"Bearer {agent.api_key()}"}
    ask = [{"role": "user", "content": "fix it"}]
    step = {"model": "spark", "stream": True, "messages": ask, "tools": [ASK_TOOL]}
    call, question = _question(httpx.post(url, json=step, headers=key).text)
    assert [o["label"] for o in question["options"]] == ["Restart spark", "Stop"]

    picked = {**step, "messages": _answered(ask, call, "Restart spark")}
    failed = httpx.post(url, json=picked, headers=key).text
    assert _events(failed)[0]["choices"][0]["delta"]["content"] == "↪ oca: starting spark…\n\n"
    call, question = _question(failed)
    assert question["question"] == (
        "spark did not start: cannot reach spark over SSH (No route to host). What should oca do?"
    )
    assert [o["label"] for o in question["options"]] == ["Try spark again", "Stop"]

    stop = httpx.post(url, json={**step, "messages": _answered(ask, call, "Stop")}, headers=key)
    said = _events(stop.text)
    assert said[0]["choices"][0]["delta"]["content"].startswith(
        "↪ oca: spark stays down. `oca qwen3.6@spark` restarts it"
    )
    assert said[-1]["choices"][0]["finish_reason"] == "stop"
    run = httpx.post(url, json={**step, "tools": []}, headers=key).text  # `opencode run`
    assert _events(run)[0]["choices"][0]["delta"]["content"] == (
        "↪ oca: spark is not answering: its server stopped: idle for 60 min, at 14:02. "
        "`oca qwen3.6@spark` restarts it; ctrl+x m picks another server.\n\n"
    )
    title = httpx.post(url, json={**step, "stream": False}, headers=key)
    assert title.status_code == 424 and "not answering" in title.json()["error"]["message"]
    endpoint.shutdown()


def test_a_reply_waits_for_a_starting_server_and_shows_its_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from oca import remotes, router

    seen: list[str] = []
    spark = _fake_server(seen)
    port = _free_port()
    s = Settings(port=port, default="spark", remotes={"spark": SPARK})
    states = iter(["loading", "loading"])  # what checks answer before it is ready
    ready: list[bool] = []

    def check(settings: Settings, name: str) -> str | None:
        state = next(states, None)
        ready.append(state is None)
        return state

    def backend(settings: Settings, name: str) -> Backend | None:
        here = f"http://127.0.0.1:{spark.server_address[1]}"
        return Backend("spark", here, "m", "", 262144) if any(ready) else None

    monkeypatch.setattr(remotes, "check", check)
    monkeypatch.setattr(remotes, "start", lambda settings, name: "loading")
    monkeypatch.setattr(remotes, "backend", backend)
    monkeypatch.setattr(router, "POLL_S", 0.05)
    endpoint = ThreadingHTTPServer(("127.0.0.1", port), router.handler(router.Router(s)))
    threading.Thread(target=endpoint.serve_forever, daemon=True).start()
    step = {"model": "spark", "stream": True, "messages": [], "tools": [ASK_TOOL]}
    key = {"Authorization": f"Bearer {agent.api_key()}"}
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    waited = httpx.post(url, json=step, headers=key).text
    notes = [e["choices"][0]["delta"]["content"] for e in _events(waited)]
    assert notes == [
        "↪ oca: spark is starting (loading); this reply waits for it.\n\n",
        "↪ oca: spark: loading\n\n",
    ]
    assert ": oca\n\n" in waited  # keeps OpenCode from giving up meanwhile
    assert waited.endswith("data: 0\n\ndata: 1\n\ndata: 2\n\n") and len(seen) == 1
    endpoint.shutdown()
    spark.shutdown()


def test_models_never_see_oca_questions_answers_or_lines() -> None:
    from oca import router

    def call(cid: str, header: str) -> dict[str, Any]:
        args = {"questions": [{"header": header, "question": "q", "options": []}]}
        fn = {"name": "question", "arguments": json.dumps(args)}
        return {"id": cid, "type": "function", "function": fn}

    said = 'User has answered your questions: "q"="Use local". You can now continue.'
    theirs = "↪ oca: local replies.\n\nDone."
    turns = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [call("oca_1", "oca: spark is down")]},
        {"role": "tool", "tool_call_id": "oca_1", "content": said},
        {"role": "assistant", "content": theirs, "tool_calls": [call("2", "Which database?")]},
        {"role": "tool", "tool_call_id": "2", "content": said},
    ]
    assert router.picked(turns[:3]) == "Use local"
    assert router.picked(turns) is None  # the model's own question
    assert router.picked([*turns[:2], {**turns[2], "content": "Error: dismissed"}]) == ""
    assert router.strip(turns) == [turns[0], {**turns[3], "content": "Done."}, turns[4]]
    r = router.Router(Settings(remotes={"spark": SPARK}))
    small = Backend("local", "u", "m", "", 65536)
    assert r.note("spark", small) == (
        "local replies while spark is down; it sees only the latest 65k tokens."
    )
    assert r.note("spark", small) == ""  # told once
    assert r.note("spark", Backend("spark", "u", "m", "", 262144)) == "spark answers again."


def test_a_down_server_offers_its_restart_then_running_servers_then_others(no_gpu: None) -> None:
    from oca import remotes

    lab = Remote("url", model="m", url="http://lab", context=32768)
    s = Settings(default="spark", remotes={"ucloud": UCLOUD, "spark": SPARK, "lab": lab})
    config.model_path(s.resolved_model).parent.mkdir(parents=True)
    config.model_path(s.resolved_model).touch()
    running = {"lab"}.__contains__
    offered = {c.label: c.description for c in remotes.choices(s, "spark", running)}
    assert list(offered) == ["Restart spark", "Use lab", "Use local", "Use ucloud"]
    assert offered["Use lab"] == (
        "running: m on http://lab, 32k context; sees only the latest 32k tokens"
    )
    assert offered["Use ucloud"].startswith("starts deepseek-v4.1-flash on UCloud")
    assert offered["Use ucloud"].endswith("a 2x B200 job, up to 16 GPU-hours")
    again = remotes.choices(s, "spark", running, failed={"spark", "lab"})
    assert [c.label for c in again] == ["Use local", "Use ucloud", "Try spark again"]
    lab_down = [c.label for c in remotes.choices(s, "lab", lambda n: False)]
    assert lab_down == [
        "Use local",
        "Use ucloud",
        "Use spark",
        "Try lab again",
    ]  # ties: config order


def test_oca_asks_what_to_do_when_its_server_is_down(
    monkeypatch: pytest.MonkeyPatch, no_gpu: None
) -> None:
    import typer

    from oca import cli, remotes, router

    s = Settings(default="spark", remotes={"spark": SPARK})
    s.save()
    config.model_path(s.resolved_model).parent.mkdir(parents=True)
    config.model_path(s.resolved_model).touch()
    gone = Down("its server stopped: idle for 60 min")
    monkeypatch.setattr(install, "is_installed", lambda: True)
    monkeypatch.setattr(router, "ensure", lambda settings: None)
    monkeypatch.setattr(remotes, "check", lambda settings, name: (_ for _ in ()).throw(gone))
    started: list[str] = []

    def start(settings: Settings, name: str, *args: Any) -> None:
        started.append(name)
        raise OcaError("cannot reach spark over SSH")

    monkeypatch.setattr(remotes, "start", start)
    monkeypatch.setattr(cli, "_local", lambda settings, yes: started.append("local"))
    asked: list[str] = []

    def prompt(question: str, **kwargs: Any) -> str:
        asked.append(question)
        return next(picks)

    picks = iter(["9", "1", "1"])  # not an option; Restart spark; it fails, so: Use local
    monkeypatch.setattr(typer, "prompt", prompt)
    assert cli._ready(s) == "local" and started == ["spark", "local"]
    assert asked[0] == "Choose (1/2/3)" and len(asked) == 3
    picks = iter(["3"])  # Cancel
    with pytest.raises(typer.Exit):
        cli._ready(s)
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: True)
    assert cli._ready(s, pick=False) == "spark"  # `oca web`: its sessions ask
    with pytest.raises(OcaError, match="cannot reach"):
        cli._ready(s, yes=True)


def test_oca_id_opens_that_model_and_ls_lists_every_pair(
    monkeypatch: pytest.MonkeyPatch, no_gpu: None
) -> None:
    from typer.testing import CliRunner

    from oca import cli, remotes

    monkeypatch.setattr(remotes, "memory_gb", lambda gpu: 64.0)  # flash-next does not fit
    Settings(remotes={"spark": SPARK, "ucloud": UCLOUD}).save()
    opened: list[Any] = []

    def ready(settings: Settings, pair: remotes.Pair | None, yes: bool) -> str:
        opened.append(pair and pair.id)
        return pair.server if pair else "local"

    monkeypatch.setattr(cli, "_ready", ready)
    monkeypatch.setattr(cli.agent, "launch", opened.append)
    here = [{"id": "ses_1abcdefghijk", "title": "new"}, {"id": "ses_2zz", "title": "old"}]
    monkeypatch.setattr(cli.agent, "sessions", lambda folder: here)
    run = CliRunner().invoke
    assert run(cli.app, ["deep", "-c"]).exit_code == 0  # part of an id
    assert run(cli.app, ["-c", "qwen3.6@spark"]).exit_code == 0  # options first
    assert run(cli.app, ["-r", "ses_1"]).exit_code == 0  # the start of a session's id
    assert run(cli.app, ["-r"], input="2\n").exit_code == 0  # alone: a menu, as in Claude Code
    assert run(cli.app, ["-r", "-y"], input="1\n").exit_code == 0
    assert opened == [
        *("deepseek@ucloud", ["--continue", "--model", "oca/ucloud"]),
        *("qwen3.6@spark", ["--continue", "--model", "oca/spark"]),
        *(None, ["--session", "ses_1abcdefghijk"]),
        *(None, ["--session", "ses_2zz"]),
        *(None, ["--session", "ses_1abcdefghijk"]),
    ]
    several = run(cli.app, ["spark"]).exception  # three models fit there; no terminal to ask
    assert "could be qwen3.6@spark, qwen3.8-27b@spark, flash-next@spark" in str(several)
    assert "nothing is called 'sprak'" in str(run(cli.app, ["sprak"]).exception)
    assert "oca <id>" in run(cli.app, ["-h"]).output
    assert "Usage: oca server add" in run(cli.app, ["server", "add", "-h"]).output
    assert "Usage: oca ucloud" in run(cli.app, ["ucloud", "-h"]).output  # a command, not a pair
    assert run(cli.app, ["-V"]).output.startswith("oca ")
    now = {
        "local": ("stopped", None, {"qwen3.8-27b"}),
        "spark": ("ready", "qwen3.6-35b-a3b", None),  # an older oca: downloads unknown
        "ucloud": ("down", None, {"deepseek-v4.1-flash", "qwen3.8-27b-fp8"}),
    }
    monkeypatch.setattr(cli, "_state", lambda settings, name: now[name])
    listed = run(cli.app, ["ls"]).output
    assert listed.startswith("local: this machine (cpu)\n * qwen3.6@local ")  # the default
    assert re.search(r"qwen3\.6@local +qwen3\.6-35b-a3b +262k  not downloaded\n", listed)
    assert re.search(r"\n   qwen3\.8-27b@local +qwen3\.8-27b +65k  downloaded\n", listed)
    assert "\nspark: spark over SSH, for all its users\n" in listed
    assert re.search(r"\n   qwen3\.6@spark +qwen3\.6-35b-a3b +262k  ready\n", listed)
    assert re.search(r"\n   flash-next@spark +qwen3\.8-flash-next +262k\n", listed)  # unknown
    assert re.search(r"\n   deepseek@ucloud +deepseek-v4\.1-flash, 2 B200s +262k  down\n", listed)
    assert re.search(r"\n   qwen3\.8-27b@ucloud .* 262k  downloaded\n", listed)
    assert re.search(r"\n   flash-next@ucloud .* 262k  not downloaded\n", listed)
    assert "flash-next@local" not in listed  # 64 GB: it does not fit
    assert run(cli.app, ["config", "default", "deep"]).exit_code == 0
    assert Settings.load().default == "deepseek@ucloud"
    assert "\ndefault = 'deepseek@ucloud'\n" in run(cli.app, ["config"]).output


def test_a_members_ucloud_server_is_listed_and_opening_it_joins_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oca import cli, cluster, remotes

    s = Settings(remotes={"ucloud": replace(UCLOUD, model="qwen3.8-flash-next")})
    monkeypatch.setattr(cluster, "read", lambda kind: None)  # nothing started or joined here
    monkeypatch.setattr(cluster, "downloaded", lambda r: {"qwen3.8-flash-next"})
    monkeypatch.setattr(cluster, "listed", lambda r: {"qwen3.8-flash-next": "bob@ws1"})
    on_drive = {"qwen3.8-flash-next"}
    assert cli._state(s, "ucloud") == ("started by bob@ws1", "qwen3.8-flash-next", on_drive)
    job = cluster.Job("serve", "42", "qwen3.8-flash-next", "t")
    monkeypatch.setattr(cluster, "join", lambda r, m: (job, "bob@ws1"))
    monkeypatch.setattr(cluster, "check", lambda r, j: None if j is job else "?")
    assert remotes.check(s, "ucloud") is None  # opening it joins bob's server
    monkeypatch.setattr(cluster, "join", lambda r, m: None)
    with pytest.raises(Down, match="no server started"):
        remotes.check(s, "ucloud")


def test_serve_says_which_models_are_downloaded() -> None:
    from oca import serve

    path = config.model_path(config.MODELS["qwen3.8-27b"])
    path.parent.mkdir(parents=True)
    path.touch()
    assert serve.status()["downloaded"] == ["qwen3.8-27b"]  # for `oca ls` on other machines


def test_tab_completes_commands_and_ids(monkeypatch: pytest.MonkeyPatch, no_gpu: None) -> None:
    import typer

    from oca import cli, remotes

    monkeypatch.setattr(remotes, "memory_gb", lambda gpu: 64.0)
    Settings(remotes={"spark": SPARK}).save()
    assert cli._complete_ids("qwen3.6") == ["qwen3.6@local", "qwen3.6@spark"]
    assert cli._complete_servers("sp") == ["spark"]
    group = typer.main.get_command(cli.app)
    ctx = group.make_context("oca", [], resilient_parsing=True)
    first = [c.value for c in group.shell_complete(ctx, "s")]
    assert {"server", "sessions", "stop", "status"} <= set(first)
    assert [c.value for c in group.shell_complete(ctx, "qwen3.8")] == [
        "qwen3.8-27b@local",
        "qwen3.8-27b@spark",
    ]


def test_tab_completion_installs_itself(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from oca import cli

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(cli.sys, "stdin", Terminal())
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("SHELL", "/bin/bash")
    cli._completion()
    path = tmp_path / "share" / "bash-completion" / "completions" / "oca"
    assert "_OCA_COMPLETE=complete_bash" in path.read_text()  # bash-completion loads it
    path.write_text("old")
    cli._completion()  # refreshed when oca's changes
    assert "complete_bash" in path.read_text() and path in install.owned_paths()
    monkeypatch.setenv("OCA_SHARED", "1")  # an install for all users is not the user's shell
    path.unlink()
    cli._completion()
    assert not path.exists()
    assert "completion" not in CliRunner().invoke(cli.app, ["-h"]).output


def test_sessions_and_server_update(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from oca import cli, remotes

    Settings(remotes={"spark": SPARK}).save()
    ran: list[list[str]] = []
    monkeypatch.setattr(cli.agent, "launch", ran.append)
    title = "Fix the router's retry when a server stops, and say why it stopped in the menu"
    here = [
        {"id": "ses_e6862e455ffeAAA", "title": title, "updated": 0, "directory": str(Path.cwd())},
        {"id": "ses_e6862e455ffeBBB", "title": "b", "updated": 0, "directory": "/x"},
    ]
    monkeypatch.setattr(cli.agent, "sessions", lambda folder: here)
    run = CliRunner().invoke
    listed = run(cli.app, ["sessions"]).output  # printed, with no pager to quit
    assert "ses_e6862e455ffeA " in listed and "ses_e6862e455ffeB " in listed  # told apart
    assert "say why it stopped" in listed.replace("\n", " ").replace("  ", " ")  # whole titles
    run(cli.app, ["sessions", "rm", "ses_e6862e455ffeB"])
    assert ran == [["session", "delete", "ses_e6862e455ffeBBB"]]
    assert "type more" in str(run(cli.app, ["sessions", "rm", "ses_e68"]).exception)
    deleted: list[str] = []
    monkeypatch.setattr(cli.agent, "opencode", lambda args: deleted.append(args[-1]) or "")
    assert "Deleted 2 sessions." in run(cli.app, ["sessions", "clear", "-y"]).output
    assert deleted == ["ses_e6862e455ffeAAA", "ses_e6862e455ffeBBB"]
    installs: list[tuple[str, str]] = []
    monkeypatch.setattr(remotes, "install", lambda host, shared: installs.append((host, shared)))
    assert run(cli.app, ["server", "update", "spark"]).exit_code == 0
    assert installs == [("spark", "/opt/oca")]  # its install for all users, again


def test_pairs_name_each_model_on_each_server(
    monkeypatch: pytest.MonkeyPatch, no_gpu: None
) -> None:
    from oca import remotes

    monkeypatch.setattr(remotes, "memory_gb", lambda gpu: 130.0)
    lab = Remote("url", model="big/model", url="http://lab:8000", context=32768)
    s = Settings(remotes={"spark": replace(SPARK, memory=34.2), "ucloud": UCLOUD, "lab": lab})
    ids = [p.id for p in remotes.pairs(s)]
    assert ids == [
        *("qwen3.6@local", "qwen3.8-27b@local", "flash-next@local"),  # 130 GB: all fit
        *("qwen3.6@spark", "qwen3.8-27b@spark"),  # 34 GB: no flash-next
        *("deepseek@ucloud", "qwen3.8-27b@ucloud", "flash-next@ucloud"),  # what it serves first
        "big/model@lab",
    ]
    find = lambda word: [p.id for p in remotes.find(s, word)]  # noqa: E731
    assert find("flash") == ["flash-next@local", "flash-next@ucloud"]  # ids before model names
    assert find("v4.1") == ["deepseek@ucloud"] and find("qwen3.6@spark") == ["qwen3.6@spark"]
    assert find("qwen3.8-27b-fp8@ucloud") == ["qwen3.8-27b@ucloud"] and find("nope") == []
    assert remotes.default_pair(s).id == "qwen3.6@local"  # a bare server: the model it serves
    s.default = "qwen3.8-27b@spark"
    s.validate()
    assert remotes.default_pair(s).server == "spark" and s.default_server == "spark"
    for wrong in ("deepseek@spark", "qwen3.6@nowhere"):
        with pytest.raises(OcaError, match="default"):
            replace(s, default=wrong).validate()


def test_opening_another_model_on_an_ssh_machine_asks_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typer

    from oca import cli, remotes

    s = Settings(remotes={"spark": SPARK})
    s.save()
    told: dict[str, Any] = {"state": "ready", "model": "qwen3.6-35b-a3b", "mine": True}
    monkeypatch.setattr(remotes, "serve_call", lambda r, args: told)
    asked: list[str] = []
    monkeypatch.setattr(typer, "confirm", lambda q, **kw: asked.append(q) or True)
    flash = remotes.Pair("flash-next@spark", "spark", "qwen3.8-flash-next", 262144)
    assert cli._use(s, flash, yes=False) == flash
    assert asked == ["spark serves qwen3.6-35b-a3b. Restart it with qwen3.8-flash-next?"]
    assert Settings.load().remotes["spark"].model == "qwen3.8-flash-next"  # what it starts
    told["mine"] = False  # someone else's: only they can switch it
    assert cli._use(s, flash, yes=False).id == "qwen3.6@spark"
    assert (
        asked[-1] == "Open qwen3.6@spark instead?" and s.remotes["spark"].model == "qwen3.6-35b-a3b"
    )


def test_stop_and_pull_act_on_the_server_named(
    monkeypatch: pytest.MonkeyPatch, no_gpu: None
) -> None:
    from typer.testing import CliRunner

    from oca import cli, remotes

    Settings(default="qwen3.6@spark", remotes={"spark": SPARK}).save()
    downs: list[str] = []
    monkeypatch.setattr(remotes, "down", lambda settings, name: downs.append(name) or True)
    monkeypatch.setattr(cli.server, "stop", lambda: downs.append("local") or False)
    monkeypatch.setattr(cli.router, "stop", lambda: downs.append("endpoint") or True)
    run = CliRunner().invoke
    assert "spark: stopped." in run(cli.app, ["stop"]).output  # the default's server
    assert "local: was not running." in run(cli.app, ["stop", "local"]).output
    assert run(cli.app, ["stop", "--all"]).exit_code == 0
    assert downs == ["spark", "local", "local", "spark", "endpoint"]
    ran: list[list[str]] = []
    monkeypatch.setattr(cli.os, "execvp", lambda exe, args: ran.append(args))
    monkeypatch.setattr(cli.install, "pull", lambda m: Path(m.id))
    assert "Saved qwen3.8-27b" in run(cli.app, ["pull", "qwen3.8-27b"]).output  # a catalog id
    run(cli.app, ["pull", "qwen3.6@spark"])
    assert ran[-1][-1].endswith("/opt/oca/bin/oca pull qwen3.6-35b-a3b")  # there, by catalog id


def test_ssh_remote_offers_the_biggest_model_that_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    from oca import cli

    assert cli._ssh_model("spark", 128.5, yes=True) == "qwen3.8-flash-next"
    assert cli._ssh_model("box", 34.2, yes=True) == "qwen3.6-35b-a3b"
    assert cli._ssh_model("box", 0, yes=True) == config.DEFAULT_MODEL["cpu"]
    asked: list[str] = []
    monkeypatch.setattr(typer, "prompt", lambda q, **kw: asked.append(q) or kw["default"])
    assert cli._ssh_model("spark", 128.5, yes=False) == "qwen3.8-flash-next"
    assert asked == ["Choose (1/2/3)"]  # numbered, the biggest first


def test_a_split_model_downloads_its_first_part_last() -> None:
    m = config.MODELS["qwen3.8-flash-next"]
    stem = "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS"
    assert install.files(m) == [f"{stem}-0000{i}-of-00003.gguf" for i in (2, 3, 1)]
    assert install.files(config.MODELS["qwen3.6-35b-a3b"]) == ["Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"]
    assert config.model_path(m).name.endswith("-00001-of-00003.gguf")


def test_a_smaller_server_sees_the_latest_turns_that_fit() -> None:
    from oca import router

    turns = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old " + "x" * 3000},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "y" * 600},
        {"role": "user", "content": [{"type": "text", "text": "now"}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "go on"},
    ]
    assert router.fit(turns, 10**6) == (turns, 0)
    kept, dropped = router.fit(turns, 200)
    assert dropped == 3 and kept[0] == turns[0] and kept[2:] == turns[5:]
    assert kept[1]["content"][0]["text"].startswith("[oca: the 3 earlier messages")
    assert kept[1]["content"][1:] == turns[4]["content"]  # starts at a user turn, untouched
    kept, dropped = router.fit(turns, 1)  # nothing fits: still the last user turn
    assert dropped == 5 and kept[-1]["content"].endswith("go on")
    small = Backend("spark", "u", "m", "", 1000)
    long_ask = {"model": "ucloud", "messages": turns, "max_tokens": 100}
    assert router._retarget(long_ask, small, room=262144)[1] == 3  # stands in for ucloud
    assert router._retarget(long_ask, small)[1] == 0  # OpenCode sized it for this server
    # OpenCode lets the requested server reply with 32k tokens; the stand-in gets a quarter.
    lab = Backend("lab", "u", "m", "", 32768)
    body, dropped = router._retarget({**long_ask, "max_tokens": 32768}, lab, room=131072)
    assert json.loads(body)["max_tokens"] == 8192 and dropped == 0


def test_tunnel_binds_localhost_and_pins_host_keys_per_job(tmp_path: Path) -> None:
    from oca import cluster

    cmd = cluster.tunnel_command("ucloud", "ssh.cloud.sdu.dk", 3421, 8001, tmp_path / "kh")
    assert "127.0.0.1:8001:127.0.0.1:8000" in cmd
    assert f"UserKnownHostsFile={tmp_path / 'kh'}" in cmd
    assert "BatchMode=yes" in cmd  # never hangs on a password prompt


@pytest.fixture
def ucloud(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake UCloud: drive files, the job state, and the jobs terminated."""
    import ucloud_api
    from ucloud_api.exceptions import APIError

    from oca import cluster

    world: dict[str, Any] = {"files": {}, "state": "RUNNING", "terminated": []}

    class Transfer:
        def __init__(self, client: object) -> None: ...

        def read_bytes(self, remote: str) -> bytes:
            if remote not in world["files"]:
                raise APIError("not found", status_code=404)
            return bytes(world["files"][remote])

        def upload(self, local: Path, remote: str) -> None:
            world["files"][remote] = local.read_bytes()

    class Files:
        def __init__(self, client: object) -> None: ...

        def mkdir(self, path: str, conflict_policy: str) -> None: ...

        def trash(self, path: str) -> None:
            del world["files"][path]

    class Jobs:
        def __init__(self, client: object) -> None: ...

        def state(self, job_id: str) -> ucloud_api.JobState:
            return ucloud_api.JobState(world["state"])

        def terminate(self, job_id: str) -> None:
            world["terminated"].append(job_id)

    monkeypatch.setattr(cluster, "_client", lambda: nullcontext(None))
    for fake in (Transfer, Files, Jobs):
        monkeypatch.setattr(ucloud_api, fake.__name__, fake)
    return world


def test_members_join_a_listed_server_and_leaving_does_not_stop_it(
    ucloud: dict[str, Any],
) -> None:
    from oca import cluster

    s, model = UCLOUD, config.CLUSTER_MODELS["deepseek-v4.1-flash"]
    started = cluster.Job("serve", "42", model.id, "serve-x", endpoint="ucloud@h:2211", key="k")
    cluster.publish(s, started)  # the starter's machine lists it
    joined = cluster.join(s, model)  # another member's machine
    assert joined is not None
    job, starter = joined
    assert (job.job_id, job.owner, job.endpoint, job.key) == ("42", False, "ucloud@h:2211", "k")
    assert "@" in starter
    assert cluster.terminate(s, "serve") and ucloud["terminated"] == []
    ucloud["state"] = "SUCCESS"
    assert cluster.join(s, model) is None  # an ended server is not joined


def test_starter_stops_and_unlists_its_server(ucloud: dict[str, Any]) -> None:
    from oca import cluster

    job = cluster.Job("serve", "42", "deepseek-v4.1-flash", "serve-x", endpoint="u@h:1")
    cluster.publish(UCLOUD, job)
    cluster._write(job, "serve")
    assert cluster.terminate(UCLOUD, "serve")
    assert ucloud["terminated"] == ["42"] and ucloud["files"] == {}


def test_web_password_is_made_once_and_private() -> None:
    first = agent.web_password()
    assert len(first) >= 20 and agent.web_password() == first
    assert (config.config_dir() / "web-password").stat().st_mode & 0o077 == 0


def test_uninstall_only_deletes_what_oca_created(tmp_path: Path) -> None:
    data = Path(os.environ["OCA_DATA"])
    for part in ("models/x.gguf", "bin/llama-server", "unrelated/keep.txt"):
        (data / part).parent.mkdir(parents=True, exist_ok=True)
        (data / part).write_text("x")
    Settings().save()
    install.remove(install.owned_paths(keep_models=True))
    assert (data / "models" / "x.gguf").exists() and (data / "unrelated" / "keep.txt").exists()
    assert not (data / "bin").exists() and not Settings.path().exists()
    install.remove(install.owned_paths())
    assert not (data / "models").exists() and (data / "unrelated" / "keep.txt").exists()


def test_ssh_remote_probe_and_serve_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    from oca import remotes

    sent: list[str] = []
    out = {"text": "aarch64\n@shared\nNVIDIA GB10, [N/A]\n@mem 125513000\n", "code": 0}

    def ssh(host: str, command: str, timeout: float = 0) -> subprocess.CompletedProcess[str]:
        sent.append(command)
        return subprocess.CompletedProcess([], out["code"], stdout=out["text"], stderr="")

    monkeypatch.setattr(remotes, "ssh", ssh)
    found = remotes.probe("spark")
    assert (found.arch, found.gpu, found.shared, found.personal) == (
        "aarch64",
        "NVIDIA GB10",
        True,
        False,
    )
    assert round(found.memory) == 129  # its GPU shares the RAM
    out["text"] = "x86_64\n@personal\nNVIDIA GeForce RTX 5090, 32607 MiB\n@mem 131000000\n"
    assert round(remotes.probe("box").memory) == 34
    out["text"] = "x86_64\n@mem 131000000\n"
    assert remotes.probe("box").memory == 0  # no GPU
    out["text"] = 'a warning\n{"state": "ready", "port": 8181}\n'
    assert remotes.serve_call(SPARK, "--status")["port"] == 8181
    assert "OCA_SHARED=1" in sent[-1] and "/opt/oca/bin/oca serve --status" in sent[-1]
    personal = Remote("ssh", model="qwen3.6-35b-a3b", host="spark")
    remotes.serve_call(personal, "--status")
    assert sent[-1].startswith('"$HOME/.local/bin/oca" serve') and "OCA_SHARED" not in sent[-1]
    out["text"] = '{"error": "this machine serves qwen3.8-27b for bob"}'
    with pytest.raises(OcaError, match=r"serves qwen3\.8-27b for bob"):
        remotes.serve_call(SPARK, "")
    out.update(text="", code=127)
    with pytest.raises(OcaError, match="failed"):
        remotes.serve_call(SPARK, "")


def test_ssh_remote_up_tunnels_to_its_server_and_down_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oca import cli, cluster, remotes

    calls: list[str] = []
    tunnels: list[list[str]] = []
    answer = {"state": "loading", "model": "qwen3.6-35b-a3b", "port": 8181, "key": "k"}
    answer |= {"context": 262144, "mine": True}
    monkeypatch.setattr(remotes, "serve_call", lambda r, args: calls.append(args) or answer)
    monkeypatch.setattr(cluster, "open_tunnel", lambda cmd, port, log: tunnels.append(cmd) or 42)
    s = Settings(remotes={"spark": SPARK})
    assert remotes.up(s, "spark") == "loading"
    assert "127.0.0.1:8002:127.0.0.1:8181" in tunnels[-1] and tunnels[-1][-1] == "spark"
    assert remotes.backend(s, "spark") == Backend(
        "spark", "http://127.0.0.1:8002", "qwen3.6-35b-a3b", "k", 262144
    )
    assert remotes.context(s, "spark") == 262144
    assert remotes.down(s, "spark") is True and calls[-1] == "--stop"
    assert remotes.read_conn("spark") is None and remotes.down(s, "spark") is None
    answer["state"] = "ready"
    monkeypatch.setattr(remotes, "healthy", lambda b: True)
    assert remotes.check(s, "spark") is None  # joins a server someone else started
    assert remotes.read_conn("spark") is not None
    answer.clear()
    answer.update(state="stopped", why="idle for 60 min, at 14:02")
    with pytest.raises(Down, match="its server stopped: idle for 60 min, at 14:02"):
        remotes.check(s, "spark")
    assert remotes.read_conn("spark") is None
    assert cli._state(s, "spark") == ("stopped (idle for 60 min, at 14:02)", None, None)
    monkeypatch.setattr(remotes, "serve_call", lambda r, args: (_ for _ in ()).throw(OcaError()))
    assert cli._state(s, "spark") == ("not answering", None, None)


def test_a_stopped_server_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    from oca import serve

    monkeypatch.setattr(serve.install, "is_installed", lambda: True)
    monkeypatch.setattr(serve, "model_path", lambda model: Path("/"))
    monkeypatch.setattr(serve.server, "start", lambda settings: None)
    monkeypatch.setattr(serve.server, "stop", lambda: True)
    monkeypatch.setattr(serve, "_idle", lambda settings, minutes: f"idle for {minutes} min")
    monkeypatch.setattr(serve.signal, "signal", lambda *args: None)
    serve.supervise(60)
    assert serve.status()["state"] == "stopped"
    assert serve.status()["why"].startswith("idle for 60 min, at ")


def test_serve_lists_one_server_that_others_join_but_cannot_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oca import serve

    spawned: list[list[str]] = []

    class Supervisor:
        pid = 777

        def __init__(self, cmd: list[str], **kwargs: Any) -> None:
            spawned.append(cmd)
            serve._set(pid=self.pid, state="starting")  # what the real one does first

        def poll(self) -> None:
            return None

    monkeypatch.setattr(serve.subprocess, "Popen", Supervisor)
    monkeypatch.setattr(serve, "process_alive", lambda pid, name: pid == 777)
    monkeypatch.setattr(serve.getpass, "getuser", lambda: "alice")
    first = serve.start("qwen3.6-35b-a3b", 30)
    assert first["state"] == "starting" and first["mine"] and first["port"] == 8081
    assert first["key"] == agent.api_key() and spawned[-1][-1] == "30"
    assert Settings.load().model == "qwen3.6-35b-a3b"
    assert serve.start(None, 30) == first and len(spawned) == 1  # joined, not restarted
    monkeypatch.setattr(serve.getpass, "getuser", lambda: "bob")
    assert serve.status()["mine"] is False
    with pytest.raises(OcaError, match=r"serves qwen3\.6-35b-a3b for alice"):
        serve.start("qwen3.8-27b", 30)
    with pytest.raises(OcaError, match="only alice"):
        serve.stop()
