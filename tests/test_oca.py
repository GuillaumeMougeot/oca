"""Offline tests: no network, no GPU, no real server."""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import tarfile
import threading
import zipfile
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from oca import agent, config, install, server
from oca.config import OcaError, Settings


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


def test_opencode_sees_one_model_served_by_the_router(no_gpu: None) -> None:
    s = Settings(port=8123)
    remote = config.CLUSTER_MODELS["deepseek-v4.1-flash"]
    cfg = agent.opencode_config(s, remote)
    provider = cfg["provider"][agent.PROVIDER]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8123/v1"
    assert cfg["model"] == cfg["small_model"] == f"{agent.PROVIDER}/{agent.MODEL}"
    assert cfg["enabled_providers"] == [agent.PROVIDER]
    model = provider["models"][agent.MODEL]
    assert (
        remote.id in model["name"]
        and s.resolved_model.id
        in agent.opencode_config(s)["provider"][agent.PROVIDER]["models"][agent.MODEL]["name"]
    )
    # Sized for the local model, so a session can always fall back to it.
    assert model["limit"]["context"] == s.resolved_context > model["limit"]["output"]
    assert cfg["permission"]["webfetch"] == "ask"


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


def test_stale_state_is_ignored() -> None:
    server.state_path().parent.mkdir(parents=True)
    server.state_path().write_text('{"pid": 999999999, "port": 8080, "cmd": []}')
    assert server.read_state() is None
    assert server.stop() is False


def test_permissions_ask_by_default_and_deny_system_installs() -> None:
    bash = agent.opencode_config(Settings())["permission"]["bash"]
    assert next(iter(bash.items())) == ("*", "ask")  # first, so later rules override it
    assert bash["*--break-system-packages*"] == "deny"
    assert list(bash)[-1] in {k for k, v in bash.items() if v == "deny"}  # denies win


def test_serve_script_is_valid_bash_and_stops_when_idle(tmp_path: Path) -> None:
    from oca import cluster

    s = Settings(cluster_idle=45)
    for model in config.CLUSTER_MODELS.values():
        script = cluster.serve_script(model, s, "serve-x")
        path = tmp_path / "serve.sh"
        path.write_text(script)
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
        assert "-ge 45 ]" in script and "/work/oca/logs/serve-x.log" in script
        assert f"--served-model-name {model.id}" in script
        assert "--host 127.0.0.1" in script  # reachable only through the SSH tunnel
        assert f"vllm=={config.VLLM_VERSION}" in script
        assert f"rm -f /work/oca/servers/{model.id}.json" in script  # unlisted on exit
        assert "trap unlist EXIT" in script and "export CUDA_HOME=" in script


def _fake_server(seen: list[str]) -> ThreadingHTTPServer:
    """An OpenAI-like server: /health answers, a completion streams three events."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: {i}\n\n".encode())
                self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_router_uses_the_cluster_while_it_answers_then_falls_back_mid_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from oca import cluster, router

    remote_seen: list[str] = []
    local_seen: list[str] = []
    remote, local = _fake_server(remote_seen), _fake_server(local_seen)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    s = Settings(port=port, cluster_port=remote.server_address[1])
    job = cluster.Job("serve", "1", "deepseek-v4.1-flash", "t")
    monkeypatch.setattr(cluster, "read", lambda kind: job)
    monkeypatch.setattr(server, "start", lambda settings: None)
    monkeypatch.setattr(server, "port", lambda settings: local.server_address[1])
    endpoint = ThreadingHTTPServer(("127.0.0.1", port), router.handler(router.Router(s)))
    threading.Thread(target=endpoint.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    ask = {"model": "auto", "stream": True, "messages": []}
    assert httpx.post(url, json=ask).text == "data: 0\n\ndata: 1\n\ndata: 2\n\n"
    remote.shutdown()
    remote.server_close()  # the cluster stops; the next request must not fail
    assert httpx.post(url, json=ask).text.startswith("data: 0")
    assert remote_seen == ["deepseek-v4.1-flash"] and local_seen == [s.resolved_model.id]
    endpoint.shutdown()
    local.shutdown()


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

    s, model = Settings(cluster_drive="/9"), config.CLUSTER_MODELS["deepseek-v4.1-flash"]
    started = cluster.Job("serve", "42", model.id, "serve-x", endpoint="ucloud@h:2211")
    cluster.publish(s, started)  # the starter's machine lists it
    joined = cluster.join(s, model)  # another member's machine
    assert joined is not None
    job, starter = joined
    assert (job.job_id, job.owner, job.endpoint) == ("42", False, "ucloud@h:2211")
    assert "@" in starter
    assert cluster.terminate(s, "serve") and ucloud["terminated"] == []
    ucloud["state"] = "SUCCESS"
    assert cluster.join(s, model) is None  # an ended server is not joined


def test_starter_stops_and_unlists_its_server(ucloud: dict[str, Any]) -> None:
    from oca import cluster

    s = Settings(cluster_drive="/9")
    job = cluster.Job("serve", "42", "deepseek-v4.1-flash", "serve-x", endpoint="u@h:1")
    cluster.publish(s, job)
    cluster._write(job, "serve")
    assert cluster.terminate(s, "serve")
    assert ucloud["terminated"] == ["42"] and ucloud["files"] == {}


def test_waiting_for_the_server_does_not_count_queue_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oca import cluster

    job = cluster.Job("serve", "1", "m", "t")
    states = iter(["in queue"] * 3 + [None])
    monkeypatch.setattr(cluster, "check", lambda s, j: next(states))
    monkeypatch.setattr(cluster.time, "sleep", lambda s: None)
    cluster.wait_ready(Settings(), job, lambda doing: None, timeout=0)  # queued: no timeout
    loading = iter(["loading"] * 3)
    monkeypatch.setattr(cluster, "check", lambda s, j: next(loading))
    with pytest.raises(OcaError, match="not ready"):
        cluster.wait_ready(Settings(), job, lambda doing: None, timeout=0)


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
