"""Offline tests: no network, no GPU, no real server."""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

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


def test_opencode_config_points_at_local_server(no_gpu: None) -> None:
    s = Settings(port=8123)
    cfg = agent.opencode_config(s)
    provider = cfg["provider"][agent.PROVIDER]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8123/v1"
    assert cfg["model"] == f"{agent.PROVIDER}/{s.resolved_model.id}"
    assert cfg["enabled_providers"] == [agent.PROVIDER]
    limit = provider["models"][s.resolved_model.id]["limit"]
    assert limit["context"] == s.resolved_context > limit["output"]


def test_llama_assets() -> None:
    tag = config.LLAMA_CPP_TAG
    assert install.llama_assets("x64", False) == [f"llama-{tag}-bin-ubuntu-x64.tar.gz"]
    cuda = install.llama_assets("x64", True)
    assert len(cuda) == 2 and all("cuda-12.8-x64" in a for a in cuda)


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
