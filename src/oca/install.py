"""Download pinned llama.cpp and OpenCode release binaries and model weights; remove them."""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import httpx
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

from .config import (
    LLAMA_CPP_TAG,
    OPENCODE_VERSION,
    Model,
    OcaError,
    config_dir,
    data_dir,
    has_nvidia_gpu,
    model_path,
)

GITHUB = "https://github.com"


def platform_id() -> tuple[str, str]:
    """This machine as release assets name it: ("linux" or "macos", "x64" or "arm64")."""
    system, machine = platform.system(), platform.machine().lower()
    os_name = {"Linux": "linux", "Darwin": "macos"}.get(system)
    cpu = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if os_name is None or cpu is None:
        raise OcaError(f"unsupported platform {system}/{machine}: Linux or macOS, x64 or arm64")
    return os_name, cpu


def llama_assets(os_name: str, cpu_arch: str, cuda: bool) -> list[str]:
    """Release tarballs for this machine. The CUDA build also runs CPU-only; macOS has Metal."""
    tag = LLAMA_CPP_TAG
    if os_name == "macos":
        return [f"llama-{tag}-bin-macos-{cpu_arch}.tar.gz"]
    if not cuda:
        return [f"llama-{tag}-bin-ubuntu-{cpu_arch}.tar.gz"]
    cu = "12.8" if cpu_arch == "x64" else "13.4"
    return [
        f"llama-{tag}-bin-ubuntu-cuda-{cu}-{cpu_arch}.tar.gz",
        f"cudart-llama-{tag}-bin-ubuntu-cuda-{cu}-{cpu_arch}.tar.gz",
    ]


def opencode_asset(os_name: str, cpu_arch: str) -> str:
    if os_name == "macos":
        return f"opencode-darwin-{cpu_arch}.zip"
    return f"opencode-linux-{cpu_arch}.tar.gz"


def llama_dir() -> Path:
    variant = "cuda" if has_nvidia_gpu() else "macos" if platform.system() == "Darwin" else "cpu"
    return data_dir() / "bin" / f"llama.cpp-{LLAMA_CPP_TAG}-{variant}"


def opencode_dir() -> Path:
    return data_dir() / "bin" / f"opencode-{OPENCODE_VERSION}"


def _find(root: Path, name: str) -> Path | None:
    return (
        next((p for p in sorted(root.rglob(name)) if p.is_file()), None) if root.is_dir() else None
    )


def llama_server() -> Path | None:
    return _find(llama_dir(), "llama-server")


def opencode() -> Path | None:
    return _find(opencode_dir(), "opencode")


def is_installed() -> bool:
    return llama_server() is not None and opencode() is not None


def _download(url: str, dest: Path) -> None:
    with (
        httpx.stream("GET", url, follow_redirects=True, timeout=60) as r,
        Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as bar,
    ):
        if r.status_code != 200:
            raise OcaError(f"download failed ({r.status_code}): {url}")
        task = bar.add_task(url.rsplit("/", 1)[-1], total=int(r.headers.get("content-length", 0)))
        with dest.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                bar.advance(task, len(chunk))


def install_release(urls: list[str], dest: Path) -> None:
    """Download and unpack archives into ``dest``, atomically (no half-installed state)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staging = Path(tmp) / "out"
        for url in urls:
            archive = Path(tmp) / "archive"
            _download(url, archive)
            if url.endswith(".zip"):
                _unzip(archive, staging)
            else:
                with tarfile.open(archive) as tf:
                    tf.extractall(staging, filter="data")
        shutil.rmtree(dest, ignore_errors=True)
        staging.rename(dest)


def _unzip(archive: Path, dest: Path) -> None:
    """Extract inside ``dest`` (zipfile drops ".." and "/"), keeping executable bits."""
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            path = Path(zf.extract(info, dest))
            if mode := (info.external_attr >> 16) & 0o777:
                path.chmod(mode)


def setup(force: bool = False) -> None:
    os_name, cpu_arch = platform_id()
    if force or llama_server() is None:
        base = f"{GITHUB}/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}"
        assets = llama_assets(os_name, cpu_arch, has_nvidia_gpu())
        install_release([f"{base}/{a}" for a in assets], llama_dir())
    if force or opencode() is None:
        base = f"{GITHUB}/anomalyco/opencode/releases/download/v{OPENCODE_VERSION}"
        install_release([f"{base}/{opencode_asset(os_name, cpu_arch)}"], opencode_dir())


def pull(model: Model) -> Path:
    from huggingface_hub import hf_hub_download  # slow import, only needed here

    dest = model_path(model)
    hf_hub_download(model.repo, model.file, local_dir=dest.parent)
    return dest


# Only what oca creates is deleted, so an OCA_DATA pointing at a shared disk is never wiped.
DATA_PARTS = ("bin", "models", "logs", "run")
CONFIG_PARTS = ("config.toml", "opencode.json", "web-password")


def opencode_state_dirs() -> list[Path]:
    """OpenCode's own sessions, logs and caches (it follows the XDG base dirs)."""
    home = Path.home()
    xdg = {
        "XDG_DATA_HOME": home / ".local" / "share",
        "XDG_CACHE_HOME": home / ".cache",
        "XDG_STATE_HOME": home / ".local" / "state",
    }
    return [Path(os.environ.get(var) or default) / "opencode" for var, default in xdg.items()]


def owned_paths(keep_models: bool = False, purge: bool = False) -> list[Path]:
    data = [data_dir() / p for p in DATA_PARTS if not (keep_models and p == "models")]
    conf = [config_dir() / p for p in CONFIG_PARTS]
    extra = opencode_state_dirs() if purge else []
    return [p for p in (*data, *conf, *extra) if p.exists() or p.is_symlink()]


def disk_usage(path: Path) -> int:
    if not path.is_dir() or path.is_symlink():
        return path.lstat().st_size
    return sum(f.lstat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def remove(paths: list[Path]) -> None:
    for p in paths:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
    for d in (data_dir(), config_dir()):
        with contextlib.suppress(OSError):
            d.rmdir()  # only succeeds when nothing else lives there


def uv_tool_installed() -> bool:
    """True when this oca runs from a `uv tool install` environment."""
    uv = shutil.which("uv")
    if uv is None:
        return False
    out = subprocess.run([uv, "tool", "dir"], capture_output=True, text=True, check=False)
    tools = Path(out.stdout.strip()).resolve()
    return out.returncode == 0 and Path(sys.prefix).resolve().is_relative_to(tools)
