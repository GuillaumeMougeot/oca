"""Paths, pinned versions, the model catalog and user settings."""

from __future__ import annotations

import getpass
import os
import shutil
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import tomli_w
from platformdirs import user_config_path, user_data_path

# Bump deliberately and journal why: upstream releases change flags and formats.
LLAMA_CPP_TAG = "b11982"
OPENCODE_VERSION = "1.19.5"
DEVICES = ("auto", "cpu", "gpu")


class OcaError(Exception):
    """A user-facing failure; the CLI prints it without a traceback."""


@dataclass(frozen=True)
class Model:
    id: str
    repo: str
    file: str
    size_gb: float
    context: int
    note: str
    # Sampling recommended by the model card.
    args: tuple[str, ...] = ("--temp", "0.6", "--top-p", "0.95", "--top-k", "20")


MODELS: dict[str, Model] = {
    m.id: m
    for m in (
        Model(
            "qwen3.6-35b-a3b",
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            22.4,
            65536,
            "MoE, 3B active, Apache-2.0: CPU default",
        ),
        Model(
            "qwen3.8-27b",
            "unsloth/Qwen3.8-27B-GGUF",
            "Qwen3.8-27B-UD-Q4_K_XL.gguf",
            17.6,
            65536,
            "dense, Apache-2.0: GPU default",
        ),
    )
}
DEFAULT_MODEL = {"cpu": "qwen3.6-35b-a3b", "gpu": "qwen3.8-27b"}


def config_dir() -> Path:
    return Path(os.environ.get("OCA_CONFIG_DIR") or user_config_path("oca"))


def data_dir() -> Path:
    """``$OCA_DATA``, else ``/data/$USER/oca`` when that disk exists, else the XDG data dir."""
    if env := os.environ.get("OCA_DATA"):
        return Path(env)
    shared = Path("/data") / getpass.getuser()
    if shared.is_dir() and os.access(shared, os.W_OK):
        return shared / "oca"
    return user_data_path("oca")


def model_path(model: Model) -> Path:
    return data_dir() / "models" / model.repo / model.file


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


@dataclass
class Settings:
    device: str = "auto"
    model: str = "auto"
    port: int = 8080
    threads: int = 0  # 0: llama.cpp picks
    context: int = 0  # 0: the model's default

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "gpu" if has_nvidia_gpu() else "cpu"
        return self.device

    @property
    def resolved_model(self) -> Model:
        return MODELS[DEFAULT_MODEL[self.resolved_device] if self.model == "auto" else self.model]

    @property
    def resolved_context(self) -> int:
        return self.context or self.resolved_model.context

    def validate(self) -> None:
        if self.device not in DEVICES:
            raise OcaError(f"device must be one of {', '.join(DEVICES)}, got {self.device!r}")
        if self.model != "auto" and self.model not in MODELS:
            raise OcaError(f"unknown model {self.model!r}; see `oca models`")

    def set(self, key: str, value: str) -> None:
        names = {f.name for f in fields(self)}
        if key not in names:
            raise OcaError(f"unknown key {key!r}; keys: {', '.join(sorted(names))}")
        current = getattr(self, key)
        try:
            setattr(self, key, type(current)(value))
        except ValueError as e:
            raise OcaError(f"{key} expects {type(current).__name__}, got {value!r}") from e
        self.validate()

    @staticmethod
    def path() -> Path:
        return config_dir() / "config.toml"

    @classmethod
    def load(cls) -> Settings:
        path = cls.path()
        raw = tomllib.loads(path.read_text()) if path.exists() else {}
        names = {f.name for f in fields(cls)}
        settings = cls(**{k: v for k, v in raw.items() if k in names})
        settings.validate()
        return settings

    def save(self) -> None:
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomli_w.dumps(asdict(self)))
