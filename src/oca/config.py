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
VLLM_VERSION = "0.32.0"
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


@dataclass(frozen=True)
class ClusterModel:
    """A model served by vLLM on UCloud B200s. Flags follow the vLLM recipe for it."""

    id: str
    repo: str
    size_gb: float
    gpus: int
    context: int
    note: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()


CLUSTER_MODELS: dict[str, ClusterModel] = {
    m.id: m
    for m in (
        ClusterModel(
            "qwen3.8-27b-fp8",
            "Qwen/Qwen3.8-27B-FP8",
            30.9,
            1,
            262144,
            "official FP8, Apache-2.0",
            (
                "--language-model-only",
                *("--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"),
                *("--reasoning-parser", "qwen3"),
            ),
        ),
        ClusterModel(
            "deepseek-v4.1-flash",
            "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            510.3,
            2,
            262144,  # model allows 1M; 256k keeps the indexer buffer small at TP2
            "original weights, MIT; recipe's verified B200 TP2 setting",
            (
                "--language-model-only",
                *("--tensor-parallel-size", "2"),
                *("--tokenizer-mode", "deepseek_v41"),
                *("--enable-auto-tool-choice", "--tool-call-parser", "deepseek_v41"),
                *("--reasoning-parser", "deepseek_v41"),
                *("--engram-config", '{"cpu_offload":true,"use_thp":true}'),
                *("--max-num-seqs", "8", "--max-num-batched-tokens", "4096"),
                *(
                    "--compilation-config",
                    '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE",'
                    '"cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,'
                    "240,288,384,480,576,768,1020,1536,2046,3072,4092]}",
                    "--max-cudagraph-capture-size",
                    "4092",
                ),
                *(
                    "--speculative-config",
                    '{"method":"dspark","num_speculative_tokens":5,'
                    '"draft_sample_method":"probabilistic","rejection_sample_method":"block",'
                    '"enable_adaptive_verification":true}',
                ),
            ),
            (("VLLM_ENGINE_READY_TIMEOUT_S", "3600"),),
        ),
    )
}


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
    cluster_drive: str = ""  # UCloud drive for weights and job files, e.g. "/12383018"
    cluster_model: str = "qwen3.8-27b-fp8"
    cluster_hours: int = 8  # job time allocation
    cluster_idle: int = 60  # minutes without requests before the job stops itself
    cluster_port: int = 8001  # local end of the SSH tunnel

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
        if self.cluster_model not in CLUSTER_MODELS:
            raise OcaError(f"unknown cluster model {self.cluster_model!r}; see `oca models`")

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
