"""Score oca's agent on the 34 Python exercises of Aider's polyglot benchmark.

Each exercise is copied without its reference solution and approach notes. The agent gets
the instructions and the stub, and may run the tests; the original tests then decide.
The model is whatever `oca/auto` serves when the run starts (see `oca status`).

    uv run python eval/polyglot.py --variant low --jobs 4 [--only wordy,forth]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import httpx

from oca import agent, cli, install, router
from oca.config import Settings, data_dir

COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"  # Aider-AI/polyglot-benchmark, 2024-12-22
SPOILERS = (".meta", ".approaches", ".articles")
TEST = ["uv", "run", "--with", "pytest", "pytest", "-q"]


def exercises() -> Path:
    """The Python exercises, downloaded once at the pinned commit."""
    root = data_dir() / "eval" / f"polyglot-{COMMIT[:12]}"
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://codeload.github.com/Aider-AI/polyglot-benchmark/tar.gz/{COMMIT}"
        data = httpx.get(url, follow_redirects=True, timeout=300).content
        with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
            with tarfile.open(fileobj=io.BytesIO(data)) as tf:
                tf.extractall(tmp, filter="data")
            shutil.move(next(Path(tmp).iterdir()) / "python" / "exercises" / "practice", root)
    return root


def run_one(exercise: Path, out: Path, variant: str | None, timeout: int) -> dict[str, object]:
    files = json.loads((exercise / ".meta" / "config.json").read_text())["files"]
    stub, tests = files["solution"][0], files["test"][0]
    work = out / exercise.name
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(exercise, work, ignore=shutil.ignore_patterns(*SPOILERS))
    original = (work / tests).read_bytes()
    prompt = (
        f"Solve this exercise: read .docs/ for the task, then implement it in {stub}, keeping "
        f"its names and signatures. Check your work with: {' '.join(TEST)}"
    )
    cmd = [str(install.opencode()), "run", *(["--variant", variant] if variant else []), prompt]
    env = dict(os.environ, OPENCODE_CONFIG=str(agent.config_path()))
    start = time.monotonic()
    with (out / f"{exercise.name}.log").open("wb") as log, suppress(subprocess.TimeoutExpired):
        subprocess.run(
            cmd, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, timeout=timeout,
        )  # fmt: skip
    seconds = time.monotonic() - start
    (work / tests).write_bytes(original)  # the agent may have edited the tests
    try:
        check = subprocess.run(
            TEST, cwd=work, stdin=subprocess.DEVNULL, capture_output=True, timeout=120
        )
        passed = check.returncode == 0
    except subprocess.TimeoutExpired:
        passed = False
    return {"exercise": exercise.name, "passed": passed, "seconds": round(seconds, 1)}


def served_since(clock: str) -> Counter[str]:
    """Which models the router used since ``clock`` (HH:MM:SS), from its log."""
    lines = router.log_path().read_text(errors="replace").splitlines()
    return Counter(
        line.rsplit("-> ", 1)[-1] for line in lines if line[:8] >= clock and "-> " in line
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--variant", help="effort: none, low, medium, high or max")
    parser.add_argument("--jobs", type=int, default=1, help="exercises run in parallel")
    parser.add_argument("--only", help="comma-separated exercise names")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per exercise")
    args = parser.parse_args()

    settings = Settings.load()
    cli._ready(settings, yes=True)  # servers, router and OpenCode config, as `oca` does
    chosen = sorted(exercises().iterdir())
    if args.only:
        chosen = [e for e in chosen if e.name in args.only.split(",")]
    label = f"{time.strftime('%Y%m%d-%H%M%S')}-{args.variant or 'default'}"
    out = data_dir() / "eval" / "runs" / label
    out.mkdir(parents=True)
    clock = time.strftime("%H:%M:%S")
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda e: run_one(e, out, args.variant, args.timeout), chosen))

    with (out / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["exercise", "passed", "seconds"])
        writer.writeheader()
        writer.writerows(results)
    passed = sum(bool(r["passed"]) for r in results)
    times = [float(str(r["seconds"])) for r in results]
    print(f"{passed}/{len(results)} passed ({100 * passed / len(results):.0f}%)")
    print(f"per exercise: median {statistics.median(times):.0f} s, max {max(times):.0f} s")
    print(f"models served: {dict(served_since(clock))}")
    print(f"details: {out}")


if __name__ == "__main__":
    main()
