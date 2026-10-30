"""Score oca's agent on a task suite; the model is whatever `oca/auto` serves at the start.

Suites:
- polyglot: the 34 Python exercises of Aider's polyglot benchmark, without their reference
  solutions. The agent gets the instructions and the stub; the exercise's tests decide.
- history: changes from oca's own history. Each task starts from a commit's parent with
  that commit's tests, which fail; the commit message is the request, and its tests decide.

The agent may run the tests; the original tests are restored before scoring.

    uv run python eval/run.py --suite history --variant high --jobs 2 [--only 24f112d]
    uv run python eval/run.py --suite history --check   # each task fails before, passes after
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import httpx

from oca import agent, cli, install, router
from oca.config import Settings, data_dir

REPO = Path(__file__).resolve().parents[1]
POLYGLOT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"  # Aider-AI/polyglot-benchmark, 2024-12-22
SPOILERS = (".meta", ".approaches", ".articles")
# Commits whose tests fail on their parent (checked with --check). Left out: the two phase
# commits (whole subsystems at once) and one-line changes.
HISTORY = ["5d78a6e", "24f112d", "082c77e", "3af6145", "848cc1f"]
HISTORY += ["bd219fb", "6d415d3", "9802449", "cbec775"]


@dataclass
class Task:
    name: str
    prepare: Callable[[Path], None]  # fills the empty work dir
    prompt: str
    tests: dict[str, bytes]  # restored before scoring
    check: list[str]  # exit 0 means solved


def polyglot() -> list[Task]:
    root = data_dir() / "eval" / f"polyglot-{POLYGLOT[:12]}"
    if not root.exists():  # download once, at the pinned commit
        root.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://codeload.github.com/Aider-AI/polyglot-benchmark/tar.gz/{POLYGLOT}"
        data = httpx.get(url, follow_redirects=True, timeout=300).content
        with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
            with tarfile.open(fileobj=io.BytesIO(data)) as tf:
                tf.extractall(tmp, filter="data")
            shutil.move(next(Path(tmp).iterdir()) / "python" / "exercises" / "practice", root)
    tasks = []
    for ex in sorted(root.iterdir()):
        files = json.loads((ex / ".meta" / "config.json").read_text())["files"]
        stub, test = files["solution"][0], files["test"][0]
        tasks.append(
            Task(
                ex.name,
                lambda work, ex=ex: shutil.copytree(
                    ex, work, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*SPOILERS)
                ),
                f"Solve this exercise: read .docs/ for the task, then implement it in {stub}, "
                "keeping its names and signatures. Check your work with: "
                "uv run --with pytest pytest -q",
                {test: (ex / test).read_bytes()},
                ["uv", "run", "--with", "pytest", "pytest", "-q"],
            )
        )
    return tasks


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True).stdout


def _unpack(rev: str, work: Path) -> None:
    """The tree at ``rev``, without git history, so the agent cannot read the answer."""
    with tarfile.open(fileobj=io.BytesIO(_git("archive", rev))) as tf:
        tf.extractall(work, filter="data")


def history() -> list[Task]:
    tasks = []
    for sha in HISTORY:
        message = _git("log", "-1", "--format=%B", sha).decode()
        request = re.sub(r"\n(Co-Authored-By|Claude-Session):.*", "", message).strip()
        changed = _git("diff", "--name-only", f"{sha}^", sha, "--", "tests").decode().split()
        tests = {path: _git("show", f"{sha}:{path}") for path in changed}

        def prepare(work: Path, sha: str = sha, tests: dict[str, bytes] = tests) -> None:
            _unpack(f"{sha}^", work)
            for path, data in tests.items():
                (work / path).write_bytes(data)
            git = ["git", "-c", "user.name=oca", "-c", "user.email=oca@localhost", "-C", str(work)]
            subprocess.run([*git, "init", "-q"], check=True)
            subprocess.run([*git, "add", "-A"], check=True)
            subprocess.run([*git, "commit", "-qm", "start"], check=True)

        tasks.append(
            Task(
                sha,
                prepare,
                f"Make this change to the repository:\n\n{request}\n\n"
                "Tests describing it are in tests/ and fail now; make them pass. Run them with: "
                "uv run pytest -q tests/test_oca.py",
                tests,
                ["uv", "run", "pytest", "-q", "tests/test_oca.py"],
            )
        )
    return tasks


def _passes(check: list[str], work: Path) -> bool:
    try:
        done = subprocess.run(
            check, cwd=work, stdin=subprocess.DEVNULL, capture_output=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


def run_one(task: Task, out: Path, variant: str | None, timeout: int) -> dict[str, object]:
    work = out / task.name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    task.prepare(work)
    cmd = [str(install.opencode()), "run", *(["--variant", variant] if variant else [])]
    # OpenCode takes its project from $PWD, which subprocess does not update.
    env = dict(os.environ, OPENCODE_CONFIG=str(agent.config_path(unattended=True)), PWD=str(work))
    start = time.monotonic()
    with (out / f"{task.name}.log").open("wb") as log, suppress(subprocess.TimeoutExpired):
        subprocess.run(
            [*cmd, task.prompt], cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, timeout=timeout,
        )  # fmt: skip
    seconds = time.monotonic() - start
    for path, data in task.tests.items():  # the agent may have edited the tests
        (work / path).write_bytes(data)
    return {"task": task.name, "passed": _passes(task.check, work), "seconds": round(seconds)}


def check(tasks: list[Task], suite: str) -> None:
    """Each task must fail as given and, for history, pass with its commit."""
    for task in tasks:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            task.prepare(work)
            before = _passes(task.check, work)
        after = None
        if suite == "history":
            with tempfile.TemporaryDirectory() as tmp:
                _unpack(task.name, Path(tmp))
                after = _passes(task.check, Path(tmp))
        print(f"{task.name}: given {'passes (trivial)' if before else 'fails'}, solution {after}")


def served_since(clock: str) -> Counter[str]:
    """Which models the router used since ``clock`` (HH:MM:SS), from its log."""
    lines = router.log_path().read_text(errors="replace").splitlines()
    return Counter(
        line.rsplit("-> ", 1)[-1] for line in lines if line[:8] >= clock and "-> " in line
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--suite", choices=["polyglot", "history"], default="polyglot")
    parser.add_argument("--variant", help="effort: none, low, medium, high or max")
    parser.add_argument("--jobs", type=int, default=1, help="tasks run in parallel")
    parser.add_argument("--only", help="comma-separated task names")
    parser.add_argument("--timeout", type=int, default=1200, help="seconds per task")
    parser.add_argument("--check", action="store_true", help="validate tasks, run no agent")
    args = parser.parse_args()

    tasks = polyglot() if args.suite == "polyglot" else history()
    if args.only:
        tasks = [t for t in tasks if t.name in args.only.split(",")]
    if args.check:
        return check(tasks, args.suite)
    cli._ready(Settings.load(), yes=True)  # servers, router and OpenCode config, as `oca` does
    label = f"{time.strftime('%Y%m%d-%H%M%S')}-{args.suite}-{args.variant or 'default'}"
    out = data_dir() / "eval" / "runs" / label
    out.mkdir(parents=True)
    clock = time.strftime("%H:%M:%S")
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda t: run_one(t, out, args.variant, args.timeout), tasks))

    with (out / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["task", "passed", "seconds"])
        writer.writeheader()
        writer.writerows(results)
    passed = sum(bool(r["passed"]) for r in results)
    times = [float(str(r["seconds"])) for r in results]
    print(f"{passed}/{len(results)} passed ({100 * passed / len(results):.0f}%)")
    print(f"per task: median {statistics.median(times):.0f} s, max {max(times):.0f} s")
    print(f"models served: {dict(served_since(clock))}")
    print(f"details: {out}")


if __name__ == "__main__":
    main()
