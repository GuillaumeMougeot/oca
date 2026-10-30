"""Command-line interface: ``oca`` launches the agent, subcommands manage the stack."""

from __future__ import annotations

import os
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import agent, cluster, install, router, server, web
from .config import (
    CLUSTER_MODELS,
    MODELS,
    ClusterModel,
    OcaError,
    Settings,
    data_dir,
    model_path,
)

app = typer.Typer(
    name="oca",
    help="Open Coding Agent. `oca` alone sets up what is missing and opens OpenCode.",
    add_completion=False,
    invoke_without_command=True,
)
cluster_app = typer.Typer(
    help="Serve a large model on UCloud B200s; `oca` uses it while it is up.",
    no_args_is_help=True,
)
app.add_typer(cluster_app, name="cluster")
console = Console()
Yes = Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")]
ClusterModelArg = Annotated[
    str | None, typer.Argument(help="Cluster model id (default: setting cluster_model).")
]


def _confirm(yes: bool, question: str, default: bool = True) -> None:
    if not (yes or typer.confirm(question, default=default)):
        raise typer.Exit(1)


def _ready(settings: Settings, yes: bool) -> None:
    if not install.is_installed():
        console.print("[bold]First run: installing llama.cpp and OpenCode[/]")
        install.setup()
    remote = None
    if job := cluster.read("serve"):
        try:
            with console.status(f"Connecting to {job.model} on UCloud"):
                doing = cluster.check(settings, job)
            if doing is None:
                remote = CLUSTER_MODELS[job.model]
            else:
                console.print(
                    f"[yellow]{job.model} on UCloud: {doing}. "
                    "Using the local model; oca switches when it is up.[/]"
                )
        except OcaError as e:
            console.print(f"[yellow]UCloud unavailable, using the local model:[/] {e}")
    if remote is None:
        model = settings.resolved_model
        if not model_path(model).exists():
            where = model_path(model).parent
            _confirm(yes, f"Download {model.id} ({model.size_gb} GB) to {where}?")
            install.pull(model)
        with console.status(f"Starting {model.id} on {settings.resolved_device}"):
            server.start(settings)
    router.ensure(settings)
    agent.write_config(settings, remote)


@app.callback()
def main(
    ctx: typer.Context,
    resume: Annotated[bool, typer.Option("--continue", "-c", help="Resume the last session.")] = (
        False
    ),
    session: Annotated[str | None, typer.Option("--session", "-s", help="Resume this one.")] = None,
    yes: Yes = False,
) -> None:
    if ctx.invoked_subcommand is None:
        settings = Settings.load()
        _ready(settings, yes)
        args = ["--continue"] if resume else []
        agent.launch(args + (["--session", session] if session else []))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context, yes: Yes = False) -> None:
    """Like `oca`, passing extra arguments to OpenCode (e.g. `oca run -- run "fix the tests"`)."""
    settings = Settings.load()
    _ready(settings, yes)
    agent.launch(ctx.args)


@app.command()
def sessions(
    delete: Annotated[str | None, typer.Option("--delete", help="Delete this session.")] = None,
    clear: Annotated[bool, typer.Option("--clear", help="Delete this folder's sessions.")] = False,
    yes: Yes = False,
) -> None:
    """List this folder's sessions (resume one with `oca -s <id>`), or delete them."""
    if not clear:
        agent.launch(["session", "delete", delete] if delete else ["session", "list"])
    ids = agent.sessions_in(Path.cwd())
    if not ids:
        console.print("No sessions started in this folder.")
        return
    _confirm(yes, f"Delete {len(ids)} sessions started in {Path.cwd()} or below?", default=False)
    for sid in ids:
        agent.opencode(["session", "delete", sid])
    console.print(f"Deleted {len(ids)} sessions.")


@app.command()
def update() -> None:
    """Install the newest oca release, then the binaries it pins."""
    if install.editable():
        raise OcaError("this oca runs from a source checkout; update it with `git pull`")
    if not install.uv_tool_installed():
        raise OcaError(
            "this oca was not installed with `uv tool`; update it the way you installed it"
        )
    tag = install.latest_release()
    console.print(f"oca {version('oca')} -> {tag}")
    source = f"git+{install.REPO}@{tag}"
    subprocess.run(["uv", "tool", "install", "--force", "--python", "3.12", source], check=True)
    os.execvp("oca", ["oca", "setup"])  # the new oca installs what its pins need


@app.command("web")
def web_ui(
    port: Annotated[int, typer.Option(help="Port on this machine.")] = 4096,
    stop: Annotated[bool, typer.Option("--stop", help="Stop the web server.")] = False,
    yes: Yes = False,
) -> None:
    """OpenCode in a browser for this folder, e.g. on a phone; runs in the background."""
    if stop:
        console.print("Stopped." if web.stop() else "Not running.")
        return
    settings = Settings.load()
    _ready(settings, yes)
    state = web.start(port, Path.cwd())
    console.print(
        f"OpenCode for {state['project']} at http://127.0.0.1:{port}, user opencode, "
        f"password {agent.web_password()}. It keeps running; `oca web --stop` ends it.\n"
        f"From a phone: `ssh -N -L {port}:127.0.0.1:{port} <you>@<this machine>`, "
        "then open that address."
    )


@app.command()
def setup(force: Annotated[bool, typer.Option(help="Reinstall binaries.")] = False) -> None:
    """Install pinned llama.cpp and OpenCode binaries; write the OpenCode config."""
    install.setup(force=force)
    settings = Settings.load()
    if not Settings.path().exists():
        settings.save()
    agent.write_config(settings)
    console.print(f"Installed into {data_dir() / 'bin'}. Next: `oca` (downloads the model).")


@app.command()
def pull(
    model: Annotated[str | None, typer.Argument(help="Model id, see `oca models`.")] = None,
) -> None:
    """Download local model weights (default: the configured model)."""
    settings = Settings.load()
    if model and model not in MODELS:
        raise OcaError(f"unknown model {model!r}; see `oca models`")
    console.print(f"Saved {install.pull(MODELS[model] if model else settings.resolved_model)}")


@app.command()
def models() -> None:
    """List the local and cluster model catalogs."""
    local = Table("local id", "GB", "context", "downloaded", "note")
    for m in MODELS.values():
        have = "yes" if model_path(m).exists() else ""
        local.add_row(m.id, f"{m.size_gb:g}", str(m.context), have, m.note)
    remote = Table("cluster id", "GB", "B200s", "context", "note")
    for c in CLUSTER_MODELS.values():
        remote.add_row(c.id, f"{c.size_gb:g}", str(c.gpus), str(c.context), c.note)
    console.print(local, remote)


@app.command()
def start() -> None:
    """Start the local model server and oca's endpoint (restarts them if settings changed)."""
    settings = Settings.load()
    with console.status("Starting llama-server"):
        server.start(settings)
    router.ensure(settings)
    agent.write_config(settings)
    console.print(
        f"Serving at http://127.0.0.1:{settings.port}/v1: the UCloud model when it is up, "
        f"else {settings.resolved_model.id}."
    )


@app.command()
def stop() -> None:
    """Stop the local model server and oca's endpoint."""
    stopped = [router.stop(), server.stop()]
    console.print("Stopped." if any(stopped) else "Not running.")


@app.command()
def status() -> None:
    """Show settings, installed parts and server health."""
    settings = Settings.load()
    model = settings.resolved_model
    state = server.read_state()
    table = Table(show_header=False, box=None)
    table.add_row("data", str(data_dir()))
    table.add_row("device", f"{settings.resolved_device} (setting: {settings.device})")
    table.add_row(
        "model", f"{model.id} ({'downloaded' if model_path(model).exists() else 'missing'})"
    )
    table.add_row("llama.cpp", str(install.llama_server() or "missing"))
    table.add_row("opencode", str(install.opencode() or "missing"))
    pid = router.running()
    table.add_row("endpoint", f"pid {pid}, port {settings.port}" if pid else "stopped")
    ui = web.running()
    table.add_row("web", f"port {ui['port']}, {ui['project']}" if ui else "stopped")
    if state:
        health = "ready" if server.healthy(state.port) else "loading"
        table.add_row("server", f"pid {state.pid}, port {state.port}, {health}")
    else:
        table.add_row("server", "stopped")
    job = cluster.read("serve")
    table.add_row("cluster", f"{job.model}, job {job.job_id}" if job else "down")
    console.print(table)


@app.command()
def logs(
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
    lines: Annotated[int, typer.Option("--lines", "-n")] = 40,
) -> None:
    """Show the local llama-server log."""
    os.execvp(
        "tail", ["tail", "-n", str(lines), *(["-F"] if follow else []), str(server.log_path())]
    )


@app.command()
def config(
    key: Annotated[str | None, typer.Argument()] = None,
    value: Annotated[str | None, typer.Argument()] = None,
) -> None:
    """Show settings, or set one: `oca config device cpu`."""
    settings = Settings.load()
    if key is None:
        console.print(f"# {Settings.path()}")
        for k, v in vars(settings).items():
            console.print(f"{k} = {v!r}")
        return
    if value is None:
        console.print(repr(getattr(settings, key, None)))
        return
    settings.set(key, value)
    settings.save()
    agent.write_config(settings)
    console.print(f"{key} = {getattr(settings, key)!r}. Applies on next `oca` or `oca start`.")


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def ucloud(ctx: typer.Context) -> None:
    """The bundled ucloud-api CLI, e.g. `oca ucloud login` or `oca ucloud jobs list`."""
    from ucloud_api.cli import app as ucloud_app

    ucloud_app(args=ctx.args, prog_name="oca ucloud")


@app.command()
def uninstall(
    keep_models: Annotated[bool, typer.Option(help="Keep downloaded local models.")] = False,
    purge: Annotated[
        bool, typer.Option(help="Also delete OpenCode's sessions, logs and caches.")
    ] = False,
    yes: Yes = False,
) -> None:
    """Remove oca: UCloud jobs, local server, data, settings and the `oca` command."""
    settings = Settings.load()
    if cluster.read("serve") or cluster.read("pull"):
        _confirm(yes, "Stop oca's UCloud jobs? Servers you joined keep running.")
        for kind in ("serve", "pull"):
            cluster.terminate(settings, kind)
    web.stop()
    router.stop()
    server.stop()
    paths = install.owned_paths(keep_models=keep_models, purge=purge)
    for p in paths:
        console.print(f"  {install.disk_usage(p) / 1e9:8.2f} GB  {p}")
    if paths:
        _confirm(yes, "Delete these?", default=False)
        install.remove(paths)
    if settings.cluster_drive:
        console.print(f"Kept UCloud files under {settings.cluster_drive}/hf and /oca.")
    if install.uv_tool_installed():
        subprocess.run(["uv", "tool", "uninstall", "oca"], check=False)
    else:
        console.print("Remove the oca package with the tool that installed it.")


# -- cluster -----------------------------------------------------------------------------


def _cluster_model(model: str | None, settings: Settings) -> ClusterModel:
    mid = model or settings.cluster_model
    if mid not in CLUSTER_MODELS:
        raise OcaError(f"unknown cluster model {mid!r}; see `oca models`")
    return CLUSTER_MODELS[mid]


@cluster_app.command("pull")
def cluster_pull(model: ClusterModelArg = None, yes: Yes = False) -> None:
    """Download weights to the UCloud drive from a 2-vCPU job (no GPU-hours)."""
    settings = Settings.load()
    m = _cluster_model(model, settings)
    if cluster.weights_ready(settings, m):
        console.print(f"{m.id} is already on {settings.cluster_drive}.")
        return
    worst = 2 * cluster.PULL_HOURS
    _confirm(yes, f"Download {m.repo} ({m.size_gb:g} GB); at most {worst} CPU core-hours?")
    job = cluster.pull(settings, m)
    console.print(f"Job {job.job_id} submitted. Follow it: `oca cluster logs pull -f`.")


@cluster_app.command("up")
def cluster_up(model: ClusterModelArg = None, yes: Yes = False) -> None:
    """Serve a model on UCloud, or join a project member's server, and make it the default."""
    settings = Settings.load()
    m = _cluster_model(model, settings)
    job = cluster.read("serve")
    if job and cluster.ended(settings, job):
        job = None  # it finished since: start or join a new one
    if job and job.model != m.id:
        what = "Stop" if job.owner else "Leave"
        _confirm(yes, f"{what} {job.model} and switch to {m.id}? `oca` works locally meanwhile.")
        cluster.terminate(settings, "serve")
        job = None
    if job is None:
        if key := cluster.unregistered_key():
            raise OcaError(
                "UCloud does not know this machine's SSH key, so the tunnel would fail. "
                f"Register it once: `oca ucloud ssh-keys add {key} --title $(hostname)`"
            )
        if shared := cluster.join(settings, m):
            job, starter = shared
            console.print(f"Joining {m.id}, started by {starter} (job {job.job_id}).")
    if job is None:
        if m.size_gb > 100 and not cluster.weights_ready(settings, m):
            raise OcaError(
                f"{m.id} ({m.size_gb:g} GB) is not on the drive yet. Run `oca cluster pull "
                f"{m.id}` first, so the download does not spend GPU-hours."
            )
        hours, gpus = settings.cluster_hours, m.gpus
        _confirm(
            yes,
            f"Serve {m.id} on {gpus}x B200 for up to {hours} h ({gpus * hours} GPU-hours max, "
            f"stops after {settings.cluster_idle} idle min)?",
        )
        job = cluster.submit_serve(settings, m)
        console.print(f"Job {job.job_id} submitted.")
    began = time.monotonic()
    with console.status("waiting") as st:
        cluster.wait_ready(settings, job, st.update)
    router.ensure(settings)
    agent.write_config(settings, m)
    minutes = (time.monotonic() - began) / 60
    console.print(f"Ready after {minutes:.0f} min. `oca` now uses {m.id}.")


@cluster_app.command("down")
def cluster_down() -> None:
    """Stop the serve job (or leave a joined one); `oca` goes back to the local model."""
    settings = Settings.load()
    job = cluster.read("serve")
    cluster.terminate(settings, "serve")
    agent.write_config(settings)
    if job is None:
        console.print("Nothing was running.")
    else:
        console.print("Stopped." if job.owner else "Left; the server keeps running for others.")


@cluster_app.command("status")
def cluster_status() -> None:
    """Show oca's UCloud jobs: state, time left, tunnel and server health."""
    settings = Settings.load()
    table = Table(show_header=False, box=None)
    for kind in ("serve", "pull"):
        job = cluster.read(kind)
        if job is None:
            continue
        status = cluster.job_status(job)
        left = "" if job.owner else ", joined"
        expires = status.get("expiresAt")
        if status.get("state") == "RUNNING" and isinstance(expires, int | float):
            left += f", {(expires / 1000 - time.time()) / 3600:.1f} h left"
        state = status.get("state", "state hidden")
        table.add_row(kind, f"{job.model}, job {job.job_id}, {state}{left}")
    if cluster.read("serve"):
        table.add_row("server", "ready" if cluster.healthy(settings) else "not reachable")
    console.print(table if table.rows else "No oca jobs.")


@cluster_app.command("logs")
def cluster_logs(
    kind: Annotated[str, typer.Argument(help="serve or pull")] = "serve",
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
    lines: Annotated[int, typer.Option("--lines", "-n")] = 40,
) -> None:
    """Show a job's log from the drive."""
    settings = Settings.load()
    job = cluster.read(kind)
    if job is None:
        raise OcaError(f"no {kind} job; see `oca cluster status`")
    if not follow:
        console.print(cluster.log_tail(settings, job, lines), markup=False, highlight=False)
        return
    shown = 0
    while True:
        text = cluster.log_text(settings, job)
        print(text[shown:], end="", flush=True)
        shown = len(text)
        time.sleep(5)


@cluster_app.command("extend")
def cluster_extend(hours: Annotated[int, typer.Argument()] = 1) -> None:
    """Add time to the serve job."""
    job = cluster.read("serve")
    if job is None:
        raise OcaError("no serve job")
    if not job.owner:
        raise OcaError("only the member who started this server can extend it")
    cluster.extend(job, hours)
    console.print(f"Extended job {job.job_id} by {hours} h.")


def entrypoint() -> None:
    try:
        app()
    except OcaError as e:
        console.print(f"[red]error:[/] {e}")
        raise SystemExit(1) from None
