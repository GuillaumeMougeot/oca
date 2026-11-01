"""Command-line interface: ``oca`` launches the agent, subcommands manage the stack."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from contextlib import suppress
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import click
import httpx
import typer
from rich.console import Console
from rich.table import Table

from . import agent, cluster, install, remotes, router, serve, server, web
from .config import (
    CLUSTER_MODELS,
    DEFAULT_MODEL,
    MODELS,
    NAME,
    ClusterModel,
    Down,
    OcaError,
    Remote,
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
remote_app = typer.Typer(
    help="Servers on other machines: UCloud, one you reach over SSH (e.g. a DGX Spark), or "
    "a URL. Alone, it lists all servers.",
    invoke_without_command=True,
)
app.add_typer(remote_app, name="remote")
web_app = typer.Typer(help="OpenCode in a browser for this folder, e.g. on a phone.")
app.add_typer(web_app, name="web")
console = Console()
Yes = Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")]
NameArg = Annotated[str | None, typer.Argument(help="The remote (default: the default one).")]
ModelArg = Annotated[str | None, typer.Argument(help="Model id (default: the remote's).")]


def _confirm(yes: bool, question: str, default: bool = True) -> None:
    if not (yes or typer.confirm(question, default=default)):
        raise typer.Exit(1)


def _ready(settings: Settings, name: str | None = None, yes: bool = False) -> str:
    """Make server ``name`` (default: the default server) answer, or another meanwhile."""
    if not install.is_installed():
        console.print("[bold]First run: installing llama.cpp and OpenCode[/]")
        install.setup()
    if not Settings.path().exists() and not yes and sys.stdin.isatty():
        _first_run(settings)
    name = name or settings.default
    if name not in settings.servers():
        raise OcaError(f"unknown server {name!r}; see `oca remote`")
    doing = None if name == "local" else _connect(settings, name, yes)
    others: list[str] = []
    if doing is not None:
        others = [n for n in settings.remotes if n != name and _answers(settings, n)]
        meanwhile = others[0] if others else "the local model"
        console.print(
            f"[yellow]{name}: {doing}. `oca` uses {meanwhile} meanwhile and switches by itself.[/]"
        )
    if name == "local" or (doing is not None and not others):
        _local(settings, yes, required=name == "local")
    router.ensure(settings)
    agent.write_config(settings, titles=name != "local" and doing is None)
    return name


def _local(settings: Settings, yes: bool, required: bool) -> None:
    """Start the local model server; download the model first if it is ``required``."""
    model = settings.resolved_model
    if not model_path(model).exists():
        if not required:
            console.print("[yellow]No local model to use meanwhile; `oca pull` downloads one.[/]")
            return
        where = model_path(model).parent
        _confirm(yes, f"Download {model.id} ({model.size_gb} GB) to {where}?")
        install.pull(model)
    with console.status(f"Starting {model.id} on {settings.resolved_device}"):
        server.start(settings)


def _answers(settings: Settings, name: str) -> bool:
    found = remotes.backend(settings, name)
    return found is not None and remotes.healthy(found)


def _connect(settings: Settings, name: str, yes: bool) -> str | None:
    """Connect to remote ``name``, offering to start it; None when it answers."""
    try:
        with console.status(f"Connecting to {name}"):
            return remotes.check(settings, name)
    except Down:
        pass
    except OcaError as e:
        console.print(f"[yellow]{name}:[/] {e}")
        return "unreachable"
    r = settings.remotes[name]
    if r.kind == "url":
        return "not answering"
    if r.kind == "ssh" and not (yes or typer.confirm(f"Start {r.model} on {r.host}?")):
        return "not started"
    try:
        return _up(settings, name, None, yes)
    except typer.Exit:
        return "not started"


def _first_run(settings: Settings) -> None:
    """Ask once which servers to use besides this machine; `oca remote add` adds more."""
    console.print("[bold]Welcome to oca.[/] It runs a model on this machine and can use others.")
    if typer.confirm("Use UCloud B200 GPUs (needs a UCloud project)?", default=False):
        try:
            _add(settings, "ucloud", "ucloud", None, yes=False)
        except OcaError as e:
            console.print(f"[yellow]Skipped UCloud:[/] {e}\nLater: `oca remote add ucloud ucloud`.")
    host = typer.prompt(
        "A machine you reach over SSH, e.g. a DGX Spark (user@host; Enter to skip)",
        default="",
        show_default=False,
    )
    if host:
        short = re.sub(r"[^a-z0-9-]", "-", host.split("@")[-1].split(".")[0].lower())
        name = typer.prompt("Its name in oca", default=short.strip("-") or "box")
        try:
            _add(settings, name, host, None, yes=False)
        except OcaError as e:
            console.print(f"[yellow]Skipped {host}:[/] {e}\nLater: `oca remote add {name} {host}`.")
    settings.save()


@app.callback()
def main(
    ctx: typer.Context,
    resume: Annotated[bool, typer.Option("--continue", "-c", help="Resume the last session.")] = (
        False
    ),
    session: Annotated[str | None, typer.Option("--session", "-s", help="Resume this one.")] = None,
    remote: Annotated[
        str | None,
        typer.Option("--remote", "-r", help="Use this server, not the default; see `oca remote`."),
    ] = None,
    yes: Yes = False,
) -> None:
    if ctx.invoked_subcommand is None:
        settings = Settings.load()
        name = _ready(settings, remote, yes)
        args = ["--continue"] if resume else []
        args += ["--session", session] if session else []
        args += ["--model", f"{agent.PROVIDER}/{name}"] if remote else []
        agent.launch(args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context, yes: Yes = False) -> None:
    """Like `oca`, passing extra arguments to OpenCode (e.g. `oca run -- run "fix the tests"`)."""
    settings = Settings.load()
    _ready(settings, None, yes)
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


@web_app.callback(invoke_without_command=True)
def web_ui(
    ctx: typer.Context,
    port: Annotated[int, typer.Option(help="Port on this machine.")] = 4096,
    yes: Yes = False,
) -> None:
    """OpenCode in a browser for this folder, e.g. on a phone; runs in the background."""
    if ctx.invoked_subcommand is not None:
        return
    settings = Settings.load()
    _ready(settings, None, yes)
    state = web.start(port, Path.cwd())
    console.print(
        f"OpenCode for {state['project']} at http://127.0.0.1:{port}, user opencode, "
        f"password {agent.web_password()}. It keeps running; `oca web stop` ends it.\n"
        f"From a phone: `ssh -N -L {port}:127.0.0.1:{port} <you>@<this machine>`, "
        "then open that address."
    )


@web_app.command("stop")
def web_stop() -> None:
    """Stop the web server."""
    console.print("Stopped." if web.stop() else "Not running.")


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
    """List the models: llama.cpp ones (this machine, SSH remotes) and UCloud ones."""
    local = Table("llama.cpp id", "GB", "context", "downloaded", "note")
    for m in MODELS.values():
        have = "yes" if model_path(m).exists() else ""
        local.add_row(m.id, f"{m.size_gb:g}", str(m.context), have, m.note)
    remote = Table("UCloud id", "GB", "B200s", "context", "note")
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
        f"Serving at http://127.0.0.1:{settings.port}/v1, one model per server "
        f"({', '.join(settings.servers())}); {settings.default} by default."
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
    for name in settings.remotes:
        table.add_row(name, f"{remotes.label(settings, name)}, {_state(settings, name)}")
    table.add_row("default", f"{settings.default} (`oca remote default <server>` changes it)")
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
        console.print(f"# {Settings.path()}; `oca remote` lists the remotes")
        for k, v in vars(settings).items():
            if k != "remotes":
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
    ucloud = next((r for r in settings.remotes.values() if r.kind == "ucloud"), None)
    if ucloud and (cluster.read("serve") or cluster.read("pull")):
        _confirm(yes, "Stop oca's UCloud jobs? Servers you joined keep running.")
        for kind in ("serve", "pull"):
            cluster.terminate(ucloud, kind)
    for name, r in settings.remotes.items():
        if r.kind == "ssh":
            with suppress(OcaError):
                remotes.down(settings, name)  # stops the servers you started there
    web.stop()
    router.stop()
    server.stop()
    paths = install.owned_paths(keep_models=keep_models, purge=purge)
    for p in paths:
        console.print(f"  {install.disk_usage(p) / 1e9:8.2f} GB  {p}")
    if paths:
        _confirm(yes, "Delete these?", default=False)
        install.remove(paths)
    if ucloud:
        console.print(f"Kept UCloud files under {ucloud.drive}/hf and /oca.")
    if install.uv_tool_installed():
        subprocess.run(["uv", "tool", "uninstall", "oca"], check=False)
    else:
        console.print("Remove the oca package with the tool that installed it.")


# -- remotes ---------------------------------------------------------------------------


def _remote(settings: Settings, name: str | None) -> str:
    """``name``, else the default server if it is a remote, else the only remote."""
    if name is None:
        if settings.default != "local":
            name = settings.default
        elif len(settings.remotes) == 1:
            name = next(iter(settings.remotes))
        else:
            raise OcaError("which remote? See `oca remote`; `oca remote add` adds one")
    if name not in settings.remotes:
        raise OcaError(f"unknown remote {name!r}; see `oca remote`")
    return name


def _cluster_model(mid: str) -> ClusterModel:
    if mid not in CLUSTER_MODELS:
        raise OcaError(f"unknown UCloud model {mid!r}; see `oca models`")
    return CLUSTER_MODELS[mid]


def _state(settings: Settings, name: str) -> str:
    if name == "local":
        st = server.read_state()
        return "stopped" if st is None else "ready" if server.healthy(st.port) else "loading"
    r, found = settings.remotes[name], remotes.backend(settings, name)
    if found is None:
        return "down" if r.kind != "url" else "not answering"
    if remotes.healthy(found):
        return "ready"
    job = cluster.read("serve") if r.kind == "ucloud" else None
    if job is None:
        return "starting"
    try:
        status = cluster.job_status(job)
    except OcaError:
        return "starting"
    state = str(status.get("state", "starting")).lower().replace("_", " ")
    expires = status.get("expiresAt")
    if isinstance(expires, int | float):
        state += f", {(expires / 1000 - time.time()) / 3600:.1f} h left"
    return state


@remote_app.callback(invoke_without_command=True)
def remote_list(ctx: typer.Context) -> None:
    """List the servers; * marks the default, which plain `oca` uses."""
    if ctx.invoked_subcommand is not None:
        return
    settings = Settings.load()
    table = Table("", "server", "model", "context", "state", box=None)
    for name in settings.servers():
        size = f"{remotes.context(settings, name) // 1000}k"
        mark = "*" if name == settings.default else ""
        table.add_row(mark, name, remotes.label(settings, name), size, _state(settings, name))
    console.print(table)
    console.print(
        "`oca -r <server>` starts a session on another one; in OpenCode, ctrl+x m switches."
    )


def _add(settings: Settings, name: str, target: str, model: str | None, yes: bool) -> None:
    if name in settings.servers():
        raise OcaError(f"{name} exists; `oca remote remove {name}` first")
    if not NAME.fullmatch(name):
        raise OcaError(f"name {name!r}: use lowercase letters, digits and dashes")
    port = max([8000, *(r.port for r in settings.remotes.values())]) + 1
    if target == "ucloud":
        remote = _new_ucloud(settings, model, port)
    elif "://" in target:
        remote = _new_url(target, model)
    else:
        remote = _new_ssh(target, model, port, yes)
    settings.remotes[name] = remote
    settings.validate()
    first = settings.default == "local"
    if (yes and first) or (not yes and typer.confirm(f"Use {name} by default?", default=first)):
        settings.default = name
    settings.save()
    agent.write_config(settings)
    console.print(
        f"Added {name}: {remotes.label(settings, name)}. "
        + ("Plain `oca` uses it; " if settings.default == name else f"`oca -r {name}` uses it; ")
        + ("it answers already." if remote.kind == "url" else f"`oca remote up {name}` starts it.")
    )


def _new_ucloud(settings: Settings, model: str | None, port: int) -> Remote:
    if any(r.kind == "ucloud" for r in settings.remotes.values()):
        raise OcaError("oca drives one UCloud remote; `oca remote` lists it")
    drives = cluster.drives()  # needs `oca ucloud login`
    table = Table("drive", "title", box=None)
    for d in drives:
        table.add_row(f"/{d.id}", d.title)
    console.print(table)
    drive = typer.prompt(
        "Drive for model weights and job files", default=f"/{drives[0].id}" if drives else None
    )
    return Remote("ucloud", model=model or "qwen3.8-flash-next", port=port, drive=drive)


def _new_url(url: str, model: str | None) -> Remote:
    base = url.rstrip("/").removesuffix("/v1")
    key = typer.prompt("Its API key (Enter for none)", default="", show_default=False)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        answer = httpx.get(f"{base}/v1/models", headers=headers, timeout=10)
        answer.raise_for_status()
        listed = answer.json()["data"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        raise OcaError(f"{base}/v1/models lists no models: {e}") from e
    ids = [m.get("id") for m in listed]
    if not model:
        model = ids[0] if len(ids) == 1 else typer.prompt("Model", default=ids[0] if ids else None)
    found = next((m.get("max_model_len") for m in listed if m.get("id") == model), None)  # vLLM
    size = int(found) if found else typer.prompt("Its context window, in tokens", type=int)
    return Remote("url", model=str(model), url=base, key=key, context=size)


def _new_ssh(host: str, model: str | None, port: int, yes: bool) -> Remote:
    with console.status(f"Looking at {host}"):
        found = remotes.probe(host)
    console.print(f"{host}: {found.arch}, {found.gpu or 'no NVIDIA GPU'}.")
    shared = ""
    if found.shared:  # someone installed it for everyone: join that
        shared = remotes.SHARED_ROOT
        console.print(f"{host} has oca for all its users in {shared}; using it.")
    else:
        who = "me"
        if not yes:
            who = typer.prompt(
                f"Install oca on {host} for you only, or for all its users?",
                type=click.Choice(["me", "all"]),
                default="me",
            )
        if who == "all":
            shared = typer.prompt("Folder for it", default=remotes.SHARED_ROOT)
            if remotes.probe(host, shared).shared:
                console.print(f"{host} has oca for all its users in {shared}; using it.")
                return Remote(
                    "ssh", model=model or DEFAULT_MODEL["cpu"], port=port, host=host, shared=shared
                )
        if shared or not found.personal:
            with console.status(f"Installing oca on {host}; this takes a few minutes"):
                remotes.install(host, shared)
    # A MoE model: fast on any GPU, even on a DGX Spark's slower memory.
    return Remote("ssh", model=model or DEFAULT_MODEL["cpu"], port=port, host=host, shared=shared)


@remote_app.command("add")
def remote_add(
    name: Annotated[str, typer.Argument(help="A short name, e.g. spark.")],
    target: Annotated[
        str,
        typer.Argument(
            help="`ucloud`; user@host of a machine you reach over SSH; or the URL of an "
            "OpenAI-compatible server."
        ),
    ],
    model: Annotated[str | None, typer.Option(help="Its model; see `oca models`.")] = None,
    yes: Yes = False,
) -> None:
    """Add a server: UCloud, a machine you reach over SSH (e.g. a DGX Spark), or a URL."""
    _add(Settings.load(), name, target, model, yes)


@remote_app.command("remove")
def remote_remove(name: Annotated[str, typer.Argument()]) -> None:
    """Forget a remote; stops the server you started there."""
    settings = Settings.load()
    name = _remote(settings, name)
    r = settings.remotes[name]
    with suppress(OcaError):
        if r.kind == "ucloud":
            cluster.terminate(r, "serve")
        elif r.kind == "ssh":
            remotes.down(settings, name)
    del settings.remotes[name]
    if settings.default == name:
        settings.default = "local"
    settings.save()
    agent.write_config(settings)
    console.print(f"Removed {name}. Plain `oca` uses {settings.default}.")


@remote_app.command("default")
def remote_default(name: Annotated[str, typer.Argument(help="A remote, or local.")]) -> None:
    """Set the server plain `oca` uses."""
    settings = Settings.load()
    if name not in settings.servers():
        raise OcaError(f"unknown server {name!r}; see `oca remote`")
    settings.default = name
    settings.save()
    agent.write_config(settings)
    console.print(f"Plain `oca` uses {name}.")


def _up(settings: Settings, name: str, model: str | None, yes: bool) -> str | None:
    """Start remote ``name``'s server, or join a running one; None once it answers."""
    r = settings.remotes[name]
    if r.kind == "url":
        return remotes.check(settings, name)
    if model and model != r.model:
        r.model = model
        settings.validate()
        settings.save()
    if r.kind == "ssh":
        with console.status(f"Starting {r.model} on {r.host}"):
            doing = remotes.up(settings, name)
        conn = remotes.read_conn(name)
        if conn and conn.context != r.context:  # what OpenCode should plan for next time
            r.context = conn.context
            settings.save()
        return doing
    m = _cluster_model(r.model)
    job = cluster.read("serve")
    if job and cluster.ended(r, job):
        job = None  # it finished since: start or join a new one
    if job and job.model != m.id:
        what = "Stop" if job.owner else "Leave"
        _confirm(yes, f"{what} {job.model} and switch to {m.id}? Another server answers meanwhile.")
        cluster.terminate(r, "serve")
        job = None
    if job is None:
        if key := cluster.unregistered_key():
            raise OcaError(
                "UCloud does not know this machine's SSH key, so the tunnel would fail. "
                f"Register it once: `oca ucloud ssh-keys add {key} --title $(hostname)`"
            )
        if shared := cluster.join(r, m):
            job, starter = shared
            console.print(f"Joining {m.id}, started by {starter} (job {job.job_id}).")
    if job is None:
        if m.size_gb > 100 and not cluster.weights_ready(r, m):
            raise OcaError(
                f"{m.id} ({m.size_gb:g} GB) is not on the drive yet. Run `oca remote pull "
                f"{name} {m.id}` first, so the download does not spend GPU-hours."
            )
        _confirm(
            yes,
            f"Serve {m.id} on {m.gpus}x B200 for up to {r.hours} h ({m.gpus * r.hours} "
            f"GPU-hours max, stops after {r.idle} idle min)?",
        )
        job = cluster.submit_serve(r, m)
        console.print(f"Job {job.job_id} submitted.")
    router.ensure(settings)  # its background check opens the tunnel once the job runs
    return cluster.check(r, job)  # one attempt: it may answer already


@remote_app.command("up")
def remote_up(
    name: NameArg = None,
    model: ModelArg = None,
    wait: Annotated[bool, typer.Option("--wait", "-w", help="Wait until it answers.")] = False,
    yes: Yes = False,
) -> None:
    """Start a remote's server, or join a running one, and return.

    `oca` uses another server meanwhile and switches once this one answers.
    """
    settings = Settings.load()
    name = _remote(settings, name)
    doing = _up(settings, name, model, yes)
    if wait and doing is not None:
        began = time.monotonic()
        with console.status("waiting") as st:
            remotes.wait_ready(settings, name, st.update)
        console.print(f"Ready after {(time.monotonic() - began) / 60:.0f} min.")
        doing = None
    router.ensure(settings)
    agent.write_config(settings, titles=doing is None and settings.default == name)
    how = "plain `oca`" if settings.default == name else f"`oca -r {name}`"
    if doing is None:
        console.print(f"{name} answers; {how} uses it.")
    else:
        console.print(
            f"{name}: {doing}. {how} switches to it once it answers; `oca remote` shows it."
        )


@remote_app.command("down")
def remote_down(name: NameArg = None) -> None:
    """Stop the server you started there, or leave one you joined."""
    settings = Settings.load()
    name = _remote(settings, name)
    r = settings.remotes[name]
    if r.kind == "url":
        raise OcaError(f"{name} is not oca's to stop; `oca remote remove {name}` forgets it")
    if r.kind == "ucloud":
        job = cluster.read("serve")
        cluster.terminate(r, "serve")
        stopped = None if job is None else job.owner
    else:
        stopped = remotes.down(settings, name)
    agent.write_config(settings)
    if stopped is None:
        console.print("Nothing was running.")
    else:
        console.print("Stopped." if stopped else "Left; the server keeps running for others.")


@remote_app.command("pull")
def remote_pull(name: NameArg = None, model: ModelArg = None, yes: Yes = False) -> None:
    """Download a model there; on UCloud from a 2-vCPU job, without GPU-hours."""
    settings = Settings.load()
    name = _remote(settings, name)
    r = settings.remotes[name]
    if r.kind == "ssh":
        command = f"{remotes.oca_command(r)} pull {shlex.quote(model or r.model)}"
        os.execvp("ssh", ["ssh", r.host, command])
    if r.kind != "ucloud":
        raise OcaError(f"{name} is not oca's to download to")
    m = _cluster_model(model or r.model)
    if cluster.weights_ready(r, m):
        console.print(f"{m.id} is already on {r.drive}.")
        return
    worst = 2 * cluster.PULL_HOURS
    _confirm(yes, f"Download {m.repo} ({m.size_gb:g} GB); at most {worst} CPU core-hours?")
    job = cluster.pull(r, m)
    console.print(f"Job {job.job_id} submitted. Follow it: `oca remote logs {name} --pull -f`.")


@remote_app.command("logs")
def remote_logs(
    name: NameArg = None,
    pull: Annotated[bool, typer.Option("--pull", help="UCloud: the download job's log.")] = False,
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
    lines: Annotated[int, typer.Option("--lines", "-n")] = 40,
) -> None:
    """Show the server's log there."""
    settings = Settings.load()
    name = _remote(settings, name)
    r = settings.remotes[name]
    if r.kind == "ssh":
        command = f"{remotes.oca_command(r)} logs -n {lines}" + (" -f" if follow else "")
        os.execvp("ssh", ["ssh", *(["-t"] if follow else []), r.host, command])
    if r.kind != "ucloud":
        raise OcaError(f"{name}'s log is not oca's")
    job = cluster.read("pull" if pull else "serve")
    if job is None:
        raise OcaError(f"no {'download' if pull else 'server'} job; see `oca remote`")
    if not follow:
        console.print(cluster.log_tail(r, job, lines), markup=False, highlight=False)
        return
    shown = 0
    while True:
        text = cluster.log_text(r, job)
        print(text[shown:], end="", flush=True)
        shown = len(text)
        time.sleep(5)


@remote_app.command("extend")
def remote_extend(hours: Annotated[int, typer.Argument()] = 1) -> None:
    """Add time to your UCloud server's job."""
    job = cluster.read("serve")
    if job is None:
        raise OcaError("no UCloud server job")
    if not job.owner:
        raise OcaError("only the member who started this server can extend it")
    cluster.extend(job, hours)
    console.print(f"Extended job {job.job_id} by {hours} h.")


@app.command("serve")
def serve_cmd(
    model: Annotated[
        str | None, typer.Option(help="Model id (default: the configured one).")
    ] = None,
    idle: Annotated[int, typer.Option(help="Idle minutes before it stops; 0: never.")] = 60,
    status: Annotated[bool, typer.Option("--status", help="Only report.")] = False,
    stop: Annotated[bool, typer.Option("--stop", help="Stop the server you started.")] = False,
) -> None:
    """Serve this machine's model to other machines' oca over SSH; answers in JSON.

    `oca remote up` on the other machine runs it for you.
    """
    answer: dict[str, object]
    try:
        if stop:
            answer = {"stopped": serve.stop()}
        else:
            answer = serve.status() if status else serve.start(model, idle)
    except OcaError as e:
        answer = {"error": str(e)}
    print(json.dumps(answer))


def entrypoint() -> None:
    try:
        app()
    except OcaError as e:
        console.print(f"[red]error:[/] {e}")
        raise SystemExit(1) from None
