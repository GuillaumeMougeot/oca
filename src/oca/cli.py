"""Command-line interface: ``oca`` launches the agent, subcommands manage the stack."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from typer.core import TyperGroup

from . import agent, cluster, install, remotes, router, serve, server, web
from .config import (
    CLUSTER_MODELS,
    DEFAULT_MODEL,
    MODELS,
    NAME,
    ClusterModel,
    OcaError,
    Remote,
    Settings,
    data_dir,
    model_path,
)
from .remotes import Choice


class _Oca(TyperGroup):
    """`oca <server>` opens OpenCode on that server, as `oca -r <server>` does."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        word = args[0] if args else ""
        if word and not word.startswith("-") and word not in self.commands:
            if word not in Settings.load().servers():
                ctx.fail(f"no command or server {word!r}; `oca ls` lists the servers")
            args = ["--remote", *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=_Oca,
    name="oca",
    help="Open Coding Agent. `oca` alone sets up what is missing and opens OpenCode on the "
    "default server; `oca <server>` opens it on another (`oca ls` lists them).",
    add_completion=False,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
remote_app = typer.Typer(
    help="Add, remove and configure servers on other machines: UCloud, one you reach over SSH "
    "(e.g. a DGX Spark), or a URL. Alone, it lists all servers, as `oca ls` does.",
    invoke_without_command=True,
)
app.add_typer(remote_app, name="remote", rich_help_panel="Servers")
web_app = typer.Typer(help="OpenCode in a browser for this folder, e.g. on a phone.")
app.add_typer(web_app, name="web", rich_help_panel="Sessions")
console = Console()
Yes = Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")]
NameArg = Annotated[str | None, typer.Argument(help="The remote (default: the default one).")]
ModelArg = Annotated[str | None, typer.Argument(help="Model id (default: the remote's).")]


def _confirm(yes: bool, question: str, default: bool = True) -> None:
    if not (yes or typer.confirm(question, default=default)):
        raise typer.Exit(1)


def _choose(question: str, answers: list[str], default: str) -> str:
    """Ask until the answer is one of ``answers``, which the question lists.
    (typer bundles its own click, so click's Choice neither shows nor checks them there.)"""
    while (
        got := str(typer.prompt(f"{question} ({'/'.join(answers)})", default=default))
    ) not in answers:
        console.print(f"Type one of: {', '.join(answers)}.")
    return got


def _ready(
    settings: Settings, name: str | None = None, yes: bool = False, pick: bool = True
) -> str:
    """Make server ``name`` (default: the default server) answer or start; returns the server
    to use. When it is down, asks: restart it, use another server, or cancel. Without
    ``pick``, only offers the restart; OpenCode's sessions ask about the rest."""
    if not install.is_installed():
        console.print("[bold]First run: installing llama.cpp and OpenCode[/]")
        install.setup()
    if not Settings.path().exists() and not yes and sys.stdin.isatty():
        _first_run(settings)
    name = name or settings.default
    if name not in settings.servers():
        raise OcaError(f"unknown server {name!r}; see `oca ls`")
    try:
        if name == "local":
            _local(settings, yes)
            doing = None
        else:
            with console.status(f"Connecting to {name}"):
                doing = remotes.check(settings, name)
    except OcaError as e:
        name, doing = _instead(settings, name, f"{name} is not answering: {_why(e)}", yes, pick)
    router.ensure(settings)
    agent.write_config(settings, titles=name == settings.default != "local" and doing is None)
    if doing not in (None, "down"):
        console.print(f"{name}: {doing}. OpenCode's first reply waits for it.")
    return name


def _why(e: Exception) -> str:
    """The first line of an error, which says what happened."""
    return (str(e).strip().splitlines() or [type(e).__name__])[0].rstrip(".")


def _instead(
    settings: Settings, down: str, title: str, yes: bool, pick: bool
) -> tuple[str, str | None]:
    """Ask what to do while ``down`` is down, and do it: restart it, start or use another
    server, or cancel. Returns the server to use and what it is doing (None: it answers)."""
    if yes or not pick:  # restart only
        offered = remotes.choices(settings, down, lambda n: False)
        restart = [c for c in offered if c.label == f"Restart {down}"]
        console.print(f"[yellow]{escape(title)}[/]")
        if restart and (yes or typer.confirm(f"Restart it ({restart[0].description})?")):
            try:
                return down, _start(settings, down)
            except OcaError as e:
                if yes:
                    raise
                console.print(f"[yellow]{down} did not start:[/] {escape(_why(e))}")
        console.print("Its sessions will ask what to do.")
        return down, "down"
    failed: set[str] = set()
    while True:
        options = remotes.choices(settings, down, lambda n: _live(settings, n), failed)
        console.print(f"[yellow]{escape(title)}[/]")
        for i, c in enumerate([*options, Choice("Cancel", "Do not open OpenCode")], 1):
            console.print(f"  {i}. [bold]{escape(c.label)}[/]: {escape(c.description)}")
        n = int(_choose("Choose", [str(i) for i in range(1, len(options) + 2)], "1"))
        if n > len(options):
            raise typer.Exit(1)
        verb, server_name = options[n - 1].label.split()[:2]
        if verb == "Use" and _live(settings, server_name):
            return server_name, None
        try:
            return server_name, _start(settings, server_name)
        except OcaError as e:
            failed.add(server_name)
            title = f"{server_name} did not start: {_why(e)}"


def _start(settings: Settings, name: str) -> str | None:
    """Start server ``name``, already approved; None once it answers, else what it does."""
    if name == "local":
        _local(settings, yes=False)
        return None
    return _up(settings, name, None, yes=True)


def _local(settings: Settings, yes: bool) -> None:
    """Start the local model server, downloading the model first."""
    model = settings.resolved_model
    if not model_path(model).exists():
        where = model_path(model).parent
        _confirm(yes, f"Download {model.id} ({model.size_gb} GB) to {where}?")
        install.pull(model)
    with console.status(f"Starting {model.id} on {settings.resolved_device}"):
        server.start(settings)


def _live(settings: Settings, name: str) -> bool:
    """Whether server ``name`` answers now."""
    if name == "local":
        state = server.read_state()
        return state is not None and server.healthy(state.port)
    found = remotes.backend(settings, name)
    return found is not None and remotes.healthy(found)


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
        typer.Option("--remote", "-r", help="Open this server, not the default; `oca <server>`."),
    ] = None,
    yes: Yes = False,
) -> None:
    if ctx.invoked_subcommand is None:
        settings = Settings.load()
        name = _ready(settings, remote, yes)
        args = ["--continue"] if resume else []
        args += ["--session", session] if session else []
        args += (
            ["--model", f"{agent.PROVIDER}/{name}"] if remote or name != settings.default else []
        )
        agent.launch(args)


@app.command(
    rich_help_panel="Sessions",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(ctx: typer.Context, yes: Yes = False) -> None:
    """Like `oca`, passing extra arguments to OpenCode (e.g. `oca run -- run "fix the tests"`)."""
    settings = Settings.load()
    name, args = _ready(settings, None, yes), ctx.args
    if name != settings.default and args[:1] in ([], ["run"]):
        args = [*args, "--model", f"{agent.PROVIDER}/{name}"]
    agent.launch(args)


@app.command(rich_help_panel="Sessions")
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


ServerArg = Annotated[str | None, typer.Argument(help="A server (default: this machine's).")]


@app.command("ls", rich_help_panel="Servers")
def ls() -> None:
    """List the servers: where each runs, its model and state; * marks the default."""
    _list(Settings.load())


def _list(settings: Settings) -> None:
    names = settings.servers()
    with ThreadPoolExecutor(len(names)) as pool:  # each may ask its machine
        states = dict(zip(names, pool.map(lambda n: _state(settings, n), names), strict=True))
    table = Table("", "server", "where", "model", "context", "state", box=None)
    for name in names:
        r = settings.remotes.get(name)
        where = remotes.where(settings, name) + (" (all users)" if r and r.shared else "")
        model = r.model if r else settings.resolved_model.id
        size = f"{remotes.context(settings, name) // 1000}k"
        mark = "*" if name == settings.default else ""
        table.add_row(mark, name, where, model, size, states[name])
    console.print(table)
    console.print(
        "* is the default, which plain `oca` opens. `oca <server>` opens another; "
        "`oca start <server>` only starts it."
    )


@app.command(rich_help_panel="Servers")
def start(
    name: ServerArg = None,
    model: ModelArg = None,
    wait: Annotated[bool, typer.Option("--wait", "-w", help="Wait until it answers.")] = False,
    yes: Yes = False,
) -> None:
    """Start a server without opening OpenCode, or join one someone else runs.

    Alone, it starts this machine's, and restarts it if settings changed.
    """
    settings = Settings.load()
    if name in (None, "local"):
        if model:
            settings.set("model", model)
            settings.save()
        with console.status("Starting llama-server"):
            server.start(settings)
        router.ensure(settings)
        agent.write_config(settings)
        console.print(
            f"Serving at http://127.0.0.1:{settings.port}/v1, one model per server "
            f"({', '.join(settings.servers())}); {settings.default} by default."
        )
        return
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
    how = "plain `oca`" if settings.default == name else f"`oca {name}`"
    if doing is None:
        console.print(f"{name} answers; {how} opens it.")
    else:
        console.print(f"{name}: {doing}. {how} waits for it; `oca ls` shows its state.")


@app.command(rich_help_panel="Servers")
def stop(name: ServerArg = None) -> None:
    """Stop a server you started, or leave one you joined.

    Alone, it stops this machine's and oca's endpoint.
    """
    if name in (None, "local"):
        here = [server.stop(), *([router.stop()] if name is None else [])]
        console.print("Stopped." if any(here) else "Not running.")
        return
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


@app.command(rich_help_panel="Setup")
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
    _ready(settings, None, yes, pick=False)
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


@app.command(rich_help_panel="Setup")
def setup(force: Annotated[bool, typer.Option(help="Reinstall binaries.")] = False) -> None:
    """Install pinned llama.cpp and OpenCode binaries; write the OpenCode config."""
    install.setup(force=force)
    settings = Settings.load()
    if not Settings.path().exists():
        settings.save()
    agent.write_config(settings)
    console.print(f"Installed into {data_dir() / 'bin'}. Next: `oca` (downloads the model).")


@app.command(rich_help_panel="Setup")
def pull(
    model: Annotated[str | None, typer.Argument(help="Model id, see `oca models`.")] = None,
) -> None:
    """Download local model weights (default: the configured model)."""
    settings = Settings.load()
    if model and model not in MODELS:
        raise OcaError(f"unknown model {model!r}; see `oca models`")
    console.print(f"Saved {install.pull(MODELS[model] if model else settings.resolved_model)}")


@app.command(rich_help_panel="Setup")
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


@app.command(rich_help_panel="Servers")
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


@app.command(rich_help_panel="Servers")
def logs(
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
    lines: Annotated[int, typer.Option("--lines", "-n")] = 40,
) -> None:
    """Show the local llama-server log."""
    os.execvp(
        "tail", ["tail", "-n", str(lines), *(["-F"] if follow else []), str(server.log_path())]
    )


@app.command(rich_help_panel="Setup")
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
    rich_help_panel="Setup",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def ucloud(ctx: typer.Context) -> None:
    """The bundled ucloud-api CLI, e.g. `oca ucloud login` or `oca ucloud jobs list`."""
    from ucloud_api.cli import app as ucloud_app

    ucloud_app(args=ctx.args, prog_name="oca ucloud")


@app.command(rich_help_panel="Setup")
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
            raise OcaError("which remote? See `oca ls`; `oca remote add` adds one")
    if name not in settings.remotes:
        raise OcaError(f"unknown remote {name!r}; see `oca ls`")
    return name


def _cluster_model(mid: str) -> ClusterModel:
    if mid not in CLUSTER_MODELS:
        raise OcaError(f"unknown UCloud model {mid!r}; see `oca models`")
    return CLUSTER_MODELS[mid]


def _state(settings: Settings, name: str) -> str:
    """What server ``name`` is doing; an SSH machine is asked, since it may have stopped it."""
    if name == "local":
        st = server.read_state()
        return "stopped" if st is None else "ready" if server.healthy(st.port) else "loading"
    r, found = settings.remotes[name], remotes.backend(settings, name)
    if found is not None and remotes.healthy(found):
        return "ready"
    if r.kind == "url":
        return "not answering"
    if r.kind == "ssh":
        try:
            told = remotes.serve_call(r, "--status")
        except OcaError:
            return "not answering"
        return str(told.get("state")) + (f" ({told['why']})" if told.get("why") else "")
    job = cluster.read("serve")
    if job is None:
        return "down"
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
    """List the servers, as `oca ls` does."""
    if ctx.invoked_subcommand is None:
        _list(Settings.load())


def _add(settings: Settings, name: str, target: str, model: str | None, yes: bool) -> None:
    if name in settings.servers():
        raise OcaError(f"{name} exists; `oca remote remove {name}` first")
    if name in _commands():
        raise OcaError(f"{name} is an oca command, so `oca {name}` could not open it: pick another")
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
        + ("Plain `oca` opens it; " if settings.default == name else f"`oca {name}` opens it; ")
        + ("it answers already." if remote.kind == "url" else f"`oca start {name}` starts it.")
    )


def _commands() -> set[str]:
    """oca's own words, which server names must not take."""
    named = (c.name or (c.callback.__name__ if c.callback else "") for c in app.registered_commands)
    return {n.replace("_", "-") for n in named} | {g.name or "" for g in app.registered_groups}


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
    gpu = f"{found.gpu} with {found.memory:.0f} GB" if found.gpu else "no NVIDIA GPU"
    console.print(f"{host}: {found.arch}, {gpu}.")
    model = model or _ssh_model(host, found.memory, yes)
    shared = ""
    if found.shared:  # someone installed it for everyone: join that
        shared = remotes.SHARED_ROOT
        console.print(f"{host} has oca for all its users in {shared}; using it.")
    else:
        who = "me"
        if not yes:
            who = _choose(
                f"Install oca on {host} for you only, or for all its users?", ["me", "all"], "me"
            )
        if who == "all":
            shared = typer.prompt("Folder for it", default=remotes.SHARED_ROOT)
            if remotes.probe(host, shared).shared:
                console.print(f"{host} has oca for all its users in {shared}; using it.")
                return Remote("ssh", model=model, port=port, host=host, shared=shared)
        if shared or not found.personal:
            with console.status(f"Installing oca on {host}; this takes a few minutes"):
                remotes.install(host, shared)
    return Remote("ssh", model=model, port=port, host=host, shared=shared)


def _ssh_model(host: str, memory: float, yes: bool) -> str:
    """Ask which model it serves, offering the biggest that fits its GPU first."""
    fits = [m for m in MODELS.values() if m.size_gb * 1.2 <= memory]  # cache and buffers too
    if not fits:
        return DEFAULT_MODEL["cpu"]
    fits.sort(key=lambda m: -m.size_gb)
    if yes:
        return fits[0].id
    for m in fits:
        console.print(f"  [bold]{m.id}[/]: {m.size_gb:g} GB; {escape(m.note)}")
    return _choose(f"Model for {host}", [m.id for m in fits], fits[0].id)


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
        raise OcaError(f"unknown server {name!r}; see `oca ls`")
    settings.default = name
    settings.save()
    agent.write_config(settings)
    console.print(f"Plain `oca` uses {name}.")


SETTABLE = ("idle", "hours", "model", "port", "context", "key")


@remote_app.command("set")
def remote_set(
    name: Annotated[str, typer.Argument()],
    key: Annotated[str, typer.Argument(help=", ".join(SETTABLE))],
    value: Annotated[str, typer.Argument()],
) -> None:
    """Change a remote's setting, e.g. `oca remote set ucloud idle 30`."""
    settings = Settings.load()
    name = _remote(settings, name)
    r = settings.remotes[name]
    if key not in SETTABLE:
        raise OcaError(f"unknown key {key!r}; keys: {', '.join(SETTABLE)}")
    current = getattr(r, key)
    try:
        setattr(r, key, type(current)(value))
    except ValueError as e:
        raise OcaError(f"{key} expects {type(current).__name__}, got {value!r}") from e
    settings.validate()
    settings.save()
    agent.write_config(settings)
    console.print(f"{name} {key} = {getattr(r, key)!r}. Applies to the next server you start.")


def _up(settings: Settings, name: str, model: str | None, yes: bool) -> str | None:
    """Start remote ``name``'s server, or join a running one; None once it answers."""
    r = settings.remotes[name]
    if model and model != r.model:
        r.model = model
        settings.validate()
        settings.save()
    if r.kind == "ucloud":
        doing = remotes.start(settings, name, lambda q: _confirm(yes, q), console.print)
        router.ensure(settings)  # its background check opens the tunnel once the job runs
        return doing
    with console.status(f"Starting {r.model} on {r.host}" if r.kind == "ssh" else "Connecting"):
        doing = remotes.start(settings, name)
    conn = remotes.read_conn(name)
    if conn and conn.context != r.context:  # what OpenCode should plan for next time
        r.context = conn.context
        settings.save()
    return doing


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


@app.command("serve", rich_help_panel="Servers")
def serve_cmd(
    model: Annotated[
        str | None, typer.Option(help="Model id (default: the configured one).")
    ] = None,
    idle: Annotated[int, typer.Option(help="Idle minutes before it stops; 0: never.")] = 60,
    status: Annotated[bool, typer.Option("--status", help="Only report.")] = False,
    stop: Annotated[bool, typer.Option("--stop", help="Stop the server you started.")] = False,
) -> None:
    """Serve this machine's model to other machines' oca over SSH; answers in JSON.

    `oca start <server>` on the other machine runs it for you.
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
