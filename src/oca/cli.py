"""Command-line interface: ``oca`` opens the agent on a model, subcommands manage the rest.

A server (this machine, UCloud, a machine over SSH, a URL) serves one model at a time. A
pair is a model on a server, such as flash-next@spark2: `oca ls` lists them, `oca <id>`
opens one, and plain `oca` the default.
"""

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
    OcaError,
    Remote,
    Settings,
    data_dir,
    model_path,
)
from .remotes import Choice, Pair

VALUED = ("-r", "--resume", "-m", "--model")  # options of plain `oca` that take a value


def _complete_ids(incomplete: str) -> list[str]:
    """Tab completion: the ids `oca ls` lists that start with what is typed."""
    with suppress(Exception):  # a broken setting must not break the shell
        return [p.id for p in remotes.pairs(Settings.load()) if p.id.startswith(incomplete)]
    return []


def _complete_servers(incomplete: str) -> list[str]:
    with suppress(Exception):
        return [n for n in Settings.load().remotes if n.startswith(incomplete)]
    return []


class _Oca(TyperGroup):
    """`oca <id>` opens that pair, as `oca -m <id>` does; `-r` alone picks a session."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        i = 0
        while i < len(args) and args[i].startswith("-"):
            after = args[i + 1] if i + 1 < len(args) else "-"
            if args[i] in ("-r", "--resume") and after.startswith("-"):
                args = [*args[: i + 1], "", *args[i + 1 :]]  # no session given: pick one
            i += 2 if args[i] in VALUED else 1
        if i < len(args) and args[i] not in self.commands:
            args = [*args[:i], "--model", *args[i:]]
        return super().parse_args(ctx, args)

    def shell_complete(self, ctx: Any, incomplete: str) -> list[Any]:
        """The first word completes to a command or an id."""
        from typer._click.shell_completion import CompletionItem  # typer bundles its click

        found = super().shell_complete(ctx, incomplete)
        if not incomplete.startswith("-"):
            found += [CompletionItem(i) for i in _complete_ids(incomplete)]
        return found

    def get_params(self, ctx: Any) -> list[Any]:
        """Without typer's --install-completion and --show-completion: oca installs it."""
        tabs = ("install_completion", "show_completion")
        return [p for p in super().get_params(ctx) if p.name not in tabs]


def _completion() -> None:
    """Install tab completion for the user's shell, or refresh it; oca does it, not the user.
    Bash and fish load it without a line in their rc file; zsh gets typer's."""
    from typer._completion_shared import get_completion_script, install_zsh

    shell = Path(os.environ.get("SHELL", "")).name
    path = install.completion_paths().get(shell)
    if path is None or os.environ.get("OCA_SHARED") or not sys.stdin.isatty():
        return
    script = get_completion_script(prog_name="oca", complete_var="_OCA_COMPLETE", shell=shell)
    with suppress(OSError):
        if path.is_file() and path.read_text() == script:
            return
        new = not path.exists()
        if shell == "zsh":
            install_zsh(prog_name="oca", complete_var="_OCA_COMPLETE", shell=shell)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script)
        if new:
            Console(stderr=True).print(f"Tab completion is on for {shell}, in new terminals.")


app = typer.Typer(
    cls=_Oca,
    name="oca",
    help="Open Coding Agent. `oca` opens OpenCode on the default model, `oca <id>` on another: "
    "`oca ls` lists them, each a model on a server, such as flash-next@spark2.",
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
server_app = typer.Typer(
    help="Add, remove or configure a server: UCloud, a machine you reach over SSH (e.g. a "
    "DGX Spark), or a URL.",
    no_args_is_help=True,
)
app.add_typer(server_app, name="server", rich_help_panel="Servers")
sessions_app = typer.Typer(help="List this folder's sessions, or delete them.")
app.add_typer(sessions_app, name="sessions", rich_help_panel="Sessions")
web_app = typer.Typer(help="OpenCode in a browser for this folder, e.g. on a phone.")
app.add_typer(web_app, name="web", rich_help_panel="Sessions")
console = Console()
Yes = Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")]
Target = Annotated[
    str | None,
    typer.Argument(
        help="An id from `oca ls`, or part of one (default: the default).",
        autocompletion=_complete_ids,
    ),
]
ModelOpt = Annotated[
    str | None,
    typer.Option(
        "--model", "-m", help="An id from `oca ls`, or part of one.", autocompletion=_complete_ids
    ),
]
ServerName = Annotated[str, typer.Argument(help="Its name.", autocompletion=_complete_servers)]


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


def _pick(settings: Settings, word: str) -> Pair:
    """The pair ``word`` names; a numbered menu when it names several."""
    found = remotes.find(settings, word)
    if not found:
        raise OcaError(
            f"nothing is called {word!r}; `oca ls` lists the models, `oca -h` the commands"
        )
    if len(found) == 1:
        return found[0]
    if not sys.stdin.isatty():
        raise OcaError(f"{word!r} could be {', '.join(p.id for p in found)}; type more of one")
    console.print(f"{word} could be:")
    for i, p in enumerate(found, 1):
        where = remotes.where(settings, p.server)
        console.print(f"  {i}. [bold]{p.id}[/]: {p.model} on {escape(where)}")
    return found[int(_choose("Choose", [str(i) for i in range(1, len(found) + 1)], "1")) - 1]


def _server(settings: Settings, word: str | None) -> str:
    """The server ``word`` names, by its name or a pair's; default: the default's."""
    if word is None:
        return settings.default_server
    return word if word in settings.servers() else _pick(settings, word).server


def _use(settings: Settings, pair: Pair, yes: bool) -> Pair:
    """Set ``pair``'s server to serve its model, and return the pair to open. A machine over
    SSH serving another model restarts with this one, after asking; one that someone else
    started keeps its model, which oca offers instead. UCloud asks in `remotes.start`."""
    name = pair.server
    r = settings.remotes.get(name)
    if r is not None and r.kind == "ssh":
        with suppress(OcaError):  # unreachable: `remotes.check` says so next
            told = remotes.serve_call(r, "--status")
            now = str(told.get("model", "")) if told.get("state") != "stopped" else ""
            if now and now != pair.model and not told.get("mine"):
                other = remotes.pair_id(settings, name, now)
                console.print(f"{name} serves {now} for someone else; only they can switch it.")
                _confirm(yes, f"Open {other} instead?")
                pair = Pair(other, name, now, int(told.get("context") or 0))
            elif now and now != pair.model:
                _confirm(yes, f"{name} serves {now}. Restart it with {pair.model}?")
    if settings.model_of(name) != pair.model:
        if r is None:
            settings.set("model", pair.model)
        else:
            r.model = pair.model
        settings.save()
    return pair


def _ready(
    settings: Settings, pair: Pair | None = None, yes: bool = False, pick: bool = True
) -> str:
    """Make ``pair`` (default: the default) answer or start; returns the server to use. When
    its server is down, asks: restart it, use another server, or cancel. Without ``pick``,
    only offers the restart; OpenCode's sessions ask about the rest."""
    if not install.is_installed():
        console.print("[bold]First run: installing llama.cpp and OpenCode[/]")
        install.setup()
    if not Settings.path().exists() and not yes and sys.stdin.isatty():
        _first_run(settings)
    name = _use(settings, pair or remotes.default_pair(settings), yes).server
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
    titles = name == settings.default_server != "local" and doing is None
    agent.write_config(settings, titles=titles)
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
    """Ask once which servers to use besides this machine; `oca server add` adds more."""
    console.print("[bold]Welcome to oca.[/] It runs a model on this machine and can use others.")
    if typer.confirm("Use UCloud B200 GPUs (needs a UCloud project)?", default=False):
        try:
            _add(settings, "ucloud", "ucloud", None, yes=False)
        except OcaError as e:
            console.print(f"[yellow]Skipped UCloud:[/] {e}\nLater: `oca server add ucloud ucloud`.")
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
            console.print(f"[yellow]Skipped {host}:[/] {e}\nLater: `oca server add {name} {host}`.")
    settings.save()


def _version(shown: bool) -> None:
    if shown:
        console.print(f"oca {version('oca')}")
        raise typer.Exit


@app.callback()
def main(
    ctx: typer.Context,
    last: Annotated[bool, typer.Option("--continue", "-c", help="Resume the last session.")] = (
        False
    ),
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume", "-r", help="Resume a session: give the start of its id, or pick one."
        ),
    ] = None,
    model: ModelOpt = None,
    yes: Yes = False,
    show_version: Annotated[
        bool,
        typer.Option("--version", "-V", callback=_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    _completion()
    if ctx.invoked_subcommand is None:
        settings = Settings.load()
        session = None if resume is None else _session(resume)  # before any server starts
        name = _ready(settings, _pick(settings, model) if model else None, yes)
        args = ["--continue"] if last else []
        args += ["--session", session] if session else []
        args += ["--model", f"{agent.PROVIDER}/{name}"] if name != settings.default_server else []
        agent.launch(args)


# -- sessions --------------------------------------------------------------------------------


@app.command(
    rich_help_panel="Sessions",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(ctx: typer.Context, model: ModelOpt = None, yes: Yes = False) -> None:
    """Run one task unattended: `oca run "fix the failing test"`.

    Other options go to `opencode run`, such as `--variant low`.
    """
    if not ctx.args:
        raise OcaError('say what to do: `oca run "fix the failing test"`')
    settings = Settings.load()
    name = _ready(settings, _pick(settings, model) if model else None, yes)
    agent.launch(["run", *ctx.args, "--model", f"{agent.PROVIDER}/{name}"])


@sessions_app.callback(invoke_without_command=True)
def sessions_list(ctx: typer.Context) -> None:
    """List this folder's sessions, the latest first; `oca -r <id>` resumes one."""
    if ctx.invoked_subcommand is not None:
        return
    found = agent.sessions(Path.cwd())
    if not found:
        console.print("No sessions started in this folder.")
        return
    short = _short([s["id"] for s in found])
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")  # whole titles: they wrap
    for s in found:
        when = time.strftime("%m-%d %H:%M", time.localtime(int(s.get("updated") or 0) / 1000))
        folder = os.path.relpath(s.get("directory") or ".", Path.cwd())
        title = s.get("title") or "(untitled)"
        table.add_row(short[s["id"]], when, title + ("" if folder == "." else f" ({folder})"))
    console.print(table, highlight=False)
    console.print(
        "`oca -r <id>` resumes one (the start of its id will do; `oca -r` alone picks); "
        "`oca sessions rm <id>` deletes one."
    )


def _short(ids: list[str]) -> dict[str, str]:
    """Each id cut to the length at which all are still told apart, as git shortens hashes."""
    size = 12
    while len({i[:size] for i in ids}) < len(ids):
        size += 1
    return {i: i[:size] for i in ids}


def _session(word: str) -> str:
    """The id of this folder's session that ``word`` starts; a numbered menu when it is ""."""
    found = agent.sessions(Path.cwd())
    if not word:
        if not found:
            raise OcaError("no session started in this folder")
        shown = found[:9]
        for i, s in enumerate(shown, 1):
            console.print(f"  {i}. {escape(s.get('title') or '(untitled)')}")
        return str(
            shown[int(_choose("Resume", [str(i) for i in range(1, len(shown) + 1)], "1")) - 1]["id"]
        )
    matches = [s["id"] for s in found if s["id"].startswith(word)]
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        raise OcaError(f"{word!r} starts {len(matches)} sessions; type more of its id")
    return word  # a whole id, maybe from another folder: OpenCode says if it does not exist


@sessions_app.command("rm")
def sessions_rm(
    session: Annotated[str, typer.Argument(help="Its id, or its start; `oca sessions`.")],
) -> None:
    """Delete a session."""
    agent.launch(["session", "delete", _session(session)])


@sessions_app.command("clear")
def sessions_clear(yes: Yes = False) -> None:
    """Delete the sessions started in this folder or below."""
    ids = [s["id"] for s in agent.sessions(Path.cwd())]
    if not ids:
        console.print("No sessions started in this folder.")
        return
    _confirm(yes, f"Delete {len(ids)} sessions started in {Path.cwd()} or below?", default=False)
    for sid in ids:
        agent.opencode(["session", "delete", sid])
    console.print(f"Deleted {len(ids)} sessions.")


@web_app.callback(invoke_without_command=True)
def web_ui(
    ctx: typer.Context,
    port: Annotated[int, typer.Option(help="Port on this machine.")] = 4096,
    vpn: Annotated[
        bool,
        typer.Option("--vpn", help="Listen on this machine's WireGuard address, not on localhost."),
    ] = False,
    yes: Yes = False,
) -> None:
    """OpenCode in a browser for this folder, e.g. on a phone; runs in the background."""
    if ctx.invoked_subcommand is not None:
        return
    host = web.vpn_address() if vpn else "127.0.0.1"
    settings = Settings.load()
    _ready(settings, None, yes, pick=False)
    state = web.start(port, Path.cwd(), host)
    console.print(
        f"OpenCode for {state['project']} at http://{host}:{port}, user opencode, "
        f"password {agent.web_password()}. It keeps running; `oca web stop` ends it."
    )
    if vpn:
        console.print("Open that address on a phone or computer that joined the VPN.")
    else:
        console.print(
            f"From a phone: `oca web --vpn` if it joined this machine's WireGuard VPN, else "
            f"`ssh -N -L {port}:127.0.0.1:{port} <you>@<this machine>` and that address."
        )


@web_app.command("stop")
def web_stop() -> None:
    """Stop the web server."""
    console.print("Stopped." if web.stop() else "Not running.")


# -- servers ---------------------------------------------------------------------------------


@app.command("ls", rich_help_panel="Servers")
def ls() -> None:
    """List every model on every server, with the id that opens it; * marks the default."""
    settings = Settings.load()
    names = settings.servers()
    with ThreadPoolExecutor(len(names)) as pool:  # each may ask its machine
        states = dict(zip(names, pool.map(lambda n: _state(settings, n), names), strict=True))
    default, every = remotes.default_pair(settings).id, remotes.pairs(settings)
    models = {p.id: _model(settings, p) for p in every}
    wide = max(map(len, models)), max(map(len, models.values()))
    for name in names:  # a line per server, then one per model: fits 80 columns
        console.print(f"[bold]{escape(name)}[/]: {escape(_where(settings, name))}")
        for p in (p for p in every if p.server == name):
            mark = "*" if p.id == default else " "
            state = _row(states[name], p.model, settings.model_of(name))
            row = f" {mark} {p.id:<{wide[0]}}  {models[p.id]:<{wide[1]}}  {p.context // 1000:>3}k"
            console.print(escape(f"{row}  {state}".rstrip()), highlight=False)


def _row(now: tuple[str, str | None, set[str] | None], model: str, chosen: str) -> str:
    """A model's line in `oca ls`: the server's state if it serves the model, or would start
    with it; else whether the model is downloaded there (blank if unknown)."""
    state, serving, have = now
    if model == serving:
        return state
    if have is not None and model not in have:
        return "not downloaded"
    if serving is None and model == chosen:
        return state
    return "" if have is None else "downloaded"
    console.print(
        "`oca <id>` opens one, and part of an id will do; plain `oca` opens the default, *. "
        "A server serves one model at a time."
    )


def _where(settings: Settings, name: str) -> str:
    r = settings.remotes.get(name)
    if r is not None and r.kind == "ssh":
        return f"{r.host} over SSH, " + ("for all its users" if r.shared else "for you")
    return remotes.where(settings, name)


def _model(settings: Settings, p: Pair) -> str:
    r = settings.remotes.get(p.server)
    if r is not None and r.kind == "ucloud":
        gpus = CLUSTER_MODELS[p.model].gpus
        return f"{p.model}, {gpus} B200" + ("s" if gpus > 1 else "")
    return p.model


def _state(settings: Settings, name: str) -> tuple[str, str | None, set[str] | None]:
    """What server ``name`` is doing, the model it serves if any, and the models it has
    downloaded (None if unknown). Its machine is asked: it may have stopped, or a project
    member may have started it."""
    if name == "local":
        have = {m.id for m in MODELS.values() if model_path(m).exists()}
        st = server.read_state()
        if st is None:
            return "stopped", None, have
        path = st.cmd[st.cmd.index("--model") + 1] if "--model" in st.cmd else ""
        now = next((m.id for m in MODELS.values() if str(model_path(m)) == path), None)
        return ("ready" if server.healthy(st.port) else "loading"), now, have
    r, found = settings.remotes[name], remotes.backend(settings, name)
    if r.kind == "url":
        return ("ready" if found and remotes.healthy(found) else "not answering"), r.model, None
    if r.kind == "ssh":
        try:
            told = remotes.serve_call(r, "--status")
        except OcaError:
            return "not answering", None, None
        there = set(told["downloaded"]) if "downloaded" in told else None  # older oca: unknown
        if told.get("state") == "stopped":
            return "stopped" + (f" ({told['why']})" if told.get("why") else ""), None, there
        state = str(told.get("state"))
        if not told.get("mine") and told.get("started_by"):
            state += f", started by {told['started_by']}"
        return state, told.get("model"), there
    on_drive: set[str] | None = None
    with suppress(OcaError, OSError):
        on_drive = cluster.downloaded(r)
    job = cluster.read("serve")
    if job is None:  # nothing joined from here; a project member may have started one
        running: dict[str, str] = {}
        with suppress(OcaError, OSError):
            running = cluster.listed(r)
        if not running:
            return "down", None, on_drive
        model = r.model if r.model in running else next(iter(running))
        return f"started by {running[model]}", model, on_drive
    if found is not None and remotes.healthy(found):
        return "ready", job.model, on_drive
    try:
        status = cluster.job_status(job)
    except OcaError:
        return "starting", job.model, on_drive
    state = str(status.get("state", "starting")).lower().replace("_", " ")
    expires = status.get("expiresAt")
    if isinstance(expires, int | float):
        state += f", {(expires / 1000 - time.time()) / 3600:.1f} h left"
    return state, job.model, on_drive


@app.command(rich_help_panel="Servers")
def stop(
    target: Target = None,
    everything: Annotated[
        bool, typer.Option("--all", help="Every server you started, and oca's endpoint.")
    ] = False,
) -> None:
    """Stop the server you started, or leave one you joined."""
    settings = Settings.load()
    if not everything:
        _stop(settings, _server(settings, target))
        return
    for name in settings.servers():
        if name == "local" or settings.remotes[name].kind != "url":
            try:
                _stop(settings, name)
            except OcaError as e:
                console.print(f"{name}: {e}")
    console.print("oca's endpoint: " + ("stopped." if router.stop() else "was not running."))


def _stop(settings: Settings, name: str) -> None:
    if name == "local":
        console.print("local: " + ("stopped." if server.stop() else "was not running."))
        return
    r = settings.remotes[name]
    if r.kind == "url":
        raise OcaError(f"{name} is not oca's to stop; `oca server rm {name}` forgets it")
    if r.kind == "ucloud":
        job = cluster.read("serve")
        cluster.terminate(r, "serve")
        stopped = None if job is None else job.owner
    else:
        stopped = remotes.down(settings, name)
    agent.write_config(settings)
    said = {None: "was not running.", True: "stopped.", False: "left; it serves others still."}
    console.print(f"{name}: {said[stopped]}")


@app.command(rich_help_panel="Servers")
def logs(
    target: Target = None,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Keep showing new lines.")] = False,
    lines: Annotated[int, typer.Option("--lines", "-n", help="How many lines.")] = 40,
    pull: Annotated[bool, typer.Option("--pull", help="UCloud: the download job's log.")] = False,
) -> None:
    """Show a server's log."""
    settings = Settings.load()
    name = _server(settings, target)
    if name == "local":
        log = str(server.log_path())
        os.execvp("tail", ["tail", "-n", str(lines), *(["-F"] if follow else []), log])
    r = settings.remotes[name]
    if r.kind == "ssh":  # its own `oca logs` shows its llama-server's log
        command = f"{remotes.oca_command(r)} logs -n {lines}" + (" -f" if follow else "")
        os.execvp("ssh", ["ssh", *(["-t"] if follow else []), r.host, command])
    if r.kind != "ucloud":
        raise OcaError(f"{name}'s log is not oca's")
    job = cluster.read("pull" if pull else "serve")
    if job is None:
        raise OcaError(f"no {'download' if pull else 'server'} job; see `oca ls`")
    if not follow:
        console.print(cluster.log_tail(r, job, lines), markup=False, highlight=False)
        return
    shown = 0
    while True:
        text = cluster.log_text(r, job)
        print(text[shown:], end="", flush=True)
        shown = len(text)
        time.sleep(5)


@app.command(rich_help_panel="Servers")
def pull(target: Target = None, yes: Yes = False) -> None:
    """Download a model where it runs, ahead of time; on UCloud, from a CPU job."""
    settings = Settings.load()
    if target in MODELS:  # a catalog id, as `oca pull` over SSH gives it: this machine's
        console.print(f"Saved {install.pull(MODELS[target])}")
        return
    pair = _pick(settings, target) if target else remotes.default_pair(settings)
    r = settings.remotes.get(pair.server)
    if r is None:
        console.print(f"Saved {install.pull(MODELS[pair.model])}")
        return
    if r.kind == "ssh":
        command = f"{remotes.oca_command(r)} pull {shlex.quote(pair.model)}"
        os.execvp("ssh", ["ssh", r.host, command])
    if r.kind != "ucloud":
        raise OcaError(f"{pair.server} is not oca's to download to")
    m = CLUSTER_MODELS[pair.model]
    if cluster.weights_ready(r, m):
        console.print(f"{m.id} is already on {r.drive}.")
        return
    worst = 2 * cluster.PULL_HOURS
    _confirm(yes, f"Download {m.repo} ({m.size_gb:g} GB); at most {worst} CPU core-hours?")
    job = cluster.pull(r, m)
    console.print(f"Job {job.job_id} submitted. Follow it: `oca logs {pair.id} --pull -f`.")


@server_app.command("add")
def server_add(
    name: Annotated[str, typer.Argument(help="A short name, e.g. spark.")],
    target: Annotated[
        str,
        typer.Argument(
            help="`ucloud`; user@host of a machine you reach over SSH; or the URL of an "
            "OpenAI-compatible server."
        ),
    ],
    model: Annotated[str | None, typer.Option(help="Its model's id; `oca ls` lists them.")] = None,
    yes: Yes = False,
) -> None:
    """Add a server: UCloud, a machine you reach over SSH (e.g. a DGX Spark), or a URL."""
    _add(Settings.load(), name, target, model, yes)


@server_app.command("rm")
def server_rm(name: ServerName) -> None:
    """Forget a server; stops the one you started there."""
    settings = Settings.load()
    r = settings.remotes[_remote(settings, name)]
    with suppress(OcaError):
        if r.kind == "ucloud":
            cluster.terminate(r, "serve")
        elif r.kind == "ssh":
            remotes.down(settings, name)
    del settings.remotes[name]
    if settings.default_server == name:
        settings.default = "local"
    settings.save()
    agent.write_config(settings)
    console.print(f"Removed {name}. Plain `oca` opens {remotes.default_pair(settings).id}.")


SETTABLE = ("idle", "hours", "model", "port", "context", "key")


@server_app.command("set")
def server_set(
    name: ServerName,
    key: Annotated[str, typer.Argument(help=", ".join(SETTABLE))],
    value: Annotated[str, typer.Argument(help="Its new value.")],
) -> None:
    """Change a server's setting, e.g. `oca server set ucloud idle 30`."""
    settings = Settings.load()
    r = settings.remotes[_remote(settings, name)]
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


@server_app.command("update")
def server_update(name: ServerName) -> None:
    """Reinstall oca on a machine over SSH, for you or for all its users, at the version this
    oca installs (`OCA_SOURCE`, as for install.sh, picks another)."""
    settings = Settings.load()
    r = settings.remotes[_remote(settings, name)]
    if r.kind != "ssh":
        raise OcaError(f"{name} is not a machine oca installs itself on")
    with console.status(f"Updating oca on {r.host}; this takes a few minutes"):
        remotes.install(r.host, r.shared)
    console.print(f"Updated oca on {r.host}. A running server keeps its version until it stops.")


@server_app.command("extend")
def server_extend(hours: Annotated[int, typer.Argument(help="Hours to add.")] = 1) -> None:
    """Add time to your UCloud server's job."""
    job = cluster.read("serve")
    if job is None:
        raise OcaError("no UCloud server job")
    if not job.owner:
        raise OcaError("only the member who started this server can extend it")
    cluster.extend(job, hours)
    console.print(f"Extended job {job.job_id} by {hours} h.")


def _remote(settings: Settings, name: str) -> str:
    if name == "local":
        raise OcaError("local is this machine: `oca config` holds its settings")
    if name not in settings.remotes:
        raise OcaError(f"no server {name!r}; `oca ls` lists them")
    return name


def _add(settings: Settings, name: str, target: str, model: str | None, yes: bool) -> None:
    if name in settings.servers():
        raise OcaError(f"{name} exists; `oca server rm {name}` first")
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
    pair = remotes.pair_id(settings, name)
    first = settings.default_server == "local"
    if (yes and first) or (not yes and typer.confirm(f"Make {pair} the default?", default=first)):
        settings.default = pair
    settings.save()
    agent.write_config(settings)
    how = "Plain `oca`" if settings.default == pair else f"`oca {pair}`"
    console.print(f"Added {name}; `oca ls` lists its models. {how} opens {pair}.")


def _new_ucloud(settings: Settings, model: str | None, port: int) -> Remote:
    if any(r.kind == "ucloud" for r in settings.remotes.values()):
        raise OcaError("oca drives one UCloud server; `oca ls` lists it")
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
    memory = round(found.memory, 1)
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
                return Remote("ssh", model, port, host=host, shared=shared, memory=memory)
        if shared or not found.personal:
            with console.status(f"Installing oca on {host}; this takes a few minutes"):
                remotes.install(host, shared)
    return Remote("ssh", model, port, host=host, shared=shared, memory=memory)


def _ssh_model(host: str, memory: float, yes: bool) -> str:
    """Ask which model it serves first, the biggest that fits its GPU on top."""
    fits = [m for m in MODELS.values() if m.size_gb * 1.2 <= memory]  # cache and buffers too
    if not fits:
        return DEFAULT_MODEL["cpu"]
    fits.sort(key=lambda m: -m.size_gb)
    if yes:
        return fits[0].id
    console.print(f"Models that fit {host}:")
    for i, m in enumerate(fits, 1):
        console.print(f"  {i}. [bold]{m.id}[/]: {m.size_gb:g} GB; {escape(m.note)}")
    return fits[int(_choose("Choose", [str(i) for i in range(1, len(fits) + 1)], "1")) - 1].id


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


@app.command(rich_help_panel="Servers")
def status() -> None:
    """Show this machine's install, endpoints and model server."""
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
    where = f"http://{ui.get('host', '127.0.0.1')}:{ui['port']}, {ui['project']}" if ui else ""
    table.add_row("web", where or "stopped")
    if state:
        health = "ready" if server.healthy(state.port) else "loading"
        table.add_row("server", f"pid {state.pid}, port {state.port}, {health}")
    else:
        table.add_row("server", "stopped")
    default = remotes.default_pair(settings).id
    table.add_row("default", f"{default} (`oca config default <id>` changes it)")
    console.print(table)
    console.print("`oca ls` shows every server.")


# -- setup -----------------------------------------------------------------------------------


@app.command(rich_help_panel="Setup")
def config(
    key: Annotated[
        str | None,
        typer.Argument(help="device, model (this machine's), port, threads, context or default."),
    ] = None,
    value: Annotated[str | None, typer.Argument(help="Its new value.")] = None,
) -> None:
    """Show the settings, or set one: `oca config device cpu`, `oca config default deepseek`."""
    settings = Settings.load()
    if key is None:
        console.print(f"# {Settings.path()}; `oca server` configures the servers")
        for k, v in vars(settings).items():
            if k != "remotes":
                console.print(f"{k} = {v!r}")
        return
    if value is None:
        console.print(repr(getattr(settings, key, None)))
        return
    settings.set(key, _pick(settings, value).id if key == "default" else value)
    settings.save()
    agent.write_config(settings)
    console.print(f"{key} = {getattr(settings, key)!r}. Applies to the next `oca`.")


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


@app.command(
    rich_help_panel="Setup",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def ucloud(ctx: typer.Context) -> None:
    """The bundled ucloud-api CLI, e.g. `oca ucloud login` or `oca ucloud jobs list`."""
    from ucloud_api.cli import app as ucloud_app

    args = ["--help" if a == "-h" else a for a in ctx.args]  # it knows only --help
    ucloud_app(args=args, prog_name="oca ucloud")


@app.command(rich_help_panel="Setup")
def uninstall(
    keep_models: Annotated[
        bool, typer.Option("--keep-models", help="Keep downloaded local models.")
    ] = False,
    purge: Annotated[
        bool, typer.Option("--purge", help="Also delete OpenCode's sessions, logs and caches.")
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


# -- plumbing: run for you, so out of the help -----------------------------------------------


@app.command(hidden=True)
def setup(
    force: Annotated[bool, typer.Option("--force", help="Reinstall binaries.")] = False,
) -> None:
    """Install pinned llama.cpp and OpenCode binaries; write the OpenCode config."""
    install.setup(force=force)
    settings = Settings.load()
    if not Settings.path().exists():
        settings.save()
    agent.write_config(settings)
    console.print(f"Installed into {data_dir() / 'bin'}. Next: `oca` (downloads the model).")


@app.command("serve", hidden=True)
def serve_cmd(
    model: Annotated[
        str | None, typer.Option(help="Model id (default: the configured one).")
    ] = None,
    idle: Annotated[int, typer.Option(help="Idle minutes before it stops; 0: never.")] = 60,
    status: Annotated[bool, typer.Option("--status", help="Only report.")] = False,
    stop: Annotated[bool, typer.Option("--stop", help="Stop the server you started.")] = False,
) -> None:
    """Serve this machine's model to other machines' oca over SSH; answers in JSON.

    `oca <id>` on the other machine runs it for you.
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
        Console(stderr=True).print(f"[red]error:[/] {e}")
        raise SystemExit(1) from None
