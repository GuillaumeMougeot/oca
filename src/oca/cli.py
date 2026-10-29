"""Command-line interface: ``oca`` launches the agent, subcommands manage the stack."""

from __future__ import annotations

import os
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import agent, install, server
from .config import MODELS, OcaError, Settings, data_dir, model_path

app = typer.Typer(
    name="oca",
    help="Open Coding Agent. `oca` alone sets up what is missing and opens OpenCode.",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()
Yes = Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before downloading.")]


def _ready(settings: Settings, yes: bool) -> None:
    if not install.is_installed():
        console.print("[bold]First run: installing llama.cpp and OpenCode[/]")
        install.setup()
    model = settings.resolved_model
    if not model_path(model).exists():
        where = model_path(model).parent
        if not (yes or typer.confirm(f"Download {model.id} ({model.size_gb} GB) to {where}?")):
            raise typer.Exit(1)
        install.pull(model)
    with console.status(f"Starting {model.id} on {settings.resolved_device}"):
        server.start(settings)
    agent.write_config(settings)


@app.callback()
def main(ctx: typer.Context, yes: Yes = False) -> None:
    if ctx.invoked_subcommand is None:
        settings = Settings.load()
        _ready(settings, yes)
        agent.launch([])


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context, yes: Yes = False) -> None:
    """Like `oca`, passing extra arguments to OpenCode (e.g. `oca run -- run "fix the tests"`)."""
    settings = Settings.load()
    _ready(settings, yes)
    agent.launch(ctx.args)


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
    """Download model weights (default: the configured model)."""
    settings = Settings.load()
    if model and model not in MODELS:
        raise OcaError(f"unknown model {model!r}; see `oca models`")
    console.print(f"Saved {install.pull(MODELS[model] if model else settings.resolved_model)}")


@app.command()
def models() -> None:
    """List the model catalog."""
    table = Table("id", "GB", "context", "downloaded", "note")
    for m in MODELS.values():
        have = "yes" if model_path(m).exists() else ""
        table.add_row(m.id, f"{m.size_gb:g}", str(m.context), have, m.note)
    console.print(table)


@app.command()
def start() -> None:
    """Start the local model server (restarts it if settings changed)."""
    settings = Settings.load()
    with console.status("Starting llama-server"):
        state = server.start(settings)
    agent.write_config(settings)
    console.print(f"Serving {settings.resolved_model.id} at http://127.0.0.1:{state.port}/v1")


@app.command()
def stop() -> None:
    """Stop the local model server."""
    console.print("Stopped." if server.stop() else "Not running.")


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
    if state:
        health = "ready" if server.healthy(state.port) else "loading"
        table.add_row("server", f"pid {state.pid}, port {state.port}, {health}")
    else:
        table.add_row("server", "stopped")
    console.print(table)


@app.command()
def logs(
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
    lines: Annotated[int, typer.Option("--lines", "-n")] = 40,
) -> None:
    """Show the llama-server log."""
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


def entrypoint() -> None:
    try:
        app()
    except OcaError as e:
        console.print(f"[red]error:[/] {e}")
        raise SystemExit(1) from None
