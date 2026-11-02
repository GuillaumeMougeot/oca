# CLI review

**Status:** RESOLVED. Built on `remotes` the same day, with one change of the user's: what you open is a model on a server, not a server.

## Question
Is oca's command line easy to learn, consistent, and fit for daily use? Reviewed on branch `remotes` (c94477c) against the tools its users know: Claude Code first, which the user came from, then uv, Ollama, git, docker and gh, and the [Command Line Interface Guidelines](https://clig.dev).

## What works
- The daily loop is short: `oca` opens the default server, `oca <server>` another, `oca ls` lists them, `oca start`/`stop <server>` manage them.
- Help is grouped (Sessions, Servers, Setup), `-h` works, and errors name the command that fixes them.
- Steps that cost or delete something ask first, `-y` skips the questions, and a down server gets a numbered menu.

## Findings
| # | Finding | Proposal | When |
|---|---|---|---|
| 1 | `oca ucloud` ran the bundled UCloud tool. So the server named `ucloud`, the name oca gives migrated UCloud settings, could not be opened with `oca ucloud`; `oca remote add ucloud ucloud` was refused; `oca ucloud -h` failed. | `oca ucloud-api …` for the tool. **Done.** | bug |
| 2 | `-r` means "remote server" here but "resume" in Claude Code, and `oca <server>` makes it redundant. | `-r/--resume <id>` resumes a session, as in Claude Code, replacing `-s/--session`. `-s/--server` picks a server in commands that take one as an option. | before merge |
| 3 | `oca run -- run "fix the tests"` says run twice and cannot pick a server. | `oca run "<task>" [-s server]` runs one task unattended, like `claude -p`. Raw OpenCode arguments go through a hidden `oca opencode …`. | before merge |
| 4 | Without a server name, `oca` uses the default server; `oca start`, `stop`, `logs` and `pull` use this machine; `oca remote logs` and `pull` the default remote. | One rule: no name means the default server. `oca stop --all` also stops oca's endpoint. | before merge |
| 5 | Two nouns for one thing. Help, menus, `oca ls` and OpenCode say "server"; the group, `-r` and some messages say "remote". `oca remote default local` reads oddly. | Rename the group `oca server` (`add`, `rm`, `default`, `set`, `extend`). `oca logs` and `oca pull` take a server. | before merge |
| 6 | The default server can be set in two places: `oca remote default` and `oca config default`. `oca status` repeats `oca ls`. | One place each: `oca server default`; `status` covers this machine only. | soon |
| 7 | `oca serve` is plumbing (JSON for oca on other machines) but is listed with user commands. | Hide it, as git hides its plumbing. | soon |
| 8 | Errors go to stdout; there is no `--version`. | Errors to stderr; `oca --version`. | soon |
| 9 | Boolean flags show twice (`--force/--no-force`); `-f` and `-n` have no help; `oca config -h` does not list its keys. | Single flags, a line of help each, keys listed. | soon |
| 10 | Choosing a model means typing an id: `(qwen3.8-flash-next/qwen3.6-35b-a3b/qwen3.8-27b)`. | A numbered menu, like the down-server one: Enter takes 1. | soon |
| 11 | `oca <server>` must come first: `oca -c spark` fails. | Accept options before the server name. | later |
| 12 | No shell completion. | `oca --install-completion`, completing server names and model ids. | later |
| 13 | `oca sessions --delete <id>` and `--clear` are actions written as flags, while `oca web stop` is a subcommand. | `oca sessions rm <id>` and `oca sessions clear`. | later |

"Before merge" means before `remotes` reaches main: these change names that main's users would otherwise learn twice.

## Decisions (2026-11-02)
The user approved all three, and corrected the unit: `oca ls` lists every server and each model it can serve; each such pair has an id, `oca <id>` opens it, and one pair is the default. They also kept `oca ucloud` for the UCloud tool (finding 1 reverted): ids contain `@`, so it no longer clashes with the server named ucloud.

- **Ids:** the model's short name at the server: `qwen3.6@local`, `flash-next@spark`, `deepseek@ucloud`. Not row numbers, which shift when a server is added, nor random codes, which nobody remembers. Part of an id will do: an exact id first, then ids containing the word, then model names (`oca v4.1`); a word that fits several gets a numbered menu, or an error listing them without a terminal.
- **What is listed:** UCloud's models, the llama.cpp models that fit this machine (its GPU, or RAM on CPU) and an SSH machine (its memory, recorded when added), and a URL's one model.
- **One model per server at a time:** opening another model on a running server restarts it. On an SSH machine oca asks first; on one someone else started, it offers the running model instead, since only the starter may switch. UCloud already asked. `oca ls` shows the state only on the model a server serves.
- **OpenCode** still lists one model per server, named by the id it serves. Keyed by server, a session outlives a change of model; and a keyboard shortcut mid-session should not start a 2×B200 job.
- **Fewer commands:** `oca start` went (`oca <id>` starts what it opens, and replies wait while it starts), as did `oca models` (`oca ls`) and `oca remote default` (`oca config default <id>`). `oca stop`, `logs` and `pull` take an id or a server, and act on the default without one; `oca stop --all` stops everything.
- **Is a server group still needed?** Yes, for the rare actions on machines: `oca server add|rm|set|extend`. Without it they would be four more top-level verbs, or hidden in `oca config`. Pairs are what you choose; servers are what you add, configure and stop.
- **Also done:** `-r` resumes and `-m` picks a model; `oca run "<task>"`; `--version`; errors to stderr; `serve` and `setup` hidden; single flags; help on every option; a numbered menu for the model when adding a machine; `oca status` covers this machine; `oca ls` prints a line per server, then one per model, to fit 80 columns. Help now lists 13 commands, against 17 and the `remote` group's 7.
- **Then, the same day:** findings 12 and 13. Tab completion completes commands, ids, also as the first word, and server names; at the user's request it is on by default, without typer's two options: the first `oca` in a terminal writes it where bash-completion and fish load it by themselves (zsh gets typer's line in ~/.zshrc), 0.1 s per Tab. `oca sessions rm <id>` and `oca sessions clear` replaced the flags.
- **`oca sessions`** used OpenCode's `session list`, which opened a pager (ctrl+c to leave) and cut titles. oca now prints the list itself: the latest first, whole titles, ids cut to the length that tells them apart. `oca -r` takes the start of an id, and alone offers a numbered list, like Claude Code's `--resume`.
- **States in `oca ls`:** the user asked why some rows said "not downloaded" and others nothing. Blank had meant "not the model its server serves", whether downloaded or not. Now the served model, or the one a stopped server would start with, shows the server's state, and the others say "downloaded" or "not downloaded": this machine checks its files, UCloud its drive, and a machine over SSH reports it in `oca serve --status` (blank for an older oca there, which `oca server update` fixes).

## The command line now
```
oca [id] [-c | -r <session>] [-y]   OpenCode on a model (the default if none)
oca run "<task>" [-m id]            one task, unattended
oca ls                              each model on each server, with its id; * the default
oca stop [id|server] [--all]        stop or leave its server
oca logs [id|server] [-f]           its log
oca pull [id]                       download its model ahead of time
oca server add|rm|set|extend        configure servers
oca sessions [rm <id> | clear]      this folder's sessions
oca web · status · config · update · ucloud · uninstall
oca -V | -h
```
