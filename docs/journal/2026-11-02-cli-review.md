# CLI review

**Status:** OPEN. One bug fixed; the rest waits for the user's decisions.

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

## After the changes
```
oca [server] [-c | -r <session>] [-y]   OpenCode on a server (the default if none)
oca run "<task>" [-s server]            one task, unattended
oca ls                                  servers: where, model, state; * marks the default
oca start [server] [model] [-w]         start a server (the default if none)
oca stop [server] [--all]               stop it; --all also stops oca's endpoint
oca logs [server] [-f] [-n N]           its log
oca server add|rm|default|set|extend    configure servers
oca sessions [rm <id> | clear]          this folder's sessions
oca web [stop]                          OpenCode in a browser
oca models | pull [model] [-s server] | config [key [value]] | status | update | setup | uninstall
oca ucloud-api …                        the bundled UCloud tool
oca --version
```

## Decisions for the user
1. Flags as in Claude Code: `-r` resumes, `-s` picks a server (finding 2). Recommended.
2. Rename `oca remote` to `oca server` (finding 5). Recommended.
3. No server name means the default server, for every command (finding 4). Recommended.
