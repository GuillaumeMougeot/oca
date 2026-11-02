# Phone access without Termux

**Status:** OPEN. Reopened 2026-11-02: chat and coding from the phone, through no server but the user's own. Proposed: Matrix on the user's VPS; waiting for the user.

## Question
The phone reaches `oca web` through an SSH tunnel opened in Termux, the hardest setup step. Would OpenClaw, which connects chat apps to an agent, do better?

## Findings (2026-11-01)
- OpenClaw is a second agent with shell access, not a front end to OpenCode: chats bypass OpenCode's sessions, permissions and AGENTS.md. It accepts OpenAI-compatible servers, so it could use oca's router.
- Security: hundreds of advisories in 2026, including a 9.9 approval bypass leading to code execution. Koi Security found 341 of 2,857 ClawHub skills malicious.
- Its WhatsApp channel uses Baileys, an unofficial WhatsApp Web client. That breaks Meta's terms, risks a ban, and breaks on protocol changes. Telegram's bot API is official.
- Telegram bridges to `opencode serve` already exist: prompts, sessions, permission approval. They only connect outward, so no tunnel is needed. Messages pass through Telegram's servers.
- Tailscale would let the phone open `oca web` directly, with the full UI, if IT allows it on a university machine.

## Recommendation
Not OpenClaw. Build a small `oca telegram` bridge to OpenCode's server, locked to one Telegram account. Add Tailscale if IT allows it.

## Reopened (2026-11-02)
The user wants to chat and code from the phone, with the local model or any remote, as with Claude's apps, and without Termux or anyone else's server: Telegram is out. Their workstation already has a WireGuard link to a VPS of theirs that faces the internet.
- OpenCode's web page is clumsy on a phone: the send arrow opens the variant picker instead. Its phone apps are young one-person projects, and the three Matrix-to-OpenCode bridges found are too: none handles approvals or oca's menus, and one routes through Beeper.
- Proposed: a Matrix server on the VPS (Tuwunel, a single small program; Synapse if it misbehaves; not Dendrite, which only gets security fixes), Element X on the phone, and an `oca matrix` bot of oca's own that drives OpenCode's server. Rooms per project plus a chat room; polls for oca's menus and approvals; notifications when a task ends or needs the user; obeys only the user's account.
- WireGuard on the phone stays useful for the full OpenCode page.
- Open: a domain for the VPS (the app needs HTTPS), and whether it runs Docker.

Sources: [OpenClaw security](https://www.techradar.com/pro/here-are-the-openclaw-security-risks-you-should-know-about), [OpenClaw and WhatsApp](https://zenvanriel.com/ai-engineer-blog/openclaw-whatsapp-risks-engineers-guide/), [OpenClaw with vLLM](https://documentation.openclaw.ai/providers/vllm), [an OpenCode Telegram bridge](https://github.com/vineetkishore01/Opencode-Telegram).
