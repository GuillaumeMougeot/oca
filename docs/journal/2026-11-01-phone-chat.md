# Phone access without Termux

**Status:** OPEN, on hold since 2026-11-01. Telegram came first, but messages would pass through its servers without end-to-end encryption; revisit, or drop if it brings too little.

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

Sources: [OpenClaw security](https://www.techradar.com/pro/here-are-the-openclaw-security-risks-you-should-know-about), [OpenClaw and WhatsApp](https://zenvanriel.com/ai-engineer-blog/openclaw-whatsapp-risks-engineers-guide/), [OpenClaw with vLLM](https://documentation.openclaw.ai/providers/vllm), [an OpenCode Telegram bridge](https://github.com/vineetkishore01/Opencode-Telegram).
