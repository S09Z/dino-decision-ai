@AGENTS.md

## Claude Code

- Shared guidance lives in AGENTS.md (imported above); edit it there, not here. Keep this file Claude-only so it doesn't drift from what Codex reads.
- **Current work:** read `TODO.md` first for the current phase and its exit criteria, and tick items off as they are completed. `PLAN.md` is background, not status: it has outdated claims (e.g. `laya>=1.0.0`, "✅" on unbuilt features).
- **Planning:** for multi-step work, state a short plan with a verify step per item before editing.
- **Shipping:** use the `push-draft-pr` skill for commit, push and Draft PR. Never push or open a PR without the user's go-ahead.
- **Skills:** add new project skills in `.claude/skills/` and symlink them into `.agents/skills/` so Codex sees them.
