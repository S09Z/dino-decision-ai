---
name: push-draft-pr
description: Use when the user wants to ship the current work in this repo for review — validates it (make lint, make test), commits only the relevant files, pushes a feature branch, and opens a Draft PR on GitHub. Trigger on "commit and push", "open a PR", "ship this", "push a draft PR", "send this for review", even if they don't say "draft" explicitly.
---

# push-draft-pr

## Purpose

Take the current, finished change in this repo (Chrome Dino AI: DQN/PPO agents, Laya router,
FastAPI dashboard) to a reviewable Draft PR in one verifiable increment: validate, commit only
what belongs, push a feature branch, open a Draft PR.

## Safety rules

- Never force push (no `--force`, no `--force-with-lease`).
- Never commit directly to `main`/`master`. Always use a feature branch.
- Never use `--no-verify` or skip hooks. If a hook fails, fix the cause.
- Never discard, stash, or overwrite unrelated changes. Check `git status` first; leave
  unrelated work alone and tell the user.
- Only open a **Draft** PR (`gh pr create --draft`).
- One logical change per PR. Do not bundle unrelated fixes.
- Never commit secrets or artifacts: `.env*` (except `.env.example`), model weights
  (`/models/`, `*.pth`, `*.pt`, `*.ckpt`, `*.zip`), logs, datasets, databases.
- Pushing and opening a PR are visible actions on a shared repo. Get the user's go-ahead in chat
  first. Approval for one push does not cover a later push.

## Preconditions

Check before starting:

- [ ] `git rev-parse --is-inside-work-tree` succeeds. If not, stop and tell the user to
      `git init` (do not init for them unprompted).
- [ ] `git remote -v` shows `origin` on GitHub. If not, stop and ask for the remote.
- [ ] `gh auth status` is authenticated
- [ ] `git status` reviewed; unrelated changes identified and left untouched
- [ ] Base branch known (default `main`; confirm with `git symbolic-ref refs/remotes/origin/HEAD`)

## Workflow

### 1. Confirm scope

State in one or two lines what change is being shipped and which files it touches, and the base
branch. Redirect now if wrong.

### 2. Validate

Run and read the real output. Do not claim success on "should work":

```bash
make format    # black + isort; review the resulting diff
make lint      # black --check + mypy src
make test      # pytest tests/ -v --cov=src
```

If a check fails, fix the cause (or report it to the user if unrelated to this change). Never
weaken tests or lint config to get green. Note any check you could not run and why.

### 3. Commit

Stage by explicit path, never `git add -A` or `git add .`. Run `git status` and `git diff --staged`
to confirm nothing from the safety list slipped in. Match the style in `git log`
(`type(scope): summary`, e.g. `feat(routing): add Laya agent router`). Keep the message about
why, not just what.

### 4. Branch

Branch name: `claude/<topic-slug>` (hyphens). Branch from the base unless the user is already on a
relevant feature branch. If the work is stacked on another open PR (`gh pr list --state open`),
branch from that PR's branch and pass `--base <that branch>` in step 5.

### 5. Confirm, push, open the Draft PR

Show the user one summary: files in the commit, branch, base, PR title and body. After a single
go-ahead covering both, run:

```bash
git push -u origin <branch>
gh pr create --draft [--base <stack-base>] --title "<title>" --body "$(cat <<'EOF'
## Summary
<what changed and why>

## Changes
- <bullet per notable change>

## Validation
- [ ] `make lint` passes
- [ ] `make test` passes

## Out of scope
<follow-ups deliberately left for another PR>
EOF
)"
```

Tick (`[x]`) a validation box only for a command actually run in step 2 that passed; leave the rest
unchecked and say why. Report the PR URL when done.

### 6. Follow-ups

If `PLAN.md` tracks progress for this work, update it in a follow-up commit on the **same feature
branch** (never `main`). That is a second push; ask first.

## Notes

- Many `src/` modules are scaffolds; a green test run may cover little. Say so if coverage of the
  change is thin rather than implying more assurance than exists.
- If there is nothing to ship (clean tree, no new commits vs base), say so and stop.
