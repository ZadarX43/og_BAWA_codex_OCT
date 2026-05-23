# Scripts Archive Review - 2026-05-23

This folder preserves untracked review/experiment scripts that should not sit in the active `scripts/` surface.

The files here were moved, not deleted. They are kept for forensic review, possible future promotion, or later deletion after explicit approval.

## Why Archived

- Dated one-off fixture research
- FTR / BTTS / OU25 / team-goal recovery experiments
- Phase8 shadow and replay experiments
- Live shadow/repeat QA helpers
- API-football backfill/validation helpers not yet promoted into the active foundation package
- Research dashboards and exploratory scorecards

## Promotion Rule

To move a script back into active use:

1. Confirm its input and output contracts.
2. Confirm it does not change protected production-spine behavior.
3. Compile or smoke-test it.
4. Move it to the appropriate active `scripts/` area in a dedicated commit.
5. Document it in the relevant runbook.

## Not Included

The API-football `.bak_*` files remain local ignored backup files. They are not part of this tracked archive.
