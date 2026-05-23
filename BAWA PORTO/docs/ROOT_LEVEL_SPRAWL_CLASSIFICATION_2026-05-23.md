# Root-Level Sprawl Classification - 2026-05-23

## Purpose

Classify the remaining untracked root-level files after generated payload cleanup, documentation commits, API-football commits, site/player-event commits, and World Cup commits.

This pass separates root markdown notes, generated CSV/TXT report outputs, and source-like files that need dependency review.

## Actions Taken

- Moved root-level markdown notes to `docs/_archive_review/2026_05_23/root_notes/`.
- Added root-only ignore rules for generated CSV/TXT outputs:
  - `/*.csv`
  - `/*.txt`
- Left root-level source-like files in place for dependency review before moving or committing.

## Why Root Python Was Not Moved Yet

Root-level Python includes files that may still be imported by older scripts or production-adjacent tooling, including examples such as:

- `constants.py`
- `deploy_presets.py`
- `deploy_gates.py`
- `prediction_overlay.py`
- `train_markets.py`
- `train_investor_leagues_v2.py`
- `train_hybrid_xgb.py`
- `team_rating_engine.py`
- `streaks_module.py`

Moving these without an import/dependency review could break local workflows even if they are currently untracked.

## Buckets

### Archived Notes

Root markdown notes are now preserved under:

- `docs/_archive_review/2026_05_23/root_notes/`

These are review material, not canonical docs.

### Ignored Generated Outputs

Root CSV/TXT files are treated as generated reports, manifests, audits, or local output unless explicitly promoted.

They remain on disk but should not dirty Git.

Examples:

- walk-forward CSVs
- branch comparison CSVs
- modelstore dir lists
- leakage/audit CSVs
- local report text files

### Source-Like Files Requiring Review

Root Python, shell, JSON, YAML, TOML, INI, and odd extension files remain visible until they are sorted into one of:

- active source to commit in place;
- source to move under `scripts/`;
- source to archive under `scripts/_archive_review/`;
- obsolete local file to ignore or delete after approval.

## Recommended Next Step

Run a dependency/import scan over root source-like files:

1. Identify files imported by tracked production, training, or publish scripts.
2. Commit true active dependencies in their expected locations.
3. Move old experiments into `scripts/_archive_review/`.
4. Add only narrow ignore rules for confirmed local-only leftovers.

## Guardrail

Do not move root source-like files just to reduce `git status` noise. Move them only after confirming they are not imported by the active system.
