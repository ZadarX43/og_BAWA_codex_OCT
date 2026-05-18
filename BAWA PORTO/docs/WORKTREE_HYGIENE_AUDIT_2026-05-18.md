# Worktree Hygiene Audit - 2026-05-18

Purpose: classify the current dirty worktree into safe commit groups and cleanup/review buckets.

Important caveat: file modification time is filesystem evidence, not authorship proof. It is still useful for reconstructing when a workstream likely happened.

## Current Head

- Branch: `dev`
- Head: `bfb21e1912 Surface real fixture model outputs`
- That commit contains the site-wide fixture market-card model-output fix.

## Inventory Snapshot

From `git status --porcelain=v1`:

| Group | Status entries | Tracked modified | Untracked status entries | Size | Mtime range |
|---|---:|---:|---:|---:|---|
| API-football/data refresh artifacts | 560 | 48 | 512 | 296.3 MB status entries / 3.2 GB actual untracked files | 2026-04-28 to 2026-05-18 |
| Website fixture/H2H/lineup intelligence payloads | 322 | 322 | 0 | 1.17 MB | 2026-05-13 to 2026-05-16 |
| Docs/runbooks | 144 status entries / 188 actual files | 0 | 144 status entries | 1.34 MB | 2025-12-17 to 2026-05-18 |
| Stray experimental/untracked files | 476 status entries / 890 actual files | 0 | 476 status entries | 14.4 MB status entries / 54.4 MB actual files | 2024-09-13 to 2026-05-18 |
| Player-event beta/research artifacts | 9 status entries / 193 actual files | 0 | 9 status entries | 0.12 MB status entries / 1.76 MB actual files | 2026-05-02 to 2026-05-17 |
| Cloudflare/account smoke tooling | 7 | 1 | 6 | 0.06 MB | 2026-05-04 to 2026-05-18 |
| Results settlement/proof outputs | 2 | 2 | 0 | 0.13 MB | 2026-05-18 |
| Frontend internal/admin data | 1 | 0 | 1 | 0.47 MB | 2026-05-18 |

`git ls-files --others --exclude-standard` shows 3,984 actual untracked files. `git status` collapses some untracked directories, so both counts matter.

## Recommended Commit Groups

### 1. Website Fixture/H2H/Lineup Intelligence Payloads

Likely scope:

- `fixture_h2h_support_engine.py`
- `fixture_lineup_intelligence_engine.py`
- `publish_fixture_intelligence.py`
- `scripts/export_site_sqlite.py`
- `scripts/external_content/build_fixture_context_signals.py`
- `docs/FIXTURE_DECISION_RECONCILER_SPEC.md`
- `frontend/public/data/fixture_h2h_support/**`
- `frontend/public/data/fixture_lineup_intelligence/**`
- `frontend/public/data/fixture_intelligence_public.json`
- `frontend/public/data/external_content/fixture_media/2026_05_10_FC_Barcelona_Real_Madrid.json`

When made:

- Main payload/code mtimes cluster around 2026-05-13 21:57 to 2026-05-13 22:32.
- Spec doc was updated 2026-05-16 22:32.

What it appears to be:

- Adds snapshot/freshness metadata to fixture, H2H, and lineup publish payloads.
- Regenerates publish-safe H2H/lineup JSON for the May 9-11 fixture set.
- Adds SQLite export schema support for snapshot metadata.

Recommended action:

- Commit as a single coherent website-publish metadata commit, after rerunning the relevant frontend/static smoke.

### 2. Results Settlement/Proof Outputs

Likely scope:

- `frontend/public/data/weekly_results.json`
- `frontend/public/data/results_archive.json`

When made:

- Current dirt from 2026-05-18 21:20.

What it appears to be:

- Timestamp-only churn from smoke/readiness commands.
- `generated_at` and repeated `settled_at` values changed from an earlier run to a later run.

Recommended action:

- Do not commit as-is unless intentionally refreshing proof timestamps.
- Prefer reverting this timestamp-only churn or making the settlement command deterministic/stable for smoke runs.

### 3. Cloudflare/Account Smoke Tooling

Likely scope:

- `worker/src/auth.js`
- `scripts/smoke_live_worker_account.py`
- `docs/ACCOUNT_FLOW_UX_SPEC.md`
- `docs/AUTH_ROUTE_CONTRACT_SPEC.md`
- `docs/REAL_AUTH_MAGIC_LINK_PLAN.md`
- `CLOUDFLARE_PAGES_CONNECT_STEPS.md`
- `WEBSITE_GIT_CLOUDFLARE_MASTER_RUNBOOK.md`

When made:

- Cloudflare docs: 2026-05-04.
- Account/auth docs: 2026-05-08.
- `worker/src/auth.js`: 2026-05-18 15:44.
- live account smoke script: 2026-05-18 17:40.

What it appears to be:

- Account/checkout gating documentation and smoke tooling.
- Worker auth change exposes `price_id` and `access_tier` from premium access verification.

Recommended action:

- Commit code/smoke script separately from docs.
- Run account/checkout smoke before commit if this is still intended.

### 4. Docs/Runbooks

Likely scope:

- Many root markdown files from March/April.
- Many `docs/*` canonicalization/runbook files from May.
- `.codex/skills/**` local repo skills and references.

When made:

- Broad range from 2025-12-17 to 2026-05-18.
- Repo-local `.codex/skills` cluster around 2026-04-28.
- Website launch docs cluster around 2026-05-12 to 2026-05-18.

What it appears to be:

- Mixed historical docs, repo-local skills, canonical docs, and working notes.

Recommended action:

- Do not commit as one giant docs commit.
- Split into:
  - repo-local `.codex/skills` commit
  - canonical `docs/` launch/runbook commit
  - root legacy markdown archive/review commit

### 5. API-Football/Data Refresh Artifacts

Likely scope:

- `data_sources/api_football/normalized/**`
- `data_sources/api_football/features/**`
- `data_sources/api_football/raw/**`
- `data_sources/api_football/features_calendar_year/**`
- `scripts/api_football/**`
- `data_sources/hybrid/**`

When made:

- Earliest cluster starts 2026-04-28.
- Large refresh/update cluster continues through 2026-05-18.

What it appears to be:

- New API-football foundation scripts plus large normalized/raw/features outputs.
- Tracked normalized files were expanded from a small current-window slice into much larger season/full-season files.

Recommended action:

- Do not commit the whole 3.2 GB untracked data estate casually.
- Split code from generated data:
  - commit reviewed `scripts/api_football/**` if desired
  - decide whether raw/features/normalized generated outputs belong in git, storage, or `.gitignore`
  - if tracking generated data, commit only a defined manifest/window, not every artifact by accident

### 6. Player-Event Beta/Research Artifacts

Likely scope:

- `scripts/player_events/**`
- player-event beta builders and refresh runners under `scripts/`

When made:

- 2026-05-02 to 2026-05-17.

What it appears to be:

- Beta/research scripts for player-event boards, live feature joins, tactical registry, and exact interaction refreshes.

Recommended action:

- Commit on a research/beta branch or in a clearly labelled player-event beta commit.
- Do not mix with website launch or production deploy commits.

### 7. Stray Experimental/Untracked Files

Likely scope:

- Old root scripts and reports
- recovery folders
- historical experiments
- duplicate trainers/backtests
- ad hoc CSVs
- old runbooks outside canonical `docs/`

When made:

- 2024-09-13 to 2026-05-18.

What it appears to be:

- Mixed legacy project debris, recovery snapshots, experiments, and possibly useful historical references.

Recommended action:

- Do not commit wholesale.
- Create an archive/ignore review:
  - keep known canonical files
  - archive recovery dumps outside active repo
  - add generated output folders to `.gitignore` where safe
  - explicitly promote only still-used scripts

## Immediate Cleanup Candidates

Low-risk cleanup candidates:

- Revert timestamp-only churn in:
  - `frontend/public/data/weekly_results.json`
  - `frontend/public/data/results_archive.json`

Needs review before cleanup:

- Any API-football generated data.
- Any root-level legacy scripts.
- Any modified tracked publish payloads, because they may represent real website intelligence work.

## Proposed Commit Order

1. Revert or stabilize results timestamp churn.
2. Commit website fixture/H2H/lineup intelligence metadata payloads.
3. Commit Cloudflare/account smoke tooling.
4. Commit canonical docs/runbooks and repo-local skills.
5. Separately review API-football code vs generated data.
6. Separately review player-event beta research.
7. Archive/ignore/review stray experimental files.

