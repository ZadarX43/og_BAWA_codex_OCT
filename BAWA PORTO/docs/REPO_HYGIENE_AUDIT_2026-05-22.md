# Repo Hygiene Audit - 2026-05-22

Scope: classify the current dirty working tree without reverting or staging generated artifacts.

## Headline

- Branch: `dev`
- Git root: `/Users/hughwade/Documents/Code/OG_master`
- Project folder: `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO`
- Current status entries: 1,536
- Tracked modified entries: 261
- Untracked entries: 1,275
- Protected production spine dirty files: 0

The working tree is dirty mainly because generated payloads are already tracked and because a large research/archive layer is untracked. Existing `.gitignore` rules do not silence files that are already tracked.

## Bucket Split

| Bucket | Modified | Untracked | Decision |
| --- | ---: | ---: | --- |
| Generated API-football normalized data | 47 | 478 | Expected output; should not keep dirtying normal development. Decide whether to untrack generated CSV families or treat only manifests as tracked. |
| Generated frontend public data | 210 | 66 | Expected output; this is legacy static-data churn. R2/D1 compact publish path should become source of truth, with only public proof/core manifests tracked if needed. |
| Root experiments, reports, configs | 0 | 449 | Archive/review. Mostly old experiments, walk-forward outputs, model studies, loose scripts, CSV/MD reports. |
| Script research/new pipeline candidates | 0 | 110 | Review one by one. Some are useful Brain/World Cup/player-event candidates, but they are not yet committed production code. |
| Docs/runbooks | 1 | 74 | Review/consolidate. Keep canonical docs under `docs/`; archive stale markdown after merge. |
| API-football pipeline candidates | 1 | 58 | Review/commit real pipeline modules; archive one-off audits after extracting useful contracts. |
| Other archive/review directories | 0 | 34 | Recovery folders, old app/config experiments, backup-style directories. Archive or ignore. |
| Tooling/tests | 0 | 5 | Review; likely keep if they are current and runnable. |
| Worker/local site data | 0 | 1 | Local R2-style payload sample. Should be ignored unless intentionally used as a checked fixture. |

## Keep / Commit Candidates

These are the only tracked non-generated changes seen during this audit:

- `docs/SITE_INCREMENTAL_PUBLISH_COMPILER_RUNBOOK.md`
  - Updates the run order to use `scripts/site_publish/orchestrator.py`.
  - Documents settlement, SQLite export, injury shock market impact, fixture brain, summary dry-run, publish compiler, and completeness audit.
- `scripts/api_football/refresh_current_context_overlay_window.py`
  - Adds `_read_csv_optional`.
  - Makes current-context overlay refresh tolerate empty/missing feature CSVs.

The website/frontend/Worker code that was just built is clean and committed. `scripts/site_publish/*` is also clean apart from ignored `__pycache__`.

## Generated But Expected

These are expected generated assets, not product source changes:

- `build/` - ignored, about 609 MB
- `build/site_data/` - ignored, about 523 MB
- `build/site_publish/` - ignored, about 69 MB
- `frontend/public/data/` - tracked legacy/static payload area, about 122 MB
- `data_sources/api_football/normalized/` - tracked normalized API-football data, about 327 MB

Important: `frontend/public/data/*.json` is listed in `.gitignore`, but many files under that tree are already tracked, so they still appear as modified.

## Archive / Review

The untracked root and script sprawl includes:

- old training experiments
- walk-forward reports
- one-off audit scripts
- duplicate recovery/backups
- World Cup research scaffolds
- player-event research scripts
- loose CSV/MD outputs

These should be reviewed in batches, not blindly committed. The likely path is:

1. extract durable contracts into `docs/`
2. keep active scripts under `scripts/` with clear ownership
3. archive or ignore old study outputs
4. avoid committing loose root-level research files

## Ignore Candidates

Safe ignore candidates for future hygiene:

- `worker/site-data/`
- `frontend/public/data/internal/`
- local R2 payload mirrors under `worker/site-data/v*/payloads/`
- generated D1 chunks and SQLite artifacts are already covered by `build/`
- generated `__pycache__/` is already covered

Needs a deliberate index decision, not just `.gitignore`:

- `frontend/public/data/**`
- `data_sources/api_football/normalized/**`
- `reports/latest/**`

Because those paths already have tracked files, ignoring alone will not stop dirty status. If we want them quiet, use a controlled `git rm --cached` plan after deciding which manifests/results remain tracked.

## Danger Zone

The protected production spine is clean:

- `footystats_drop_ingest.py`
- `etl_press_intensity.py`
- `build_merged.py`
- `patch_merge_add_streaks.py`
- `team_ratings.py`
- `patch_merge_add_power_ratings.py`
- `make_fd_odds_enriched_synth.py`
- `patch_merge_add_synth_odds.py`
- `pipeline_qa_gate.py`
- `bookie_allmarkets.py`
- `deploy_rulebook.py`
- `slip_formatter.py`

No cleanup action should modify these files unless a separate production-spine task explicitly requires it.

## Recommended Safe Cleanup Order

1. Commit this audit plus the current runbook/API-football hardening changes as a small hygiene commit.
2. Add ignore coverage for local mirrors such as `worker/site-data/`.
3. Create a generated-data tracking decision:
   - keep only stable public proof/manifests in `frontend/public/data`, or
   - keep legacy static payloads tracked until R2/D1 fully replaces them.
4. If approved, untrack repeatable generated data with `git rm --cached` in scoped batches.
5. Move old root-level experiments/reports into an archive folder or leave ignored after documenting where durable knowledge moved.
6. Review untracked `scripts/` and `scripts/api_football/` by subsystem before committing anything.

