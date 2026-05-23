# BAWA PORTO Root Document Index

Purpose: make the messy root safer to navigate without moving historical runbooks or changing production behavior.

This file is a pointer only. It does not replace `AGENTS.md`, the production spine, or any live runbook.

## Start Here

| Need | Read |
| --- | --- |
| Repo safety rules for Codex/agents | `AGENTS.md` |
| Current repo hygiene plan | `docs/REPO_HYGIENE_SAFE_REORG_PLAN_2026-05-12.md` |
| First cleanup batch report | `docs/REPO_HYGIENE_BATCH_1_EXECUTION_REPORT_2026-05-12.md` |
| Root markdown index | `docs/ROOT_MARKDOWN_INDEX_2026-05-12.csv` |
| Root markdown summary | `docs/ROOT_MARKDOWN_INDEX_SUMMARY_2026-05-12.md` |
| Root markdown temporal review | `docs/ROOT_MARKDOWN_TEMPORAL_REVIEW_2026-05-12.md` |
| Root markdown temporal index | `docs/ROOT_MARKDOWN_TEMPORAL_INDEX_2026-05-12.csv` |
| Current system integral file map | `docs/CURRENT_SYSTEM_INTEGRAL_FILE_MAP_2026-05-12.md` |
| Current system candidate CSV | `docs/CURRENT_SYSTEM_FILE_CANDIDATE_INDEX_2026-05-12.csv` |
| Walk-forward/backtest spine audit | `docs/WALKFORWARD_BACKTEST_SPINE_AUDIT_2026-05-12.md` |
| Walk-forward/backtest runbook draft | `docs/WALKFORWARD_BACKTEST_CANONICAL_RUNBOOK_DRAFT_2026-05-12.md` |
| Working state freeze, 2026-04-17 | `docs/WORKING_STATE_FREEZE_2026-04-17.md` |
| BTTS markdown cluster review | `docs/BTTS_MARKDOWN_CLUSTER_REVIEW_2026-05-12.md` |
| Existing root restructure map | `docs/ROOT_RESTRUCTURE_MOVE_MAP.md` |

## Protected Zones

Do not sweep these into generic folders during repo cleanup.

| Zone | Protected examples |
| --- | --- |
| Prediction production spine | `footystats_drop_ingest.py`, `etl_press_intensity.py`, `build_merged.py`, `patch_merge_add_streaks.py`, `team_ratings.py`, `patch_merge_add_power_ratings.py`, `make_fd_odds_enriched_synth.py`, `patch_merge_add_synth_odds.py`, `pipeline_qa_gate.py`, `bookie_allmarkets.py`, `deploy_rulebook.py`, `slip_formatter.py` |
| Website publish spine | `publish_predictions.py`, `validate_public_export.py`, `frontend/public/data/*`, `frontend/assets/app.js`, `worker/src/index.js`, `worker/wrangler.toml` |
| Walk-forward / backtest / model-generation spine candidates | `run_walkforward_windows.py`, `backtest_deploy_csv.py`, `train_investor_leagues_v2.py`, `train_markets.py`, late-April/May validation and trainer files |
| Active research/intelligence spine | `scripts/api_football/`, `scripts/player_events/`, `scripts/build_live_shadow_research_dashboard.py`, fixture-market intelligence, tactical registry, player-event trackers |
| Sensitive roots | `Matches/`, `ModelStore/`, `predictions_output/`, `data_sources/` |

## Operator Runbooks Held In Root

These are command/path dense or likely used by operator muscle memory. Do not move until each has a reviewed replacement or a root pointer.

| File | Current action |
| --- | --- |
| `OG : BAWA Runbook (One-Page).md` | hold root or replace with pointer later |
| `OG Deploy + Backtesting Runbook.md` | hold root or replace with pointer later |
| `DEPLOY_BACKTEST_RUNBOOK.md` | hold root or replace with pointer later |
| `DEPLOYMENT_RUNBOOK.md` | hold root or replace with pointer later |
| `DATA_UPDATE_CHECKLIST.md` | hold root or replace with pointer later |
| `DATA_UPDATE_CHECKLIST_RUNBOOK.md` | hold root or replace with pointer later |
| `LIVE_DEPLOYMENT_AUDIT.md` | hold root or replace with pointer later |
| `deploy_weekend_runner.md` | hold root or replace with pointer later |
| `OG Backtesting Runbook (19 Leagues, 3 Years).md` | hold root or replace with pointer later |
| `OG : BAWA WEEKEND RUNBOOK April_26.md` | hold root or replace with pointer later |

## Command-Dense Docs Held For Reconciliation

These likely overlap with canonical docs, but they contain enough exact commands and path assumptions that they should be merged manually.

| Root file | Possible canonical doc |
| --- | --- |
| `TRAINING_PIPELINE__MARKETS_AND_MODELS.md` | `docs/TRAINING_PIPELINE.md` |
| `FTR_DEPLOY_RULEBOOK.md` | `docs/DEPLOY_RULEBOOK.md` |
| `MERGED_REBUILD_PIPELINE_README.md` | `docs/PIPELINE_RUNBOOK.md` |
| `ODDS_SYNTH__UNDER25_AND_MERGE_PIPELINE.md` | `docs/PIPELINE_RUNBOOK.md` |
| `FootyStats_Ingest_README.md` | `docs/PIPELINE_RUNBOOK.md` |
| `BTTS_POLICY_DECISION.md` | `docs/BTTS_RULEBOOK.md` |
| `BTTS_RULEBOOK_AUDIT_REPRODUCIBILITY.md` | `docs/BTTS_RULEBOOK.md` |
| `BTTS_WALKFORWARD_SUMMARY.md` | `docs/BTTS_RULEBOOK.md` |

## Website Publish Docs Held In Root

These stay put until the frontend/worker bridge is mapped.

| File | Reason |
| --- | --- |
| `README_FRONTEND.md` | frontend operator entrypoint |
| `CLOUDFLARE_PAGES.md` | Cloudflare Pages setup |
| `CLOUDFLARE_PAGES_CONNECT_STEPS.md` | Cloudflare connection steps |
| `WEBSITE_GIT_CLOUDFLARE_MASTER_RUNBOOK.md` | publish/deploy runbook |
| `PUBLIC_EXPORT_POLICY.md` | publish data policy |
| `LAUNCH_PREDEPLOY_CHECKLIST.md` | website launch validation |
| `WORKER_BACKEND_PLAN.md` | worker/backend planning |
| `WORKER_SUBSCRIPTION_STATE.md` | worker subscription state |

## Active Research/Intelligence Docs Held In Root

These are not production deploy, but they are active system intelligence.

| File | Reason |
| --- | --- |
| `PLAYER_PROP_BETA_README.md` | player prop beta/research boundary |
| `TEAM_SNAPSHOT_WEEKEND_SUPPORT_BOARD_README.md` | active snapshot intelligence |
| `OG_Snapshot_Weekend_Runner_README.md` | snapshot workflow |
| `OG Snapshot Weekend Findings + Next-Stag.md` | snapshot findings/history |
| `WEATHER_BACKFILL_README.md` | weather research boundary |
| `WEATHER_STADIUM_REGISTRY_README.md` | stadium/weather commands |

## Cleanup Policy

Before moving any Python file, generate a reference report covering:

- imports
- shell calls
- docs references
- default path assumptions
- website publish references
- research/intelligence references

Before moving root markdown, reconcile it against its canonical `docs/` target and leave a pointer if operators may still expect the old root path.
