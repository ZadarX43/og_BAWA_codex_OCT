# Odds Genius System Operating Map - 2026-05-23

## Purpose

This document is the working map for navigating Odds Genius without accidentally breaking the production football prediction spine.

Use it when planning upgrades, debugging a failed publish, checking data freshness, or deciding where a new feature belongs.

## Golden Rule

The production prediction spine is protected. Do not casually change:

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

If a website, publishing, account, intelligence-overlay, or UI change appears to require editing these files, pause and document the reason first.

## System Layers

### 1. Source Data Layer

Primary football data comes from:

- FootyStats drops in `FOOTYSTATS_DROP/`
- API-football raw and normalized data under `data_sources/api_football/`
- Local canonical merged match files under `Matches/__merged__/`
- Weather and contextual research layers
- Manual or official/team news overlays where available

Important rule:

- Training and production model work should use canonical merged inputs, not raw season CSVs.

### 2. Data Refresh Spine

Safe data refresh order:

1. `footystats_drop_ingest.py`
2. `etl_press_intensity.py`
3. `build_merged.py --recursive --rolling-press`
4. `patch_merge_add_streaks.py`
5. `team_ratings.py`
6. `patch_merge_add_power_ratings.py`
7. `make_fd_odds_enriched_synth.py --emit-ou25-novig`
8. `patch_merge_add_synth_odds.py`
9. `pipeline_qa_gate.py`

Hard stop:

- If integrity checks fail, do not run `bookie_allmarkets.py`.

### 3. Prediction / Deploy Layer

Core sequence:

1. `bookie_allmarkets.py`
2. `deploy_rulebook.py`
3. `slip_formatter.py`

Responsibilities:

- `bookie_allmarkets.py` creates market outputs.
- `deploy_rulebook.py` owns routing, tiers, vetoes, gates, and deploy status.
- `slip_formatter.py` formats outputs only. It must not rescue picks, invent filler, or override gates.

Public product rule:

- `OBSERVE` is not deployable unless explicitly producing research output.

### 4. API-Football Foundation

Tracked foundation lives in `scripts/api_football/`.

Main responsibilities:

- Fetch fixtures, player data, injuries, sidelined data, odds, events, lineups, and fixture stats.
- Normalize source data into compact local tables/files.
- Build player, team, H2H, injury, lineup, odds, referee, and live feature families.
- Support current-window context refreshes for website intelligence.

Important scripts include:

- `scripts/api_football/run_api_foundation.py`
- `scripts/api_football/refresh_current_injury_lineup_window.py`
- `scripts/api_football/refresh_current_context_overlay_window.py`
- `scripts/api_football/import_current_site_normalized.py`
- `scripts/api_football/import_latest_fixture_masters.py`

### 5. Intelligence Overlay Layer

Tracked overlay scripts live mostly in `scripts/` and `scripts/player_events/`.

Purpose:

- Explain whether team, player, fixture, H2H, injury, and lineup context supports or contradicts FTR, BTTS, OU25, TG1.5, and player-event reads.
- Create fixture-level audit and summary material for website cards and future GPT summaries.

Key families:

- Injury shock:
  - `scripts/build_injury_shock_coverage_scan.py`
  - `scripts/build_injury_shock_market_impact_sidecar.py`
- Goal market support:
  - `scripts/build_goal_market_overlay_support_audit.py`
  - `scripts/build_goal_market_signal_matrix.py`
- Fixture intelligence:
  - `scripts/build_fixture_market_intelligence_board.py`
  - `scripts/build_fixture_market_outcome_tracker.py`
  - `scripts/build_fixture_summary_dry_run.py`
- Player events:
  - `scripts/player_events/`
  - `scripts/build_player_event_current_board_fixture_inputs.py`
  - `scripts/build_player_event_live_feature_join.py`
  - `scripts/build_player_event_live_interaction_features.py`

Product rule:

- Overlay systems can explain support/contradiction and help product presentation, but they must not silently override deploy gates.

### 6. Website Static Contract

The launch static contract still includes selected top-level files in `frontend/public/data/`:

- `publish_summary.json`
- `public_predictions.json`
- `premium_predictions.json`
- `fixture_intelligence_public.json`
- `covered_fixture_universe.json`
- `weekly_results.json`
- `results_archive.json`
- `live_results_feed.json`

Deep generated payload mirrors are intentionally untracked:

- `fixture_decision_intelligence/`
- `fixture_lineup_intelligence/`
- `fixture_h2h_support/`
- `player_intelligence/`
- `team_intelligence/`

Reason:

- Top-level files are small public/static launch contract files.
- Deep payload families are regenerated and should move through compact R2/D1 publishing.

### 7. Site Brain / Publish Layer

The target publishing model is:

```text
local source DB/files
  -> export compact site SQLite
  -> fixture brain compiler
  -> publish compiler
  -> changed-only R2 upload
  -> D1 index delta
  -> Worker/site smoke checks
```

Important files:

- `scripts/site_publish/orchestrator.py`
- `scripts/export_site_sqlite.py`
- `scripts/site_publish/fixture_brain_compiler.py`
- `scripts/site_publish/publish_compiler.py`
- `scripts/cloudflare_preview_readiness.py`
- `docs/SITE_INCREMENTAL_PUBLISH_COMPILER_RUNBOOK.md`

Design:

- D1 is the index/control plane.
- R2 stores compact heavy payloads.
- Worker reads D1 first, then R2 payload objects.
- Static frontend files remain launch fallback until replaced by Worker/R2 contract routes.

### 8. Worker / Account / Paywall Layer

Worker responsibilities include:

- Checkout
- Account/session state
- Magic links
- Billing portal
- Premium route gating
- R2/D1 site-data reads

Important paths:

- `worker/src/index.js`
- `worker/src/site_data_store.js`
- `worker/wrangler.toml`
- `worker/test_worker_local.js`

Important smoke areas:

- Stripe checkout
- post-checkout redirect
- session restore
- billing portal
- inactive/payment issue lockout
- premium route gating
- R2 fixture payload route

### 9. Frontend Product Layer

Important frontend areas:

- Homepage launch copy and Founder offer
- Fixture pages
- Results proof page
- Matches feed
- Search/typeahead
- Team pages
- Account/paywall states

Current product direction:

- Standard: top public cards only.
- Founder/Premium: fixture context and readable intelligence.
- Pro: player-event intelligence.
- Pro+: audit/explainability and downloadable intelligence.

### 10. Public Proof / Settlement Layer

Public trust layer:

- Published picks must settle automatically.
- Results must distinguish won, lost, void, and pending.
- Stats must split FTR, BTTS, OU25, and TG1.5 instead of blending all markets.

Important files:

- `scripts/settle_published_results.py`
- `frontend/public/data/weekly_results.json`
- `frontend/public/data/results_archive.json`
- `docs/RESULTS_SETTLEMENT_AND_ARCHIVE_PLAN.md`

### 11. World Cup Layer

World Cup scripts are tracked in `scripts/`.

Purpose:

- Prepare World Cup coverage to work like domestic competitions where possible.
- Add graceful H2H fallback where historical matchup data is thin.
- Build player-event, travel/weather, macro prior, qualification context, and historical priors.

Important scripts:

- `scripts/build_world_cup_data_foundation.py`
- `scripts/build_world_cup_api_footystats_bridge.py`
- `scripts/build_world_cup_recent_history_and_h2h_sidecar.py`
- `scripts/build_world_cup_player_event_fixture_inputs.py`
- `scripts/build_world_cup_venue_weather_travel_scaffold.py`
- `scripts/report_world_cup_2026_coverage_gaps.py`

## Fresh Data Operating Order

When new FootyStats/API-football data lands:

1. Refresh data and run integrity gates.
2. Run `bookie_allmarkets.py` only after integrity passes.
3. Run `deploy_rulebook.py`.
4. Run overlay/intelligence scripts.
5. Export site SQLite.
6. Run injury shock market impact sidecar.
7. Run fixture brain compiler.
8. Run publish compiler.
9. Upload changed R2 objects and D1 delta.
10. Run Cloudflare preview readiness.
11. Smoke key fixture pages and account routes.
12. Promote only when preview is clean.

## Repo Hygiene Policy

Tracked:

- production spine
- canonical docs
- source scripts for API-football, site brain, player events, World Cup, Worker, frontend
- small static launch contract files

Untracked/ignored:

- raw generated payloads
- local normalized API-football exports
- deep frontend generated data mirrors
- backups
- generated report CSV/TXT files
- old root-level research/output sprawl unless deliberately promoted

Archive review:

- `scripts/_archive_review/`

Use this area to preserve experiments without leaving them in the active script surface.
