# Pre-Deploy Intelligence Run Order - 2026-05-13

## Purpose

Define how Odds Genius should run the normal prediction model and the newer team/player/fixture intelligence layers before deploy decisions are made.

This is a workflow contract only. It does not change `deploy_rulebook.py`.

## Core Rule

Deploy policy can only learn from intelligence layers that were generated before kickoff and marked:

```text
pre_kickoff_eligible = true
```

Post-kickoff or backfilled intelligence can be used for discovery, UI testing, and post-match review, but not as proof that the system knew something before kickoff.

## Safe Run Order

### 1. Refresh source data

Run the protected data-refresh stack:

```bash
python3 footystats_drop_ingest.py
# etl_press_intensity.py is league-scoped; run it per eligible Matches/<league> + Players/<league> folder.
python3 etl_press_intensity.py --match-dir "Matches/<league>" --player-dir "Players/<league>" --out "Matches/__merged__/<LEAGUE_TAG>__merged.csv"
python3 build_merged.py --all --recursive --rolling-press
python3 patch_merge_add_streaks.py
# team_ratings.py is league-scoped; run it once per merged league.
python3 team_ratings.py --league "<League Name>" --mode rolling
python3 patch_merge_add_power_ratings.py
python3 make_fd_odds_enriched_synth.py --emit-ou25-novig
python3 patch_merge_add_synth_odds.py --root "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" --overwrite --harmonize-duplicates
python3 pipeline_qa_gate.py
```

Hard stop if integrity fails.

### 2. Generate model outputs

Only after integrity passes:

```bash
python3 bookie_allmarkets.py
python3 deploy_rulebook.py
python3 slip_formatter.py
```

`OBSERVE` remains non-deployable.

### 3. Generate website intelligence beside model outputs

Run the website/premium intelligence exporters against the same active fixture window:

```bash
python3 publish_fixture_intelligence.py
python3 team_rating_engine.py --config ratings_publish_sources.json
python3 player_rating_engine.py --config ratings_publish_sources.json
python3 fixture_lineup_intelligence_engine.py --config ratings_publish_sources.json --fixture-feed frontend/public/data/fixture_intelligence_public.json
python3 fixture_h2h_support_engine.py --fixture-feed frontend/public/data/fixture_intelligence_public.json
python3 fixture_decision_reconciler.py
python3 fixture_preview_generator.py
```

### 3.5 Generate injury shock and lineup risk sidecar

Run this before manual deploy review whenever API injuries/player stats or a curated context file are available:

```bash
python3 injury_shock_engine.py \
  --fixtures-csv data_sources/api_football/normalized/fixtures_master__<LEAGUE_TAG>__<SEASON>.csv \
  --injuries-csv data_sources/api_football/normalized/injuries__<LEAGUE_TAG>__<SEASON>.csv \
  --player-stats-csv data_sources/api_football/normalized/match_player_stats__<LEAGUE_TAG>__<SEASON>.csv \
  --context-csv docs/INJURY_SHOCK_CONTEXT_FLAGS_TEMPLATE.csv \
  --output-csv reports/latest/injury_shock_engine_<WINDOW>/INJURY_SHOCK_BOARD.csv \
  --output-md reports/latest/injury_shock_engine_<WINDOW>/INJURY_SHOCK_BOARD.md
```

This is a warning sidecar only until token-level backtesting proves which flags should affect deployment. It should surface `ATTACK_SHOCK`, `DEFENCE_SPINE_SHOCK`, `MOTIVATION_VOLATILITY`, and `REQUIRE_LINEUP_CONFIRMATION` for the pre-slip review queue.

Then apply/verify snapshot metadata:

```bash
python3 scripts/apply_snapshot_metadata_to_publish_estate.py
python3 scripts/audit_pre_kickoff_intelligence_snapshots.py
```

### 4. Export website database

```bash
python3 scripts/export_site_sqlite.py
```

The D1/SQLite route layer should read compact page-shaped payloads, not raw historical source files.

## What Deploy Review Should Compare

For each fixture, review:

- model deploy state and tier
- EV state
- market intelligence state
- team ratings: Goal Heat, BTTS Pressure, Attack Flow, Defensive Lock, First Strike, Chaos
- player event shortlist strength
- lineup status: predicted, confirmed, or unavailable
- H2H availability and fallback mode
- weather/environment context
- timestamp metadata

## Cost-Control Principle

Pre-launch, publish only:

- active competitions
- current season
- active fixture window
- compact fixture/team/player route payloads

Keep deeper historical rows for high-tier/pro routes or local analysis until traffic and query cost justify widening.

## Website Results Rule

The public Results page should only update after:

```bash
python3 grade_weekend_results.py --src <settled deploy scored file>
python3 validate_weekly_results.py
```

Current intelligence scoring reports are analysis artifacts until promoted through the results-publish flow.
