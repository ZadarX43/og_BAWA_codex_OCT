# API_FOOTBALL_PLAN

## Purpose
Build API-Football as an additive enrichment layer alongside the existing FootyStats pipeline.

This layer is intentionally parallel-first:
- do not replace FootyStats yet
- do not alter existing production model files
- do not promote API-enriched features until baseline-vs-enriched audits prove value

## Mission
Create normalized API-Football source tables and derived feature tables that can be joined onto existing Odds Genius fixture inputs for controlled audits across:
- FTR
- BTTS
- BTTS NO
- OU25
- Team Goals Home O1.5
- Team Goals Away O1.5
- Correct Score
- Player Shots
- Player SOT
- Player Fouls
- Player Tackles
- Player Bookings

## Folder Layout
```text
data_sources/api_football/
  raw/
  normalized/
  features/
reports/api_football/
scripts/api_football/
```

## Execution Phases
### Phase 1: Foundation
Build the scaffolding now, without live keys:
- folder structure
- config loader
- endpoint map
- path helpers
- normalized-table builder stubs
- feature-builder stubs
- audit stubs
- leakage-check framework

### Phase 2: Normalized Source Tables
Priority order:
1. `fixtures_master.csv`
2. `match_team_stats.csv`
3. `match_events.csv`
4. `match_player_stats.csv`
5. `lineups.csv`
6. `injuries.csv`
7. `odds_prematch_long.csv`
8. `odds_live_long.csv`
9. `teams_master.csv`
10. `players_master.csv`
11. `bookmaker_map.csv`
12. `bet_market_map.csv`

### Phase 3: Pre-match Feature Layer
Initial outputs:
- `api_team_rolling_features.csv`
- `api_lineup_features.csv`
- `api_injury_features.csv`
- `api_event_features.csv`
- `api_odds_features.csv`
- `api_enriched_fixture_features.csv`

This is the fastest route to useful FTR / BTTS / OU25 / team-goal / correct-score enrichment.

### Phase 4: Player and Live Expansion
Later outputs:
- `api_player_rolling_features.csv`
- `api_live_features.csv`
- `live_minute_15_dataset.csv`
- `live_minute_30_dataset.csv`
- `live_minute_45_dataset.csv`
- `live_minute_60_dataset.csv`
- `live_minute_75_dataset.csv`

## Required Join Keys
Every feature output must preserve:
- `fixture_id`
- `fixture_key`
- `league`
- `league_id`
- `season`
- `match_date`
- `home_team_id`
- `away_team_id`
- `home_team_name`
- `away_team_name`

Canonical fixture key:
```python
fixture_key = f"{match_date}_{home_team_name}_{away_team_name}"
```

Also support a fuzzy FootyStats join helper with:
- date ± 1 day
- normalized home team
- normalized away team
- normalized league name

## Output Targets
### Feature tables
- `data_sources/api_football/features/api_team_rolling_features.csv`
- `data_sources/api_football/features/api_player_rolling_features.csv`
- `data_sources/api_football/features/api_lineup_features.csv`
- `data_sources/api_football/features/api_injury_features.csv`
- `data_sources/api_football/features/api_event_features.csv`
- `data_sources/api_football/features/api_odds_features.csv`
- `data_sources/api_football/features/api_live_features.csv`
- `data_sources/api_football/features/api_enriched_fixture_features.csv`

### Audit outputs
- `reports/api_football/api_feature_coverage_report.csv`
- `reports/api_football/api_missing_values_report.csv`
- `reports/api_football/api_feature_uplift_matrix.csv`
- `reports/api_football/api_footystats_join_audit.csv`
- `reports/api_football/api_odds_market_coverage_report.csv`

## Strict Leakage Rules
Pre-match features must not use:
- current match stats
- current match events
- current match player stats
- post-kickoff odds snapshots
- lineup info not known before kickoff
- injury info not known before kickoff

Live models must remain separate from pre-match datasets.

## Baseline vs Enriched Audit Rule
Do not promote API-enriched model paths unless they improve one or more of:
- accuracy
- ROI
- volume / deployable coverage
- false-positive suppression
- league stability

And do so without breaking protected floors.

## Protected Benchmark Floors
- `BTTS ELITE`: `90.68`
- `FTR ELITE`: `92.64`
- `FTR STANDARD`: `83.36`
- `OU25 STANDARD`: `90.06`
- `BTTS calibrated`: `93.55`
- `OU25 calibrated`: `95.35`

## Promotion Status Labels
- `LOCKED`
- `PROMOTE`
- `WATCH`
- `REJECT`

## Current Foundation Files
Scaffolding created under `scripts/api_football/` includes:
- config / path helpers
- normalized-table builders
- feature-builder stubs
- audit stubs
- leakage-check framework
- a foundation runner

## Next Step When Keys Arrive
1. add env vars or local config for API auth
2. implement fetch layer per endpoint
3. populate normalized tables
4. run join audit
5. build feature tables
6. run baseline-vs-enriched comparison
