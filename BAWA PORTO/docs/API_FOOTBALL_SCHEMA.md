# API_FOOTBALL_SCHEMA

## Purpose
Define the normalized table contracts, key fields, and feature output expectations for the API-Football enrichment layer.

## Normalized Source Tables
### `fixtures_master.csv`
One row per fixture.

Required columns:
- `fixture_id`
- `fixture_key`
- `league`
- `league_id`
- `season`
- `match_date`
- `kickoff_ts_utc`
- `status`
- `home_team_id`
- `away_team_id`
- `home_team_name`
- `away_team_name`
- `venue_id`
- `venue_name`

### `match_team_stats.csv`
One row per fixture/team side.

Required columns:
- `fixture_id`
- `team_id`
- `team_name`
- `is_home`
- `goals_for`
- `goals_against`
- `ht_goals_for`
- `ht_goals_against`
- `shots_total`
- `shots_on_goal`
- `shots_inside_box`
- `shots_outside_box`
- `blocked_shots`
- `possession_pct`
- `passes_total`
- `passes_accurate`
- `corners_for`
- `fouls_for`
- `yellow_cards`
- `red_cards`

### `match_events.csv`
One row per event.

Required columns:
- `fixture_id`
- `event_id`
- `minute`
- `extra_minute`
- `team_id`
- `player_id`
- `event_type`
- `event_detail`
- `is_home`
- `score_home_after`
- `score_away_after`

### `match_player_stats.csv`
One row per fixture/player.

Required columns:
- `fixture_id`
- `player_id`
- `team_id`
- `player_name`
- `position`
- `minutes`
- `started_flag`
- `subbed_on_flag`
- `subbed_off_flag`
- `rating`
- `goals`
- `assists`
- `shots_total`
- `shots_on_target`
- `passes_total`
- `passes_key`
- `passes_accurate`
- `tackles`
- `interceptions`
- `blocks`
- `duels_total`
- `duels_won`
- `dribbles_attempted`
- `dribbles_successful`
- `dribbled_past`
- `fouls_drawn`
- `fouls_committed`
- `yellow_cards`
- `red_cards`
- `saves`
- `goals_conceded`

### `lineups.csv`
One row per fixture/team/player lineup slot.

Required columns:
- `fixture_id`
- `team_id`
- `player_id`
- `player_name`
- `formation`
- `is_starting_xi`
- `position`
- `lineup_known_pre_kickoff_flag`
- `lineup_published_ts_utc`

### `injuries.csv`
One row per player injury / absence record.

Required columns:
- `fixture_id` when fixture-scoped, else nullable
- `team_id`
- `player_id`
- `player_name`
- `absence_type`
- `reason`
- `status`
- `known_pre_kickoff_flag`
- `published_ts_utc`

### `odds_prematch_long.csv`
One row per bookmaker / market / outcome snapshot pre-kickoff.

Required columns:
- `fixture_id`
- `bookmaker_id`
- `bookmaker_name`
- `market_code`
- `market_name`
- `selection_code`
- `selection_name`
- `line_value`
- `odds`
- `snapshot_ts_utc`
- `is_opening`
- `is_latest_pre_kickoff`

### `odds_live_long.csv`
One row per bookmaker / market / outcome snapshot after kickoff.

Required columns:
- `fixture_id`
- `live_minute`
- `bookmaker_id`
- `market_code`
- `selection_code`
- `odds`
- `snapshot_ts_utc`

### Lookup tables
- `teams_master.csv`
- `players_master.csv`
- `bookmaker_map.csv`
- `bet_market_map.csv`

## Feature Table Outputs
### `api_team_rolling_features.csv`
One row per fixture with team-level rolling form, goals, OU, BTTS, halftime, shot, passing, corners, discipline, and volatility features.

### `api_player_rolling_features.csv`
One row per fixture/player candidate with rolling and per-90 player features plus player target labels.

### `api_lineup_features.csv`
One row per fixture with parsed formation, XI aggregate, and shape features.

### `api_injury_features.csv`
One row per fixture with absence counts and severity metrics.

### `api_event_features.csv`
One row per fixture with event-derived rolling timing/volatility features.

### `api_odds_features.csv`
One row per fixture with normalized bookmaker pricing, margin-normalized probabilities, drift, and disagreement fields.

### `api_live_features.csv`
One row per fixture/live snapshot for in-play modeling.

### `api_enriched_fixture_features.csv`
Final pre-match enriched fixture table containing:
- fixture identifiers
- rolling team features
- lineup features
- injury features
- event features
- odds features
- selected targets

## Join Rules
Every feature table must preserve:
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

## Leakage Contract
### Pre-match allowed
- completed historical fixtures only
- prematch odds only
- lineups only if timestamped before kickoff
- injuries only if known before kickoff

### Pre-match forbidden
- current fixture match stats
- current fixture events
- current fixture player stats
- post-kickoff odds

### Live datasets
Separate live datasets must be built for minute checkpoints:
- 15
- 30
- 45
- 60
- 75

## Promotion Audit Contract
Every enriched feature family must later support:
- coverage report
- missing values report
- uplift matrix
- FootyStats join audit
- odds market coverage audit

## Config Contract
The fetch/auth layer should remain pluggable via env vars or local config, e.g.:
- `API_FOOTBALL_BASE_URL`
- `API_FOOTBALL_KEY`
- `API_FOOTBALL_HOST`
- `API_FOOTBALL_LEAGUE_IDS`
- `API_FOOTBALL_SEASONS`

No live key is required for the current scaffolding pass.
