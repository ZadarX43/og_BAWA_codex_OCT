# CURRENT_CONTEXT_OVERLAY_REFRESH_RUNBOOK

## Purpose

Build an isolated current-window API-Football overlay bundle for non-routed `CONTEXT` / `MONITOR` enrichment without touching production normalized season files.

This path is intended to feed:

- current injuries
- current lineups
- current player-stat availability
- current prematch odds snapshots
- current additive team / matchup / referee overlays

The output bundle is consumed opportunistically by the covered-fixture universe builder when present.

## Output Root

Default output path:

`reports/latest/api_current_context_overlay_window`

Primary artifacts:

- `CURRENT_CONTEXT_OVERLAY_MANIFEST.json`
- `CURRENT_CONTEXT_OVERLAY_BUILD_SUMMARY.csv`
- `CURRENT_CONTEXT_OVERLAY_SUMMARY.json`
- `CURRENT_CONTEXT_OVERLAY_SUMMARY.csv`
- `raw/`
- `normalized/`
- `features/`

## Safe Usage

Run from repo root:

```bash
python3 scripts/api_football/refresh_current_context_overlay_window.py \
  --league-tags "Belgium_Pro,Brazil_Serie_A,Portugal_Liga,USA_MLS" \
  --from-date 2026-05-09 \
  --to-date 2026-05-11
```

Use all default board leagues:

```bash
python3 scripts/api_football/refresh_current_context_overlay_window.py \
  --from-date 2026-05-09 \
  --to-date 2026-05-11
```

Optional safety controls:

- `--max-fixtures-per-league`
- `--chunk-size`
- `--max-pages-per-fixture`
- `--sleep-seconds`
- `--daily-cap`

## What It Builds

### Raw

- fixtures
- fixture bundle
- wrapped fixture statistics
- wrapped fixture events
- injuries
- prematch odds

### Normalized

- `fixtures_master__<TAG>__<SEASON>.csv`
- `lineups__<TAG>__<SEASON>.csv`
- `match_player_stats__<TAG>__<SEASON>.csv`
- `match_team_stats__<TAG>__<SEASON>.csv`
- `match_events__<TAG>__<SEASON>.csv`
- `injuries__<TAG>__<SEASON>.csv`
- `odds_prematch_long__<TAG>__<SEASON>.csv`

### Feature Families

- `api_team_rolling_features__...`
- `api_player_rolling_features__...`
- `api_lineup_features__...`
- `api_injury_features__...`
- `api_event_features__...`
- `api_odds_features__...`
- `api_enriched_fixture_features__...`
- `api_team_identity_features__...`
- `api_matchup_interaction_features__...`
- `api_referee_profile_features__...`

## Intake Behavior

When `CURRENT_CONTEXT_OVERLAY_SUMMARY.json` exists:

- `build_covered_fixture_universe.py` loads it
- covered fixtures gain `source_availability.current_overlay`
- live current-overlay availability can upgrade:
  - `injuries`
  - `lineups`
  - `player_stats`
  - `prematch_odds`
  - `match_events`
  - `team_stats`
- `publish_fixture_intelligence.py` uses current overlay notes ahead of historical fallback for non-routed `CONTEXT`

## Important Boundary

This runner is additive only.

It does **not**:

- touch production normalized season files
- alter deploy routing
- alter `bookie_allmarkets.py`
- alter `deploy_rulebook.py`

It exists purely to improve the non-routed intelligence lane when a fresh current-window overlay bundle is available.
