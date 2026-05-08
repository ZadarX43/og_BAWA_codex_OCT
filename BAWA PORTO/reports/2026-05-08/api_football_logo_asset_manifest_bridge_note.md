# API_FOOTBALL_LOGO_ASSET_MANIFEST Bridge Note

Date: 2026-05-08

## Purpose

Add website badge/logo enrichment as a presentation layer only.

This does not touch:
- `bookie_allmarkets.py`
- `deploy_rulebook.py`
- `slip_formatter.py`
- live routing, tiers, gates, vetoes, or value policy

## Built

- `scripts/api_football/build_logo_asset_manifest.py`
- `data_sources/api_football/normalized/api_football_logo_asset_manifest.csv`
- `frontend/public/data/api_football_logo_asset_manifest.json`

The manifest is built from existing raw API-Football fixture JSONL files, so it uses no new API calls.

Extracted fields include:
- league ID / name / country / season
- league logo URL
- league flag URL
- team ID / name / season
- team logo URL
- first seen fixture ID
- appearance count

## Publisher Enrichment

`publish_predictions.py` now joins logo fields into public and premium website JSON:

- `home_team_logo_url`
- `away_team_logo_url`
- `league_logo_url`
- `league_flag_url`
- `logo_join_status`

Join logic uses:
- direct league/team normalized matching
- approved `configs/team_name_join_map.generated.csv`
- approved `configs/team_name_join_map.csv`
- global team-logo fallback when the logo identity is unambiguous

## Frontend Display

`frontend/assets/app.js` now renders:
- home badge
- away badge
- league badge where available

`frontend/assets/styles.css` adds badge styling and initials fallback.

If a logo is unavailable or fails to load, the UI falls back to club initials.

## Weekend Board QA

Manifest build:
- raw fixture files scanned: `55`
- fixture rows read: `21,239`
- league assets written: `54`
- team assets written: `3,621`

Weekend publish QA:
- public records: `6`
- premium records: `28`
- premium logo status:
  - `FULL_MATCH`: `26`
  - `PARTIAL_MATCH`: `1`
  - `NO_MATCH`: `1`

Remaining misses are source-coverage gaps in the current cached API-Football raw estate:
- `Remo` partial match: Palmeiras/league found, Remo missing
- `Al Ittihad v Dhamk` no match: Saudi source logos not present in current manifest cache

## Verification

Passed:
- `python3 -m py_compile scripts/api_football/build_logo_asset_manifest.py publish_predictions.py validate_public_export.py`
- `node --check frontend/assets/app.js`
- `.venv/bin/python validate_public_export.py`

## Later Hardening

Do not download/cache remote logo images until API-Sports asset terms are checked.

Short term:
- use API-Sports media URLs directly
- keep initials fallback
- publish logo join status in summaries

Long term:
- cache approved images under `frontend/public/assets/...`
- add asset freshness checks
- add a missing-logo review queue for newly promoted/current-season teams
