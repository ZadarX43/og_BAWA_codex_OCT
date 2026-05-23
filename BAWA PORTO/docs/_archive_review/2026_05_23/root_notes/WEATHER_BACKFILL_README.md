# Weather Backfill Audit

This is the first historical weather audit layer for the football stack.

It is designed to:

- join fixtures to `stadium_id`
- fetch historical weather around kickoff
- enrich fixtures with weather summaries
- audit:
  - upset sensitivity
  - BTTS sensitivity
  - OU25 sensitivity
  - favorite failure in bad weather

## Scripts

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/build_stadium_registry.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/weather_data.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/weather_backfill_audit.py`

## Inputs

- merged fixtures:
  - `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/Matches/__merged__`
- stadium registry:
  - `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/reports/2026-04-23/WEATHER_REGISTRY__V1/STADIUM_REGISTRY__STARTER.csv`

## Outputs

The audit writes:

- `WEATHER_BACKFILL__ENRICHED_FIXTURES.csv`
- `WEATHER_BACKFILL__AUDIT_BY_CONDITION.csv`
- `WEATHER_BACKFILL__AUDIT_FLAGS.csv`
- `WEATHER_BACKFILL__LEAGUE_FLAGS.csv`
- `WEATHER_BACKFILL__SUMMARY.md`

## Dry-Run Smoke Test

This checks joins and audit outputs without calling the weather API:

```bash
python3 '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/weather_backfill_audit.py' \
  --stadium-registry '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/reports/2026-04-23/WEATHER_REGISTRY__V1/STADIUM_REGISTRY__STARTER.csv' \
  --outdir '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/reports/2026-04-23/WEATHER_BACKFILL__SMOKE' \
  --max-fixtures 25 \
  --skip-fetch
```

## Live Weather Smoke Test

This attempts a tiny real weather pull:

```bash
python3 '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/weather_backfill_audit.py' \
  --stadium-registry '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/reports/2026-04-23/WEATHER_REGISTRY__V1/STADIUM_REGISTRY__STARTER.csv' \
  --outdir '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/reports/2026-04-23/WEATHER_BACKFILL__LIVE_SMOKE' \
  --max-fixtures 10 \
  --allow-geocode-fallback
```

## Full Historical Backfill

Once the stadium registry has verified coordinates filled in:

```bash
python3 '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/weather_backfill_audit.py' \
  --stadium-registry '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/reports/2026-04-23/WEATHER_REGISTRY__V1/STADIUM_REGISTRY__STARTER.csv' \
  --outdir '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/reports/2026-04-23/WEATHER_BACKFILL__V1'
```

## Current Definitions

- `favorite_side` is the lowest-implied-odds side from `odds_ft_home_team_win`, `odds_ft_draw`, `odds_ft_away_team_win`
- `upset_flag` means a home or away favorite with implied probability `>= 0.50` failed to win
- `draw_upset_flag` means that same strong favorite drew

## Weather Features

Current per-fixture summaries include:

- `weather_temp_c_mean`
- `weather_precip_mm`
- `weather_rain_mm`
- `weather_snow_mm`
- `weather_windspeed_kmh`
- `weather_windgust_kmh`
- `weather_cloud_cover_pct`
- `weather_pressure_hpa`
- `weather_condition`
- `weather_is_wet`
- `weather_is_windy`
- `weather_is_cold`
- `weather_severity_score`

## Important Note

The audit is intended for:

- internal research
- historical pattern finding
- annotation
- soft context

It is not intended yet as a hard live gating layer.
