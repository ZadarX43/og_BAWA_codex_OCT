# Weather Stadium Registry

This is the starter workflow for football weather enrichment.

The goal is simple:

- build a stable stadium registry once
- verify coordinates once
- then attach weather by `stadium_id` instead of fuzzy text every time

## Why This Exists

For weather research we need a canonical venue layer before we do any backfills.

That means owning:

- `league`
- `home_team_name`
- `stadium_name`
- `latitude`
- `longitude`
- `timezone`

The registry is the join surface for:

- historical weather backfills
- weekend fixture weather enrichment
- upset / volatility mapping
- OU25 / BTTS weather context
- later player-props context

## Current Script

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/build_stadium_registry.py`

It scans:

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/Matches/__merged__`

and builds a starter registry from:

- `league`
- `home_team_name`
- `stadium_name`
- fixture dates

## Outputs

Running the script writes:

- `STADIUM_REGISTRY__STARTER.csv`
- `STADIUM_REGISTRY__REVIEW_QUEUE.csv`
- `STADIUM_REGISTRY__SUMMARY.md`

Default output folder:

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/reports/<today>/WEATHER_REGISTRY__V1`

## Run Command

```bash
python3 '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/build_stadium_registry.py'
```

Optional custom output:

```bash
python3 '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/build_stadium_registry.py' \
  --outdir '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/reports/2026-04-23/WEATHER_REGISTRY__V1'
```

## What The Starter Registry Gives Us

Each row includes:

- `stadium_id`
- `league`
- `competition_scope`
- `home_team_name`
- `stadium_name`
- `city`
- `country_name`
- `country_code`
- `timezone`
- `latitude`
- `longitude`
- `weather_location_query`
- `fixture_count`
- `is_primary_home_venue`
- `multi_venue_team_flag`
- `coord_confidence`
- `notes`

## What Still Needs Manual Verification

The starter file is intentionally conservative.

You should still review:

1. `latitude`
2. `longitude`
3. `timezone`
4. teams with multiple venues
5. international competition rows
6. stadiums where city could not be inferred from the venue name

## Suggested Manual Workflow

1. Build the starter registry.
2. Work through `STADIUM_REGISTRY__REVIEW_QUEUE.csv`.
3. Fill verified venue coordinates and timezone.
4. Save the cleaned result as the canonical `stadium_registry.csv`.
5. Use that file for weather backfills and live weekend enrichment.

## Weather Fields We Expect To Attach Later

Recommended fixture-level weather fields:

- `weather_kickoff_utc`
- `weather_temp_c_mean`
- `weather_temp_c_min`
- `weather_temp_c_max`
- `weather_relative_humidity_pct`
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
- `weather_source`

## Next Practical Step

Once the registry is verified, the next script should be:

- `weather_backfill_audit.py`

That script should:

- join historical fixtures to `stadium_id`
- fetch hourly weather around kickoff
- save per-fixture weather summaries
- let us audit:
  - upsets in bad weather
  - BTTS / OU25 behavior in rain or wind
  - whether certain leagues are weather-sensitive
