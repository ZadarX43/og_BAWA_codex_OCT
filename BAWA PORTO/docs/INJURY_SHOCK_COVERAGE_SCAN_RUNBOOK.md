# Injury Shock Coverage Scan Runbook

Purpose: recurring pre-deploy injury, lineup, and motivation risk radar across covered leagues.

Standard command:

```bash
python3 scripts/build_injury_shock_coverage_scan.py \
  --outdir reports/latest/injury_shock_coverage_scan \
  --admin-json frontend/public/data/internal/injury_shock_admin_dashboard.json
```

Required outputs:

- `INJURY_SHOCK_COVERAGE_SCAN.csv`
- `INJURY_SHOCK_PLAYER_IMPACT.csv`
- `INJURY_SHOCK_PLAYER_RATING_JOIN_AUDIT.csv`
- `INJURY_SHOCK_LEAGUE_COVERAGE.csv`
- `INJURY_SHOCK_BACKTEST_LINK.csv`
- `INJURY_SHOCK_COVERAGE_SCAN.md`
- `frontend/public/data/internal/injury_shock_admin_dashboard.json`

Guardrail: this is a reporting layer only. It must not mutate `deploy_rulebook.py` or promote vetoes without walk-forward proof.

Fresh source refresh command when leagues are `RED_MISSING_SOURCE`:

```bash
python3 scripts/api_football/refresh_current_context_overlay_window.py \
  --from-date YYYY-MM-DD \
  --to-date YYYY-MM-DD \
  --outdir reports/latest/api_current_context_overlay_window_YYYY_MM_DD_to_YYYY_MM_DD
```

Structured injury availability refresh command for the Injury Shock engine:

```bash
python3 scripts/api_football/refresh_current_injury_lineup_window.py \
  --from-date YYYY-MM-DD \
  --to-date YYYY-MM-DD \
  --injury-query-scopes fixture,league_season,league_date,team_season \
  --include-sidelined \
  --outdir reports/latest/api_current_injury_lineup_window_YYYY_MM_DD_to_YYYY_MM_DD
```

Interpretation:

- `fixture` injury scope is late match confirmation.
- `league_season`, `league_date`, and `team_season` are the early-warning availability layer.
- `availability_first_seen_ts_utc` comes from the persistent local seen registry.
- `fixture_only_late_confirmation_flag=1` means OG only learned it through fixture-level confirmation.

When a large refresh is split into multiple league batches, rebuild the combined manifest before scanning:

```bash
python3 scripts/api_football/build_current_injury_lineup_window_manifest.py \
  --outdir reports/latest/api_current_injury_lineup_window_YYYY_MM_DD_to_YYYY_MM_DD \
  --from-date YYYY-MM-DD \
  --to-date YYYY-MM-DD \
  --injury-query-scopes fixture,league_season,league_date,team_season \
  --include-sidelined \
  --seen-registry-csv data_sources/api_football/availability_seen_registry.csv
```

The refresh runner also rebuilds this combined manifest automatically at the end of each completed run.

The scan only trusts `reports/latest/**/normalized` folders whose parent has `CURRENT_CONTEXT_OVERLAY_MANIFEST.json`.
It also trusts the lighter `CURRENT_INJURY_LINEUP_WINDOW_MANIFEST.json` produced by `refresh_current_injury_lineup_window.py`.
Smoke-test folders are ignored so partial test pulls cannot contaminate production coverage.
