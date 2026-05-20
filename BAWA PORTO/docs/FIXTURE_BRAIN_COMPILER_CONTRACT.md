# Fixture Brain Compiler Contract

The fixture brain compiler turns local site intelligence into compact per-fixture payloads for the website/Worker layer.

It does not create predictions, change deploy routing, or touch `deploy_rulebook.py`.

## Command

```bash
python3 scripts/site_publish/fixture_brain_compiler.py --from-date 2026-05-20 --days 14
```

Useful options:

- `--all-fixtures`
- `--fixture-key 2026_05_09_Stuttgart_Bayer_Leverkusen`
- `--output-dir build/site_brain/current`
- `--injury-fixture-csv reports/latest/injury_shock_coverage_scan/INJURY_SHOCK_COVERAGE_SCAN.csv`
- `--injury-player-csv reports/latest/injury_shock_coverage_scan/INJURY_SHOCK_PLAYER_IMPACT.csv`

## Inputs

- `build/site_data/odds_genius.sqlite`
- `reports/latest/injury_shock_coverage_scan/INJURY_SHOCK_COVERAGE_SCAN.csv`
- `reports/latest/injury_shock_coverage_scan/INJURY_SHOCK_PLAYER_IMPACT.csv`
- `reports/latest/injury_shock_coverage_scan/SUNDAY_2026_05_17_INJURY_LINEUP_IMPACT.csv`
- `frontend/public/data/internal/injury_shock_admin_dashboard.json`

## Output Shape

Each fixture payload is written to:

```text
build/site_brain/current/payloads/fixtures/<fixture_key>.json
```

Top-level sections:

- `fixture_core`
- `market_cards`
- `decision`
- `h2h`
- `weather`
- `team_context`
- `player_context`
- `lineup_context`
- `injury_context`
- `fixture_stats`
- `tier_visibility`
- `freshness`
- `coverage`
- `source_refs`

`market_cards` keeps the actual all-markets model output when available. TG1.5 remains support-only until odds/model output exists.

`injury_context` is research-only and includes fixture shock scores, warning tokens, matched player impacts, and source references.

## Publish Rule

Cloudflare should receive compact fixture-brain payloads plus a tiny index/manifest, not raw player/team/match source tables.

The compiler writes:

- `build/site_brain/current/manifest.json`
- `reports/latest/FIXTURE_BRAIN_COMPILER_REPORT.json`
- `reports/latest/FIXTURE_BRAIN_COMPILER_REPORT.md`

The manifest hashes each fixture payload so the publish layer can upload changed fixtures only.
