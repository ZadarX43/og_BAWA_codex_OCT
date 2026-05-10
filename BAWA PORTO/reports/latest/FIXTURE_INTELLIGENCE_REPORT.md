# FIXTURE_INTELLIGENCE_REPORT

Generated: `2026-05-10T16:55:25+00:00`
Source run id: `2026-05-09`
Source window: `2026-05-09` to `2026-05-11`

## Source Files
- `predictions_output/2026-05-09/BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11__DEPLOY_TIER_ELITE__PRESET_V1__FTR_accuracy.csv`
- `predictions_output/2026-05-09/BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11__DEPLOY_TIER_STANDARD__PRESET_V1__FTR_accuracy.csv`
- `predictions_output/2026-05-09/BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11__DEPLOY_TIER_OBSERVE__PRESET_V1__FTR_accuracy.csv`
- `covered_universe`: `frontend/public/data/covered_fixture_universe.json`

## Counts
- Source rows read: `333`
- Published fixtures written: `156`

## Publish Class Counts
- `CONTEXT`: `24`
- `DEPLOY`: `26`
- `OBSERVE`: `106`

## Primary Market Counts
- `BTTS`: `39`
- `FTR`: `54`
- `OU25`: `39`

## Fallbacks And Drops
- `context_monitor:context`: `24`
- `kickoff_time:date_only`: `317`
- `kickoff_time:match_date`: `333`
- `logo_join:FULL_MATCH`: `246`
- `logo_join:NO_MATCH`: `52`
- `logo_join:PARTIAL_MATCH`: `35`
- `logo_manifest:global_alias_assets`: `1276`
- `logo_manifest:global_team_assets`: `1294`
- `logo_manifest:league_assets`: `19`
- `logo_manifest:league_team_assets`: `1619`
- `logo_manifest:loaded`: `1`
- `logo_manifest:scoped_alias_assets`: `1753`
- `publish_class:CONTEXT`: `24`
- `publish_class:DEPLOY`: `26`
- `publish_class:OBSERVE`: `106`
- `source_bundle:expanded_from_preset`: `1`
- `source_bundle:files`: `3`
- `source_rows:BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11__DEPLOY_TIER_ELITE__PRESET_V1__FTR_accuracy.csv`: `6`
- `source_rows:BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11__DEPLOY_TIER_OBSERVE__PRESET_V1__FTR_accuracy.csv`: `305`
- `source_rows:BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11__DEPLOY_TIER_STANDARD__PRESET_V1__FTR_accuracy.csv`: `22`

## Notes
- Exporter is additive only and does not alter deploy routing.
- `OBSERVE` rows are translated into public-safe intelligence, never premium picks.
- Non-routed fixtures are classified from the covered universe into `CONTEXT` or `MONITOR` using safe missing-coverage language.
- Run `python3 validate_fixture_intelligence.py` after publishing.
