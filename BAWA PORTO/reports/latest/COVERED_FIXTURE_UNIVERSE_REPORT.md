# COVERED_FIXTURE_UNIVERSE_REPORT

Generated: `2026-05-09T02:33:05+00:00`
Source run id: `2026-05-09`
Source window: `2026-05-09` to `2026-05-11`

## Source Files
- `allmarkets`: `predictions_output/2026-05-09/BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11.csv`
- `deploy_candidates_raw`: `predictions_output/2026-05-09/DEPLOY_CANDIDATES_RAW.csv`
- `deploy_candidates_after_gates`: `predictions_output/2026-05-09/DEPLOY_CANDIDATES_AFTER_GATES.csv`
- `deploy_elite`: `predictions_output/2026-05-09/BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11__DEPLOY_TIER_ELITE__PRESET_V1__FTR_accuracy.csv`
- `deploy_standard`: `predictions_output/2026-05-09/BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11__DEPLOY_TIER_STANDARD__PRESET_V1__FTR_accuracy.csv`
- `deploy_observe`: `predictions_output/2026-05-09/BOOKIE_IMP20_ALLMARKETS_2026-05-09_to_2026-05-11__DEPLOY_TIER_OBSERVE__PRESET_V1__FTR_accuracy.csv`
- `loss_details`: `predictions_output/2026-05-09/PRE_ALLMARKETS_FIXTURE_LOSS_DETAILS_2026-05-09_to_2026-05-11.csv`
- `loss_report`: `predictions_output/2026-05-09/PRE_ALLMARKETS_FIXTURE_LOSS_REPORT_2026-05-09_to_2026-05-11.csv`

## Coverage Summary
- Total fixtures: `156`
- Routed fixtures: `132`
- Non-routed fixtures: `24`
- Hidden fixtures: `0`
- Covered leagues: `22`

## Availability Counters
- `availability:goal_shape_base`: `132`
- `availability:prematch_odds`: `132`
- `availability:routed_deploy`: `26`
- `availability:routed_observe`: `131`
- `availability:team_stats`: `9`
- `logo_join:FULL_MATCH`: `103`
- `logo_join:NO_MATCH`: `29`
- `logo_join:PARTIAL_MATCH`: `24`
- `logo_manifest:global_alias_assets`: `1276`
- `logo_manifest:global_team_assets`: `1294`
- `logo_manifest:league_assets`: `19`
- `logo_manifest:league_team_assets`: `1619`
- `logo_manifest:loaded`: `1`
- `logo_manifest:scoped_alias_assets`: `1753`
- `loss_details:fixture_added_to_universe`: `24`
- `loss_details:fixture_already_in_universe`: `15`
- `routing_status:non_routed`: `24`
- `routing_status:routed`: `132`
- `source_rows:allmarkets`: `378`
- `source_rows:deploy_candidates_after_gates`: `132`
- `source_rows:deploy_candidates_raw`: `132`
- `source_rows:deploy_elite`: `6`
- `source_rows:deploy_observe`: `131`
- `source_rows:deploy_standard`: `21`
- `source_rows:loss_details_not_emitted`: `39`

## Notes
- Builder uses the current-window ALLMARKETS file as the primary live fixture intake base.
- The pre-ALLMARKETS loss detail report fills upstream fixtures that never emitted into ALLMARKETS.
- Historical normalized API-Football fixture master is joined opportunistically when identity matches exist.
- This builder is an intake artifact for later CONTEXT and MONITOR classification.
