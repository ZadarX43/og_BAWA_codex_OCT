# Untracked Docs / Research Classification - 2026-05-23

## Purpose

Classify the remaining untracked docs and research/script sprawl after the generated payload cleanup batches.

This is a classification pass only. It does not move, delete, or promote production behavior.

## Current Snapshot

- Total dirty entries after payload cleanup: `723`
- Untracked docs: `74`
- Untracked `scripts/` entries: `168`
- Untracked root-level files: `459`
- Protected production spine: clean

## Keep / Commit

These are useful product, pipeline, website, or system-contract assets and should be reviewed for a deliberate commit batch.

### Canonical Docs

Commit or merge into existing canonical docs:

- `docs/AGREEMENT_SCORE_SPEC.md`
- `docs/API_FOOTBALL_PLAN.md`
- `docs/API_FOOTBALL_SCHEMA.md`
- `docs/BTTS_RULEBOOK.md`
- `docs/COLUMN_DICTIONARY.md`
- `docs/DEPLOY_RULEBOOK.md`
- `docs/DESKTOP_APP_WORKFLOW.md`
- `docs/FIXTURE_BRAIN_SUMMARY_GENERATOR_CONTRACT.md`
- `docs/HYBRID_CONSENSUS_LAYER_SPEC.md`
- `docs/HYBRID_FEATURE_SYSTEM_SPEC.md`
- `docs/HYBRID_SIDE_MARKETS_PLAN.md`
- `docs/INJURY_SHOCK_COVERAGE_SCAN_RUNBOOK.md`
- `docs/INJURY_SHOCK_ENGINE_SPEC_2026-05-16.md`
- `docs/MODELSTORE_CONTRACT.md`
- `docs/PHASE_LOCKS_AND_BASELINES.md`
- `docs/PIPELINE_RUNBOOK.md`
- `docs/PLAYER_EVENTS_API_FOOTBALL_PLAN.md`
- `docs/PLAYER_EVENTS_FRAMEWORK.md`
- `docs/PLAYER_EVENTS_PRE_KICKOFF_READINESS_CHECKLIST.md`
- `docs/PLAYER_PROP_BETA.md`
- `docs/PREDICTION_OVERLAY.md`
- `docs/PREMIUM_PAYLOAD_ALLOWLIST.md`
- `docs/PRE_DEPLOY_INTELLIGENCE_RUN_ORDER_2026-05-13.md`
- `docs/PRE_KICKOFF_SNAPSHOT_DISCIPLINE_2026-05-13.md`
- `docs/PUBLIC_EXPLANATION_STYLE_GUIDE.md`
- `docs/REASON_TOKENS.md`
- `docs/REFEREE_PROFILE_ENGINE_SPEC_2026-05-17.md`
- `docs/SLIP_ENGINE.md`
- `docs/SYSTEM_MAP.md`
- `docs/TEAM_PLAYER_RATINGS_ENGINE_SPEC.md`
- `docs/TRAINING_PIPELINE.md`
- `docs/WEATHER_SYSTEM.md`
- `docs/WEBSITE_10K_USER_ARCHITECTURE_PLAN.md`
- `docs/WEBSITE_FULL_HANDOFF_2026-05-12.md`
- `docs/WEBSITE_LAUNCH_PLAN_2026-06-01.md`
- `docs/WORLD_CUP_INSTITUTIONAL_MODEL_BENCHMARK_AND_EDGE_MAP_2026-05-19.md`

CSV/template docs that can be committed if still used by scripts:

- `docs/INJURY_SHOCK_CONTEXT_FLAGS_TEMPLATE.csv`
- `docs/PLAYER_EVENTS_INPUT_SCHEMA.csv`
- `docs/PLAYER_EVENTS_PRE_KICKOFF_READINESS_FIELDS.csv`

### API-Football Foundation Scripts

Keep/commit as a coherent package, excluding `.bak_*` files:

- `scripts/api_football/__init__.py`
- `scripts/api_football/client.py`
- `scripts/api_football/config.py`
- `scripts/api_football/paths.py`
- `scripts/api_football/raw_helpers.py`
- `scripts/api_football/schema_contracts.py`
- `scripts/api_football/team_name_map.py`
- `scripts/api_football/utils.py`
- `scripts/api_football/fetch_*.py`
- `scripts/api_football/normalize_*.py`
- `scripts/api_football/build_*features.py`
- `scripts/api_football/build_current_injury_lineup_window_manifest.py`
- `scripts/api_football/refresh_current_injury_lineup_window.py`
- `scripts/api_football/run_api_foundation.py`
- `scripts/api_football/leakage_checks.py`
- `scripts/api_football/feature_family_stacks.py`
- `scripts/api_football/hybrid_training_utils.py`

Reason:

- This is the local API-football source/normalization/feature foundation.
- It feeds player, team, fixture, injury, lineup, H2H, and odds intelligence.
- It should be tracked if it is part of the future brain/orchestrator spine.

### Site / Brain / Overlay Scripts

Keep/commit after quick smoke/import checks:

- `scripts/analyze_pre_kickoff_intelligence_signals.py`
- `scripts/audit_pre_kickoff_intelligence_snapshots.py`
- `scripts/audit_walkforward_intelligence_estate.py`
- `scripts/build_api_context_overlay_board.py`
- `scripts/build_api_intelligence_overlay_audit.py`
- `scripts/build_fixture_identity_map.py`
- `scripts/build_fixture_market_intelligence_board.py`
- `scripts/build_fixture_market_outcome_tracker.py`
- `scripts/build_fixture_summary_dry_run.py`
- `scripts/build_goal_market_overlay_support_audit.py`
- `scripts/build_goal_market_signal_matrix.py`
- `scripts/build_injury_shock_coverage_scan.py`
- `scripts/build_injury_shock_market_impact_sidecar.py`
- `scripts/build_player_event_current_board_fixture_inputs.py`
- `scripts/build_player_event_live_feature_join.py`
- `scripts/build_player_event_live_interaction_features.py`
- `scripts/build_player_event_market_coverage_audit.py`
- `scripts/build_player_event_tactical_registry_overlay.py`
- `scripts/build_tactical_feature_registry.py`
- `scripts/build_team_goals_15_shortlist.py`
- `scripts/build_team_shots_profile.py`
- `scripts/compare_live_deploy_with_site_intelligence.py`
- `scripts/score_cross_layer_intelligence_filters.py`
- `scripts/score_weekend_deploy_tier_results.py`
- `scripts/score_weekend_intelligence_audit.py`
- `scripts/run_walkforward_intelligence_overnight.sh`
- `scripts/player_events/`

Reason:

- These are close to the current website brain, player-event, injury shock, and overlay systems.
- They should be reviewed for imports, usage paths, and naming before commit.

### World Cup Build Scripts

Keep/commit as a World Cup research/build package after one naming pass:

- `scripts/build_world_cup_2026_launch_scaffold.py`
- `scripts/build_world_cup_data_foundation.py`
- `scripts/build_world_cup_api_backfill_manifest.py`
- `scripts/build_world_cup_api_footystats_bridge.py`
- `scripts/build_world_cup_recent_history_and_h2h_sidecar.py`
- `scripts/build_world_cup_venue_weather_travel_scaffold.py`
- `scripts/build_world_cup_player_event_fixture_inputs.py`
- `scripts/build_world_cup_player_intelligence_scaffold.py`
- `scripts/build_world_cup_intelligence_overlay_matrix.py`
- `scripts/build_world_cup_modelling_target_harness.py`
- `scripts/report_world_cup_2026_coverage_gaps.py`
- `scripts/audit_api_football_world_cup_coverage.py`
- `scripts/audit_footystats_world_cup_drop.py`
- `scripts/audit_world_cup_additions_coverage.py`
- `scripts/audit_world_cup_api_historical_backfill_readiness.py`
- `scripts/audit_world_cup_player_event_source_coverage.py`
- `scripts/fetch_api_football_world_cup_players.py`
- `scripts/inspect_world_cup_historical_source_requirements.py`

Reason:

- These line up with the revised June 7 launch and World Cup readiness work.
- They are research/build tooling, not production prediction routing.

## Archive / Review

These may contain useful thinking, but should not be committed blindly into the active repo surface.

### Docs To Merge Or Archive

- root markdown cleanup/index/restructure docs:
  - `docs/ROOT_MARKDOWN_*`
  - `docs/ROOT_RESTRUCTURE_MOVE_MAP.md`
  - `docs/DOCS_MERGE_REPORT.md`
- repo hygiene historical drafts:
  - `docs/REPO_HYGIENE_*`
- one-off run logs and dated summaries:
  - `docs/DATA_REFRESH_RUN_LOG_2026-05-14.md`
  - `docs/WEEKEND_INTELLIGENCE_OVERLAY_RUN_LOG_2026-05-14.md`
  - `docs/WEEKEND_PREDICTION_INTELLIGENCE_SCORING_2026-05-13.md`
  - `docs/weekend_prediction_summary_pack_2026_05_15_to_2026_05_19/`
- old handoff/freeze material:
  - `docs/WORKING_STATE_FREEZE_2026-04-17.md`
  - `docs/API_FOOTBALL_HYBRID_PROGRESS_SUMMARY_2026-04-28.md`
  - `docs/ACTIVE_SITE_INTELLIGENCE_PUBLISH_ESTATE_COMPLETION_2026-05-12.md`

Action:

- Keep only if they are still referenced from current runbooks.
- Otherwise move to an archive branch or `docs/archive/` in a separate commit.

### One-Off Research Scripts

Archive/review before committing:

- dated Chelsea/Spurs research:
  - `scripts/audit_chelsea_spurs_2026_05_19_predictions.py`
  - `scripts/build_chelsea_spurs_research_preview_2026_05_19.py`
  - `scripts/build_chelsea_spurs_upgraded_player_event_interaction_2026_05_19.py`
- phase/recovery experiments:
  - `scripts/audit_phase8h_c4_window_stability.py`
  - `scripts/audit_phase8h_c4d_shadow_stage_stability.py`
  - `scripts/build_phase8h_replay_sweeps.py`
  - `scripts/classify_phase8h_c4_recovery_rings.py`
  - `scripts/simulate_phase8h_c3_policy.py`
  - `scripts/simulate_phase8h_live_research_order.py`
- team-goal/combo research that is not launch-critical:
  - `scripts/audit_team_goal_combo_c4_proof_window_stability.py`
  - `scripts/build_team_goal_combo_c4_full_estate_validation.py`
  - `scripts/build_team_goal_shadow_market_backtest.py`
  - `scripts/build_team_goal_threshold_stability_report.py`
  - `scripts/run_team_goal_combo_live_shadow_qa.py`
- FTR/BTTS/OU25 discovery scripts that are separate from live deploy:
  - `scripts/audit_ftr_btts_combo_window_stability.py`
  - `scripts/build_ftr_btts_combo_discovery_audit.py`
  - `scripts/build_ftr_c5_side_shape_ring_classifier.py`
  - `scripts/build_ftr_loss_autopsy.py`
  - `scripts/build_ftr_shadow_throttle_backtest.py`
  - `scripts/run_ftr_btts_combo_live_shadow_qa.py`
  - `scripts/run_ou25_c4_live_shadow_repeat_qa.py`
- broad World Cup experiments:
  - `scripts/backtest_world_cup_player_event_markets.py`
  - `scripts/build_footystats_world_cup_model_matrix.py`
  - `scripts/build_footystats_world_cup_research_foundation.py`
  - `scripts/build_world_cup_2026_player_event_board.py`
  - `scripts/build_world_cup_2026_player_power_projection_board.py`
  - `scripts/build_world_cup_2026_research_feature_matrix.py`
  - `scripts/build_world_cup_historical_full_stack_backbuild.py`
  - `scripts/build_world_cup_initial_goal_market_prediction_list.py`
  - `scripts/build_world_cup_selective_overlay_audit.py`
  - `scripts/build_world_cup_selective_policy_simulator.py`
  - `scripts/run_world_cup_research_model_ablation.py`
  - `scripts/run_world_cup_trainer_native_research.py`

Action:

- Review each for dependencies and output paths.
- Promote only the scripts that are part of the central brain/orchestrator contract.
- Archive dated or superseded experiments.

## Ignore / Generated

These should not be committed as source.

### Backup / Patch Files

- `scripts/api_football/*.bak_*`
- root `*.bak_*`
- `*.patch_*`
- `deploy_rulebook_backup.py`
- `deploy_gates_backup.py`
- `deploy_rulebook_research_*.py`
- `bookie_allmarkets_research_*.py`

Action:

- Archive if needed for forensic history.
- Otherwise ignore via a dedicated cleanup commit only after confirming no current script imports them.

### Generated CSV / TXT Reports

Examples:

- root `*_audit.csv`
- root `*_comparison.csv`
- root `*_manifest.csv`
- root `*_summary.csv`
- `mypy_report.txt`
- `pyflakes_report.txt`
- `MODELSTORE_*_DIRLIST.txt`
- `tmp_*`

Action:

- Do not commit as source.
- If the content matters, regenerate through scripts and publish into `reports/` or a dated artifact archive.

### Root-Level Script Sprawl

The 459 root-level untracked files are mostly old research scripts, notes, CSV reports, backups, and duplicates. They should not be mass-committed.

Initial policy:

- Keep root production spine protected.
- Move active scripts into `scripts/` only after review.
- Move documentation into `docs/` only after dedupe.
- Archive old root research/output files in a separate branch or archive commit.

## Recommended Next Batches

1. Commit this classification report.
2. Commit canonical docs package.
3. Commit API-football foundation package, excluding backups.
4. Commit site/brain/player-event/injury overlay scripts after import checks.
5. Commit World Cup build package after naming/dependency check.
6. Archive/review old one-off research and root-level sprawl.
7. Add narrow ignore rules only for confirmed generated report/backup patterns.

## Safety Notes

- Do not move or modify protected production spine files during these cleanup batches.
- Do not commit generated payloads or raw data dumps.
- Do not promote research scripts into production runbooks until their input/output contracts are documented.
- Prefer small commits with clear batch names.
