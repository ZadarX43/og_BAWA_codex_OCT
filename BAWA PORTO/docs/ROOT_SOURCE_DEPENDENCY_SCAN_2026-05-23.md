# Root Source Dependency Scan - 2026-05-23

Purpose: classify the remaining untracked root-level Python/config/shell sprawl without touching the protected production spine or moving source-like files blindly.

## Scope

- Root-level untracked source/config/shell files scanned: 323
- Python files: 290
- Config files: 13
- Shell runners: 20
- Full machine-readable classification: `docs/ROOT_SOURCE_DEPENDENCY_SCAN_2026-05-23.csv`

## Bucket Counts

- `archive_review_research_harness`: 169
- `archive_review_scratch_backup`: 12
- `archive_review_shell_runner`: 18
- `keep_commit_candidate`: 34
- `keep_or_review_imported`: 21
- `needs_manual_review_config`: 1
- `needs_manual_review_source`: 68

## Strong Keep / Commit Candidates

These files are either known support contracts/configs/runners or are directly related to website/pipeline validation. They should be inspected and committed in small themed groups, not archived.

- `btts_league_policy.json` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `btts_no_live_allowlist.json` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `constants.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `correct_score_product.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `deploy_gates.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `deploy_presets.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `deploy_weekend_runner.sh` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `game_context.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `injury_shock_engine.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `manifest_and_calendar_flags.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `market_cs_proxy_promotion_config.json` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `market_proxy_league_config.json` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `market_proxy_promotion_config.json` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `mypy.ini` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `odds_synth.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `og_model_paths.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `ou25_league_policy.json` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `over25_deploy_policy.json` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `over25_from_ou25_policy.json` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `player_prop_models.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `player_usage_profiles.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `prediction_overlay.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `prediction_report.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `pyproject.toml` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `ratings_publish_sources.sample.json` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `rebuild_all_merged.sh` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `seasonal_market_ledger.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `signal_layers.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `streaks_module.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `team_rating_engine.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `travel_fatigue.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `uefa_context.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `under25_btts_no_support_allowlist.json` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.
- `weather_data.py` - Named production/support contract, config policy, or operational runner; inspect then track in a focused commit.

## Imported Source That Needs Owner Review

These files are imported by other local Python files, so they are not safe to archive until we confirm whether the importing path is still active or historical.

- `_baseline_ftr_pipeline.py` - 9 refs. Imported by: probe_goal_model.py; weekend_smoke_test.py; build_ftr_apples_to_apples_eval.py; shadow_goal_rebuild_runner.py; train_investor_leagues_v2.py; prediction_overlay.py; prediction_overlay copy.py; side_prob_models.py
- `acca_builder.py` - 1 refs. Imported by: slip_score_report.py
- `backtest_deploy_csv.py` - 1 refs. Imported by: scripts/_archive_review/2026_05_23/build_phase8h_replay_sweeps.py
- `batch_run_hybrid_upstream_pipeline.py` - 1 refs. Imported by: batch_fetch_api_historical_context.py
- `btts_over25_pipeline.py` - 2 refs. Imported by: btts_over25_predict.py; BAWA-PORTO-recovery-2025-08-24-2308/btts_over25_predict.py
- `build_hybrid_consensus_inputs.py` - 2 refs. Imported by: batch_run_hybrid_upstream_pipeline.py; batch_build_hybrid_consensus_inputs.py
- `build_hybrid_goal_mass_inputs.py` - 1 refs. Imported by: batch_run_hybrid_upstream_pipeline.py
- `build_hybrid_threshold_policy.py` - 2 refs. Imported by: batch_run_hybrid_upstream_pipeline.py; batch_build_hybrid_threshold_policies.py
- `deploy_rulebook_research_phase8h_c4_shadow.py` - 2 refs. Imported by: scripts/_archive_review/2026_05_23/run_ou25_c4_live_shadow_repeat_qa.py; scripts/_archive_review/2026_05_23/build_live_shadow_research_dashboard.py
- `ftr_consensus.py` - 1 refs. Imported by: backtest_ftr_consensus.py
- `leak_tests.py` - 11 refs. Imported by: train_draw_classifier.py; -recover_rebuilt_.py; _baseline_ftr_pipeline.py; BAWA-PORTO-recovery-2025-08-24-2308/train_draw_classifier.py; BAWA-PORTO-recovery-2025-08-24-2308/00_baseline_ftr_pipeline.py; tests/test_no_leak.py; MAIN BACKUP BACKUPS/_baseline_ftr_pipeline - august 24, 25 9pm.py; MAIN BACKUP BACKUPS/_baseline_ftr_pipeline.py
- `run_pre_post_bypass_compare.py` - 1 refs. Imported by: run_pre_post_overlay_compare.py
- `run_walkforward_windows.py` - 2 refs. Imported by: run_walkforward_windows_research_harness.py; backfill_walkforward_exact_scores.py
- `side_prob_models.py` - 10 refs. Imported by: -recover_rebuilt_.py; weekend_bets.py; _baseline_ftr_pipeline.py; prediction_overlay.py; prediction_overlay copy.py; BAWA-PORTO-recovery-2025-08-24-2308/00_baseline_ftr_pipeline.py; MAIN BACKUP BACKUPS/_baseline_ftr_pipeline - august 24, 25 9pm.py; MAIN BACKUP BACKUPS/_baseline_ftr_pipeline.py
- `train_draw_classifier.py` - 4 refs. Imported by: -recover_rebuilt_.py; _baseline_ftr_pipeline.py; MAIN BACKUP BACKUPS/_baseline_ftr_pipeline.py; BAWA-PORTO-recovery-2025-08-24-2308/ModelStore/England_Premier_League/_baseline_ftr_pipeline.py
- `train_hybrid_goal_mass.py` - 1 refs. Imported by: batch_run_hybrid_upstream_pipeline.py
- `train_hybrid_ou25_tuned.py` - 1 refs. Imported by: batch_run_hybrid_upstream_pipeline.py
- `train_hybrid_side_markets.py` - 1 refs. Imported by: batch_run_hybrid_upstream_pipeline.py
- `train_investor_leagues_v2.py` - 3 refs. Imported by: build_ftr_xgb_hyperparam_sweep.py; build_ftr_apples_to_apples_eval.py; build_ftr_xgb_feature_pruning_audit.py
- `train_markets.py` - 7 refs. Imported by: bookie_allmarkets.py; bookie_allmarkets_research_no_recent_btts_regime.py; bookie_allmarkets_research_pre_hybrid_modelstore.py; threshold_calibration.py; backup_pre_rebuild_2026-03-16/bookie_allmarkets.py; BAWA-PORTO-recovery-2025-08-24-2308/threshold_calibration.py; scripts/run_world_cup_trainer_native_research.py
- `weekend_smoke_test.py` - 2 refs. Imported by: meta_builder.py; weekend_bets.py

## Top Imported Root Modules

- `prediction_overlay.py` - 48 refs; sample: -recover_rebuilt_.py; side_meta_backtest.py; bookie_allmarkets.py; 94%_train_ftr_with_team_stats 2.py; margin_sweep_ftr.py; train_ftr_with_team_stats.py; run_fast_ftr_predictions.py; top50_check.py
- `constants.py` - 31 refs; sample: train_draw_classifier.py; -recover_rebuilt_.py; bookie_allmarkets.py; bookie_allmarkets_research_no_recent_btts_regime.py; weekend_smoke_test.py; ftr_meta_backtest.py; debug_draw.py; _baseline_ftr_pipeline.py
- `prediction_report.py` - 16 refs; sample: -recover_rebuilt_.py; 94%_train_ftr_with_team_stats 2.py; train_ftr_with_team_stats.py; btts_over25_predict.py; 94%_train_ftr_with_team_stats.py; _baseline_ftr_pipeline.py; BAWA-PORTO-recovery-2025-08-24-2308/94%_train_ftr_with_team_stats 2.py; BAWA-PORTO-recovery-2025-08-24-2308/train_ftr_with_team_stats.py
- `leak_tests.py` - 11 refs; sample: train_draw_classifier.py; -recover_rebuilt_.py; _baseline_ftr_pipeline.py; BAWA-PORTO-recovery-2025-08-24-2308/train_draw_classifier.py; BAWA-PORTO-recovery-2025-08-24-2308/00_baseline_ftr_pipeline.py; tests/test_no_leak.py; MAIN BACKUP BACKUPS/_baseline_ftr_pipeline - august 24, 25 9pm.py; MAIN BACKUP BACKUPS/_baseline_ftr_pipeline.py
- `side_prob_models.py` - 10 refs; sample: -recover_rebuilt_.py; weekend_bets.py; _baseline_ftr_pipeline.py; prediction_overlay.py; prediction_overlay copy.py; BAWA-PORTO-recovery-2025-08-24-2308/00_baseline_ftr_pipeline.py; MAIN BACKUP BACKUPS/_baseline_ftr_pipeline - august 24, 25 9pm.py; MAIN BACKUP BACKUPS/_baseline_ftr_pipeline.py
- `streaks_module.py` - 10 refs; sample: bookie_allmarkets.py; bookie_allmarkets_research_no_recent_btts_regime.py; weekend_smoke_test.py; patch_merge_add_streaks.py; build_merged.py; backtest_ftr_consensus.py; _baseline_ftr_pipeline.py; bookie_allmarkets_research_pre_hybrid_modelstore.py
- `_baseline_ftr_pipeline.py` - 9 refs; sample: probe_goal_model.py; weekend_smoke_test.py; build_ftr_apples_to_apples_eval.py; shadow_goal_rebuild_runner.py; train_investor_leagues_v2.py; prediction_overlay.py; prediction_overlay copy.py; side_prob_models.py
- `signal_layers.py` - 7 refs; sample: bookie_allmarkets.py; bookie_allmarkets_research_no_recent_btts_regime.py; _baseline_ftr_pipeline.py; bookie_allmarkets_research_pre_hybrid_modelstore.py; prediction_overlay.py; v2_window_signals.py; backup_pre_rebuild_2026-03-16/bookie_allmarkets.py
- `train_markets.py` - 7 refs; sample: bookie_allmarkets.py; bookie_allmarkets_research_no_recent_btts_regime.py; bookie_allmarkets_research_pre_hybrid_modelstore.py; threshold_calibration.py; backup_pre_rebuild_2026-03-16/bookie_allmarkets.py; BAWA-PORTO-recovery-2025-08-24-2308/threshold_calibration.py; scripts/run_world_cup_trainer_native_research.py
- `og_model_paths.py` - 5 refs; sample: bookie_allmarkets.py; bookie_allmarkets_research_no_recent_btts_regime.py; bookie_allmarkets_research_pre_hybrid_modelstore.py; pack_builder.py; backup_pre_rebuild_2026-03-16/bookie_allmarkets.py
- `train_draw_classifier.py` - 4 refs; sample: -recover_rebuilt_.py; _baseline_ftr_pipeline.py; MAIN BACKUP BACKUPS/_baseline_ftr_pipeline.py; BAWA-PORTO-recovery-2025-08-24-2308/ModelStore/England_Premier_League/_baseline_ftr_pipeline.py
- `uefa_context.py` - 4 refs; sample: bookie_allmarkets.py; bookie_allmarkets_research_no_recent_btts_regime.py; bookie_allmarkets_research_pre_hybrid_modelstore.py; backup_pre_rebuild_2026-03-16/bookie_allmarkets.py
- `deploy_gates.py` - 3 refs; sample: deploy_rulebook.py; deploy_rulebook_research_no_ou25_hybrid_meta.py; deploy_rulebook_research_pre_hybrid_modelstore.py
- `train_investor_leagues_v2.py` - 3 refs; sample: build_ftr_xgb_hyperparam_sweep.py; build_ftr_apples_to_apples_eval.py; build_ftr_xgb_feature_pruning_audit.py
- `btts_over25_pipeline.py` - 2 refs; sample: btts_over25_predict.py; BAWA-PORTO-recovery-2025-08-24-2308/btts_over25_predict.py
- `build_hybrid_consensus_inputs.py` - 2 refs; sample: batch_run_hybrid_upstream_pipeline.py; batch_build_hybrid_consensus_inputs.py
- `build_hybrid_threshold_policy.py` - 2 refs; sample: batch_run_hybrid_upstream_pipeline.py; batch_build_hybrid_threshold_policies.py
- `correct_score_product.py` - 2 refs; sample: build_correct_score_backtest_summary.py; build_correct_score_deploy.py
- `deploy_rulebook_research_phase8h_c4_shadow.py` - 2 refs; sample: scripts/_archive_review/2026_05_23/run_ou25_c4_live_shadow_repeat_qa.py; scripts/_archive_review/2026_05_23/build_live_shadow_research_dashboard.py
- `run_walkforward_windows.py` - 2 refs; sample: run_walkforward_windows_research_harness.py; backfill_walkforward_exact_scores.py

## Archive / Review Candidates

These are mostly audit, backtest, training, walk-forward, frozen/sandbox, shell-runner, or backup/scratch files. The next cleanup batch can move them into `archive/review` only after checking that none are referenced by current docs/runbooks.

- `-recover_rebuilt_.py` - `archive_review_scratch_backup`
- `aggregate_gated_backtests.py` - `archive_review_research_harness`
- `apply_frozen_product_gates.py` - `archive_review_research_harness`
- `apply_frozen_specialist_profiles.py` - `archive_review_research_harness`
- `audit_backtest_run.py` - `archive_review_research_harness`
- `audit_btts_full_universe_scenarios.py` - `archive_review_research_harness`
- `audit_btts_proxy_feature_ablation_from_a.py` - `archive_review_research_harness`
- `audit_btts_proxy_feature_ablation_from_allmarkets.py` - `archive_review_research_harness`
- `audit_btts_proxy_lift_from_allmarkets.py` - `archive_review_research_harness`
- `audit_btts_scenario_grid.py` - `archive_review_research_harness`
- `audit_btts_shadow_bucket.py` - `archive_review_research_harness`
- `audit_ftr_dual_paths.py` - `archive_review_research_harness`
- `audit_market_proxy_delta.py` - `archive_review_research_harness`
- `audit_merged_side_coverage.py` - `archive_review_research_harness`
- `audit_ou25_btts_support_experiment.py` - `archive_review_research_harness`
- `audit_ou25_fullgrid_signal.py` - `archive_review_research_harness`
- `audit_ou25_proxy_feature_ablation_from_a.py` - `archive_review_research_harness`
- `audit_ou25_proxy_feature_ablation_from_allmarkets.py` - `archive_review_research_harness`
- `audit_ou25_proxy_lift_from_allmarkets.py` - `archive_review_research_harness`
- `audit_power_freshness.py` - `archive_review_research_harness`
- `audit_snapshot_proxy_feature_coverage.py` - `archive_review_research_harness`
- `audit_snapshot_proxy_feature_coverage_v2_batch.py` - `archive_review_research_harness`
- `backfill_walkforward_exact_scores.py` - `archive_review_research_harness`
- `backtest_ftr_consensus.py` - `archive_review_research_harness`
- `backtest_ftr_from_merged.py` - `archive_review_research_harness`
- `backtest_ou25_from_merged.py` - `archive_review_research_harness`
- `backtest_uefa_ab.py` - `archive_review_research_harness`
- `batch_build_hybrid_consensus_inputs.py` - `archive_review_research_harness`
- `batch_build_hybrid_threshold_policies.py` - `archive_review_research_harness`
- `batch_fetch_api_historical_context.py` - `archive_review_research_harness`
- `bookie_allmarkets_proxy_experiment.py` - `archive_review_research_harness`
- `bookie_allmarkets_research_no_recent_btts_regime.py` - `archive_review_scratch_backup`
- `bookie_allmarkets_research_pre_hybrid_modelstore.py` - `archive_review_scratch_backup`
- `BTTS_WALKFORWARD_SUMMARY.py` - `archive_review_research_harness`
- `build_acca_extension_audit.py` - `archive_review_research_harness`
- `build_batch_join_quality_audit.py` - `archive_review_research_harness`
- `build_branch_comparison.py` - `archive_review_research_harness`
- `build_branch_cumulative_stats.py` - `archive_review_research_harness`
- `build_btts_deploy_policy.py` - `archive_review_research_harness`
- `build_btts_from_walkforward_audit.py` - `archive_review_research_harness`
- `build_btts_model_vs_valueev_comparison.py` - `archive_review_research_harness`
- `build_btts_walkforward_league_audit.py` - `archive_review_research_harness`
- `build_btts_xgb_consensus_audit.py` - `archive_review_research_harness`
- `build_correct_score_backtest_summary.py` - `archive_review_research_harness`
- `build_correct_score_deploy.py` - `archive_review_research_harness`
- `build_cs_market_bucket_features.py` - `archive_review_research_harness`
- `build_ftr_apples_to_apples_eval.py` - `archive_review_research_harness`
- `build_ftr_btts_ou25_rescue_audit.py` - `archive_review_research_harness`
- `build_ftr_combo_rescue_audit.py` - `archive_review_research_harness`
- `build_ftr_consensus_elite_policy.py` - `archive_review_research_harness`
- `build_ftr_league_calibration.py` - `archive_review_research_harness`
- `build_ftr_league_weighting.py` - `archive_review_research_harness`
- `build_ftr_meta_inject_into_tiers.py` - `archive_review_research_harness`
- `build_ftr_meta_priority_walkforward_audit.py` - `archive_review_research_harness`
- `build_ftr_meta_super_score_walkforward.py` - `archive_review_research_harness`
- `build_ftr_meta_vs_consensus_summary.py` - `archive_review_research_harness`
- `build_ftr_priority_walkforward_audit.py` - `archive_review_research_harness`
- `build_ftr_validation_backtests_3y19l.sh` - `archive_review_shell_runner`
- `build_ftr_xgb_consensus_elite_audit.py` - `archive_review_research_harness`
- `build_ftr_xgb_feature_pruning_audit.py` - `archive_review_research_harness`
- `build_ftr_xgb_hyperparam_sweep.py` - `archive_review_research_harness`
- `build_ftr_xgb_side_by_side_audit.py` - `archive_review_research_harness`
- `build_full_walkforward_backtest_summary.py` - `archive_review_research_harness`
- `build_fullgrid_cs_market_bucket_features.py` - `archive_review_research_harness`
- `build_fullgrid_cs_market_shape_features.py` - `archive_review_research_harness`
- `build_hybrid_benchmark_scoreboard_v1.py` - `archive_review_research_harness`
- `build_hybrid_benchmark_scoreboard_v2.py` - `archive_review_research_harness`
- `build_hybrid_estate_scorecard.py` - `archive_review_research_harness`
- `build_hybrid_parity_review.py` - `archive_review_research_harness`
- `build_learned_gates.py` - `archive_review_research_harness`
- `build_monster_release_gate_audit.py` - `archive_review_research_harness`
- `build_months.sh` - `archive_review_shell_runner`
- `build_months_2022_2024.sh` - `archive_review_shell_runner`
- `build_months_specialists.sh` - `archive_review_shell_runner`
- `build_ou25_branch_comparison.py` - `archive_review_research_harness`
- `build_ou25_cumulative_stats.py` - `archive_review_research_harness`
- `build_ou25_investor_summary.py` - `archive_review_research_harness`
- `build_ou25_rulebook_audit.py` - `archive_review_research_harness`
- `build_ou25_walkforward_comparison.py` - `archive_review_research_harness`
- `build_ou25_walkforward_league_audit.py` - `archive_review_research_harness`
- `build_ou25_xgb_consensus_audit.py` - `archive_review_research_harness`
- `build_over25_from_ou25_audit.py` - `archive_review_research_harness`
- `build_over25_model_vs_filtered_ou25_comp.py` - `archive_review_research_harness`
- `build_phase7c_cross_market_rescue_audit.py` - `archive_review_research_harness`
- `build_player_prop_beta_board.py` - `archive_review_research_harness`
- `build_product_branch_comparison.py` - `archive_review_research_harness`
- `build_product_branch_cumulative_stats.py` - `archive_review_research_harness`
- `build_same_fixture_composite_audit.py` - `archive_review_research_harness`
- `build_same_fixture_composite_from_walkforward.py` - `archive_review_research_harness`
- `build_slip_audit_dataset.py` - `archive_review_research_harness`
- `build_snapshot_proxy_features_from_merge.py` - `archive_review_research_harness`
- `build_snapshot_proxy_features_from_merged_v2_batch.py` - `archive_review_research_harness`
- `build_stadium_registry.py` - `archive_review_research_harness`
- `build_team_snapshot_matchup_features.py` - `archive_review_research_harness`
- `build_team_snapshot_matchup_features_all.py` - `archive_review_research_harness`
- `build_uefa_team_map.py` - `archive_review_research_harness`
- `build_under25_btts_no_audit.py` - `archive_review_research_harness`
- `build_validated_league_manifest.py` - `archive_review_research_harness`
- `build_weakest_failed_leg_audit.py` - `archive_review_research_harness`
- `compare_goal_manifest_sets.py` - `archive_review_research_harness`
- `compare_rulebook_agg.py` - `archive_review_research_harness`
- `debug_draw.py` - `archive_review_research_harness`
- `debug_home_away_swap.py` - `archive_review_research_harness`
- `deploy_gates_backup.py` - `archive_review_scratch_backup`
- `deploy_rulebook_backup.py` - `archive_review_scratch_backup`
- `deploy_rulebook_research_no_ou25_hybrid_meta.py` - `archive_review_scratch_backup`
- `deploy_rulebook_research_pre_hybrid_modelstore.py` - `archive_review_scratch_backup`
- `eval_ftr_consensus.py` - `archive_review_research_harness`
- `fit_draw_prone.py` - `archive_review_research_harness`
- `fit_signal_bands.py` - `archive_review_research_harness`
- `forensic_ou25_audit.py` - `archive_review_research_harness`
- `forensic_sideproduct_walkforward_audit.py` - `archive_review_research_harness`
- `forensic_walkforward_audit.py` - `archive_review_research_harness`
- `ftr_meta_backtest.py` - `archive_review_research_harness`
- `FTR_MetaPredictor.py` - `archive_review_research_harness`
- `ftr_variant_tests.py` - `archive_review_research_harness`
- `ge2_defence_candidate_rule_table.py` - `archive_review_research_harness`
- `ge2_defence_deploy_triage.py` - `archive_review_research_harness`
- `leak_probes.py` - `archive_review_research_harness`
- `make_feb_sandbox_modelstore.sh` - `archive_review_shell_runner`
- ...plus 79 more in `ROOT_SOURCE_DEPENDENCY_SCAN_2026-05-23.csv`

## Manual Review

- ` run_fast_ftr_predictions.py` - `needs_manual_review_source`
- `94%_train_ftr_with_ROI_reports.py` - `needs_manual_review_source`
- `94%_train_ftr_with_team_stats 2.py` - `needs_manual_review_source`
- `94%_train_ftr_with_team_stats.py` - `needs_manual_review_source`
- `apply_btts_proxy_overlay_from_config.py` - `needs_manual_review_source`
- `apply_ou25_proxy_overlay_from_config.py` - `needs_manual_review_source`
- `apply_specialist_deploy_config.py` - `needs_manual_review_source`
- `apply_specialist_overlay_bridge.py` - `needs_manual_review_source`
- `apply_stadium_coord_overrides.py` - `needs_manual_review_source`
- `attack_identity_audit.py` - `needs_manual_review_source`
- `attack_identity_refined_audit.py` - `needs_manual_review_source`
- `auto_leagues.py` - `needs_manual_review_source`
- `auto_leagues.py.py` - `needs_manual_review_source`
- `bookie_allmarkets_btts_proxy_wrapper.py` - `needs_manual_review_source`
- `bookie_allmarkets_ou25_proxy_wrapper.py` - `needs_manual_review_source`
- `BTTS_MODEL_VS_VALUEEV_POLICY_NOTE.py` - `needs_manual_review_source`
- `btts_over25 match goals.py` - `needs_manual_review_source`
- `btts_over25_match_goals.py` - `needs_manual_review_source`
- `btts_over25_predict.py` - `needs_manual_review_source`
- `check_ftr_v2_mapping_all.py` - `needs_manual_review_source`
- `config.yaml.py` - `needs_manual_review_source`
- `conflict_zones.py` - `needs_manual_review_source`
- `cross_audit_shortlist.py` - `needs_manual_review_source`
- `cs_signal_audit.py` - `needs_manual_review_source`
- `deploy_top_slips.py` - `needs_manual_review_source`
- `extract_predictions.py` - `needs_manual_review_source`
- `fetch_api_league_catalog.py` - `needs_manual_review_source`
- `fill_stadium_registry_coords.py` - `needs_manual_review_source`
- `filter_deploy_by_rulebook.py` - `needs_manual_review_source`
- `fix_names.py` - `needs_manual_review_source`
- `ftr_accuracy_report.py` - `needs_manual_review_source`
- `goal_bias_holdout_eval.py` - `needs_manual_review_source`
- `inject_btts_proxy_columns_from_merged.py` - `needs_manual_review_source`
- `inject_ou25_proxy_columns_from_merged.py` - `needs_manual_review_source`
- `league_competition_audit.py` - `needs_manual_review_source`
- `live_feature_signal_audit.py` - `needs_manual_review_source`
- `make_league_odds_enriched_synth.py` - `needs_manual_review_source`
- `merge_fd_odds_into_matches.py` - `needs_manual_review_source`
- `merged_pipeline_dashboard.py` - `needs_manual_review_source`
- `model_coverage_report.py` - `needs_manual_review_source`
- `modelstore_goal_ensemble_audit.py` - `needs_manual_review_source`
- `modelstore_goal_ensemble_audit_shadow.py` - `needs_manual_review_source`
- `OG : BAWA WEEKEND RUNBOOK April_26.py` - `needs_manual_review_source`
- `og_deploy_rulepack.py` - `needs_manual_review_source`
- `og_goal_tracker.py` - `needs_manual_review_source`
- `og_public_match_cards.py` - `needs_manual_review_source`
- `og_xray.py` - `needs_manual_review_source`
- `pack_builder.py` - `needs_manual_review_source`
- `patch_merge_add_fd_odds.py` - `needs_manual_review_source`
- `patch_merge_add_legacy_derived.py` - `needs_manual_review_source`
- `patch_overrides_missing_teams.py` - `needs_manual_review_source`
- `poisson_source_audit.py` - `needs_manual_review_source`
- `promote_shadow_goal_ensembles_to_prod.py` - `needs_manual_review_source`
- `README.ini` - `needs_manual_review_config`
- `retrain-over25_expanded.py` - `needs_manual_review_source`
- `score_public_match_cards.py` - `needs_manual_review_source`
- `segment_stats.py` - `needs_manual_review_source`
- `segment_stats_full.py` - `needs_manual_review_source`
- `slip_reconstruct.py` - `needs_manual_review_source`
- `slip_score_report.py` - `needs_manual_review_source`
- `tests:test_no_leak.py` - `needs_manual_review_source`
- `tests:test_poisson_cutoff.py` - `needs_manual_review_source`
- `threshold_calibration.py` - `needs_manual_review_source`
- `tier_audit.py` - `needs_manual_review_source`
- `top50_check.py` - `needs_manual_review_source`
- `v2_window_signals.py` - `needs_manual_review_source`
- `walk_forward_runner.py` - `needs_manual_review_source`
- `weather_backfill_audit.py` - `needs_manual_review_source`
- `weekend_bets.py` - `needs_manual_review_source`

## Recommended Next Cleanup Order

1. Commit keep candidates by product family: core support modules, public proof/export validators, fixture intelligence validators, then configs.
2. For `keep_or_review_imported`, inspect the importing files first. If the import path is historical only, archive both sides together.
3. Move archive/review research harnesses into a dated archive folder in batches by family: FTR, BTTS, OU25, hybrid, walk-forward, shell runners.
4. Add ignore rules only for repeatable generated outputs, not source-like scripts.
5. Keep production spine clean throughout; this scan made no production-spine edits.

## Notes

- Python AST parse errors encountered: 21. Most are malformed scratch files, old notes saved as `.py`, or filenames with awkward syntax history.
- The scan intentionally did not move or modify root source files. It only classifies them so the next action is controlled.

