# Backtest / Walk-Forward Estate Map - 2026-05-23

Purpose: preserve the validation estate that proves model, route, and slip quality over time.

This estate is separate from production prediction logic. It may audit or replay production outputs, but it must not override `deploy_rulebook.py`, `bookie_allmarkets.py`, or `slip_formatter.py`.

## Critical Validation Estate

These scripts are strong preservation candidates because they build or summarize recurring walk-forward/backtest evidence:

- `walk_forward_runner.py`: monthly walk-forward runner and summary generator.
- `run_frozen_walkforward.py`: frozen monthly walk-forward runner.
- `run_frozen_specialist_walkforward.py`: frozen specialist-overlay walk-forward runner.
- `run_slip_walkforward_audit.py`: walk-forward slip audit for the current slip policy stack.
- `build_full_walkforward_backtest_summary.py`: aggregate 3-year walk-forward evidence into one clean summary.
- `backfill_walkforward_exact_scores.py`: enrich historical scored outputs with actual exact goals.
- `forensic_walkforward_audit.py`: forensic review of walk-forward outputs.
- `forensic_sideproduct_walkforward_audit.py`: forensic side-product walk-forward review.
- `build_monster_release_gate_audit.py`: classify windows into release buckets from walk-forward evidence.

## Market-Specific Evidence

These scripts preserve market proof layers and policy evidence:

- FTR: `backtest_ftr_consensus.py`, `backtest_ftr_from_merged.py`, `eval_ftr_consensus.py`, `ftr_accuracy_report.py`, `ftr_meta_backtest.py`
- BTTS: `build_btts_from_walkforward_audit.py`, `build_btts_walkforward_league_audit.py`, `build_btts_model_vs_valueev_comparison.py`, `build_btts_xgb_consensus_audit.py`
- OU25 / goal markets: `backtest_ou25_from_merged.py`, `build_ou25_rulebook_audit.py`, `build_ou25_walkforward_comparison.py`, `build_ou25_walkforward_league_audit.py`, `build_ou25_xgb_consensus_audit.py`, `build_over25_from_ou25_audit.py`, `build_under25_btts_no_audit.py`, `forensic_ou25_audit.py`
- Correct score / composite: `build_correct_score_backtest_summary.py`, `cs_signal_audit.py`, `build_same_fixture_composite_audit.py`, `build_same_fixture_composite_from_walkforward.py`

## Slip / Portfolio Evidence

These scripts protect portfolio construction evidence and should not be casually discarded:

- `build_acca_extension_audit.py`
- `build_slip_audit_dataset.py`
- `build_weakest_failed_leg_audit.py`
- `deploy_top_slips.py`
- `slip_reconstruct.py`
- `slip_score_report.py`
- `run_weekend_portfolio.py`

## ModelStore / Feature / Coverage Audits

These scripts diagnose whether the model estate and feature estate are coherent:

- `audit_merged_side_coverage.py`
- `audit_power_freshness.py`
- `audit_snapshot_proxy_feature_coverage.py`
- `audit_snapshot_proxy_feature_coverage_v2_batch.py`
- `league_competition_audit.py`
- `live_feature_signal_audit.py`
- `modelstore_goal_ensemble_audit.py`
- `modelstore_goal_ensemble_audit_shadow.py`
- `poisson_source_audit.py`
- `weather_backfill_audit.py`

## Caution: Alias / Stub Files

These files compile but appear to be one-line aliases or markdown-renamed stubs. Keep them out of critical commits until reviewed:

- `BTTS_WALKFORWARD_SUMMARY.py`
- `audit_btts_proxy_feature_ablation_from_a.py`
- `audit_ou25_proxy_feature_ablation_from_a.py`
- `backtest_ftr_from_merged.py`
- `walkforward_fullgrid_cs_bucket_promotion.py`

## Recommended Commit Order

1. Track critical walk-forward/backtest spine.
2. Track market-specific evidence by family: FTR, BTTS, OU25/goal markets, correct-score/composite.
3. Track slip/portfolio evidence.
4. Track ModelStore/feature/coverage audits.
5. Archive one-off, duplicate, or stub aliases after confirming no active docs refer to them.

## Verification

The estate scan compiled the candidate Python files with `python3 -m py_compile` on 2026-05-23.

No model training was run. No predictions were run. No ModelStore artifacts were changed.

