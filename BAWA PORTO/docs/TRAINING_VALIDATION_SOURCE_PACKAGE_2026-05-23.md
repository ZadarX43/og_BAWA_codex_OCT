# Training / Validation Source Package - 2026-05-23

Purpose: track the remaining imported root-level training and validation helpers without changing production prediction or deploy behavior.

This package follows the training guardrails:

- train only from `Matches/__merged__/<LEAGUE_TAG>__merged.csv`
- choose one trainer per run: `train_markets.py` or `train_investor_leagues_v2.py`
- do not mix ModelStore artifact naming schemes
- do not run predictions before integrity checks
- do not let validation helpers override `deploy_rulebook.py`

## Core Training / Validation

These files are the first-class training and validation support package:

- `_baseline_ftr_pipeline.py`: baseline FTR/training helpers and leak stripping used by trainer and inference helpers.
- `train_markets.py`: canonical market trainer option.
- `train_investor_leagues_v2.py`: canonical V2 league trainer option.
- `train_draw_classifier.py`: draw classifier training/helper path used by baseline/recovery scripts.
- `side_prob_models.py`: side-probability support models referenced by baseline/overlay helpers.
- `leak_tests.py`: leakage checks imported by training/baseline smoke paths.
- `weekend_smoke_test.py`: weekend model/prediction validation smoke harness.
- `ftr_consensus.py`: FTR consensus validation helper.
- `run_walkforward_windows.py`: walk-forward validation runner.
- `backtest_deploy_csv.py`: deploy CSV backtest helper used by archived/replay validation paths.

## Hybrid / Research Training Helpers

These are tracked to preserve imported research/training workflows, but they are not the production spine:

- `batch_run_hybrid_upstream_pipeline.py`
- `build_hybrid_consensus_inputs.py`
- `build_hybrid_goal_mass_inputs.py`
- `build_hybrid_threshold_policy.py`
- `train_hybrid_goal_mass.py`
- `train_hybrid_ou25_tuned.py`
- `train_hybrid_side_markets.py`
- `btts_over25_pipeline.py`

## Imported Validation / Portfolio Helpers

These are imported by other validation or archived analysis scripts. They remain separate from production deploy logic:

- `acca_builder.py`
- `run_pre_post_bypass_compare.py`
- `deploy_rulebook_research_phase8h_c4_shadow.py`

## Verification Run

All files in the imported group passed `python3 -m py_compile` on 2026-05-23.

No model training was run. No ModelStore artifacts were created or renamed. No production spine files were edited.

