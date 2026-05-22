# PREDICTION_OVERLAY

## Purpose
Code-derived documentation for `prediction_overlay.py`.

`prediction_overlay.py` is the feature-and-overlay layer that sits between canonical data and final prediction/deploy preparation. It attaches power ratings, synth odds, calibration, signal bands, derived probabilities, and optional market overlays.

## Inputs
Typical input frame:
- fixture-level pre-match dataframe produced from merged/enriched match sources

Frequently expected columns:
- identity: `league`, `fixture_key`, `match_date`, `home_team_name`, `away_team_name`
- pricing: market odds / implied columns
- model probabilities: `p_model`, `model_p_for_bookie`, per-market probabilities
- correct-score / lambda fields for derived overlays
- optional raw columns from merged sources used for feature alignment

External dependencies:
- `ModelStore/` bundles and market calibrators
- market threshold JSONs
- draw-threshold JSONs
- rolling power ratings from `team_ratings.py`

## Outputs
Typical added / mutated columns:
- power fields: `home_power_rating`, `away_power_rating`, `power_diff`
- signal fields: `signal_over25`, `signal_btts`, related runtime/side labels
- calibrated probabilities
- FTR draw-mix / margin outputs
- inferred odds / synth odds helpers
- scenario support / deployability helpers

This file also writes optional:
- confidence summaries
- ROI / P&L CSVs
- betslips / reports in older workflows

## Key Functions
Core attach / enrichment:
- `attach_team_power_ratings_if_available(...)`
- `attach_signal_layers_if_available(...)`
- `attach_synth_odds(...)`
- `attach_ftr_with_draw_mix(...)`
- `attach_decisive_over_btts_scores(...)`
- `attach_multi_book_odds(...)`
- `attach_clv(...)`
- `attach_win_to_nil_proxy(...)`

Model / threshold loading:
- `_load_signal_band_config(...)`
- `load_market_thresholds_for_league(...)`
- `_load_draw_threshold(...)`
- `_load_market_model(...)`
- `score_trained_markets_if_available(...)`
- `enrich_with_models_or_odds(...)`

Deploy preparation / validation:
- `prepare_deployable_overlay(...)`
- `validate_market_coherence(...)`
- `ensure_minimal_signals(...)`
- `log_resolved_odds_columns(...)`

Utility / keying:
- `_match_key(...)`
- `_coalesce_match_date_series(...)`
- `align_features_for_sklearn(...)`
- `apply_safe_renames_and_whitelist(...)`

## Input Columns of Interest
Common columns referenced across overlay logic:
- `fixture_key`
- `match_date`
- `league`
- odds columns for FTR / BTTS / OU25
- probability columns per market
- CS / goal-mass / entropy fields
- feature-family columns from merged data

## Output Columns of Interest
Representative overlay outputs:
- `home_power_rating`, `away_power_rating`, `power_diff`
- `ftr_margin`
- `signal_over25`, `signal_btts`
- threshold flags / draw-threshold flags
- canonical `p_model` / `od` helper columns in older composer flows

## Gates / Vetoes
This file is mostly additive, but it still contains important constraints:
- secondary scores must never overwrite core model probabilities unless explicitly designed to do so
- draw threshold handling can flag risky rows
- market coherence validation can detect inconsistent score/probability shapes
- feature alignment protects inference by filling missing expected columns safely

## Reason Tokens / Labels
This module emits more labels than deploy tokens. Common signal labels include:
- `STRONG_OVER`, `VERY_STRONG_OVER`, `WEAK_OVER`
- `STRONG_YES`, `VERY_STRONG_YES`, `WEAK_YES`
- `STRONG_NO`, `VERY_STRONG_NO`, `WEAK_NO`
- draw / threshold diagnostic labels

It also supplies upstream context later consumed by `deploy_rulebook.py`.

## Protected Rules
- Additive overlays must not silently override core deploy routing.
- Synthetic / inferred odds should remain explicit and auditable.
- Fallback or rescue probabilities must not masquerade as first-class model outputs.
- Value or scenario layers must remain secondary to deploy gates.

## Known Negative Controls
- missing feature columns are aligned rather than causing unsafe inference drift
- post-match / debug columns are excluded from strict pre-match rating attachment
- market coherence checks are used to spot broken probability shapes

## Test Commands
Smoke / overlay validation:
```bash
python3 - <<'PY'
import prediction_overlay as po
po.run_overlay_smoke_tests()
PY
```

Typical integrated use is through:
```bash
./.venv/bin/python bookie_allmarkets.py --date-from <FROM> --date-to <TO> --strict --debug
```

## Risks / TODOs
- The file currently carries a very broad responsibility set.
- Some older writer/report paths overlap with newer deploy-only workflows.
- This layer will likely be one of the biggest beneficiaries of API-Football enrichment later.
