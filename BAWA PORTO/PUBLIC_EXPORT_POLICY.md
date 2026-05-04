# PUBLIC_EXPORT_POLICY

Updated: `2026-05-04`

## Purpose

Define the strict public and premium publishing boundary for Odds Genius website data exports.

This policy is fail-closed:
- if a field is not explicitly allowed, it must not be published
- if a value appears unsafe or ambiguous, it must be dropped or replaced with a safer fallback
- model logic remains private

## Source Of Truth

- Source CSVs: newest true `predictions_output/**/DEPLOY_COMBINED_*.csv`
- Exclude scored files such as `DEPLOY_COMBINED_SCORED_*.csv`
- Only `publish_predictions.py` may transform deploy CSV rows into website JSON

## Publishable Products

### Public
Public export is a limited free board.

Rules:
- only website-safe fields
- no raw reason tokens
- no raw deploy logic
- rounded or bucketed confidence/value only
- limited board, not full routing estate

### Premium
Premium export is the richer paid board.

Rules:
- can include more model-facing detail than public
- must still avoid private formulas, file paths, bundle references, and raw gate internals
- reason tokens must be sanitized before publishing

## Exact Public Field Allowlist

- `fixture_id`
- `fixture_key`
- `kickoff_time`
- `league`
- `home_team`
- `away_team`
- `market`
- `pick`
- `confidence_tier`
- `display_confidence`
- `bookie_od`
- `model_prob_display`
- `value_edge_display`
- `short_reason`
- `is_free`

No additional public fields are allowed.

## Exact Premium Field Allowlist

- `fixture_id`
- `fixture_key`
- `kickoff_time`
- `league`
- `home_team`
- `away_team`
- `market`
- `pick`
- `confidence_tier`
- `model_prob`
- `bookie_implied_prob`
- `value_edge`
- `bookie_od`
- `reason_tokens`
- `human_reason`
- `slip_role_hint`
- `safe_for_small_acca_flag`
- `safe_for_large_acca_flag`
- `correct_score_shortlist`
- `premium_tier`

No additional premium fields are allowed.

## Forbidden Field Classes

The following must never be published directly from deploy CSVs into website JSON:

- thresholds and threshold helpers
- veto and gate outputs
- raw feature columns
- bundle/model path references
- filesystem paths
- API keys, env values, secrets
- raw xG, lambda, p00, H2H, streak, or policy fields
- metadata/debug/reporting helper fields that reveal routing internals

Representative blocked substrings:

- `threshold`
- `thr`
- `gate`
- `veto`
- `lambda`
- `p00`
- `meta`
- `support`
- `raw`
- `model_path`
- `bundle`
- `feature`
- `xg`
- `h2h`
- `streak`
- `power_diff`
- `draw_risk`
- `draw_chaos`
- `policy`
- `branch`
- `state`
- `source_path`
- `api`
- `secret`

## Safe Mapping Rules

### Identity
- `fixture_key` comes from source `fixture_key`
- `fixture_id` is derived if no source fixture id exists
- `kickoff_time` comes from `match_date` or fallback equivalent

### Picks
- `market` comes from normalized source `market`
- `pick` comes from `selection`, else `bookie_pick`, else `pick`
- `confidence_tier` comes from normalized `deploy_tier`

### Public display values
- `model_prob_display` is rounded/bucketed only
- `value_edge_display` is rounded/bucketed only
- `display_confidence` is a user-facing bucket label, not a raw score

### Premium values
- `model_prob` can be numeric and rounded
- `bookie_implied_prob` can be numeric and rounded
- `value_edge` can be numeric and rounded
- `reason_tokens` must be sanitized and filtered
- `correct_score_shortlist` may include only redacted top-3 scoreline summaries

## Board Scope Rules

### Premium board
- include deployable rows only
- deployable means `ELITE` or `STANDARD`
- exclude `OBSERVE`

### Public board
- limited free board only
- default posture: `ELITE` rows only
- if no `ELITE` rows exist in the selected source, fallback may use a small `STANDARD` subset and must be recorded in publish summary/report

## Validation Rules

`validate_public_export.py` must fail if:
- required fields are missing
- unexpected fields appear
- public JSON contains forbidden-looking keys
- values contain obvious path/model/env leakage
- critical identity fields are null/blank
- NaN or Infinity values exist

## Operational Note

This policy protects the model edge first.
When in doubt:
- drop the field
- replace with a safer display string
- record the fallback in `publish_summary.json`
