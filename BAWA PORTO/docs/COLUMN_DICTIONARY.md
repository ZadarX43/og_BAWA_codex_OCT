# COLUMN_DICTIONARY

## Purpose
Canonical high-value column guide for the current Odds Genius production path.

This dictionary focuses on columns most relevant to:
- canonical merged training inputs
- all-markets prediction output
- deploy routing
- slip formatting

## Canonical Training Input
Source:
- `Matches/__merged__/<LEAGUE_TAG>__merged.csv`

Minimum families enforced by `pipeline_qa_gate.py`:
- core: `fixture_key`, `home_team_name`, `away_team_name`, `status`
- FTR odds: `odds_ft_home_team_win`, `odds_ft_draw`, `odds_ft_away_team_win`
- BTTS odds: `odds_btts_yes`, `odds_btts_no`
- OU25 odds: `odds_ft_over25`, `odds_ft_under25`

## Identity Columns
- `fixture_key`: canonical fixture join key used widely across overlays and routing
- `league`: competition name
- `match_date`: normalized fixture date
- `home_team_name`, `away_team_name`: canonical team labels

## Market Columns
- `market`: `ftr`, `btts`, `ou25`, plus some research / secondary lanes
- `selection`: routed selection for the row
- `bookie_pick`: bookmaker-facing selected side for the row
- `model_top_pick`: model-preferred side where applicable

## Price / Probability Columns
- `bookie_od`: decimal odds used for the routed pick
- `bookie_implied`: raw implied probability from odds
- `bookie_implied_used`: effective implied probability used downstream
- `bookie_implied_novig`: no-vig implied probability where available
- `model_p_for_bookie`: model probability for the selected bookmaker side
- `gap`: simple model minus bookie gap
- `gap_novig`: model minus no-vig implied gap
- `model_strength`: bundle validation strength from source model

## FTR-Specific Columns
- `ftr_margin`: separation between top FTR probabilities
- `power_diff`: home minus away power rating differential
- `bookie_spread`: FTR spread helper
- `pick_side_mass_top3`: top-3 correct-score mass supporting selected FTR side
- `pick_side_margin_top3`: selected-side top-3 support minus strongest competing side

## BTTS-Specific Columns
- `signal_btts`: signal label such as `STRONG_YES` / `STRONG_NO`
- `p_home_fts`, `p_away_fts`: estimated fail-to-score probabilities
- `p00_est`: estimated 0-0 probability
- `btts_alignment`: alignment marker between model and selection

## OU25-Specific Columns
- `signal_over25`: OU25 signal label such as `STRONG_OVER`
- `ou25_policy_state`: `live` or `review` posture used downstream
- `ou25_over_struct_pass`: structural pass flag
- `bookie_lambda_total_fit`: lambda fitted from odds / totals
- `exp_goals_sum`: expected total goals proxy
- `top3_over_count`: count of top-3 correct scores that imply over 2.5 goals

## Overlay / Context Columns
- `context_reason_codes`: pipe-delimited reasons and context tags
- `reason_codes`: secondary reason field when present
- `team_intel_overlay_action`: team-intelligence action
- `team_intel_overlay_policy_bucket`: overlay policy grouping

## Deploy / Tier Columns
- `deploy_tier`: final routed tier (`ELITE`, `STANDARD`, `OBSERVE`)
- `tier`: synchronized alias of `deploy_tier`
- `standard_reporting_bucket`: standard-lane subgroup used by reporting and slips
- `profit_first_keep`: optional keep flag used by some slip/product filters
- `profit_first_review_flag`, `profit_first_review_reason`: review-only product metadata

## Slip Formatter Columns
Consumed directly by `slip_formatter.py`:
- `fixture_key`
- `league`
- `home_team_name`, `away_team_name`
- `match_date`
- `market`
- `selection`
- `bookie_pick`
- `model_top_pick`
- `model_p_for_bookie`
- `bookie_od`
- `deploy_tier`
- `standard_reporting_bucket`
- `context_reason_codes`

## QA Columns
Used by `pipeline_qa_gate.py` reports:
- `<column>__nonnull`
- baseline row counts from merge reports
- duplicate counts from dedupe reports

## Important Conventions
- `OBSERVE` rows are non-deployable even if other fields look attractive.
- `selection` and `bookie_pick` are often equivalent but both should be preserved.
- Signal labels are not the same thing as deploy reason tokens.
- No-vig fields should be preferred where explicitly available.

## Test Commands
Integrity gate:
```bash
./.venv/bin/python pipeline_qa_gate.py
```

All-markets sample schema inspection:
```bash
python3 - <<'PY'
import pandas as pd
p='predictions_output/<DATE>/<ALLMARKETS_FILE>.csv'
df=pd.read_csv(p)
print(sorted(df.columns.tolist()))
PY
```

## Risks / TODOs
- A fuller field-by-field schema should eventually be generated directly from canonical merged samples and live output samples.
- Some older scripts still use variant column names and require alias handling.
