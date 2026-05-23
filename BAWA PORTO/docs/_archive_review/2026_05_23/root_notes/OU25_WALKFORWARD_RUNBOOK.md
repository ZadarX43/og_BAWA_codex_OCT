# OU25 Walk-Forward Runbook
## Odds Genius — OU2.5 Forward Validation

## Mission

Validate that the strongest OU25 branches discovered on the canonical 19-league / 3-year corpus remain stable under true forward-style month-by-month application.

This phase is not discovery.  
This phase is **forward validation**.

The purpose is to answer:

1. Does the branch still work when applied month-by-month?
2. Which branch is most stable across months?
3. Which branch is best for deploy:
   - premium quality
   - directional OVER-only
   - balanced benchmark
   - scale-friendly branch

---

## Locked candidate branches

These are the OU25 branches now carried into walk-forward validation:

1. `OU25_COMBINED_BASELINE`
2. `OU25_COMBINED_TOPQ_080`
3. `OU25_OVER_ONLY`
4. `OU25_BAND2_178_195`
5. `OU25_BAND1_124_176`

These names must remain stable across the run outputs.

---

## Inputs

This walk-forward phase expects **monthly scored backtest CSVs** already produced by your frozen walk-forward pipeline.

Typical expected input pattern:

- `walkforward_frozen/YYYY-MM/backtest_YYYY-MM.csv`

Examples:

- `walkforward_frozen/2024-08/backtest_2024-08.csv`
- `walkforward_frozen/2024-09/backtest_2024-09.csv`

Each monthly input must be a row-level scored backtest file containing at minimum:

- `league`
- `market`
- `bookie_pick`
- `bookie_od`
- `score`
- `correct`

For OU25 rows specifically:
- `market = ou25`
- `bookie_pick` expected to normalize to `OVER` / `UNDER` or `OVER25` / `UNDER25`

---

## Downstream dependency

This runbook depends on:

- `apply_frozen_product_gates.py`

That script must already support:

- `--include-ou25`
- `--ou25-only`
- `--ou25-pick-mode {combined,over_only,under_only}`
- `--ou25-band1-low`
- `--ou25-band1-high`
- `--ou25-band2-low`
- `--ou25-band2-high`
- `--top-q`
- `--tag`

---

## Output structure

Suggested output root:

- `predictions_output/ou25_walkforward/<RUN_TAG>/`

Inside that:

- one folder per month
- inside each month, one folder per branch

Example:

- `predictions_output/ou25_walkforward/ou25_walkforward_v1/2024-08/ou25_combined_baseline/`
- `predictions_output/ou25_walkforward/ou25_walkforward_v1/2024-08/ou25_combined_topq_080/`
- `predictions_output/ou25_walkforward/ou25_walkforward_v1/2024-08/ou25_mode_over_only/`
- `predictions_output/ou25_walkforward/ou25_walkforward_v1/2024-08/ou25_band2_178_195/`
- `predictions_output/ou25_walkforward/ou25_walkforward_v1/2024-08/ou25_band1_124_176/`

Each branch folder should contain:

- filtered CSV
- markdown summary
- market summary CSV
- league summary CSV
- summary JSON
- tier CSVs

---

## Branch definitions

### 1. Combined baseline
Purpose:
- reference branch
- benchmark against later tighter variants

Parameters:
- `pick-mode = combined`
- `band1 = 1.24–1.72`
- `band2 = 1.82–1.91`
- `top_q = 0.70`

Tag:
- `OU25_COMBINED_BASELINE`

### 2. Combined top-q 0.80
Purpose:
- premium quality branch
- strongest frozen ROI result so far

Parameters:
- `pick-mode = combined`
- `band1 = 1.24–1.72`
- `band2 = 1.82–1.91`
- `top_q = 0.80`

Tag:
- `OU25_COMBINED_TOPQ_080`

### 3. Over-only
Purpose:
- directional specialist lane
- strongest directional OU25 result from frozen discovery

Parameters:
- `pick-mode = over_only`
- `band1 = 1.24–1.72`
- `band2 = 1.82–1.91`
- `top_q = 0.70`

Tag:
- `OU25_OVER_ONLY`

### 4. Band2 widened branch
Purpose:
- scale-friendly combined branch
- larger sample than premium branch while retaining strong ROI

Parameters:
- `pick-mode = combined`
- `band1 = 1.24–1.72`
- `band2 = 1.78–1.95`
- `top_q = 0.70`

Tag:
- `OU25_BAND2_178_195`

### 5. Band1 widened branch
Purpose:
- balanced benchmark branch
- strong ROI and good sample size

Parameters:
- `pick-mode = combined`
- `band1 = 1.24–1.76`
- `band2 = 1.82–1.91`
- `top_q = 0.70`

Tag:
- `OU25_BAND1_124_176`

---

## Run order

### Step 1 — preflight
Confirm:

```bash
python apply_frozen_product_gates.py --help | rg "ou25-only|ou25-pick-mode|include-ou25"