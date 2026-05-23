Perfect. Here’s a repo-ready markdown version with a clean structure, patch log section, and a reusable Weekend Deploy Execution Checklist.

# OG Deploy + Backtesting Runbook
## Current Working Operations Log (Rulebook / Gates / Backtests / Weekend Deploy Prep)

_Last updated: 2026-02-26_

---

## Purpose

This document is the operational runbook for the current OG deploy + backtesting stack.

It records:

- **which files are used**
- **what each file does**
- **commands used to run + test**
- **how core V3 market bundles are resolved (shared resolver)**
- **how to reproduce the 3-year / 19-league FTR Value vs Accuracy investor tables**
- **how `deploy_rulebook.py` gates and tiers rows**
- **current threshold references**
- **recent FTR demotion → re-promotion patch behavior**
- **the separate backtest rulebook pipeline (`filter_deploy_by_rulebook.py`) used for historical gating comparisons (value vs accuracy FTR profiles)**
- **a reusable weekend deploy checklist**

This should act as the **single clean log** for how the current chain works before live test deploys.

---

# 1) Core Files in Use

## 1.1 Backtesting / Research

### `backtest_deploy_csv.py`
Scores a deploy CSV against truth (`Matches/*`) and computes:

- row-level correctness (`correct`)
- hit-rate (`hit = mean(correct)`)
- average odds (`avg_od`)
- flat-stake ROI (`roi`)

Used for:
- month backtests
- gated deploy validation
- market/source/pick-level breakdowns

**Important:**
- Joins truth by `fixture_key`
- Loads truth from `Matches/<League>/fd_odds_enriched.csv` (current layout) and does **not** scan the entire Matches tree
- For UEFA competitions, qualifiers can land in different UEFA folders, so missing `correct` can occur unless you widen truth scope (see “Known Issues” in section 17)

---

### `filter_deploy_by_rulebook.py`
Applies a **deterministic historical gating rulebook** to a backtested deploy CSV (typically `__BACKTEST.csv`) and outputs gated subsets + summaries for comparison.

Key responsibilities:
- populates `bookie_od` from market-specific odds columns (FTR / OU25 / BTTS)
- derives proxy fair odds for TG15/TG25 when bookmaker TG odds are missing
- applies odds-band gates by market
- applies top-quantile score gating per market
- writes audit-friendly summaries (including market/source splits)
- supports **FTR profile modes** (e.g. value vs accuracy) for side-by-side backtest comparisons

Outputs typically include:
- `__GATED.csv`
- `__GATED_SUMMARY.csv`
- `__RULEBOOK_SNAPSHOT.json`

Used for:
- historical rulebook backtests
- comparing **FTR value** vs **FTR accuracy** profiles
- validating market mix / hit / ROI trade-offs before changing deploy-facing logic

**Critical distinction:**
- `ftr_profile=value` means `bookie_od >= 2.14` (price-seeking)
- `ftr_profile=accuracy` in this repo’s current definition means `bookmaker implied >= 0.68` i.e. `bookie_od <= 1/0.68 ≈ 1.470588` (favourites)
- Model confidence is **not** part of the accuracy definition unless explicitly enabled
# 2) High-Level Flow (Current Working Chain)

# 2.5 Stack Map (One Page)

bookie_allmarkets.py
  -> produces: BOOKIE_IMP*_ALLMARKETS_<date_from>_to_<date_to>.csv (one row per fixture per market)
  -> uses: og_model_paths.py (shared resolver; core markets strict v3)

deploy_rulebook.py (live deploy path)
  -> consumes: ALLMARKETS
  -> produces: __DEPLOY_* tiers (ELITE / STANDARD / OBSERVE)

backtest_deploy_csv.py (truth join)
  -> consumes: ALLMARKETS or DEPLOY csv
  -> produces: __BACKTEST.csv (row-level with correct) + __BACKTEST_SUMMARY.csv

filter_deploy_by_rulebook.py (historical gating / research)
  -> consumes: __BACKTEST.csv ONLY
  -> produces: __GATED.csv + __GATED_SUMMARY.csv + __RULEBOOK_SNAPSHOT.json

Guard rails / diagnostics (sit alongside the chain)
  - og_model_paths.py: bundle resolver used by bookie_allmarkets.py + pack_builder.py
  - audit_backtest_run.py: coverage + resolver-policy audit for ALLMARKETS or __BACKTEST
  - investor_table_ftr_value_vs_accuracy_IMP40.csv: per-league investor table output (generated)


---

### `seasonal_market_ledger.py`
Builds a market-level seasonal ledger.

Used for:
- tracking performance by product (FTR / OU25 / BTTS / TG15 / TG25)
- identifying stable vs fragile markets over time

---

### `build_months.sh`
Batch runner for monthly builds/backtests.

Used for:
- repeatable month-by-month evaluation
- generating comparable output artefacts

---

### `aggregate_gated_backtests.py`
Aggregates multiple monthly gated backtest summaries.

Used for:
- multi-month rollups
- combined hit-rate / ROI views
- checking whether one strong month persists across windows

---

### `season_leave_one_out.py`
Leave-one-out market ablation analysis.

Used for:
- identifying drag markets
- measuring hit-rate / ROI uplift when a market is excluded

---

### `build_months_2022_2024.sh`
Historical batch runner focused on 2022–2024 windows.

Used for:
- long-range historical comparisons
- robust backtest passes over multiple seasons

---

### `season_leave_one_out_2023_2024.py`
Season-specific leave-one-out analysis (2023/24).

Used for:
- focused diagnostics on 2023/24 only
- checking consistency vs aggregate findings

---

### `season_leave_one_out_2022_2023.py`
Season-specific leave-one-out analysis (2022/23).

Used for:
- focused diagnostics on 2022/23 only
- cross-season comparison vs 2023/24

---

## 1.2 Deploy / Generation

### `deploy_presets.py`
Preset parameter definitions for deploy profiles.

Used for:
- centralising parameter sets
- switching profiles without editing rulebook internals each run

---

### `bookie_allmarkets.py`
Builds the ALLMARKETS long-format market stack (one row per fixture per market).

Used for:
- the main source artefact before rulebook gating/tiering
- feeding `deploy_rulebook.py`

Typical markets:
- `ftr`
- `ou25`
- `btts`
- `tg15`
- `tg25`

---

### `deploy_gates.py`
Shared gate logic/helpers (where imported in the current stack).

Used for:
- reusable gating functions
- keeping rule logic cleaner / modular where applicable

---

### `deploy_rulebook.py`
Main deterministic rulebook + tiering script (current deploy engine in this chain).

Used for:
- runtime pass filtering
- market-specific gates (FTR / OU25 / BTTS / TG)
- tier assignment (`ELITE`, `STANDARD`, `OBSERVE`)
- FTR demotion + selective re-promotion into `STANDARD` (patched behavior)

Outputs typically include:
- `__DEPLOY_PRESET_V1.csv`
- `__DEPLOY_PRESET_V1.md`
- `__DEPLOY_TIER_ELITE.csv`
- `__DEPLOY_TIER_STANDARD.csv`
- `__DEPLOY_TIER_OBSERVE.csv`

---

# 2) High-Level Flow (Current Working Chain)

## A) Generate ALLMARKETS
`bookie_allmarkets.py` produces the long-format source stack.

Example output:
- `BOOKIE_IMP62_ALLMARKETS_<date_from>_to_<date_to>.csv`

This is the **candidate pool** before deploy filtering/tiering.

---

## B) Apply rulebook + tiers
`deploy_rulebook.py` reads ALLMARKETS and applies:

- runtime gates
- market-specific rulebook filters
- tier assignment
- demoted FTR row handling (including selective re-promotion)

Outputs:
- deploy preset CSV/MD
- tier CSVs (`ELITE`, `STANDARD`, `OBSERVE`)

---

## C) Backtest historical deploys (row-level truth join)
`backtest_deploy_csv.py` scores deploy rows against truth to validate:

- hit-rate
- ROI
- product-level behavior
- whether gates are robust vs overfit

Important output:
- `...__BACKTEST.csv` (row-level, includes `correct` and is the input to historical rulebook filtering)
- `...__BACKTEST_SUMMARY.csv` (aggregate only; not suitable as input to `filter_deploy_by_rulebook.py`)

---

## D) Apply historical rulebook filter (comparison pipeline)
`filter_deploy_by_rulebook.py` reads a **row-level `__BACKTEST.csv`** and applies historical gating for analysis/comparison.

Used for:
- recreating/benchmarking gated backtest profiles
- testing alternate FTR gate philosophies (e.g. value vs accuracy)
- auditing bookmaker vs model-fair odds contributions (especially TG markets)

Typical outputs:
- `__GATED.csv`
- `__GATED_SUMMARY.csv`
- `__RULEBOOK_SNAPSHOT.json`

---

## E) Aggregate / compare / ablate
Use:
- `aggregate_gated_backtests.py`
- `seasonal_market_ledger.py`
- `season_leave_one_out*.py`

This stage decides:
- which markets stay in deploy mix
- which markets get suppressed
- what belongs in `OBSERVE` only

Important: season_leave_one_out*.py and seasonal summary tools are for portfolio ablation/diagnostics and do not replace row-level FTR profile validation. FTR value vs accuracy comparisons must be run via filter_deploy_by_rulebook.py on row-level __BACKTEST.csv inputs.

---

# 3) Command Log (Run + Test)

## 3.1 CLI sanity check (post-patch)
Use this after any patch to confirm imports/argparse still work.

```bash
python deploy_rulebook.py --help >/tmp/deploy_rulebook_help.txt && tail -n 20 /tmp/deploy_rulebook_help.txt
```

Purpose:
- catches syntax/import breakage
- verifies CLI is still wired correctly

⸻

## 3.1B Resolver smoke tests (core V3 strictness)

- Export debug flags:
```
export OG_DEBUG_BUNDLES=1
export OG_DEBUG_MODEL_RESOLVER=1
mkdir -p run_logs
```

- Tiny run (example):
```
python bookie_allmarkets.py \
  --date-from 2025-01-10 \
  --date-to 2025-01-12 \
  --leagues "England Premier League,Germany Bundesliga,Champions League" \
  --debug 2>&1 | tee run_logs/tiny_resolver_smoke_$(date +%Y%m%d_%H%M%S).log
```

- Confirm resolver lines:
```
grep "BUNDLE_RESOLVER" run_logs/tiny_resolver_smoke_*.log
```

- Intentional MISS test (restore file afterwards):
```
mv "ModelStore/Germany_Bundesliga/btts_v3.pkl" "ModelStore/Germany_Bundesliga/btts_v3.pkl.bak"
python bookie_allmarkets.py --date-from 2025-01-10 --date-to 2025-01-10 --leagues "Germany Bundesliga" --debug
mv "ModelStore/Germany_Bundesliga/btts_v3.pkl.bak" "ModelStore/Germany_Bundesliga/btts_v3.pkl"
```

Note: core v2 fallback is disallowed for ftr/btts/ou25 (side models may be v2).

⸻

## 3.2 3-year / 19-league baseline build (ALLMARKETS)

These are the two canonical runs validated for the 3-year, 19-league corpus:

A) Favourite-heavy corpus (IMP40):
```
LEAGUES_CSV="England Premier League,England Championship,England EFL League 1,England FA Cup,Japan J1,Norway Eliteserien,Netherlands Eredivisie,Belgium Pro,Scotland Premiership,Brazil Serie A,USA MLS,Portugal Liga,Spain La Liga,Italy Serie A,France Ligue 1,Germany Bundesliga,Europa Conference,Europa League,Champions League"

python bookie_allmarkets.py \
  --date-from 2022-01-01 \
  --date-to 2025-12-31 \
  --leagues "$LEAGUES_CSV" \
  --implied-min 0.40 \
  --debug 2>&1 | tee run_logs/backtest_19l_3y_IMP40_$(date +%Y%m%d_%H%M%S).log
```

B) Favourites-only subset (IMP68) note that it produced no FTR odds >= 2.14 and therefore creates an FTR-value dead zone:
```
python bookie_allmarkets.py \
  --date-from 2022-01-01 \
  --date-to 2025-12-31 \
  --leagues "$LEAGUES_CSV" \
  --implied-min 0.68 \
  --debug 2>&1 | tee run_logs/backtest_19l_3y_IMP68_$(date +%Y%m%d_%H%M%S).log
```

Explanation: IMP68 is great for “accuracy favourites”, IMP40 is required to evaluate “value 2.14+”.

⸻

## 3.3 Build row-level truth backtests (__BACKTEST.csv)

```
# IMP40 truth join
python backtest_deploy_csv.py \
  --deploy-csv "predictions_output/2026-02-25/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31.csv" \
  --matches-root "Matches" \
  --outdir "predictions_output/backtests/19l_3y_IMP40"

# IMP68 truth join (optional baseline)
python backtest_deploy_csv.py \
  --deploy-csv "predictions_output/2026-02-25/BOOKIE_IMP68_ALLMARKETS_2022-01-01_to_2025-12-31.csv" \
  --matches-root "Matches" \
  --outdir "predictions_output/backtests/19l_3y_baseline"
```

**Key warning:** `filter_deploy_by_rulebook.py` must use `__BACKTEST.csv`, not summaries.

⸻

## 3.4 Run FTR Value vs FTR Accuracy sweeps (research)

Run these two sweeps against the IMP40 __BACKTEST:

- FTR Value (odds >= 2.14):
```
python filter_deploy_by_rulebook.py \
  --deploy-csv "predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv" \
  --outdir "predictions_output/rulebook_compare/19l_3y_IMP40__FTR_VALUE_214" \
  --ftr-profile value \
  --ftr-min 2.14
```

- FTR Accuracy (bookie implied >= 0.68, odds <= 1/0.68):
```
python filter_deploy_by_rulebook.py \
  --deploy-csv "predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv" \
  --outdir "predictions_output/rulebook_compare/19l_3y_IMP40__FTR_ACCURACY_68IMP" \
  --ftr-profile accuracy \
  --ftr-acc-max-od 1.470588
```

Note: we patched/validated per-league top_q to avoid global league starvation; do not use global top_q for FTR accuracy.

⸻

## 3.5 Produce the investor table (per-league VALUE vs ACCURACY)

Reference output file:
- `experiments/backtests/investor_table_ftr_value_vs_accuracy_IMP40.csv`

Minimal regeneration snippet (Python, 15-25 lines):
```python
import pandas as pd

# Read value and accuracy gated CSVs
value = pd.read_csv("predictions_output/rulebook_compare/19l_3y_IMP40__FTR_VALUE_214/__GATED.csv")
acc = pd.read_csv("predictions_output/rulebook_compare/19l_3y_IMP40__FTR_ACCURACY_68IMP/__GATED.csv")

def investor_table(df, tag):
    df = df[df["market"].str.lower().eq("ftr")]
    df = df.dropna(subset=["correct"])
    return (df.groupby("league")
              .agg(
                  n=( "correct", "size"),
                  hit=("correct", "mean"),
                  roi=(lambda x: (df.loc[x.index,"correct"] * (df.loc[x.index,"bookie_od"] - 1) - (1-df.loc[x.index,"correct"])).mean()),
                  avg_od=("bookie_od","mean"),
              )
              .rename(columns=lambda c: f"{tag}_{c}")
           )

tab = investor_table(value, "value").join(investor_table(acc, "acc"), how="outer")
tab.to_csv("experiments/backtests/investor_table_ftr_value_vs_accuracy_IMP40.csv")
print(tab)
```

This will output a per-league table with n/hit/roi/avg_od for both value and accuracy sweeps.

⸻

# 17) Diagnostics and Known Issues (2026-02-25 findings)

## 17.1 Why FTR Value died on IMP68
- At implied min 0.68, the FTR odds distribution is capped: FTR odds quantiles ~1.47 max at IMP68, so no FTR rows have odds >= 2.14. This creates a dead zone for FTR value analysis.

## 17.2 League starvation bug (global top_q)
- Using a global top_q drops league coverage down to 14; per-league top_q preserves all 19 leagues. Use per-league top_q for FTR accuracy research.

## 17.3 NaN correct rows (UEFA qualifiers truth mismatch)
- Truth is loaded per-league from Matches/<League>/fd_odds_enriched.csv only.
- UEFA qualifiers sometimes land in other UEFA folders, so the correct result is missing (correct=NaN) unless you widen the truth join.
- Fix options: join truth using a UEFA tri-pool, or fallback join on date/home/away.
- Note: `Matches/__merged__/fd_odds_enriched_synth.csv` does NOT exist in current repo layout.

3.2 Example deploy_rulebook.py run (used in current test)

python3 deploy_rulebook.py \
  --src predictions_output/2026-02-20/BOOKIE_IMP62_ALLMARKETS_2026-02-14_to_2026-02-17.csv \
  --outdir predictions_output/2026-02-20 \
  --debug

What this does:
- reads ALLMARKETS source
- applies runtime + market rulebook gates
- writes deploy preset outputs
- writes tier outputs
- prints detailed stage-by-stage diagnostics

⸻

## 3.2B Historical backtest-gating flow (the missing step)
This is the **separate analysis pipeline** used to compare historical gating profiles (including FTR value vs accuracy).

### Step 1 — Backtest a row-level deploy file
```bash
python3 backtest_deploy_csv.py \
  --deploy-csv predictions_output/YOUR_DEPLOY_OR_ALLMARKETS_FILE.csv \
  --outdir predictions_output/backtests
```

Use the resulting `...__BACKTEST.csv` as input for the next step.

### Step 2 — Apply historical rulebook filter (value profile example)
```bash
python3 filter_deploy_by_rulebook.py \
  --deploy-csv predictions_output/backtests/YOUR_FILE__BACKTEST.csv \
  --outdir predictions_output/rulebook_compare/value \
  --ftr-profile value \
  --ftr-min 2.14 \
  --top-q 0.70
```

### Step 3 — Apply historical rulebook filter (accuracy profile example)
```bash
python3 filter_deploy_by_rulebook.py \
  --deploy-csv predictions_output/backtests/YOUR_FILE__BACKTEST.csv \
  --outdir predictions_output/rulebook_compare/accuracy \
  --ftr-profile accuracy \
  --ftr-acc-max-od 1.60 \
  --ftr-acc-min-conf 0.68 \
  --ftr-acc-min-margin 0.06 \
  --ftr-acc-home-away-only \
  --top-q 0.70
```

### Important
- `filter_deploy_by_rulebook.py` expects a **row-level file** with columns like `market`, `bookie_pick`, `score`, and (for backtest metrics) `correct`.
- Do **not** pass aggregate files like `__BACKTEST_SUMMARY.csv` or season summary exports into `filter_deploy_by_rulebook.py`.

⸻

3.3 Patch verification commands (grep / inspect)

Check demotion reason tokens exist

rg -n "DEMOTED_ULTRASHORT_STANDARD|DEMOTED_MARGIN_STANDARD|DEMOTED_TO_OBSERVE" deploy_rulebook.py

Expected tokens:
	•	DEMOTED_ULTRASHORT_STANDARD
	•	DEMOTED_MARGIN_STANDARD
	•	DEMOTED_TO_OBSERVE

⸻

Inspect patched FTR demotion/promotion tiering block

sed -n '4156,4275p' deploy_rulebook.py

Confirms:
	•	demoted_promoted variable naming
	•	lane-specific reason tagging
	•	promoted rows removed from OBSERVE

⸻

Confirm Lane 2 params + FTR structural refs are wired

rg -n "ftr_demote_lane2_|draw_chaos_score|ftr_margin|bookie_od" deploy_rulebook.py

Confirms:
	•	Lane 2 thresholds exist in code
	•	key columns used by the rulebook are referenced

⸻

3.4 Tier integrity test (no cross-tier duplicates)

After a deploy run, verify no row appears in more than one tier on key columns.

Recommended key:
	•	league
	•	fixture_key
	•	market
	•	bookie_pick

Recent validation result:
	•	Total rows across tiers: 61
	•	Duplicate key rows across tiers: 0
	•	✅ No cross-tier duplicates on key columns.

This is a hard integrity requirement after the demotion → re-promotion patch.

⸻

3.5 Promotion verification test (FTR demoted → STANDARD patch)

Recent observed result from STANDARD tier:
	•	STANDARD rows: 3
	•	Promoted demoted rows in STANDARD: 3

Observed examples:
	•	Liverpool vs Brighton (ftr, HOME) → DEMOTED_MARGIN_STANDARD
	•	Burnley vs Mansfield (ftr, HOME) → DEMOTED_MARGIN_STANDARD
	•	Manchester City vs Salford (ftr, HOME) → DEMOTED_ULTRASHORT_STANDARD|DEMOTED_MARGIN_STANDARD

Confirms:
	•	Lane 2 promotion works
	•	Lane 1 and Lane 2 can both tag the same row (expected behavior)

⸻

4) How the Rulebook + Gates Work (Conceptual)

4.1 Stage 1 — Runtime gates

deploy_rulebook.py first applies a runtime pass filter across the source stack.

Recent example:
	•	raw rows: 107
	•	runtime pass: 61

This stage removes rows with:
	•	hard veto conditions
	•	signal mismatches
	•	market-shape conflicts
	•	pre-rulebook incompatibilities

Example veto labels observed:
	•	OVER25_P00_TOO_HIGH
	•	BTTS_YES_CS1_TRAP
	•	SIGNAL_MISMATCH

These are printed in the veto top12 summary for debugging.

⸻

4.2 Stage 2 — Market-specific rulebook gates

Rows are split and processed by market (FTR / OU25 / BTTS / TG15 / TG25).

Each market follows a staged funnel:
	•	start
	•	agreement / alignment checks
	•	thresholds
	•	structural filters
	•	veto layers
	•	final kept rows

This staged logging is essential for diagnosing why rows disappear.

⸻

4.3 Stage 3 — Tiering (ELITE, STANDARD, OBSERVE)

After market gates:
	•	ELITE = strictest, best deploy rows
	•	STANDARD = valid deploy rows, lower strictness than ELITE
	•	OBSERVE = runtime-pass rows not selected for ELITE/STANDARD + demoted rows

Tier invariant

Tiers must remain disjoint.

Patched behavior ensures:
	•	if a demoted FTR row is re-promoted into STANDARD,
	•	it is removed from OBSERVE using row keys.

⸻

5) FTR Gate Sequence (Detailed Working Notes)

5.1 Observed FTR debug sequence

Typical FTR logging flow:
	1.	FTR start
	2.	FTR after agree
	3.	FTR implied_min gate
	4.	FTR after gap_min
	5.	FTR after margin_min
	6.	FTR xg confirm
	7.	FTR after xg confirm
	8.	FTR after ppg confirm
	9.	FTR(GLUE) after not_glue_flag veto
	10.	FTR after od_min
	11.	FTR power_diff pre-gate
	12.	FTR after power_diff gate

This shows FTR picks can survive early model/odds checks and still drop later on:
	•	PPG confirmation
	•	GLUE veto
	•	odds floor
	•	power-diff sanity gate

⸻

5.2 FTR columns used (observed in code/logs)

Core pricing / confidence
	•	bookie_pick
	•	bookie_od
	•	bookie_implied
	•	bookie_implied_used
	•	ftr_margin
	•	gap_used

Structural / context
	•	draw_chaos_score
	•	power_diff
	•	not_glue_flag
	•	xG / PPG support columns

Odds fallback support (FTR)
	•	od_home
	•	od_draw
	•	od_away

Rulebook contains fallback logic to backfill bookie_od from 1X2 columns when needed.

⸻

5.3 FTR threshold references (current observed values)

Implied minimum (observed in debug run)
	•	FTR implied_min gate thr = 0.680

Add a short design note to the runbook / script header:
	•	FTR value profile: value-priced FTR selections (default bookie_od >= 2.14)
	•	FTR accuracy profile: moderate favourites (recommended cap testing 1.75–1.85)
	•	These are separate products, not mirror images

Also add a gotcha note:
	•	If accuracy profile shows ALL rows but zero MARKET::ftr, that means FTR accuracy dead-zone, not a failed batch run.

That’ll save future-you hours.
⸻

Gap gate (observed dynamic thresholds)

Debug text shows context-dependent thresholds, e.g.:
	•	HARD>=-0.80
	•	short>=0.65:-0.45
	•	else:-0.10
	•	ANCHOR_CAND:-0.55

This indicates gap acceptance is not a single static threshold.

⸻

Margin minimum

ftr_margin is a core structural filter.

The exact effective threshold may vary by params/preset, but it is a key gate before later confirmations.

⸻

Odds minimum (observed)
	•	FTR after od_min>=1.167

This can remove ultra-short rows even if they pass earlier stages.

⸻

Power-diff gate

Applied late as a structural sanity check for directional picks.

Observed logs:
	•	FTR power_diff pre-gate...
	•	FTR after power_diff gate

⸻

6) FTR Demotion → Re-promotion Patch (Patched Behavior)

6.1 Why this patch exists

Some strong FTR favourites were being demoted into OBSERVE despite being useful safe-leg candidates.

The patch adds selective re-promotion of certain demoted FTR rows into STANDARD.

⸻

6.2 Where it lives

deploy_rulebook.py tiering block around:
	•	~4156–4275 (current working file state)

⸻

6.3 Patched logic summary

Input
	•	demote_keys identifies rows demoted earlier in the rulebook process
	•	demoted = df_pass.loc[dk.isin(demote_keys)]

Split demoted rows into:
	•	demoted_promoted → rows promoted back to STANDARD
	•	demoted_rest → remain in OBSERVE

Promotion condition:
	•	promote_mask = is_ultra | is_lane2

⸻

6.4 Promotion lanes

Lane 1 — Ultra-short favourites

Promote if either:
	•	bookie_od <= ftr_short_od_max
	•	OR bookie_implied_used (fallback bookie_implied) >= ftr_short_imp_min

Current defaults in code:
	•	ftr_short_od_max = 1.10
	•	ftr_short_imp_min = 0.85

Reason token added:
	•	DEMOTED_ULTRASHORT_STANDARD

⸻

Lane 2 — High-margin favourites with acceptable draw chaos

Promote if all pass:
	•	bookie_od <= ftr_demote_lane2_od_max
	•	ftr_margin >= ftr_demote_lane2_margin_min
	•	draw_chaos_score <= ftr_demote_lane2_draw_chaos_max
	•	missing draw_chaos_score is treated as allowed (fail-open for missing DCS in this lane)

Current params in code:
	•	ftr_demote_lane2_od_max = 1.60
	•	ftr_demote_lane2_margin_min = 0.08
	•	ftr_demote_lane2_draw_chaos_max = 0.92

Reason token added:
	•	DEMOTED_MARGIN_STANDARD

⸻

6.5 Important patch details (naming + integrity)

Renamed variable

Old concept:
	•	demoted_ultra

Patched variable:
	•	demoted_promoted

Reason:
	•	promotions are no longer ultra-short only (Lane 2 now exists)
	•	avoids future confusion / false bug hunts

⸻

Lane-specific reason tagging

demoted_promoted rows are tagged per-lane:
	•	DEMOTED_ULTRASHORT_STANDARD
	•	DEMOTED_MARGIN_STANDARD

A row can receive both tags if it matches both lanes.

⸻

Promote to STANDARD and remove from OBSERVE

Patched flow:
	1.	append demoted_promoted to deploy_standard
	2.	remove those row keys from observe

This preserves tier disjointness.

⸻

Comment update (done)

The comment above the block was updated to reflect both lanes:
	•	Lane 1 = ultra-short favourites
	•	Lane 2 = high-margin favourites with acceptable draw chaos

This prevents future misreads of intended behavior.

⸻

7) OU25 / BTTS / TG Gating (Operational Summary)

7.1 OU25

Observed OU25 flow includes:
	•	agreement checks
	•	implied banding
	•	split OVER25 vs UNDER25
	•	model floors
	•	strong-signal gates
	•	veto layer (FTS / CS1 / under-shape etc.)
	•	structural rate / goaliness filters

Recent behavior observed:
	•	OVER side can be thinned heavily by signal + probability gates
	•	UNDER side may fail-open on one band but still survive through structural filters

⸻

7.2 BTTS

Observed BTTS flow includes:
	•	split YES / NO
	•	strong-label gates
	•	model floor
	•	FTS-based logic
	•	low-goaliness checks (especially for NO)
	•	recombination

Key point:
	•	BTTS YES and BTTS NO are intentionally handled differently

⸻

7.3 TG15 / TG25

Observed TG flow includes:
	•	agreement checks
	•	probability minimum (pmin)
	•	tg_pois_ok == 1
	•	Poisson tail thresholds (e.g. pois_ge2 for TG15)

Recent run example:
	•	TG15 reduced to zero after pois_ge2>=0.50
	•	TG25 reduced to zero after pmin

This aligns with prior evidence that TG markets (especially TG25) can become drag markets depending on threshold profile and sample window.

⸻

8) Backtest Metrics (How Accuracy / ROI Are Calculated)

For each deploy row:
	•	correct = whether selection won (market-specific truth logic)
	•	hit = mean(correct)
	•	avg_od = mean(odds)
	•	roi = flat-stake average profit at decimal odds

ROI formula (decimal odds)

Profit per bet:
	•	correct * (odds - 1) - (1 - correct)

Then:
	•	roi = mean(profit per bet)

Equivalent shorthand:
	•	roi = mean(correct * odds - 1)

(Equivalent when correct is 0/1)

⸻

9) April 2025 Gated Backtest — Reference Ledger

These are the current headline reference numbers for the April gated profile.

9.1 Combined
	•	All combined: 84.6% hit
	•	n=285
	•	avg_od=1.864
	•	ROI=+0.490

⸻

9.2 Market hit-rates (April gated)
	•	FTR: 62.7%
	•	OU25: 93.1%
	•	BTTS: 91.7%
	•	TG15: 97.6%
	•	TG25: 78.9%
	•	All combined: 84.6%

⸻

9.3 Interpretation notes
	•	TG ROI may be proxy/fair-odds ROI where bookmaker TG odds are missing
	•	Always separate:
	•	bookie_od_source = bookmaker
	•	bookie_od_source = model_fair

This avoids overstating TG performance vs real market odds.

⸻

10) Threshold Reference Notes (Current Working Values)

10.1 FTR (observed / current references)
	•	implied_min ≈ 0.680 (observed in debug run)
	•	od_min ≈ 1.167 (observed in debug run)
	•	ftr_margin used as a core structural gate
	•	draw_chaos_score used in draw/glue logic and Lane 2 demote re-promotion
	•	power_diff gate applied late

⸻

10.2 Demoted FTR re-promotion thresholds (patched)
	•	ftr_short_od_max = 1.10
	•	ftr_short_imp_min = 0.85
	•	ftr_demote_lane2_od_max = 1.60
	•	ftr_demote_lane2_margin_min = 0.08
	•	ftr_demote_lane2_draw_chaos_max = 0.92

⸻

11) Patch Log (Operational Change Record)

Patch: FTR demotion → selective re-promotion into STANDARD (Lane 1 + Lane 2)

Status

✅ Applied and tested

What changed
	•	Expanded demoted-row promotion logic from “ultra-short only” into two promotion lanes:
	•	Lane 1: ultra-short favourites
	•	Lane 2: high-margin favourites with acceptable draw chaos
	•	Renamed working variable:
	•	demoted_ultra → demoted_promoted
	•	Added lane-specific reason tokens:
	•	DEMOTED_ULTRASHORT_STANDARD
	•	DEMOTED_MARGIN_STANDARD
	•	Kept demoted leftovers tagged as:
	•	DEMOTED_TO_OBSERVE
	•	Removed promoted rows from OBSERVE after appending to STANDARD (prevents cross-tier duplication)
	•	Updated comments to reflect both lanes

Validation results (recent run)
	•	Cross-tier duplicates on key columns: 0
	•	Promoted demoted rows observed in STANDARD: 3
	•	Lane tagging working, including dual-tag case (Man City example)

⸻

12) Weekend Deploy Execution Checklist (Reusable)

12.1 Pre-run (code + patch integrity)
	•	python deploy_rulebook.py --help runs successfully
	•	grep shows all demotion tokens:
	•	DEMOTED_ULTRASHORT_STANDARD
	•	DEMOTED_MARGIN_STANDARD
	•	DEMOTED_TO_OBSERVE
	•	patched tiering block shows demoted_promoted
	•	patched tiering block removes promoted rows from OBSERVE

⸻

12.2 Input readiness (ALLMARKETS source)
	•	fresh BOOKIE_IMP*_ALLMARKETS_*.csv generated via bookie_allmarkets.py
	•	file path/date window matches intended weekend fixtures
	•	spot-check key columns exist:
	•	market
	•	bookie_pick
	•	bookie_od or 1X2 odds columns
	•	ftr_margin
	•	draw_chaos_score (if available)
	•	xG/PPG support columns for FTR confirms
	•	obvious odds look sane on random sample rows

⸻

12.3 Run deploy rulebook
	•	run deploy_rulebook.py --debug
	•	save console logs to file
	•	confirm output files written:
	•	__DEPLOY_PRESET_V1.csv
	•	__DEPLOY_PRESET_V1.md
	•	__DEPLOY_TIER_ELITE.csv
	•	__DEPLOY_TIER_STANDARD.csv
	•	__DEPLOY_TIER_OBSERVE.csv

⸻

12.4 Post-run integrity tests
	•	no cross-tier duplicates on key columns (league,fixture_key,market,bookie_pick)
	•	promoted FTR rows appear in STANDARD with reason tokens where expected
	•	OBSERVE does not still contain promoted rows
	•	tier counts look sensible (not unexpectedly tiny or huge)

⸻

12.5 Quick qualitative review (before test deploy)
	•	FTR selections look sane (no obvious nonsense favorites/underdogs)
	•	OU25 mix looks plausible (not all one side unless expected)
	•	BTTS YES/NO split looks plausible
	•	TG15/TG25 not accidentally over-represented if current profile should be selective
	•	inspect any ultra-short FTRs (1.03–1.10) for expected handling

⸻

12.6 Optional historical confidence check (recommended)
	•	run/compare latest window vs known April profile behavior
	•	check if market mix drifts sharply from prior successful gated profiles
	•	note any suspicious league pockets for observe-only treatment

⸻

13) Useful Copy/Paste Command Block

# 1) CLI sanity
python deploy_rulebook.py --help >/tmp/deploy_rulebook_help.txt && tail -n 20 /tmp/deploy_rulebook_help.txt

# 2) Check patched demotion tokens
rg -n "DEMOTED_ULTRASHORT_STANDARD|DEMOTED_MARGIN_STANDARD|DEMOTED_TO_OBSERVE" deploy_rulebook.py

# 3) Inspect patched tiering block (FTR demote promotion logic)
sed -n '4156,4275p' deploy_rulebook.py

# 4) Check Lane 2 parameter references + FTR field usage
rg -n "ftr_demote_lane2_|draw_chaos_score|ftr_margin|bookie_od" deploy_rulebook.py

# 5) Run deploy_rulebook on a target ALLMARKETS file
python3 deploy_rulebook.py \
  --src predictions_output/2026-02-20/BOOKIE_IMP62_ALLMARKETS_2026-02-14_to_2026-02-17.csv \
  --outdir predictions_output/2026-02-20 \
  --debug


⸻

14) Known Practical Notes / Gotchas
	•	Tier counts will change when the source ALLMARKETS file changes, even if code is unchanged.
	•	TG markets can look amazing in short windows and then regress; keep them under tighter scrutiny.
	•	Small-sample league/pick splits are diagnostic signals, not final conclusions.
	•	Dual lane tokens on one row are expected (e.g. ultra-short + margin both true).
	•	No cross-tier duplicates is non-negotiable after the re-promotion patch.
	•	`filter_deploy_by_rulebook.py` must be run on row-level `__BACKTEST.csv` inputs, not summary CSVs (e.g. `__BACKTEST_SUMMARY.csv` or leave-one-out summary exports).

	•	IMP68 ALLMARKETS is not suitable for FTR value (2.14+) analysis; use IMP40 (or lower) to include those odds bands.
	•	For UEFA comps, expect occasional correct=NaN unless truth loading is widened for qualifiers.

⸻

15) Next Actions (for the upcoming weekend test deploy)

Immediate plan
	1.	Generate fresh weekend ALLMARKETS using bookie_allmarkets.py
	2.	Run deploy_rulebook.py --debug
	3.	Save console output to dated log file
	4.	Run tier duplicate check
	5.	Inspect STANDARD FTR rows for:
	•	DEMOTED_MARGIN_STANDARD
	•	DEMOTED_ULTRASHORT_STANDARD
	6.	Produce a short deploy summary:
	•	counts by market
	•	counts by tier
	•	promoted FTR rows + tokens
	•	top FTR/OU25/BTTS candidates for manual review

Optional hardening before weekend
	•	Add a tiny helper script to automate:
	•	cross-tier duplicate checks
	•	promoted-row listing
	•	tier market counts
	•	Save output as __TIER_AUDIT.md alongside deploy outputs

⸻


## 16) 3-Year / 19-League FTR Validation Execution Plan

### Goal
Validate and compare two historical FTR gating products across a broad multi-season dataset using the **same row-level backtest inputs**:

- **FTR value profile** (value-priced selections; default `bookie_od >= 2.14`)
- **FTR accuracy profile** (moderate favourites with confidence/margin tightening)

This is a **research-validation workflow** (not the live deploy path) intended to test robustness across:
- ~3 years of data
- all 19 target leagues
- multiple seasonal windows

---

### 16.1 Golden rule (file-type discipline)
`filter_deploy_by_rulebook.py` must only be run on **row-level** backtest files:

- ✅ `...__BACKTEST.csv`

It must not be run on:
- ❌ `...__BACKTEST_SUMMARY.csv`
- ❌ seasonal ledger exports
- ❌ leave-one-out summary exports
- ❌ aggregated summaries

If accuracy profile shows `ALL` rows but zero `MARKET::ftr`, that is an **FTR accuracy dead-zone**, not a failed batch run.

---

### 16.2 Campaign folder layout (recommended)
Use a dedicated campaign root to prevent mixing phases and artefact types.

Example:

- `predictions_output/ftr_profile_validation_3y_19l/`
  - `01_inputs_manifest/`
  - `02_backtests_rowlevel/`
  - `03_rulebook_sweeps/`
  - `04_parsed_rankings/`
  - `05_aggregates/`
  - `06_notes/`
  - `trash_or_failed_runs/` (optional but recommended)

Keep row-level backtests, gated outputs, parser outputs, and aggregate summaries in separate folders.

---

### 16.3 Phase 0 — Define and freeze the target universe
Create a single manifest (CSV recommended) listing every planned window/file for the campaign.

Recommended manifest columns:
- `window_id`
- `date_from`
- `date_to`
- `season_tag`
- `league_scope`
- `source_type`
- `src_csv_path`
- `status_backtested`
- `status_swept`
- `notes`

Guidance:
- Prefer **ALLMARKETS raw** inputs for broad FTR coverage.
- Avoid relying on heavily filtered deploy outputs (e.g. `YESONLY`) for primary FTR profile validation because they can create FTR dead-zones.

---

### 16.4 Phase 1 — Build the row-level backtest base (truth-joined)
Use `backtest_deploy_csv.py` to produce row-level backtests for each selected source file.

Input (preferred):
- raw `BOOKIE_IMP*_ALLMARKETS_*.csv`

Outputs (keep):
- ✅ `...__BACKTEST.csv` (**row-level; core input for FTR profile validation**)
- ⚠️ `...__BACKTEST_SUMMARY.csv` (keep for reference only; **not** rulebook-filter input)

Recommended output location:
- `02_backtests_rowlevel/<window_id>/`

#### Phase 1 QC checks (quick, mandatory)
For each `__BACKTEST.csv`, confirm:
- file exists
- columns include `market`, `bookie_pick`, `score`, `correct`
- FTR rows exist (`market == ftr`)
- odds columns exist / `bookie_od` can be derived

If a file fails QC, mark it in the manifest and do **not** include it in rulebook sweeps.

---

### 16.5 Phase 2 — Run FTR profile rulebook comparisons (value + accuracy)
Run `filter_deploy_by_rulebook.py` (directly or via `run_rulebook_ftr_sweeps.sh`) on the **same row-level `__BACKTEST.csv` files**.

Profiles to run:
1. **Value baseline** (e.g. `--ftr-profile value --ftr-min 2.14`)
2. **Accuracy sweeps** (confidence / max-od / optional margin)

Inputs allowed in `run_rulebook_ftr_sweeps.sh`:
- ✅ files ending `__BACKTEST.csv`
- ❌ files ending `__BACKTEST_SUMMARY.csv`

Outputs to keep per run:
- `__GATED.csv`
- `__GATED_SUMMARY.csv`
- `__RULEBOOK_SNAPSHOT.json`

Recommended sweep output root:
- `03_rulebook_sweeps/<campaign_tag>/...`

#### Recommended first-pass sweep (compact)
Value profile:
- `ftr_min = 2.14`

Accuracy profile (initial):
- `top_q = 0.70`
- `home_away_only = true`
- fixed `margin = 0.06`
- confidence sweep: `0.62 0.64 0.66 0.68 0.70 0.72`
- max-od sweep: `1.60 1.70 1.80 1.85 1.90 2.00`

Add margin sweeps later only after confirming row-level coverage and dead-zone behavior.

---

### 16.6 Phase 3 — Parse and rank the sweep outputs
Use `parse_rulebook_ftr_sweeps.py` to build ranked comparisons across windows and parameter sets.

Outputs to keep:
- `RANKED_RULEBOOK_SWEEP_RESULTS.csv`
- `RANKED_RULEBOOK_SWEEP_RESULTS__AGG_BY_PARAM.csv`

Recommended destination:
- `04_parsed_rankings/`

This phase is used to identify:
- best parameter sets within a window
- parameter sets that remain viable across multiple windows
- dead-zone frequency (accuracy profile has `ALL` rows but no `MARKET::ftr`)

---

### 16.7 Phase 4 — Stability, season, and league coverage analysis
After parsing/ranking, evaluate robustness across seasons and leagues.

Use (as appropriate):
- `aggregate_gated_backtests.py`
- `seasonal_market_ledger.py`
- custom analysis scripts/notebooks (league coverage tables, dead-zone rate, variance)

Produce at least:
1. **Parameter stability table** (windows tested, FTR present rate, mean/median hit/ROI)
2. **Season split table** (e.g. 2022/23 vs 2023/24 vs 2024/25)
3. **League coverage table** (FTR n / hit / ROI by league with a minimum-n threshold)

This is where global-looking performance is tested for concentration risk (e.g. a few leagues carrying the result).

---

### 16.8 Phase 5 — Portfolio ablation (separate question)
Use `season_leave_one_out*.py` and related seasonal diagnostics only **after** candidate FTR profiles have been selected.

Purpose:
- test whether chosen FTR value/accuracy profiles improve or harm the **overall portfolio mix**
- identify whether FTR belongs in deploy mix, observe-only, or conditional use by season/league

Important:
- This phase does **not** replace row-level FTR profile validation.
- It answers portfolio composition questions, not profile discovery.

---

### 16.9 Exact execution order (do not reorder)
1. **Create/update manifest** (`01_inputs_manifest/`)
2. **Generate row-level backtests** with `backtest_deploy_csv.py`
3. **QC-check row-level files** (columns + FTR presence)
4. **Freeze valid `__BACKTEST.csv` list** for the campaign
5. **Run rulebook sweeps** (value + accuracy) on that frozen list only
6. **Parse sweep outputs** into ranked CSVs
7. **Aggregate by season/league/stability**
8. **Select candidate FTR value + accuracy parameter sets**
9. **Run portfolio ablation diagnostics** (`season_leave_one_out*.py`) on selected candidates
10. **Write a short decision note** in `06_notes/` (chosen/rejected params, dead-zone behavior, next test plan)

---

### 16.10 Artefact retention policy (keep vs discard)
#### Keep (campaign evidence)
- input manifest(s)
- row-level `__BACKTEST.csv` files used in final comparison
- sweep outputs (`__GATED.csv`, `__GATED_SUMMARY.csv`, `__RULEBOOK_SNAPSHOT.json`)
- parser ranking CSVs
- aggregate stability tables
- final decision notes / memos

#### Discard or move to `trash_or_failed_runs/`
- failed runs from broken script versions
- ad hoc scratch outputs
- duplicate parser outputs superseded by corrected runs
- test artefacts from incorrect inputs (e.g. summary files accidentally passed in)

---

### 16.11 File hygiene rules (strongly recommended)
- Treat `__BACKTEST.csv` as **rulebook-eligible** and `__BACKTEST_SUMMARY.csv` as **summary-only**.
- Add a preflight guard in `run_rulebook_ftr_sweeps.sh` to reject summary files before any run starts.
- Use a manifest flag (e.g. `rulebook_eligible=1`) and only populate sweep script `FILES=(...)` from that filtered list.
- If copying files for manual review, use explicit prefixes such as:
  - `ROWBACKTEST__...`
  - `GATED__...`
  - `SUMMARY__...`
  - `SNAPSHOT__...`

ftr_validation_manifest.csv (columns)

window_id,date_from,date_to,season_tag,league_scope,league_count,source_type,src_csv_path,backtest_csv_path,backtest_summary_csv_path,rulebook_eligible,status_backtested,status_swept,ftr_rows_raw,ftr_rows_backtest,notes
apr2025_allm_01,2025-04-01,2025-04-30,2024_2025,core19,19,allmarkets_raw,predictions_output/2025-04/BOOKIE_IMP0_ALLMARKETS_2025-04-01_to_2025-04-30.csv,predictions_output/ftr_profile_validation_3y_19l/02_backtests_rowlevel/apr2025_allm_01/BOOKIE_IMP0_ALLMARKETS_2025-04-01_to_2025-04-30__BACKTEST.csv,predictions_output/ftr_profile_validation_3y_19l/02_backtests_rowlevel/apr2025_allm_01/BOOKIE_IMP0_ALLMARKETS_2025-04-01_to_2025-04-30__BACKTEST_SUMMARY.csv,1,done,pending,547,547,"Pilot reference month; good FTR coverage"
mayjul2025_yesonly_01,2025-05-01,2025-07-31,2024_2025,core19,19,deploy_yesonly_backtest,predictions_output/2025-12-30/BOOKIE_IMP62_ALLMARKETS_2025-05-01_to_2025-07-31__DEPLOY_PROFIT_PRESET_FINAL_YESONLY.csv,predictions_output/ftr_profile_validation_3y_19l/02_backtests_rowlevel/mayjul2025_yesonly_01/BOOKIE_IMP62_ALLMARKETS_2025-05-01_to_2025-07-31__DEPLOY_PROFIT_PRESET_FINAL_YESONLY__BACKTEST.csv,predictions_output/ftr_profile_validation_3y_19l/02_backtests_rowlevel/mayjul2025_yesonly_01/BOOKIE_IMP62_ALLMARKETS_2025-05-01_to_2025-07-31__DEPLOY_PROFIT_PRESET_FINAL_YESONLY__BACKTEST_SUMMARY.csv,0,done,pending,14,14,"FTR dead-zone risk; keep for comparison only"

Column meanings (recommended)
	•	window_id — stable ID used in folders/logs
	•	source_type — e.g. allmarkets_raw, deploy_yesonly_backtest, deploy_preset_backtest
	•	src_csv_path — original source file you backtest from
	•	backtest_csv_path — row-level __BACKTEST.csv (the only valid rulebook sweep input)
	•	backtest_summary_csv_path — summary file (for reference only)
	•	rulebook_eligible — 1 or 0 (hard gate for sweeps)
	•	status_backtested / status_swept — pending|done|failed|skipped
	•	ftr_rows_raw / ftr_rows_backtest — quick coverage sanity
	•	notes — dead-zone / quirks / data quality flags

---

### 16.12 Decision criteria for selecting FTR paths
Do not select a profile using only one strong month.

Evaluate candidates on:
- **FTR present rate** across windows (dead-zone frequency matters)
- **sample size** (`ftr_n`) and league spread
- **hit-rate stability** (not just peak hit)
- **ROI stability** (watch for collapses / concentration)
- **portfolio compatibility** (post-ablation)

Treat the two profiles as separate products:
- **Value FTR** = price-seeking / higher-odds FTR selections
- **Accuracy FTR** = safer-leg / moderate favourite FTR selections

They are not mirror images and may both be useful in different deployment contexts.

---

### 16.13 Immediate next step (recommended)
Before launching the full 3-year / 19-league campaign:
1. build a clean manifest of candidate inputs
2. validate a pilot run on 6–10 windows end-to-end
3. confirm parser + aggregate outputs look correct
4. then launch the full campaign using the same folder and file discipline

This avoids wasting compute/time on a full pass with one bad file family in the input list.

---

Appendix A) Quick reference of files (this runbook scope)

Backtesting
	•	backtest_deploy_csv.py
	•	filter_deploy_by_rulebook.py
	•	seasonal_market_ledger.py
	•	build_months.sh
	•	aggregate_gated_backtests.py
	•	season_leave_one_out.py
	•	build_months_2022_2024.sh
	•	season_leave_one_out_2023_2024.py
	•	season_leave_one_out_2022_2023.py

Deploy
	•	deploy_presets.py
	•	bookie_allmarkets.py
	•	deploy_gates.py
	•	deploy_rulebook.py









Step A — Create the missing row-level backtest

Run:

python backtest_deploy_csv.py \
  --deploy-csv "predictions_output/2026-02-25/BOOKIE_IMP68_ALLMARKETS_2022-01-01_to_2025-12-31.csv" \
  --matches-root "Matches" \
  --outdir "predictions_output/backtests/19l_3y_baseline"

  This should write (names vary slightly but you’ll get):
	•	...__BACKTEST.csv ✅ (row-level, has correct)
	•	...__BACKTEST_SUMMARY.csv


Step B — Per-league FTR accuracy (copy/paste)

Run this exactly:

python - <<'PY'
import pandas as pd

p = "predictions_output/backtests/19l_3y_baseline/BOOKIE_IMP68_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv"
df = pd.read_csv(p, low_memory=False)

# normalise
df["market"] = df["market"].astype(str).str.lower().str.strip()
df["correct"] = pd.to_numeric(df.get("correct"), errors="coerce")

ftr = df[df["market"].eq("ftr")].dropna(subset=["correct"]).copy()

out = (ftr.groupby("league")
          .agg(
              ftr_n=("correct","size"),
              ftr_hit=("correct","mean"),
              avg_od=("bookie_od","mean"),
          )
          .reset_index())

out["breakeven_hit"] = 1.0 / out["avg_od"]
out["edge_vs_be"] = out["ftr_hit"] - out["breakeven_hit"]

out = out.sort_values(["edge_vs_be","ftr_n"], ascending=[False,False])

pd.set_option("display.max_rows", 200)
print(out.to_string(index=False))
PY


This gives you per-league:
	•	sample size
	•	hit-rate
	•	average odds
	•	break-even hit-rate
	•	“edge vs break-even” (positive = profitable zone in theory)


Now that you have __BACKTEST.csv, we should run the audit script against that too.

python audit_backtest_run.py \
  --csv "predictions_output/backtests/19l_3y_baseline/BOOKIE_IMP68_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv" \
  --log "$(ls -t run_logs/backtest_19l_3y_baseline_*.log | head -n 1)" \
  --outdir "experiments/backtests/audit_19l_3y_baseline__BACKTEST"

  1) Scoring gap: 28 rows not scored

Fast inspect:

