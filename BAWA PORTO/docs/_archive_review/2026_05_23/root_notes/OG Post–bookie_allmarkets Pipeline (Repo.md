## pack_builder.py (V3) — Pack / Pool Builder (fresh pipeline)

`pack_builder.py` is the **clean, self-contained V3 pack-building pipeline**.  
It consumes an **enriched match spine** (fixture-level CSV) *or* an **ALLMARKETS long** CSV and emits deterministic **packs/pools** (plus a stack pool) under an output directory.

It is designed to live alongside legacy V1/V2 logic (e.g. `prediction_overlay.py`, `deploy_rulebook.py`) without changing them.

V3 status (Feb 2026)
--------------------
V3 now has a validated **odds-aware evaluation loop** for ALLMARKETS outputs:
- `filter_deploy_by_rulebook.py` can populate `bookie_od` deterministically (per market) and apply a rulebook gate.
- `backtest_deploy_csv.py` can score the gated output and compute hit-rate + ROI.
- Team-goals markets (TG15/TG25) can participate in ROI backtests using **model-derived fair odds** when bookmaker odds are missing.

---

### What it builds

From one input CSV, it produces:

**Adult packs (BTTS artefact-gated by default):**
- **Adult Premium (strict)**: BTTS + O2.5 + Side/Draw safety + TG≥2 Poisson tail
- **Adult Matchup (strict)**: BTTS + stronger Side/Draw safety + TG≥2 tail
- **Mismatch Pool TierB (“wild” band)**: underdog-band subset with TG≥2 + BTTS gate

**FTR pack:**
- **FTR Core**: chooses H/D/A legs using `p_home/p_draw/p_away` already present (or injected via `--score-v3`)

**Team Goals packs:**
- **TG GE2**: picks HOME/AWAY team to score ≥2 based on best-calibrated GE2 probs (or Poisson fallback)
- **TG GE3**: same for ≥3

**U21 pool (not BTTS artefact-gated):**
- **U21 O2.5 pool** + tier split into S/A/B by `u21_o25_score`

**Stack pool (optional, default ON):**
- Writes the fully enriched, artefact-gated frame as a single CSV (useful as the “canonical V3 pool” snapshot)

---

### Inputs

#### Required
- `--src <csv>`: input CSV (enriched spine / ALLMARKETS / stacked pool)

#### Input modes
- `--src-is-allmarkets`  
  Treat `--src` as ALLMARKETS long format and build a **fixture spine** (1 row per fixture) using:
  - identifier columns: `fixture_key, league, match_date, home_team_name, away_team_name, ...`
  - plus a wide bookie snapshot from `_build_bookie_wide_by_fixture()` (FTR odds, BTTS yes/no odds, O25 over/under odds, plus CS top3 if present)

#### Optional ALLMARKETS merge (fixture-wide enrichment)
- `--allmarkets <BOOKIE_IMP*_ALLMARKETS_*.csv>`  
  Merge a fixture-level “wide snapshot” onto the current spine (requires `fixture_key` in src).
- `--emit-bookie-wide`  
  If `--allmarkets` is provided, also writes:  
  `bookie_wide_by_fixture.csv`

#### Optional V3 scoring from ModelStore
- `--score-v3`  
  Scores per-league V3 models from `ModelStore/<LeagueTag>/` and injects:
  - `prob_btts`, `prob_over25` (and also sets `*_best` equal to these)
  - `p_home, p_draw, p_away` (from a 3-class FTR model if available)
  - derived: `ftr_pick, ftr_conf, p_side_or_draw`
- `--modelstore ModelStore` (default: `ModelStore`)
- `--no-calibrators`  
  Disable calibrator application from `ModelStore/<LeagueTag>/calibrators`
- `--score-verbose`  
  Print per-league scoring diagnostics

---

### Column expectations (minimum viable spine)

The script is intentionally tolerant, but it works best when you have:

**Identifiers**
- `fixture_key` (preferred)
- `league`
- `match_date`
- `home_team_name`, `away_team_name` (or `home_team/away_team`)

**Probabilities (any of these; it selects best available)**
- BTTS: `prob_btts_v2`, `prob_btts`, `btts_confidence`, `adjusted_btts_confidence`, `prob_btts_pois`, `prob_btts_best`
- O2.5: `prob_over25_v2`, `prob_over25`, `over25_confidence`, `adjusted_over25_confidence`, `prob_over25_pois`, `prob_over25_best`

**FTR**
- ideally `p_home, p_draw, p_away` (or aliases accepted by `_best_ftr_probs()`)
- otherwise use `--score-v3` to inject them

**Lambdas (any alias; it selects best available)**
- home lambda: `lambda_home`, `home_goals_pred`, `home_lambda`, `lh`, etc.
- away lambda: `lambda_away`, `away_goals_pred`, `away_lambda`, `la`, etc.

---

### Enrichment step (what `enrich_v3()` adds)

After reading/merging/scoring, `enrich_v3()` computes a stable schema:

**Best probs (with Poisson fallback fill)**
- `prob_btts_best`
- `prob_over25_best`
- also ensures canonical `prob_btts` / `prob_over25` exist and are filled from `*_best`

**Lambdas**
- `lambda_home`, `lambda_away`

**FTR**
- `p_home, p_draw, p_away`
- `ftr_pick` in `{H,D,A}` and `ftr_conf` = max prob
- `p_side_or_draw` (derived as `p_draw + max(p_home,p_away)` if missing)

**BTTS artefact diagnostics**
- `p_home_ge1`, `p_away_ge1` (Poisson)
- `btts_indep` = `p_home_ge1 * p_away_ge1`
- `btts_gap` = `prob_btts_best - btts_indep`
- `btts_ratio` = `prob_btts_best / btts_indep`

**Team-goals**
- `prob_home_ge2_best`, `prob_away_ge2_best`, `prob_home_ge3_best`, `prob_away_ge3_best`
- canonical compat fill: `prob_home_ge2`, `prob_away_ge2`, `prob_home_ge3`, `prob_away_ge3`
- TG picks + confidence:
  - `tg_pick_ge2`, `tg_pick_ge2_conf`
  - `tg_pick_ge3`, `tg_pick_ge3_conf`
- Poisson tails used for some gates:
  - `p_team_ge2_pois`, `p_team_ge3_pois`

**U21 flags + score**
- `is_u21` (league name contains “U21”)
- `is_adult` = `1 - is_u21`
- `u21_o25_score` = `2*prob_over25_best + p_side_or_draw`

**Optional correct-score pick**
- if `cs1/cs1_p/cs2/cs2_p/cs3/cs3_p` exist:
  - `cs_pick`, `cs_pick_p`
- else schema-stable NaNs for these columns

---

### BTTS artefact gating (adult pools)

Before building adult packs, it runs:

- gap gate: `abs(btts_gap) <= --btts-max-gap` (default `0.10`)
- ratio gate (optional): `btts_ratio <= --btts-max-ratio` (default `1.20`)

Flags:
- `--btts-use-ratio` (default ON)
- `--btts-no-ratio` disables ratio, uses gap-only

If gating removes everything, it falls back to the ungated frame so outputs still write.

---

### Outputs (files written)

All outputs are written under:
- `--outdir <dir>`  
  Default: `predictions_output/V3_PACKS/<today_utc_YYYY-MM-DD>/`

#### Stack pool (default ON)
- `<stack_pool_name>` (default `stack_pool_tierA.csv`)
  - if exists and `--overwrite` is not set, it appends `<tag>_<date>` to avoid clobbering

#### Adult packs
- `PACK_PREMIUM_STRICT_BT{BT}_CLEAN_ADULT_v3_{tag}.csv`
- `TIER_S_MATCHUP_STRICT_BT{BT}_CLEAN_ADULT_v3_{tag}.csv`
- `mismatch_pool_tierB_{ud_band}_TG{min_tg}_v3_{tag}.csv`

Each of these also writes a `*_top200.csv` sibling, sorted by the relevant score:
- adult premium: `tier_score`
- adult matchup: `tier_score`
- mismatch pool: `tier_score`

#### FTR core
- `PACK_FTR_CORE_v3_{tag}.csv` (+ `_top200`, sorted by `ftr_score`)

#### U21 pools
- `U21_POOL_O25_v3_{tag}.csv` (+ `_top200`, sorted by `u21_o25_score`)
- `U21_POOL_TIER_S_O25_DC_v3_{tag}.csv`
- `U21_POOL_TIER_A_O25_DC_v3_{tag}.csv`
- `U21_POOL_TIER_B_O25_DC_v3_{tag}.csv`
(+ `_top200` for each, sorted by `u21_score`)

#### Team Goals packs
- `PACK_TG_GE2_v3_{tag}.csv` (+ `_top200`, sorted by `tg_ge2_score`)
- `PACK_TG_GE3_v3_{tag}.csv` (+ `_top200`, sorted by `tg_ge3_score`)

---

### CLI options (defaults)

**Source / mode**
- `--src` (required)
- `--src-is-allmarkets` (default: False)

**Scoring**
- `--score-v3` (default: False)
- `--modelstore ModelStore`
- `--no-calibrators` (default: False)
- `--score-verbose` (default: False)

**Output**
- `--outdir` (default: `predictions_output/V3_PACKS/<today>`)
- `--tag V3`

**BTTS artefact gating**
- `--btts-max-gap 0.10`
- `--btts-max-ratio 1.20`
- `--btts-use-ratio` (default ON)
- `--btts-no-ratio` (default OFF)

**Adult premium**
- `--thr-btts 0.62`
- `--thr-o25 0.55`
- `--thr-psod 0.78`
- `--min-tg-ge2 0.70`

**Adult matchup**
- `--thr-psod-matchup 0.80`
- `--min-tg-ge2-matchup 0.72`

**Mismatch**
- `--mismatch-ud-band wild`
- `--mismatch-min-tg-ge2 0.74`
- `--mismatch-min-btts 0.58`

**FTR core**
- `--thr-ftr 0.58`
- `--thr-psod-ftr 0.75`
- `--max-draw-ftr 0.34`

**TG packs**
- `--tg-ge2-min-prob 0.72`
- `--tg-ge2-min-psod 0.75`
- `--tg-ge2-max-draw 0.36`
- `--tg-ge3-min-prob 0.42`
- `--tg-ge3-min-psod 0.72`
- `--tg-ge3-max-draw 0.38`

**U21**
- `--u21-min-o25 0.58`
- `--u21-min-psod 0.80`
- `--u21-tier-s 2.14`
- `--u21-tier-a 2.10`
- `--u21-tier-b 2.00`

**Stack pool emission**
- `--emit-stack-pool` (default ON)
- `--no-emit-stack-pool` (default OFF)
- `--stack-pool-name stack_pool_tierA.csv`
- `--overwrite` (default OFF)

**ALLMARKETS merge**
- `--allmarkets <path>` (default None)
- `--emit-bookie-wide` (default OFF)

---

### Example commands

#### A) Build from ALLMARKETS long, score V3 models, emit packs + stack pool
```bash
python pack_builder.py \
  --src predictions_output/GOAL_TRACKER_RUN/ALLMARKETS_2025_01/BOOKIE_IMP68_ALLMARKETS_2025-01-01_to_2025-01-31.csv \
  --src-is-allmarkets \
  --score-v3 \
  --modelstore ModelStore \
  --outdir predictions_output/TEST_RUN/V3_PACKS_2025_01 \
  --tag TEST_RUN

B) Build from spine and merge bookie-wide snapshot from ALLMARKETS

python pack_builder.py \
  --src predictions_output/TEST_RUN/spine.csv \
  --allmarkets predictions_output/GOAL_TRACKER_RUN/ALLMARKETS_2025_01/BOOKIE_IMP68_ALLMARKETS_2025-01-01_to_2025-01-31.csv \
  --emit-bookie-wide \
  --outdir predictions_output/TEST_RUN/V3_PACKS_2025_01 \
  --tag TEST_RUN


⸻

Notes / gotchas
	•	FTR Core pack requires FTR probabilities (p_home/p_draw/p_away).
If your source doesn’t contain them, use --score-v3, otherwise FTR pack will be weak/empty.
	•	Adult pools are built from the artefact-gated dataframe (kept).
U21 pools use the original enriched dataframe (df) because they are O2.5-driven.
	•	Every output CSV also writes a *_top200.csv (head or score-sorted top 200).








# OG Post-`book_allmarkets.py` Pipeline (V3) — Top-Level README (Skeleton)

This README documents the current “post `book_allmarkets.py`” workflow for building **ALLMARKETS**, generating **V3 packs/pools**, and producing **truth-aligned match cards** for goal tracking + public outputs.

Core scripts in this slice:
- `book_allmarkets.py`
- `pack_builder.py` (V3 packs/pools)
- `deploy_rulebook.py` (legacy V1 deploy preset)
- `filter_deploy_by_rulebook.py` (odds-fill + April gating)
- `backtest_deploy_csv.py` (hit/ROI scoring vs Matches truth)
- `og_goal_tracker.py` (truth-aligned match cards + reporting)
- `og_public_match_cards.py` (public match cards export)
- `season_public_match_cards_runner.py` (month loop runner)
- `score_public_match_cards.py` (month/season scoring tables)

---

## 0) What this pipeline produces

### Primary artefacts
1) **ALLMARKETS CSV (long format)**  
   One row per fixture *per market* (FTR, O2.5, BTTS, etc.), with odds/implieds and model fields (where available).

1b) **Gated deploy CSV (long format)** via `filter_deploy_by_rulebook.py`
A filtered subset of ALLMARKETS rows that:
- has `bookie_od` populated per market (deterministic odds mapping)
- includes `bookie_od_source` (`bookmaker` vs `model_fair`)
- applies the “April profile” odds bands + top-quantile score gate

Outputs:
- `...__GATED.csv`
- `...__GATED_SUMMARY.csv`
- `...__RULEBOOK_SNAPSHOT.json` (the exact params used)

2) **V3 packs/pools (fixture-level outputs)** via `pack_builder.py`  
   Deterministic “ready-to-use” pools for:
   - Adult premium / matchup / mismatch pools
   - FTR core pack
   - Team-goals packs (TG≥2, TG≥3)
   - U21 O2.5 pool + S/A/B tiers
   - A “stack pool” snapshot (enriched pool) to act as the canonical V3 pool spine

3) **Goal Tracker match cards** via `og_goal_tracker.py`  
   Fixture-level match cards with *truth columns* (actual goals, actual FTR, totals) for scored fixtures.
   This is the “ground truth spine” you use for model drift checks.

4) **Public match cards** via `og_public_match_cards.py`  
   A public-facing export format (schema-safe, cleaner fields, no internal-only columns).

---

## 1) Conceptual flow (high level)

### A) Build ALLMARKETS
`book_allmarkets.py` generates:
- `BOOKIE_IMP*_ALLMARKETS_<date_from>_to_<date_to>.csv`

This is your “raw long-format market stack”.

### B) Build V3 fixture spine + packs/pools
`pack_builder.py` consumes either:
- the ALLMARKETS long file (`--src-is-allmarkets`) **or**
- an already enriched fixture spine CSV

Then it:
1) Builds/normalises a fixture spine
2) Optionally scores V3 models from `ModelStore/` (`--score-v3`)
3) Runs enrichment (`enrich_v3()`)
4) Applies BTTS artefact gating (adult pools)
5) Emits packs/pools + stack pool

### B2) Gate ALLMARKETS for deployment + ROI backtests (April profile)
`filter_deploy_by_rulebook.py` takes the ALLMARKETS long file and produces a gated deploy subset.

What it does:
1) **Populate `bookie_od`** per market:
   - FTR: HOME/DRAW/AWAY -> `od_home/od_draw/od_away` (fallback `odds_ft_*`)
   - OU25: OVER25/UNDER25 -> `od_over/od_under` (fallback `odds_ft_over25/odds_ft_under25`)
   - BTTS: YES/NO -> `od_yes/od_no` (fallback `odds_btts_yes/odds_btts_no`)
   - TG15/TG25: if bookmaker odds are missing, derive **proxy fair odds** from model probabilities
     (Poisson tails preferred, confidence fallbacks), with conservative caps + gamma + vig.

2) **Apply odds-band gates (April defaults):**
   - BTTS: `bookie_od <= 1.62`
   - OU25: `(1.24–1.72) OR (1.82–1.91)`
   - FTR:  `bookie_od >= 2.14`
   - TG caps: `tg15 <= 2.50`, `tg25 <= 6.00` (set max <= 0 to disable)

3) **Apply top-score gate within each market** (default top 30% => `q=0.70`).

It writes a `__RULEBOOK_SNAPSHOT.json` alongside the gated CSV so runs are reproducible.

Then `backtest_deploy_csv.py` scores the gated deploy file against `Matches/*` truth and produces:
- `...__BACKTEST.csv`
- `...__BACKTEST_SCORED.csv`
- `...__BACKTEST_SUMMARY.csv`

Important note on TG markets:
- TG15/TG25 ROI is **proxy/fair-odds ROI** until real bookmaker odds are available.
- Always interpret TG ROI separately from markets with true bookmaker odds.

### C) Generate Goal Tracker match cards
`og_goal_tracker.py` consumes:
- the fixture spine / stack pool / packs (depending on your run mode)
- your results/truth source (completed fixtures)

Outputs:
- `match_cards.csv` (truth-aligned)
- optionally month-range directories

### D) Generate Public match cards
`og_public_match_cards.py` consumes:
- `match_cards.csv` (Goal Tracker output)
and emits:
- `public_match_cards.csv`

---

## 2) Directory layout (current conventions)

Common outputs live under:
- `predictions_output/`

Typical structure (example):
- `predictions_output/GOAL_TRACKER_RUN/ALLMARKETS_2025_01/`
  - `BOOKIE_IMP68_ALLMARKETS_2025-01-01_to_2025-01-31.csv`

- `predictions_output/TEST_RUN/V3_PACKS_2025_01/`
  - `stack_pool_tierA.csv` (or tagged variant)
  - `PACK_PREMIUM_STRICT_BT62_CLEAN_ADULT_v3_TEST_RUN.csv`
  - `PACK_FTR_CORE_v3_TEST_RUN.csv`
  - `PACK_TG_GE2_v3_TEST_RUN.csv`
  - `U21_POOL_TIER_S_O25_DC_v3_TEST_RUN.csv`
  - plus `_top200.csv` siblings

- `predictions_output/TEST_RUN/GOAL_TRACKER_2024_08_to_2025_01/2024-08-01_to_2025-01-31/`
  - `match_cards.csv`

- `predictions_output/TEST_RUN/PUBLIC_MATCH_CARDS_2024_08_to_2025_01/2024-08-01_to_2025-01-31/`
  - `public_match_cards.csv`

---

## 3) Script responsibilities

### 3.1 `book_allmarkets.py`
**Purpose:** build ALLMARKETS long-format file(s) for a date window.  
**Outputs:** `BOOKIE_IMP*_ALLMARKETS_<from>_to_<to>.csv`

Key characteristics:
- Long format: multiple rows per fixture (one per market)
- Contains bookie odds/implieds and (optionally) model fields / lambdas / correct-score top picks

---

### 3.2 `pack_builder.py` (V3)
**Purpose:** produce a deterministic V3 fixture pool + packs from either ALLMARKETS or an enriched spine.  
**Key features:**
- `--src-is-allmarkets` to build a fixture spine from ALLMARKETS
- `--score-v3` to inject V3 model probabilities from `ModelStore/<LeagueTag>/`
- enrichment (`prob_*_best`, lambdas, TG probs/picks, BTTS artefact metrics, etc.)
- BTTS artefact gating for adult pools (gap/ratio suppression)
- emits packs and `_top200` files

---

### 3.2a `filter_deploy_by_rulebook.py`
**Purpose:** create a gated deploy subset from an ALLMARKETS file, suitable for backtesting and deployment selection.

**Key outputs:**
- `...__GATED.csv`
- `...__GATED_SUMMARY.csv`
- `...__RULEBOOK_SNAPSHOT.json`

**Key columns added/standardised:**
- `bookie_od` (deterministic odds for the `bookie_pick`)
- `bookie_od_source` (`bookmaker` or `model_fair`)

### 3.2b `backtest_deploy_csv.py`
**Purpose:** join deploy rows to realised outcomes in `Matches/*` and compute:
- **hit-rate** (win %)
- **avg_od** (average odds of selected legs)
- **ROI** (flat 1u staking)

**ROI formula (decimal odds):**
- profit per bet = `correct*(od - 1) - (1 - correct)`
- ROI = mean(profit per bet)

**Outputs:**
- `...__BACKTEST.csv` (joined rows)
- `...__BACKTEST_SCORED.csv` (row-level scoring)
- `...__BACKTEST_SUMMARY.csv` (aggregates)

Planned hardening:
- break out metrics by `bookie_od_source` so “real odds ROI” and “proxy fair odds ROI” are reported separately.

---

### 3.3 `og_goal_tracker.py`
**Purpose:** generate **truth-aligned match cards** for scored fixtures, used to:
- verify pipeline parity
- compute drift metrics
- provide a clean “fixture truth spine” for evaluation

Outputs typically include:
- `match_cards.csv`
with columns like:
- identifiers (`league`, `fixture_key`, teams, date)
- truth (`actual_home_goals`, `actual_away_goals`, `actual_total_goals`, `actual_ftr`)
- optionally prediction fields (depending on your build mode)

---

### 3.4 `og_public_match_cards.py`
**Purpose:** produce a **public-safe** export from goal-tracker match cards.  
Outputs:
- `public_match_cards.csv`

Design goals:
- stable schema
- no internal-only dev columns
- consistent naming

---

## 4) “One month” run (canonical)

### Step 1 — Build ALLMARKETS (month window)
```bash
# Example window: January 2025
python book_allmarkets.py \
  --date-from 2025-01-01 \
  --date-to 2025-01-31 \
  --outdir predictions_output/GOAL_TRACKER_RUN/ALLMARKETS_2025_01
```

Expected output:
	•	predictions_output/GOAL_TRACKER_RUN/ALLMARKETS_2025_01/BOOKIE_IMP*_ALLMARKETS_2025-01-01_to_2025-01-31.csv

### Step 1b — Gate ALLMARKETS (April profile) + backtest
```bash
python filter_deploy_by_rulebook.py \
  --deploy-csv "predictions_output/2025-04/BOOKIE_IMP0_ALLMARKETS_2025-04-01_to_2025-04-30.csv" \
  --outdir "predictions_output/2025-04"

python backtest_deploy_csv.py \
  --deploy-csv "predictions_output/2025-04/BOOKIE_IMP0_ALLMARKETS_2025-04-01_to_2025-04-30__GATED.csv" \
  --matches-root "Matches"
```

Outputs:
- `...__GATED.csv`, `...__GATED_SUMMARY.csv`, `...__RULEBOOK_SNAPSHOT.json`
- `...__BACKTEST.csv`, `...__BACKTEST_SCORED.csv`, `...__BACKTEST_SUMMARY.csv`

### Step 2 — Build V3 packs/pools (fixture spine) from ALLMARKETS

python pack_builder.py \
  --src predictions_output/GOAL_TRACKER_RUN/ALLMARKETS_2025_01/BOOKIE_IMP68_ALLMARKETS_2025-01-01_to_2025-01-31.csv \
  --src-is-allmarkets \
  --score-v3 \
  --modelstore ModelStore \
  --outdir predictions_output/TEST_RUN/V3_PACKS_2025_01 \
  --tag TEST_RUN

Key outputs (examples):
	•	stack_pool_tierA.csv
	•	PACK_PREMIUM_STRICT_BT62_CLEAN_ADULT_v3_TEST_RUN.csv
	•	PACK_FTR_CORE_v3_TEST_RUN.csv
	•	PACK_TG_GE2_v3_TEST_RUN.csv
	•	U21_POOL_TIER_S_O25_DC_v3_TEST_RUN.csv

Step 3 — Generate Goal Tracker match cards (truth spine)

python og_goal_tracker.py \
  --src predictions_output/TEST_RUN/V3_PACKS_2025_01/stack_pool_tierA.csv \
  --date-from 2025-01-01 \
  --date-to 2025-01-31 \
  --outdir predictions_output/TEST_RUN/GOAL_TRACKER_2025_01 \
  --tag TEST_RUN

Expected output:
	•	predictions_output/TEST_RUN/GOAL_TRACKER_2025_01/2025-01-01_to_2025-01-31/match_cards.csv

Step 4 — Generate Public match cards

python og_public_match_cards.py \
  --src predictions_output/TEST_RUN/GOAL_TRACKER_2025_01/2025-01-01_to_2025-01-31/match_cards.csv \
  --outdir predictions_output/TEST_RUN/PUBLIC_MATCH_CARDS_2025_01 \
  --tag TEST_RUN

Expected output:
	•	predictions_output/TEST_RUN/PUBLIC_MATCH_CARDS_2025_01/2025-01-01_to_2025-01-31/public_match_cards.csv

⸻

5) Three-month sweep (canonical drift check)

Goal: build 3 consecutive months of:
	•	ALLMARKETS
	•	V3 packs (and stack pool)
	•	Goal tracker match cards
	•	Public match cards
then aggregate into a single drift summary CSV.

5.1 Standard sweep sequence (repeat monthly)

For each month window:
  1) `book_allmarkets.py` → ALLMARKETS long
  2) `filter_deploy_by_rulebook.py` → `__GATED.csv` + snapshot
  3) `backtest_deploy_csv.py` → backtest summaries
  4) `pack_builder.py` → stack pool + packs (optional per month)
  5) `og_goal_tracker.py` → match_cards.csv (truth-aligned)
  6) `og_public_match_cards.py` → public_match_cards.csv

(Exact “3-month sweep command set” goes in the next section once you paste the exact CLI flags you’re using for book_allmarkets.py + goal tracker/public scripts, or we keep it in this generic form and align later.)
## 10) April profile (current defaults)
These are the current working defaults discovered on the April 2025 window and saved via `__RULEBOOK_SNAPSHOT.json`.

- BTTS max odds: `<= 1.62`
- OU25 odds bands: `(1.24–1.72) OR (1.82–1.91)`
- FTR min odds: `>= 2.14`
- Top quantile per market by `score`: top 30% (`q=0.70`)
- TG caps (proxy fair odds until bookie odds exist):
  - TG15 max odds: `<= 2.50`
  - TG25 max odds: `<= 6.00`
  - fair-odds parameters: `tg_vig`, `tg15_cap`, `tg25_cap`, `tg_gamma` (see snapshot)

Operational guidance:
- Treat TG ROI as **proxy signal** until bookmaker odds are available.
- Prefer reporting ROI by `bookie_od_source` to separate real vs proxy odds.

⸻

6) Drift aggregation (single CSV summary across months)

6.1 What the aggregator should report

Per month (and optionally per league):
	•	fixture counts (total / scored)
	•	truth coverage rate
	•	presence/coverage of key prob columns (btts/o25 + poisson)
	•	MAE / calibration checks (if you include predictions in match cards)
	•	“schema parity” checks (are key columns present consistently?)

6.2 Aggregator script (drop-in)

Create: tools/aggregate_month_summaries.py

#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def num(s):
    return pd.to_numeric(s, errors="coerce")

def month_from_path(p: Path) -> str:
    # expects .../<YYYY-MM-DD>_to_<YYYY-MM-DD>/match_cards.csv
    # returns the folder name as the month key
    return p.parent.name

def summarize_match_cards(match_cards_csv: Path) -> dict:
    df = pd.read_csv(match_cards_csv)
    out = {}
    out["month_window"] = month_from_path(match_cards_csv)
    out["rows"] = int(len(df))
    out["fixtures"] = int(df["fixture_key"].nunique()) if "fixture_key" in df.columns else int(len(df))

    # scored fixture definition
    scored_mask = df.get("actual_ftr", pd.Series([np.nan]*len(df))).notna()
    out["scored_rows"] = int(scored_mask.sum())
    out["scored_rate"] = float(scored_mask.mean()) if len(df) else 0.0

    # truth coverage
    for c in ["actual_home_goals","actual_away_goals","actual_total_goals","actual_ftr"]:
        if c in df.columns:
            out[f"{c}_nonnull_rate"] = float(df[c].notna().mean())
        else:
            out[f"{c}_nonnull_rate"] = np.nan

    # optional probs coverage (these may or may not exist depending on your goal tracker schema)
    prob_cols = ["prob_over25","prob_btts","prob_over25_pois","prob_btts_pois","prob_over25_best","prob_btts_best"]
    for c in prob_cols:
        out[f"{c}_nonnull_rate"] = float(num(df[c]).notna().mean()) if c in df.columns else np.nan

    # league breakdown (optional, lightweight)
    if "league" in df.columns:
        out["leagues"] = int(df["league"].astype(str).nunique())
    else:
        out["leagues"] = np.nan

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing multiple GOAL_TRACKER month outputs")
    ap.add_argument("--pattern", default="match_cards.csv", help="Filename pattern to find (default: match_cards.csv)")
    ap.add_argument("--out", required=True, help="Output CSV path for aggregated summary")
    args = ap.parse_args()

    root = Path(args.root)
    paths = sorted(root.rglob(args.pattern))

    rows = []
    for p in paths:
        try:
            rows.append(summarize_match_cards(p))
        except Exception as e:
            rows.append({"month_window": str(p), "error": str(e)})

    summ = pd.DataFrame(rows).sort_values("month_window")
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(outp, index=False)

    print(f"Wrote: {outp}")
    print(summ)

if __name__ == "__main__":
    main()

Run it like:

python tools/aggregate_month_summaries.py \
  --root predictions_output/TEST_RUN/GOAL_TRACKER_2024_08_to_2025_01 \
  --out predictions_output/TEST_RUN/DRIFT_SUMMARY_2024_08_to_2025_01.csv


⸻

7) Parity checks (Goal Tracker vs Public match cards)

You already have a parity harness proving:
	•	fixture keys match
	•	truth columns match (MAE=0)
	•	actual_ftr match rate = 1.0

Recommended: keep that script under:
	•	tools/verify_goaltracker_vs_public.py

…and treat it as a required CI-ish check for any schema changes.

⸻

8) TODOs / next hardening steps
	•	Freeze canonical schemas:
	•	match_cards.csv column contract
	•	public_match_cards.csv column contract
	•	pack outputs column contract
	•	Pin “month sweep” runner script (single command runs all months)
	•	Add per-league drift slice to aggregator
	•	Add prediction-vs-truth metrics if predictions are present in match cards
	•	Add “missing ModelStore coverage” reporting (which leagues skipped in --score-v3)

⸻

9) Quick reference

Common output artefacts
	•	ALLMARKETS long:
	•	BOOKIE_IMP*_ALLMARKETS_<from>_to_<to>.csv
	•	V3 packs/pools:
	•	stack_pool_tierA.csv
	•	PACK_PREMIUM_...csv, PACK_FTR_CORE...csv, PACK_TG_GE2...csv, etc.
	•	Goal tracker:
	•	match_cards.csv
	•	Public:
	•	public_match_cards.csv

⸻

Appendix A — Script-by-script CLI docs (placeholder)
	• book_allmarkets.py flags: TODO: paste canonical flags
	• filter_deploy_by_rulebook.py flags: see defaults + snapshot output
	• backtest_deploy_csv.py flags: TODO: paste canonical flags
	• og_goal_tracker.py flags: TODO: paste canonical flags
	• og_public_match_cards.py flags: TODO: paste canonical flags

(Once you paste the exact CLIs you’re running for these three, this README becomes “copy/paste runnable” with no guesswork.)





5) What to update in the “Post-bookie_allmarkets pipeline” README

Add a new section after deploy generation:

Deploy gating + backtest chain (canonical)

python filter_deploy_by_rulebook.py \
  --deploy-csv predictions_output/2025-04/BOOKIE_IMP0_ALLMARKETS_2025-04-01_to_2025-04-30__DEPLOY_PRESET_V1.csv \
  --outdir predictions_output/2025-04

python backtest_deploy_csv.py \
  --deploy-csv predictions_output/2025-04/BOOKIE_IMP0_ALLMARKETS_2025-04-01_to_2025-04-30__GATED.csv \
  --matches-root Matches

Then add a “Results ledger” snippet showing April’s headline numbers (hit/ROI per market).

=

What accuracies we now have (from April gated)
	•	FTR: 62.7%
	•	OU25: 93.1%
	•	BTTS: 91.7%
	•	TG15: 97.6%
	•	TG25: 78.9%
	•	All combined: 84.6%

How they’re calculated
	•	correct per market (rules you already saw in code)
	•	hit = mean(correct)
	•	roi = mean(correct * odds - 1)
	•	avg_od = mean(odds)


) What April 2025 has proven (now that we see products + leagues)

Overall edge is real at the aggregate level
	•	285 bets
	•	84.56% hit
	•	Avg odds 1.864
	•	ROI +0.490

That’s a big sample for a gated deploy month.

Product breakdown is extremely informative

OU25_UNDER25
	•	n=30, hit 1.000, avg_od 1.697, ROI +0.697
	•	This is basically “the rulebook is murdering false overs” — under25 is being selected only when it’s very safe.

OU25_OVER25
	•	n=72, hit 0.903, avg_od 1.673, ROI +0.508
Still strong, but noticeably weaker than the under side — useful signal for future tuning.

BTTS_NO
	•	n=14, hit 1.000, avg_od 1.589, ROI +0.589
Again: “NO” is being picked in very clean situations.

BTTS_YES
	•	n=22, hit 0.864, avg_od 1.544, ROI +0.326
YES is always harder; still profitable.

TG25
	•	Away TG25: n=10, hit 1.000, avg_od 2.155, ROI +1.155 (huge, but small n)
	•	Home TG25: n=28, hit 0.714, avg_od 2.215, ROI +0.246
This split screams: your away-team 3+ selection logic is extremely sharp, but we need more months.

League slice tells us where the rulebook is fragile

These are the only real “warning flags”, and they’re both small-ish sample problems:
	•	England Premier League: n=4, hit 0.25, ROI -0.522
Too small to conclude “EPL is bad”, but it is enough to say: your gates might be picking the wrong kind of EPL fixtures (e.g., overconfident spots where the market is efficient).
	•	France Ligue 1: n=11, hit 0.545, ROI -0.090
This is the first “non-trivial” negative. Still small, but meaningful enough to investigate.
