# OG / BAWA Runbook (One-Page)

This repo produces **league-level football picks** (FTR / OU2.5 / BTTS / Team Goals) plus **Poisson λ + Correct Score top-3** diagnostics, then filters into a deterministic deploy shortlist.

---

## Repo layout (what matters)
- `Matches/<League>/fd_odds_enriched.csv`
  Main merged match + odds + engineered features per league (history + forward fixtures as your feed allows).
- `Matches/<League>/fd_odds_enriched_synth.csv`
  Same, but with missing odds sides (e.g. **Under2.5**) filled where possible by conservative synthesis.
- `ModelStore/<LeagueTag>/*.pkl`
  Trained models (FTR / side models + specialist heads like GE2/GE3/FTS).
- `predictions_output/<YYYY-MM-DD>/`
  Daily outputs (ALLMARKETS pool, deploy shortlist, optional slips/legs CSVs).

**Key scripts**
- `train_markets.py` — trains specialist heads (GE2/GE3/FTS)
- `make_fd_odds_enriched_synth.py` — builds synth-enriched league files
- `bookie_allmarkets.py` — builds ALLMARKETS pool + λ + CS + TG coherence + draw/chaos flags
- `deploy_rulebook.py` — deterministic deploy shortlist (CSV + MD)
- `acca_builder.py` — optional: build slips/legs + emits `cs_note` in legs CSV

---

## What each script does (quick)

### 1) `train_markets.py`
Trains **specialist heads** per league:
- `home_ge2`, `away_ge2` (team ≥2 goals)
- `home_ge3`, `away_ge3` (team ≥3 goals)
- `home_fts`, `away_fts` (fail-to-score)

Writes:
- `ModelStore/<LeagueTag>/<market>.pkl` and `<market>_v2.pkl`
- `ModelStore/<LeagueTag>_market_thresholds.json`

### 2) `make_fd_odds_enriched_synth.py`
Creates `fd_odds_enriched_synth.csv` per league by filling missing odds sides where rules are confident (e.g. missing **Under2.5** but Over2.5 exists).

**Safety:** synthesis is for **coverage** and **diagnostics**. Gate hard and treat as optional.

### 3) `bookie_allmarkets.py`
Builds the master pick pool.

Emits:
- FTR / OU2.5 / BTTS picks when odds exist
- TG15/TG25 picks (model-only; unpriced)

Attaches **λ + correct-score + tails**:
- `home_goals_pred`, `away_goals_pred`, `lambda_home`, `lambda_away`, `exp_goals_sum`, `p00_est`
- `cs1/cs1_p`, `cs2/cs2_p`, `cs3/cs3_p`
- `p_home_pois/p_draw_pois/p_away_pois`, `cs_trunc_mass_0_6`
- TG Poisson tails: `pois_home_ge2/3`, `pois_away_ge2/3`

Attaches **TG coherence**:
- `tg_pois_ok`, `tg_pois_gap` (vetoes “GE says huge but Poisson says tiny”)

Attaches **Draw/Chaos Risk** (used to tag “NOT GLUE” even when pick is HOME):
- `draw_risk_flag`, `chaos_risk_flag`, `not_glue_flag`, `draw_chaos_score`

Writes:
- `predictions_output/<today>/BOOKIE_IMP{xx}_ALLMARKETS_<from>_to_<to>.csv`

**Note on FTR feed timing:** when `--markets` includes `ftr`, the runner prefers the source file with the most **valid 1X2 odds rows in-window** (so it won’t silently drop FTR just because one enriched file is missing 1X2 odds).

### 4) `deploy_rulebook.py`
Deterministic filtering (“deploy shortlist”):
- applies market-specific rules using bookie implieds + model probs + stability/structure features

Writes:
- `<input>__DEPLOY_PRESET_V1.csv`
- `<input>__DEPLOY_PRESET_V1.md`

### 5) `acca_builder.py` (optional)
- `build`: creates slips + `slips_legs_<tag>.csv` (**includes `cs_note`**)
- `backtest`: joins to realised Matches and outputs ROI/hit-rate summaries

You can use `build` even if you don’t want “bettable slips” — it’s a convenient **legs view** with CS notes.

---

## Accuracy snapshot (Aug–Dec 2025, monthly folds)

### A) Walk-forward coverage accuracy (all picks)
Latest batch scoreboard (monthly folds, Aug–Dec 2025; `tau=1.50`, `xog_k=0.25`):
- Norway_Eliteserien: **0.637** (n=113)
- Spain_La_Liga: **0.632** (n=516)
- England_Premier_League: **0.599** (n=558)
- Portugal_Liga: **0.589** (n=426)
- Germany_Bundesliga: **0.583** (n=405)
- France_Ligue_1: **0.576** (n=432)
- Netherlands_Eredivisie: **0.575** (n=456)
- Japan_J1: **0.562** (n=130)
- Scotland_Premiership: **0.546** (n=348)
- Italy_Serie_A: **0.498** (n=498)
- USA_MLS: **0.037** (n=294)  ← treat as **BROKEN DATA PIPELINE** until audited

These are *coverage* accuracies — i.e., the consensus produces a pick for every fixture and we score all of them.

### B) Where the ~70% accuracy comes from (deploy-grade bucket)
The **~70%** number is not the coverage accuracy — it comes from filtering to the **deploy-grade subset**:

**Deploy-grade definition (used in backtests):**
- `consensus_lane == "SIDE"`
- `xog_tier ∈ {"ELITE", "STRONG"}`

**Observed (EPL walk-forward Aug–Dec 2025):**
- Deploy-grade subset: **0.72** accuracy (n=200)

This is the subset you deploy most often (favourites / separated markets) while still keeping full-coverage reporting for everything else.

### C) Where the ~90% accuracy comes from (bankers bucket)
The **~90%** number comes from the **bankers bucket**:

**Bankers definition:**
- Rank all fixtures by `xog_spread` (descending)
- Take **Top-N** (typically Top 50 on big windows)

**Observed Top-50 by `xog_spread` (walk-forward Aug–Dec 2025):**
- EPL: **0.94** (Top50)
- Spain La Liga: **0.94** (Top50)
- Germany Bundesliga: **0.88** (Top50)

In the per-league quintile views, the **top quintile** of `xog_pick_score` and `xog_spread` is consistently the highest-accuracy band.

### D) The exact thresholds used to produce these buckets

**Consensus defaults (baked):**
- `tau = 1.50`
- `xog_k = 0.25`
- lane routing:
  - `close_spread_max = 0.09`
  - `close_ipd_max = 0.11`
  - `side_spread_min = 0.16`
  - `side_ipd_min = 0.20`

**Deploy-grade thresholds (XOG tier function used by `ftr_consensus.py`):**

- Lane = `SIDE`
  - `ELITE` if `xog_pick_score >= 2.30` **and** `xog_spread >= 1.10`
  - `STRONG` if `xog_pick_score >= 2.20` **and** `xog_spread >= 0.95`
  - `MEDIUM` if `xog_pick_score >= 2.05` **and** `xog_spread >= 0.75`
  - else `WEAK`

- Lane = `MID`
  - `ELITE` if `xog_pick_score >= 1.70` **and** `xog_spread >= 0.20`
  - `STRONG` if `xog_pick_score >= 1.69` **and** `xog_spread >= 0.17`
  - `MEDIUM` if `xog_pick_score >= 1.67` **and** `xog_spread >= 0.12`
  - else `WEAK`

- Lane = `CLOSE`
  - `ELITE` if `xog_pick_score >= 1.80` **and** `xog_spread >= 0.26`
  - `STRONG` if `xog_pick_score >= 1.74` **and** `xog_spread >= 0.20`
  - `MEDIUM` if `xog_pick_score >= 1.70` **and** `xog_spread >= 0.13`
  - else `WEAK`

**Bankers threshold:**
- Sort by `xog_spread` and select Top-N (`--bankers-n N`).

> Important: these thresholds are designed to be **monotonic selectors** (higher XOG score/spread ⇒ higher accuracy). The whole point is that you can publish full coverage but deploy only the strongest tier bands.

---

### E) CloseMatch deploy preset gates (for tight matches)
The CLOSE deploy preset is intentionally simple and deterministic:

**Eligible rows:**
- `market == ftr`
- `close_match_flag == 1`
- `pool_tier == ANCHOR_CAND`
- `candidate_rank <= 2`

**Default gates:**
- `bookie_implied (raw) >= 0.18`
- `model_p_for_bookie >= 0.27`
- `gap_used >= 0.02`

CLI knobs:
- `--close-implied-min`
- `--close-pmin`
- `--close-gap-min`
- `--close-max-ftr`

---

## Quick verification checks

### 1) Confirm CS + λ columns exist in ALLMARKETS
```bash
python - <<'PY'
import pandas as pd, glob
p=sorted(glob.glob("predictions_output/*/BOOKIE_IMP*_ALLMARKETS_*.csv"))[-1]
df=pd.read_csv(p)
cols=["home_goals_pred","away_goals_pred","exp_goals_sum","cs1","cs1_p","cs2","cs2_p","cs3","cs3_p",
      "p_home_pois","p_draw_pois","p_away_pois","cs_trunc_mass_0_6"]
print("FILE:", p)
print({c:(c in df.columns) for c in cols})
PY
```

### 2) Confirm TG Poisson coherence is present (and not vetoing everything)
```bash
python - <<'PY'
import pandas as pd, glob
p=sorted(glob.glob("predictions_output/*/BOOKIE_IMP*_ALLMARKETS_*.csv"))[-1]
df=pd.read_csv(p)
tg=df[df["market"].astype(str).str.lower().isin(["tg15","tg25"])].copy()
print("FILE:", p)
print("tg rows:", len(tg))
for c in ["pois_home_ge2","pois_away_ge2","pois_home_ge3","pois_away_ge3","tg_pois_ok","tg_pois_gap"]:
    print(c, "OK" if c in df.columns else "MISSING")
print("vetoed:", int((pd.to_numeric(tg.get("tg_pois_ok",1), errors="coerce").fillna(1)==0).sum()))
PY
```

### 3) Confirm Draw/Chaos flags exist (FTR rows)
```bash
python - <<'PY'
import pandas as pd, glob
p=sorted(glob.glob("predictions_output/*/BOOKIE_IMP*_ALLMARKETS_*.csv"))[-1]
df=pd.read_csv(p)
want=["draw_risk_flag","chaos_risk_flag","not_glue_flag","draw_chaos_score"]
print("FILE:", p)
print({c:(c in df.columns) for c in want})
ftr=df[df["market"].astype(str).str.lower().eq("ftr")].copy()
if not ftr.empty:
    cols=[c for c in ["fixture_key","bookie_pick","confidence_draw","ftr_margin","draw_chaos_score","draw_risk_flag","chaos_risk_flag","not_glue_flag"] if c in ftr.columns]
    print(ftr[cols].head(25).to_string(index=False))
else:
    print("(no FTR rows in this window)")
PY
```

---

## Troubleshooting (fast)

### “rows_in_window = 0” for a league
- your feed likely hasn’t populated fixtures that far ahead

```bash
python - <<'PY'
import pandas as pd
p="Matches/England Premier League/fd_odds_enriched.csv"
df=pd.read_csv(p, low_memory=False)
md=pd.to_datetime(df.get("match_date", df.get("date_GMT", None)), errors="coerce", utc=True)
lo=pd.Timestamp("2026-01-07", tz="UTC"); hi=pd.Timestamp("2026-01-08", tz="UTC")
print(int(((md>=lo)&(md<=hi)).sum()))
PY
```

### “OU25/BTTS missing” in ALLMARKETS
- odds columns may be missing/invalid (0/NaN) in that window
- `bookie_allmarkets.py` will fall back to OVER-only for OU25 if under odds absent (depending on safeguards)

### “Model load errors” (wrapper/pickle)
- you already moved to `--calibration none` in `train_markets.py` to avoid custom wrapper unpickling

### “TG outputs nonsense”
- run with `--tg-use-dir-gate`
- ensure Poisson coherence is present (`pois_*` cols)
- check `tg_pois_gap` (big gap = mismatch)

---

## Safety notes
- **Synth odds:** treat as coverage tooling, not “must bet”. Gate hard (conf thresholds + sanity checks).
- **Unpriced legs (TG):** fine for accuracy tracking and builder notes; ROI/staking must exclude or handle separately.
- **Draw/Chaos flags:** use them as *pool hygiene* (“NOT GLUE”), not as a claim that a draw is “predicted”.

---

## V2/V3 daily run order (keep)

This is the **existing** V2/V3 operating flow. Keep using it for your proven deploy path; the CloseMatch + XOG layers sit alongside it.

### 0) (Optional) Build synth-enriched league files
Only if you want missing odds sides (e.g. Under2.5) filled conservatively for coverage/diagnostics.

```bash
python make_fd_odds_enriched_synth.py
```

### 1) (Optional) Retrain specialist heads (GE2/GE3/FTS)
Run when you’ve added completed matches or changed feature/whitelist/leak policy.

```bash
python train_markets.py \
  --leagues "England Premier League,Germany Bundesliga,Spain La Liga,Italy Serie A,France Ligue 1,Netherlands Eredivisie,Portugal Liga,Belgium Pro,Scotland Premiership" \
  --matches-csv "Matches/England Premier League/fd_odds_enriched_synth.csv" \
  --markets "home_ge2,away_ge2,home_ge3,away_ge3,home_fts,away_fts" \
  --calibration sigmoid
```

### 2) Generate ALLMARKETS pool (master output)
```bash
python bookie_allmarkets.py \
  --date-from 2026-01-19 --date-to 2026-01-23 \
  --leagues "England Premier League,Italy Serie A,Spain La Liga,Germany Bundesliga,France Ligue 1,Netherlands Eredivisie,Portugal Liga,Belgium Pro,Scotland Premiership" \
  --markets "ftr,ou25,btts,tg15,tg25" \
  --tg-use-dir-gate \
  --debug
```

### 3) Deploy shortlist (V1)
```bash
python deploy_rulebook.py \
  --src "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv" \
  --preset V1 \
  --debug
```

### 4) CloseMatch shortlist (CLOSE)
```bash
python deploy_rulebook.py \
  --src "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv" \
  --preset CLOSE \
  --debug
```

### 5) FTR coverage + ranking (Consensus/XOG)
```bash
python ftr_consensus.py \
  --src "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv"
```

### 6) Optional: legs view with correct-score notes
Use this even if you’re not building slips — it’s a convenient “legs table” view.

```bash
python acca_builder.py build \
  --pool "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv" \
  --tag "CS_VIEW_K4" \
  --out-dir "predictions_output/<today>" \
  --k 4 --slips 1 \
  --allow-unpriced \
  --max-per-league 99 --max-per-date 99
```

Outputs:
- `predictions_output/<today>/slips_legs_<tag>.csv` (includes `cs_note`)

# OG / BAWA Runbook (One-Page)

This repo produces **coverage-first football predictions** across leagues and competitions:

- **FTR (1X2)**: HOME / DRAW / AWAY probabilities + picks
- **OU2.5 / BTTS**: priced picks when odds exist
- **Team Goals (TG15/TG25)**: model-only (unpriced) candidates
- **Poisson λ diagnostics** + **Correct Score top-3** + Poisson 1X2 masses
- **CloseMatch routing** (candidate rows + CLOSE deploy preset)
- **FTR Consensus / XOG**: always outputs a pick + strength ranking for every fixture

---

## What’s new (Jan 2026)

### 1) CloseMatch stack (tight 1X2 clusters)
Tight clusters (UCL/UEL and parity domestic games) were getting floored/dropped by GLUE-style gating.

**Now:**
- `bookie_allmarkets.py` stamps close diagnostics:
  - `close_match_flag`, `candidate_rank`, `xg_diff_abs`, `implied_prob_diff`, `odds_diff`
- It emits **FTR candidate rows** (HOME/DRAW/AWAY) as:
  - `pool_tier="ANCHOR_CAND"`, with `candidate_rank` (1/2)
- `deploy_rulebook.py` has **two presets**:
  - `--preset V1` (existing GLUE-style deploy rules)
  - `--preset CLOSE` (deploys only close-match candidate rows; no `agree_model_vs_bookie` requirement)

### 2) FTR Consensus + XOG ranking (coverage layer)
`ftr_consensus.py` produces a coverage report for **all fixtures**:
- consensus pick (HOME/DRAW/AWAY)
- consensus lane: `SIDE / MID / CLOSE / AVOID`
- XOG scores (0–3) + `xog_tier` (ELITE/STRONG/MEDIUM/WEAK)

**Defaults (validated via walk-forward):**
- `tau = 1.50`
- `xog_k = 0.25`
- `close_spread_max = 0.09`
- `close_ipd_max = 0.11`
- `side_spread_min = 0.16`
- `side_ipd_min = 0.20`

**Auto outputs (unless disabled):**
- `__FTR_CONSENSUS.csv` (coverage)
- `__FTR_XOG_TIER_SHORTLIST.csv` (DEPLOY shortlist: `SIDE` + `xog_tier ∈ {ELITE, STRONG}`)
- `__FTR_BANKERS_TOP{N}.csv` (Top-N by `xog_spread`, default N=50)

CLI:
- `--bankers-n 50`
- `--no-extra-outputs`

### 3) Walk-forward backtest harness
`backtest_ftr_consensus.py` runs a true walk-forward backtest:
- train on matches strictly before each fold
- predict the fold window
- evaluate overall + buckets (lane/tier/spread)

---

## Repo layout (what matters)

- `Matches/<League>/`  
  League match CSVs (multi-season). Many leagues include `matches.csv` plus season files.

- `ModelStore/`  
  Trained artifacts per league:
  - V2 market models: `ModelStore/<LeagueTag>/{ftr_v2.pkl, over25_v2.pkl, under25_v2.pkl, btts_v2.pkl}`
  - Specialist heads: `ModelStore/<LeagueTag>/{home_ge2.pkl, away_ge2.pkl, home_ge3.pkl, away_ge3.pkl, home_fts.pkl, away_fts.pkl, btts_fh.pkl}`
  - Thresholds: `ModelStore/<LeagueTag>_market_thresholds.json`

- `predictions_output/<YYYY-MM-DD>/`  
  Daily window outputs:
  - `BOOKIE_IMP{xx}_ALLMARKETS_<from>_to_<to>.csv`
  - `...__DEPLOY_PRESET_V1.csv/.md`
  - `...__DEPLOY_PRESET_CLOSE.csv/.md`
  - `...__FTR_CONSENSUS.csv`
  - `...__FTR_XOG_TIER_SHORTLIST.csv`
  - `...__FTR_BANKERS_TOP{N}.csv`

---

## Key scripts

### Training
- `train_investor_leagues_v2.py` — trains V2 market models (FTR/OU25/BTTS) per league
- `train_markets.py` — trains specialist heads (GE2/GE3/FTS + BTTS_FH)
- `team_ratings.py` — rolling pre-match power ratings

### Pools / inference
- `bookie_allmarkets.py` — builds ALLMARKETS pool + attaches:
  - odds (novig), `bookie_spread`
  - Poisson λ (`lambda_home/away`, `exp_goals_sum`, `p00_est`)
  - correct score top-3 (`cs1..cs3` + probs)
  - Poisson 1X2 masses (`p_home_pois/p_draw_pois/p_away_pois`)
  - specialist heads (GE2/GE3/FTS, BTTS_FH)
  - rolling team rates / H2H (when available)
  - draw/chaos flags
  - close-match candidates

### Deploy / coverage
- `deploy_rulebook.py` — deterministic deploy filtering
  - `--preset V1`
  - `--preset CLOSE`
- `ftr_consensus.py` — coverage-first FTR consensus + XOG ranking

### Backtesting
- `backtest_ftr_consensus.py` — walk-forward backtest

---

## What it predicts (outputs)

### FTR
- `confidence_home/confidence_draw/confidence_away`
- `model_strength`, `ftr_margin`
- consensus: `consensus_pick`, `consensus_lane`, `consensus_confidence`, `consensus_margin`
- XOG: `xog_home/xog_draw/xog_away`, `xog_pick_score`, `xog_spread`, `xog_tier`

### Poisson + correct score diagnostics
- `lambda_home`, `lambda_away`, `exp_goals_sum`, `p00_est`
- `cs1/cs1_p`, `cs2/cs2_p`, `cs3/cs3_p`
- `p_home_pois/p_draw_pois/p_away_pois`, `cs_trunc_mass_0_6`

### CloseMatch
- `close_match_flag`, `candidate_rank`, `implied_prob_diff`, `odds_diff`, `xg_diff_abs`

---

## Recommended run order (per window)

### 1) (Optional) Train models
V2 markets:
```bash
python train_investor_leagues_v2.py \
  --leagues "England Premier League,Spain La Liga,Germany Bundesliga,France Ligue 1,Italy Serie A,Portugal Liga,Netherlands Eredivisie" \
  --matches-root Matches \
  --modelstore ModelStore \
  --markets btts,over25,under25,ftr
```

Specialist heads:
```bash
python train_markets.py \
  --league "Champions League" \
  --matches-csv "Matches/Champions League/__TRAIN_MULTISEASON_completed.csv" \
  --outdir ModelStore \
  --markets "btts_fh,home_ge2,away_ge2,home_ge3,away_ge3,home_fts,away_fts" \
  --calibration sigmoid
```

### 2) Build ALLMARKETS
```bash
python bookie_allmarkets.py \
  --date-from 2026-01-19 --date-to 2026-01-23 \
  --leagues "Champions League,Europa League,England Premier League,Spain La Liga,Germany Bundesliga" \
  --emit-ftr-candidates \
  --ftr-implied-min 0.0 \
  --debug
```

### 3) Deploy rulebook (run both)
```bash
python deploy_rulebook.py \
  --src "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv" \
  --preset V1 \
  --debug

python deploy_rulebook.py \
  --src "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv" \
  --preset CLOSE \
  --debug
```

### 4) Coverage + ranking (Consensus / XOG)
```bash
python ftr_consensus.py \
  --src "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv"
```

---

## Backtesting (walk-forward)

Monthly folds example:
```bash
python backtest_ftr_consensus.py \
  --league "England Premier League" \
  --fold monthly \
  --from 2025-08-01 \
  --to 2025-12-31 \
  --tau 1.50 \
  --xog-k 0.25 \
  --iters 800
```

---

## Accuracy snapshot (Aug–Dec 2025, monthly folds)

Latest batch scoreboard:
- Norway_Eliteserien: **0.637** (n=113)
- Spain_La_Liga: **0.632** (n=516)
- England_Premier_League: **0.599** (n=558)
- Portugal_Liga: **0.589** (n=426)
- Germany_Bundesliga: **0.583** (n=405)
- France_Ligue_1: **0.576** (n=432)
- Netherlands_Eredivisie: **0.575** (n=456)
- Japan_J1: **0.562** (n=130)
- Scotland_Premiership: **0.546** (n=348)
- Italy_Serie_A: **0.498** (n=498)
- USA_MLS: **0.037** (n=294)  ← treat as **BROKEN DATA PIPELINE** until audited

Key finding: XOG ranking is monotonic across EPL/LaLiga/Bundesliga, and top-N by `xog_spread` yields banker-grade accuracy.

---

## Quick verification checks

### 1) Confirm consensus outputs exist
```bash
ls predictions_output/<today>/*__FTR_CONSENSUS.csv
ls predictions_output/<today>/*__FTR_XOG_TIER_SHORTLIST.csv
ls predictions_output/<today>/*__FTR_BANKERS_TOP*.csv
```

### 2) Confirm CloseMatch candidates exist
```bash
python - <<'PY'
import pandas as pd
p="predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv"
df=pd.read_csv(p)
ftr=df[df['market'].astype(str).str.lower().eq('ftr')]
cand=ftr[ftr.get('pool_tier','').astype(str).str.upper().eq('ANCHOR_CAND')]
print('FTR rows:', len(ftr), 'ANCHOR_CAND rows:', len(cand))
if len(cand):
    print('close_match_flag:', cand.get('close_match_flag',0).value_counts(dropna=False).to_dict())
    print('candidate_rank:', cand.get('candidate_rank',0).value_counts(dropna=False).to_dict())
PY
```

---

## Troubleshooting

- **BANKERS writes fewer rows than requested**: normal; filename reflects actual rows written.
- **Italy Serie A underperforms**: treat as tuning target (thresholds/feature coverage/odds consistency).
- **USA MLS is unusable**: almost certainly data/format mismatch; do not deploy until fixed.

---

## Safety notes
- Unpriced TG is for tracking and notes; treat separately from priced ROI.
- CLOSE matches should be deployed via `--preset CLOSE` or via the XOG shortlist, not GLUE-only.








===============================

BELOW IS THE READ ME POST CREATION OF xOG

===============================



# OG / BAWA Runbook (One-Page)

This repo produces **league-level football picks** (FTR / OU2.5 / BTTS / Team Goals) plus **Poisson λ + Correct Score top-3** diagnostics, then filters into deterministic deploy shortlists.

**Core principle**
- **Keep V2/V3 deploy path unchanged** (it’s proven and profitable).
- Add **XOG/XAG coverage + ranking** as an *additional* layer for:
  - total fixture coverage
  - close-match handling
  - banker ranking
  - error checking / cross-validation during test windows

---

## Repo layout (what matters)

- `Matches/<League>/...csv`  
  Multi-season match datasets and league feeds.

- `ModelStore/`  
  Trained models & thresholds:
  - **V2 market models** (CatBoost): `ModelStore/<LeagueTag>/{ftr_v2.pkl, over25_v2.pkl, under25_v2.pkl, btts_v2.pkl}`
  - **Specialist heads** (train_markets): `ModelStore/<LeagueTag>/{home_ge2*.pkl, away_ge2*.pkl, home_ge3*.pkl, away_ge3*.pkl, home_fts*.pkl, away_fts*.pkl, btts_fh*.pkl}`
  - **Threshold JSON**: `ModelStore/<LeagueTag>_market_thresholds.json`
  - Optional draw bundle assets: `<LeagueTag>__draw_bundle.joblib`, `<LeagueTag>_draw_threshold.json` etc.

- `predictions_output/<YYYY-MM-DD>/`  
  Window outputs:
  - `BOOKIE_IMP{xx}_ALLMARKETS_<from>_to_<to>.csv`
  - Deploy: `...__DEPLOY_PRESET_V1.csv/.md`, `...__DEPLOY_PRESET_CLOSE.csv/.md`
  - Coverage: `...__FTR_CONSENSUS.csv`
  - XOG outputs: `...__FTR_XOG_TIER_SHORTLIST.csv`, `...__FTR_BANKERS_TOP{N}.csv`

---

## Key scripts

### V2/V3 (original “proven” path)
- `make_fd_odds_enriched_synth.py` — build synth-enriched files (coverage tooling; optional)
- `train_investor_leagues_v2.py` — batch-train V2 market models (FTR/OU25/BTTS)
- `train_markets.py` — train specialist heads (GE2/GE3/FTS + BTTS_FH)
- `bookie_allmarkets.py` — build ALLMARKETS pool + λ + CS + tails + risk flags
- `deploy_rulebook.py` — deterministic deploy shortlist (V1 preset)

### New (additive) CloseMatch + Consensus/XOG layer
- `deploy_rulebook.py --preset CLOSE` — deploy close-match candidates (no agree gate)
- `ftr_consensus.py` — coverage-first consensus + XOG ranking + bankers + shortlist
- `backtest_ftr_consensus.py` — true walk-forward backtest (train-before-test)
- `eval_ftr_consensus.py` — ad-hoc evaluator for consensus outputs

---

## What each script does (quick)

### 1) `train_investor_leagues_v2.py` (V2 market models)
Trains per league:
- FTR (multi-class)
- Over2.5 / Under2.5 / BTTS (binary)
Writes:
- `ModelStore/<LeagueTag>/{ftr_v2.pkl, over25_v2.pkl, under25_v2.pkl, btts_v2.pkl}`
- `ModelStore/<LeagueTag>_market_thresholds.json`
- Optional: per-market calibrators

### 2) `train_markets.py` (specialist heads)
Trains per league:
- `home_ge2`, `away_ge2`, `home_ge3`, `away_ge3`
- `home_fts`, `away_fts`
- `btts_fh`
Writes:
- `ModelStore/<LeagueTag>/{market}.pkl` and `{market}_v2.pkl`
- updates `ModelStore/<LeagueTag>_market_thresholds.json`

### 3) `bookie_allmarkets.py` (master pool)
Produces the **master pool CSV** with:
- FTR / OU25 / BTTS picks (priced where odds exist)
- TG15/TG25 candidates (unpriced)
- Poisson λ + CS top-3 + tails:
  - `lambda_home/lambda_away/exp_goals_sum/p00_est`
  - `cs1/cs1_p`, `cs2/cs2_p`, `cs3/cs3_p`
  - `p_home_pois/p_draw_pois/p_away_pois`, `cs_trunc_mass_0_6`
  - TG Poisson tails: `pois_home_ge2/3`, `pois_away_ge2/3`
- Risk flags:
  - `draw_risk_flag`, `chaos_risk_flag`, `not_glue_flag`, `draw_chaos_score`
- CloseMatch support:
  - `close_match_flag`, `candidate_rank`, `xg_diff_abs`, `implied_prob_diff`, `odds_diff`
  - `pool_tier="ANCHOR_CAND"` rows for top-2 FTR candidates

**Note on FTR feed timing:** if `--markets` includes `ftr`, the runner prefers the source file with the most **valid 1X2 odds rows in-window** (prevents silent FTR drops).

### 4) `deploy_rulebook.py` (deploy shortlist)
- `--preset V1`: existing GLUE-style deploy filtering (proven path)
- `--preset CLOSE`: close-match candidate deploy filtering (new, additive)

### 5) `ftr_consensus.py` (coverage + XOG/XAG)
Produces a per-fixture coverage report:
- always emits a **consensus pick** + strength
- XOG (0–3) outcome scores + `xog_tier`
- writes extra outputs (shortlist + bankers) for deployment

---

## What’s new (Jan 2026)

### A) CloseMatch stack
**Goal:** handle tight markets (parity games) without GLUE flooring.
- Candidate rows: `pool_tier="ANCHOR_CAND"` + `candidate_rank<=2`
- CLOSE deploy preset gates:
  - eligible when: `market=ftr`, `close_match_flag==1`, `pool_tier==ANCHOR_CAND`, `candidate_rank<=2`
  - default gates: `bookie_implied>=0.18`, `model_p_for_bookie>=0.27`, `gap_used>=0.02`

### B) XOG/XAG (coverage + ranking)
**Goal:** publish/score every fixture, but deploy only top bands.
- consensus uses multiple blocks:
  - model distribution + margin
  - market microstructure (novig implieds, spread, IPD)
  - strength alignment (power_diff, ppg_diff_pre, xg diff sign)
  - draw regime (exp_goals_sum, p00, poisson draw)
  - specialist heads (GE2/GE3/FTS, BTTS_FH soft)
  - rolling/H2H small adjustments
  - chaos penalties

Outputs:
- `__FTR_CONSENSUS.csv` (coverage)
- `__FTR_XOG_TIER_SHORTLIST.csv` (deploy shortlist: SIDE + ELITE/STRONG)
- `__FTR_BANKERS_TOP{N}.csv` (bankers by spread)

---

## Accuracy snapshot (Aug–Dec 2025, monthly folds)

### A) Walk-forward coverage accuracy (all fixtures)
Monthly folds, Aug–Dec 2025; `tau=1.50`, `xog_k=0.25`:

- Norway_Eliteserien: **0.637** (n=113)
- Spain_La_Liga: **0.632** (n=516)
- England_Premier_League: **0.599** (n=558)
- Portugal_Liga: **0.589** (n=426)
- Germany_Bundesliga: **0.583** (n=405)
- France_Ligue_1: **0.576** (n=432)
- Netherlands_Eredivisie: **0.575** (n=456)
- Japan_J1: **0.562** (n=130)
- Scotland_Premiership: **0.546** (n=348)
- Italy_Serie_A: **0.498** (n=498)
- USA_MLS: **0.037** (n=294)  ← treat as **BROKEN DATA PIPELINE** until audited

These are *coverage* accuracies (we score every fixture the system outputs).

### 2) Deploy-grade accuracy (thresholded bucket)
**Definition**
- `consensus_lane == SIDE`
- `xog_tier ∈ {ELITE, STRONG}`

**Observed / logged**
- England_Premier_League: **0.72** (n=200)

> Add additional leagues here as we compute them (same query can be run for each league’s preds.csv).

### 3) Bankers accuracy (Top-50 by `xog_spread`)
**Definition**
- sort all predictions by `xog_spread` desc
- take Top-50

**Observed**
- England_Premier_League: **0.94** (Top50)
- Spain La Liga: **0.94** (Top50)
- Germany Bundesliga: **0.88** (Top50)

### 4) Monotonicity checks (by threshold bins)
In the league backtest reports, accuracy rises with:
- top quintile of `xog_pick_score`
- top quintile of `xog_spread`

This is the reason XOG works as a publishable “OG ranking” and as a deploy selector.

### B) Where the ~70% accuracy comes from (deploy-grade bucket)
The ~70% comes from filtering to **deploy-grade**:
- `consensus_lane == "SIDE"`
- `xog_tier ∈ {"ELITE","STRONG"}`

Observed:
- EPL walk-forward Aug–Dec 2025: **0.72** (n=200)

### C) Where the ~90% accuracy comes from (bankers bucket)
Bankers are:
- sort by `xog_spread` descending
- take Top-N (typically Top 50)

Observed Top-50 by `xog_spread`:
- EPL: **0.94**
- Spain La Liga: **0.94**
- Germany Bundesliga: **0.88**

### D) Defaults + thresholds used

#### Consensus defaults (baked)
- `tau = 1.50`
- `xog_k = 0.25`
- lane routing:
  - `close_spread_max = 0.09`
  - `close_ipd_max = 0.11`
  - `side_spread_min = 0.16`
  - `side_ipd_min = 0.20`

#### XOG tier thresholds (used by `ftr_consensus.py`)
Lane = `SIDE`
- ELITE if `xog_pick_score>=2.30` and `xog_spread>=1.10`
- STRONG if `xog_pick_score>=2.20` and `xog_spread>=0.95`
- MEDIUM if `xog_pick_score>=2.05` and `xog_spread>=0.75`
- else WEAK

Lane = `MID`
- ELITE if `xog_pick_score>=1.70` and `xog_spread>=0.20`
- STRONG if `xog_pick_score>=1.69` and `xog_spread>=0.17`
- MEDIUM if `xog_pick_score>=1.67` and `xog_spread>=0.12`
- else WEAK

Lane = `CLOSE`
- ELITE if `xog_pick_score>=1.80` and `xog_spread>=0.26`
- STRONG if `xog_pick_score>=1.74` and `xog_spread>=0.20`
- MEDIUM if `xog_pick_score>=1.70` and `xog_spread>=0.13`
- else WEAK

#### CLOSE deploy gates (deploy_rulebook preset)
Eligible rows:
- `market==ftr`, `close_match_flag==1`, `pool_tier==ANCHOR_CAND`, `candidate_rank<=2`

Default gates:
- `bookie_implied(raw) >= 0.18`
- `model_p_for_bookie >= 0.27`
- `gap_used >= 0.02`

---

## Recommended daily run order (keep V2/V3 unchanged + add XOG)

### 0) Environment / quick sanity
```bash
python -V
python -c "import pandas, numpy; print('ok')"

1) Optional: build synth-enriched files (V2/V3 legacy)

python make_fd_odds_enriched_synth.py

2) Optional: retrain V2 market models (when data/features updated)

python train_investor_leagues_v2.py \
  --leagues "England Premier League,Germany Bundesliga,Spain La Liga,Italy Serie A,France Ligue 1,Netherlands Eredivisie,Portugal Liga" \
  --matches-root Matches \
  --modelstore ModelStore \
  --markets btts,over25,under25,ftr

3) Optional: retrain specialist heads (GE/FTS/BTTS_FH)

python train_markets.py \
  --league "England Premier League" \
  --matches-csv "Matches/England Premier League/__TRAIN_MULTISEASON_completed.csv" \
  --outdir ModelStore \
  --markets "btts_fh,home_ge2,away_ge2,home_ge3,away_ge3,home_fts,away_fts" \
  --calibration sigmoid

4) Generate ALLMARKETS pool (master output)

python bookie_allmarkets.py \
  --date-from 2026-01-19 --date-to 2026-01-23 \
  --leagues "England Premier League,Italy Serie A,Spain La Liga,Germany Bundesliga,France Ligue 1,Netherlands Eredivisie,Portugal Liga,Belgium Pro,Scotland Premiership,Champions League,Europa League" \
  --markets "ftr,ou25,btts,tg15,tg25" \
  --emit-ftr-candidates \
  --ftr-implied-min 0.0 \
  --tg-use-dir-gate \
  --debug

5) Deploy (V2/V3 proven path)

python deploy_rulebook.py \
  --src "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv" \
  --preset V1 \
  --debug

6) Deploy close matches (additive)

python deploy_rulebook.py \
  --src "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv" \
  --preset CLOSE \
  --debug

7) Run XOG/XAG coverage layer (additive, error checking + bankers)

python ftr_consensus.py \
  --src "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv"

Optional:
	•	--bankers-n 50
	•	--no-extra-outputs

8) Optional: legs view with CS notes (keep legacy)

python acca_builder.py build \
  --pool "predictions_output/<today>/BOOKIE_IMP68_ALLMARKETS_<from>_to_<to>.csv" \
  --tag "CS_VIEW_K4" \
  --out-dir "predictions_output/<today>" \
  --k 4 --slips 1 \
  --allow-unpriced \
  --max-per-league 99 --max-per-date 99


⸻

Quick verification checks

1) Confirm λ + CS exist

python - <<'PY'
import pandas as pd, glob
p=sorted(glob.glob("predictions_output/*/BOOKIE_IMP*_ALLMARKETS_*.csv"))[-1]
df=pd.read_csv(p)
cols=["lambda_home","lambda_away","exp_goals_sum","p00_est","cs1","cs1_p","cs2","cs2_p","cs3","cs3_p","p_home_pois","p_draw_pois","p_away_pois"]
print("FILE:", p)
print({c:(c in df.columns) for c in cols})
PY

2) Confirm CloseMatch candidates exist

python - <<'PY'
import pandas as pd, glob
p=sorted(glob.glob("predictions_output/*/BOOKIE_IMP*_ALLMARKETS_*.csv"))[-1]
df=pd.read_csv(p)
ftr=df[df["market"].astype(str).str.lower().eq("ftr")]
cand=ftr[ftr.get("pool_tier","").astype(str).str.upper().eq("ANCHOR_CAND")]
print("FTR rows:", len(ftr), "ANCHOR_CAND rows:", len(cand))
if len(cand):
    print("close_match_flag:", cand.get("close_match_flag",0).value_counts(dropna=False).to_dict())
    print("candidate_rank:", cand.get("candidate_rank",0).value_counts(dropna=False).to_dict())
PY

3) Confirm consensus outputs exist

ls predictions_output/<today>/*__FTR_CONSENSUS.csv
ls predictions_output/<today>/*__FTR_XOG_TIER_SHORTLIST.csv
ls predictions_output/<today>/*__FTR_BANKERS_TOP*.csv


⸻

League onboarding (checklist)
	1.	Add CSVs under Matches/<League>/
	2.	Confirm date parsing works and goals exist for realised rows
	3.	Train V2 models: train_investor_leagues_v2.py
	4.	Train specialist heads: train_markets.py
	5.	Build power ratings (rolling) availability check (join rate)
	6.	Run bookie_allmarkets.py on a small window
	7.	Run deploy presets: deploy_rulebook.py --preset V1 and --preset CLOSE
	8.	Run ftr_consensus.py and inspect shortlist/bankers

⸻

Troubleshooting (fast)

“No rows produced”
	•	odds columns missing/invalid in-window OR wrong league file selection
	•	confirm rows_in_window and window_valid_1x2_rows logs

“BANKERS requested 50 but file is TOP25”

Normal: file name reflects actual rows written (min(requested, available rows)).

USA MLS ≈ 0.037 accuracy

Treat as broken data pipeline until audited:
	•	date parsing, team naming, season windows, odds columns, or fixture_key drift

⸻

Safety notes
	•	Synth odds are for coverage/diagnostics; deploy only behind hard gates.
	•	TG markets are unpriced; keep separate from ROI logic.
	•	CLOSE fixtures are higher variance; deploy via CLOSE preset and/or banker-only.

If you want the **per-league deploy-grade accuracy** section fully populated (like the coverage section), we just need to run the same 10-line summary snippet against each league’s `preds.csv` from the backtest run dirs and paste the results into that block.

















======

# OG / BAWA Runbook (One-Page)

This repo produces **league-level football picks** (FTR / OU2.5 / BTTS / Team Goals) plus **Poisson λ + Correct Score top-3** diagnostics, then filters into a deterministic deploy shortlist.

---

## Repo layout (what matters)
- `Matches/<League>/fd_odds_enriched.csv`
  Main merged match + odds + engineered features per league (history + forward fixtures as your feed allows).
- `Matches/<League>/fd_odds_enriched_synth.csv`
  Same, but with missing odds sides (e.g. **Under2.5**) filled where possible by conservative synthesis.
- `ModelStore/<LeagueTag>/*.pkl`
  Trained models (FTR / side models + specialist heads like GE2/GE3/FTS).
- `predictions_output/<YYYY-MM-DD>/`
  Daily outputs (ALLMARKETS pool, deploy shortlist, optional slips/legs CSVs).

**Key scripts**
- `train_markets.py` — trains specialist heads (GE2/GE3/FTS)
- `make_fd_odds_enriched_synth.py` — builds synth-enriched league files
- `bookie_allmarkets.py` — builds ALLMARKETS pool + λ + CS + TG coherence + draw/chaos flags
- `deploy_rulebook.py` — deterministic deploy shortlist (CSV + MD)
- `acca_builder.py` — optional: build slips/legs + emits `cs_note` in legs CSV

---

## What each script does (quick)

### 1) `train_markets.py`
Trains **specialist heads** per league:
- `home_ge2`, `away_ge2` (team ≥2 goals)
- `home_ge3`, `away_ge3` (team ≥3 goals)
- `home_fts`, `away_fts` (fail-to-score)

Writes:
- `ModelStore/<LeagueTag>/<market>.pkl` and `<market>_v2.pkl`
- `ModelStore/<LeagueTag>_market_thresholds.json`

### 2) `make_fd_odds_enriched_synth.py`
Creates `fd_odds_enriched_synth.csv` per league by filling missing odds sides where rules are confident (e.g. missing **Under2.5** but Over2.5 exists).

**Safety:** synthesis is for **coverage** and **diagnostics**. Gate hard and treat as optional.

### 3) `bookie_allmarkets.py`
Builds the master pick pool.

Emits:
- FTR / OU2.5 / BTTS picks when odds exist
- TG15/TG25 picks (model-only; unpriced)

Attaches **λ + correct-score + tails**:
- `home_goals_pred`, `away_goals_pred`, `lambda_home`, `lambda_away`, `exp_goals_sum`, `p00_est`
- `cs1/cs1_p`, `cs2/cs2_p`, `cs3/cs3_p`
- `p_home_pois/p_draw_pois/p_away_pois`, `cs_trunc_mass_0_6`
- TG Poisson tails: `pois_home_ge2/3`, `pois_away_ge2/3`

Attaches **TG coherence**:
- `tg_pois_ok`, `tg_pois_gap` (vetoes “GE says huge but Poisson says tiny”)

Attaches **Draw/Chaos Risk** (used to tag “NOT GLUE” even when pick is HOME):
- `draw_risk_flag`, `chaos_risk_flag`, `not_glue_flag`, `draw_chaos_score`

Writes:
- `predictions_output/<today>/BOOKIE_IMP{xx}_ALLMARKETS_<from>_to_<to>.csv`

**Note on FTR feed timing:** when `--markets` includes `ftr`, the runner prefers the source file with the most **valid 1X2 odds rows in-window** (so it won’t silently drop FTR just because one enriched file is missing 1X2 odds).

### 4) `deploy_rulebook.py`
Deterministic filtering (“deploy shortlist”):
- applies market-specific rules using bookie implieds + model probs + stability/structure features

Writes:
- `<input>__DEPLOY_PRESET_V1.csv`
- `<input>__DEPLOY_PRESET_V1.md`

### 5) `acca_builder.py` (optional)
- `build`: creates slips + `slips_legs_<tag>.csv` (**includes `cs_note`**)
- `backtest`: joins to realised Matches and outputs ROI/hit-rate summaries

You can use `build` even if you don’t want “bettable slips” — it’s a convenient **legs view** with CS notes.

---

## Recommended run order (daily)

### A) Refresh league data (your ETL step)
Update files under `Matches/<League>/...`.

### B) Optional: build synth-enriched files
Only if you want `fd_odds_enriched_synth.csv` present for window selection / missing under odds coverage.

```bash
python make_fd_odds_enriched_synth.py
```

### C) Optional: retrain specialist heads
Do this if:
- you added new completed matches
- you changed feature engineering / leak drops / schemas
- you fixed a bug affecting GE/FTS

```bash
python train_markets.py \
  --leagues "England Premier League,Germany Bundesliga,Spain La Liga,Italy Serie A,France Ligue 1,Netherlands Eredivisie,Portugal Liga,Belgium Pro,Scotland Premiership" \
  --matches-csv "Matches/England Premier League/fd_odds_enriched_synth.csv" \
  --markets "home_ge2,away_ge2,home_ge3,away_ge3,home_fts,away_fts" \
  --calibration none
```

> Note: `--matches-csv` is used as a template path in the runner; it still trains per league from the correct inputs inside the league loop.

### D) Generate ALLMARKETS pool (master output)
```bash
python bookie_allmarkets.py \
  --date-from 2026-01-07 --date-to 2026-01-08 \
  --leagues "England Premier League,Italy Serie A,Spain La Liga,Germany Bundesliga,France Ligue 1,Netherlands Eredivisie,Portugal Liga,Belgium Pro,Scotland Premiership" \
  --markets "ftr,ou25,btts,tg15,tg25" \
  --tg-use-dir-gate \
  --debug
```

### E) Generate deploy shortlist
```bash
python deploy_rulebook.py \
  --src "predictions_output/2026-01-07/BOOKIE_IMP62_ALLMARKETS_2026-01-07_to_2026-01-08.csv" \
  --outdir "predictions_output/2026-01-07" \
  --tag "DEPLOY_PRESET_V1"
```

### F) Optional: create a legs CSV with `cs_note` (no betting automation)
```bash
python acca_builder.py build \
  --pool "predictions_output/2026-01-07/BOOKIE_IMP68_ALLMARKETS_2026-01-07_to_2026-01-08.csv" \
  --tag "CS_VIEW_K4" \
  --out-dir "predictions_output/2026-01-07" \
  --k 4 --slips 1 \
  --allow-unpriced \
  --max-per-league 99 --max-per-date 99
```

Output:
- `predictions_output/<date>/slips_legs_<tag>.csv` (has `cs_note`)

---

## Quick verification checks

### 1) Confirm CS + λ columns exist in ALLMARKETS
```bash
python - <<'PY'
import pandas as pd, glob
p=sorted(glob.glob("predictions_output/*/BOOKIE_IMP*_ALLMARKETS_*.csv"))[-1]
df=pd.read_csv(p)
cols=["home_goals_pred","away_goals_pred","exp_goals_sum","cs1","cs1_p","cs2","cs2_p","cs3","cs3_p",
      "p_home_pois","p_draw_pois","p_away_pois","cs_trunc_mass_0_6"]
print("FILE:", p)
print({c:(c in df.columns) for c in cols})
PY
```

### 2) Confirm TG Poisson coherence is present (and not vetoing everything)
```bash
python - <<'PY'
import pandas as pd, glob
p=sorted(glob.glob("predictions_output/*/BOOKIE_IMP*_ALLMARKETS_*.csv"))[-1]
df=pd.read_csv(p)
tg=df[df["market"].astype(str).str.lower().isin(["tg15","tg25"])].copy()
print("FILE:", p)
print("tg rows:", len(tg))
for c in ["pois_home_ge2","pois_away_ge2","pois_home_ge3","pois_away_ge3","tg_pois_ok","tg_pois_gap"]:
    print(c, "OK" if c in df.columns else "MISSING")
print("vetoed:", int((pd.to_numeric(tg.get("tg_pois_ok",1), errors="coerce").fillna(1)==0).sum()))
PY
```

### 3) Confirm Draw/Chaos flags exist (FTR rows)
```bash
python - <<'PY'
import pandas as pd, glob
p=sorted(glob.glob("predictions_output/*/BOOKIE_IMP*_ALLMARKETS_*.csv"))[-1]
df=pd.read_csv(p)
want=["draw_risk_flag","chaos_risk_flag","not_glue_flag","draw_chaos_score"]
print("FILE:", p)
print({c:(c in df.columns) for c in want})
ftr=df[df["market"].astype(str).str.lower().eq("ftr")].copy()
if not ftr.empty:
    cols=[c for c in ["fixture_key","bookie_pick","confidence_draw","ftr_margin","draw_chaos_score","draw_risk_flag","chaos_risk_flag","not_glue_flag"] if c in ftr.columns]
    print(ftr[cols].head(25).to_string(index=False))
else:
    print("(no FTR rows in this window)")
PY
```

---

## Troubleshooting (fast)

### “rows_in_window = 0” for a league
- your feed likely hasn’t populated fixtures that far ahead

```bash
python - <<'PY'
import pandas as pd
p="Matches/England Premier League/fd_odds_enriched.csv"
df=pd.read_csv(p, low_memory=False)
md=pd.to_datetime(df.get("match_date", df.get("date_GMT", None)), errors="coerce", utc=True)
lo=pd.Timestamp("2026-01-07", tz="UTC"); hi=pd.Timestamp("2026-01-08", tz="UTC")
print(int(((md>=lo)&(md<=hi)).sum()))
PY
```

### “OU25/BTTS missing” in ALLMARKETS
- odds columns may be missing/invalid (0/NaN) in that window
- `bookie_allmarkets.py` will fall back to OVER-only for OU25 if under odds absent (depending on safeguards)

### “Model load errors” (wrapper/pickle)
- you already moved to `--calibration none` in `train_markets.py` to avoid custom wrapper unpickling

### “TG outputs nonsense”
- run with `--tg-use-dir-gate`
- ensure Poisson coherence is present (`pois_*` cols)
- check `tg_pois_gap` (big gap = mismatch)

---

## Safety notes
- **Synth odds:** treat as coverage tooling, not “must bet”. Gate hard (conf thresholds + sanity checks).
- **Unpriced legs (TG):** fine for accuracy tracking and builder notes; ROI/staking must exclude or handle separately.
- **Draw/Chaos flags:** use them as *pool hygiene* (“NOT GLUE”), not as a claim that a draw is “predicted”.



=======================


=======================



## UEFA Context Overlay (UCL / UEL / UECL)

This project includes a deterministic UEFA context overlay that converts the “table pressure + rotation risk” narrative into structured features, attached directly onto ALLMARKETS rows so downstream modules (rulebook, gates, overlays) can consume them.

### What it does

**1) Build a per-team UEFA snapshot (from Teams CSV)**
Derives:
- `state_bucket` ∈ {`TOP8`, `TOP24`, `OUTSIDE`}
- `gap_to24`, `gap_to8`
- flags: `must_win_flag`, `must_avoid_loss_flag`, `rotation_risk_flag`, `eliminated_flag`
- volatility: `volatility_ratio`, derived `vol_band_n`

**2) Attach snapshot to match rows in ALLMARKETS**
Adds match-level columns:
- `uefa_home_state`, `uefa_away_state`
- `uefa_home_gap24`, `uefa_away_gap24`, `uefa_gap24_diff`
- `uefa_home_rotation_risk`, `uefa_away_rotation_risk`
- `uefa_home_must_win`, `uefa_away_must_win`
- `uefa_home_must_avoid_loss`, `uefa_away_must_avoid_loss`
- `uefa_home_eliminated`, `uefa_away_eliminated`
- `uefa_both_must_win`, `uefa_goal_hunt_flag`, `uefa_pride_only_flag`
- `uefa_live_table_volatility`, `uefa_vol_band_n`

**3) Derived match-level aggregates**
Computed after the UEFA merge:
- `uefa_rotation_any = max(uefa_home_rotation_risk, uefa_away_rotation_risk)`
- `uefa_rotation_both = uefa_home_rotation_risk & uefa_away_rotation_risk`
- `uefa_pressure_sum = uefa_home_must_win + uefa_away_must_win + uefa_home_must_avoid_loss + uefa_away_must_avoid_loss`
- `uefa_pressure_asym = abs(uefa_home_gap24 - uefa_away_gap24)`

### How deploy_rulebook uses it (tier nudges, not vetoes)

`deploy_rulebook.py` consumes UEFA fields as **tier adjustments**:
- Rotation risk + short odds ⇒ downgrade tier to `OBSERVE`
- High pressure (`uefa_pressure_sum` high) ⇒ allow strong OU/BTTS picks to promote one tier (only if deterministic gates pass + signal alignment is clean)

A guarded “UEFA not_glue fail-open” exists to avoid wiping legitimate strong favourites, but includes false-positive guards (e.g. require non-negative edge and non-leaky home xGA) so draw-trap favourites are not rescued.

---

## Commands

### Run ALLMARKETS with debug
```bash
cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"
source .venv/bin/activate

OUTDIR="predictions_output/2026-01-28/MIDWEEK_UCL_UEL_DBG"

python bookie_allmarkets.py \
  --date-from 2026-01-28 \
  --date-to 2026-01-30 \
  --outdir "$OUTDIR" \
  --leagues "Champions League,Europa League" \
  --markets "ftr,btts,ou25" \
  --debug