TRAINING_PIPELINE__MARKETS_AND_MODELS.md
# TRAINING_PIPELINE__MARKETS_AND_MODELS.md

A single, canonical runbook for **training + validating** OddsGenius market models and goal (λ) regressors.

This mirrors the style of `ODDS_SYNTH__UNDER25_AND_MERGE_PIPELINE.md` so you can add leagues without digging through old notes.

---

## Source-of-truth scripts

### Merge + rolling features
- `build_merged.py`  
  Builds `Matches/__merged__/<LEAGUE_TAG>__merged.csv` from league folder CSVs.
  
### Market model training
You currently have two training paths:

1) **`train_markets.py`** (preferred “production” path if that’s what runtime expects)
- Use this if your runtime / overlays / deploy gates are already wired to its artifact naming, calibrators, and thresholds.

2) **`train_investor_leagues_v2.py`** (batch trainer; recently updated)
- Produces market bundles + thresholds and (when not in `--lite`) calibrators.
- 4-days-old skeleton is correct; update it to **prefer synth odds inputs** (`fd_odds_enriched_synth.csv`) and keep naming consistent with v3.

### Goal (λ) regressors
- Implemented in `_baseline_ftr_pipeline.quick_fit_goal_regressors(df, league_name, ...)`
- Writes the **5-fold ensembles**:
  - `ModelStore/<LEAGUE_TAG>/goal_ensembles/home_goals_fold5.pkl`
  - `ModelStore/<LEAGUE_TAG>/goal_ensembles/away_goals_fold5.pkl`
  - plus `lambda_models_manifest.json`

---

## Target leagues

### Normalized league names (spaces)
Use these with CLI flags that accept comma-separated league strings:
- England Premier League
- England Championship
- England EFL League 1
- England FA Cup
- Japan J1
- Norway Eliteserien
- Netherlands Eredivisie
- Belgium Pro
- Scotland Premiership
- Brazil Serie A
- USA MLS
- Portugal Liga
- Spain La Liga
- Italy Serie A
- France Ligue 1
- Germany Bundesliga
- Europa Conference
- Europa League
- Champions League

### League tags (underscores)
File paths and ModelStore folders use tags:
- England_Premier_League
- England_Championship
- England_EFL_League_1
- England_FA_Cup
- Japan_J1
- Norway_Eliteserien
- Netherlands_Eredivisie
- Belgium_Pro
- Scotland_Premiership
- Brazil_Serie_A
- USA_MLS
- Portugal_Liga
- Spain_La_Liga
- Italy_Serie_A
- France_Ligue_1
- Germany_Bundesliga
- Europa_Conference
- Europa_League
- Champions_League

---

## What “healthy” means (pass criteria)

### A) Inputs are correct (merge stage)
Per league merged CSV in `Matches/__merged__/` must have:
- timeline parseable: `match_date` or `date_GMT`
- teams: `home_team_name`, `away_team_name`
- realised goals: `home_team_goal_count`, `away_team_goal_count`

### B) Odds coverage (core markets)
We expect (most leagues):
- OU25: `odds_ft_over25` and `odds_ft_under25`
- BTTS: `odds_btts_yes` and `odds_btts_no`
- FTR: `odds_ft_home_team_win`, `odds_ft_draw`, `odds_ft_away_team_win`

If your league has missing canonical under 2.5, it must be filled via synth (see synth doc):
- `odds_ft_under25_synth` + `odds_ft_under25` populated
- `odds_source_under25` should become `synth_ou25` for synth-filled rows

### C) Rolling press features are present (recommended)
Merged should contain these 6 columns when `--rolling-press` is used:
- `rolling5_home_press_intensity`
- `rolling5_away_press_intensity`
- `rolling5_press_intensity_diff`
- `rolling5_home_press_z`
- `rolling5_away_press_z`
- `rolling5_press_z_diff`

### D) Goal (λ) artifacts exist
After training goal regressors:
- `ModelStore/<TAG>/goal_ensembles/home_goals_fold5.pkl`
- `ModelStore/<TAG>/goal_ensembles/away_goals_fold5.pkl`
- `ModelStore/<TAG>/goal_ensembles/lambda_models_manifest.json`

### E) Market artifacts exist (core)
After market training:
- `ModelStore/<TAG>/ftr_v3.pkl` (or your chosen canonical name)
- `ModelStore/<TAG>/over25_v3.pkl`
- `ModelStore/<TAG>/under25_v3.pkl`
- `ModelStore/<TAG>/btts_v3.pkl`

…and compatibility copies if you keep runtime backwards compatible:
- `*_v2.pkl` copies and `ou25_*` aliases (if required by `bookie_allmarkets.py`)

### F) Export QC is green
`bookie_allmarkets.py --strict` must pass:
- OU25 no-vig is present, sums to 1
- BTTS + FTR non-null critical fields present
- strict asserts pass without re-reading file

---

## Phase 0 — Pre-merge: synth odds where needed

If league is missing canonical under 2.5:
- Generate / update:
  - `Matches/<League>/fd_odds_enriched_synth.csv`
- Ensure:
  - `odds_ft_under25` is populated (synth where required)
  - `odds_source_under25` shows `synth_ou25` for synth rows

Example from England Championship:
- wrote `Matches/England Championship/fd_odds_enriched_synth.csv`
- under25_synth nonnull: `4281/4452`

---

## Phase A — Build merged CSVs (single source of truth)

### Build merged + rolling press + report (recommended)

```bash
python build_merged.py --leagues "England Premier League,England Championship,England EFL League 1,England FA Cup,Japan J1,Norway Eliteserien,Netherlands Eredivisie,Belgium Pro,Scotland Premiership,Brazil Serie A,USA MLS,Portugal Liga,Spain La Liga,Italy Serie A,France Ligue 1,Germany Bundesliga,Europa Conference,Europa League,Champions League" \
  --recursive \
  --rolling-press \
  --write-report
```

Outputs:
- `Matches/__merged__/<TAG>__merged.csv`
- `Matches/__merged__/__merge_report__.csv`

Interpretation of report fields:
- `inputs` should include `fd_odds_enriched_synth.csv` when available
- `under25_synth_rate` should be >0 when league needed synth
- `rolling_press_cols_present` should be `6` if `--rolling-press`

---

## Phase B — Train goal regressors (λ home/away)

This uses `_baseline_ftr_pipeline.quick_fit_goal_regressors` and persists fold5 ensembles.

### Batch retrain (fixed leagues)

```bash
FORCE_RETRAIN_LAMBDA=1 VERBOSE_QUICK=1 python - <<'PY'
import os
import pandas as pd
import _baseline_ftr_pipeline as m

ROOT = os.path.dirname(m.__file__)
MERGED = os.path.join(ROOT, "Matches", "__merged__")

LEAGUES = [
  "England Premier League",
  "England Championship",
  "England EFL League 1",
  "England FA Cup",
  "Japan J1",
  "Norway Eliteserien",
  "Netherlands Eredivisie",
  "Belgium Pro",
  "Scotland Premiership",
  "Brazil Serie A",
  "USA MLS",
  "Portugal Liga",
  "Spain La Liga",
  "Italy Serie A",
  "France Ligue 1",
  "Germany Bundesliga",
  "Europa Conference",
  "Europa League",
  "Champions League",
]

ok, fail = 0, 0
for league in LEAGUES:
    merged_path = os.path.join(MERGED, f"{league.replace(' ','_')}__merged.csv")
    if not os.path.exists(merged_path):
        print("❌ missing merged:", merged_path)
        fail += 1
        continue

    df = pd.read_csv(merged_path, low_memory=False)
    try: df = m._canonize(df)
    except Exception: pass
    try: df = m._sanitize_pd_na_boolean(df)
    except Exception: pass

    print(f"\n=== TRAIN GOALS: {league} | rows={len(df)} cols={len(df.columns)} ===")
    try:
        m.quick_fit_goal_regressors(df, league)
        ok += 1
    except Exception as e:
        print("❌ train failed:", league, "|", e)
        fail += 1

print(f"\nDONE | ok={ok} fail={fail}")
PY
```

### If “train_goals=True” produces FTR instead (diagnosis)
If calling something like `train_ftr_for_league(... train_goals=True)` still trains FTR, likely:
1) `train_ftr_for_league` ignores `train_goals` in that branch.
2) goal block is swallowed by `try/except` and silently continues.
3) fold5 writer is in a different helper (the one you pasted shows it’s in `quick_fit_goal_regressors`).
4) you’re checking the wrong output path.

The fold5 writer we care about is the block that does:
- `ensemble_dir = ModelStore/<TAG>/goal_ensembles`
- `joblib.dump(... home_goals_fold5.pkl)`
- `joblib.dump(... away_goals_fold5.pkl)`
- writes `lambda_models_manifest.json`

---

## Phase C — Train core market models (FTR / OU25 / BTTS)

### Decide the single source-of-truth trainer

**Recommendation:** pick ONE trainer as canonical for core markets to prevent drift.

- If runtime expects the “investor v2/v3” bundle names + calibrators → use `train_investor_leagues_v2.py`.
- If runtime expects `train_markets.py` artifacts (and extra manifests/calibrators) → use `train_markets.py`.

### Required update: prefer synth inputs
Regardless of trainer, training must read merged CSVs that:
- prefer `fd_odds_enriched_synth.csv` when present
- have `odds_ft_under25` populated (orig or synth)

### Build missing market pack for the 3 leagues (thresholds + calibrators)

```bash
python train_investor_leagues_v2.py \
  --leagues "England Championship,England EFL League 1,England FA Cup" \
  --markets "btts,over25,under25,ftr" \
  --market-version v3 \
  --ftr-version v2 \
  --write-v2-compat \
  --val-frac 0.2 \
  --iters 1800 \
  --threads 4 \
  --overwrite
```

Verify calibrators exist:
```bash
find ModelStore -maxdepth 4 -type f | rg -n "England_(Championship|EFL_League_1|FA_Cup).*calibrators"
```

---

## Phase D — Signal bands (required if runtime expects them)

Builder: `fit_signal_bands.py`

```bash
python fit_signal_bands.py --league "England Championship" --outdir ModelStore
python fit_signal_bands.py --league "England EFL League 1" --outdir ModelStore
python fit_signal_bands.py --league "England FA Cup" --outdir ModelStore
```

Verify:
```bash
find ModelStore -maxdepth 3 -type f | rg -n "England_(Championship|EFL_League_1|FA_Cup).*(signal_bands\.json)$"
```

---

## Phase E — Export + strict QC (single command “QC green”)

Run your export with `--strict` so it fails fast if any market is broken.

Example:

```bash
OG_DEBUG_POWER=1 OG_DEBUG_PICK=1 python3 bookie_allmarkets.py \
  --date-from 2026-02-14 \
  --date-to 2026-02-17 \
  --leagues "England Premier League,England Championship,England EFL League 1,England FA Cup,Japan J1,Norway Eliteserien,Netherlands Eredivisie,Belgium Pro,Scotland Premiership,Brazil Serie A,USA MLS,Portugal Liga,Spain La Liga,Italy Serie A,France Ligue 1,Germany Bundesliga,Europa Conference,Europa League,Champions League" \
  --markets ftr,ou25,btts,tg15,tg25 \
  --implied-min 0.62 \
  --ou25-implied-min 0.50 \
  --btts-implied-min 0.50 \
  --tg15-pmin 0.52 \
  --tg25-pmin 0.35 \
  --strict
```

Pass criteria:
- strict asserts pass
- output written
- “Coverage by market” shows novig/overround where expected

---

## Phase F — Secondary markets (only after core is stable)

Targets (examples):
- fail to score (home/away)
- win to nil
- team goals (home_ge2 / away_ge2, etc.)
- BTTS 1st half

Rule:
- do not add secondary markets until core retrain is fully clean across all 19 leagues.

---

## New League Kickoff Checklist (Training) — copy/paste

```bash
# 1) Put league season CSVs into Matches/<League Name>/
# 2) If under 2.5 odds are missing, generate synth file first (see synth doc)

python build_merged.py --leagues "<LEAGUE NAME>" --recursive --rolling-press --write-report

FORCE_RETRAIN_LAMBDA=1 VERBOSE_QUICK=1 python - <<'PY'
import os, pandas as pd
import _baseline_ftr_pipeline as m
league = "<LEAGUE NAME>"
root=os.path.dirname(m.__file__)
mp=os.path.join(root,"Matches","__merged__",f"{league.replace(' ','_')}__merged.csv")
df=pd.read_csv(mp, low_memory=False)
try: df=m._canonize(df)
except: pass
m.quick_fit_goal_regressors(df, league)
print("✅ goal fold5 written")
PY

# Train core markets (choose ONE canonical trainer)
python train_investor_leagues_v2.py --leagues "<LEAGUE NAME>" --markets "btts,over25,under25,ftr" --market-version v3 --write-v2-compat --overwrite

# Optional if runtime expects it
python fit_signal_bands.py --league "<LEAGUE NAME>" --outdir ModelStore

# Smoke test export with strict QC
python3 bookie_allmarkets.py --date-from 2026-02-14 --date-to 2026-02-17 --leagues "<LEAGUE NAME>" --markets ftr,ou25,btts --strict
```

---

## Notes / gotchas

- USA MLS / Japan / Brazil / UEFA comps may be missing “up to date” data; that’s fine for training as long as merged is coherent.
- If `date_GMT` parsing warnings appear (“Could not infer format”), it’s usually safe, but if you want to silence it:
  - standardize date formats upstream OR add explicit parsing format in the relevant helper.
- Keep `--strict` in export runs. It’s your “one command = QC green” gate.


# TRAIN_MARKETS — Models, Scripts, Running Order (Source of Truth)

This document defines the canonical training workflow for OG/BAWA market models
(FTR, Over/Under, BTTS, and secondary markets), and how it fits with the odds
synthesis + merge pipeline.

If you follow this, you never need to dig through scattered notes again.

---

## What this pipeline produces (per league)

Inside:

ModelStore/<LEAGUE_TAG>/

Canonical (versioned):
- ftr_<version>.pkl
- over25_<version>.pkl
- under25_<version>.pkl
- btts_<version>.pkl
- (optional secondary markets) home_ge2_<version>.pkl, away_ge2_<version>.pkl, etc.

Optional compatibility outputs:
- <market>.pkl               (legacy alias for runtime that expects unversioned name)
- <market>_v2.pkl            (compat copy when requested)

Metrics:
- <LEAGUE_TAG>_market_metrics.csv   (saved in the league folder)
Threshold map:
- ModelStore/<LEAGUE_TAG>_market_thresholds.json

---

## Dependencies / prerequisites

### A) Clean merged matches CSV for the league
Your trainer expects the league CSV already contains prematch-safe features and odds.

Minimum required labels:
- home_team_goal_count
- away_team_goal_count
- (optional) half-time goal counts if you want BTTS_FH:
  - home_team_goal_count_half_time
  - away_team_goal_count_half_time

### B) Odds synth / merge already run (important)
Under 2.5 is often missing. The odds synth pipeline should create a synth under25 column.
Training must NOT generate odds — it only maps synth outputs into canonical names.

Expected synth outputs (any one of these is enough):
- odds_ft_under25_synth
- odds_under25_synth
- synth_odds_ft_under25
- odds_ft_u25_synth

Trainer will fill canonical:
- odds_ft_under25

---

## Running order (global)

### Step 1 — Build / refresh merged per-league CSV
Run your merge pipeline(s) so each league has a single merged dataset.

(You already documented the synth odds merge in:
ODDS_SYNTH__UNDER25_AND_MERGE_PIPELINE.md)

✅ Pass criteria:
- league merged CSV exists
- columns include your pre-match features
- under25 synth column exists if real under25 odds often missing

### Step 2 — Train core market models (per league)
Core markets (stable first):
- ftr
- over25
- under25
- btts

Do NOT add secondary markets until core is clean across all target leagues.

Recommended run:
1 league smoke test → validate artifacts → scale batch.

### Step 3 — Train secondary markets (after core is stable)
Examples:
- btts_fh
- home_ge2 / away_ge2
- home_ge3 / away_ge3
- home_fts / away_fts

Rule:
Only do this after Step 2 is proven clean across the whole league list.

---

## CLI usage (train_markets.py)

### Single league smoke test (core only)
python train_markets.py \
  --league "England Premier League" \
  --matches-csv "Matches/__merged__/England_Premier_League__merged.csv" \
  --outdir "ModelStore" \
  --markets "ftr,over25,under25,btts" \
  --market-version v3 \
  --write-v2-compat

### Full run (all markets configured in DEFAULT_CFG)
python train_markets.py \
  --league "England Premier League" \
  --matches-csv "Matches/__merged__/England_Premier_League__merged.csv" \
  --outdir "ModelStore" \
  --market-version v3 \
  --write-v2-compat

---

## Validation checks (must pass per league)

### A) Artifacts exist
ModelStore/<TAG>/ftr_v3.pkl
ModelStore/<TAG>/over25_v3.pkl
ModelStore/<TAG>/under25_v3.pkl
ModelStore/<TAG>/btts_v3.pkl

If legacy alias is enabled:
ModelStore/<TAG>/ftr.pkl etc.

If v2 compat enabled:
ModelStore/<TAG>/ftr_v2.pkl etc.

### B) Threshold file updated
ModelStore/<TAG>_market_thresholds.json exists and contains thresholds for any trained markets.

### C) Feature sanity
Check your press intensity columns appear in the saved bundle features when present:
- pre_match_press_intensity_home
- pre_match_press_intensity_away

### D) Under25 odds mapping sanity
If the merged CSV contains *only* synth under25 odds (not the canonical), trainer must still include:
- odds_ft_under25 (filled from synth column)

---

## Target league list (canonical)

- England Premier League
- England Championship
- England EFL League 1
- England FA Cup
- Japan J1
- Norway Eliteserien
- Netherlands Eredivisie
- Belgium Pro
- Scotland Premiership
- Brazil Serie A
- USA MLS
- Portugal Liga
- Spain La Liga
- Italy Serie A
- France Ligue 1
- Germany Bundesliga
- Europa Conference
- Europa League
- Champions League

---

## Notes / rules to avoid breaking production

1) Training does not synth odds. Only consumes merge outputs.
2) Versioned model filenames are canonical. Legacy alias exists only for runtime compatibility.
3) Do core first, then secondary markets.
4) If a league is missing half-time goals → btts_fh will be skipped (expected).







































# TRAINING_PIPELINE__MARKETS_AND_MODELS.md

A single, canonical runbook for **training + validating** OddsGenius (OG/BAWA) market models and goal (λ) regressors.

This mirrors the style of `ODDS_SYNTH__UNDER25_AND_MERGE_PIPELINE.md` so you can add leagues without digging through old notes.

---

## Source-of-truth scripts

### Odds synth + merge (inputs)
- `odds_synth.py`
  - Creates per-league `fd_odds_enriched_synth.csv` when canonical Under 2.5 (or BTTS No) is missing.
  - Promotes synth into canonical columns (e.g. fills `odds_ft_under25` when missing) and writes provenance.
- `build_merged.py`
  - Builds `Matches/__merged__/<LEAGUE_TAG>__merged.csv`.
  - **Prefers** `fd_odds_enriched_synth.csv` when present (so synth-under25 is already promoted into canonical `odds_ft_under25`).
  - Optionally adds rolling press features (`--rolling-press`) and writes a merge health report (`--write-report`).

### Goal (λ) regressors
- `_baseline_ftr_pipeline.quick_fit_goal_regressors(df, league_name, ...)`
  - Writes the **5-fold ensembles**:
    - `ModelStore/<LEAGUE_TAG>/goal_ensembles/home_goals_fold5.pkl`
    - `ModelStore/<LEAGUE_TAG>/goal_ensembles/away_goals_fold5.pkl`
    - `ModelStore/<LEAGUE_TAG>/goal_ensembles/lambda_models_manifest.json`

### Market model training
You currently have two training paths:

1) **`train_markets.py`** (preferred when runtime expects its pickles)
- Lightweight, sklearn-based (logistic regression) with pickle-safe calibration wrapper.
- Produces per-market bundles as `ModelStore/<TAG>/<market>.pkl` (and optional `_v2` copies).
- **Patch required / recommended:** make `--matches-csv` optional and default to the canonical merged input:
  - `Matches/__merged__/<LEAGUE_TAG>__merged.csv`
  - This keeps training perfectly aligned with the synth + merge contract.

2) **`train_investor_leagues_v2.py`** (batch trainer; CatBoost)
- Produces market bundles + thresholds and (when not in `--lite`) calibrators.
- Update it to **prefer merged inputs** (and/or synth-enriched inputs) and keep naming consistent with v3 if you use it as canonical.

#### Bundle Map (CatBoost + XGBoost side-by-side)
When running `train_investor_leagues_v2.py` with `--ftr-engine both` and `--ftr-engine-subdir`, bundles are written to:
- CatBoost (primary, legacy-compatible):
  - `ModelStore/<LEAGUE_TAG>/ftr_v2.pkl`
  - `ModelStore/<LEAGUE_TAG>/cat/ftr_v2.pkl`
- XGBoost (parallel):
  - `ModelStore/<LEAGUE_TAG>/ftr_v2_xgb.pkl`
  - `ModelStore/<LEAGUE_TAG>/xgb/ftr_v2.pkl`

Runtime (`bookie_allmarkets.py`) loads both and emits side-by-side columns:
- `ftr_p_home_xgb`, `ftr_p_draw_xgb`, `ftr_p_away_xgb`
- `ftr_pick_xgb`, `ftr_margin_xgb`, `model_p_for_bookie_xgb`

This allows clean Cat vs XGB comparisons without changing deploy logic.

---

## Target leagues

### Normalized league names (spaces)
Use these with CLI flags that accept comma-separated league strings:
- England Premier League
- England Championship
- England EFL League 1
- England FA Cup
- Japan J1
- Norway Eliteserien
- Netherlands Eredivisie
- Belgium Pro
- Scotland Premiership
- Brazil Serie A
- USA MLS
- Portugal Liga
- Spain La Liga
- Italy Serie A
- France Ligue 1
- Germany Bundesliga
- Europa Conference
- Europa League
- Champions League

### League tags (underscores)
File paths and ModelStore folders use tags:
- England_Premier_League
- England_Championship
- England_EFL_League_1
- England_FA_Cup
- Japan_J1
- Norway_Eliteserien
- Netherlands_Eredivisie
- Belgium_Pro
- Scotland_Premiership
- Brazil_Serie_A
- USA_MLS
- Portugal_Liga
- Spain_La_Liga
- Italy_Serie_A
- France_Ligue_1
- Germany_Bundesliga
- Europa_Conference
- Europa_League
- Champions_League

---

## What “healthy” means (pass criteria)

### A) Inputs are correct (merge stage)
Per league merged CSV in `Matches/__merged__/` must have:
- timeline parseable: `match_date` or `date_GMT` (or another supported timestamp column)
- teams: `home_team_name`, `away_team_name`
- realised goals: `home_team_goal_count`, `away_team_goal_count`

### B) Odds coverage (core markets)
We expect (most leagues):
- OU25: `odds_ft_over25` and `odds_ft_under25`
- BTTS: `odds_btts_yes` and `odds_btts_no`
- FTR: `odds_ft_home_team_win`, `odds_ft_draw`, `odds_ft_away_team_win`

If canonical Under 2.5 is missing, it must be filled via synth:
- `odds_ft_under25_synth` exists (diagnostic)
- canonical `odds_ft_under25` populated
- provenance column `odds_source_under25 = "synth_ou25"` for synth-filled rows

**Note:** you already verified your merged files are clean (no odds columns contain `<= 1.0001`).

### C) Rolling press features are present (recommended)
Merged should contain these 6 columns when `--rolling-press` is used:
- `rolling5_home_press_intensity`
- `rolling5_away_press_intensity`
- `rolling5_press_intensity_diff`
- `rolling5_home_press_z`
- `rolling5_away_press_z`
- `rolling5_press_z_diff`

### D) Goal (λ) artifacts exist
After training goal regressors:
- `ModelStore/<TAG>/goal_ensembles/home_goals_fold5.pkl`
- `ModelStore/<TAG>/goal_ensembles/away_goals_fold5.pkl`
- `ModelStore/<TAG>/goal_ensembles/lambda_models_manifest.json`

### E) Market artifacts exist (core)
After market training (choose your canonical naming):
- `ModelStore/<TAG>/ftr_<version>.pkl`
- `ModelStore/<TAG>/over25_<version>.pkl`
- `ModelStore/<TAG>/under25_<version>.pkl`
- `ModelStore/<TAG>/btts_<version>.pkl`

…and compatibility copies if runtime requires:
- `*_v2.pkl` copies and/or unversioned aliases.

### F) Export QC is green
`bookie_allmarkets.py --strict` must pass:
- OU25 no-vig is present (requires both sides)
- BTTS + FTR critical fields present
- strict asserts pass without “silent drift”

---

## Canonical running order

### Phase 0 — Pre-merge: synth odds where needed
If a league is missing canonical Under 2.5:
1) Ensure ODDS synth tables exist:
```bash
python odds_synth.py fit \
  --src "Matches/*/fd_odds_enriched.csv" \
  --out ModelStore/ODDS_SYNTH_TABLES.json \
  --group-by league \
  --markets ou25 btts \
  --known-side over yes \
  --bins 10 \
  --min-n 200
```
2) Apply synth for the league:
```bash
LEAGUE="England EFL League 1"
python odds_synth.py apply \
  --src "Matches/$LEAGUE/fd_odds_enriched.csv" \
  --tables "ModelStore/ODDS_SYNTH_TABLES.json" \
  --out "Matches/$LEAGUE/fd_odds_enriched_synth.csv" \
  --group-by league \
  --conf-min 0.25 \
  --min-n 200
```

### Phase A — Build merged CSVs (single source of truth)
Build merged + rolling press + report (recommended):
```bash
python build_merged.py --leagues "England Premier League,England Championship,England EFL League 1,England FA Cup,Japan J1,Norway Eliteserien,Netherlands Eredivisie,Belgium Pro,Scotland Premiership,Brazil Serie A,USA MLS,Portugal Liga,Spain La Liga,Italy Serie A,France Ligue 1,Germany Bundesliga,Europa Conference,Europa League,Champions League" \
  --recursive \
  --rolling-press \
  --write-report
```
Outputs:
- `Matches/__merged__/<TAG>__merged.csv`
- `Matches/__merged__/__merge_report__.csv`

### Phase B — Train goal regressors (λ home/away)
Batch retrain (per league):
```bash
FORCE_RETRAIN_LAMBDA=1 VERBOSE_QUICK=1 python - <<'PY'
import os
import pandas as pd
import _baseline_ftr_pipeline as m

ROOT = os.path.dirname(m.__file__)
MERGED = os.path.join(ROOT, "Matches", "__merged__")

LEAGUES = [
  "England Premier League",
  "England Championship",
  "England EFL League 1",
  "England FA Cup",
  "Japan J1",
  "Norway Eliteserien",
  "Netherlands Eredivisie",
  "Belgium Pro",
  "Scotland Premiership",
  "Brazil Serie A",
  "USA MLS",
  "Portugal Liga",
  "Spain La Liga",
  "Italy Serie A",
  "France Ligue 1",
  "Germany Bundesliga",
  "Europa Conference",
  "Europa League",
  "Champions League",
]

ok, fail = 0, 0
for league in LEAGUES:
    merged_path = os.path.join(MERGED, f"{league.replace(' ','_')}__merged.csv")
    if not os.path.exists(merged_path):
        print("❌ missing merged:", merged_path)
        fail += 1
        continue

    df = pd.read_csv(merged_path, low_memory=False)
    try: df = m._canonize(df)
    except Exception: pass
    try: df = m._sanitize_pd_na_boolean(df)
    except Exception: pass

    print(f"\n=== TRAIN GOALS: {league} | rows={len(df)} cols={len(df.columns)} ===")
    try:
        m.quick_fit_goal_regressors(df, league)
        ok += 1
    except Exception as e:
        print("❌ train failed:", league, "|", e)
        fail += 1

print(f"\nDONE | ok={ok} fail={fail}")
PY
```

### Phase C — Train core market models (FTR / OU25 / BTTS)

#### Decide the single canonical trainer
Pick ONE for core markets to prevent drift:
- If runtime expects `train_markets.py` bundles → use `train_markets.py`.
- If runtime expects investor-v2/v3 naming + calibrators → use `train_investor_leagues_v2.py`.

#### Required rule (both trainers)
**Train from the merged canonical CSV**:
- `Matches/__merged__/<LEAGUE_TAG>__merged.csv`
This ensures synth-under25 is already promoted into canonical `odds_ft_under25`.

#### `train_markets.py` (recommended default behaviour)
After patching `train_markets.py` to default to merged, the simplest run is:
```bash
python train_markets.py --league "England EFL League 1" --outdir ModelStore
```
It should auto-resolve:
- `Matches/__merged__/England_EFL_League_1__merged.csv`

If you want an explicit smoke test (core only):
```bash
python train_markets.py \
  --league "England EFL League 1" \
  --matches-csv "Matches/__merged__/England_EFL_League_1__merged.csv" \
  --outdir ModelStore \
  --markets "ftr,over25,under25,btts"
```

#### `train_investor_leagues_v2.py` (example)
```bash
python train_investor_leagues_v2.py \
  --leagues "England Championship,England EFL League 1,England FA Cup" \
  --markets "btts,over25,under25,ftr" \
  --market-version v3 \
  --ftr-version v2 \
  --write-v2-compat \
  --val-frac 0.2 \
  --iters 1800 \
  --threads 4 \
  --overwrite
```

### Phase D — Signal bands (only if runtime expects them)
Builder: `fit_signal_bands.py`
```bash
python fit_signal_bands.py --league "England Championship" --outdir ModelStore
python fit_signal_bands.py --league "England EFL League 1" --outdir ModelStore
python fit_signal_bands.py --league "England FA Cup" --outdir ModelStore
```
Verify:
```bash
find ModelStore -maxdepth 3 -type f | rg -n "England_(Championship|EFL_League_1|FA_Cup).*(signal_bands\\.json)$"
```

### Phase E — Export + strict QC (one command gate)
Run export with `--strict` so it fails fast if any market is broken:
```bash
OG_DEBUG_POWER=1 OG_DEBUG_PICK=1 python3 bookie_allmarkets.py \
  --date-from 2026-02-14 \
  --date-to 2026-02-17 \
  --leagues "England Premier League,England Championship,England EFL League 1,England FA Cup,Japan J1,Norway Eliteserien,Netherlands Eredivisie,Belgium Pro,Scotland Premiership,Brazil Serie A,USA MLS,Portugal Liga,Spain La Liga,Italy Serie A,France Ligue 1,Germany Bundesliga,Europa Conference,Europa League,Champions League" \
  --markets ftr,ou25,btts,tg15,tg25 \
  --implied-min 0.62 \
  --ou25-implied-min 0.50 \
  --btts-implied-min 0.50 \
  --tg15-pmin 0.52 \
  --tg25-pmin 0.35 \
  --strict
```

### Phase F — Secondary markets (only after core is stable)
Targets (examples):
- `home_fts` / `away_fts`
- `home_ge2` / `away_ge2`
- `home_ge3` / `away_ge3`
- `btts_fh`

Rule:
- Do not add secondary markets until core retrain is clean across all target leagues.

---

## New League Kickoff Checklist (copy/paste)

```bash
# 0) Put league season CSVs into Matches/<League Name>/
# 1) If canonical under 2.5 odds are missing, synth first (see synth doc)

LEAGUE="<LEAGUE NAME>"

# (optional) enrich season sources if your pipeline uses it
python etl_press_intensity.py --match-dir "Matches/$LEAGUE" --force --overwrite-intensity

# synth odds (only if needed)
python odds_synth.py apply --src "Matches/$LEAGUE/fd_odds_enriched.csv" --tables "ModelStore/ODDS_SYNTH_TABLES.json" --out "Matches/$LEAGUE/fd_odds_enriched_synth.csv" --group-by league --conf-min 0.25 --min-n 200

# build merged canonical dataset (prefers *_synth.csv automatically)
python build_merged.py --leagues "$LEAGUE" --recursive --rolling-press --write-report

# train goal (λ) fold5
FORCE_RETRAIN_LAMBDA=1 VERBOSE_QUICK=1 python - <<'PY'
import os, pandas as pd
import _baseline_ftr_pipeline as m
league = "<LEAGUE NAME>"
root=os.path.dirname(m.__file__)
mp=os.path.join(root,"Matches","__merged__",f"{league.replace(' ','_')}__merged.csv")
df=pd.read_csv(mp, low_memory=False)
try: df=m._canonize(df)
except Exception: pass
try: df=m._sanitize_pd_na_boolean(df)
except Exception: pass
m.quick_fit_goal_regressors(df, league)
print("✅ goal fold5 written")
PY

# train core markets (choose ONE canonical trainer)
python train_markets.py --league "$LEAGUE" --outdir ModelStore
# OR
# python train_investor_leagues_v2.py --leagues "$LEAGUE" --markets "btts,over25,under25,ftr" --market-version v3 --write-v2-compat --overwrite

# optional if runtime expects it
python fit_signal_bands.py --league "$LEAGUE" --outdir ModelStore

# strict QC smoke test
python3 bookie_allmarkets.py --date-from 2026-02-14 --date-to 2026-02-17 --leagues "$LEAGUE" --markets ftr,ou25,btts --strict
```

---

## Notes / gotchas

- MLS / Japan / Brazil / UEFA comps can be sparse/out-of-sync — training is fine as long as the merged file is coherent and completed fixtures are filtered correctly.
- If you see date parsing warnings, it’s usually safe; standardize date formats upstream if needed.
- Keep `--strict` in export runs. It’s your “one command = QC green” gate.
