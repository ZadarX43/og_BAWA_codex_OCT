

# ODDS_SYNTH: Under 2.5 Synthesis + Merge Pipeline (OG / BAWA)

This document is the single source of truth for:
1) how we synthesize missing Under 2.5 odds (and BTTS NO) from the known side, and  
2) how we ensure each league’s merged dataset uses those synthesized odds downstream.

Goal: forward fixtures often have only one side quoted (e.g. Over2.5 but not Under2.5).  
We synthesize the missing side safely, then merge it into the canonical league data so no-vig OU25 and OU25 models can run consistently.

---

## 1. What this solves

### Common feed problem
For a large number of fixtures, we have:
- `odds_ft_over25` present
- `odds_ft_under25` missing

Without under odds we cannot:
- compute proper OU25 no-vig (requires both sides)
- build stable OU25 features
- run OU25 predictions consistently

### The approach
We learn a robust mapping from the known side implied probability to the missing side:
- Known side: Over 2.5 (or BTTS YES)
- Missing side: Under 2.5 (or BTTS NO)

The method is “rulebook” + “safety gates”, not a fragile regression.
It refuses to hallucinate when confidence is low.

---

## 2. Files and outputs

### Code
- `odds_synth.py`  
  CLI supports:
  - `fit`: builds rulebook tables (JSON)
  - `backtest`: evaluates holdout window
  - `apply`: writes per-league CSV with synth columns + canonical promotion

### Tables
- `ModelStore/ODDS_SYNTH_TABLES.json`  
Contains:
- `tables.ou25` (known_side = over)
- `tables.btts` (known_side = yes)
- per-group bins + a global fallback group

### Per-league output
Example:
- `Matches/England EFL League 1/fd_odds_enriched_synth.csv`

Adds:
- `odds_ft_under25_synth`
- `under25_synth_conf`
- `under25_synth_reason`
- `under25_synth_group`
- `under25_synth_bin`

Also (important):
- **promotes synth into canonical** `odds_ft_under25` when missing
- writes provenance tag `odds_source_under25 = "synth_ou25"` for rows filled by synth

---

## 3. Odds synthesis method (OU25 + BTTS)

### Inputs (OU25)
Known-side candidates (deploy-time):
- priority list prefers OG columns first, then FD columns
- main expected known column: `odds_ft_over25`

Missing-side truth (training-time):
- priority list prefers FD under first, then fallback columns
- main expected missing column: `fd_odds_ft_under25` or `odds_ft_under25`

### Core model concept
Let:
- `p_known = 1 / odds_known`
- `p_miss  = 1 / odds_missing`
- `R = p_known + p_miss`  (overround-like total implied)

We bin fixtures by `p_known` quantiles (default 10 bins):
- for each bin we store:
  - `R_median`, `R_q25`, `R_q75`, `R_p05`, `R_p95`
  - missing odds quantiles: `miss_q10`, `miss_q25`, `miss_med`, `miss_q75`, `miss_q90`
  - support count `n`

### Prediction-time synthesis
For a row with known odds:
1) compute `p_known`
2) select the bin based on `p_known`
3) estimate `p_miss_hat = R_median - p_known`
4) convert to odds: `odds_miss_hat = 1 / p_miss_hat`
5) clamp `odds_miss_hat` to `[miss_q10, miss_q90]`
6) compute confidence score from:
   - bin support size (n)
   - how tight R distribution is (IQR)
7) drop if confidence < `--conf-min` (default 0.25)

### Safety gates (refusal behavior)
Synthesis drops a row if:
- known odds missing or invalid (<=1.0001)
- computed `p_miss_hat` is non-positive
- group/bin has insufficient support
- confidence is below threshold

This is intentional: “no hallucinations”.

### Global fallback
If a league (group) does not meet `--min-n` (default 200),
`apply` falls back to a `__GLOBAL__` rulebook built from all leagues.

This means:
- We can synthesize for leagues with weak/partial history
- But it’s not “bespoke league mapping” unless that league has enough historical both-sided rows

That’s expected and desired for forward-fixture coverage.

---

## 4. CLI commands

### 4.1 Fit tables
Example:
```bash
python odds_synth.py fit \
  --src "Matches/*/fd_odds_enriched.csv" \
  --out ModelStore/ODDS_SYNTH_TABLES.json \
  --group-by league \
  --markets ou25 btts \
  --known-side over yes \
  --bins 10 \
  --min-n 200

Expected print:
	•	[OK] wrote tables: ...
	•	ou25: groups=... known_side=over
	•	btts: groups=... known_side=yes

4.2 Apply to a league (generate synth file)

Example:

python odds_synth.py apply \
  --src "Matches/England EFL League 1/fd_odds_enriched.csv" \
  --tables "ModelStore/ODDS_SYNTH_TABLES.json" \
  --out "Matches/England EFL League 1/fd_odds_enriched_synth.csv" \
  --group-by league \
  --conf-min 0.25 \
  --min-n 200

Expected print:
	•	[OK] wrote: ..._synth.csv
	•	under25_synth nonnull: N/Total
	•	bttsno_synth nonnull: N/Total

4.3 QA: reason counts

Run after apply:

python - <<'PY'
import pandas as pd
p="Matches/England EFL League 1/fd_odds_enriched_synth.csv"
df=pd.read_csv(p, low_memory=False)
print(df["under25_synth_reason"].fillna("NA").value_counts())
print(df["bttsno_synth_reason"].fillna("NA").value_counts())
PY

Interpretation:
	•	ok = synthesized successfully
	•	known_missing = known side missing/invalid (common if odds are 0.0 placeholders)
	•	low_conf = bin/group exists but confidence below threshold
	•	no_group/no_bins = table coverage issue (should be rare if global fallback exists)

⸻

5. Canonical columns + downstream expectations

Canonical OU25 columns (what downstream should read)
	•	odds_ft_over25
	•	odds_ft_under25

After apply:
	•	if odds_ft_under25 was missing AND synth produced odds:
	•	odds_ft_under25 is filled with synth value
	•	odds_source_under25 set to synth_ou25

Diagnostics columns (do not depend on downstream)
	•	odds_ft_under25_synth
	•	under25_synth_conf
	•	under25_synth_reason
	•	under25_synth_group
	•	under25_synth_bin

Special note on invalid odds placeholders

Some historical rows contain odds_ft_over25 = 0.0 (and/or odds_over25 = 0.0).
These are treated as missing and will produce:
	•	under25_synth_reason = known_missing

We should convert 0.0 odds to NaN during merge/clean to avoid confusion.

⸻

## 6. Merge pipeline: making synth permanent per league

### The rule

For each league, the “merged” canonical dataset must prefer the synthesized file if it exists.

Priority:
1. `fd_odds_enriched_synth.csv` (preferred; includes canonical under25 filled + provenance)
2. `fd_odds_enriched.csv` (fallback)
3. otherwise: merge all readable league CSVs (season files etc.) excluding obvious artifacts (upcoming/fixtures/reports)

### Output target

We standardize per league merged files into:
- `Matches/__merged__/<League_With_Underscores>__merged.csv`

This merged file is the input for:
- feature enrichment pipelines
- goal regressors / lambda models
- OU25 no-vig calculations and OU25 models

### Contract

After merge:
- `odds_ft_over25` exists for the league rows where over odds are known
- `odds_ft_under25` is present whenever synth filled it
- downstream code should read canonical columns only

---

## 6.1 `build_merged.py` (the canonical merger script)

We now use a dedicated merger script so we stop re-copying notebook snippets.

### What it does

- Builds `Matches/__merged__/<league>__merged.csv` for one or more leagues.
- Prefers `fd_odds_enriched_synth.csv` when present (so synthesized Under2.5 is already promoted into canonical `odds_ft_under25`).
- Otherwise falls back to `fd_odds_enriched.csv`, or (if needed) merges the league’s season CSVs.
- Optional: computes rolling press features directly inside the merged output.
- Optional: writes a small per-league health report CSV.

### How to use it for your exact current situation (the “9 failing leagues”)

1) Rebuild merged (using synth where available; otherwise merges the league CSVs):

```bash
python build_merged.py --leagues \
  "Brazil Serie A" \
  "Champions League" \
  "England Championship" \
  "England EFL League 1" \
  "England FA Cup" \
  "England U21" \
  "Europa Conference" \
  "Japan J1" \
  "Norway Eliteserien" \
  --recursive
```

2) Then compute rolling press features in one pass (this replaces the old Step 3 notebook snippet):

```bash
python build_merged.py --leagues \
  "Brazil Serie A" \
  "Champions League" \
  "England Championship" \
  "England EFL League 1" \
  "England FA Cup" \
  "England U21" \
  "Europa Conference" \
  "Japan J1" \
  "Norway Eliteserien" \
  --recursive \
  --rolling-press
```

3) Then retrain goal regressors (your existing command stays the same).

### Useful “pro” option: write a health report

This dumps a compact CSV so you can see coverage at a glance before training.

```bash
python build_merged.py --leagues \
  "Brazil Serie A" \
  "Champions League" \
  "England Championship" \
  "England EFL League 1" \
  "England FA Cup" \
  "England U21" \
  "Europa Conference" \
  "Japan J1" \
  "Norway Eliteserien" \
  --recursive \
  --rolling-press \
  --write-report
```

Default report path:
- `Matches/__merged__/__merge_report__.csv`

(If supported in your script build) you can also set:
- `--report-path ModelStore/merge_health_report.csv`

---

## 6.2 Where ETL press intensity + streaks should live

### Press intensity ETL

**Recommendation:** keep the raw press-intensity ETL as a separate explicit step (it writes into the season CSVs), and keep `build_merged.py` responsible for *merging + optional rolling features*.

Reason: ETL steps can change source files (overwrite columns), so you want to run them intentionally (with `--force/--overwrite-*`) rather than “silently” inside merge.

Typical order for a new / fixed league:
1) Run ETL that enriches season CSVs (press intensity, other enrichers)
2) Run `build_merged.py` to build the merged canonical file
3) (Optional) Run `build_merged.py --rolling-press` to add rolling features
4) Train goal regressors / models

If you later want full consolidation, add an opt-in flag like `--etl-press` that shells out to `etl_press_intensity.py` before merging — but keep it **off by default**.

### Streaks module (team form features)

Same principle:
- If streaks are derived purely from match results inside the merged file, they *can* live in `build_merged.py` behind a flag (e.g. `--streaks`).
- If streaks require external inputs / rewrite season sources, keep it as an explicit ETL step.

For now, focus `build_merged.py` on:
- deterministic merging
- robust timeline parsing
- optional rolling press columns
- report output

That’s enough to stop us looping and gives a clear “ingest → merge → train” runway.

7. Recommended QA checks after merge

7.1 Coverage sanity
	•	over25 nn should be near total if feed provides it
	•	under25 canonical nn should increase after synth

Example:
	•	over25 nn: 4405
	•	under25 canonical nn: 4135
	•	source_under25 synth nn: 4135

7.2 Check remaining missing under25

If odds_ft_under25 still missing after synth, inspect:
	•	under25_synth_reason value counts
	•	common causes:
	•	known side is 0.0 placeholder (known_missing)
	•	confidence too low in bin (low_conf)

7.3 Ensure no-vig OU25 can be computed

No-vig requires both sides (implied sum).
Any match with missing under25 cannot get a no-vig OU25 line.

⸻

8. “Done” definition for this stage

We consider this stage complete when:
	1.	ODDS_SYNTH_TABLES.json exists and is reproducible from fit
	2.	apply produces _synth.csv per league
	3.	Merge pipeline prefers _synth.csv so canonical odds_ft_under25 is populated
	4.	Downstream OU25 no-vig + OU25 model steps can run without special casing per league
	5.	QA logs (reason counts) are easy to run and interpret

⸻

Appendix A: What “global fallback” means in practice
	•	If a league lacks enough historical both-sided rows (min_n=200),
it won’t have a stable league-specific rulebook.
	•	apply will use the __GLOBAL__ rulebook instead.
	•	This is acceptable and expected for leagues with sparse history, and still improves
forward-fixture coverage dramatically.

If you later want league-bespoke behavior everywhere:
	•	lower --min-n (at some risk)
	•	or fit tables with additional grouping logic (league,od_source) where reliable
	•	or introduce a hierarchical fallback: league → country → global

For now, global fallback is the correct operational choice.

⸻


---

## New League Kickoff Checklist (copy/paste bootstrap)

```bash
# 0) choose your league folder name exactly as in Matches/
LEAGUE="England EFL League 1"

# 1) (optional) enrich season sources (press intensity) if you rely on it
python etl_press_intensity.py --match-dir "Matches/$LEAGUE" --force --overwrite-intensity

# 2) ensure ODDS synth tables exist (run occasionally; safe to rerun)
python odds_synth.py fit --src "Matches/*/fd_odds_enriched.csv" --out ModelStore/ODDS_SYNTH_TABLES.json --group-by league --markets ou25 btts --known-side over yes --bins 10 --min-n 200

# 3) synthesize missing Under2.5 + BTTS NO and promote into canonical odds
python odds_synth.py apply --src "Matches/$LEAGUE/fd_odds_enriched.csv" --tables "ModelStore/ODDS_SYNTH_TABLES.json" --out "Matches/$LEAGUE/fd_odds_enriched_synth.csv" --group-by league --conf-min 0.25 --min-n 200

# 4) build merged canonical dataset (prefers *_synth.csv automatically)
python build_merged.py --leagues "$LEAGUE" --recursive --rolling-press --write-report

# 5) retrain goal regressors / lambda ensembles for the league
FORCE_RETRAIN_LAMBDA=1 VERBOSE_QUICK=1 python - <<'PY'
import os, pandas as pd
import _baseline_ftr_pipeline as m
league=os.environ.get('LEAGUE','England EFL League 1')
root=os.path.dirname(m.__file__)
mp=os.path.join(root,'Matches','__merged__', f"{league.replace(' ','_')}__merged.csv")
df=pd.read_csv(mp, low_memory=False)
try: df=m._canonize(df)
except Exception: pass
try: df=m._sanitize_pd_na_boolean(df)
except Exception: pass
m.quick_fit_goal_regressors(df, league)
print('OK trained goal regressors for:', league)
PY
```
---





usage: merge_fd_odds_into_matches.py [-h] [--matches-root MATCHES_ROOT] [--fd-root FD_ROOT] [--out-tag OUT_TAG] [--leagues LEAGUES] [--overrides OVERRIDES] [--min-match-score MIN_MATCH_SCORE]
                                     [--ftr-src {PS,B365}] [--over25-src {P,B365}] [--under25-only] [--overwrite]

Merge football-data.co.uk odds (incl Under 2.5) into OG Matches

options:
  -h, --help            show this help message and exit
  --matches-root MATCHES_ROOT
  --fd-root FD_ROOT
  --out-tag OUT_TAG
  --leagues LEAGUES     Comma-separated OG leagues; default = multi-file mapped leagues
  --overrides OVERRIDES
                        CSV with columns: league,fd_team,og_team_override
  --min-match-score MIN_MATCH_SCORE
  --ftr-src {PS,B365}   Preferred 1X2 odds source
  --over25-src {P,B365}
                        Preferred O/U 2.5 odds source
  --under25-only        Only merge Under 2.5 odds into OG frames (leave 1X2 and Over2.5 untouched).
  --overwrite, --overwrites
                        Overwrite existing enriched file
