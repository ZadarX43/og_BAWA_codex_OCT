# BAWA PORTO - Data Update Checklist Runbook

This runbook is the operational version of `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/DATA_UPDATE_CHECKLIST.md`.

Use it when:
- full fresh data has been downloaded for all leagues
- weekend prediction prep is starting
- merged files need to be rebuilt safely
- we want one repeatable recovery path if any stage fails

The rule is simple:

- run every stage in order
- do not skip source-generation stages
- do not skip patch stages
- do not run predictions until the integrity spot-check passes

---

## Stage Order

1. Ingest new match data
2. Press ETL loop
3. Base merged rebuild
4. H2H + streak patch
5. Power ratings generation
6. Power ratings merge-back
7. Synth odds generation
8. Synth odds patch
9. Integrity spot-check
10. Predictions or retrain

---

## Stage 1 - Ingest New Match Data

Run your normal ingest/update process first.

Goal:
- latest results and upcoming fixtures are present in the league source CSVs

Verify:
- open one recently updated league file and confirm this week's matches are there
- confirm dates/results are current before doing any merge work

If this fails, do this:
- stop immediately
- fix the raw source ingest first
- do not run any rebuild or patch step on stale source data

---

## Stage 2 - Press ETL

```bash
cd '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO'

for dir in Matches/*/; do
  league=$(basename "$dir")
  if [ "$league" = "__merged__" ] || [ "$league" = "Upcoming Fixtures" ]; then
    continue
  fi
  league_tag=$(printf '%s' "$league" | tr ' ' '_')
  ./.venv/bin/python etl_press_intensity.py \
    --match-dir "$dir" \
    --merged-csv "Matches/__merged__/${league_tag}__merged.csv" \
    2>&1
done | tee /tmp/press_etl.log
```

Verify:
- no `ERROR` lines in `/tmp/press_etl.log`

Expected behavior:
- some leagues will still have sparse press intensity after this
- that is acceptable if it is source-data sparsity, not ETL failure

If this fails, do this:
- inspect `/tmp/press_etl.log`
- check whether one league folder has malformed/new headers
- rerun just the failed league manually first
- if multiple leagues fail, do not continue to merged rebuild until ETL is clean

---

## Stage 3 - Rebuild Base Merged Files

```bash
GOAL_REGRESSOR_SCHEMA=legacy \
./.venv/bin/python build_merged.py --recursive --rolling-press \
  2>&1 | tee /tmp/build_merged.log
```

Verify:
- merged files are recreated under `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/Matches/__merged__`
- spot-check one merged file and confirm base column count is around `450`

Important:
- this stage creates the base merged file only
- this stage does **not** restore all enriched families by itself

If this fails, do this:
- inspect `/tmp/build_merged.log`
- fix the failing league/source issue
- rerun Stage 3 only after source data is healthy
- do not move to predictions from here because enriched families are still missing

---

## Stage 4 - Patch H2H + Streaks

```bash
./.venv/bin/python patch_merge_add_streaks.py \
  2>&1 | tee /tmp/streaks_patch.log
```

Verify:
- all leagues patch cleanly
- merged column count rises to about `537+`

Expected additions:
- H2H block
- streak block
- related composites used by BTTS/FTR/OU25 models

If this fails, do this:
- inspect `/tmp/streaks_patch.log`
- if only one or two leagues fail, rerun the script after fixing those league rows
- if the wrapper fails globally, do not continue; merged files are incomplete

---

## Stage 5 - Generate Power Ratings

```bash
for league in $(python3 - <<'PY'
from pathlib import Path
for f in Path('Matches/__merged__').glob('*__merged.csv'):
    print(f.stem.replace('__merged',''))
PY
); do
  league_name=$(printf '%s' "$league" | tr '_' ' ')
  ./.venv/bin/python team_ratings.py \
    --league "$league_name" \
    --mode rolling \
    2>&1
done | tee /tmp/power_ratings.log
```

Verify:
- no `ERROR` lines
- side artifacts are written into `ModelStore/`

If this fails, do this:
- check whether the failing league name/league tag conversion broke
- confirm the merged file exists for that league
- rerun just the failed league first

---

## Stage 6 - Merge Back Power Ratings

```bash
./.venv/bin/python patch_merge_add_power_ratings.py \
  2>&1 | tee /tmp/power_patch.log
```

Verify:
- `29/29` leagues patched
- no `SKIP` or `ERROR` rows
- patched files now contain `home_power_rating`, `away_power_rating`, `power_diff`

If this fails, do this:
- inspect `/tmp/power_patch.log`
- confirm the side artifact exists in `ModelStore/`
- confirm join keys still match current merged schema
- rerun Stage 5 and Stage 6 together if in doubt

---

## Stage 7 - Generate Synth Odds

```bash
./.venv/bin/python make_fd_odds_enriched_synth.py --emit-ou25-novig \
  2>&1 | tee /tmp/synth_gen.log
```

Verify:
- EPL and Scotland show `ou25_novig_rows > 0`

Notes:
- synth is source-limited
- EPL and Scotland are the key leagues where this is expected to matter immediately

If this fails, do this:
- inspect `/tmp/synth_gen.log`
- confirm the underlying odds source files still exist and are current
- if EPL and Scotland produce zero rows, do not treat synth as healthy

---

## Stage 8 - Patch Synth Odds Into Merged

```bash
./.venv/bin/python patch_merge_add_synth_odds.py \
  --root "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" \
  --overwrite --harmonize-duplicates \
  2>&1 | tee /tmp/synth_patch.log
```

Verify:
- EPL `under25_nonnull` is about `97%+`
- Scotland `under25_nonnull` is about `98%+`

Important:
- as of 2026-04-17, synth training wiring also depends on the identity-mapping fix in:
  - `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/prediction_overlay.py`

If this fails, do this:
- inspect `/tmp/synth_patch.log`
- confirm the synth generation stage produced source rows
- rerun Stage 7 then Stage 8 together

---

## Stage 9 - Integrity Spot-Check

Run this before any prediction run or retrain.

```bash
python3 -c "
import pandas as pd
from pathlib import Path

check_leagues = ['England_Premier_League', 'Netherlands_Eredivisie', 'Japan_J1', 'Europa_Conference']
families = {
    'h2h': 'h2h_btts_rate',
    'streak': 'btts_streak_home_x',
    'power': 'home_power_rating',
    'press': 'home_press_baseline',
    'synth': 'p_over25_novig',
}

print(f'League                  | cols | h2h | streak | power | press | synth')
print('-' * 72)
for lg in check_leagues:
    path = Path(f'Matches/__merged__/{lg}__merged.csv')
    if not path.exists():
        print(f'{lg:<24}| MISSING')
        continue
    df = pd.read_csv(path, nrows=50)
    cols = len(df.columns)
    checks = {}
    for fam, col in families.items():
        if col not in df.columns:
            checks[fam] = 'MISSING'
        else:
            nan_pct = df[col].isna().mean()
            checks[fam] = 'OK' if nan_pct < 0.5 else f'{nan_pct:.0%}NaN'
    print(f'{lg:<24}| {cols:<5}| {checks[\"h2h\"]:<5}| {checks[\"streak\"]:<7}| {checks[\"power\"]:<6}| {checks[\"press\"]:<6}| {checks[\"synth\"]}')
"
```

Pass condition:
- merged cols are `550+`
- `h2h: OK`
- `streak: OK`
- `power: OK`
- `press: OK`
- `synth: OK` for EPL/Scotland expectations, or known high-NaN where source-limited elsewhere

If this fails, do this:
- do **not** run predictions
- identify which family is missing
- map that family to its generating and patch stages:
  - H2H/streak -> Stages 3 and 4
  - power -> Stages 5 and 6
  - synth -> Stages 7 and 8
  - press -> Stage 2 plus Stage 3
- rerun from the failed family’s source stage, not just the final patch

---

## Stage 10 - Retrain Or Run Predictions

Only run this after Stage 9 passes.

### Predictions

```bash
./.venv/bin/python bookie_allmarkets.py \
  [your normal prediction flags] \
  2>&1 | tee /tmp/predictions_$(date +%Y%m%d).log
```

### Retrain

If a rebuild was intended to update model state too, retrain after the spot-check passes.

If retrain fails, do this:
- inspect the trainer log
- confirm merged files used in training still contain the restored feature families
- verify `features_manifest.json` in the touched `ModelStore` league folder

---

## Failure Map

### If H2H or streak columns are missing

Likely issue:
- Stage 4 skipped or failed

Do this:
- rerun Stage 4
- re-run Stage 9

### If power columns are missing

Likely issue:
- Stage 5 did not create side artifacts
- or Stage 6 failed to merge them back

Do this:
- rerun Stages 5 and 6
- re-run Stage 9

### If synth columns are missing from merged

Likely issue:
- Stage 7 or Stage 8 failed

Do this:
- rerun Stages 7 and 8
- re-run Stage 9

### If synth columns exist in merged but not in trained bundles

Likely issue:
- trainer/rename layer dropped them before feature-frame construction

As of 2026-04-17:
- this was fixed in `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/prediction_overlay.py`
- canonical synth columns must survive `apply_safe_renames_and_whitelist()`

Do this:
- verify `features_manifest.json` includes synth fields
- retrain with `--overwrite`
- re-check the actual `over25_v3.pkl` and `under25_v3.pkl` bundle feature lists

### If press looks sparse

Likely issue:
- either ETL failed
- or source data is genuinely sparse

Do this:
- check Stage 2 log first
- if log is clean and baseline/pre-match/snap press columns exist, treat remaining sparsity as source-limited rather than pipeline failure

---

## Weekend Operating Rule

For prediction weekends:

- always finish Stage 9 before any live picks run
- if one feature family is questionable, fall back to the last locked baseline rather than deploying on half-rebuilt data
- if a two-league targeted enhancement is being tested, validate it separately before making it part of the general weekend model state

---

## Integration Notes For Desktop Workflow

Current desktop app:
- `OG All-In-One Pipeline.app`
- source script:
  - `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/apps/OG_All_In_One_Pipeline.applescript`

Current gap:
- the desktop workflow description still assumes:
  - ingest
  - merged/dashboard
  - picks pipeline

But the real production-safe rebuild path now also requires:
- press ETL
- streak patch
- power generation + patch
- synth generation + patch
- integrity spot-check

That means the future desktop workflow should be updated to expose:

1. A full “Refresh Data Stack” action
2. Per-stage logs
3. A mandatory integrity status screen
4. A weekend-safe block that prevents picks if Stage 9 fails

---

## Next App / Script Upgrade Targets

Once fresh full-league data is downloaded, the next maintenance phase should update:

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/footystats_drop_ingest.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/merged_pipeline_dashboard.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/apps/OG_All_In_One_Pipeline.applescript`

So the workflow supports:
- full update chain visibility
- stage ordering
- stage logs
- integrity pass/fail gating
- synth and enrichment awareness

That is the path to safer future prediction weekends.
