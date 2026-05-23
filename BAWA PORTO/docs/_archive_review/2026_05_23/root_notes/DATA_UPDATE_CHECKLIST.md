# BAWA PORTO - Data Update Master Checklist

Run every step in this exact order. Do not skip steps even if you think the data has not changed for a league.

## Stage 1 - Ingest New Match Data

```bash
# Update season match CSVs with latest results and fixtures
# Whatever your normal ingest process is - run it first
# Verify new rows landed in the source match files before proceeding
```

Verify:
- Spot-check one league's most recent match CSV has this week's results.

## Stage 2 - Press ETL (per-league loop)

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
- No `ERROR` lines in `/tmp/press_etl.log`.

## Stage 3 - Rebuild Merged Files

```bash
GOAL_REGRESSOR_SCHEMA=legacy \
./.venv/bin/python build_merged.py --recursive --rolling-press \
  2>&1 | tee /tmp/build_merged.log
```

Verify:
- Spot-check one merged file.
- Column count should be about `450` before enrichment patches.

## Stage 4 - H2H + Streaks Patch

```bash
./.venv/bin/python patch_merge_add_streaks.py \
  2>&1 | tee /tmp/streaks_patch.log
```

Verify:
- All 29 leagues show success.
- No errors.
- Column count should rise to about `537+`.

## Stage 5 - Power Ratings Generation (per-league loop)

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
- No `ERROR` lines.
- Side artifacts written into `ModelStore/`.

## Stage 6 - Power Ratings Merge-Back

```bash
./.venv/bin/python patch_merge_add_power_ratings.py \
  2>&1 | tee /tmp/power_patch.log
```

Verify:
- `29/29` leagues patched.
- `max_nan < 20%` on all.
- No `SKIP` or `ERROR` rows.

## Stage 7 - Synth Odds Generation

```bash
./.venv/bin/python make_fd_odds_enriched_synth.py --emit-ou25-novig \
  2>&1 | tee /tmp/synth_gen.log
```

Verify:
- EPL and Scotland show `ou25_novig_rows > 0`.

## Stage 8 - Synth Odds Patch

```bash
./.venv/bin/python patch_merge_add_synth_odds.py \
  --root "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" \
  --overwrite --harmonize-duplicates \
  2>&1 | tee /tmp/synth_patch.log
```

Verify:
- EPL shows `under25_nonnull` about `97%+`.
- Scotland shows `under25_nonnull` about `98%+`.

## Stage 9 - Quick Integrity Spot-Check

Run this before any prediction run or retrain:

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
- All four check leagues should show:
  - `cols: 550+`
  - `h2h: OK`
  - `streak: OK`
  - `power: OK`
  - `press: OK`
  - `synth: OK` for EPL and Scotland only; others can show high NaN and that is expected.

If anything shows `MISSING`:
- Do not run predictions.
- Find which stage failed.
- Rerun from that stage.

## Stage 10 - Run Predictions

Only run after Stage 9 passes cleanly.

```bash
./.venv/bin/python bookie_allmarkets.py \
  [your normal prediction flags] \
  2>&1 | tee /tmp/predictions_$(date +%Y%m%d).log
```

## Key Principle

The merged files are not self-contained.

- `build_merged.py` produces a base file.
- Stages 4, 6, and 8 each add feature families on top.
- If you rebuild merged without running those patch stages afterward, you silently lose:
  - H2H / streaks
  - power / ELO
  - synth odds

Stages 2, 5, and 7 generate the source data.

Stages 4, 6, and 8 patch it into merged.

Both halves of each pair must run.

Stage 9 is the safety net.

- Run it every time before predictions.
- It takes seconds.
- It catches missed stages before they turn into another Phase 3A-style integrity problem.
