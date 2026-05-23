# MERGED_REBUILD_PIPELINE_README.md

## Purpose

This document is the source of truth for rebuilding clean merged match datasets in OG / BAWA after FootyStats raw files are refreshed.

It covers:

1. how `build_merged.py` now selects valid source files
2. how merged files are rebuilt safely
3. how duplicate contamination was fixed
4. how to QA merged outputs before training or deployment

---

## Problem this fixed

Historically, some merged league files became contaminated by bad fallback inputs such as:

- `__TRAIN_MULTISEASON_completed.csv`
- `matches.csv`
- other non-canonical helper / audit / artifact CSVs

This caused:

- duplicate fixtures
- duplicate `fixture_key`
- duplicate `home_team_name + away_team_name + date`
- polluted merged datasets for cup / European competitions
- downstream training and audit instability

Examples that were fixed:

- Champions League
- England FA Cup
- Europa Conference

---

## Current merge rules

`build_merged.py` now works with this priority:

### Preferred inputs
If present, it uses exactly one of:

1. `fd_odds_enriched_synth.csv`
2. `fd_odds_enriched.csv`

These are preferred because they are already enriched / canonical.

### Fallback mode
If neither preferred file exists, it falls back to season match files only.

Accepted fallback filenames must match the season raw FootyStats pattern:

- `england-championship-matches-2020-to-2021-stats.csv`
- `england-fa-cup-matches-2023-to-2024-stats.csv`
- `europe-uefa-champions-league-matches-2025-to-2026-stats.csv`
- and variants with duplicate browser suffixes:
  - `...stats (1).csv`
  - `...stats (2).csv`

### Explicitly rejected in fallback mode
Files like these must never be merged:

- `__TRAIN_MULTISEASON_completed.csv`
- `matches.csv`
- `matches.csv.bak`
- `fd_odds_enriched.csv` during fallback collection
- `fd_odds_enriched_synth.csv` during fallback collection
- `fd_ou25_novig.csv`
- prediction / report / upcoming / audit outputs
- other artifact files

---

## What `build_merged.py` now guarantees

### 1. Stable datetime parsing
Merged files automatically backfill datetime using this order:

1. `match_date`
2. `date_GMT`
3. `timestamp`

### 2. Stable `match_date`
If `match_date` is blank, it is rebuilt from the parsed datetime in normalized date form.

### 3. Stable `fixture_key`
If `fixture_key` is blank, it is rebuilt from:

- normalized date
- normalized home team
- normalized away team

Format:

`YYYY_MM_DD_Home_Team_Away_Team`

Example:

`2025_08_02_AFC_Croydon_Athletic_Roffey`

### 4. Under 2.5 canonical promotion
If `odds_ft_under25_synth` exists and canonical `odds_ft_under25` is missing, it is promoted automatically.

`odds_source_under25` is also maintained properly:
- `synth_ou25`
- `orig`

### 5. Odds placeholder cleanup
Invalid odds placeholders are converted to `NaN`, including values like:

- `0.0`
- `1.0`
- `1.0001`

---

## Canonical rebuild workflow

## Step 1 — Refresh raw FootyStats files
Use the FootyStats ingest workflow to place the newest season files into the correct `Matches/`, `Teams/`, and `Players/` league folders.

## Step 2 — Build / refresh all-seasons enriched file if needed
For leagues where `fd_odds_enriched.csv` must be rebuilt from season files, concatenate only valid season match CSVs.

This is mainly relevant for leagues where enriched files do not yet exist or need rebuilding.

## Step 3 — Fit / refresh synth tables
Run:

```bash
python odds_synth.py fit \
  --src "Matches/*/fd_odds_enriched.csv" \
  --out ModelStore/ODDS_SYNTH_TABLES.json \
  --group-by league \
  --markets ou25 btts \
  --known-side over yes \
  --bins 10 \
  --min-n 200

Step 4 — Apply synth where appropriate

Run per league:

python odds_synth.py apply \
  --src "Matches/<LEAGUE>/fd_odds_enriched.csv" \
  --tables "ModelStore/ODDS_SYNTH_TABLES.json" \
  --out "Matches/<LEAGUE>/fd_odds_enriched_synth.csv" \
  --group-by league \
  --conf-min 0.25 \
  --min-n 200

Step 5 — Build merged files

Single league:

python build_merged.py --league "England EFL League 1" --recursive --rolling-press --write-report

Multiple leagues:

python build_merged.py --leagues \
  "England Championship" \
  "England EFL League 1" \
  "France Ligue 1" \
  "Spain La Liga" \
  --recursive --rolling-press --write-report

All leagues:

python build_merged.py --all --recursive --rolling-press --write-report

### Built-in dedupe report

You can now ask `build_merged.py` to rebuild and write a duplicate-audit report in the same run.

Single league:

```bash
python3 build_merged.py --league "Champions League" --recursive --write-report --dedupe-report
```

Batch example:

```bash
python3 build_merged.py --leagues "Champions League" "England FA Cup" "Europa Conference" --recursive --write-report --dedupe-report
```

Default dedupe output:

- `Matches/__merged__/__merged_dedupe_report__.csv`

Custom dedupe report path:

```bash
python3 build_merged.py --all --recursive --write-report --dedupe-report \
  --dedupe-report-path "ModelStore/logs/merged_dedupe_report.csv"
```

⸻

Rebuild script

We also use:

./rebuild_all_merged.sh

This handles the multi-step process:
	1.	rebuild all-seasons fd_odds_enriched.csv for selected leagues
	2.	fit / refresh ODDS synth tables
	3.	apply synth where source enriched files exist
	4.	rebuild merged files
	5.	write QA logs

Typical outputs:
	•	ModelStore/logs/rebuild_all_merged__synth_log.txt
	•	ModelStore/logs/rebuild_all_merged__merge_log.txt
	•	ModelStore/logs/rebuild_all_merged__qa.csv
	•	Matches/__merged__/__merged_dedupe_report__.csv (when `--dedupe-report` is used)

⸻

QA checks

If you use `--dedupe-report`, the duplicate audit is written automatically during the rebuild run.

After rebuild, merged files should be audited for:
	1.	fixture_key duplicate rows
	2.	exact duplicate rows
	3.	duplicate home_team_name + away_team_name + normalized date
	4.	key coverage:
	•	fixture_key
	•	match_date
	•	odds columns
	•	synth columns
	•	rolling press columns

Good result

A healthy merged file should show:
	•	fixture_key duplicate rows: 0
	•	exact duplicate rows: 0
	•	home-away-date duplicate rows: 0

Verified clean after patch

These were confirmed clean after the file-selection fix:
	•	Matches/__merged__/Champions_League__merged.csv
	•	Matches/__merged__/England_FA_Cup__merged.csv
	•	Matches/__merged__/Europa_Conference__merged.csv

Result:
	•	duplicate fixture_key: 0
	•	exact duplicate rows: 0
	•	duplicate home/away/date rows: 0

⸻

Why cup and European competitions needed special care

League competitions are usually simpler because each season file is straightforward and unique.

Cup and European competitions were more prone to merge contamination because:
	•	extra helper files existed in the folder
	•	duplicated raw exports with (1) / (2) existed
	•	train artifact CSVs were sitting beside raw season feeds
	•	broad fallback globbing accidentally pulled in non-season files

The fix was to make fallback input selection strict and filename-pattern based.

⸻

Accepted fallback filename pattern

Fallback season source files must match this logical shape:

<competition-slug>-matches-YYYY-to-YYYY-stats.csv

Also allowed:

<competition-slug>-matches-YYYY-to-YYYY-stats (1).csv
<competition-slug>-matches-YYYY-to-YYYY-stats (2).csv

Examples:
	•	england-fa-cup-matches-2025-to-2026-stats.csv
	•	europe-uefa-europa-conference-league-matches-2024-to-2025-stats (2).csv

⸻

Files that must never be used as fallback merge inputs

Examples:
	•	__TRAIN_MULTISEASON_completed.csv
	•	matches.csv
	•	matches.csv.bak
	•	fd_odds_enriched.csv
	•	fd_odds_enriched_synth.csv
	•	fd_ou25_novig.csv
	•	predictions_*.csv
	•	*report*.csv
	•	*upcoming*.csv
	•	*audit*.csv

⸻

Safe operating rule

For any future league rebuild:
	•	if fd_odds_enriched_synth.csv exists, prefer it
	•	else if fd_odds_enriched.csv exists, use it
	•	else fallback only to strict season *-matches-YYYY-to-YYYY-stats*.csv files
	•	never merge helper, training, report, audit, or backup files

⸻

Recommended post-rebuild spot checks

For any rebuilt league, immediately verify:

python3 - <<'PY'
import pandas as pd
p = "Matches/__merged__/England_EFL_League_1__merged.csv"
df = pd.read_csv(p, low_memory=False)
print("rows:", len(df))
print("fixture_key non-null:", int(df["fixture_key"].astype("string").fillna("").str.strip().ne("").sum()))
print("match_date non-null:", int(df["match_date"].astype("string").fillna("").str.strip().ne("").sum()))
PY

Then run duplicate audit checks.

⸻

Done definition

A merged rebuild is considered complete when:
	1.	merged CSV is written successfully
	2.	fixture_key coverage is complete or near-complete
	3.	match_date coverage is complete or near-complete
	4.	no exact duplicate rows remain
	5.	no duplicate fixture_key rows remain
	6.	no duplicate home/away/date rows remain
	7.	preferred enriched/synth source selection worked as expected
	8.	no junk files were pulled into the merge

⸻

Change log summary

Permanent fixes now in place
	•	strict fallback file filtering for season match CSVs only
	•	duplicate-suffix support for (1), (2), etc.
	•	automatic match_date repair from match_date -> date_GMT -> timestamp
	•	automatic fixture_key rebuild from normalized date + teams
	•	safer source/provenance handling for synth Under 2.5 odds
	•	cleaner merged QA process
	•	optional built-in `--dedupe-report` output for rebuild + audit in one pass

This should stop the earlier duplicate-merge problem from reappearing.

# MERGED_REBUILD_PIPELINE_README.md

## Purpose

This document is the source of truth for rebuilding clean merged match datasets in OG / BAWA after FootyStats raw files are refreshed.

It covers:

1. how `build_merged.py` selects valid source files
2. how `fd_odds_enriched.csv` and `fd_odds_enriched_synth.csv` are rebuilt
3. how Under 2.5 synth is generated and promoted into canonical columns
4. how duplicate contamination was fixed
5. how to QA merged outputs before training or deployment
6. how to freeze a clean merged baseline across the full league set

---

## Scope: current league set

The current full rebuild baseline covers these 20 league folders:

1. `Belgium Pro`
2. `Brazil Serie A`
3. `Champions League`
4. `Engl***and U21`
5. `England Championship`
6. `England EFL League 1`
7. `England FA Cup`
8. `England Premier League`
9. `Europa Conference`
10. `Europa League`
11. `France Ligue 1`
12. `Germany Bundesliga`
13. `Italy Serie A`
14. `Japan J1`
15. `Netherlands Eredivisie`
16. `Norway Eliteserien`
17. `Portugal Liga`
18. `Scotland Premiership`
19. `Spain La Liga`
20. `USA MLS`

---

## Problem this fixed

Historically, some merged league files became contaminated by bad fallback inputs such as:

- `__TRAIN_MULTISEASON_completed.csv`
- `matches.csv`
- `matches.csv.bak`
- other non-canonical helper / audit / artifact CSVs

This caused:

- duplicate fixtures
- duplicate `fixture_key`
- duplicate `home_team_name + away_team_name + date`
- polluted merged datasets for cup / European competitions
- unstable Under 2.5 coverage in merged outputs
- downstream training and audit instability

Examples that were fixed:

- Champions League
- England FA Cup
- Europa Conference

---

## Current merge rules

`build_merged.py` now works with this priority.

### Preferred inputs

If present, it uses exactly one of:

1. `fd_odds_enriched_synth.csv`
2. `fd_odds_enriched.csv`

These are preferred because they are already enriched / canonical.

### Fallback mode

If neither preferred file exists, it falls back to season match files only.

Accepted fallback filenames must match the season raw FootyStats pattern:

- `england-championship-matches-2020-to-2021-stats.csv`
- `england-fa-cup-matches-2023-to-2024-stats.csv`
- `europe-uefa-champions-league-matches-2025-to-2026-stats.csv`

Duplicate browser suffix variants are allowed:

- `...stats (1).csv`
- `...stats (2).csv`
- `...stats (3).csv`

### Explicitly rejected in fallback mode

Files like these must never be merged:

- `__TRAIN_MULTISEASON_completed.csv`
- `matches.csv`
- `matches.csv.bak`
- `fd_odds_enriched.csv` during fallback collection
- `fd_odds_enriched_synth.csv` during fallback collection
- `fd_ou25_novig.csv`
- prediction / report / upcoming / audit outputs
- helper / artifact CSVs

---

## What `build_merged.py` now guarantees

### 1. Stable datetime parsing

Merged files automatically backfill datetime using this order:

1. `match_date`
2. `date_GMT`
3. `timestamp`

### 2. Stable `match_date`

If `match_date` is blank, it is rebuilt from the parsed datetime in normalized date form.

### 3. Stable `fixture_key`

If `fixture_key` is blank, it is rebuilt from:

- normalized date
- normalized home team
- normalized away team

Format:

`YYYY_MM_DD_Home_Team_Away_Team`

Example:

`2025_08_02_AFC_Croydon_Athletic_Roffey`

### 4. Under 2.5 canonical promotion

If `odds_ft_under25_synth` exists and canonical `odds_ft_under25` is missing, it is promoted automatically.

`odds_source_under25` is also maintained properly:

- `synth_ou25`
- `orig`

### 5. Odds placeholder cleanup

Invalid odds placeholders are converted to `NaN`, including values like:

- `0.0`
- `1.0`
- `1.0001`

### 6. Optional built-in dedupe audit

`build_merged.py` can now write a duplicate audit in the same run via `--dedupe-report`.

---

## Under 2.5 synth: what it is for

A large number of leagues / rows have:

- `odds_ft_over25` present
- `odds_ft_under25` missing

Without Under 2.5, the merged files are incomplete for:

- OU25 no-vig calculation
- OU25 feature building
- OU25 model inputs
- downstream all-markets consistency

The synth pipeline solves this by:

1. fitting synthesis rulebooks from existing leagues with usable both-sided odds
2. applying those rulebooks to league-level `fd_odds_enriched.csv`
3. writing `fd_odds_enriched_synth.csv`
4. promoting `odds_ft_under25_synth` into canonical `odds_ft_under25`
5. ensuring merged files prefer the synth file automatically

---

## Core Under 2.5 synth files

### Inputs

Per league:

- `Matches/<LEAGUE>/fd_odds_enriched.csv`

Tables:

- `ModelStore/ODDS_SYNTH_TABLES.json`

### Synth output

Per league:

- `Matches/<LEAGUE>/fd_odds_enriched_synth.csv`

Important columns written by synth:

- `odds_ft_under25_synth`
- `under25_synth_conf`
- `under25_synth_reason`
- `under25_synth_group`
- `under25_synth_bin`
- `odds_source_under25`

After promotion / merge, canonical merged files should contain:

- `odds_ft_over25`
- `odds_ft_under25`
- `odds_ft_under25_synth`
- `under25_synth_conf`
- `under25_synth_reason`
- `odds_source_under25`

---

## Canonical rebuild workflow

## Step 1 — Refresh raw FootyStats files

Use the FootyStats ingest workflow to place the newest season files into the correct `Matches/`, `Teams/`, and `Players/` league folders.

## Step 2 — Build / refresh all-seasons enriched file if needed

For leagues where `fd_odds_enriched.csv` must be rebuilt from season files, concatenate only valid season match CSVs.

This is mainly relevant for leagues where enriched files do not yet exist or need rebuilding.

Typical examples where this was required:

- `Champions League`
- `Engl***and U21`
- `England FA Cup`
- `Europa Conference`
- `Europa League`
- `Japan J1`
- `Norway Eliteserien`

Example pattern:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

LEAGUES = [
  "Champions League",
  "Engl***and U21",
  "England FA Cup",
  "Europa Conference",
  "Europa League",
  "Japan J1",
  "Norway Eliteserien",
]

def is_season_file(p: Path) -> bool:
  n = p.name.lower()
  if not n.endswith('.csv'):
    return False
  if 'matches-' not in n:
    return False
  if 'stats' not in n:
    return False
  if '__train_' in n or 'train_multiseason' in n:
    return False
  if n in {'matches.csv', 'matches.csv.bak'}:
    return False
  if 'fd_odds_enriched' in n:
    return False
  if 'fd_ou25_novig' in n:
    return False
  if '_hold_' in n:
    return False
  return True

for lg in LEAGUES:
  d = Path('Matches') / lg
  season_files = sorted([p for p in d.glob('*.csv') if is_season_file(p)])
  frames = []
  for p in season_files:
    df = pd.read_csv(p, low_memory=False)
    df['__src_csv'] = p.name
    frames.append(df)
  out = pd.concat(frames, axis=0, ignore_index=True)
  out.to_csv(d / 'fd_odds_enriched.csv', index=False)
  print(f'[OK] {lg}: wrote {d / "fd_odds_enriched.csv"} | rows={len(out)} | cols={out.shape[1]} | inputs={len(season_files)}')
PY
```

## Step 3 — Fit / refresh synth tables

Run:

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

## Step 4 — Apply Under 2.5 synth where appropriate

Run per league:

```bash
python odds_synth.py apply \
  --src "Matches/<LEAGUE>/fd_odds_enriched.csv" \
  --tables "ModelStore/ODDS_SYNTH_TABLES.json" \
  --out "Matches/<LEAGUE>/fd_odds_enriched_synth.csv" \
  --group-by league \
  --conf-min 0.25 \
  --min-n 200
```

### Example batch for the seven previously missing Under 2.5 leagues

```bash
for L in \
  "Champions League" \
  "Engl***and U21" \
  "England FA Cup" \
  "Europa Conference" \
  "Europa League" \
  "Japan J1" \
  "Norway Eliteserien"
do
  python3 odds_synth.py apply \
    --src "Matches/$L/fd_odds_enriched.csv" \
    --tables "ModelStore/ODDS_SYNTH_TABLES.json" \
    --out "Matches/$L/fd_odds_enriched_synth.csv" \
    --group-by league \
    --conf-min 0.25 \
    --min-n 200
done
```

Expected style of output:

```text
[OK] wrote: Matches/Champions League/fd_odds_enriched_synth.csv
  under25_synth nonnull: 1768/1781
  bttsno_synth nonnull: 1761/1781
```

## Step 5 — Build merged files

Single league:

```bash
python build_merged.py --league "England EFL League 1" --recursive --rolling-press --write-report
```

Multiple leagues:

```bash
python build_merged.py --leagues \
  "England Championship" \
  "England EFL League 1" \
  "France Ligue 1" \
  "Spain La Liga" \
  --recursive --rolling-press --write-report
```

All leagues:

```bash
python build_merged.py --all --recursive --rolling-press --write-report
```

### Built-in dedupe report

You can now ask `build_merged.py` to rebuild and write a duplicate-audit report in the same run.

Single league:

```bash
python3 build_merged.py --league "Champions League" --recursive --write-report --dedupe-report
```

Batch example:

```bash
python3 build_merged.py --leagues "Champions League" "England FA Cup" "Europa Conference" --recursive --write-report --dedupe-report
```

Default dedupe output:

- `Matches/__merged__/__merged_dedupe_report__.csv`

Custom dedupe report path:

```bash
python3 build_merged.py --all --recursive --write-report --dedupe-report \
  --dedupe-report-path "ModelStore/logs/merged_dedupe_report.csv"
```

---

## Full baseline rebuild command

When you want to freeze a full clean baseline across the whole 20-league set, use:

```bash
python3 build_merged.py --all --recursive --rolling-press --write-report --dedupe-report \
  --report-path "ModelStore/logs/full_merged_baseline__merge_report.csv" \
  --dedupe-report-path "ModelStore/logs/full_merged_baseline__dedupe_report.csv"
```

This gives you:

- merged outputs in `Matches/__merged__/`
- merge coverage report in `ModelStore/logs/full_merged_baseline__merge_report.csv`
- dedupe audit in `ModelStore/logs/full_merged_baseline__dedupe_report.csv`

---

## Rebuild helper script

We also use:

```bash
./rebuild_all_merged.sh
```

This handles the multi-step process:

1. rebuild all-seasons `fd_odds_enriched.csv` for selected leagues
2. fit / refresh ODDS synth tables
3. apply synth where source enriched files exist
4. rebuild merged files
5. write QA logs

Typical outputs:

- `ModelStore/logs/rebuild_all_merged__synth_log.txt`
- `ModelStore/logs/rebuild_all_merged__merge_log.txt`
- `ModelStore/logs/rebuild_all_merged__qa.csv`
- `Matches/__merged__/__merged_dedupe_report__.csv` when `--dedupe-report` is used

---

## Under 2.5 synth QA checks

After synth apply, immediately check that the synth file actually contains Under 2.5 coverage.

Example:

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

LEAGUES = [
  "Champions League",
  "Engl***and U21",
  "England FA Cup",
  "Europa Conference",
  "Europa League",
  "Japan J1",
  "Norway Eliteserien",
]

rows = []
for lg in LEAGUES:
  p = Path("Matches") / lg / "fd_odds_enriched_synth.csv"
  if not p.exists():
    rows.append({"league": lg, "exists": False})
    continue
  df = pd.read_csv(p, low_memory=False)
  rows.append({
    "league": lg,
    "exists": True,
    "rows": len(df),
    "odds_ft_over25_nn": int(pd.to_numeric(df.get("odds_ft_over25"), errors="coerce").notna().sum()) if "odds_ft_over25" in df.columns else 0,
    "odds_ft_under25_nn": int(pd.to_numeric(df.get("odds_ft_under25"), errors="coerce").notna().sum()) if "odds_ft_under25" in df.columns else 0,
    "odds_ft_under25_synth_nn": int(pd.to_numeric(df.get("odds_ft_under25_synth"), errors="coerce").notna().sum()) if "odds_ft_under25_synth" in df.columns else 0,
    "source_under25_nn": int(df.get("odds_source_under25", pd.Series(dtype='string')).astype('string').fillna('').str.strip().ne('').sum()) if "odds_source_under25" in df.columns else 0,
  })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
PY
```

Good result:

- `odds_ft_under25_synth_nn` is materially above zero
- `odds_ft_under25_nn` is materially above zero
- `source_under25_nn` is materially above zero

---

## Merged-file QA checks

If you use `--dedupe-report`, the duplicate audit is written automatically during the rebuild run.

After rebuild, merged files should be audited for:

1. `fixture_key` duplicate rows
2. exact duplicate rows
3. duplicate `home_team_name + away_team_name + normalized date`
4. key coverage:
   - `fixture_key`
   - `match_date`
   - odds columns
   - synth columns
   - rolling press columns

### Good result

A healthy merged file should show:

- `fixture_key duplicate rows: 0`
- `exact duplicate rows: 0`
- `home-away-date duplicate rows: 0`

### Verified clean after patch

These were confirmed clean after the file-selection fix:

- `Matches/__merged__/Champions_League__merged.csv`
- `Matches/__merged__/England_FA_Cup__merged.csv`
- `Matches/__merged__/Europa_Conference__merged.csv`

Result:

- duplicate `fixture_key`: `0`
- exact duplicate rows: `0`
- duplicate home / away / date rows: `0`

---

## Why cup and European competitions needed special care

League competitions are usually simpler because each season file is straightforward and unique.

Cup and European competitions were more prone to merge contamination because:

- extra helper files existed in the folder
- duplicated raw exports with `(1)` / `(2)` existed
- train artifact CSVs were sitting beside raw season feeds
- broad fallback globbing accidentally pulled in non-season files

The fix was to make fallback input selection strict and filename-pattern based.

---

## Accepted fallback filename pattern

Fallback season source files must match this logical shape:

`<competition-slug>-matches-YYYY-to-YYYY-stats.csv`

Also allowed:

- `<competition-slug>-matches-YYYY-to-YYYY-stats (1).csv`
- `<competition-slug>-matches-YYYY-to-YYYY-stats (2).csv`
- `<competition-slug>-matches-YYYY-to-YYYY-stats (3).csv`

Examples:

- `england-fa-cup-matches-2025-to-2026-stats.csv`
- `europe-uefa-europa-conference-league-matches-2024-to-2025-stats (2).csv`

---

## Files that must never be used as fallback merge inputs

Examples:

- `__TRAIN_MULTISEASON_completed.csv`
- `matches.csv`
- `matches.csv.bak`
- `fd_odds_enriched.csv`
- `fd_odds_enriched_synth.csv`
- `fd_ou25_novig.csv`
- `predictions_*.csv`
- `*report*.csv`
- `*upcoming*.csv`
- `*audit*.csv`

---

## Safe operating rule

For any future league rebuild:

- if `fd_odds_enriched_synth.csv` exists, prefer it
- else if `fd_odds_enriched.csv` exists, use it
- else fallback only to strict season `*-matches-YYYY-to-YYYY-stats*.csv` files
- never merge helper, training, report, audit, or backup files

---

## Recommended post-rebuild spot checks

For any rebuilt league, immediately verify:

```bash
python3 - <<'PY'
import pandas as pd
p = "Matches/__merged__/England_EFL_League_1__merged.csv"
df = pd.read_csv(p, low_memory=False)
print("rows:", len(df))
print("fixture_key non-null:", int(df["fixture_key"].astype("string").fillna("").str.strip().ne("").sum()))
print("match_date non-null:", int(df["match_date"].astype("string").fillna("").str.strip().ne("").sum()))
print("odds_ft_under25 non-null:", int(pd.to_numeric(df["odds_ft_under25"], errors="coerce").notna().sum()) if "odds_ft_under25" in df.columns else 0)
print("odds_ft_under25_synth non-null:", int(pd.to_numeric(df["odds_ft_under25_synth"], errors="coerce").notna().sum()) if "odds_ft_under25_synth" in df.columns else 0)
PY
```

Then run duplicate audit checks.

---

## Done definition

A merged rebuild is considered complete when:

1. merged CSV is written successfully
2. `fixture_key` coverage is complete or near-complete
3. `match_date` coverage is complete or near-complete
4. no exact duplicate rows remain
5. no duplicate `fixture_key` rows remain
6. no duplicate home / away / date rows remain
7. preferred enriched / synth source selection worked as expected
8. canonical `odds_ft_under25` is populated where synth was available
9. `odds_ft_under25_synth` is present where synth was applied
10. no junk files were pulled into the merge

---

## Change log summary

Permanent fixes now in place:

- strict fallback file filtering for season match CSVs only
- duplicate-suffix support for `(1)`, `(2)`, `(3)`, etc.
- automatic `match_date` repair from `match_date -> date_GMT -> timestamp`
- automatic `fixture_key` rebuild from normalized date + teams
- safer source / provenance handling for synth Under 2.5 odds
- cleaner merged QA process
- optional built-in `--dedupe-report` output for rebuild + audit in one pass
- explicit Under 2.5 synth rebuild path for leagues missing canonical Under 2.5 odds

This should stop the earlier duplicate-merge problem from reappearing and keep OU25 coverage intact in future rebuilds.