# OG Snapshot Weekend Runner & Dashboard

## Purpose

This toolchain gives you a fast way to:

1. rebuild team-snapshot matchup layers
2. generate the all-leagues weekend snapshot support board
3. open a desktop dashboard that highlights:
   - strong **home FTR** structural support
   - strong **Over 2.5** structural support
   - **away/draw danger** on weak home-side FTR rows
   - **OU2.5 oppose** rows

It is a support layer, not the final prediction engine.

The point is to give you a fast structural readout alongside the core model stack.

---

## Main files

### Core build / board scripts

- `build_team_snapshot_matchup_features.py`
- `build_team_snapshot_matchup_features_all.py`
- `weekend_snapshot_support_board.py`
- `weekend_snapshot_support_board_all_leagues.py`

### Dashboard

- `weekend_snapshot_board_dashboard.py`

### AppleScript launcher / Script Editor app

This is the macOS launcher you saved as your **OG Snapshot Weekend Runner** app.

It is responsible for:
- choosing mode
- choosing prediction window
- launching the terminal command
- optionally opening the dashboard

---

## Main outputs

### Matchup build board
- `predictions_output/TEAM_SNAPSHOT_MATCHUP_BUILD_BOARD.csv`

### Weekend board
- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv`

### Weekend build board
- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES__BUILD_BOARD.csv`

### Logs
- `ModelStore/logs/snapshot_weekend_app__YYYY-MM-DD_HH-MM-SS.log`

---

## High-level flow of data

## Stage 1: raw fixtures

Source:
- `Matches/<League>/*.csv`

These are the fixture-level match files.

---

## Stage 2: raw team snapshots

Source:
- `Teams/<League>/*.csv`

These are the team summary tables used to build structural matchup context.

---

## Stage 3: enriched matchup files

Built by:
- `build_team_snapshot_matchup_features.py`
- `build_team_snapshot_matchup_features_all.py`

Output pattern:
- `Matches/<League>/*__team_snapshot_matchups.csv`
- `Matches/<League>/*__team_snapshot_matchups__audit.csv`

These files join:
- home team snapshot data
- away team snapshot data
- derived `snap_...` matchup features

This is the bridge layer between raw team summaries and fixture-level support scoring.

---

## Stage 4: weekend board generation

Built by:
- `weekend_snapshot_support_board.py`
- `weekend_snapshot_support_board_all_leagues.py`

The all-leagues runner scans the matchup CSVs, runs the single-league board script for each, and combines them into:

- `WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv`
- `WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES__BUILD_BOARD.csv`

---

## Stage 5: dashboard

Read by:
- `weekend_snapshot_board_dashboard.py`

The dashboard reads:

- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv`

It does not rebuild anything itself.

It is a viewer / explainer layer on top of the already-generated weekend CSV.

---

## What the dashboard shows

The dashboard is currently built around two structural views:

### 1. FTR home-side support

Key columns:
- `snapshot_ftr_support_score`
- `snapshot_ftr_backed_side`
- `snapshot_ftr_support_band`
- `snapshot_ftr_direction_note`

Interpretation:
- `70+` → strong home FTR support
- `60–69.99` → decent home FTR support
- `40–59.99` → neutral / mixed
- `<40` → away/draw danger rises

Operational bands:
- `strong_home`
- `home`
- `neutral`
- `away_or_draw_danger`

Row colours:
- `strong_home` → green
- `home` → amber
- `neutral` → grey
- `away_or_draw_danger` → red

---

### 2. Over 2.5 structural support

Key columns:
- `snapshot_ou25_support_score`
- `snapshot_ou25_support_bucket`

Interpretation:
- `70+` → strong OU2.5 support
- `60–69.99` → decent OU2.5 support
- `40–59.99` → neutral / mixed
- `<40` or bucket `oppose` → OU2.5 oppose / resistance

Dashboard filters now support:
- Strong OU2.5 support (70+)
- OU2.5 support (60–69.99)
- OU2.5 oppose

Note:
the current row colouring is still driven by the **FTR band** tag, not a separate OU-only colour system.

---

## Dashboard filter options

Current top-right filter options:

- `All`
- `Strong home FTR support (70+)`
- `Home FTR support (60-69.99)`
- `FTR away/draw danger`
- `Strong OU2.5 support (70+)`
- `OU2.5 support (60-69.99)`
- `OU2.5 oppose`

---

## Summary bar counts

The dashboard summary line now shows:

- total rows shown
- strong home FTR count
- FTR away/draw danger count
- strong OU2.5 count
- OU2.5 oppose count

This gives a dual-market snapshot in one line.

---

## AppleScript app workflow

## Run Weekend Board mode

This mode:
1. asks for date-from
2. asks for date-to
3. launches `weekend_snapshot_support_board_all_leagues.py` in Terminal
4. writes a log file
5. lets you open the dashboard afterwards

Use this when matchup layers are already built and you just want a fresh weekend board.

---

## Rebuild Matchups + Run Weekend Board mode

This mode:
1. rebuilds matchup layers first
2. then runs the all-leagues weekend board
3. writes everything to the same log
4. lets you open the dashboard afterwards

Use this when you want the structural layer refreshed before generating the board.

---

## Expected successful terminal ending

A successful all-leagues weekend run should end with something like:

```text
=== ALL-LEAGUES WEEKEND SNAPSHOT SUMMARY ===
TOTAL_INPUT_FILES: 144
STATUS_OK: 144
STATUS_FAILED: 0
COMBINED_ROWS_OUT: 129
OUTPUT_CSV_WRITTEN: .../predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv
BOARD_CSV_WRITTEN: .../predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES__BUILD_BOARD.csv
```

If you see that, the run itself worked.

---

## Manual commands

## Rebuild matchup layers only

```bash
cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"

python3 build_team_snapshot_matchup_features_all.py \
  --matches-root "Matches" \
  --teams-root "Teams" \
  --single-league-script "build_team_snapshot_matchup_features.py"
```

---

## Run weekend board only

```bash
cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"

python3 weekend_snapshot_support_board_all_leagues.py \
  --matches-root "Matches" \
  --single-league-script "weekend_snapshot_support_board.py" \
  --date-from "2026-03-13" \
  --date-to "2026-03-16"
```

---

## Open dashboard manually

```bash
cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"

python3 weekend_snapshot_board_dashboard.py \
  --input-csv "predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv"
```

If your venv python is needed:

```bash
cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"

./.venv/bin/python weekend_snapshot_board_dashboard.py \
  --input-csv "predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv"
```

---

## Files the dashboard depends on

The dashboard depends on:

### Required script
- `weekend_snapshot_board_dashboard.py`

### Required CSV
- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv`

### Important expected columns
- `league`
- `home_team_name`
- `away_team_name`
- `__match_time__`
- `snapshot_ftr_support_score`
- `snapshot_ftr_backed_side`
- `snapshot_ftr_support_band`
- `snapshot_ftr_direction_note`
- `snapshot_ou25_support_score`
- `snapshot_ou25_support_bucket`

The dashboard will create some blanks if columns are missing, but it works best when all of the above are present.

---

## Explainability logic now embedded in the dashboard

### FTR logic

The dashboard infers and displays:

- backed side
- support band
- direction note

if they are missing or incomplete.

Rules:
- `score >= 70` → backed side = home team, band = `strong_home`
- `60 <= score < 70` → backed side = home team, band = `home`
- `40 <= score < 60` → band = `neutral`
- `score < 40` → backed side = `away_or_draw_danger`, band = `away_or_draw_danger`

Direction note examples:
- `Strong home FTR support for Inter Milan`
- `Decent home FTR support for Grêmio`
- `Home side weakly supported; away/draw danger rises vs St. Pauli`

---

## Common troubleshooting

## 1. App says script was not found

Cause:
the AppleScript path does not match the actual file name.

Check:
- `weekend_snapshot_support_board_all_leagues.py`
- `weekend_snapshot_support_board.py`
- `build_team_snapshot_matchup_features_all.py`
- `build_team_snapshot_matchup_features.py`
- `weekend_snapshot_board_dashboard.py`

A common mistake was mixing up:
- `weekend_snapshot_support_board_dashboard.py`
- `weekend_snapshot_board_dashboard.py`

The correct dashboard file you ended up using is:

- `weekend_snapshot_board_dashboard.py`

---

## 2. Error: path breaks at “BAWA”

Cause:
shell commands were being built in a way that broke on the space in:

- `BAWA PORTO`

Fix:
use fully quoted paths in AppleScript shell commands.

You already moved to quoted form, which solved this.

---

## 3. App appears blank after pressing Run

Cause:
the original app was waiting on a long-running shell command without clear UI behaviour.

Fix:
launching through Terminal is better because:
- you see live progress
- you see where it hangs if anything fails
- long rebuilds do not make the app feel frozen

---

## 4. Terminal run succeeds but dashboard does not open

Cause:
the dashboard launch command may fail silently, or the app may try to open it before the CSV is ready.

Best check:
run it manually in Terminal:

```bash
cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"
./.venv/bin/python weekend_snapshot_board_dashboard.py \
  --input-csv "predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv"
```

If that works, the issue is the launcher, not the dashboard script.

---

## 5. Dashboard crashes with `'float' object has no attribute 'strip'`

Cause:
some CSV cells are NaN floats, not strings.

Fix already applied:
the helper functions now coerce values safely before calling `.strip()`.

Key patched functions:
- `infer_band()`
- `infer_backed_side()`
- `infer_direction_note()`

---

## 6. Dashboard shows syntax error near the filter list

Cause:
missing comma after the filter `values=[...]` list in the `ttk.Combobox(...)` block.

The correct block must look like:

```python
self.filter_box = ttk.Combobox(
    top,
    textvariable=self.filter_var,
    state="readonly",
    values=[
        "All",
        "Strong home FTR support (70+)",
        "Home FTR support (60-69.99)",
        "FTR away/draw danger",
        "Strong OU2.5 support (70+)",
        "OU2.5 support (60-69.99)",
        "OU2.5 oppose",
    ],
    width=28,
)
```

Note the comma after the closing `]`.

---

## 7. `weekend_snapshot_support_board_all_leagues.py` suddenly behaves like the dashboard

Cause:
at one point the dashboard code was accidentally saved over the all-leagues runner file.

Check quickly:

```bash
head -n 20 weekend_snapshot_support_board_all_leagues.py
head -n 20 weekend_snapshot_board_dashboard.py
```

The all-leagues runner should begin with imports like:
- `csv`
- `sys`
- typing helpers

The dashboard should begin with:
- `tkinter`
- `ttk`
- `ScrolledText`

If both files look the same, one got overwritten and needs restoring.

---

## 8. Error: unrecognized arguments on weekend board run

Cause:
the dashboard script was accidentally being called instead of the all-leagues runner.

Symptom:
a command with `--matches-root --single-league-script --date-from --date-to` fails, because the dashboard only understands `--input-csv`.

Fix:
make sure the run step points to:

- `weekend_snapshot_support_board_all_leagues.py`

not the dashboard file.

---

## 9. “Maximum of 3 buttons allowed”

Cause:
AppleScript `display dialog` only allows up to 3 buttons.

Fix:
keep result dialogs to 3 buttons max.

---

## 10. Rebuild finishes but weekend board step fails

Check the end of the log.

If you see the rebuild summary but the weekend board does not start correctly, verify:
- `weekend_snapshot_support_board_all_leagues.py` is the correct runner file
- AppleScript is calling the correct script
- date arguments are valid `YYYY-MM-DD`

---

## Quick sanity checks

## Check that the weekend CSV exists

```bash
ls -l "predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv"
```

---

## Check that the explainability columns exist

```bash
python3 - <<'PY'
import pandas as pd
p = "predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv"
df = pd.read_csv(p, low_memory=False)
want = [
    "snapshot_ftr_support_score",
    "snapshot_ftr_backed_side",
    "snapshot_ftr_support_band",
    "snapshot_ftr_direction_note",
    "snapshot_ou25_support_score",
    "snapshot_ou25_support_bucket",
]
for c in want:
    print(c, c in df.columns)
PY
```

---

## Check top strong home FTR rows

```bash
python3 - <<'PY'
import pandas as pd
p = "predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv"
df = pd.read_csv(p, low_memory=False)
df["snapshot_ftr_support_score"] = pd.to_numeric(df["snapshot_ftr_support_score"], errors="coerce")
out = df[df["snapshot_ftr_support_score"] >= 70]
cols = [
    "league",
    "home_team_name",
    "away_team_name",
    "snapshot_ftr_support_score",
    "snapshot_ftr_backed_side",
    "snapshot_ftr_support_band",
]
print(out[cols].sort_values("snapshot_ftr_support_score", ascending=False).head(25).to_string(index=False))
PY
```

---

## Recommended operating flow

### Fastest normal flow
1. open the app
2. choose `Run Weekend Board`
3. choose date window
4. let Terminal complete
5. open dashboard
6. use filter box for:
   - strong home FTR
   - away/draw danger
   - strong OU2.5
   - OU2.5 oppose

### Full refresh flow
1. open the app
2. choose `Rebuild Matchups + Run Weekend Board`
3. choose date window
4. let rebuild finish
5. let weekend board finish
6. open dashboard
7. inspect build boards if anything looks off

---

## Current limitations

- row colours are still tied to the FTR band, not a separate OU row-colour system
- the lower detail pane currently focuses on the FTR direction note only
- the dashboard is a support view, not the final merged model-vs-snapshot comparison layer
- `core_ftr_pick` / `core_ou25_pick` still need live integration if you want conflict logic inside the dashboard later

---

## Next sensible upgrades

### 1. Dual detail pane
Add:
- FTR direction note
- OU2.5 direction note / OU support explanation

### 2. OU-specific row tags
Let OU filters recolour rows by OU band instead of still using FTR row tags.

### 3. Combined conflict mode
Once live model picks are merged into the weekend board:
- model agrees with snapshot
- model neutral vs snapshot
- model conflict with snapshot

### 4. Quick export buttons
Add dashboard buttons for:
- export filtered view
- open build board
- open matchup build board

---

## Bottom line

You now have a working macOS flow that can:

- rebuild matchup layers
- generate the all-leagues weekend support board
- open a dashboard with FTR explainability
- filter strong home FTR rows
- filter away/draw danger rows
- filter strong OU2.5 rows
- filter OU2.5 oppose rows

The most important live file in the viewing layer is:

- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv`

and the most important viewer is:

- `weekend_snapshot_board_dashboard.py`
