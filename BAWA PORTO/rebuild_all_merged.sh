#!/bin/bash
set -euo pipefail

ROOT="/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"
cd "$ROOT"

LEAGUES=(
  "England Premier League"
  "England Championship"
  "England EFL League 1"
  "England FA Cup"
  "Japan J1"
  "Norway Eliteserien"
  "Netherlands Eredivisie"
  "Belgium Pro"
  "Scotland Premiership"
  "Brazil Serie A"
  "USA MLS"
  "Portugal Liga"
  "Spain La Liga"
  "Italy Serie A"
  "France Ligue 1"
  "Germany Bundesliga"
  "Australia A-League"
  "Austria Bundesliga"
  "Czech First League"
  "Denmark Superliga"
  "Germany Bundesliga 2"
  "Saudi Pro League"
  "South Korea K League"
  "Sweden Allsvenskan"
  "Swiss Super League"
  "Turkey Super Lig"
  "Europa Conference"
  "Europa League"
  "Champions League"
)
REBUILD_ENRICHED_LEAGUES=(
  "England Premier League"
  "England Championship"
  "England EFL League 1"
  "France Ligue 1"
  "Spain La Liga"
  "Germany Bundesliga"
  "Italy Serie A"
  "Netherlands Eredivisie"
  "Belgium Pro"
  "Scotland Premiership"
  "Brazil Serie A"
  "Portugal Liga"
  "USA MLS"
  "Australia A-League"
  "Austria Bundesliga"
  "Czech First League"
  "Denmark Superliga"
  "Germany Bundesliga 2"
  "Saudi Pro League"
  "South Korea K League"
  "Sweden Allsvenskan"
  "Swiss Super League"
  "Turkey Super Lig"
  "Champions League"
  "England FA Cup"
  "Europa Conference"
  "Europa League"
  "Japan J1"
  "Norway Eliteserien"
)

MERGE_REPORT_OUT="ModelStore/logs/full_merged_baseline__merge_report.csv"
DEDUPE_REPORT_OUT="ModelStore/logs/full_merged_baseline__dedupe_report.csv"

echo "============================================================"
echo "STEP 0: Rebuild all-seasons fd_odds_enriched.csv for selected leagues"
echo "============================================================"

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

$PY - <<'PY'
from pathlib import Path
import pandas as pd

LEAGUES = [
  "England Championship",
  "England EFL League 1",
  "France Ligue 1",
  "Spain La Liga",
  "Germany Bundesliga",
  "Italy Serie A",
  "Netherlands Eredivisie",
  "Portugal Liga",
  "USA MLS",
  "Australia A-League",
  "Austria Bundesliga",
  "Czech First League",
  "Denmark Superliga",
  "Germany Bundesliga 2",
  "Saudi Pro League",
  "South Korea K League",
  "Sweden Allsvenskan",
  "Swiss Super League",
  "Turkey Super Lig",
  "Champions League",
  "England FA Cup",
  "Europa Conference",
  "Europa League",
  "Japan J1",
  "Norway Eliteserien",
]

def is_season_file(p: Path) -> bool:
  n = p.name.lower()
  if not n.endswith(".csv"):
    return False
  if "matches-" not in n:
    return False
  if "stats" not in n:
    return False
  if "__train_" in n or "train_multiseason" in n:
    return False
  if "fd_ou25_novig" in n:
    return False
  if "fd_odds_enriched" in n:
    return False
  if "_hold_" in n:
    return False
  return True

for lg in LEAGUES:
  d = Path("Matches") / lg
  if not d.exists():
    print(f"[MISS] league folder not found: {d}")
    continue

  season_files = sorted([p for p in d.glob("*.csv") if is_season_file(p)])
  if not season_files:
    print(f"[MISS] no season files found for {lg} in {d}")
    continue

  frames = []
  for p in season_files:
    df = pd.read_csv(p, low_memory=False)
    df["__src_csv"] = p.name
    frames.append(df)

  out = pd.concat(frames, axis=0, ignore_index=True)
  out_path = d / "fd_odds_enriched.csv"
  out.to_csv(out_path, index=False)

  print(f"[OK] {lg}: wrote {out_path} | rows={len(out)} | cols={out.shape[1]} | inputs={len(season_files)}")
  need = [
      "odds_ft_over25",
      "odds_btts_yes",
      "odds_btts_no",
      "odds_ft_home_team_win",
      "odds_ft_draw",
      "odds_ft_away_team_win",
  ]
  present = [c for c in need if c in out.columns]
  print(f"     odds-cols-present: {present}")
PY

echo
echo "============================================================"
echo "STEP 1: Fit / refresh ODDS synth tables"
echo "============================================================"

$PY odds_synth.py fit \
  --src "Matches/*/fd_odds_enriched.csv" \
  --out ModelStore/ODDS_SYNTH_TABLES.json \
  --group-by league \
  --markets ou25 btts \
  --known-side over yes \
  --bins 10 \
  --min-n 200

echo
echo "============================================================"
echo "STEP 2: Apply synth per league where fd_odds_enriched.csv exists"
echo "============================================================"

mkdir -p ModelStore/logs
SYNTH_LOG="ModelStore/logs/rebuild_all_merged__synth_log.txt"
MERGE_LOG="ModelStore/logs/rebuild_all_merged__merge_log.txt"
QA_OUT="ModelStore/logs/rebuild_all_merged__qa.csv"
: > "$SYNTH_LOG"
: > "$MERGE_LOG"
rm -f "$MERGE_REPORT_OUT" "$DEDUPE_REPORT_OUT"

for LEAGUE in "${REBUILD_ENRICHED_LEAGUES[@]}"; do
  SRC="Matches/$LEAGUE/fd_odds_enriched.csv"
  OUT="Matches/$LEAGUE/fd_odds_enriched_synth.csv"

  echo
  echo "---- $LEAGUE ----"

  if [ -f "$SRC" ]; then
    echo "Applying synth: $SRC"
    {
      echo "[$LEAGUE] APPLY START"
      $PY odds_synth.py apply \
        --src "$SRC" \
        --tables "ModelStore/ODDS_SYNTH_TABLES.json" \
        --out "$OUT" \
        --group-by league \
        --conf-min 0.25 \
        --min-n 200
      echo "[$LEAGUE] APPLY OK"
    } >> "$SYNTH_LOG" 2>&1
  else
    echo "SKIP synth: missing $SRC"
    echo "[$LEAGUE] SKIP synth: missing $SRC" >> "$SYNTH_LOG"
  fi
done

echo
echo "============================================================"
echo "STEP 3: Rebuild merged files for all ${#LEAGUES[@]} leagues"
echo "============================================================"

{
  echo "[ALL] MERGE START"
  $PY build_merged.py \
    --all \
    --recursive \
    --rolling-press \
    --write-report \
    --dedupe-report \
    --report-path "$MERGE_REPORT_OUT" \
    --dedupe-report-path "$DEDUPE_REPORT_OUT"
  echo "[ALL] MERGE OK"
} >> "$MERGE_LOG" 2>&1

echo "Merged all leagues with stable report outputs:"
echo "  Merge report : $MERGE_REPORT_OUT"
echo "  Dedupe report: $DEDUPE_REPORT_OUT"

echo
echo "============================================================"
echo "STEP 4: QA merged outputs"
echo "============================================================"

$PY - <<'PY'
import os
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
MERGED = ROOT / "Matches" / "__merged__"
OUT = ROOT / "ModelStore" / "logs" / "rebuild_all_merged__qa.csv"

leagues = [
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
  "Australia A-League",
  "Austria Bundesliga",
  "Czech First League",
  "Denmark Superliga",
  "Germany Bundesliga 2",
  "Saudi Pro League",
  "South Korea K League",
  "Sweden Allsvenskan",
  "Swiss Super League",
  "Turkey Super Lig",
  "Europa Conference",
  "Europa League",
  "Champions League",
]

required_cols = [
    "fixture_key",
    "home_team_name",
    "away_team_name",
    "status",
    "odds_ft_over25",
    "odds_ft_under25",
    "odds_source_under25",
    "odds_btts_yes",
    "odds_btts_no",
    "odds_ft_home_team_win",
    "odds_ft_draw",
    "odds_ft_away_team_win",
    "odds_ft_under25_synth",
    "under25_synth_conf",
    "under25_synth_reason",
    "rolling5_home_press_intensity",
    "rolling5_away_press_intensity",
    "rolling5_press_intensity_diff",
]

NUMERIC_PREFIXES = ("odds_ft_", "odds_btts_", "odds_over", "odds_under")
STRING_COLS = {"odds_source_under25", "od_source", "fixture_key", "home_team_name", "away_team_name", "status"}

rows = []

for league in leagues:
    merged_path = MERGED / f"{league.replace(' ', '_')}__merged.csv"
    row = {
        "league": league,
        "merged_path": str(merged_path),
        "exists": merged_path.exists(),
        "rows": 0,
        "cols": 0,
    }

    if merged_path.exists():
        try:
            df = pd.read_csv(merged_path, low_memory=False)
            row["rows"] = len(df)
            row["cols"] = len(df.columns)
            for c in required_cols:
                if c in df.columns:
                    row[f"{c}__present"] = True
                    if (c in STRING_COLS) or (not c.startswith(NUMERIC_PREFIXES)):
                        row[f"{c}__nonnull"] = int(df[c].astype("string").fillna("").str.strip().ne("").sum())
                    else:
                        row[f"{c}__nonnull"] = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
                else:
                    row[f"{c}__present"] = False
                    row[f"{c}__nonnull"] = 0
        except Exception as e:
            row["read_error"] = str(e)
    rows.append(row)

out = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print(f"[OK] wrote QA report: {OUT}")
print(out[['league','exists','rows','cols']].to_string(index=False))

synth_cols = [
    "league",
    "odds_ft_over25__nonnull",
    "odds_ft_under25__nonnull",
    "odds_ft_under25_synth__nonnull",
    "odds_source_under25__nonnull",
]
avail = [c for c in synth_cols if c in out.columns]
if avail:
    print()
    print(out[avail].to_string(index=False))
PY

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Synth log     : $SYNTH_LOG"
echo "Merge log     : $MERGE_LOG"
echo "QA report     : $QA_OUT"
echo "Merge report  : $MERGE_REPORT_OUT"
echo "Dedupe report : $DEDUPE_REPORT_OUT"