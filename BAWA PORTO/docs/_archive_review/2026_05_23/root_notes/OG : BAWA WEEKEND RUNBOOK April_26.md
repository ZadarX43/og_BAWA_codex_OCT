# ============================================================
# OG / BAWA WEEKEND RUNBOOK
# clean → enrich → synth → merge → QA
# ============================================================

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" || exit 1

# ------------------------------------------------------------
# LEAGUES
# ------------------------------------------------------------
LEAGUES=(
  "England Premier League"
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
  "Europa Conference"
  "Europa League"
  "Champions League"
  "England EFL League 1"
  "England FA Cup"
  "England Championship"
)

# ------------------------------------------------------------
# 0) INGEST RAW SEASON FILES FIRST
# ------------------------------------------------------------
# Run your FootyStats ingest app / ingest workflow first.
# This is the real start of the chain:
# season files -> fd_odds_enriched.csv -> fd_odds_enriched_synth.csv -> __merged__.csv -> predictions
#
# If using your app, do that now before continuing.
# If using your script directly, run your normal footystats ingest step here.

echo
echo "============================================================"
echo "STEP 0 — CONFIRM RAW INGEST IS DONE"
echo "============================================================"
echo "Make sure latest season/raw files are already ingested before continuing."
echo

# ------------------------------------------------------------
# 1) CLEAN
# ------------------------------------------------------------
echo
echo "============================================================"
echo "STEP 1 — CLEAN STALE GENERATED FILES"
echo "============================================================"

for LEAGUE in "${LEAGUES[@]}"; do
  rm -f "Matches/$LEAGUE/fd_odds_enriched.csv"
  rm -f "Matches/$LEAGUE/fd_odds_enriched_synth.csv"
  rm -f "Matches/$LEAGUE/fd_ou25_novig.csv"
done

rm -f Matches/__merged__/England_Premier_League__merged.csv
rm -f Matches/__merged__/Japan_J1__merged.csv
rm -f Matches/__merged__/Norway_Eliteserien__merged.csv
rm -f Matches/__merged__/Netherlands_Eredivisie__merged.csv
rm -f Matches/__merged__/Belgium_Pro__merged.csv
rm -f Matches/__merged__/Scotland_Premiership__merged.csv
rm -f Matches/__merged__/Brazil_Serie_A__merged.csv
rm -f Matches/__merged__/USA_MLS__merged.csv
rm -f Matches/__merged__/Portugal_Liga__merged.csv
rm -f Matches/__merged__/Spain_La_Liga__merged.csv
rm -f Matches/__merged__/Italy_Serie_A__merged.csv
rm -f Matches/__merged__/France_Ligue_1__merged.csv
rm -f Matches/__merged__/Germany_Bundesliga__merged.csv
rm -f Matches/__merged__/Europa_Conference__merged.csv
rm -f Matches/__merged__/Europa_League__merged.csv
rm -f Matches/__merged__/Champions_League__merged.csv
rm -f Matches/__merged__/England_EFL_League_1__merged.csv
rm -f Matches/__merged__/England_FA_Cup__merged.csv
rm -f Matches/__merged__/England_Championship__merged.csv

echo "CLEAN DONE"

# ------------------------------------------------------------
# 2) ENRICH
# ------------------------------------------------------------
echo
echo "============================================================"
echo "STEP 2 — REBUILD FRESH fd_odds_enriched.csv"
echo "============================================================"

python merge_fd_odds_into_matches.py \
  --matches-root "Matches" \
  --fd-root "football_data.co.uk" \
  --overwrite \
  --leagues "England Premier League,Japan J1,Norway Eliteserien,Netherlands Eredivisie,Belgium Pro,Scotland Premiership,Brazil Serie A,USA MLS,Portugal Liga,Spain La Liga,Italy Serie A,France Ligue 1,Germany Bundesliga,Europa Conference,Europa League,Champions League,England EFL League 1,England FA Cup,England Championship"

# ------------------------------------------------------------
# 3) SYNTH
# ------------------------------------------------------------
echo
echo "============================================================"
echo "STEP 3 — REBUILD FRESH fd_odds_enriched_synth.csv"
echo "============================================================"

for LEAGUE in "${LEAGUES[@]}"; do
  echo
  echo "==================== $LEAGUE ===================="
  python odds_synth.py apply \
    --src "Matches/$LEAGUE/fd_odds_enriched.csv" \
    --tables "ModelStore/ODDS_SYNTH_TABLES.json" \
    --out "Matches/$LEAGUE/fd_odds_enriched_synth.csv" \
    --group-by league \
    --conf-min 0.25 \
    --min-n 200
done

# ------------------------------------------------------------
# 4) MERGE
# ------------------------------------------------------------
echo
echo "============================================================"
echo "STEP 4 — REBUILD FRESH __merged__.csv"
echo "============================================================"

python build_merged.py --leagues \
  "England Premier League" \
  "Japan J1" \
  "Norway Eliteserien" \
  "Netherlands Eredivisie" \
  "Belgium Pro" \
  "Scotland Premiership" \
  "Brazil Serie A" \
  "USA MLS" \
  "Portugal Liga" \
  "Spain La Liga" \
  "Italy Serie A" \
  "France Ligue 1" \
  "Germany Bundesliga" \
  "Europa Conference" \
  "Europa League" \
  "Champions League" \
  "England EFL League 1" \
  "England FA Cup" \
  "England Championship" \
  --recursive \
  --rolling-press \
  --write-report

# ------------------------------------------------------------
# 5) QA — MERGED FILE PRESENCE / BASIC ODDS COVERAGE
# ------------------------------------------------------------
echo
echo "============================================================"
echo "STEP 5A — QA BASIC MERGED COVERAGE"
echo "============================================================"

python3 - <<'PY'
from pathlib import Path
import pandas as pd

targets = [
    "England_Premier_League__merged.csv",
    "Japan_J1__merged.csv",
    "Norway_Eliteserien__merged.csv",
    "Netherlands_Eredivisie__merged.csv",
    "Belgium_Pro__merged.csv",
    "Scotland_Premiership__merged.csv",
    "Brazil_Serie_A__merged.csv",
    "USA_MLS__merged.csv",
    "Portugal_Liga__merged.csv",
    "Spain_La_Liga__merged.csv",
    "Italy_Serie_A__merged.csv",
    "France_Ligue_1__merged.csv",
    "Germany_Bundesliga__merged.csv",
    "Europa_Conference__merged.csv",
    "Europa_League__merged.csv",
    "Champions_League__merged.csv",
    "England_EFL_League_1__merged.csv",
    "England_FA_Cup__merged.csv",
    "England_Championship__merged.csv",
]

root = Path("Matches/__merged__")

for name in targets:
    p = root / name
    if not p.exists():
        print(name, "| MISSING FILE")
        continue

    df = pd.read_csv(p, low_memory=False)

    over = int(pd.to_numeric(df.get("odds_ft_over25"), errors="coerce").notna().sum()) if "odds_ft_over25" in df.columns else 0
    under = int(pd.to_numeric(df.get("odds_ft_under25"), errors="coerce").notna().sum()) if "odds_ft_under25" in df.columns else 0
    under_s = int(pd.to_numeric(df.get("odds_ft_under25_synth"), errors="coerce").notna().sum()) if "odds_ft_under25_synth" in df.columns else 0
    src = int(df["odds_source_under25"].astype(str).str.len().gt(0).sum()) if "odds_source_under25" in df.columns else 0

    print(name, "| over:", over, "| under:", under, "| under_synth:", under_s, "| source:", src)
PY

# ------------------------------------------------------------
# 6) QA — WEEKEND LEAGUES LIST
# ------------------------------------------------------------
echo
echo "============================================================"
echo "STEP 5B — QA LEAGUES PLAYING THIS WEEKEND"
echo "============================================================"

python3 - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("Matches/__merged__")
targets = sorted(root.glob("*__merged.csv"))

dates = {"2026-03-21", "2026-03-22", "2026-03-23"}
rows = []

for p in targets:
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        continue

    if "match_date" not in df.columns:
        continue

    md = pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sub = df.loc[md.isin(dates)].copy()
    if sub.empty:
        continue

    league = p.name.replace("__merged.csv", "").replace("_", " ")
    rows.append({
        "league": league,
        "fixtures_21_23": int(len(sub)),
        "latest_date_in_subset": str(md[md.isin(dates)].max()),
    })

out = pd.DataFrame(rows).sort_values(["fixtures_21_23", "league"], ascending=[False, True])
if out.empty:
    print("No leagues found with fixtures on 2026-03-21 to 2026-03-23")
else:
    print(out.to_string(index=False))
PY

# ------------------------------------------------------------
# 7) QA — WEEKEND FIXTURE COMPLETENESS
# ------------------------------------------------------------
echo
echo "============================================================"
echo "STEP 5C — QA WEEKEND FTR + OU25 COMPLETENESS"
echo "============================================================"

python3 - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("Matches/__merged__")
targets = sorted(root.glob("*__merged.csv"))
dates = {"2026-03-21", "2026-03-22", "2026-03-23"}

rows = []

for p in targets:
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        continue

    if "match_date" not in df.columns:
        continue

    md = pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sub = df.loc[md.isin(dates)].copy()
    if sub.empty:
        continue

    ftr_complete = (
        pd.to_numeric(sub.get("odds_ft_home_team_win"), errors="coerce").notna()
        & pd.to_numeric(sub.get("odds_ft_draw"), errors="coerce").notna()
        & pd.to_numeric(sub.get("odds_ft_away_team_win"), errors="coerce").notna()
    )

    ou25_complete = (
        pd.to_numeric(sub.get("odds_ft_over25"), errors="coerce").notna()
        & pd.to_numeric(sub.get("odds_ft_under25"), errors="coerce").notna()
    )

    rows.append({
        "league": p.name.replace("__merged.csv", "").replace("_", " "),
        "fixtures_21_23": int(len(sub)),
        "ftr_complete": int(ftr_complete.sum()),
        "ou25_complete": int(ou25_complete.sum()),
        "missing_ftr": int((~ftr_complete).sum()),
        "missing_ou25": int((~ou25_complete).sum()),
        "latest_date_in_subset": str(md[md.isin(dates)].max()),
    })

out = pd.DataFrame(rows).sort_values(["fixtures_21_23", "league"], ascending=[False, True])
if out.empty:
    print("No leagues found with fixtures on 2026-03-21 to 2026-03-23")
else:
    print(out.to_string(index=False))
PY

# ------------------------------------------------------------
# 8) QA — DETAILED WEEKEND FIXTURES FOR LEAGUES WITH GAMES
# ------------------------------------------------------------
echo
echo "============================================================"
echo "STEP 5D — QA DETAILED WEEKEND FIXTURES"
echo "============================================================"

python3 - <<'PY'
from pathlib import Path
import pandas as pd

targets = [
    "England_Premier_League__merged.csv",
    "Japan_J1__merged.csv",
    "Norway_Eliteserien__merged.csv",
    "Netherlands_Eredivisie__merged.csv",
    "Belgium_Pro__merged.csv",
    "Scotland_Premiership__merged.csv",
    "Brazil_Serie_A__merged.csv",
    "USA_MLS__merged.csv",
    "Portugal_Liga__merged.csv",
    "Spain_La_Liga__merged.csv",
    "Italy_Serie_A__merged.csv",
    "France_Ligue_1__merged.csv",
    "Germany_Bundesliga__merged.csv",
    "Europa_Conference__merged.csv",
    "Europa_League__merged.csv",
    "Champions_League__merged.csv",
    "England_EFL_League_1__merged.csv",
    "England_FA_Cup__merged.csv",
    "England_Championship__merged.csv",
]

dates = {"2026-03-21", "2026-03-22", "2026-03-23"}
root = Path("Matches/__merged__")

for name in targets:
    p = root / name
    if not p.exists():
        continue

    df = pd.read_csv(p, low_memory=False)
    if "match_date" not in df.columns:
        continue

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sub = df[df["match_date"].isin(dates)].copy()
    if sub.empty:
        continue

    sub["ftr_complete"] = (
        pd.to_numeric(sub.get("odds_ft_home_team_win"), errors="coerce").notna()
        & pd.to_numeric(sub.get("odds_ft_draw"), errors="coerce").notna()
        & pd.to_numeric(sub.get("odds_ft_away_team_win"), errors="coerce").notna()
    )
    sub["ou25_complete"] = (
        pd.to_numeric(sub.get("odds_ft_over25"), errors="coerce").notna()
        & pd.to_numeric(sub.get("odds_ft_under25"), errors="coerce").notna()
    )

    cols = [
        "match_date",
        "home_team_name",
        "away_team_name",
        "status",
        "odds_ft_home_team_win",
        "odds_ft_draw",
        "odds_ft_away_team_win",
        "odds_ft_over25",
        "odds_ft_under25",
        "odds_source_under25",
        "ftr_complete",
        "ou25_complete",
    ]
    cols = [c for c in cols if c in sub.columns]

    print()
    print("=" * 140)
    print("LEAGUE:", name.replace("__merged.csv", "").replace("_", " "))
    print("=" * 140)
    print(sub[cols].sort_values(["match_date", "home_team_name", "away_team_name"]).to_string(index=False))
    print()
    print(
        "SUMMARY: fixtures=", len(sub),
        "| ftr_complete=", int(sub["ftr_complete"].sum()),
        "| ou25_complete=", int(sub["ou25_complete"].sum())
    )
PY

echo
echo "============================================================"
echo "RUNBOOK COMPLETE"
echo "============================================================"
echo "Next stage after this: downstream prediction / deploy pipeline."