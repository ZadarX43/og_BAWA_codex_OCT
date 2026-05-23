

NEW_LEAGUE_MODEL_ONBOARDING_README.md

# New League Model Onboarding README

## Purpose

This document explains the step-by-step process used to onboard new leagues into the OG / BAWA model pipeline.

It covers:

- folder setup
- historical data backfill
- enriched file creation
- synth creation
- merged file creation
- main market model training
- goal lambda / home-away fold5 training
- artifact verification
- parity checks against existing leagues

This is the operational checklist for bringing a new league up to the same standard as the existing model portfolio.

---

## League onboarding target

A league is only considered fully onboarded when it has:

- raw historical files in:
  - `Matches/<League>`
  - `Players/<League>`
  - `Teams/<League>`
- `fd_odds_enriched.csv`
- `fd_odds_enriched_synth.csv`
- merged output in:
  - `Matches/__merged__/<LeagueTag>__merged.csv`
- main market models in:
  - `ModelStore/<LeagueTag>/`
- thresholds
- features manifest
- calibrator metadata
- goal lambda fold5 artifacts:
  - `home_goals_fold5.pkl`
  - `away_goals_fold5.pkl`
  - `lambda_models_manifest.json`

---

## New leagues onboarded in this run

The following leagues were onboarded successfully:

- Australia A-League
- Austria Bundesliga
- Denmark Superliga
- Swiss Super League
- Germany Bundesliga 2
- Czech First League
- South Korea K League
- Saudi Pro League
- Turkey Super Lig

Sweden Allsvenskan was intentionally excluded for now because current-season ingest was not active.

---

## Folder naming convention

### Human-readable folder names
Used in:

- `Matches/`
- `Players/`
- `Teams/`

Examples:

- `Australia A-League`
- `Germany Bundesliga 2`
- `South Korea K League`

### ModelStore tags
Used in:

- `ModelStore/<LeagueTag>/`

Examples:

- `Australia_A_League`
- `Germany_Bundesliga_2`
- `South_Korea_K_League`

### Important note
Some leagues require explicit tag overrides.

Example:

- folder / league name: `Australia A-League`
- merged filename tag: `Australia_A-League`
- ModelStore tag: `Australia_A_League`

This means path resolvers must be handled carefully.

---

## Step 1 — Create league folders

Create folders under:

- `Matches/<League>`
- `Players/<League>`
- `Teams/<League>`

Example command:

```bash
cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && mkdir -p \
"Matches/Australia A-League" "Teams/Australia A-League" "Players/Australia A-League" \
"Matches/Austria Bundesliga" "Teams/Austria Bundesliga" "Players/Austria Bundesliga" \
"Matches/Denmark Superliga" "Teams/Denmark Superliga" "Players/Denmark Superliga" \
"Matches/Swiss Super League" "Teams/Swiss Super League" "Players/Swiss Super League" \
"Matches/Germany Bundesliga 2" "Teams/Germany Bundesliga 2" "Players/Germany Bundesliga 2" \
"Matches/Czech First League" "Teams/Czech First League" "Players/Czech First League" \
"Matches/South Korea K League" "Teams/South Korea K League" "Players/South Korea K League" \
"Matches/Saudi Pro League" "Teams/Saudi Pro League" "Players/Saudi Pro League" \
"Matches/Turkey Super Lig" "Teams/Turkey Super Lig" "Players/Turkey Super Lig"


⸻

Step 2 — Create ModelStore folders

Create canonical ModelStore folders:

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && mkdir -p \
"ModelStore/Australia_A_League" \
"ModelStore/Austria_Bundesliga" \
"ModelStore/Czech_First_League" \
"ModelStore/Denmark_Superliga" \
"ModelStore/Germany_Bundesliga_2" \
"ModelStore/Saudi_Pro_League" \
"ModelStore/South_Korea_K_League" \
"ModelStore/Swiss_Super_League" \
"ModelStore/Turkey_Super_Lig"


⸻

Step 3 — Backfill historical FootyStats files manually

For first-time onboarding, the strict ingest app is not appropriate because it only accepts the latest season/year.

Instead, historical files should be copied manually into the correct league folders.

Manual historical backfill command

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
python3 - <<'PY'
from pathlib import Path
import shutil

drop = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP")

league_map = {
    "australia-a-league": "Australia A-League",
    "austria-bundesliga": "Austria Bundesliga",
    "denmark-superliga": "Denmark Superliga",
    "switzerland-super-league": "Swiss Super League",
    "germany-2-bundesliga": "Germany Bundesliga 2",
    "czech-republic-first-league": "Czech First League",
    "south-korea-k-league-1": "South Korea K League",
    "saudi-arabia-professional-league": "Saudi Pro League",
    "saudi-arabia-pro-league": "Saudi Pro League",
    "turkey-super-lig": "Turkey Super Lig",
}

type_map = {
    "matches": "Matches",
    "players": "Players",
    "teams": "Teams",
}

copied = 0
skipped = 0
unmatched = 0

for p in sorted(drop.glob("*.csv")):
    name = p.name
    matched = False

    for slug, league in league_map.items():
        for kind, root in type_map.items():
            prefix = f"{slug}-{kind}-"
            if name.startswith(prefix):
                dest_dir = Path(root) / league
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / name

                if dest.exists():
                    print(f"SKIP exists: {dest}")
                    skipped += 1
                else:
                    shutil.copy2(p, dest)
                    print(f"COPIED: {p.name} -> {dest}")
                    copied += 1

                matched = True
                break
        if matched:
            break

    if not matched:
        print(f"UNMATCHED: {p.name}")
        unmatched += 1

print(f"\nDone. copied={copied} skipped_existing={skipped} unmatched={unmatched}")
PY


⸻

Step 4 — Verify raw file counts

Check that the historical files landed correctly:

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
python3 - <<'PY'
from pathlib import Path

targets = {
    "Australia A-League": ["Matches", "Players", "Teams"],
    "Austria Bundesliga": ["Matches", "Players", "Teams"],
    "Denmark Superliga": ["Matches", "Players", "Teams"],
    "Swiss Super League": ["Matches", "Players", "Teams"],
    "Germany Bundesliga 2": ["Matches", "Players", "Teams"],
    "Czech First League": ["Matches", "Players", "Teams"],
    "South Korea K League": ["Matches", "Players", "Teams"],
    "Saudi Pro League": ["Matches", "Players", "Teams"],
    "Turkey Super Lig": ["Matches", "Players", "Teams"],
}

for league, roots in targets.items():
    print(f"\n=== {league} ===")
    for root in roots:
        d = Path(root) / league
        n = len(list(d.glob("*.csv"))) if d.exists() else 0
        print(f"{root:8} {n}")
PY


⸻

Step 5 — Build fd_odds_enriched.csv

Script used:
	•	inline Python concat workflow
	•	source files read from Matches/<League>/
	•	output written to Matches/<League>/fd_odds_enriched.csv

Command used:

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
python3 - <<'PY'
from pathlib import Path
import pandas as pd

LEAGUES = [
    "Australia A-League",
    "Austria Bundesliga",
    "Denmark Superliga",
    "Swiss Super League",
    "Germany Bundesliga 2",
    "Czech First League",
    "South Korea K League",
    "Saudi Pro League",
    "Turkey Super Lig",
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


⸻

Step 6 — Fit or refresh ODDS synth tables

Script used:
	•	odds_synth.py

Command:

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
python3 odds_synth.py fit \
  --src "Matches/*/fd_odds_enriched.csv" \
  --out ModelStore/ODDS_SYNTH_TABLES.json \
  --group-by league \
  --markets ou25 btts \
  --known-side over yes \
  --bins 10 \
  --min-n 200


⸻

Step 7 — Apply synth to the new leagues

Script used:
	•	odds_synth.py

Command:

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
for LEAGUE in \
  "Australia A-League" \
  "Austria Bundesliga" \
  "Denmark Superliga" \
  "Swiss Super League" \
  "Germany Bundesliga 2" \
  "Czech First League" \
  "South Korea K League" \
  "Saudi Pro League" \
  "Turkey Super Lig"
do
  echo ""
  echo "==== $LEAGUE ===="
  python3 odds_synth.py apply \
    --src "Matches/$LEAGUE/fd_odds_enriched.csv" \
    --tables "ModelStore/ODDS_SYNTH_TABLES.json" \
    --out "Matches/$LEAGUE/fd_odds_enriched_synth.csv" \
    --group-by league \
    --conf-min 0.25 \
    --min-n 200
done


⸻

Step 8 — Build merged files for the new leagues only

Script used:
	•	build_merged.py

Command:

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
python3 build_merged.py \
  --leagues \
  "Australia A-League" \
  "Austria Bundesliga" \
  "Denmark Superliga" \
  "Swiss Super League" \
  "Germany Bundesliga 2" \
  "Czech First League" \
  "South Korea K League" \
  "Saudi Pro League" \
  "Turkey Super Lig" \
  --recursive \
  --rolling-press

Expected output:
	•	Matches/__merged__/<LeagueTag>__merged.csv

Example:
	•	Australia_A-League__merged.csv
	•	Germany_Bundesliga_2__merged.csv

⸻

Step 9 — Train main market models

Script used:
	•	train_investor_leagues_v2.py

Markets trained:
	•	BTTS
	•	Over 2.5
	•	Under 2.5
	•	FTR

Command:

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
python3 train_investor_leagues_v2.py \
  --use-merged \
  --leagues "Australia A-League,Austria Bundesliga,Denmark Superliga,Swiss Super League,Germany Bundesliga 2,Czech First League,South Korea K League,Saudi Pro League,Turkey Super Lig" \
  --markets "btts,over25,under25,ftr"

Expected outputs per league:
	•	ModelStore/<LeagueTag>/btts_v3.pkl
	•	ModelStore/<LeagueTag>/over25_v3.pkl
	•	ModelStore/<LeagueTag>/under25_v3.pkl
	•	ModelStore/<LeagueTag>/ftr_v2.pkl
	•	ModelStore/<LeagueTag>/market_thresholds.json
	•	ModelStore/<LeagueTag>/features_manifest.json
	•	ModelStore/<LeagueTag>/calibrators/calibrators_meta.json

Important tag note

train_investor_leagues_v2.py required a split between:
	•	merged input tag resolver
	•	ModelStore output tag resolver

Australia required:
	•	merged file tag: Australia_A-League
	•	ModelStore tag: Australia_A_League

⸻

Step 10 — Train goal lambda / home-away fold5 regressors

Script used:
	•	_baseline_ftr_pipeline.quick_fit_goal_regressors(df, league)

Command:

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
for LEAGUE in \
  "Australia A-League" \
  "Austria Bundesliga" \
  "Denmark Superliga" \
  "Swiss Super League" \
  "Germany Bundesliga 2" \
  "Czech First League" \
  "South Korea K League" \
  "Saudi Pro League" \
  "Turkey Super Lig"
do
  echo ""
  echo "=== $LEAGUE ==="
  FORCE_RETRAIN_LAMBDA=1 VERBOSE_QUICK=1 LEAGUE="$LEAGUE" python3 - <<'PY'
import os
import pandas as pd
import _baseline_ftr_pipeline as m

league = os.environ["LEAGUE"]
root = os.path.dirname(m.__file__)
mp = os.path.join(root, "Matches", "__merged__", f"{league.replace(' ','_')}__merged.csv")

print(f"Using merged file: {mp}")
df = pd.read_csv(mp, low_memory=False)

try:
    df = m._canonize(df)
except Exception:
    pass

try:
    df = m._sanitize_pd_na_boolean(df)
except Exception:
    pass

m.quick_fit_goal_regressors(df, league)
print(f"✅ goal fold5 written for {league}")
PY
done

Expected outputs:
	•	ModelStore/<LeagueTag>/goal_ensembles/home_goals_fold5.pkl
	•	ModelStore/<LeagueTag>/goal_ensembles/away_goals_fold5.pkl
	•	ModelStore/<LeagueTag>/goal_ensembles/lambda_models_manifest.json

Australia note

Australia initially wrote lambda artifacts into:
	•	ModelStore/Australia_A-League

These were moved into the canonical folder:
	•	ModelStore/Australia_A_League/goal_ensembles/

Later, _baseline_ftr_pipeline.py was patched so future writes use the canonical tag.

⸻

Step 11 — Verify artifacts exist

Audit new leagues only

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
python3 - <<'PY'
from pathlib import Path

checks = {
    "Australia_A_League": "Australia A-League",
    "Austria_Bundesliga": "Austria Bundesliga",
    "Denmark_Superliga": "Denmark Superliga",
    "Swiss_Super_League": "Swiss Super League",
    "Germany_Bundesliga_2": "Germany Bundesliga 2",
    "Czech_First_League": "Czech First League",
    "South_Korea_K_League": "South Korea K League",
    "Saudi_Pro_League": "Saudi Pro League",
    "Turkey_Super_Lig": "Turkey Super Lig",
}

required = [
    "btts_v3.pkl",
    "over25_v3.pkl",
    "under25_v3.pkl",
    "ftr_v2.pkl",
    "market_thresholds.json",
    "features_manifest.json",
    "calibrators/calibrators_meta.json",
    "goal_ensembles/home_goals_fold5.pkl",
    "goal_ensembles/away_goals_fold5.pkl",
    "goal_ensembles/lambda_models_manifest.json",
]

for tag, league in checks.items():
    base = Path("ModelStore") / tag
    print(f"\n=== {league} | {tag} ===")
    ok = 0
    for rel in required:
        p = base / rel
        if p.exists():
            print(f"OK      {rel:<45} size={p.stat().st_size}")
            ok += 1
        else:
            print(f"MISSING {rel}")
    print(f"SUMMARY {ok}/{len(required)} present")
PY


⸻

Step 12 — Compare parity against established leagues

Use reference leagues such as:
	•	Germany_Bundesliga
	•	Italy_Serie_A
	•	Spain_La_Liga
	•	England_Premier_League

Compact parity table command:

cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
python3 - <<'PY'
from pathlib import Path
import pandas as pd

tags = {
    "Australia_A_League": "Australia A-League",
    "Austria_Bundesliga": "Austria Bundesliga",
    "Denmark_Superliga": "Denmark Superliga",
    "Swiss_Super_League": "Swiss Super League",
    "Germany_Bundesliga_2": "Germany Bundesliga 2",
    "Czech_First_League": "Czech First League",
    "South_Korea_K_League": "South Korea K League",
    "Saudi_Pro_League": "Saudi Pro League",
    "Turkey_Super_Lig": "Turkey Super Lig",
    "Germany_Bundesliga": "REF Germany Bundesliga",
    "Italy_Serie_A": "REF Italy Serie A",
    "Spain_La_Liga": "REF Spain La Liga",
    "England_Premier_League": "REF England Premier League",
}

required = {
    "btts": "btts_v3.pkl",
    "over25": "over25_v3.pkl",
    "under25": "under25_v3.pkl",
    "ftr": "ftr_v2.pkl",
    "thresholds": "market_thresholds.json",
    "manifest": "features_manifest.json",
    "cal_meta": "calibrators/calibrators_meta.json",
    "lambda_home": "goal_ensembles/home_goals_fold5.pkl",
    "lambda_away": "goal_ensembles/away_goals_fold5.pkl",
    "lambda_manifest": "goal_ensembles/lambda_models_manifest.json",
}

rows = []
for tag, label in tags.items():
    base = Path("ModelStore") / tag
    row = {"tag": tag, "league": label}
    for k, rel in required.items():
        row[k] = (base / rel).exists()
    row["present_count"] = sum(int(v) for k, v in row.items() if k in required)
    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_string(index=False))
PY

Target result:
	•	all new leagues = present_count 10
	•	reference leagues = present_count 10

⸻

Scripts used in this onboarding workflow

Raw ingest / file placement
	•	footystats_drop_ingest.py
	•	manual historical backfill Python one-off

Enriched build
	•	inline Python concat workflow from Matches/<League>/

Synth
	•	odds_synth.py

Merged build
	•	build_merged.py

Main market training
	•	train_investor_leagues_v2.py

Goal lambda / fold5 training
	•	_baseline_ftr_pipeline.py
	•	function:
	•	quick_fit_goal_regressors(df, league_name)

⸻

Important path/tag lessons

Australia A-League

Requires careful handling because:
	•	merged file tag = Australia_A-League
	•	ModelStore tag = Australia_A_League

England EFL League 1

Also requires special handling because:
	•	folder name = England EFL League 1
	•	canonical ModelStore tag may not be simple space replacement depending on legacy setup

Rule

Do not assume one single tag resolver is safe for:
	•	merged input paths
	•	ModelStore output paths
	•	runtime loaders

Use explicit overrides where needed.

⸻

Healthy final state checklist

A league is fully onboarded when all are true:
	•	raw historical files present
	•	fd_odds_enriched.csv present
	•	fd_odds_enriched_synth.csv present
	•	merged file present
	•	main market models present
	•	thresholds present
	•	features manifest present
	•	calibrator metadata present
	•	goal lambda fold5 artifacts present
	•	parity table matches reference leagues

⸻

Recommended next step after onboarding

Once structural parity is confirmed:
	1.	run runtime loader checks
	2.	run bookie_allmarkets.py for the new leagues only
	3.	confirm output columns populate correctly
	4.	verify deploy compatibility
	5.	add leagues into broader weekend workflow
	6.	include them in walkforward / audit framework

⸻

Notes
	•	Sweden Allsvenskan was excluded in this run
	•	the strict ingest app should remain for latest-season maintenance
	•	historical backfill should be done manually or with a dedicated one-off script
	•	after onboarding, future updates should use normal ingest rules

