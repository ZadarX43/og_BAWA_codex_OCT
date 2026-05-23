# Team Snapshot Weekend Support Board - All Leagues

## Purpose

This layer is a **fast structural support system** that sits beside the main model stack.

It does **not** replace your core prediction models.

It gives you a second lens based on team-level snapshot statistics so you can:

- sanity-check weekend FTR, OU2.5, and later BTTS views
- spot fixtures where the structural matchup is very strong before model outputs are merged
- compare model outputs against a simpler matchup-based support layer
- identify agreement, weak agreement, or future conflict between the snapshot layer and the main deployment stack

In plain English:

- the **main models** remain the decision engine
- the **snapshot board** is the support / confirmation layer

---

## What has been built

### 1) Single-league team snapshot matchup builder

**File:** `build_team_snapshot_matchup_features.py`

This script reads one fixture CSV and one team snapshot CSV for the same league/season, joins the home and away team snapshots onto each fixture row, creates matchup features, and writes an enriched fixture-level CSV.

Example output file pattern:

- `Matches/<League>/<league-matches...>__team_snapshot_matchups.csv`
- `Matches/<League>/<league-matches...>__team_snapshot_matchups__audit.csv`

---

### 2) All-leagues matchup builder wrapper

**File:** `build_team_snapshot_matchup_features_all.py`

This script scans the `Matches/` tree, finds eligible match CSVs, finds the matching team snapshot CSV in `Teams/`, runs the single-league builder for each league, and writes a build board.

Main board output:

- `predictions_output/TEAM_SNAPSHOT_MATCHUP_BUILD_BOARD.csv`

From your run:

- Total leagues processed: **151**
- OK: **144**
- Missing teams: **7**
- Failed: **0**

---

### 3) Snapshot family test harness

**File:** `test_team_snapshot_feature_families.py`

This script reads an enriched fixture-level matchup CSV, detects the `snap_` columns, buckets them into feature families, and tests baseline vs snapshot-enhanced feature sets for:

- FTR
- OU2.5
- BTTS

It supports:

- baseline only
- baseline + all snapshot fields
- baseline + one family at a time
- baseline + pair combinations of families

This is the script used to find which snapshot families help and which are noise.

---

### 4) Single-league weekend support board

**File:** `weekend_snapshot_support_board.py`

This script reads one enriched fixture-level matchup CSV, filters fixtures to a chosen date window, computes snapshot support scores for the weekend, and writes a league-level support board.

Example output:

- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD.csv`

Example use:

- Spain La Liga, March 13 to March 16, 2026

---

### 5) All-leagues weekend support board wrapper

**File:** `weekend_snapshot_support_board_all_leagues.py`

This script scans all `__team_snapshot_matchups.csv` files under `Matches/`, runs the single-league weekend board script for each one, then combines all output rows into a single all-leagues weekend board.

Main outputs:

- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv`
- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES__BUILD_BOARD.csv`

From your successful run:

- Total input files: **144**
- Status OK: **144**
- Status failed: **0**
- Combined rows out: **124**

---

## Core data flow

### Stage A - raw league fixture CSVs

Source folder:

- `Matches/<League>/...matches...csv`

These are the fixture / match-level files.

They contain the row-per-match structure used as the base layer.

---

### Stage B - raw team snapshot CSVs

Source folder:

- `Teams/<League>/...teams...csv`

These are the team summary tables for the same league/season.

They contain cumulative team performance and regime stats such as:

- points per game
- home / away splits
- league position
- goal difference
- goals scored / conceded per match
- clean sheet %
- FTS %
- BTTS %
- first team to score %
- half-time regime stats
- xG averages
- shots / shots on target
- corners / cards / fouls / possession
- goal timing buckets

---

### Stage C - enriched fixture-level matchup files

Built by:

- `build_team_snapshot_matchup_features.py`
- `build_team_snapshot_matchup_features_all.py`

Output pattern:

- `Matches/<League>/<fixture_file>__team_snapshot_matchups.csv`

These files are the bridge between raw team snapshots and fixture-level decision support.

Each fixture row is enriched with:

- joined home team snapshot fields
- joined away team snapshot fields
- derived `snap_...` matchup features

---

### Stage D - optional snapshot family testing

Built by:

- `test_team_snapshot_feature_families.py`

This stage is for validation, not operational weekend output.

It tells you which snapshot families actually add signal for a target market.

---

### Stage E - weekend support boards

Built by:

- `weekend_snapshot_support_board.py`
- `weekend_snapshot_support_board_all_leagues.py`

These produce the operational weekend board that you can compare against:

- `bookie_allmarkets.py`
- `deploy_gates.py`
- `deploy_thresholds.py` / `deploy_rulebook.py`

---

## Snapshot feature families

The first-pass matchup layer was built around simple, interpretable families.

### 1) Strength

Examples:

- `snap_strength_ppg_home_vs_away`
- `snap_strength_position_edge_home`
- `snap_strength_rank_edge_home`
- `snap_strength_goal_diff_home_vs_away`

Meaning:

- Is the home side structurally stronger than the away side?
- Useful mainly for FTR support and suppressing weak home-side inflation.

---

### 2) Attack vs defence

Examples:

- `snap_home_attack_vs_away_def_goals`
- `snap_away_attack_vs_home_def_goals`
- `snap_home_attack_vs_away_def_xg`
- `snap_away_attack_vs_home_def_xg`
- `snap_xg_total_pressure`

Meaning:

- How well does one side's attacking profile match up against the opponent's defensive profile?
- Useful for FTR, OU2.5, team-to-score, and team goals.

---

### 3) Clean sheet / FTS / BTTS regime

Examples:

- `snap_home_clean_sheet_vs_away_fts_edge`
- `snap_away_clean_sheet_vs_home_fts_edge`
- `snap_btts_regime_blend`

Meaning:

- Does one team keep clean sheets often?
- Does the opponent fail to score often?
- Does the fixture structurally resemble a BTTS game or a no-BTTS game?

This family matters a lot for BTTS and also for OU2.5.

---

### 4) Half-time regime

Examples:

- `snap_ht_home_attack_vs_away_def`
- `snap_ht_away_attack_vs_home_def`
- `snap_ht_home_lead_vs_away_trail_edge`
- `snap_ht_away_lead_vs_home_trail_edge`
- `snap_ht_goal_regime_blend`

Meaning:

- Does the matchup lean toward early control, early pressure, or stronger first-half action?
- Useful for FTR overlays and future first-half / in-play logic.

---

### 5) Style / chaos

Examples:

- `snap_style_chaos_index`
- `snap_ou25_over_regime_blend`
- `snap_ou25_under_regime_blend`
- `snap_cards_chaos_blend`
- `snap_corners_chaos_blend`
- `snap_fouls_chaos_blend`
- `snap_possession_control_edge_home`

Meaning:

- Is the game structurally stretched, compressed, chaotic, territorial, or low-event?
- Most relevant to OU2.5, BTTS, and tempo interpretation.

---

### 6) Timing pressure

Examples:

- `snap_timing_early_goal_pressure`
- `snap_timing_late_goal_pressure`
- `snap_timing_second_half_acceleration`
- `snap_timing_both_teams_late_risk`

Meaning:

- Do the two teams show early scoring threat, late scoring threat, or second-half acceleration patterns?
- Strongly useful for OU2.5 support and live model direction later.

---

## Key commands

## A) Build one league's matchup layer

```bash
python3 build_team_snapshot_matchup_features.py \
  --fixtures-csv "Matches/Spain La Liga/spain-la-liga-matches-2025-to-2026-stats.csv" \
  --teams-csv "Teams/Spain La Liga/spain-la-liga-teams-2025-to-2026-stats.csv"
```

This writes:

- `Matches/Spain La Liga/spain-la-liga-matches-2025-to-2026-stats__team_snapshot_matchups.csv`
- `Matches/Spain La Liga/spain-la-liga-matches-2025-to-2026-stats__team_snapshot_matchups__audit.csv`

---

## B) Build matchup layers for all leagues

```bash
python3 build_team_snapshot_matchup_features_all.py \
  --matches-root "Matches" \
  --teams-root "Teams" \
  --single-league-script "build_team_snapshot_matchup_features.py"
```

Main result:

- `predictions_output/TEAM_SNAPSHOT_MATCHUP_BUILD_BOARD.csv`

---

## C) Test snapshot families on one enriched fixture file

```bash
python3 test_team_snapshot_feature_families.py \
  --input-csv "Matches/Spain La Liga/spain-la-liga-matches-2025-to-2026-stats__team_snapshot_matchups.csv" \
  --combo-mode all_pairs
```

Main outputs:

- `predictions_output/team_snapshot_family_tests/...__long.csv`
- `predictions_output/team_snapshot_family_tests/...__summary.csv`
- `predictions_output/team_snapshot_family_tests/...__family_audit.csv`

---

## D) Build one league's weekend support board

```bash
python3 weekend_snapshot_support_board.py \
  --input-csv "Matches/Spain La Liga/spain-la-liga-matches-2025-to-2026-stats__team_snapshot_matchups.csv" \
  --date-from "2026-03-13" \
  --date-to "2026-03-16"
```

Main output:

- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD.csv`

---

## E) Build the all-leagues weekend support board

```bash
python3 weekend_snapshot_support_board_all_leagues.py \
  --matches-root "Matches" \
  --single-league-script "weekend_snapshot_support_board.py" \
  --date-from "2026-03-13" \
  --date-to "2026-03-15"
```

Main outputs:

- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv`
- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES__BUILD_BOARD.csv`

---

## How the weekend board works

The weekend board is built from the enriched fixture-level matchup files.

For each fixture inside the date window, it calculates snapshot-derived support values and outputs a compact board with fields like:

- `core_ftr_pick`
- `core_ftr_home_win_proba`
- `snapshot_ftr_support_score`
- `snapshot_ftr_support_bucket`
- `core_ou25_pick`
- `core_ou25_over_proba`
- `snapshot_ou25_support_score`
- `snapshot_ou25_support_bucket`
- `conflict_flag`

At the moment, based on your run, the board is still in **snapshot-first mode**.

That means:

- `snapshot_*` fields are working
- `core_*` model picks are still `unknown`
- `conflict_flag` remains `neutral`

Once real model outputs are merged in, the board becomes the true comparison surface.

---

## What `snapshot_ftr_support_score` means

This is a **fixture-level structural support score for the home-side FTR angle**.

It is based on a weighted blend of matchup information such as:

- home vs away strength mismatch
- home attack vs away defence
- first-half edge / lead-trail edge
- timing pressure
- likely regime of the matchup

In simple terms:

- higher score = stronger structural support for the **home side** in FTR terms
- lower score = weaker or opposing structural shape for the **home side**

This is important because the current snapshot layer is **not symmetric yet**.
It is not directly choosing between home and away as equal opposing picks.
It is effectively answering:

**“How strongly does the snapshot layer support the home team in FTR terms?”**

### UI / board interpretation

This explainability rule should be surfaced clearly in the UI:

- rows with **`snapshot_ftr_support_score >= 70`** are strong **home-team FTR support** rows
- those fixtures should visually point toward the **home team** as the snapshot-backed side

Examples:

- Wycombe Wanderers vs Luton Town → snapshot FTR is backing **Wycombe Wanderers**
- Lincoln City vs Stockport County → snapshot FTR is backing **Lincoln City**
- Inter Milan vs Atalanta → snapshot FTR is backing **Inter Milan**
- PSG vs Nantes → snapshot FTR is backing **PSG**

### Practical interpretation

A simple working framework is:

- **70+** → strong home FTR support
- **60-69.99** → decent / supportive home edge
- **40-59.99** → neutral / mixed / messy
- **below ~40** → home side is not well supported, so away / draw danger rises

These are not automatically final picks. They are **structural support signals**.

### Why this matters in the UI

When the board is shown in the app or weekend support UI, a user should be able to tell at a glance:

- whether the snapshot layer strongly likes the **home team**
- whether the fixture is neutral / messy
- whether the home side is structurally weak and therefore carries away/draw danger

That means the UI should not only show the raw score, but also make the direction obvious:

- `70+` = **home-backed snapshot row**
- low score = **home not supported**

### Example reading logic

- model says home + snapshot `70+` → healthier home case
- model says home + snapshot neutral → caution
- model says home + snapshot oppose / very low score → conflict or downgrade
- model says away + snapshot very low home score → cleaner away/draw case

## What `snapshot_ftr_support_bucket` means

This is the bucketed interpretation of the raw FTR support score.

Observed bucket values include:

- `strong_support`
- `support`
- `neutral`
- `weak`
- `oppose`

Interpretation:

- `strong_support` = the snapshot layer strongly likes the home-side structural case
- `support` = reasonably favourable structural profile
- `neutral` = mixed / not decisive
- `weak` = not enough structure to back the home-side view strongly
- `oppose` = structural signals push against the home-side view

---

## What `snapshot_ou25_support_score` means

This is the **fixture-level structural support score for an Over 2.5 angle**.

It is built from matchup fields related to:

- attack vs defence pressure
- xG pressure
- OU2.5 regime blends
- style / chaos profile
- timing pressure
- first-half and late-goal behaviour

In plain English:

- higher score = stronger structural case for goals / over environment
- lower score = more resistance to a free-scoring game

### Practical interpretation

Again, from your outputs, the live working interpretation looks roughly like:

- **70+** → strong over support
- **60-69** → support
- **40-59** → neutral / mixed
- **below ~40** → oppose

### Examples from your board

Top OU2.5 support examples:

- PSV vs NEC → 82.46
- Charlotte vs Inter Miami → 80.72
- Lincoln City vs Stockport → 80.62
- Barcelona vs Sevilla → 79.07
- Coventry vs Southampton → 77.99
- Sporting vs Tondela → 77.92
- Inter vs Atalanta → 77.62

---

## What `snapshot_ou25_support_bucket` means

This is the bucketed form of the OU2.5 support score.

Observed values include:

- `strong_support`
- `support`
- `neutral`
- `oppose`

Interpretation:

- `strong_support` = matchup shape strongly supports an over environment
- `support` = favourable over structure
- `neutral` = not enough structural conviction either way
- `oppose` = the snapshot layer leans against the over case

---

## What `core_ftr_pick` and `core_ou25_pick` currently mean

At this stage they are placeholders for the **main model outputs**.

Right now, your all-leagues weekend board shows `unknown` because the snapshot board has not yet been merged with the real deploy outputs.

So the current board is telling you:

- “Here is the structural support profile”
- not yet: “Here is the final model pick versus snapshot opinion”

---

## What `conflict_flag` means

This field is intended to tell you whether the snapshot layer agrees or disagrees with the main model output.

Example future uses:

- model says home win, snapshot says strong_support → healthy agreement
- model says home win, snapshot says neutral → caution
- model says home win, snapshot says oppose → conflict
- model says over 2.5, snapshot says oppose → conflict

From your run, the conflict table was empty because:

- `core_ftr_pick = unknown`
- `core_ou25_pick = unknown`

So there is currently nothing real to compare against.

That is normal.

---

## How to use the board this weekend

### Current best use

Use the snapshot board as a **preliminary support board** before the main deployment outputs are merged.

That means:

- shortlist strong structural fixtures
- note neutral fixtures
- flag structurally weak or opposing fixtures

Then compare the shortlist against:

- `bookie_allmarkets.py`
- `deploy_gates.py`
- `deploy_thresholds.py` / `deploy_rulebook.py`

### Operational interpretation

#### For FTR

- strong model home pick + strong snapshot support = healthier
- strong model home pick + neutral snapshot = caution
- strong model home pick + oppose snapshot = review / possible downgrade

#### For OU2.5

- over model signal + strong snapshot OU25 support = healthier
- over signal + neutral snapshot = caution
- over signal + oppose snapshot = review / possible downgrade

#### For BTTS

The current weekend board is FTR / OU25 oriented, but the same snapshot layer can support BTTS logic:

- BTTS Yes healthier when:
  - clean sheet edges are weak
  - BTTS regime blend is high
  - attack pressure is balanced
  - timing / late-risk is high

- BTTS No healthier when:
  - clean sheet / FTS edges are strong
  - one side suppresses the other structurally
  - goal environment is flatter

---

## What the family tests already suggested

From your Spain La Liga family test outputs:

### FTR
Best helpful snapshot additions looked to be:

- timing pressure
- half-time + timing pressure
- clean_sheet_fts_btts + timing pressure

This suggests the easier edge for FTR support is not broad chaos stacking, but more targeted timing and regime structure.

### OU2.5
Baseline was already strongest in the sample you showed.

That suggests:

- snapshot additions may be useful as confirmation or caution
- but not necessarily as an automatic improvement layer for OU2.5 training in that sample

### BTTS
Baseline also looked stronger than all broad snapshot additions.

That suggests:

- avoid blindly shoving all snapshot columns into BTTS training
- use the snapshot layer more as an interpretation / support filter
- especially the clean-sheet / FTS / BTTS regime family, but selectively

---

## Important limitations

### 1) This is a support layer, not the final engine

The snapshot board is not your primary pick generator.

It is a structural validation layer.

### 2) The current all-leagues weekend board does not yet contain live model picks

That is why:

- `core_ftr_pick` is `unknown`
- `core_ou25_pick` is `unknown`
- `conflict_flag` is neutral everywhere

### 3) Team snapshot stats are cumulative abstractions

They are useful, but they are still summary-layer statistics. They do not replace richer fixture-level context from the main models.

### 4) More columns is not automatically better

Your family tests already showed that broad snapshot stacking can degrade performance.

So the correct use is:

- targeted support
- selective family testing
- side-by-side comparison against the core model stack

---

## Recommended next workflow

### Step 1
Keep the all-leagues snapshot weekend board as your structural reference file:

- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv`

### Step 2
Run the core model weekend pipeline:

- `bookie_allmarkets.py`
- `deploy_gates.py`
- `deploy_thresholds.py` / `deploy_rulebook.py`

### Step 3
Merge model outputs against the snapshot board.

That is where the board becomes fully operational.

### Step 4
Use merged logic like:

- model strong + snapshot strong = green
- model strong + snapshot neutral = amber
- model strong + snapshot oppose = red / review

---

## File list summary

### Raw inputs

- `Matches/<League>/*matches*.csv`
- `Teams/<League>/*teams*.csv`

### Build scripts

- `build_team_snapshot_matchup_features.py`
- `build_team_snapshot_matchup_features_all.py`

### Testing script

- `test_team_snapshot_feature_families.py`

### Weekend board scripts

- `weekend_snapshot_support_board.py`
- `weekend_snapshot_support_board_all_leagues.py`

### Main build / output files

- `Matches/<League>/*__team_snapshot_matchups.csv`
- `Matches/<League>/*__team_snapshot_matchups__audit.csv`
- `predictions_output/TEAM_SNAPSHOT_MATCHUP_BUILD_BOARD.csv`
- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD.csv`
- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES.csv`
- `predictions_output/WEEKEND_SNAPSHOT_SUPPORT_BOARD__ALL_LEAGUES__BUILD_BOARD.csv`

---

## Short version

You have successfully built a working all-leagues structural support layer that:

- joins team snapshots onto fixture rows
- derives interpretable matchup features
- produces weekend support boards across many leagues
- is ready to be used as a comparison layer against this weekend's model deployments

The next real move is not more structural tinkering.

It is to run the main weekend model outputs and compare them directly against this board.




## Selecting the prediction window

The weekend support board does not hardcode a weekend.

You choose the prediction window directly on the command line with:

- `--date-from`
- `--date-to`

Example for one league:

```bash
python3 weekend_snapshot_support_board.py \
  --input-csv "Matches/Spain La Liga/spain-la-liga-matches-2025-to-2026-stats__team_snapshot_matchups.csv" \
  --date-from "2026-03-13" \
  --date-to "2026-03-16"

  Example for all leagues:

  python3 weekend_snapshot_support_board_all_leagues.py \
  --matches-root "Matches" \
  --single-league-script "weekend_snapshot_support_board.py" \
  --date-from "2026-03-13" \
  --date-to "2026-03-15"

  The script parses fixture kick-off times and filters rows to only include matches inside the chosen window.




Why: the current snapshot_ftr_support_score is built from home-minus-away style matchup edges. A high positive score means the home side has the stronger snapshot profile across the support components. It is not choosing “either side” symmetrically yet; it is basically answering:

“How strongly does the snapshot layer support the home side in FTR terms?”

How to read it:
	•	strong_support / support = supports the home team
	•	neutral = no strong home-side edge from snapshot layer
	•	oppose = works against the home team, which usually implies the away side is the healthier FTR direction
	•	very low home support scores are the ones where you’d start thinking “away/draw danger”

Important detail:
	•	core_ftr_pick = unknown means the weekend snapshot board is not itself producing the final FTR pick
	•	it is a support layer
	•	once you overlay it onto bookie_allmarkets.py / deploy_gates.py / thresholds output, then it becomes useful as:
	•	model says home + snapshot strong_support → healthier
	•	model says home + snapshot oppose → caution
	•	model says away + snapshot strong home support → conflict
	•	model says away + snapshot oppose home → cleaner away case

A simple rule for now:
	•	70+ = strong home FTR support
	•	60–69.99 = decent home support
	•	40–59.99 = neutral/messy
	•	below ~40 = home side not well supported, so away/draw risk rises

For your two direct questions:
	•	Wycombe or Luton? → snapshot says Wycombe
	•	Lincoln or Stockport? → snapshot says Lincoln City
