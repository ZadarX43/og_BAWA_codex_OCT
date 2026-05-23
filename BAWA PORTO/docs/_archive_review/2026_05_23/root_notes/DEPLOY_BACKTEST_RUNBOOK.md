Absolutely — below is a clean markdown runbook / notes draft you can paste into your repo (e.g. DEPLOY_BACKTEST_RUNBOOK.md).

It captures:
	•	what files are used
	•	what each file does
	•	commands used to run + test
	•	how rulebook + gates work
	•	thresholds (including the new FTR demote → STANDARD promotions)
	•	what to verify before a weekend test deploy

⸻

OG Deploy + Backtesting Runbook (Current Working Notes)

Purpose

This document is the working runbook for the current OG deploy / gating / backtesting flow.

It records:
	•	the core files in use
	•	the command chain for backtests and deploy generation
	•	how deploy_rulebook.py applies gates
	•	key thresholds and current assumptions
	•	recent FTR demotion-promotion patch behavior (Lane 1 / Lane 2)
	•	sanity checks to run before a weekend test deploy

⸻

1) Core Files in Use

Backtesting / research files

backtest_deploy_csv.py

Scores a deploy CSV against truth (Matches/*) and computes:
	•	correct per row (market-specific)
	•	hit = mean(correct)
	•	avg_od = mean(odds)
	•	roi = flat-stake ROI using decimal odds

Used for:
	•	month backtests
	•	gated deploy ROI validation
	•	market-level / source-level performance breakdowns

⸻

filter_deploy_by_rulebook.py

Applies historical rulebook-style gating to a row-level backtest CSV.

Expects:
	•	row-level CSV with columns like: market, bookie_pick, score, correct
	•	Typically input is a __BACKTEST.csv (not a summary or aggregate file)

Outputs:
	•	__GATED.csv (row-level, after gating)
	•	__GATED_SUMMARY.csv (aggregate stats)
	•	__RULEBOOK_SNAPSHOT.json (rulebook config snapshot)

Used for:
	•	value-vs-accuracy FTR profile comparisons
	•	testing how a rulebook would have performed historically on specific rows

Note: filter_deploy_by_rulebook.py is not for use directly on summary exports or season_leave_one_out* outputs unless they are row-level and include the required columns.

⸻

seasonal_market_ledger.py

Builds a seasonal ledger view of market performance.

Used for:
	•	tracking per-market behavior over time
	•	understanding which products are stable / fragile across seasons

⸻

build_months.sh

Batch runner for month-by-month builds / backtests.

Used for:
	•	repeatable monthly runs
	•	creating comparable monthly artifacts

⸻

aggregate_gated_backtests.py

Aggregates multiple monthly gated backtest summaries.

Used for:
	•	multi-month rollups
	•	combined ledger / trend summaries
	•	checking if April-style edge persists across windows

⸻

season_leave_one_out.py

Leave-one-out market analysis (ablation style).

Used for:
	•	identifying drag markets (e.g., TG25 in some windows)
	•	quantifying ROI / hit-rate uplift when a market is removed

⸻

build_months_2022_2024.sh

Batch runner focused on historical backtest windows (2022–2024).

Used for:
	•	reproducible historical comparisons
	•	testing rule/gate behavior over older seasons

⸻

season_leave_one_out_2023_2024.py

Season-specific LOO (2023/24).

Used for:
	•	focused diagnostics on one season
	•	validating whether broad findings hold in that season only

⸻

season_leave_one_out_2022_2023.py

Season-specific LOO (2022/23).

Used for:
	•	same as above, prior season
	•	cross-season consistency checks

⸻

Deploy / generation files

deploy_presets.py

Preset parameter definitions for deploy profiles.

Used for:
	•	centralizing parameter sets
	•	switching deploy behavior without editing rulebook internals each time

⸻

bookie_allmarkets.py

Generates ALLMARKETS long-format prediction stack (one row per fixture per market).

Used for:
	•	source artifact before deploy filtering / gating
	•	core output feeding rulebook / deploy stages

Markets typically include:
	•	FTR
	•	OU25
	•	BTTS
	•	TG15
	•	TG25

⸻

deploy_gates.py

Reusable gate logic / gate helpers (depending on current codebase usage).

Used for:
	•	shared gating checks
	•	structured rule filtering support (where imported)

⸻

deploy_rulebook.py

Main rulebook application script (legacy V1 deploy preset + tiering logic in current flow).

Used for:
	•	deterministic gating per market
	•	deploy output generation
	•	tier outputs (ELITE, STANDARD, OBSERVE)
	•	FTR demotion and post-demotion selective re-promotion logic (recent patch)

⸻

2) High-Level Flow (Current Working Chain)

A) Generate ALLMARKETS

bookie_allmarkets.py produces a long-format market stack.

Output example:
	•	BOOKIE_IMP62_ALLMARKETS_<date_from>_to_<date_to>.csv

This is the raw market candidate set before deploy gating/tiering.

⸻

B) Run deploy_rulebook.py

deploy_rulebook.py applies:
	•	runtime gates (market-agnostic / consistency checks)
	•	market-specific gates (FTR / OU25 / BTTS / TG)
	•	tier assignment (ELITE, STANDARD, OBSERVE)
	•	special handling for FTR demoted rows

Outputs typically include:
	•	__DEPLOY_PRESET_V1.csv
	•	__DEPLOY_PRESET_V1.md
	•	tiered CSVs:
	•	__DEPLOY_TIER_ELITE.csv
	•	__DEPLOY_TIER_STANDARD.csv
	•	__DEPLOY_TIER_OBSERVE.csv

⸻

C) Build/locate source historical CSV (e.g. _SEASON_LEAVE_ONE_OUT__*.csv or other deploy-like export)

This step obtains a historical export for backtesting. Note that _SEASON_LEAVE_ONE_OUT__*.csv files may be summary or ablation outputs and are not always row-level deploy rows. Ensure that the file you use contains row-level deploy-like predictions, not just market or season summaries.

⸻

D) Run backtest_deploy_csv.py to create row-level __BACKTEST.csv with truth joins and correct

Use backtest_deploy_csv.py to score the deploy-like source rows against truth data (Matches/*). This produces a row-level __BACKTEST.csv with columns such as market, bookie_pick, score, correct, etc.

This enables:
	•	per-row scoring and truth joins
	•	downstream rulebook-style gating and analysis

⸻

E) Run filter_deploy_by_rulebook.py on the row-level __BACKTEST.csv (not on summary files) to create gated comparisons

filter_deploy_by_rulebook.py expects a row-level __BACKTEST.csv (with market, bookie_pick, etc). It applies the rulebook logic historically and outputs:
	•	__GATED.csv (row-level after gating)
	•	__GATED_SUMMARY.csv (aggregate stats)
	•	__RULEBOOK_SNAPSHOT.json (rulebook config)

This step is essential for value-vs-accuracy FTR profile comparisons and for simulating historical gating.

⸻

F) Aggregate / compare / ablate using aggregate/ledger/leave-one-out scripts

Use:
	•	aggregate_gated_backtests.py
	•	seasonal_market_ledger.py
	•	leave-one-out scripts

This is where you:
	•	roll up results across months/seasons
	•	compare different gates or ablations
	•	decide what to deploy, suppress, or treat as observe-only

⸻

3) Command Log (Run + Test)

3.1 Help / sanity (CLI wiring check)

python deploy_rulebook.py --help >/tmp/deploy_rulebook_help.txt && tail -n 20 /tmp/deploy_rulebook_help.txt

Purpose:
	•	confirms script loads
	•	confirms CLI parses successfully
	•	catches syntax/import breakage after patching

⸻

3.2 Current deploy_rulebook run (example used)

python3 deploy_rulebook.py \
  --src predictions_output/2026-02-20/BOOKIE_IMP62_ALLMARKETS_2026-02-14_to_2026-02-17.csv \
  --outdir predictions_output/2026-02-20 \
  --debug

What this does:
	•	reads ALLMARKETS source
	•	applies runtime + market gates
	•	writes deploy preset outputs
	•	writes tiered outputs
	•	prints detailed gate-stage diagnostics

⸻

3.2b Historical gated backtest compare (value vs accuracy)

# 1. Run backtest_deploy_csv.py on a historical deploy-like export (must be row-level, e.g. _SEASON_LEAVE_ONE_OUT__2024_2025.csv)
python3 backtest_deploy_csv.py \
  --src predictions_output/_SEASON_LEAVE_ONE_OUT__2024_2025.csv \
  --out predictions_output/_SEASON_LEAVE_ONE_OUT__2024_2025__BACKTEST.csv

# 2. Run filter_deploy_by_rulebook.py for value profile (on the row-level __BACKTEST.csv output)
python3 filter_deploy_by_rulebook.py \
  --src predictions_output/_SEASON_LEAVE_ONE_OUT__2024_2025__BACKTEST.csv \
  --profile value \
  --outdir predictions_output/

# 3. Run filter_deploy_by_rulebook.py for accuracy profile (on the same __BACKTEST.csv)
python3 filter_deploy_by_rulebook.py \
  --src predictions_output/_SEASON_LEAVE_ONE_OUT__2024_2025__BACKTEST.csv \
  --profile accuracy \
  --outdir predictions_output/

Note: Do not pass __BACKTEST_SUMMARY.csv or seasonal summary exports directly into filter_deploy_by_rulebook.py; it expects row-level rows with market/bookie_pick.

⸻

3.3 Quick grep checks (patch validation)

Check demotion tokens exist in rulebook

rg -n "DEMOTED_ULTRASHORT_STANDARD|DEMOTED_MARGIN_STANDARD|DEMOTED_TO_OBSERVE" deploy_rulebook.py

Expected (after patch):
	•	DEMOTED_ULTRASHORT_STANDARD
	•	DEMOTED_MARGIN_STANDARD
	•	DEMOTED_TO_OBSERVE

⸻

Check tiering block in code (demote/promotion logic)

sed -n '4156,4275p' deploy_rulebook.py

Purpose:
	•	confirms patched block is present
	•	confirms demoted_promoted naming
	•	confirms lane-specific reason tagging

⸻

Check parameter references exist

rg -n "ftr_demote_lane2_|draw_chaos_score|ftr_margin|bookie_od" deploy_rulebook.py

Purpose:
	•	ensure Lane 2 parameters and required columns are actually wired in rulebook

⸻

3.4 Tier duplication test (cross-tier integrity)

After a deploy run, verify no row appears in more than one tier on key columns.

Key test result (recent):
	•	Total rows across tiers: 61
	•	Duplicate key rows across tiers: 0
	•	✅ No cross-tier duplicates on key columns

Recommended key columns:
	•	league
	•	fixture_key
	•	market
	•	bookie_pick

⸻

3.5 Promotion verification test (FTR demoted → STANDARD patch)

Recent validation result from STANDARD tier:
	•	STANDARD rows: 3
	•	Promoted demoted rows in STANDARD: 3

Example promoted rows observed:
	•	Liverpool vs Brighton (FTR HOME) → DEMOTED_MARGIN_STANDARD
	•	Burnley vs Mansfield (FTR HOME) → DEMOTED_MARGIN_STANDARD
	•	Man City vs Salford (FTR HOME) → DEMOTED_ULTRASHORT_STANDARD|DEMOTED_MARGIN_STANDARD

This confirms:
	•	Lane 2 promotions are working
	•	Lane 1 + Lane 2 can both apply on the same row (Man City case)

⸻

4) How deploy_rulebook.py Gates Work (Conceptual)

Stage 1 — Runtime gates

The rulebook first applies broad “runtime pass” filtering.

From recent example:
	•	raw = 107
	•	runtime pass = 61

This stage removes rows with:
	•	hard veto conditions
	•	signal mismatches
	•	inconsistent structures
	•	market-specific pre-veto issues (logged in veto top12 summary)

Example veto logs observed:
	•	OVER25_P00_TOO_HIGH
	•	BTTS_YES_CS1_TRAP
	•	SIGNAL_MISMATCH

⸻

Stage 2 — Market-specific rulebook gates

Each market is processed separately with tailored thresholds.

Markets seen in logs:
	•	FTR
	•	OU25
	•	BTTS
	•	TG15
	•	TG25

Each market has a staged funnel (start → agree → thresholds → structural checks → final kept rows).

⸻

Stage 3 — Tiering (ELITE, STANDARD, OBSERVE)

After rulebook pass:
	•	ELITE = strictest subset
	•	STANDARD = deployable but less strict than ELITE
	•	OBSERVE = runtime-pass rows not selected for ELITE/STANDARD + explicitly demoted rows

Important invariant:
	•	tiers must be disjoint
	•	patched code now explicitly removes promoted demoted rows from OBSERVE after re-adding to STANDARD

⸻

5) FTR Rulebook Gating (Detailed Notes)

Below reflects the observed gate path from debug logs and code references.


	•	FTR value profile: value-priced FTR selections (default bookie_od >= 2.14)
	•	FTR accuracy profile: moderate favourites (recommended cap testing 1.75–1.85)
	•	These are separate products, not mirror images


	•	If accuracy profile shows ALL rows but zero MARKET::ftr, that means FTR accuracy dead-zone, not a failed batch run.

That’ll save future-you hours.

FTR gate sequence (observed)

Typical log flow:
	1.	FTR start
	2.	FTR after agree
	3.	FTR implied_min gate
	4.	FTR after gap_min
	5.	FTR after margin_min
	6.	FTR xg confirm
	7.	FTR ppg confirm
	8.	FTR(GLUE) not_glue_flag veto
	9.	FTR od_min
	10.	FTR power_diff gate

This means a row can survive early model/odds checks and still get dropped later on:
	•	PPG confirmation
	•	glue veto
	•	odds minimum
	•	power diff constraints

⸻

FTR columns used (seen in code/logs)

Core:
	•	bookie_pick
	•	bookie_od
	•	bookie_implied
	•	bookie_implied_used
	•	ftr_margin
	•	gap_used

Structural/context:
	•	draw_chaos_score
	•	power_diff
	•	not_glue_flag
	•	PPG / xG columns (various confirmations)

Fallback support:
	•	od_home, od_draw, od_away (used for FTR bookie_od backfill when missing)

⸻

FTR thresholds seen in logs / code (current working references)

Implied minimum

From debug run:
	•	FTR implied_min gate thr=0.680

⸻

Gap gate

Log shows dynamic gap handling:
	•	HARD>=-0.80
	•	short>=0.65:-0.45
	•	else -0.10
	•	ANCHOR_CAND:-0.55

(These appear to be context-dependent gap acceptance thresholds)

⸻

Margin minimum

ftr_margin is a core structural filter.

Code references show:
	•	default / params usage around ftr_margin_min
	•	observed lane2 promotion uses its own separate margin threshold (see below)

⸻

Odds minimum

Observed log:
	•	FTR after od_min>=1.167

This can remove very short-price rows even if they pass earlier gates.

⸻

Power-diff gate

Observed log:
	•	FTR power_diff pre-gate...
	•	FTR after power_diff gate

This is a structural sanity layer (particularly for directional picks).

⸻

6) FTR Demotion → Re-promotion Patch (Recent Change)

Why this exists

Some strong FTR favourites were being demoted into OBSERVE despite being useful “safe-leg” candidates.

The patch adds selective re-promotion of certain demoted FTR rows into STANDARD.

⸻

Where it lives

deploy_rulebook.py tiering block (around lines ~4156–4275 in current file)

⸻

Logic summary (patched behavior)

Source set
	•	Start from rows in demote_keys (demoted rows)
	•	Build demoted = df_pass[demoted rows]

Split demoted rows into:
	•	demoted_promoted = rows meeting Lane 1 or Lane 2 promotion criteria
	•	demoted_rest = rows remaining in OBSERVE

⸻

Promotion lanes

Lane 1 — Ultra-short favourites

Promote if either condition is true:
	•	bookie_od <= ftr_short_od_max
	•	OR bookie_implied_used (fallback: bookie_implied) >= ftr_short_imp_min

Current defaults in code block:
	•	ftr_short_od_max = 1.10
	•	ftr_short_imp_min = 0.85

Reason token added:
	•	DEMOTED_ULTRASHORT_STANDARD

⸻

Lane 2 — High-margin favourites with acceptable draw chaos

Promote if all conditions pass:
	•	bookie_od <= ftr_demote_lane2_od_max
	•	ftr_margin >= ftr_demote_lane2_margin_min
	•	draw_chaos_score <= ftr_demote_lane2_draw_chaos_max (or missing = allowed)

Current params used in code:
	•	ftr_demote_lane2_od_max = 1.60
	•	ftr_demote_lane2_margin_min = 0.08
	•	ftr_demote_lane2_draw_chaos_max = 0.92

Reason token added:
	•	DEMOTED_MARGIN_STANDARD

⸻

Important patched behavior (naming + integrity)

Renamed working variable
	•	old concept: demoted_ultra
	•	patched variable: demoted_promoted

Why:
	•	promotions are no longer only “ultra-short”
	•	avoids future confusion/false bug hunts

⸻

OBSERVE removal for promoted rows

After appending promoted rows to STANDARD, the code removes them from OBSERVE using row keys.

This prevents cross-tier duplication.

⸻

Comment updated (important)

The code comment was updated from “ultra-short only” wording to reflect both lanes:
	•	Lane 1 = ultra-short favourites
	•	Lane 2 = margin/chaos-qualified favourites

⸻

7) OU25 / BTTS / TG Gating (Operational Summary)

OU25

Observed flow includes:
	•	agree
	•	implied-band filters
	•	split OVER vs UNDER
	•	model floors
	•	signal gates
	•	structural vetoes (FTS / CS1 / under-shape etc.)
	•	goaliness / rate checks

Important observation from recent run:
	•	OVER side can get heavily thinned by strong-label + probability gates
	•	UNDER side may fail-open on some bands but still passes via structural filters

⸻

BTTS

Observed flow includes:
	•	split YES / NO
	•	implied max (YES)
	•	strong-label gates
	•	model floor
	•	FTS-based veto/structure
	•	low-goaliness checks (especially NO)
	•	recombine

Important behavior:
	•	YES and NO are treated differently (as they should be)
	•	BTTS_NO is often highly selective and can be very strong in small samples

⸻

TG15 / TG25

Observed flow includes:
	•	agree
	•	pmin
	•	tg_pois_ok==1
	•	Poisson tail thresholds (e.g., pois_ge2 for TG15)

Recent run example:
	•	TG15 narrowed to zero after pois_ge2>=0.50
	•	TG25 narrowed to zero after pmin

This matches prior backtesting evidence that TG markets (especially TG25) are sensitive and can become drag markets depending on thresholds/window.

⸻

8) Backtest Metrics — How Accuracy / ROI Are Calculated

For each deploy row:
	•	correct = whether the selection won (market-specific truth logic)
	•	hit = mean(correct)
	•	avg_od = mean(odds)
	•	roi = flat-stake average profit using decimal odds

ROI formula (decimal odds)
	•	profit per bet = correct * (odds - 1) - (1 - correct)
	•	roi = mean(profit per bet)

Equivalent shorthand often referenced:
	•	roi = mean(correct * odds - 1)

(These are algebraically equivalent if correct is 0/1)

⸻

9) April 2025 Gated Backtest (Reference Ledger)

These are the current headline reference numbers for the April gated profile.

Combined
	•	All combined: 84.6% hit
	•	n=285
	•	avg_od=1.864
	•	ROI=+0.490

Market accuracies (April gated)
	•	FTR: 62.7%
	•	OU25: 93.1%
	•	BTTS: 91.7%
	•	TG15: 97.6% (proxy/fair odds for ROI interpretation if no real bookie odds)
	•	TG25: 78.9%
	•	All combined: 84.6%

Notes
	•	TG ROI is currently proxy/fair-odds ROI if bookmaker TG odds are missing.
	•	Always separate bookie_od_source = bookmaker vs model_fair in interpretation.

⸻

10) Current Threshold Notes to Keep in Mind

FTR (from current observations / logs)
	•	implied_min ≈ 0.680 (seen in run)
	•	od_min ≈ 1.167 (seen in run)
	•	ftr_margin used as core structural gate
	•	draw_chaos_score used in glue/draw-trap contexts and Lane 2 demote promotions
	•	power_diff gate applied late

Demoted FTR re-promotion thresholds (patched)
	•	ftr_short_od_max = 1.10
	•	ftr_short_imp_min = 0.85
	•	ftr_demote_lane2_od_max = 1.60
	•	ftr_demote_lane2_margin_min = 0.08
	•	ftr_demote_lane2_draw_chaos_max = 0.92

⸻

11) Pre-Weekend Test Deploy Checklist (Recommended)

A) Code / patch integrity
	•	python deploy_rulebook.py --help works
	•	grep shows all three demotion tokens:
	•	DEMOTED_ULTRASHORT_STANDARD
	•	DEMOTED_MARGIN_STANDARD
	•	DEMOTED_TO_OBSERVE
	•	tiering block shows demoted_promoted variable and promoted rows removed from OBSERVE

B) Data / inputs
	•	fresh BOOKIE_IMP*_ALLMARKETS_*.csv generated from bookie_allmarkets.py
	•	expected columns present for FTR/OU25/BTTS/TG
	•	bookie_od / implied columns look sane (spot check 10 rows)

C) Deploy run
	•	run deploy_rulebook.py --debug
	•	capture console logs to file for audit
	•	confirm tier outputs written (ELITE, STANDARD, OBSERVE)

D) Tier integrity tests
	•	no cross-tier duplicates on key columns
	•	promoted FTR rows actually show lane reason tokens in STANDARD

E) Sanity checks on results
	•	row counts are not suspiciously low/high
	•	market mix looks sensible (not all one market)
	•	obvious short-price monsters (e.g., 1.03–1.10) handled as expected

⸻

12) Useful Commands (Copy/Paste Block)

# 1) CLI sanity
python deploy_rulebook.py --help >/tmp/deploy_rulebook_help.txt && tail -n 20 /tmp/deploy_rulebook_help.txt

# 2) Check patched demotion tokens
rg -n "DEMOTED_ULTRASHORT_STANDARD|DEMOTED_MARGIN_STANDARD|DEMOTED_TO_OBSERVE" deploy_rulebook.py

# 3) Inspect tiering block (FTR demote promotion logic)
sed -n '4156,4275p' deploy_rulebook.py

# 4) Check lane2 parameter references + FTR fields
rg -n "ftr_demote_lane2_|draw_chaos_score|ftr_margin|bookie_od" deploy_rulebook.py

# 5) Run deploy_rulebook (example window)
python3 deploy_rulebook.py \
  --src predictions_output/2026-02-20/BOOKIE_IMP62_ALLMARKETS_2026-02-14_to_2026-02-17.csv \
  --outdir predictions_output/2026-02-20 \
  --debug


⸻

13) Known Practical Notes / Gotchas
	•	_Naming trap_: _SEASON_LEAVE_ONE_OUT__*.csv may not always be row-level deploy rows in every workflow/export. If filter_deploy_by_rulebook.py raises "Deploy CSV must include columns: market, bookie_pick", inspect your columns and run backtest_deploy_csv.py first to produce the row-level __BACKTEST.csv.
	•	Tier counts can change between runs if source files differ, even when code is unchanged.
	•	TG markets may look incredible in some windows but are often the first to overfit / destabilize.
	•	Small-sample splits (especially by league or pick side) are useful as signals, not conclusions.
	•	Man City-style ultra-short favourites can now carry multiple promotion reason tokens (Lane 1 + Lane 2) — this is expected, not a bug.
	•	No cross-tier duplicates is a hard requirement after the demote-promotion patch.

⸻

14) Next Step for Weekend Deploy Prep (Suggested mini-plan)
	1.	Generate latest ALLMARKETS for weekend window via bookie_allmarkets.py
	2.	Run deploy_rulebook.py --debug
	3.	Save console output to a log file
	4.	Run tier duplicate check
	5.	Inspect STANDARD FTR rows for:
	•	DEMOTED_MARGIN_STANDARD
	•	DEMOTED_ULTRASHORT_STANDARD
	6.	Produce a short weekend deploy summary:
	•	counts by market
	•	counts by tier
	•	top FTR standard picks + reason tokens
	•	OU25/BTTS distribution

⸻

If you want, next I can turn this into a repo-ready final markdown version with:
	•	a tighter header
	•	a “change log” section (Patch A, Patch B, etc.)
	•	a Weekend Deploy Execution Checklist template you can reuse every Friday/Saturday.