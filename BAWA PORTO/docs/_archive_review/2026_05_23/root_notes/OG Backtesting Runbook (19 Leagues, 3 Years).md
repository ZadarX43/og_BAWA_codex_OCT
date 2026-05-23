
⸻

OG Backtesting Runbook (19 Leagues, 3 Years)

Mission goal

Validate that:
	1.	Core markets (FTR / BTTS / OU25) load STRICT V3-only
	2.	Shared resolver behavior is consistent across:
	•	pack_builder.py
	•	bookie_allmarkets.py
	3.	19-league backtests run cleanly over the target window
	4.	Output quality is stable (coverage, skips, no silent fallback weirdness)

⸻

Phase 0 — Pre-flight checklist (do this once)

Files involved (core)

You asked to include the files — these are the ones that matter in this run:

Runtime / scoring
	•	bookie_allmarkets.py ✅
	•	pack_builder.py ✅
	•	og_model_paths.py ✅ (shared resolver)
	•	_baseline_ftr_pipeline.py (goal preds / lambda fallback path used indirectly)
	•	prediction_overlay.py (optional helper paths used in runtime)
	•	train_markets.py (specialist side heads scoring if imported)
	•	streaks_module.py (team rates / streaks if used)
	•	signal_layers.py (optional signal labels)
	•	uefa_context.py (optional UEFA context)
	•	constants.py (draw thresholds etc.)

Model artifacts (core V3 only)

Under ModelStore/<LeagueTag>/:
	•	ftr_v3.pkl
	•	btts_v3.pkl
	•	over25_v3.pkl or ou25_v3.pkl (alias accepted by resolver)

Side model artifacts (V2 allowed only)

Under relevant side model dirs / bundle locations:
	•	home_fts_v2.pkl
	•	away_fts_v2.pkl
	•	home_ge2_v2.pkl
	•	away_ge2_v2.pkl
	•	home_ge3_v2.pkl
	•	away_ge3_v2.pkl

Data inputs

Whatever your current merged / league CSV sources are (examples from your stack):
	•	league CSVs used by bookie_allmarkets.py
	•	any consolidated prediction input frames
	•	odds feeds merged into final selection frame
	•	historical match CSVs for the 3-year window

⸻

Environment sanity (important)

Before running anything:
	•	Make sure you are in the project root (BAWA PORTO)
	•	Activate the same Python env you used for training/scoring
	•	Confirm imports resolve in that env

Quick terminal checks

pwd
python --version
python -c "import joblib,pandas,numpy; print('ok')"
python -c "import og_model_paths; print('resolver import ok')"


⸻

Phase 1 — Tiny debug scoring pass (prove resolver behavior)

This is the first thing to run.

Objective

Prove that:
	•	resolver is being called
	•	core markets resolve to v3
	•	OU25 aliasing works (over25_v3.pkl or ou25_v3.pkl)
	•	missing core models produce MISS + candidates
	•	no silent v2 fallback for core markets

⸻

Step 1.1 — Enable resolver debug logs

You added support for debug via:
	•	--debug
	•	env flags (OG_DEBUG_BUNDLES, OG_DEBUG_MODEL_RESOLVER)

Use both to be explicit.

macOS/Linux

export OG_DEBUG_BUNDLES=1
export OG_DEBUG_MODEL_RESOLVER=1

(If you want to disable later:)

unset OG_DEBUG_BUNDLES
unset OG_DEBUG_MODEL_RESOLVER


⸻

Step 1.2 — Run a tiny bookie_allmarkets.py window (2–3 days)

Pick a tiny date range where you know there are fixtures.

Example template

python bookie_allmarkets.py \
  --date_from 2025-01-10 \
  --date_to 2025-01-12 \
  --leagues "England Premier League,Germany Bundesliga,Champions League" \
  --debug

If your CLI uses different flags for leagues/modelstore/input paths, keep the same structure but include --debug.

If your script expects no --leagues arg and uses defaults, do:

python bookie_allmarkets.py --date_from 2025-01-10 --date_to 2025-01-12 --debug


⸻

Step 1.3 — What to look for in logs (success signatures)

✅ Expected resolver success log

You want lines like:
	•	path=.../ftr_v3.pkl
	•	alias=over25 or alias=ou25
	•	version=v3

Example pattern (not exact):
	•	[BUNDLE_RESOLVER] league=England Premier League market=ftr path=.../ftr_v3.pkl alias=ftr version=v3
	•	[BUNDLE_RESOLVER] league=... market=over25 path=.../ou25_v3.pkl alias=ou25 version=v3

✅ Expected MISS behavior (if a file is absent)

You want:
	•	MISS ... exists=False
	•	CANDIDATES ... tried=...

This proves the resolver is transparent and not silently failing.

⸻

Step 1.4 — Force one intentional MISS test (recommended)

Pick one league folder and temporarily simulate absence of a core file (example: rename only for a minute).

Example

mv "ModelStore/Scotland_Premiership/btts_v3.pkl" "ModelStore/Scotland_Premiership/btts_v3.pkl.bak"
python bookie_allmarkets.py --date_from 2025-01-10 --date_to 2025-01-10 --leagues "Scotland Premiership" --debug
mv "ModelStore/Scotland_Premiership/btts_v3.pkl.bak" "ModelStore/Scotland_Premiership/btts_v3.pkl"

✅ Pass condition

You see:
	•	MISS
	•	candidate list includes btts_v3.pkl
	•	no fallback to btts_v2.pkl

⸻

Step 1.5 — Tiny pack_builder.py debug pass (if available / used in this run)

If pack_builder.py has a debug/verbose mode, run a tiny sample there too so both scripts are validated.

Example template

python pack_builder.py \
  --score-v3 \
  --date_from 2025-01-10 \
  --date_to 2025-01-12 \
  --verbose

(Use your actual flags; the goal is to hit score_v3_models_in_df(...) and watch resolver output.)

✅ Pass condition for pack_builder
	•	Resolver logs show core markets resolved as v3 only
	•	Missing core v3 models cause skip (with candidate list)
	•	No core v2 fallback

⸻

Phase 2 — Freeze checkpoint (strongly recommended)

Once tiny debug pass is clean, checkpoint before the big run.

Git checkpoint

git status
git add og_model_paths.py pack_builder.py bookie_allmarkets.py
git commit -m "Shared bundle resolver + strict core v3 loading + resolver debug/miss logging"

If you’re not ready to commit, at least save a manual checkpoint note:
	•	date
	•	branch
	•	files changed
	•	known state = “resolver smoke test passed”

⸻

Phase 3 — 19-league / 3-year baseline backtest run

Objective

Run your baseline backtest with current stable resolver rules, before adding new UEFA post-Europe overlays.

This gives you a clean truth baseline.

⸻

Step 3.1 — Define the league set (19-league locked list)

Create a text file so you don’t keep retyping and changing it.

Suggested file

configs/backtest_19_leagues.txt

Example contents (edit to your exact 19):

	•	England Premier League
	•	England Championship
	•	England EFL League 1
	•	England FA Cup
	•	Japan J1
	•	Norway Eliteserien
	•	Netherlands Eredivisie
	•	Belgium Pro
	•	Scotland Premiership
	•	Brazil Serie A
	•	USA MLS
	•	Portugal Liga
	•	Spain La Liga
	•	Italy Serie A
	•	France Ligue 1
	•	Germany Bundesliga
	•	Europa Conference
	•	Europa League
	•	Champions League



⸻

Step 3.2 — Define the 3-year backtest window

Use exact dates and keep them stable across runs.

Example (adjust to your actual dataset coverage)

date_from = 2022-01-01
date_to   = 2025-12-31

If data starts later, use your real start date. The key is consistency.

⸻

Step 3.3 — Run the main bookie_allmarkets.py backtest

This depends on your CLI shape, but here’s a robust template.

Template command (single run)

python bookie_allmarkets.py \
  --date-from 2022-01-01 \
  --date-to 2025-12-31 \
  --leagues-file configs/backtest_19_leagues.txt \
  --debug

If your script doesn’t support --leagues-file, convert file to CSV string and pass --leagues.

Alternative (manual CSV list)

python bookie_allmarkets.py \
  --date-from 2022-01-01 \
  --date-to 2025-12-31 \
  --leagues "England Premier League,Spain La Liga,Italy Serie A,Germany Bundesliga,France Ligue 1,Portugal Liga,Netherlands Eredivisie,Belgium Pro,Scotland Premiership,Turkey Super Lig,Austria Bundesliga,Switzerland Super League,Denmark Superliga,Norway Eliteserien,Sweden Allsvenskan,USA MLS,Brazil Serie A,Champions League,Europa League"

Optional log capture (recommended)

mkdir -p run_logs
python bookie_allmarkets.py ... 2>&1 | tee run_logs/backtest_19l_3y_baseline_$(date +%Y%m%d_%H%M%S).log


⸻

Step 3.4 — Run pack_builder.py scoring pass (if part of your production backtest flow)

If your pack workflow is part of the validation path, run a mirrored window.

Template

python pack_builder.py \
  --score-v3 \
  --date_from 2022-01-01 \
  --date_to 2025-12-31 \
  --leagues-file configs/backtest_19_leagues.txt \
  --verbose

Again: use your real CLI flags. The point is to validate scoring and skips under the same window.

⸻

Phase 4 — Outputs to capture (don’t skip this)

This is where a lot of time gets lost later. Capture these every run.

1) Raw log file

Save:
	•	run_logs/backtest_19l_3y_baseline_<timestamp>.log

This is your forensic source for:
	•	resolver behavior
	•	skips
	•	missing bundles
	•	weird alias resolutions

⸻

2) Output CSV(s)

From bookie_allmarkets.py, capture the generated file path, typically something like:
	•	predictions_output/<today>/BOOKIE_IMP{...}_ALLMARKETS_<date_from>_to_<date_to>.csv

Copy/rename to a stable experiment artifact name if needed:

cp "predictions_output/.../BOOKIE_....csv" "experiments/backtests/19l_3y_baseline_allmarkets.csv"


⸻

3) League-level summary (must capture)

Create a quick summary table from output CSV:
	•	rows per league
	•	rows per market
	•	non-null rate of key columns
	•	skipped leagues count (if visible in logs)
	•	coverage by market

If you want, I can give you a tiny Python script next for this summary, but for now the key fields are:

Minimum summary fields to compute
	•	league
	•	market
	•	n_rows
	•	n_unique_fixture_key
	•	% non-null model_p_for_bookie
	•	% non-null bookie_implied_novig
	•	% non-null prob_over25
	•	% non-null prob_btts
	•	% non-null ftr_margin (FTR only)

⸻

4) Resolver audit extract (optional but powerful)

Grep the log for resolver lines so you can inspect only path loading.

Example

grep "BUNDLE_RESOLVER" run_logs/backtest_19l_3y_baseline_*.log > run_logs/resolver_audit_19l_3y.txt

Then spot-check:
	•	no core *_v2.pkl
	•	alias behavior correct
	•	MISS lines explain skips

⸻

Phase 5 — Pass / Fail checks (explicit)

A. Resolver behavior (hard pass/fail)

✅ PASS if ALL true
	•	Core markets only resolve to v3
	•	OU25 resolves via over25_v3.pkl or ou25_v3.pkl
	•	MISS lines include candidate list
	•	No core v2 file paths appear in resolver logs

❌ FAIL if ANY occur
	•	ftr_v2.pkl, btts_v2.pkl, over25_v2.pkl, ou25_v2.pkl used for core markets
	•	Missing model causes silent skip with no debug trail
	•	Different scripts resolve different bundle names for same league/market

⸻

B. Coverage sanity (run-level)

✅ PASS (baseline sanity)
	•	Majority of target leagues produce output rows for all core markets
	•	No large unexplained league drops
	•	FTR rows have non-null ftr_margin and model_p_for_bookie
	•	OU25 rows have p_over25_novig / p_under25_novig
	•	BTTS rows have prob_btts (or canonical backfilled value)

❌ FAIL signals
	•	One or more major leagues produce zero rows unexpectedly
	•	Many rows have missing core confidence columns
	•	OU25/BTTS canonical columns all-NA after export
	•	Huge difference vs prior row counts without code/data reason

⸻

C. Strict QC (if you enable _run_strict_qc_asserts)

If your script already calls the strict QC helper in this run path:

✅ PASS
	•	OU25 novig probs sum to ~1
	•	BTTS required columns non-null
	•	FTR required columns non-null
	•	“ALL ASSERTS PASSED ✅”

❌ FAIL
	•	Any strict QC exception / SystemExit

⸻

Phase 6 — Fast failure triage (what to do if it breaks)

Problem 1: Core market missing for a league

Symptoms
	•	resolver MISS
	•	skip in scoring
	•	fewer rows than expected

Check
	•	ModelStore/<LeagueTag>/ contains:
	•	ftr_v3.pkl
	•	btts_v3.pkl
	•	over25_v3.pkl or ou25_v3.pkl

Fix
	•	retrain/re-copy missing V3 bundle
	•	confirm filename exactly matches resolver expectations

⸻

Problem 2: OU25 weirdness (alias mismatch)

Symptoms
	•	O25 MISS but file exists
	•	file named differently (ou25_v3.pkl vs over25_v3.pkl)

Check
	•	resolver candidates printed in MISS log
	•	actual filename in folder

Fix
	•	ensure og_model_paths.py alias list includes both
	•	normalize filenames if needed (or keep alias support)

⸻

Problem 3: Silent different behavior between pack_builder and bookie_allmarkets

Symptoms
	•	one script loads, other skips
	•	inconsistent output by league

Check
	•	both import resolve_market_bundle_path from og_model_paths.py
	•	no leftover duplicate local resolver in either script
	•	compare resolver logs side-by-side for same league

Fix
	•	remove duplicate resolver code
	•	use shared resolver only

⸻

Problem 4: Backtest window returns no rows for some leagues

Symptoms
	•	zero rows after filter
	•	known fixtures missing

Check
	•	date parsing / match_date vs date_GMT
	•	_coalesce_match_date_series behavior
	•	source CSV actually contains rows in date window

Fix
	•	inspect source file headers/date format
	•	test _count_rows_in_window(...) behavior on one affected file
	•	confirm timezone/date parsing isn’t shifting out-of-window

⸻

Phase 7 — Recommended run sequence (exact order)

Run sequence (copy/paste checklist)

1) Tiny resolver smoke test
	•	Export debug env flags
	•	Run bookie_allmarkets.py for 2–3 days / 2–3 leagues
	•	Confirm v3-only core loads
	•	Confirm OU25 alias logs
	•	Confirm MISS candidate list works (intentional test)

2) pack_builder smoke test
	•	Run tiny pack_builder.py --score-v3 window
	•	Confirm same resolver behavior

3) Checkpoint commit
	•	Commit resolver hardening changes

4) Full 19-league / 3-year baseline run
	•	Run bookie_allmarkets.py
	•	Capture full logs with tee
	•	Save/copy output CSV

5) Post-run QA
	•	Extract resolver audit lines
	•	Build league/market summary
	•	Check row coverage and key columns
	•	Record baseline metrics in notes

⸻

Tiny debug scoring pass (proving resolver behavior) — copy/paste version

Use this as your immediate first action.

A) Turn on debug flags

export OG_DEBUG_BUNDLES=1
export OG_DEBUG_MODEL_RESOLVER=1

B) Run a tiny window (bookie_allmarkets)

python bookie_allmarkets.py \
  --date-from 2025-01-10 \
  --date-to 2025-01-12 \
  --leagues "England Premier League,Germany Bundesliga,Champions League" \
  --debug 2>&1 | tee run_logs/tiny_resolver_smoke_$(date +%Y%m%d_%H%M%S).log

C) What to confirm in the log

Search for:
	•	BUNDLE_RESOLVER
	•	version=v3
	•	alias=ou25 or alias=over25
	•	MISS
	•	CANDIDATES

Quick grep

grep "BUNDLE_RESOLVER" run_logs/tiny_resolver_smoke_*.log

D) Intentional MISS test (optional but strongly recommended)

mv "ModelStore/Scotland_Premiership/btts_v3.pkl" "ModelStore/Scotland_Premiership/btts_v3.pkl.bak"
python bookie_allmarkets.py \
  --date-from 2025-01-10 \
  --date-to 2025-01-10 \
  --leagues "Scotland Premiership" \
  --debug 2>&1 | tee run_logs/tiny_resolver_miss_test_$(date +%Y%m%d_%H%M%S).log
mv "ModelStore/Scotland_Premiership/btts_v3.pkl.bak" "ModelStore/Scotland_Premiership/btts_v3.pkl"

PASS criteria for this test
	•	MISS printed for BTTS
	•	candidate list printed
	•	no core v2 fallback path shown

⸻

Notes for your “don’t lose the thread” workflow

Create one run note file per experiment

Suggested:
	•	experiments/backtests/19l_3y_baseline_RUN_NOTES.md

Template:

# 19L 3Y Baseline Backtest Run Notes

## Run ID
2026-02-25_baseline_19l_3y_v3core

## Code checkpoint
<git commit hash>

## Window
2022-01-01 to 2025-12-31

## Leagues
<locked 19-league list>

## Resolver policy
Core markets strict v3 only; OU25 alias support (over25/ou25); no core v2 fallback.

## Smoke test status
PASS / FAIL

## Full run status
PASS / FAIL

## Key observations
- ...
- ...

## Issues found
- ...

This will save you a lot of mental load later.

⸻

If you want next, I can give you a compact post-run audit script (Python) that reads the ALLMARKETS CSV + log and outputs:
	•	league/market coverage
	•	missing core columns by market
	•	resolver v2 path violations (if any)
	•	top skip reasons from logs


Below is a compact post-run audit script you can drop into your repo (e.g. audit_backtest_run.py) and run against:
	•	your ALLMARKETS CSV
	•	your run log (tee output)

It will output:
	•	league/market coverage
	•	missing core columns by market
	•	resolver v2 path violations (if any)
	•	top skip reasons from logs

⸻

audit_backtest_run.py


How to run it

Minimal run

python audit_backtest_run.py \
  --csv "experiments/backtests/19l_3y_baseline_allmarkets.csv" \
  --log "run_logs/backtest_19l_3y_baseline_20260225_170500.log"

With output files saved

python audit_backtest_run.py \
  --csv "experiments/backtests/19l_3y_baseline_allmarkets.csv" \
  --log "run_logs/backtest_19l_3y_baseline_20260225_170500.log" \
  --outdir "experiments/backtests/audit_19l_3y_baseline"

What “good” looks like (quick read)

✅ Resolver V2 violations

Should say:
	•	✅ None found.

If not, it will print the offending log lines (e.g. ftr_v2.pkl, btts_v2.pkl, ou25_v2.pkl etc.).

⸻

✅ Coverage table

You should see rows grouped by:
	•	league
	•	_market_norm (FTR, BTTS, OU25)

with sensible:
	•	n_rows
	•	n_unique_fixture_key
	•	non-null percentages

⸻

✅ Missing core columns by market

For each market:
	•	col_present::<column> should be True for expected columns
	•	pct_non_null::<column> should be high (ideally ~100% on required fields)

⸻

✅ Top skip reasons

Useful for immediate triage:
	•	resolver MISS
	•	missing manifest/core V3 models
	•	skip missing (generic)
	•	error/exception
