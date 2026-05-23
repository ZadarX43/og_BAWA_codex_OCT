Perfect. Here’s the one-page stack map with your core chain, plus where the new guard-rail files sit.

OG Stack Map

What runs the system vs what keeps it safe

                           ┌──────────────────────────────┐
                           │        og_model_paths.py     │
                           │  (shared model bundle resolver)
                           │  - exact path rules          │
                           │  - alias support (over25/ou25)
                           │  - strict core V3 policy     │
                           └──────────────┬───────────────┘
                                          │
                                          │ imported by
                                          ▼
┌──────────────────────┐        ┌──────────────────────┐
│  bookie_allmarkets.py│        │    pack_builder.py   │   (optional pack flow)
│  Build ALLMARKETS    │        │ Score/pack from      │
│  long-format rows    │        │ ALLMARKETS / spine   │
│  (ftr/ou25/btts/...) │        │ (uses same resolver) │
└──────────┬───────────┘        └──────────────────────┘
           │
           │ outputs
           ▼
┌──────────────────────────────────────────────────────────┐
│ BOOKIE_IMP*_ALLMARKETS_<date_from>_to_<date_to>.csv      │
│ (candidate pool / scored market rows)                    │
└──────────┬───────────────────────────────────────────────┘
           │
           ├──────────────────────────────► deploy_rulebook.py
           │                                (live/deploy gating + tiers)
           │                                - runtime gates
           │                                - market-specific gates
           │                                - tiering (ELITE/STANDARD/OBSERVE)
           │                                - FTR demotion/re-promotion patch
           │
           │                                outputs:
           │                                __DEPLOY_PRESET_V1.csv/.md
           │                                __DEPLOY_TIER_ELITE.csv
           │                                __DEPLOY_TIER_STANDARD.csv
           │                                __DEPLOY_TIER_OBSERVE.csv
           │
           ▼
┌──────────────────────┐
│ backtest_deploy_csv.py│
│ truth join + scoring  │
│ (correct / hit / ROI) │
└──────────┬───────────┘
           │ outputs
           ▼
┌──────────────────────────────────────────────┐
│ ...__BACKTEST.csv        (row-level, usable) │
│ ...__BACKTEST_SUMMARY.csv (aggregate only)   │
└──────────┬───────────────────────────────────┘
           │
           │ row-level only
           ▼
┌──────────────────────────────┐
│ filter_deploy_by_rulebook.py │
│ historical rulebook gating   │
│ for comparisons              │
│ (e.g. FTR value vs accuracy) │
└──────────┬───────────────────┘
           │ outputs
           ▼
┌──────────────────────────────────────────────┐
│ __GATED.csv                                  │
│ __GATED_SUMMARY.csv                          │
│ __RULEBOOK_SNAPSHOT.json                     │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────┐
│ Analysis / Portfolio / Stability Layer                    │
│  - aggregate_gated_backtests.py                           │
│  - seasonal_market_ledger.py                              │
│  - season_leave_one_out*.py                               │
│  - parse_rulebook_ftr_sweeps.py (if running FTR sweeps)   │
└────────────────────────────────────────────────────────────┘


⸻

Where the new files sit (guard rails + diagnostics)

og_model_paths.py = guard rail before scoring

It sits upstream, inside loading/scoring scripts.

It makes sure:
	•	core bundles resolve consistently
	•	alias rules match everywhere
	•	strict V3-only core policy is enforced (when configured)

Think of it as:
	•	“How do we load models correctly?”

⸻

audit_backtest_run.py = diagnostic after a run

It sits after bookie_allmarkets.py (and can be used after major backtest runs).

It checks:
	•	coverage by league/market
	•	missing columns / null rates
	•	resolver path violations (v2 core leaks)
	•	skip reasons from logs

Think of it as:
	•	“Can we trust this run’s output before analyzing performance?”

⸻

Super simple flow by purpose

A) Production / deploy path

bookie_allmarkets.py -> deploy_rulebook.py -> tier outputs

B) Historical performance path

(bookie/deploy output) -> backtest_deploy_csv.py -> __BACKTEST.csv

C) Historical gating comparison path (research)

__BACKTEST.csv -> filter_deploy_by_rulebook.py -> __GATED outputs

D) Portfolio / stability analysis path

__GATED summaries -> aggregate / seasonal / leave-one-out scripts

E) Guard rails

og_model_paths.py (during loading) + audit_backtest_run.py (after run)


⸻

Your remaining files and where they fit

You listed these — here’s exactly where they belong:

Build / orchestration scripts

build_months.sh
	•	Batch runner for monthly backtests / builds
	•	Sits above the chain (automation wrapper)

build_months_2022_2024.sh
	•	Historical batch runner for a fixed long period
	•	Same role, just a historical campaign wrapper

build_ftr_validation_backtests_3y19l.sh
	•	Campaign builder for the 3-year / 19-league FTR validation
	•	Likely orchestrates repeated backtest_deploy_csv.py runs (and maybe sweeps pipeline prep)
	•	Sits above the research chain as a campaign script

⸻

Core scoring / truth

backtest_deploy_csv.py
	•	Core truth join + scoring engine
	•	Converts generated rows into measurable backtests (correct, hit, ROI)

⸻

Portfolio / research diagnostics

seasonal_market_ledger.py
	•	Market performance ledger over seasons
	•	Helps identify stable/fragile markets

season_leave_one_out__2024_2025.py

season_leave_one_out_2022_2023.py

season_leave_one_out_2023_2024.py
	•	Market ablation by season
	•	“What happens if we remove FTR / TG / BTTS / OU25 from the portfolio?”
	•	These are portfolio composition tools, not core prediction scorers

⸻

Important distinction (the confusion point you flagged earlier)

Why audit doesn’t show “FTR accuracy per league”

Because audit_backtest_run.py is checking:
	•	structure
	•	coverage
	•	resolver correctness
	•	missing data

It is not checking truth outcomes.

FTR accuracy per league comes from:
	•	backtest_deploy_csv.py outputs (row-level correct)
	•	then a summary/grouping script on market == 'ftr' by league

So:
	•	audit_backtest_run.py = “Was the run valid?”
	•	backtest_deploy_csv.py (+ summaries) = “Was the model accurate/profitable?”

⸻

One-line cheat sheet you can keep in your head
	•	Generate: bookie_allmarkets.py
	•	Gate live: deploy_rulebook.py
	•	Score truth: backtest_deploy_csv.py
	•	Compare historical rulebooks: filter_deploy_by_rulebook.py
	•	Analyze stability: seasonal_market_ledger.py + season_leave_one_out*.py
	•	Keep loading sane: og_model_paths.py
	•	Audit run integrity: audit_backtest_run.py

⸻

