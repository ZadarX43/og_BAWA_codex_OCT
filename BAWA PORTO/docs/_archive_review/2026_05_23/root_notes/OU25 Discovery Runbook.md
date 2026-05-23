OU25 Discovery Runbook

Purpose

This document defines the permanent discovery, backtesting, threshold-sweep, walk-forward, and audit process for OU2.5 product development inside OG.

It mirrors the structure now proven on FTR:
	•	candidate product branches
	•	threshold sweeps
	•	fixed output folders
	•	required CSV artifacts
	•	audit requirements
	•	branch comparison and cumulative stats
	•	investor-facing positioning

The goal is not just to find one profitable OU25 setting. The goal is to establish distinct, auditable OU25 products that can sit alongside the now-locked FTR products.

⸻

1) Mission Goal

Establish OU25 in the same productized way FTR has now been established.

Target outcome:
	•	OU25 Accuracy Lane
	•	OU25 Value / Edge Lane
	•	month-by-month walk-forward validation
	•	forensic audit parity with FTR
	•	investor-ready summary tables

This is discovery work, not rescue work.

⸻

2) Product Framing

Product A — OU25 Accuracy

Positioning:
	•	higher strike-rate OU25 product
	•	tighter odds structure
	•	stable month-to-month
	•	suitable for “safer” investor/product narrative

Primary characteristics expected:
	•	higher hit rate
	•	lower average odds
	•	positive but more modest ROI
	•	lower variance than value-oriented OU25

Product B — OU25 Value / Edge

Positioning:
	•	more selective OU25 product
	•	stronger ROI / edge emphasis
	•	likely requires ranking or edge proxy, not just simple odds banding

Primary characteristics expected:
	•	lower hit rate than accuracy lane, but still robust
	•	higher average odds than accuracy lane
	•	materially stronger ROI if valid
	•	more variance, but still stable enough to defend

⸻

3) Canonical Corpus Policy

For product-level OU25 discovery, use the same canonical corpus logic already proven for FTR.

Canonical product corpus
	•	IMP40 row-level backtest CSV
	•	used when broad market coverage / value-oriented universes are needed
	•	primary corpus for OU25 discovery unless testing shows otherwise

Path pattern:
	•	predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_<date_from>_to_<date_to>__BACKTEST.csv

Fast regression corpus
	•	IMP68 row-level backtest CSV
	•	smaller, faster subset for quick checks / stability checks
	•	not assumed to be the canonical OU25 product corpus unless evidence supports it

Path pattern:
	•	predictions_output/backtests/19l_3y_baseline/BOOKIE_IMP68_ALLMARKETS_<date_from>_to_<date_to>__BACKTEST.csv

Working rule
	•	Use IMP40 as the default discovery universe for OU25.
	•	Use IMP68 only for fast sanity/regression tests unless the sweep evidence says an accuracy-style OU25 lane is better on a tighter upstream universe.

⸻

4) Core OU25 Candidate Branches

We should not assume one single OU25 gate family is correct. Discovery should test several candidate branch families.

Branch OU25-A — Frozen Band Accuracy

Core idea:
	•	odds-band-based OU25 selection
	•	minimal complexity
	•	mirrors current frozen gate style already present in apply_frozen_product_gates.py

Candidate shape:
	•	odds band 1 + odds band 2
	•	optional top-q ranking trim
	•	combined Over/Under unless evidence suggests split needed

Likely first test family:
	•	band1 low/high around current range
	•	band2 low/high around current range
	•	rank by score if present
	•	fallback rank by model_p_for_bookie

Branch OU25-B — Band + Rank Accuracy

Core idea:
	•	same odds banding, but adds stronger quantile trim
	•	attempts to isolate more stable, higher-confidence rows within the banded universe

Candidate shape:
	•	fixed odds bands
	•	sweep top-q
	•	compare no rank trim vs ranked trim

Branch OU25-C — Over/Under Split Accuracy

Core idea:
	•	treat Over 2.5 and Under 2.5 separately
	•	one side may materially outperform the other

Candidate shape:
	•	OVER-only band tests
	•	UNDER-only band tests
	•	combined comparison after side-level validation

Branch OU25-D — Value / Edge OU25

Core idea:
	•	not purely odds-band selection
	•	uses ranking or edge-like proxy
	•	may require score, model_p_for_bookie, implied-vs-model gap, or related proxy

Candidate shape:
	•	minimum odds floor or target odds band
	•	rank by score / probability / edge proxy
	•	top-q trim
	•	possibly separate OVER and UNDER

⸻

5) Threshold Families To Sweep

This is the permanent first discovery sweep set for OU25.

5.1 Odds band sweep

Current frozen reference:
	•	band1: 1.24–1.72
	•	band2: 1.82–1.91

Initial sweep family:

Band 1 candidates
	•	1.24–1.68
	•	1.24–1.72
	•	1.24–1.76
	•	1.28–1.72
	•	1.30–1.75

Band 2 candidates
	•	1.80–1.90
	•	1.82–1.91
	•	1.84–1.92
	•	1.85–1.95

Goal:
	•	identify which band windows produce the best balance of rows / hit / ROI / avg odds

5.2 Rank trim sweep

Apply to each candidate band family:
	•	no top-q trim
	•	top_q = 0.60
	•	top_q = 0.70
	•	top_q = 0.75
	•	top_q = 0.80

Goal:
	•	determine whether ranked trimming materially improves stability or just reduces row count without enough benefit

5.3 Side split sweep

For each branch family, test:
	•	combined OU25
	•	OVER-only
	•	UNDER-only

Goal:
	•	determine whether one side is carrying the other

5.4 Optional odds-floor / odds-cap sweep for value lane

If OU25 Value lane is tested separately, candidate sweeps may include:
	•	minimum odds floor
	•	tighter value-oriented odds windows
	•	score-rank-only selection within broader odds universe

This should not be introduced until the simpler band sweeps are completed.

⸻

6) Required Files Involved

Runtime / generation files
	•	bookie_allmarkets.py
	•	pack_builder.py
	•	og_model_paths.py
	•	prediction_overlay.py
	•	train_markets.py
	•	signal_layers.py
	•	constants.py
	•	uefa_context.py (if active in current pipeline)

Backtest / scoring / filtering files
	•	backtest_deploy_csv.py
	•	apply_frozen_product_gates.py
	•	build_branch_comparison.py
	•	build_branch_cumulative_stats.py
	•	forensic_walkforward_audit.py
	•	poisson_source_audit.py (less central to OU25 than FTR, but still useful for pipeline sanity)

Optional runner files to create for OU25
	•	build_ou25_validation_backtests_3y19l.sh
	•	run_ou25_frozen_sweeps.sh
	•	parse_ou25_sweep_results.py
	•	build_ou25_branch_comparison.py
	•	build_ou25_branch_cumulative_stats.py
	•	forensic_ou25_walkforward_audit.py

⸻

7) Output Folder Convention

Use dedicated branch roots exactly like the FTR branch layout.

Accuracy branch root
	•	walkforward_frozen_ou25_accuracy/

Value branch root
	•	walkforward_frozen_ou25_value/

Optional side-specific roots if needed later
	•	walkforward_frozen_ou25_over_accuracy/
	•	walkforward_frozen_ou25_under_accuracy/
	•	walkforward_frozen_ou25_over_value/
	•	walkforward_frozen_ou25_under_value/

Month folder pattern
	•	walkforward_frozen_ou25_accuracy/YYYY-MM/
	•	walkforward_frozen_ou25_value/YYYY-MM/

Required monthly outputs

For each branch/month folder, require:
	•	backtest_YYYY-MM.csv
	•	backtest_unscored_YYYY-MM.csv
	•	backtest_summary_YYYY-MM.csv
	•	frozen_gated_YYYY-MM.csv
	•	frozen_gated_YYYY-MM.md
	•	frozen_tier_elite_YYYY-MM.csv
	•	frozen_tier_standard_YYYY-MM.csv
	•	frozen_tier_observe_YYYY-MM.csv
	•	summary_YYYY-MM.json

If running direct source-level gate outputs in addition, also preserve the raw source-tagged artifacts:
	•	BOOKIE_IMPxx_ALLMARKETS_...__BACKTEST__OU25_<PROFILE>.csv
	•	...__SUMMARY.json
	•	...__MARKET_SUMMARY.csv
	•	...__LEAGUE_SUMMARY.csv

⸻

8) Minimum Required CSV Artifacts

For a branch to count as established, the following must exist.

A. Monthly branch comparison source

Each month must produce summary stats with:
	•	month
	•	rows
	•	hit
	•	roi
	•	avg_odds

B. Master branch comparison CSV

Required output:
	•	walkforward_ou25_branch_comparison_<window>.csv

Required columns:
	•	month
	•	branch
	•	rows
	•	hit
	•	roi
	•	avg_odds

C. Cumulative branch stats CSV

Required output:
	•	walkforward_ou25_branch_cumulative_stats_<window>.csv

Required columns:
	•	branch
	•	months_present
	•	total_rows
	•	weighted_hit
	•	weighted_roi
	•	weighted_avg_odds
	•	best_month_by_hit
	•	best_hit
	•	worst_month_by_hit
	•	worst_hit
	•	best_month_by_roi
	•	best_roi
	•	worst_month_by_roi
	•	worst_roi
	•	hit_std
	•	roi_std

D. Forensic audit CSV

Required output:
	•	walkforward_ou25_forensic_audit.csv

E. Spotcheck sample CSV

Required output:
	•	walkforward_ou25_fixture_spotcheck_samples.csv

⸻

9) Discovery Workflow

Phase 1 — Build corpus

Build or confirm the row-level backtest corpus exists:
	•	BOOKIE_IMP40_ALLMARKETS_...__BACKTEST.csv
	•	optionally IMP68 quick regression corpus

Requirement:
	•	row-level backtest only
	•	must include market, bookie_pick, score, correct

Phase 2 — Run threshold sweeps

For each candidate branch:
	•	odds bands
	•	top-q
	•	side split (combined / over-only / under-only)

Track results into a sweep comparison table.

Phase 3 — Select 1–2 finalist branches

Finalists should be selected based on:
	•	repeatability
	•	stability
	•	not just one explosive ROI month
	•	adequate row coverage
	•	auditability

Phase 4 — Freeze walk-forward branch test

Run month-by-month frozen walk-forward for at least:
	•	2024-10
	•	2024-11
	•	2024-12
	•	2025-01
	•	2025-02
	•	2025-03

Phase 5 — Build branch comparison and cumulative stats

Outputs:
	•	month-by-month branch table
	•	cumulative branch stats
	•	best/worst month summary

Phase 6 — Run forensic audit

Same standard as FTR.

Phase 7 — Lock investor lane summary

Once stable:
	•	OU25 Accuracy Lane one-pager
	•	OU25 Value Lane one-pager

⸻

10) Audit Requirements

OU25 must meet audit parity with FTR before being called established.

Required audit checks

A. Fixture join cleanliness

Confirm:
	•	filtered rows exist in backtest corpus
	•	no merge misses
	•	no duplicate join rows
	•	no duplicate filtered fixture rows

B. Filtering vs presence distinction

Audit must distinguish:
	•	columns merely present in filtered exports
	•	columns actually used in filtering/ranking logic

Required audit columns:
	•	filtering_columns_checked
	•	ranking_columns_checked
	•	filtering_columns_present_in_filtered
	•	ranking_columns_present_in_filtered
	•	selection_columns_present_in_filtered
	•	selection_leak_suspected
	•	postmatch_selection_columns_present
	•	postmatch_cols_present_but_not_used

C. Post-match leakage audit

Allowed:
	•	post-match columns present in filtered exports for scoring and audit only

Not allowed:
	•	evidence they were used in filtering/ranking

D. Spotcheck samples

Inspect random winners and losers by month / branch.

Minimum fields:
	•	month
	•	branch
	•	sample_type
	•	league
	•	match_date
	•	teams
	•	market
	•	bookie_pick
	•	bookie_od
	•	score / model field used for rank
	•	OU25 probability fields if present
	•	correct

E. Market-specific sanity

For OU25-specific audit, additionally inspect whether:
	•	over/under rows are distributed sensibly
	•	band definitions are actually selecting intended price zones
	•	one side is not silently dominating the branch without being acknowledged

⸻

11) OU25-Specific Audit Questions

These must be answered before OU25 is considered locked.
	1.	Were the winning months driven mainly by OVER or UNDER?
	2.	Does combined OU25 hide one weak side and one strong side?
	3.	Were any post-match columns used in selection logic? (must be no)
	4.	Are fixture joins clean? (must be yes)
	5.	Are row counts sufficient to defend the product? (must not be tiny sample theatre)
	6.	Does ranked trimming genuinely improve ROI / hit, or merely over-concentrate the sample?
	7.	Is IMP40 truly the best corpus for OU25, or does a tighter upstream universe outperform it for the accuracy lane?

⸻

12) Success Criteria

OU25 branch is considered viable if:
	•	positive ROI across the walk-forward window
	•	hit rate materially above naive market baseline
	•	no forensic red flags
	•	month-to-month stability acceptable
	•	row count large enough to matter
	•	investor positioning is explainable in plain English

Accuracy lane preferred target
	•	80%+ hit rate
	•	positive ROI
	•	moderate odds
	•	lower variance

Value lane preferred target
	•	lower hit than accuracy lane is acceptable
	•	materially stronger ROI
	•	higher avg odds than accuracy lane
	•	still stable enough month-to-month to defend

⸻

13) What We Need To Prove

At the end of OU25 discovery, we need to be able to say:
	•	OU25 is not just “present at scale”; it is productized.
	•	We have identified at least one robust accuracy-style OU25 lane.
	•	We have tested whether a distinct value-style OU25 lane exists.
	•	The selected lane(s) hold up across multiple months, not just one sweet spot.
	•	The filtered outputs are audit-clean.
	•	The product can be described clearly to investors.

⸻

14) Permanent Deliverables To Produce

Once discovery is complete, lock the following:
	1.	OU25_DISCOVERY_RUNBOOK.md
	2.	walkforward_ou25_branch_comparison_<window>.csv
	3.	walkforward_ou25_branch_cumulative_stats_<window>.csv
	4.	walkforward_ou25_forensic_audit.csv
	5.	walkforward_ou25_fixture_spotcheck_samples.csv
	6.	OU25_INVESTOR_SUMMARY.md
	7.	OU25_AUDIT_INTERPRETATION_AND_CAVEATS.md

⸻

15) Recommended Immediate Next Steps

Step 1

Patch or extend apply_frozen_product_gates.py so OU25 branch tagging is explicit enough for discovery runs.

Step 2

Create a dedicated OU25 branch comparison builder mirroring the FTR scripts.

Step 3

Run the first OU25 sweep grid on the canonical IMP40 corpus.

Step 4

Compare:
	•	combined OU25
	•	OVER-only
	•	UNDER-only
	•	ranked vs non-ranked

Step 5

Pick finalist OU25 branch families and run frozen walk-forward month-by-month.

⸻

16) Strategic Note

FTR is now locked.

The next stage is not “prove the model works.”
The next stage is:
	•	establish OU25
	•	establish BTTS
	•	unify them into a multi-product OG portfolio
	•	then extend the walk-forward horizon and investor narrative across all established markets

That means OU25 discovery is now one of the highest-value tasks in the project.