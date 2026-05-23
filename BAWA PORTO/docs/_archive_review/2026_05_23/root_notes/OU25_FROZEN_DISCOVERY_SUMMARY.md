Paste this into a repo file like OU25_FROZEN_DISCOVERY_SUMMARY.md.

OG OU25 Frozen Discovery Summary

Purpose

This document locks the current OU2.5 frozen discovery result into a permanent written record.

It captures:
	•	the mission of the OU25 discovery phase
	•	the command chain used
	•	smoke-test results
	•	forensic audit results
	•	the current OU25 branch leaderboard
	•	interpretation and caveats
	•	the current recommended next position

This is the OU25 counterpart to the locked FTR summary. It establishes that OU25 is now a functioning, audited product lane within the frozen walk-forward framework.

⸻

Mission

The OU25 discovery phase was designed to answer the following:
	1.	Can OU25 be run as a standalone frozen product lane from the canonical scored backtest corpus?
	2.	Can we sweep:
	•	combined over/under selection
	•	over-only selection
	•	under-only selection
	•	odds-band variants
	•	top-q ranking trims
	3.	Can we produce:
	•	repeatable branch outputs
	•	branch comparison tables
	•	cumulative ranking tables
	•	forensic audit outputs
	4.	Can we identify a provisional OU25 leaderboard suitable for product positioning and later investor packaging?

The answer is now yes.

⸻

Canonical Source Corpus

The canonical source used for OU25 discovery is:

predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv

This remains the main truth-backed product corpus because it contains:
	•	FTR rows at scale
	•	BTTS rows at scale
	•	OU25 rows at scale
	•	row-level scoring columns including correct, score, and bookie_od

For OU25 specifically, the market inspection confirmed:
	•	market = ou25
	•	total OU25 rows in canonical corpus: 19,923
	•	bookie_pick values:
	•	OVER25: 13,549
	•	UNDER25: 6,374

⸻

Core Files Used

Product gate layer

apply_frozen_product_gates.py

Used to apply frozen learned product gates directly to the canonical scored backtest CSV.

Key OU25 additions now supported:
	•	--include-ou25
	•	--ou25-only
	•	--ou25-pick-mode {combined,over_only,under_only}
	•	--ou25-band1-low
	•	--ou25-band1-high
	•	--ou25-band2-low
	•	--ou25-band2-high
	•	--top-q
	•	--tag

Sweep runner

run_ou25_frozen_sweeps.sh

Used to run the full OU25 frozen sweep batch across the canonical backtest corpus.

Branch comparison

build_ou25_branch_comparison.py

Used to gather all branch outputs and produce a sortable comparison table.

Outputs:
	•	ou25_branch_comparison.csv
	•	ou25_branch_comparison.md

Forensic audit

forensic_ou25_audit.py

Used to verify:
	•	filtered rows map cleanly back to the canonical scored backtest
	•	post-match columns are present only for scoring/audit
	•	no evidence of selection leakage
	•	joins are clean

Outputs:
	•	ou25_forensic_audit.csv
	•	ou25_fixture_spotcheck_samples.csv

Cumulative ranking

build_ou25_cumulative_stats.py

Used to compute branch-level cumulative ranking statistics from the branch comparison file.

Output:
	•	ou25_branch_cumulative_stats.csv

⸻

Command Chain Used

1. Help / CLI validation

python apply_frozen_product_gates.py --help | rg "ou25-only|ou25-pick-mode|include-ou25|ou25-band1-low|ou25-band2-high"

Confirmed the OU25 flags were live and correctly wired.

⸻

2. Baseline OU25 combined smoke test

python apply_frozen_product_gates.py \
  --src predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv \
  --outdir predictions_output/ou25_smoke/combined_baseline \
  --include-ou25 \
  --ou25-only \
  --ou25-pick-mode combined \
  --ou25-band1-low 1.24 \
  --ou25-band1-high 1.72 \
  --ou25-band2-low 1.82 \
  --ou25-band2-high 1.91 \
  --top-q 0.70 \
  --tag OU25_COMBINED_BASELINE

Result:
	•	rows: 4288
	•	hit: 0.814832
	•	ROI: 0.405690
	•	avg odds: 1.738406

⸻

3. OVER-only smoke test

python apply_frozen_product_gates.py \
  --src predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv \
  --outdir predictions_output/ou25_smoke/over_only \
  --include-ou25 \
  --ou25-only \
  --ou25-pick-mode over_only \
  --ou25-band1-low 1.24 \
  --ou25-band1-high 1.72 \
  --ou25-band2-low 1.82 \
  --ou25-band2-high 1.91 \
  --top-q 0.70 \
  --tag OU25_OVER_ONLY

Result:
	•	rows: 2946
	•	hit: 0.828921
	•	ROI: 0.428578
	•	avg odds: 1.736130

⸻

4. UNDER-only smoke test

python apply_frozen_product_gates.py \
  --src predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv \
  --outdir predictions_output/ou25_smoke/under_only \
  --include-ou25 \
  --ou25-only \
  --ou25-pick-mode under_only \
  --ou25-band1-low 1.24 \
  --ou25-band1-high 1.72 \
  --ou25-band2-low 1.82 \
  --ou25-band2-high 1.91 \
  --top-q 0.70 \
  --tag OU25_UNDER_ONLY

Result:
	•	rows: 1347
	•	hit: 0.789161
	•	ROI: 0.372530
	•	avg odds: 1.750747

⸻

5. Full OU25 frozen sweep batch

bash run_ou25_frozen_sweeps.sh

Confirmed final output root:

predictions_output/ou25_frozen_compare/rulebook_ftr_validation_3yr_19lg_v1

⸻

6. Branch comparison build

python build_ou25_branch_comparison.py

Outputs written:
	•	ou25_branch_comparison.csv
	•	ou25_branch_comparison.md

⸻

7. Forensic OU25 audit

python forensic_ou25_audit.py

Outputs written:
	•	ou25_forensic_audit.csv
	•	ou25_fixture_spotcheck_samples.csv

⸻

8. Cumulative branch stats

python build_ou25_cumulative_stats.py

Output written:
	•	ou25_branch_cumulative_stats.csv

⸻

Smoke Tests Passed

Market label validation

The canonical backtest correctly contains OU25 rows:
	•	market = ou25
	•	OU25 rows present: 19,923

Pick label validation

OU25 pick labels were confirmed present and usable:
	•	OVER25
	•	UNDER25

The OU25 standalone gate initially returned zero rows until the pick normalization issue was corrected. After patching, the smoke tests passed.

Post-patch smoke result summary

branch	rows	hit	ROI	avg odds
OU25 combined baseline	4288	0.814832	0.405690	1.738406
OU25 over_only	2946	0.828921	0.428578	1.736130
OU25 under_only	1347	0.789161	0.372530	1.750747

Conclusion:
	•	combined works
	•	over-only works
	•	under-only works
	•	the OU25 product lane is operational

⸻

Forensic Audit Result

The forensic audit is now clean.

The final staged join audit showed:
	•	join_stage_used = full for every branch checked
	•	full_match_count = filtered_rows
	•	merge_miss_rows = 0
	•	join_probe_any_match = True
	•	filtered_rows_missing_from_backtest = 0
	•	duplicate_join_rows = 0
	•	duplicate_filtered_fixture_rows = 0

This means:
	1.	filtered OU25 rows map cleanly back to the canonical scored backtest
	2.	no staged fallback join was required in the final audit state
	3.	branch outputs are genuine filtered subsets of the source corpus
	4.	there is no evidence of branch-generation corruption or join drift

Leakage check

The forensic audit also showed:
	•	selection_leak_suspected = False

Post-match columns present in filtered exports:
	•	home_team_goal_count
	•	away_team_goal_count
	•	correct

These are present for scoring / audit only, not selection.

No evidence was found that post-match columns were used in filtering or ranking.

⸻

OU25 Branch Leaderboard

Current top branches by weighted ROI

rank	branch	sweep_type	pick_mode	rows	hit	ROI	avg odds
1	ou25_combined_topq_080	top_q	combined	2863	0.820817	0.443485	1.768241
2	ou25_mode_over_only	pick_mode	over_only	2946	0.828921	0.428578	1.736130
3	ou25_band2_178_195	band2	combined	4956	0.809726	0.416825	1.761004
4	ou25_band1_124_176	band1	combined	4866	0.817098	0.415307	1.742960
5	ou25_band1_120_176	band1	combined	4881	0.817251	0.415239	1.742596
6	ou25_band1_128_176	band1	combined	4842	0.816605	0.414939	1.743561
7	ou25_band2_178_191	band2	combined	4811	0.811266	0.410804	1.750496
8	ou25_band2_182_195	band2	combined	4434	0.811231	0.409962	1.751830

Baseline references

branch	rows	hit	ROI	avg odds
ou25_combined_baseline	4288	0.814832	0.405690	1.738406
ou25_mode_combined	4288	0.814832	0.405690	1.738406
ou25_combined_topq_070	4288	0.814832	0.405690	1.738406

Under-only reference

branch	rows	hit	ROI	avg odds
ou25_mode_under_only	1347	0.789161	0.372530	1.750747


⸻

What We Learned

1. OU25 can stand as a real product lane

The frozen OU25 framework now works end-to-end:
	•	canonical scored corpus
	•	OU25-only gate application
	•	branch sweeps
	•	branch comparison
	•	cumulative ranking
	•	forensic audit

That is the structural win.

2. Tightening top-q improved quality

The best ROI branch is:
	•	ou25_combined_topq_080
	•	ROI: 0.443485

That is materially better than:
	•	topq_070 / baseline: 0.405690
	•	topq_060: 0.364302
	•	topq_050: 0.319281

Interpretation:
	•	looser inclusion diluted edge
	•	tighter rank trimming improved product quality

3. OVER-only is stronger than UNDER-only

OVER-only:
	•	rows: 2946
	•	hit: 82.89%
	•	ROI: 0.4286

UNDER-only:
	•	rows: 1347
	•	hit: 78.92%
	•	ROI: 0.3725

Interpretation:
	•	OVER-only is currently the stronger OU25 directional lane
	•	UNDER-only is not dead, but it is clearly weaker in this frozen discovery run

4. Band widening can preserve quality while increasing scale

The best wider-scale band branch is:
	•	ou25_band2_178_195
	•	rows: 4956
	•	ROI: 0.416825

This is important because it offers a scale-friendly alternative to the tighter top-q winner.

⸻

Recommended Provisional OU25 Product Positioning

Flagship accuracy / premium lane

ou25_combined_topq_080

Why:
	•	highest ROI
	•	still meaningful row count
	•	strong hit rate
	•	best pure quality branch in the current discovery table

Directional specialist lane

ou25_mode_over_only

Why:
	•	best hit rate among meaningful branches
	•	very strong ROI
	•	cleaner directional interpretation
	•	likely useful as a separate product story

Scale-friendly branch

ou25_band2_178_195

Why:
	•	larger row count than the tighter leader
	•	still excellent ROI
	•	strong candidate for broader deployment packaging

Balanced default reference

ou25_band1_124_176

Why:
	•	strong hit
	•	strong ROI
	•	wider sample
	•	useful as a stable benchmark branch

⸻

Interpretation & Caveats

1. This is a branch-ranking result, not yet a full stability proof

The current cumulative table shows:
	•	single_variant_only = True

That means each branch currently reflects a single dataset context in the cumulative file.

So this table should be interpreted as:
	•	single-corpus branch ranking

not yet as:
	•	multi-window robustness proof

2. The OU25 discovery is real, but still provisional

What is proven now:
	•	the framework works
	•	the branches are real
	•	the audit is clean
	•	the leaderboard is meaningful

What is not yet fully proven:
	•	that these exact same rankings hold under broader multi-window or month-by-month decomposition

3. Post-match fields are present only for audit/scoring

Post-match columns in filtered exports:
	•	home_team_goal_count
	•	away_team_goal_count
	•	correct

These exist because the source is a scored backtest CSV.

The forensic audit found no evidence that these fields were used in filtering/ranking.

4. OVER-only may deserve separate packaging

Because OVER-only materially outperformed UNDER-only, the eventual OU25 investor/product story may need to distinguish:
	•	OU25 combined lane
	•	OU25 over-only lane
	•	OU25 under-only lane

rather than pretending OU25 is one monolithic product.

5. Wider sample vs tighter edge is the central tradeoff

The current OU25 ranking clearly shows the core product decision:
	•	tighter trim = higher ROI, lower row count
	•	broader bands = lower but still strong ROI, larger row count

That tradeoff should be made explicit in future investor-facing material.

⸻

Locked Conclusion

OU25 is now established as a functioning, audited frozen product lane.

The current locked state is:
	•	framework: working
	•	sweeps: working
	•	forensic audit: passed
	•	branch comparison: working
	•	cumulative ranking: working
	•	leaderboard: identified

Locked provisional leaderboard
	1.	ou25_combined_topq_080
	2.	ou25_mode_over_only
	3.	ou25_band2_178_195
	4.	ou25_band1_124_176

Locked practical takeaway
	•	best premium branch: ou25_combined_topq_080
	•	best directional branch: ou25_mode_over_only
	•	best scale branch: ou25_band2_178_195
	•	best balanced benchmark: ou25_band1_124_176

⸻

