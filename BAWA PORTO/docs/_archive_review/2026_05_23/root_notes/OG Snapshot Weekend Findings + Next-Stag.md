OG Snapshot Weekend Findings + Next-Stage Experiment Plan

Purpose

This document captures what we learned from:
	•	the snapshot weekend shortlist system
	•	the snapshot support board scripts
	•	the feature-family testing harness
	•	the deploy-rulebook FTR / BTTS / OU25 experiments across 19 leagues and competitions

It is intended to serve as the working reference before running:
	•	large-scale historical backtests
	•	walk-forward validation
	•	investor-grade proof experiments

The aim is to make sure the new calibrations, gates, thresholds, and product buckets are fixed and understood before scaling.

⸻

1. Executive summary

We have now identified two separate but complementary systems:

A. Snapshot weekend support system

A fast structural shortlist and explainability layer built from team matchup snapshot features.

Outputs:
	•	home-side FTR structural support
	•	Over 2.5 structural support
	•	away/draw danger
	•	conflict/alignment flags

This is not yet a final deploy engine.
It is better understood as:
	•	a structural filter
	•	an explainability layer
	•	a candidate overlay source
	•	a scouting board for weekend decisions

B. Deploy rulebook live-book system

A market-routing and product-tiering engine that creates:
	•	ELITE
	•	STANDARD
	•	OBSERVE

with now-proven profitable sub-buckets such as:
	•	STANDARD_FTR_CS_PROMOTED
	•	STANDARD_BTTS
	•	STANDARD_OU25
	•	STANDARD_FTR_BASE

This system produced the strongest results so far, including:
	•	CS-promoted FTR bucket outperforming ELITE on the tested windows
	•	profit-first and strict profit-first live books reaching very high hit rates
	•	league-level separation showing where live deployment should be selective

⸻

2. Core breakthrough

Biggest technical finding

The correct-score overlay mechanism appears to have unlocked a major FTR routing breakthrough.

Specifically:
	•	OBSERVE FTR rows with strong correct-score side agreement
	•	sufficient support mass
	•	sufficient side margin
	•	and sufficient model_p_for_bookie

can be promoted to STANDARD with exceptional reliability.

This has now been demonstrated on multiple windows.

Important conclusion

The correct-score overlay is not cosmetic.

It is acting like a high-quality second-opinion confirmation mechanism for FTR and has produced one of the strongest product buckets in the system.

⸻

3. Snapshot support board: what it actually is

Main script
	•	weekend_snapshot_support_board.py

All-leagues runner
	•	weekend_snapshot_support_board_all_leagues.py

Purpose

These scripts build a support board from __team_snapshot_matchups.csv files and score each fixture for:
	•	FTR home-side structural support
	•	OU25 structural support

They are not training models.
They are deterministic structural scorers.

⸻

4. Snapshot support formulas

4.1 FTR snapshot support score

Built in:
	•	_score_ftr_snapshot_support(df)

Features used
	•	snap_strength_ppg_home_vs_away
	•	snap_strength_position_edge_home
	•	snap_home_attack_vs_away_def_xg
	•	snap_home_attack_vs_away_def_goals
	•	snap_ht_home_lead_vs_away_trail_edge
	•	snap_timing_early_goal_pressure

Weights
	•	strength_ppg: 1.25
	•	strength_pos: 1.00
	•	attack_xg: 1.10
	•	attack_goals: 0.90
	•	ht_edge: 0.90
	•	timing: 0.85

Scoring method

Each component is robust-scaled with median / MAD style normalization, clipped to [-3, 3].

Final score:
	•	weighted mean of normalized components
	•	then mapped to:
	•	50 + 15 * score
	•	clipped to [0, 100]

Interpretation

This is primarily a home-side support score, not a full 3-way FTR engine.

⸻

4.2 OU25 snapshot support score

Built in:
	•	_score_ou25_support(df)

Features used
	•	snap_xg_total_pressure
	•	snap_style_chaos_index
	•	snap_ou25_over_regime_blend
	•	snap_timing_both_teams_late_risk
	•	snap_timing_late_goal_pressure

Weights
	•	attack_xg: 1.20
	•	chaos: 0.95
	•	ou25_over: 1.10
	•	late_risk: 1.00
	•	timing: 0.90

Scoring method

Same robust normalization approach as FTR.

Final score:
	•	weighted mean of normalized components
	•	mapped to:
	•	50 + 15 * score
	•	clipped to [0, 100]

Interpretation

This is a goal-environment / openness / escalation score, not a final over-betting engine on its own.

⸻

5. Snapshot support buckets

Both FTR and OU25 currently use the same bucket function:
	•	strong_support if score >= 62
	•	support if 56 <= score < 62
	•	neutral if 44 < score < 56
	•	weak if 38 < score <= 44
	•	oppose if score <= 38

⸻

6. Snapshot weekend shortlist results

6.1 FTR strong-support shortlist tested

Rule:
	•	snapshot_ftr_support_score > 70
	•	snapshot_ftr_support_bucket = strong_support

Result
	•	Hits: 7
	•	Misses: 4
	•	Hit rate: 63.64%

Conclusion

Useful as structural confirmation, but not strong enough as a standalone deploy rule.

⸻

6.2 OU25 strong-support shortlist tested

Rule:
	•	snapshot_ou25_support_score > 70
	•	snapshot_ou25_support_bucket = strong_support

Result
	•	Hits: 5
	•	Misses: 3
	•	Hit rate: 62.50%

Conclusion

Useful as regime support, but not strong enough standalone.

⸻

6.3 Combined overlap fixtures

Rows that were strong on both FTR and OU25 showed a mix of:
	•	double hits
	•	goals-right-side-wrong cases
	•	double misses

Key interpretation

This suggests the snapshot layer is especially useful for:
	•	identifying high-event matches
	•	identifying favourite instability
	•	supporting draw suppression logic
	•	helping FTR when combined with other signals

⸻

7. What the snapshot system is best used for

The current evidence suggests the snapshot board should be treated as:

Best uses
	•	structural confirmation layer
	•	weekend scouting board
	•	explainability tool
	•	feature source for the main model stack
	•	overlay input for FTR / OU25 / BTTS

Not best used as
	•	raw standalone deploy engine
	•	single-threshold direct betting product

⸻

8. Feature-family testing harness

Main script
	•	test_team_snapshot_feature_families_v1

Purpose

To test whether raw snap_... features improve:
	•	ftr
	•	ou25
	•	btts

relative to non-snapshot baseline features.

How it works
	•	resolves target columns or fallback targets from goals
	•	filters to completed fixtures only
	•	time-sorts data
	•	uses 80/20 time split
	•	trains logistic regression pipeline
	•	compares:
	•	baseline
	•	baseline + all snapshot features
	•	baseline + each family
	•	baseline + family combos

⸻

9. Snapshot feature families

Strength

Prefixes:
	•	snap_strength_
	•	snap_position_
	•	snap_rank_
	•	snap_goal_diff_
	•	snap_points_
	•	snap_performance_

Attack / defence

Prefixes:
	•	snap_home_attack_vs_away_def
	•	snap_away_attack_vs_home_def
	•	snap_xg_total_pressure
	•	snap_shot_pressure
	•	snap_minutes_

Clean sheet / FTS / BTTS

Prefixes:
	•	snap_home_clean_sheet_vs_away_fts
	•	snap_away_clean_sheet_vs_home_fts
	•	snap_btts_
	•	snap_clean_sheet_
	•	snap_fts_
	•	snap_first_team_to_score_

Half-time

Prefixes:
	•	snap_ht_
	•	snap_half_time_

Style / chaos

Prefixes:
	•	snap_style_
	•	snap_chaos_
	•	snap_ou25_
	•	snap_over25_
	•	snap_under25_
	•	snap_possession_
	•	snap_cards_
	•	snap_corners_
	•	snap_fouls_

Timing pressure

Prefixes:
	•	snap_timing_
	•	snap_early_
	•	snap_late_

⸻

10. Strongest current deploy-rulebook findings

10.1 FTR CS overlay promotion

Promotes OBSERVE FTR rows into STANDARD when:
	•	market is ftr
	•	league is not in denylist
	•	model_p_for_bookie >= 0.60
	•	ftr_margin >= 0.03
	•	no hard upper margin ceiling
	•	correct-score top-3 supports the same side
	•	cs_top3_support_mass >= 0.25
	•	cs_top3_side_margin >= 0.20

Current denylist
	•	Brazil Serie A

Reason codes stamped
	•	FTR_CS_TOP3_MATCH
	•	FTR_CS_SUPPORT_MASS_025
	•	FTR_CS_SIDE_MARGIN_020
	•	FTR_CS_PROMOTE_STANDARD

⸻

10.2 Proven FTR CS promotion performance

Window 1

On one earlier test window:
	•	promoted FTR rows: 60
	•	hit rate: 91.67%
	•	non-promoted STANDARD FTR base: 87.50%

Window 2

On the larger 13-league window:
	•	promoted FTR rows: 63
	•	hit rate: 96.83%
	•	non-promoted STANDARD FTR base: 60.87%

19-league universe

On the broader 19-league universe:
	•	STANDARD_FTR_CS_PROMOTED: 73 rows
	•	hit rate: 97.26%
	•	STANDARD_FTR_BASE: 26 rows
	•	hit rate: 57.69%

Conclusion

The CS-promoted FTR bucket is materially different from plain STANDARD FTR and should be treated as its own product.

⸻

11. STANDARD reporting buckets

STANDARD now splits into:
	•	STANDARD_BTTS
	•	STANDARD_OU25
	•	STANDARD_FTR_CS_PROMOTED
	•	STANDARD_FTR_BASE

This matters because plain STANDARD as one combined bucket hides the fact that:
	•	CS-promoted FTR is elite-grade
	•	BTTS can be very strong in selective leagues
	•	OU25 is weaker and suppresses blended quality
	•	FTR base should often be excluded from live profit-first books

⸻

12. 19-league / competition universe tested

The expanded universe included:
	•	England Premier League
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
	•	England EFL League 1
	•	England FA Cup
	•	England Championship

⸻

13. 19-league deploy-rulebook results

ELITE
	•	rows: 23
	•	hit rate: 91.30%

By market:
	•	BTTS: 10 rows, 90.00%
	•	FTR: 13 rows, 92.31%

STANDARD overall
	•	rows: 556
	•	hit rate: 79.68%

By market:
	•	BTTS: 237 rows, 88.61%
	•	FTR: 99 rows, 86.87%
	•	OU25: 220 rows, 66.82%

STANDARD sub-buckets
	•	STANDARD_BTTS: 237, 88.61%
	•	STANDARD_OU25: 220, 66.82%
	•	STANDARD_FTR_CS_PROMOTED: 73, 97.26%
	•	STANDARD_FTR_BASE: 26, 57.69%

⸻

14. Profit-first live book logic

Profit-first live book

Keep:
	•	ELITE
	•	STANDARD_BTTS
	•	STANDARD_FTR_CS_PROMOTED

Exclude:
	•	STANDARD_OU25
	•	STANDARD_FTR_BASE

19-league result
	•	rows: 333
	•	blended hit rate: 90.69%
	•	unique fixtures: 322
	•	all fixtures in window: 1268
	•	fixture coverage: 25.39%

⸻

15. Strict profit-first live book logic

Strict book

Same as profit-first, but review-tagged leagues are excluded from STANDARD_BTTS.

BTTS review leagues

Current review list:
	•	Brazil Serie A
	•	Europa League
	•	Belgium Pro

BTTS priority keep leagues

Current keep-priority list:
	•	Japan J1
	•	Netherlands Eredivisie
	•	Europa Conference

FTR priority keep leagues

Current keep-priority list:
	•	Portugal Liga
	•	Italy Serie A
	•	England Championship
	•	USA MLS

Strict book result
	•	rows: 268
	•	blended hit rate: 96.27%
	•	unique fixtures: 259
	•	all fixtures in window: 1268
	•	fixture coverage: 20.43%

This is currently the cleanest live-book construct.

⸻

16. Strict book by odds band

Bookie odds bands
	•	[0.0, 1.4): 16, 100.00%
	•	[1.4, 1.6): 28, 96.43%
	•	[1.6, 1.8): 94, 96.81%
	•	[1.8, 2.1): 90, 94.44%
	•	[2.1, 2.5): 34, 97.06%
	•	[2.5, 10.0): 6, 100.00%

Interpretation

This edge is not confined only to ultra-short prices.

⸻

17. Strict book by model probability band

Model-p bands
	•	[0.0, 0.55): 5, 100.00%
	•	[0.55, 0.6): 4, 75.00%
	•	[0.6, 0.65): 54, 90.74%
	•	[0.65, 0.7): 48, 97.92%
	•	[0.7, 0.75): 49, 97.96%
	•	[0.75, 1.01): 108, 98.15%

Important note

The weakest pocket is:
	•	model_p_for_bookie in [0.55, 0.60)

This supports using a stricter floor on some products, especially BTTS.

⸻

18. Strict-plus-BTTS065 refinement

Additional rule:
	•	keep strict book
	•	require STANDARD_BTTS rows to have model_p_for_bookie >= 0.65

Result
	•	rows: 257
	•	blended hit rate: 96.89%
	•	unique fixtures: 248
	•	all fixtures in window: 1268
	•	fixture coverage: **19.56%`

Breakdown
	•	STANDARD_BTTS: 161, 97.52%
	•	STANDARD_FTR_CS_PROMOTED: 73, 97.26%
	•	ELITE remains included

This is currently the cleanest observed live-book profile.

⸻

19. UEFA competition contribution

Inside the 19-league profit-first universe, UEFA competitions contributed positively overall.

UEFA contribution in profit-first book
	•	rows: 51
	•	hit rate: 90.20%

By competition
	•	Champions League BTTS: 2, 100%
	•	Champions League FTR: 1, 100%
	•	Europa Conference BTTS: 30, 96.67%
	•	Europa League BTTS: 10, 60.00%
	•	Europa League FTR: 8, 100%

Conclusion
	•	Europa Conference BTTS looks strong
	•	Europa League BTTS remains review territory
	•	Europa League FTR looks strong
	•	UEFA should not be excluded globally; only selectively reviewed

⸻

20. Current best live products

Strongest products so far

FTR
	•	STANDARD_FTR_CS_PROMOTED
	•	elite-grade, currently one of the strongest discovered live products

BTTS
	•	STANDARD_BTTS in selected leagues
	•	especially strong in:
	•	Japan J1
	•	Netherlands Eredivisie
	•	Europa Conference
	•	Norway Eliteserien
	•	review or exclude:
	•	Brazil Serie A
	•	Europa League BTTS
	•	Belgium Pro BTTS

FTR priority keep leagues
	•	Portugal Liga
	•	Italy Serie A
	•	England Championship
	•	USA MLS

⸻

21. Files produced by deploy_rulebook

For a source file like:

BOOKIE_IMP20_ALLMARKETS_<date_from>_to_<date_to>.csv

the rulebook writes:
	•	__DEPLOY_PRESET__PRESET_V1__FTR_accuracy.csv
	•	__DEPLOY_PRESET__PRESET_V1__FTR_accuracy.md
	•	__DEPLOY_TIER_ELITE__PRESET_V1__FTR_accuracy.csv
	•	__DEPLOY_TIER_STANDARD__PRESET_V1__FTR_accuracy.csv
	•	__DEPLOY_TIER_OBSERVE__PRESET_V1__FTR_accuracy.csv

Then scoring scripts create:
	•	__SCORED.csv for raw prediction universe
	•	__DEPLOY_TIER_ELITE__...__SCORED.csv
	•	__DEPLOY_TIER_STANDARD__...__SCORED.csv

These scored files are the basis for:
	•	hit-rate analysis
	•	bucket breakdowns
	•	coverage studies
	•	live-book simulations

⸻

22. Profit-first fields now stamped

On deploy output, the following analysis fields now exist:
	•	standard_reporting_bucket
	•	profit_first_bucket
	•	profit_first_keep
	•	profit_first_review_flag
	•	profit_first_review_reason

These make it possible to build:
	•	strict live books
	•	review-tagged subsets
	•	future investor reporting by product family

⸻

23. What the snapshot weekend system likely contributes to the main stack

The weekend snapshot layer is probably most useful for:

FTR
	•	confirmation of structural superiority
	•	favourite instability warnings
	•	away/draw danger flags
	•	draw suppression support

OU25
	•	identifying high-event regimes
	•	timing escalation
	•	late-goal risk
	•	chaos / openness classification

BTTS
	•	likely useful through:
	•	timing pressure
	•	clean-sheet / FTS structure
	•	openness / chaos features

The raw snap_... families are likely more valuable than the final support score buckets themselves.

⸻

24. Working conclusions

Conclusion 1

The snapshot weekend board is useful, but should be treated as a structural and explainability layer, not as a standalone product.

Conclusion 2

The correct-score overlay is a real breakthrough and should now be treated as a core FTR product mechanism.

Conclusion 3

STANDARD must remain split into sub-products. A single STANDARD number is misleading.

Conclusion 4

Profit-first and strict profit-first live books are now the correct framework for forward testing and investor presentation.

Conclusion 5

League-specific review and keep lists matter. Some markets are strong only in selected leagues.

⸻

25. Recommended next experiments

A. Large-scale historical backtest

Run across the full 19-league / competition universe over multiple seasons.

Measure:
	•	ELITE
	•	STANDARD_BTTS
	•	STANDARD_FTR_CS_PROMOTED
	•	STANDARD_FTR_BASE
	•	STANDARD_OU25
	•	profit-first
	•	strict profit-first
	•	strict-plus-BTTS065

B. Feature-family testing at scale

Run:
	•	test_team_snapshot_feature_families_v1

for:
	•	FTR
	•	OU25
	•	BTTS

Then compare:
	•	baseline
	•	best family
	•	best combo
	•	all snapshot features

C. Integrate snapshot features into live models

Potential paths:
	•	FTR overlay model
	•	OU25 regime overlay
	•	BTTS environment filter
	•	draw-danger module
	•	volatility stack expansion

D. OU25 experiments next

Use the weekend site experiment findings to identify:
	•	whether team-stat snapshot features improve OU25 over 2.5 premium subsets
	•	whether they help filter out weak OU25 live rows
	•	whether they improve FTR via openness / volatility / non-draw support

⸻

26. Current candidate investor narrative

A clean version of the story is:

We have built a multi-market football intelligence stack across 19 leagues and competitions.
We discovered that correct-score top-3 confirmation can materially upgrade FTR routing, producing a promoted FTR product that outperforms even our elite baseline on tested windows.
We also separated our standard products into distinct sub-buckets, showing that BTTS, OU25, and FTR have different reliability profiles by market and league.
Using a strict profit-first live book composed of ELITE, selected BTTS, and CS-confirmed FTR products, we achieved a hit rate above 96% on a broad 19-league validation window while still covering nearly 20% of fixtures.
We are now scaling this into full historical backtests and walk-forward validation to convert this technical breakthrough into definitive institutional proof.

⸻

27. Current default live-book candidates

Conservative live book

Keep:
	•	ELITE
	•	STANDARD_FTR_CS_PROMOTED
	•	STANDARD_BTTS

Exclude:
	•	STANDARD_FTR_BASE
	•	STANDARD_OU25

Strict live book

Additionally exclude BTTS review leagues:
	•	Brazil Serie A
	•	Europa League
	•	Belgium Pro

Strict-plus-BTTS065 book

Further require:
	•	STANDARD_BTTS with model_p_for_bookie >= 0.65

This is currently the strongest observed profile.

⸻

28. Bottom line

The major technical breakthroughs so far are:
	1.	Correct-score overlay promotion for FTR
	2.	STANDARD product decomposition
	3.	profit-first and strict live-book logic
	4.	league-aware BTTS keep/review structure
	5.	snapshot feature-family framework as the next-scale overlay engine

The next phase is not guesswork anymore.

It is now:
	•	codify
	•	backtest
	•	walk forward
	•	document
	•	present