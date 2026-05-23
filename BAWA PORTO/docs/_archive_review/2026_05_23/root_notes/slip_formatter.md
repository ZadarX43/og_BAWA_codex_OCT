slip_formatter.py Guide

Purpose

slip_formatter.py is a thin formatter that sits after deploy_rulebook.py.

It does not try to re-predict matches.
It does not invent filler legs.
It does not rescue weak picks.
It does not depend on acca_builder.py.

It simply takes the routed deploy output that already passed your deploy logic and turns that into:
	•	ranked boards
	•	singles
	•	doubles
	•	trebles
	•	yankees
	•	heinz
	•	fixed accas by market or mixed board

This keeps the logic clean:

deploy_rulebook output CSV(s)
→ slip_formatter.py
→ ranked board(s)
→ slip products


⸻

Why this is the correct architecture

The old acca_builder layer overcomplicated the process.

You already proved the strongest value comes from the actual routed deploy outputs from deploy_rulebook.py.

That means the correct flow is:
	•	let deploy_rulebook.py decide what is valid
	•	let slip_formatter.py format those valid picks into boards and slip products
	•	do not add junk just to inflate slip size

So slip_formatter.py is intentionally conservative:
	•	one leg per fixture
	•	top-down from real routed quality
	•	optional removal of FTR base legs
	•	optional exclusion of review / regime flagged rows
	•	optional market-only boards
	•	optional acca size cap

⸻

What it reads

It reads one or more routed deploy CSV files, typically:
	•	...__DEPLOY_TIER_ELITE__...csv
	•	...__DEPLOY_TIER_STANDARD__...csv

Optionally:
	•	...__DEPLOY_TIER_OBSERVE__...csv if you explicitly enable --include-observe

⸻

What it uses from those files

It uses the real routed columns already stamped by deploy_rulebook.py, such as:
	•	market
	•	selection
	•	bookie_pick
	•	model_top_pick
	•	model_p_for_bookie
	•	bookie_od
	•	deploy_tier
	•	standard_reporting_bucket
	•	profit_first_keep
	•	profit_first_review_flag
	•	profit_first_review_reason
	•	context_reason_codes
	•	product_profile
	•	product_lane

From those it derives the internal signal families:
	•	FTR_CS_PROMOTED
	•	HW_HGE2_COMBINED
	•	BTTS_STRONG
	•	BTTS_NO_STRONG
	•	OU25_PREMIUM
	•	FTR_BASE
	•	OTHER

⸻

Default philosophy

By default:
	•	ELITE and STANDARD rows are treated as already valid routed picks
	•	OBSERVE is excluded unless requested
	•	profit_first_keep is optional, not mandatory
	•	FTR_BASE is included unless explicitly excluded
	•	OU25_PREMIUM is included unless explicitly excluded
	•	one fixture can only appear once in the ranked board
	•	ranking is global, then split into market boards

⸻

Main outputs

A run can produce:
	•	ranked_board_<tag>.csv
	•	ranked_board_ftr_<tag>.csv
	•	ranked_board_btts_<tag>.csv
	•	ranked_board_ou25_<tag>.csv
	•	ranked_board_family_summary_<tag>.csv
	•	slips_singles_<tag>.csv
	•	slips_doubles_<tag>.csv
	•	slips_trebles_<tag>.csv
	•	slips_yankee_<tag>.csv
	•	slips_heinz_<tag>.csv

And fixed accas:
	•	mixed runs:
	•	slips_acca03_<tag>.csv
	•	slips_acca04_<tag>.csv
	•	slips_acca05_<tag>.csv
	•	etc
	•	market-specific runs:
	•	slips_acca_ftr_03_<tag>.csv
	•	slips_acca_btts_03_<tag>.csv
	•	slips_acca_ou25_03_<tag>.csv
	•	etc

⸻

Global rank vs market rank

The formatter writes:
	•	a global ranked board
	•	split market boards for:
	•	FTR
	•	BTTS
	•	OU25

Each split board includes:
	•	rank = global board rank
	•	market_rank = local rank within that market

So you can audit both:
	•	where the pick sits in the total board
	•	where it sits inside its own market

⸻

Core command patterns

1. Simplest run: one file

python3 slip_formatter.py \
  --input "/path/to/DEPLOY_TIER_ELITE.csv" \
  --outdir "./slips"

Use this when you just want to format one routed deploy file.

⸻

2. Standard real-world run: ELITE + STANDARD

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips"

This is the normal production pattern.

It combines both routed files, applies risk policy, and builds the boards/slips.

⸻

3. FTR-only market run

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_ftr" \
  --market ftr

This will still build the full ranked board for audit, but fixed acca products will be written as:
	•	slips_acca_ftr_*

instead of mixed generic acca files.

⸻

4. FTR-only, no base legs

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_ftr_no_base" \
  --market ftr \
  --monster-ftr-no-base

This removes FTR_BASE rows from FTR and mixed acca products.

This is useful when you want FTR slips built only from:
	•	FTR_CS_PROMOTED
	•	HW_HGE2_COMBINED

and do not want base FTR legs padding the stack.

⸻

5. Cap the acca size

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_capped" \
  --market ftr \
  --monster-ftr-no-base \
  --max-acca-size 8

This means:
	•	the board may still have capacity for bigger accas
	•	but only fixed accas up to 8-fold are written

The summary now separates:
	•	Largest eligible acca
	•	Largest written acca

so you can see both.

⸻

6. Exclude review-flagged rows

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_clean" \
  --block-review-flags

This removes rows where:
	•	profit_first_review_flag == -1

Only use this when those flags are actually populated in the input files.

⸻

7. Exclude regime / demotion / blocked rows

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_no_regime" \
  --block-regime-flags

This removes rows whose reason fields contain tokens such as:
	•	DOWNGRADE_REGIME
	•	DEMOTED_TO_OBSERVE
	•	DEMOTED_MARGIN_STANDARD
	•	BLOCKED_LIVE
	•	BLOCKED_POWER
	•	BLOCKED_POWER_DIRECTION
	•	BLOCK_PICK_SIDE_MARGIN
	•	SELECTION_MISMATCH
	•	DOUBLE_CHAOS_WARN
	•	SHORTPRICE_MODEL_LT_0P41_BLOCK

Use this if you want a stricter board made only from clean routed survivors.

⸻

8. Require profit-first rows only

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_profit_first" \
  --keep-only

This keeps only rows where:
	•	profit_first_keep == 1

This is optional and should be used carefully, because some routed files may contain valid rows outside that subset.

⸻

9. Exclude FTR base rows entirely

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_no_ftr_base" \
  --exclude-ftr-base

This excludes rows mapped to FTR_BASE.

⸻

10. Exclude OU25 premium rows entirely

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_no_ou25" \
  --exclude-ou25

This excludes rows mapped to OU25_PREMIUM.

⸻

11. Include OBSERVE rows

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  "/path/to/DEPLOY_TIER_OBSERVE.csv" \
  --outdir "./slips_with_observe" \
  --include-observe

Use this only when you intentionally want OBSERVE-tier rows included.

Default behavior is to exclude them.

⸻

12. Cap family concentration

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_family_cap" \
  --max-per-family 6

This prevents one family from dominating the full board.

⸻

13. Cap league concentration

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_league_cap" \
  --max-per-league 4

This limits how many legs from one league can survive into the ranked board.

⸻

14. Apply rank filters

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_rank_window" \
  --min-rank 1 \
  --max-rank 18

This lets you restrict the board to a top segment only.

⸻

15. Apply minimum model probability

python3 slip_formatter.py \
  --inputs \
  "/path/to/DEPLOY_TIER_ELITE.csv" \
  "/path/to/DEPLOY_TIER_STANDARD.csv" \
  --outdir "./slips_model_p" \
  --min-model-p 0.60

This adds an extra hard model probability floor.

Use sparingly, because the routed deploy logic has already done the main filtering.

⸻

Recommended command patterns

A. Main mixed-board production run

python3 slip_formatter.py \
  --inputs \
  "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/2026-03-30/BOOKIE_IMP20_ALLMARKETS_2026-03-13_to_2026-03-17__DEPLOY_TIER_ELITE__PRESET_V1__FTR_accuracy.csv" \
  "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/2026-03-30/BOOKIE_IMP20_ALLMARKETS_2026-03-13_to_2026-03-17__DEPLOY_TIER_STANDARD__PRESET_V1__FTR_accuracy.csv" \
  --outdir "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/slips_test_mixed"


⸻

B. Main FTR-only no-base run

python3 slip_formatter.py \
  --inputs \
  "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/2026-03-30/BOOKIE_IMP20_ALLMARKETS_2026-03-13_to_2026-03-17__DEPLOY_TIER_ELITE__PRESET_V1__FTR_accuracy.csv" \
  "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/2026-03-30/BOOKIE_IMP20_ALLMARKETS_2026-03-13_to_2026-03-17__DEPLOY_TIER_STANDARD__PRESET_V1__FTR_accuracy.csv" \
  --outdir "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/slips_test_ftr" \
  --market ftr \
  --monster-ftr-no-base \
  --max-acca-size 8


⸻

C. Clean strict routed board

python3 slip_formatter.py \
  --inputs \
  "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/2026-03-30/BOOKIE_IMP20_ALLMARKETS_2026-03-13_to_2026-03-17__DEPLOY_TIER_ELITE__PRESET_V1__FTR_accuracy.csv" \
  "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/2026-03-30/BOOKIE_IMP20_ALLMARKETS_2026-03-13_to_2026-03-17__DEPLOY_TIER_STANDARD__PRESET_V1__FTR_accuracy.csv" \
  --outdir "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/slips_test_strict" \
  --block-regime-flags \
  --max-per-family 6 \
  --max-per-league 4


⸻

How ranking works

Rows are sorted by:
	1.	family priority
	2.	deploy tier priority
	3.	descending model_p
	4.	ascending odds

Current family priority:
	•	FTR_CS_PROMOTED = 1
	•	BTTS_STRONG = 2
	•	BTTS_NO_STRONG = 2
	•	HW_HGE2_COMBINED = 3
	•	OU25_PREMIUM = 7
	•	FTR_BASE = 99
	•	OTHER = 999

Deploy tier priority:
	•	ELITE = 1
	•	STANDARD = 2
	•	OBSERVE = 3

So the board is always trying to preserve your strongest routed families first.

⸻

One-leg-per-fixture rule

The risk policy includes:
	•	only one row per fixture survives the board

That means if a fixture appears in both:
	•	FTR
	•	BTTS
	•	OU25

the highest-ranked valid routed row for that fixture survives, and the others are dropped.

This is intentional. It prevents cross-market duplication and slip contamination from the same match.

⸻

What the summary means

At the end of a run you now get:
	•	Ranked board
	•	Markets
	•	Families
	•	FTR rows
	•	BTTS rows
	•	OU25 rows
	•	Largest eligible acca
	•	Largest written acca
	•	Largest mixed acca
	•	Largest FTR acca
	•	Largest BTTS acca
	•	Largest OU25 acca

Interpretation:
	•	Largest eligible acca
biggest acca size supported by the filtered primary acca board for that run
	•	Largest written acca
biggest acca actually written after applying --max-acca-size
	•	Largest mixed acca
maximum possible from the mixed board
	•	Largest FTR / BTTS / OU25 acca
maximum possible inside each market-specific board

⸻

Important note on --market

--market controls the fixed acca product generation, not the global ranked board creation.

So even if you run:

--market ftr

the formatter still writes:
	•	global ranked board
	•	market split boards
	•	singles
	•	doubles
	•	trebles
	•	yankee
	•	heinz

But the fixed acca files become market-specific, for example:
	•	slips_acca_ftr_08_*.csv

and it does not write plain mixed slips_acca08_*.csv in an FTR-only run.

⸻

When to use --monster-ftr-no-base

Use it when you want FTR accas built without FTR_BASE.

That means the FTR acca board will use only:
	•	FTR_CS_PROMOTED
	•	HW_HGE2_COMBINED

This is useful when you want the purest high-confidence FTR stack and do not want base legs diluting acca quality.

⸻

When not to use too many switches

The formatter is meant to stay thin.

So the default best practice is:
	1.	let deploy_rulebook.py do the real filtering
	2.	use slip_formatter.py to format the routed output
	3.	only add switches when you are deliberately testing a policy change

That keeps the system auditable and prevents drifting back into over-engineered selection layers.

⸻

Suggested workflow going forward

Weekly flow
	1.	ingest fresh data
	2.	run your source / deploy pipeline
	3.	produce ELITE + STANDARD deploy outputs
	4.	run slip_formatter.py
	5.	inspect:
	•	global ranked board
	•	FTR board
	•	BTTS board
	•	OU25 board
	•	family summary
	6.	choose product format:
	•	singles
	•	doubles
	•	trebles
	•	yankee
	•	heinz
	•	capped market accas

⸻

Step status

This step is now done.

You now have:
	•	routed deploy outputs
	•	clean formatter logic
	•	market split boards
	•	global and local ranks
	•	explicit acca capacity summary
	•	market-specific acca naming
	•	no dependency on acca_builder.py

Exciting.