Canonical answer: which file is “the” backtest?

✅ Canonical “investor / product” corpus = IMP40

Use this as the main truth-backed dataset for everything product-level:
	•	FTR Value (odds ≥ 2.14) ✅ (you have 5539 rows)
	•	FTR Accuracy (bookie implied ≥ 68% ⇒ odds ≤ 1/0.68) ✅ (you have 3451 rows)
	•	OU25 / BTTS at scale ✅ (≈20k each)

Path:
predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv

✅ “Short favourites sanity baseline” = IMP68

This is a subset that basically equals “bookie implied ≥ 68% for FTR”.
It’s good for:
	•	quick regression tests (fast, small)
	•	confirming the short-fav slice is stable
	•	confirming the pipeline works end-to-end without waiting

Path:
predictions_output/backtests/19l_3y_baseline/BOOKIE_IMP68_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv

And your check proves exactly why:
	•	IMP68: FTR ≥ 2.14 = 0 (so it cannot represent the “value product” at all)
	•	IMP40: FTR ≥ 2.14 = 5539 (so it can represent value + accuracy in one universe)

So: IMP40 is canonical. IMP68 is a “quick subset test”.

⸻

What have we proven (stock check)

Proven ✅
	1.	Both files are row-level backtests (have correct, score, bookie_od).
	2.	IMP68 = FTR short-fav slice only
	•	3451/3451 FTR rows are ≤ 1/0.68
	3.	IMP40 contains BOTH key FTR products
	•	Value pool exists (≥ 2.14)
	•	Accuracy pool exists (≤ 1/0.68)
	4.	19 leagues present in both.

New “quality signal” we learned ✅
	•	Truth-join completeness is better on IMP68 (28 NaN correct) than IMP40 (175 NaN correct), which is expected because IMP40 includes a lot more matches, weird cups/qualifiers edge cases, and more long-tail fixtures.

⸻

Why does IMP40 have more NaN correct rows?

Simple math + reality:
	•	IMP40 is ~10x bigger (58,721 rows vs 5,946)
	•	It includes more obscure fixtures / naming variants / qualifier rounds that don’t always land cleanly in Matches/<League>/fd_odds_enriched.csv under the same fixture_key.
	•	You’re joining by fixture_key, which is brittle for UEFA qualifiers + weird team-name punctuation.

Also important: this is not catastrophic.
	•	IMP68 NaNs: 28 / 5946 = 0.47%
	•	IMP40 NaNs: 175 / 58721 = 0.30%

So proportionally IMP40 is actually fine — it’s just bigger, so you see more absolute misses.

⸻

What’s next to test (the right next 3 steps)

Step 1 — Lock the “canonical corpus” naming (so you never lose it again)

Create a tiny note file (or just a convention) like:
	•	CANONICAL_CORPUS_FTR_PRODUCTS = IMP40 __BACKTEST.csv
	•	FAST_REGRESSION_CORPUS = IMP68 __BACKTEST.csv

(If you want, I’ll give you a 12-line README snippet you can paste into your runbook.)

Step 2 — Produce the final “investor tables” for ALL markets from IMP40

You already have the FTR investor table. Next is:
	•	OU25 per-league: n, hit, ROI, avg_od (for gated profiles you care about)
	•	BTTS per-league: same
	•	plus “portfolio” rollups

If you tell me what your OU25/BTTS gating definitions are for the investor story (odds bands etc), I’ll give you one exact command/script to generate the CSVs in one go.

Step 3 — Reduce NaN correct rows (optional polish, but nice)

This is the “truth join hardening” work. Cheapest is usually:
	•	Option C (UEFA comps only): if fixture_key miss, fallback join on (match_date, home_team, away_team) within the UEFA matches file.

Because your Matches layout is “Matches//fd_odds_enriched.csv”, Option A “scan entire Matches root” is brute-force and can accidentally match the wrong fixture unless carefully scoped.

If you want the surgical approach: paste the _load_matches() block from backtest_deploy_csv.py (you already found it), and I’ll tell you exactly where to add the fallback join (minimal code change).



Bottom line: how good is OG right now?

From what you’ve shown in this run, the system is now doing what we need structurally:
	•	It can generate a single, consistent 3-year/19-league truth-backed corpus (IMP40)
	•	It supports both headline products in the same data universe:
	•	FTR Value (≥2.14) and it’s profitable in your gated run
	•	FTR Accuracy (≤1/0.68) and it’s high hit-rate in your gated run
	•	OU25/BTTS are present at scale for the same window (and you can now generate investor tables for them too)