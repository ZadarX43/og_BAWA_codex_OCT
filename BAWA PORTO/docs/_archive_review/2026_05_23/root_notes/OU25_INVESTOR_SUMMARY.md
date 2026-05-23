# OU25 Product Summary

## Product status

OU25 is now established as a second major OG product family alongside FTR.

This sweep tested frozen OU25 branches on the canonical 19-league, 3-year, truth-backed IMP40 backtest corpus.

The purpose was to identify:
- stable OU25 candidate lanes
- optimal top-q ranking trims
- over-only vs under-only behavior
- odds-band refinements suitable for investor-facing deployment

---

## Canonical corpus

- Source corpus: `predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv`
- Coverage:
  - 19 leagues
  - 3-year window
  - row-level truth-backed backtest
- OU25 universe in canonical corpus:
  - market label: `ou25`
  - selections observed: `OVER25`, `UNDER25`

---

## Headline findings

### Baseline combined OU25 lane
- Rows: 4,288
- Hit rate: 81.48%
- ROI: +40.57%
- Avg odds: 1.738

### Best ROI branch
`ou25_combined_topq_080`
- Rows: 2,863
- Hit rate: 82.08%
- ROI: +44.35%
- Avg odds: 1.768

### Best over-only branch
`ou25_mode_over_only`
- Rows: 2,946
- Hit rate: 82.89%
- ROI: +42.86%
- Avg odds: 1.736

### Under-only branch
`ou25_mode_under_only`
- Rows: 1,347
- Hit rate: 78.92%
- ROI: +37.25%
- Avg odds: 1.751

---

## Interpretation

The OU25 product is not weak, marginal, or noisy.

It has now shown:
- scale
- strong strike rate
- strong flat-stake ROI
- branch sensitivity that is logical rather than random

The current evidence suggests:

1. **OU25 combined is already investor-viable**
   - even the baseline combined branch is strong

2. **Over-only is likely the cleaner deploy lane**
   - highest hit rate among the practical branches
   - strong ROI
   - simpler public story

3. **Top-q tightening improves product quality**
   - top-q 0.80 produced the best ROI in the current sweep
   - suggests score/ranking discipline matters

4. **Under-only is viable but weaker**
   - still profitable
   - lower league coverage
   - probably a secondary lane, not the flagship

---

## Current candidate product lanes

### OU25 Accuracy / Core Lane
Recommended current candidate:
- `ou25_mode_over_only`

Reason:
- strong hit rate
- strong ROI
- simpler to explain and deploy
- cleaner product narrative

### OU25 Premium / Trimmed Lane
Recommended current candidate:
- `ou25_combined_topq_080`

Reason:
- best observed ROI
- tighter rank discipline
- smaller but higher-quality branch

---

## What has been proven

We have now proven that OG can produce a second non-FTR product family with serious backtested strength.

Specifically:
- OU25 works on the same canonical truth-backed corpus used for FTR
- frozen gates can extract multiple viable OU25 branches
- branch outputs are reproducible and artifact-backed
- the product is strong enough to proceed into cumulative stats, forensic audit, and investor packaging

---

## Remaining work

The remaining work is not “rescue OU25.”

The remaining work is:
- formal cumulative stats
- forensic audit
- investor summary tables
- month-by-month / season-by-season stability review
- then BTTS replication using the same framework

---

## Current conclusion

FTR is locked.

OU25 is now entering lock phase.

If BTTS can be brought to comparable quality, OG will have:
- one elite result product (FTR)
- one elite goals product (OU25)
- one additional correlation-capable market family (BTTS)

That is the foundation for a serious multi-product investor narrative.



# Odds Genius — OU2.5 Product Summary
## Frozen Discovery Result (19 Leagues, 3 Years)

### Investor summary

Odds Genius has now completed a full frozen-discovery cycle for the **Over/Under 2.5 Goals** market across a canonical **19-league, 3-year truth-backed backtest corpus**.

This work confirms that OU2.5 is no longer just present in the model stack — it is now a **separate, functioning product lane** with:

- repeatable frozen gate application
- branch-level threshold discovery
- directional splits (combined / over-only / under-only)
- ranked leaderboard outputs
- clean forensic audit
- permanent summary artifacts

The current discovery result identifies multiple viable OU2.5 product branches, including:

- a **premium high-edge branch**
- a **directional over-only branch**
- a **scale-friendly combined branch**
- a **balanced benchmark branch**

This positions OU2.5 as the second major product family after FTR, with BTTS next in sequence.

---

## What was tested

We ran frozen OU25 discovery against the canonical scored backtest corpus:

`predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv`

This corpus contains:

- 19 leagues
- 3 years of truth-backed scored rows
- row-level bookmaker odds
- row-level correctness
- row-level model ranking fields

For OU25 specifically:

- total OU25 rows in corpus: **19,923**
- OVER25 rows: **13,549**
- UNDER25 rows: **6,374**

---

## Product discovery objective

The goal was to discover whether OU25 could support a real product architecture similar to FTR.

We tested:

- combined over/under branch
- over-only branch
- under-only branch
- band1 odds-range variants
- band2 odds-range variants
- top-q rank trims

This let us evaluate the core tradeoff between:

- **quality**
- **scale**
- **directional specialization**

---

## Headline result

OU25 passed discovery.

The lane now has:

- valid standalone gate logic
- successful smoke tests
- successful full branch sweep execution
- successful branch comparison build
- successful cumulative ranking build
- successful forensic audit

This means OU25 is now structurally deployable as its own product family.

---

## Current branch leaderboard

### Top branches by ROI

| Branch | Type | Pick Mode | Rows | Hit Rate | ROI | Avg Odds |
|---|---|---:|---:|---:|---:|---:|
| `ou25_combined_topq_080` | top_q | combined | 2863 | 82.08% | 0.4435 | 1.7682 |
| `ou25_mode_over_only` | pick_mode | over_only | 2946 | 82.89% | 0.4286 | 1.7361 |
| `ou25_band2_178_195` | band2 | combined | 4956 | 80.97% | 0.4168 | 1.7610 |
| `ou25_band1_124_176` | band1 | combined | 4866 | 81.71% | 0.4153 | 1.7430 |

---

## What the leaderboard means

### 1. Premium branch
`ou25_combined_topq_080`

This is the strongest current branch on pure return.

Why it matters:
- highest ROI in current discovery set
- strong hit rate
- still meaningful sample size
- strongest premium candidate

### 2. Directional branch
`ou25_mode_over_only`

This is the strongest directional OU25 lane.

Why it matters:
- strongest directional hit rate
- very strong ROI
- clean product story
- suggests OVER may be a better standalone commercial lane than UNDER

### 3. Scale-friendly branch
`ou25_band2_178_195`

This is the best large-sample branch among the current leaders.

Why it matters:
- more rows than the tighter premium branch
- still strong ROI
- strong candidate for higher-volume deployment

### 4. Balanced benchmark
`ou25_band1_124_176`

This is the stable reference branch.

Why it matters:
- strong hit rate
- strong ROI
- reasonable scale
- useful benchmark for future walk-forward comparison

---

## Directional insight

One of the clearest discoveries is that **OVER-only is currently stronger than UNDER-only**.

### OVER-only
- rows: **2946**
- hit: **82.89%**
- ROI: **0.4286**

### UNDER-only
- rows: **1347**
- hit: **78.92%**
- ROI: **0.3725**

Interpretation:

The OU25 product is not behaving like a symmetric market lane.  
At the current frozen thresholds, **OVER25 is materially stronger than UNDER25**.

That gives Odds Genius optionality:

- market combined lane
- over-only specialist lane
- under-only secondary lane

---

## Audit result

The OU25 forensic audit passed.

Confirmed:

- filtered rows map cleanly back to the canonical scored backtest
- no merge misses remain
- no duplicate join rows
- no duplicate filtered fixture rows
- no evidence of post-match leakage into selection logic

Important note:

Post-match fields such as:
- `home_team_goal_count`
- `away_team_goal_count`
- `correct`

are present in filtered exports **for scoring and audit only**.

No evidence was found that these fields were used for branch selection or ranking.

---

## What this proves

This discovery phase proves that Odds Genius can now:

1. run OU25 as a standalone frozen product lane
2. compare multiple branch families systematically
3. rank branches by ROI, hit rate, and sample size
4. audit filtered outputs back to source truth rows
5. package OU25 as a real product family, not just a feature column

---

## What this does not yet prove

This is still a **frozen discovery result**.

It does **not yet fully prove** that the top OU25 branch is the final deploy branch under month-by-month forward conditions.

That is the purpose of the next phase:

## OU25 walk-forward validation

That step will test whether the best frozen branches remain stable when applied in true forward windows rather than across the whole canonical corpus at once.

---

## Decision taken

At this stage, the current provisional OU25 hierarchy is:

1. `ou25_combined_topq_080`
2. `ou25_mode_over_only`
3. `ou25_band2_178_195`
4. `ou25_band1_124_176`

These are now the locked candidate branches for the next phase.

---

## Next milestone

The next required milestone is:

**OU25 walk-forward validation**

Goal:
- test branch stability month by month
- confirm whether the frozen leaderboard survives forward deployment conditions
- lock the final OU25 deploy branch

Once that is complete, OU25 can be promoted from:

**discovery winner**  
to  
**validated deploy product**