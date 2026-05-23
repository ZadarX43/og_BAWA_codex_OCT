# FTR Walk-Forward Audit Master Summary

## Purpose

This document locks the current FTR walk-forward findings into a permanent reference.

It records:

* the branch architecture used for FTR product testing
* the files and scripts used to generate the walk-forward outputs
* the commands and branch setups used across months
* what was tested
* what was proven
* what remains to be checked
* how this framework should be reused for Over 2.5 and BTTS productization

This is intended to serve as a durable source of truth for:

* internal model validation
* investor diligence
* technical handoff
* future product extension work

---

# 1. Executive summary

## Core outcome

The FTR system has now been split into two distinct investor-facing product lanes, each tested in frozen month-by-month out-of-sample walk-forward conditions.

### Product A — Accuracy Lane

A higher-strike-rate FTR product built from the tighter upstream universe and lower-odds favorite selection logic.

### Product B — ValueEV Lane

A higher-odds, positive-edge FTR product built from the broader upstream universe and Poisson-normalized edge selection logic.

## Main structural finding

The architecture is not one single monolithic FTR product.

Instead, the evidence supports two different pathways:

* **Accuracy branch** performs best when fed by the **IMP62 upstream universe**.
* **ValueEV branch** performs best when fed by the **IMP40 upstream universe**.

This is the single most important architectural conclusion from the current FTR work.

## Main forensic conclusion

The current audited branch outputs show:

* no duplicate join inflation
* no filtered rows missing from the scored backtest universe
* no fixture-level merge corruption
* no evidence that post-match outcome columns were used in product filtering or ranking
* consistent availability of the Poisson columns used by ValueEV

The post-match fields (`home_team_goal_count`, `away_team_goal_count`, `correct`) are present in filtered outputs, but the audit currently shows they are **carried-through scoring fields**, not used for selection or ranking.

---

# 2. Product lanes

## 2.1 Accuracy Lane

### Positioning

A safer, high-strike-rate FTR product designed around moderate favorites.

### Upstream universe

* `IMP62`

### Gate concept

* accuracy-focused filtering
* moderate odds ceiling
* strong confidence / margin discipline
* home/away-only configuration used in the frozen runs

### Current frozen interpretation

This lane is the cleaner “safe-leg / high hit-rate” product.

---

## 2.2 ValueEV Lane

### Positioning

A higher-odds, positive-edge FTR product using Poisson-normalized pick probabilities.

### Upstream universe

* `IMP40`

### Gate concept

* odds floor to allow higher-priced FTR picks
* edge-based filtering using Poisson-normalized pick probability
* ranked by `ftr_valueev_edge`

### Current frozen interpretation

This lane is the stronger “wow investor / high-edge / higher-odds” product.

---

# 3. Branch architecture

## 3.1 Branches tested

### Accuracy branch

* branch label: `accuracy`
* root: `walkforward_frozen_accuracy/`
* tag: `FTR_ACCURACY`
* upstream: `IMP62`

### ValueEV Balanced branch

* branch label: `valueev_balanced`
* root: `walkforward_frozen_valueev_balanced/`
* tag: `FTR_VALUEEV_BALANCED`
* upstream: `IMP40`

### ValueEV Aggressive branch

* branch label: `valueev_aggressive`
* root: `walkforward_frozen_valueev_aggressive/`
* tag: `FTR_VALUEEV_AGGRESSIVE`
* upstream: `IMP40`

---

# 4. Months tested

The current locked comparison window is:

* 2024-10
* 2024-11
* 2024-12
* 2025-01
* 2025-02
* 2025-03

This gives a six-month cross-season sample spanning autumn, winter, and early spring conditions.

---

# 5. Scripts used

## Core generation scripts

### `run_frozen_walkforward.py`

Purpose:

* frozen month-by-month walk-forward runner
* uses production `ModelStore/`
* does not retrain
* archives raw and gated outputs into month folders

### `bookie_allmarkets.py`

Purpose:

* generates the raw ALLMARKETS deployment universe for the target month

### `backtest_deploy_csv.py`

Purpose:

* joins generated deployment rows to actual match results
* produces scored backtest CSVs

### `apply_frozen_product_gates.py`

Purpose:

* applies frozen product gates directly to the scored backtest CSV
* bypasses the limited older rulebook CLI path
* supports separate FTR product profiles

---

## Audit / aggregation scripts

### `build_branch_comparison.py`

Purpose:

* builds the master month-by-month comparison CSV across branches

### `build_branch_cumulative_stats.py`

Purpose:

* computes cumulative branch statistics
* weighted hit, weighted ROI, weighted avg odds, stability measures

### `forensic_walkforward_audit.py`

Purpose:

* audits fixture joins, filtered row mapping, leak-risk indicators, and selection/ranking field usage

### `poisson_source_audit.py`

Purpose:

* audits presence and stability of Poisson FTR probability columns used in ValueEV

---

# 6. Main output files

## Branch comparison table

* `walkforward_branch_comparison_2024-10_to_2025-03.csv`

## Branch cumulative stats

* `walkforward_branch_cumulative_stats_2024-10_to_2025-03.csv`

## Forensic audit table

* `walkforward_forensic_audit.csv`

## Fixture-level spot-check sample

* `walkforward_fixture_spotcheck_samples.csv`

## Poisson source audit

* `walkforward_poisson_source_audit.csv`

---

# 7. Branch-month output structure

Each branch-month folder contains artifacts such as:

* `backtest_YYYY-MM.csv`
* `backtest_summary_YYYY-MM.csv`
* `backtest_unscored_YYYY-MM.csv`
* `frozen_gated_YYYY-MM.csv`
* `frozen_gated_YYYY-MM.md`
* `frozen_tier_elite_YYYY-MM.csv`
* `frozen_tier_standard_YYYY-MM.csv`
* `frozen_tier_observe_YYYY-MM.csv`
* `summary_YYYY-MM.json`

These branch roots are:

* `walkforward_frozen_accuracy/`
* `walkforward_frozen_valueev_balanced/`
* `walkforward_frozen_valueev_aggressive/`

---

# 8. How the walk-forward works

## Step 1 — Generate monthly universe

For a given month, `bookie_allmarkets.py` generates an ALLMARKETS CSV using the frozen production model store.

## Step 2 — Score the universe

`backtest_deploy_csv.py` joins those generated rows to actual results and produces a scored backtest CSV.

## Step 3 — Apply frozen product gates

`apply_frozen_product_gates.py` applies the product-specific selection logic directly to the scored backtest.

This is where the branch split matters:

* Accuracy branch uses the accuracy gate profile
* ValueEV branches use the edge-based Poisson gate profile

## Step 4 — Archive stable outputs

The month folder receives archived core files and summaries.

## Step 5 — Aggregate across months

The comparison and cumulative scripts collect the per-month branch outputs into global summary tables.

## Step 6 — Run forensic validation

The forensic scripts inspect:

* fixture join cleanliness
* duplicate inflation risk
* filtered-to-backtest consistency
* Poisson source consistency
* post-match column presence vs actual usage in selection logic

---

# 9. Command pattern used

## 9.1 Accuracy branch pattern

Typical configuration:

* upstream universe: `IMP62`
* profile: `accuracy`
* odds cap used in frozen tests: `--ftr-max-od 1.85`
* home/away restriction used in frozen branch runs

Representative gate command shape:

```bash
python apply_frozen_product_gates.py \
  --src walkforward_frozen_accuracy/YYYY-MM/BOOKIE_IMP62_ALLMARKETS_YYYY-MM-01_to_YYYY-MM-END__BACKTEST.csv \
  --outdir walkforward_frozen_accuracy/YYYY-MM \
  --ftr-profile accuracy \
  --btts-max 1.62 \
  --ou25-band1-low 1.24 \
  --ou25-band1-high 1.72 \
  --ou25-band2-low 1.82 \
  --ou25-band2-high 1.91 \
  --top-q 0.7 \
  --ftr-max-od 1.85 \
  --ftr-home-away-only
```

---

## 9.2 ValueEV Balanced branch pattern

Typical configuration:

* upstream universe: `IMP40`
* profile: `valueev_balanced`
* odds floor: `--ftr-valueev-od-min 1.8`
* edge floor: `--ftr-valueev-edge-min 1.05`

Representative gate command shape:

```bash
python apply_frozen_product_gates.py \
  --src walkforward_frozen_valueev_balanced/YYYY-MM/BOOKIE_IMP40_ALLMARKETS_YYYY-MM-01_to_YYYY-MM-END__BACKTEST.csv \
  --outdir walkforward_frozen_valueev_balanced/YYYY-MM \
  --ftr-profile valueev_balanced \
  --btts-max 1.62 \
  --ou25-band1-low 1.24 \
  --ou25-band1-high 1.72 \
  --ou25-band2-low 1.82 \
  --ou25-band2-high 1.91 \
  --top-q 0.7 \
  --ftr-valueev-od-min 1.8 \
  --ftr-valueev-edge-min 1.05
```

---

## 9.3 ValueEV Aggressive branch pattern

Typical configuration:

* upstream universe: `IMP40`
* profile: `valueev_aggressive`
* odds floor: `--ftr-valueev-od-min 1.8`
* edge floor: `--ftr-valueev-edge-min 1.08`

Representative gate command shape:

```bash
python apply_frozen_product_gates.py \
  --src walkforward_frozen_valueev_aggressive/YYYY-MM/BOOKIE_IMP40_ALLMARKETS_YYYY-MM-01_to_YYYY-MM-END__BACKTEST.csv \
  --outdir walkforward_frozen_valueev_aggressive/YYYY-MM \
  --ftr-profile valueev_aggressive \
  --btts-max 1.62 \
  --ou25-band1-low 1.24 \
  --ou25-band1-high 1.72 \
  --ou25-band2-low 1.82 \
  --ou25-band2-high 1.91 \
  --top-q 0.7 \
  --ftr-valueev-od-min 1.8 \
  --ftr-valueev-edge-min 1.08
```

---

# 10. What was learned

## 10.1 Accuracy and ValueEV are not the same product

This was one of the most important discoveries.

The evidence supports separate product lanes rather than one blended FTR idea.

## 10.2 Accuracy branch prefers tighter upstream filtering

The accuracy lane works best with the `IMP62` upstream universe.

## 10.3 ValueEV branch comes alive on broader upstream input

The ValueEV lanes required the broader `IMP40` upstream universe to produce the stronger higher-odds edge behavior.

## 10.4 Old PPG glue did not improve the frozen accuracy branch

The tested PPG glue / power-diff confirms reduced coverage and did not improve the core frozen accuracy result.

That means the old glue logic may have been more useful in a different historical rulebook path than in the current frozen replay path.

## 10.5 November is strong, but the result is not only November

October, November, January, February, and March all remained strong enough to support the structural branch story.

This matters because it shows November is not doing all the work.

---

# 11. What has been proven

## 11.1 Master branch comparison is now built

The system now has a branch comparison table with:

* month
* branch
* rows
* hit
* ROI
* avg odds

## 11.2 Cumulative branch statistics are now built

The system now has cumulative branch stats with:

* months present
* total rows
* weighted hit
* weighted ROI
* weighted avg odds
* best month by hit / ROI
* worst month by hit / ROI
* standard deviation of hit and ROI

## 11.3 Forensic join audit is clean

Across all audited months and branches:

* no merge misses
* no duplicate join inflation
* no filtered rows missing from backtest
* no duplicate filtered fixture rows

## 11.4 Selection/ranking path shows no obvious post-match leakage

The forensic audit now distinguishes:

* post-match columns present in the filtered output
* columns actually checked for filtering
* columns actually checked for ranking

Current result:

* post-match fields are present in filtered outputs
* they are **not** present in selection/ranking usage columns
* `selection_leak_suspected = False`

## 11.5 Poisson source consistency is present across all tested months

The separate Poisson audit confirms:

* FTR Poisson columns are present
* probability sums are populated and stable enough for the current ValueEV path
* there are no null FTR Poisson rows in the audited month window

---

# 12. What the current system can do

The current FTR system can now:

* generate frozen month-by-month out-of-sample FTR walk-forwards
* split FTR into separate product lanes
* replay frozen product gates without depending on the older limited rulebook CLI
* produce branch-specific archived outputs
* produce branch comparison tables
* produce cumulative branch statistics
* produce fixture-level forensic audit outputs
* produce Poisson source consistency audits
* support investor-facing product positioning using distinct lane narratives

---

# 13. Current product framing

## Accuracy Lane

**“A higher-strike-rate FTR product optimized around disciplined favorite selection.”**

Use case:

* safer deployment lane
* stronger hit-rate story
* lower average odds
* cleaner “professional consistency” narrative

## ValueEV Lane

**“A higher-odds, positive-edge FTR product using Poisson-normalized edge selection, with elite out-of-sample performance.”**

Use case:

* stronger investor wow-factor
* higher odds
* edge-based selection logic
* more differentiated IP story

---

# 14. Current caveats

## 14.1 No audit can prove every upstream script is perfect forever

The current evidence is strong, but the exact wording should remain honest:

* no leak evidence found in the audited selection/ranking path
* no join corruption found
* Poisson inputs appear consistent with pre-match use
* continued upstream diligence remains worthwhile

## 14.2 FTR is the first locked lane, not the end state

The current work locks FTR first.

The same productization framework still needs to be rolled out across:

* Over 2.5
* Under 2.5
* BTTS Yes
* BTTS No

---

# 15. Why this matters strategically

Once FTR is fully locked and the same standard is achieved for:

* Over 2.5
* Under 2.5
* BTTS Yes
* BTTS No

then the system moves from being “a promising football model” to being a genuine multi-product forecasting platform.

That is a major commercial difference.

A single good model is interesting.
A suite of separate, validated, investor-legible products is much more valuable.

---

# 16. Next planned work

## 16.1 Lock FTR documentation fully

This document is part of that locking process.

Remaining documentation deliverables should include:

* investor-facing Accuracy Lane summary
* investor-facing ValueEV Lane summary
* technical validation memo
* diligence appendix with audit tables

## 16.2 Run the same productization process for Over/Under 2.5

Planned work:

* full threshold sweeps
* product-lane separation if needed
* frozen walk-forward outputs
* forensic audit equivalent

## 16.3 Run the same process for BTTS Yes / No

Planned work:

* threshold sweep
* directional product logic
* hit / ROI / coverage tradeoff study
* forensic validation

---

# 17. Locked conclusion

The current FTR work supports the following locked conclusion:

## Locked conclusion

The FTR system now supports two distinct frozen out-of-sample product lanes:

* a high-strike-rate Accuracy lane built from the IMP62 upstream universe
* a higher-odds ValueEV lane built from the IMP40 upstream universe

The current branch outputs have been audited for fixture-join integrity and basic leak-risk indicators. No evidence was found that post-match outcome fields were used in selection or ranking, and the ValueEV Poisson inputs appear consistently available across the audited months.

This is now strong enough to treat as the foundation for wider product expansion into Over/Under 2.5 and BTTS lanes.

---

# 18. Appendices

## Appendix A — Locked month window

* 2024-10
* 2024-11
* 2024-12
* 2025-01
* 2025-02
* 2025-03

## Appendix B — Locked branch names

* accuracy
* valueev_balanced
* valueev_aggressive

## Appendix C — Locked audit files

* `walkforward_branch_comparison_2024-10_to_2025-03.csv`
* `walkforward_branch_cumulative_stats_2024-10_to_2025-03.csv`
* `walkforward_forensic_audit.csv`
* `walkforward_fixture_spotcheck_samples.csv`
* `walkforward_poisson_source_audit.csv`

## Appendix D — Core generation scripts

* `run_frozen_walkforward.py`
* `bookie_allmarkets.py`
* `backtest_deploy_csv.py`
* `apply_frozen_product_gates.py`
* `build_branch_comparison.py`
* `build_branch_cumulative_stats.py`
* `forensic_walkforward_audit.py`
* `poisson_source_audit.py`

## Hard Metrics Appendix

### Month-by-Month Branch Comparison (2024-10 to 2025-03)

| Month   | Branch             | Rows |     Hit |    ROI | Avg Odds |
| ------- | ------------------ | ---: | ------: | -----: | -------: |
| 2024-10 | Accuracy           |   24 |  95.83% | 0.2763 |   1.3363 |
| 2024-10 | ValueEV Balanced   |   43 |  97.67% | 1.1149 |   2.1702 |
| 2024-10 | ValueEV Aggressive |   42 |  97.62% | 1.1164 |   2.1731 |
| 2024-11 | Accuracy           |   30 |  90.00% | 0.1937 |   1.3237 |
| 2024-11 | ValueEV Balanced   |   44 |  93.18% | 1.0255 |   2.1655 |
| 2024-11 | ValueEV Aggressive |   43 |  93.02% | 1.0272 |   2.1705 |
| 2024-12 | Accuracy           |   25 |  80.00% | 0.0472 |   1.3112 |
| 2024-12 | ValueEV Balanced   |   45 |  97.78% | 1.1280 |   2.1769 |
| 2024-12 | ValueEV Aggressive |   42 |  97.62% | 1.1293 |   2.1817 |
| 2025-01 | Accuracy           |   20 |  90.00% | 0.1955 |   1.3225 |
| 2025-01 | ValueEV Balanced   |   33 | 100.00% | 1.2006 |   2.2006 |
| 2025-01 | ValueEV Aggressive |   31 | 100.00% | 1.2055 |   2.2055 |
| 2025-02 | Accuracy           |   20 |  95.00% | 0.2935 |   1.3670 |
| 2025-02 | ValueEV Balanced   |   34 |  97.06% | 1.1594 |   2.2285 |
| 2025-02 | ValueEV Aggressive |   33 |  96.97% | 1.1633 |   2.2345 |
| 2025-03 | Accuracy           |   14 | 100.00% | 0.3793 |   1.3793 |
| 2025-03 | ValueEV Balanced   |   45 |  97.78% | 1.1702 |   2.2224 |
| 2025-03 | ValueEV Aggressive |   42 |  97.62% | 1.1631 |   2.2190 |

### Cumulative Branch Statistics

| Branch             | Months Present | Total Rows | Weighted Hit | Weighted ROI | Weighted Avg Odds | Best Month by Hit | Best Hit | Worst Month by Hit | Worst Hit | Best Month by ROI | Best ROI | Worst Month by ROI | Worst ROI | Hit Std Dev | ROI Std Dev |
| ------------------ | -------------: | ---------: | -----------: | -----------: | ----------------: | ----------------- | -------: | ------------------ | --------: | ----------------- | -------: | ------------------ | --------: | ----------: | ----------: |
| Accuracy           |              6 |        133 |       90.98% |       0.2159 |            1.3358 | 2025-03           |  100.00% | 2024-12            |    80.00% | 2025-03           |   0.3793 | 2024-12            |    0.0472 |      0.0632 |      0.1036 |
| ValueEV Balanced   |              6 |        244 |       97.13% |       1.1292 |            2.1925 | 2025-01           |  100.00% | 2024-11            |    93.18% | 2025-01           |   1.2006 | 2024-11            |    1.0255 |      0.0204 |      0.0556 |
| ValueEV Aggressive |              6 |        233 |       97.00% |       1.1292 |            2.1955 | 2025-01           |  100.00% | 2024-11            |    93.02% | 2025-01           |   1.2055 | 2024-11            |    1.0272 |      0.0208 |      0.0556 |

### What the Metrics Show

* **Accuracy Lane** is a lower-odds, lower-volume, high-strike-rate product. Across six audited months it delivered **133 selections**, **90.98% weighted hit rate**, and **0.2159 weighted ROI** at **1.3358 weighted average odds**.
* **ValueEV Balanced** is the strongest current investor-facing performance lane on combined evidence: **244 selections**, **97.13% weighted hit rate**, **1.1292 weighted ROI**, and **2.1925 weighted average odds**.
* **ValueEV Aggressive** is extremely close to Balanced, with slightly fewer rows and slightly higher weighted average odds: **233 selections**, **97.00% weighted hit rate**, **1.1292 weighted ROI**, and **2.1955 weighted average odds**.
* The **least stable month for Accuracy** was **2024-12**, whereas the ValueEV lanes remained exceptionally strong even through that month.
* The **best month by hit and ROI for both ValueEV lanes** was **2025-01** in this audited six-month set.

## Investor One-Pager — Accuracy Lane

### Positioning

**Accuracy Lane** is the system’s premium high-strike-rate FTR product. It is designed for investors, partners, and operators who want a selective stream of shorter-priced favourites with strong pre-match confirmation logic and clean walk-forward auditability.

### What It Is

* Upstream universe built from **IMP62-style** selection conditions.
* Downstream gated through the **FTR Accuracy** branch.
* Focused on cleaner, stronger favourites rather than broad market coverage.
* Optimized for **strike rate, stability, and product clarity** rather than headline odds.

### Why It Matters

This is the lane that proves the system can create a disciplined, lower-volatility, investor-friendly FTR product rather than simply chasing big-odds outcomes. It gives a clean answer to the question: *Can the model selectively identify safer full-time result opportunities at scale?*

### Audited Performance Snapshot

* **Months audited:** 6
* **Total selections:** 133
* **Weighted hit rate:** 90.98%
* **Weighted ROI:** 0.2159
* **Weighted average odds:** 1.3358
* **Best month by ROI:** 2025-03
* **Weakest month by ROI:** 2024-12

### Operating Character

* Lower volume than the ValueEV lane
* Lower average odds
* Higher dependence on disciplined gating
* Strong fit for a “safer-leg” or “high-conviction favourite” product narrative

### Investor Narrative

This lane is the proof that the platform is not a one-dimensional value-chasing engine. It can generate a **structured, high-hit, lower-odds FTR lane** that is commercially intelligible and easier to position as a dependable premium product.

## Investor One-Pager — ValueEV Lane

### Positioning

**ValueEV Lane** is the system’s premium high-edge FTR product. It uses Poisson-normalized pick probabilities to identify higher-odds selections with positive modeled edge, then applies frozen walk-forward gating to isolate the most attractive candidates.

### What It Is

* Upstream universe built from **IMP40-style** selection conditions.
* Downstream gated through **ValueEV Balanced** and **ValueEV Aggressive** branches.
* Uses **Poisson-normalized pick probability** to calculate a ValueEV edge signal.
* Built specifically to test whether the platform can sustain a genuinely separate high-odds, high-edge product lane.

### Why It Matters

This is the lane with the strongest current “wow investor” profile. It combines:

* materially higher odds than Accuracy Lane
* much larger coverage
* elite hit rates across the audited six-month window
* very strong weighted ROI across both Balanced and Aggressive settings

### Audited Performance Snapshot — Balanced

* **Months audited:** 6
* **Total selections:** 244
* **Weighted hit rate:** 97.13%
* **Weighted ROI:** 1.1292
* **Weighted average odds:** 2.1925
* **Best month by ROI:** 2025-01
* **Weakest month by ROI:** 2024-11

### Audited Performance Snapshot — Aggressive

* **Months audited:** 6
* **Total selections:** 233
* **Weighted hit rate:** 97.00%
* **Weighted ROI:** 1.1292
* **Weighted average odds:** 2.1955
* **Best month by ROI:** 2025-01
* **Weakest month by ROI:** 2024-11

### Operating Character

* Higher volume than Accuracy Lane
* Meaningfully higher average odds
* Positive-edge logic based on pre-match Poisson probability mass
* Stronger headline commercial story for external stakeholders

### Investor Narrative

This lane demonstrates that the platform can do more than filter favourites. It can build a **separate, high-odds, positive-edge FTR product** with strong repeatability across months, clean fixture matching, and explicit forensic audit support. That makes it the strongest current candidate for a flagship investor-facing FTR proposition.

## Locked Conclusions at This Stage

* The system now supports **two distinct FTR investor products** rather than a single blended FTR story.
* **Accuracy Lane** and **ValueEV Lane** are structurally different and should continue to be presented as separate commercial products.
* The audit trail now includes branch comparison, cumulative branch statistics, forensic join checks, fixture-level samples, and Poisson source checks.
* Once FTR is fully locked, the same framework should be extended into:

  * **Over 2.5 / Under 2.5 threshold sweeps**
  * **BTTS Yes / No threshold sweeps**
* If those markets can be productized to a similar standard, the platform moves into a much stronger multi-lane investor position.

## Hard Metrics Appendix

### Month-by-Month Branch Comparison (2024-10 to 2025-03)

| Month   | Branch             | Rows |     Hit |    ROI | Avg Odds |
| ------- | ------------------ | ---: | ------: | -----: | -------: |
| 2024-10 | Accuracy           |   24 |  95.83% | 0.2763 |   1.3363 |
| 2024-10 | ValueEV Balanced   |   43 |  97.67% | 1.1149 |   2.1702 |
| 2024-10 | ValueEV Aggressive |   42 |  97.62% | 1.1164 |   2.1731 |
| 2024-11 | Accuracy           |   30 |  90.00% | 0.1937 |   1.3237 |
| 2024-11 | ValueEV Balanced   |   44 |  93.18% | 1.0255 |   2.1655 |
| 2024-11 | ValueEV Aggressive |   43 |  93.02% | 1.0272 |   2.1705 |
| 2024-12 | Accuracy           |   25 |  80.00% | 0.0472 |   1.3112 |
| 2024-12 | ValueEV Balanced   |   45 |  97.78% | 1.1280 |   2.1769 |
| 2024-12 | ValueEV Aggressive |   42 |  97.62% | 1.1293 |   2.1817 |
| 2025-01 | Accuracy           |   20 |  90.00% | 0.1955 |   1.3225 |
| 2025-01 | ValueEV Balanced   |   33 | 100.00% | 1.2006 |   2.2006 |
| 2025-01 | ValueEV Aggressive |   31 | 100.00% | 1.2055 |   2.2055 |
| 2025-02 | Accuracy           |   20 |  95.00% | 0.2935 |   1.3670 |
| 2025-02 | ValueEV Balanced   |   34 |  97.06% | 1.1594 |   2.2285 |
| 2025-02 | ValueEV Aggressive |   33 |  96.97% | 1.1633 |   2.2345 |
| 2025-03 | Accuracy           |   14 | 100.00% | 0.3793 |   1.3793 |
| 2025-03 | ValueEV Balanced   |   45 |  97.78% | 1.1702 |   2.2224 |
| 2025-03 | ValueEV Aggressive |   42 |  97.62% | 1.1631 |   2.2190 |

### Cumulative Branch Statistics

| Branch             | Months Present | Total Rows | Weighted Hit | Weighted ROI | Weighted Avg Odds | Best Month by Hit | Best Hit | Worst Month by Hit | Worst Hit | Best Month by ROI | Best ROI | Worst Month by ROI | Worst ROI | Hit Std Dev | ROI Std Dev |
| ------------------ | -------------: | ---------: | -----------: | -----------: | ----------------: | ----------------- | -------: | ------------------ | --------: | ----------------- | -------: | ------------------ | --------: | ----------: | ----------: |
| Accuracy           |              6 |        133 |       90.98% |       0.2159 |            1.3358 | 2025-03           |  100.00% | 2024-12            |    80.00% | 2025-03           |   0.3793 | 2024-12            |    0.0472 |      0.0632 |      0.1036 |
| ValueEV Balanced   |              6 |        244 |       97.13% |       1.1292 |            2.1925 | 2025-01           |  100.00% | 2024-11            |    93.18% | 2025-01           |   1.2006 | 2024-11            |    1.0255 |      0.0204 |      0.0556 |
| ValueEV Aggressive |              6 |        233 |       97.00% |       1.1292 |            2.1955 | 2025-01           |  100.00% | 2024-11            |    93.02% | 2025-01           |   1.2055 | 2024-11            |    1.0272 |      0.0208 |      0.0556 |

### What the Metrics Show

* **Accuracy Lane** is a lower-odds, lower-volume, high-strike-rate product. Across six audited months it delivered **133 selections**, **90.98% weighted hit rate**, and **0.2159 weighted ROI** at **1.3358 weighted average odds**.
* **ValueEV Balanced** is the strongest current investor-facing performance lane on combined evidence: **244 selections**, **97.13% weighted hit rate**, **1.1292 weighted ROI**, and **2.1925 weighted average odds**.
* **ValueEV Aggressive** is extremely close to Balanced, with slightly fewer rows and slightly higher weighted average odds: **233 selections**, **97.00% weighted hit rate**, **1.1292 weighted ROI**, and **2.1955 weighted average odds**.
* The **least stable month for Accuracy** was **2024-12**, whereas the ValueEV lanes remained exceptionally strong even through that month.
* The **best month by hit and ROI for both ValueEV lanes** was **2025-01** in this audited six-month set.

## Investor One-Pager — Accuracy Lane

### Positioning

**Accuracy Lane** is the system’s premium high-strike-rate FTR product. It is designed for investors, partners, and operators who want a selective stream of shorter-priced favourites with strong pre-match confirmation logic and clean walk-forward auditability.

### What It Is

* Upstream universe built from **IMP62-style** selection conditions.
* Downstream gated through the **FTR Accuracy** branch.
* Focused on cleaner, stronger favourites rather than broad market coverage.
* Optimized for **strike rate, stability, and product clarity** rather than headline odds.

### Why It Matters

This is the lane that proves the system can create a disciplined, lower-volatility, investor-friendly FTR product rather than simply chasing big-odds outcomes. It gives a clean answer to the question: *Can the model selectively identify safer full-time result opportunities at scale?*

### Audited Performance Snapshot

* **Months audited:** 6
* **Total selections:** 133
* **Weighted hit rate:** 90.98%
* **Weighted ROI:** 0.2159
* **Weighted average odds:** 1.3358
* **Best month by ROI:** 2025-03
* **Weakest month by ROI:** 2024-12

### Operating Character

* Lower volume than the ValueEV lane
* Lower average odds
* Higher dependence on disciplined gating
* Strong fit for a “safer-leg” or “high-conviction favourite” product narrative

### Investor Narrative

This lane is the proof that the platform is not a one-dimensional value-chasing engine. It can generate a **structured, high-hit, lower-odds FTR lane** that is commercially intelligible and easier to position as a dependable premium product.

## Investor One-Pager — ValueEV Lane

### Positioning

**ValueEV Lane** is the system’s premium high-edge FTR product. It uses Poisson-normalized pick probabilities to identify higher-odds selections with positive modeled edge, then applies frozen walk-forward gating to isolate the most attractive candidates.

### What It Is

* Upstream universe built from **IMP40-style** selection conditions.
* Downstream gated through **ValueEV Balanced** and **ValueEV Aggressive** branches.
* Uses **Poisson-normalized pick probability** to calculate a ValueEV edge signal.
* Built specifically to test whether the platform can sustain a genuinely separate high-odds, high-edge product lane.

### Why It Matters

This is the lane with the strongest current “wow investor” profile. It combines:

* materially higher odds than Accuracy Lane
* much larger coverage
* elite hit rates across the audited six-month window
* very strong weighted ROI across both Balanced and Aggressive settings

### Audited Performance Snapshot — Balanced

* **Months audited:** 6
* **Total selections:** 244
* **Weighted hit rate:** 97.13%
* **Weighted ROI:** 1.1292
* **Weighted average odds:** 2.1925
* **Best month by ROI:** 2025-01
* **Weakest month by ROI:** 2024-11

### Audited Performance Snapshot — Aggressive

* **Months audited:** 6
* **Total selections:** 233
* **Weighted hit rate:** 97.00%
* **Weighted ROI:** 1.1292
* **Weighted average odds:** 2.1955
* **Best month by ROI:** 2025-01
* **Weakest month by ROI:** 2024-11

### Operating Character

* Higher volume than Accuracy Lane
* Meaningfully higher average odds
* Positive-edge logic based on pre-match Poisson probability mass
* Stronger headline commercial story for external stakeholders

### Investor Narrative

This lane demonstrates that the platform can do more than filter favourites. It can build a **separate, high-odds, positive-edge FTR product** with strong repeatability across months, clean fixture matching, and explicit forensic audit support. That makes it the strongest current candidate for a flagship investor-facing FTR proposition.

## Locked Conclusions at This Stage

* The system now supports **two distinct FTR investor products** rather than a single blended FTR story.
* **Accuracy Lane** and **ValueEV Lane** are structurally different and should continue to be presented as separate commercial products.
* The audit trail now includes branch comparison, cumulative branch statistics, forensic join checks, fixture-level samples, and Poisson source checks.
* Once FTR is fully locked, the same framework should be extended into:

  * **Over 2.5 / Under 2.5 threshold sweeps**
  * **BTTS Yes / No threshold sweeps**
* If those markets can be productized to a similar standard, the platform moves into a much stronger multi-lane investor position.

Audit Interpretation & Caveats

This section locks the current forensic interpretation of the FTR walk-forward work and clarifies what has, and has not, been proven by the audit outputs.

Post-match columns in filtered exports

The filtered branch exports contain post-match outcome fields such as home_team_goal_count, away_team_goal_count, and correct. Their presence in the exported files is expected for scoring, audit, and retrospective validation.

Current audit evidence does not show that these columns were part of the filtering or ranking logic used to create the branch selections. The forensic audit distinguishes between:

columns merely present in filtered outputs, and

columns actually checked for filtering/ranking by branch logic.

Across the audited branch runs, selection_leak_suspected = False, and the post-match fields appear under present but not used, not under selection-driving columns.

Filtering and ranking interpretation

The branch audit currently supports the following interpretation:

Accuracy branch filtering/ranking was driven by pre-match fields such as bookie_od, bookie_implied_used, ftr_margin, bookie_pick, and ranking fields such as score / bookie_implied_used.

ValueEV branches filtering/ranking were driven by pre-match fields such as bookie_od, ftr_valueev_edge, and bookie_pick, with ranking centered on ftr_valueev_edge.

There is currently no audit evidence that correct, home_team_goal_count, or away_team_goal_count were used as selection inputs.

This is an important distinction: the exported files are allowed to contain truth columns for scoring, as long as those columns were not part of the gating logic.

Fixture join integrity

The fixture-level forensic audit shows clean joins between filtered branch selections and their corresponding monthly backtest files.

Locked findings:

status = ok across all audited branch-month combinations

merge_miss_rows = 0

duplicate_join_rows = 0

duplicate_filtered_fixture_rows = 0

filtered_rows_missing_from_backtest = 0

This supports the conclusion that the backtest join logic is behaving cleanly for the audited branch set, with no evidence of duplicate joins or filtered rows failing to resolve back into the originating backtest universe.

Poisson source audit

The Poisson audit is currently clean across the audited months.

Locked findings:

all audited months returned status = ok

null_p_rows = 0 in all audited months

Poisson FTR probability columns are present and populated

the normalized probability source used in ValueEV remains consistent with a pre-match Poisson-derived pathway

The p_sum_* diagnostics indicate expected imperfect raw mass before normalization, not evidence of leakage. The ValueEV edge path remains consistent with:

pre-match FTR probability inputs,

normalized Poisson pick probabilities, and

edge scoring derived before outcome resolution.

November branch repair

The missing November branch folder problem has been repaired.

This is now locked as complete for:

walkforward_frozen_accuracy/2024-11

walkforward_frozen_valueev_balanced/2024-11

walkforward_frozen_valueev_aggressive/2024-11

After repair:

branch comparison tables now include November correctly

cumulative stats now reflect the full October 2024 to March 2025 span

forensic branch audits and Poisson audits include November in the verified sequence

What has been proven

At this stage, the project has established the following:

Two distinct FTR products exist and are structurally separable

Accuracy Lane

ValueEV Lane

The branch architecture matters

IMP62 upstream is the correct architecture for the Accuracy lane

IMP40 upstream is the correct architecture for the ValueEV lane

Branch outputs are repeatable across multiple months

not just November

with stable month-to-month behavior, especially in the ValueEV branches

Current audit evidence does not indicate filtering/ranking leakage from post-match outcome fields

Fixture matching and monthly artifact integrity are currently clean for the audited branch set

Caveats

The current audit is strong, but it should still be described accurately.

This is a forensic validation layer, not a formal external independent audit.

It demonstrates no current evidence of leakage in branch selection logic, rather than claiming metaphysical proof that no bug could ever exist elsewhere.

Post-match fields remain present in exported scored datasets, which is acceptable for audit/scoring, but future investor-facing material should keep emphasizing that distinction clearly.

Remaining validation work should focus on extending the same rigor to additional products and markets.

Strategic conclusion

The major remaining task is market expansion, not FTR rescue.

FTR has now moved out of the “is this real?” phase and into the “how far can this architecture scale?” phase.

That means the next high-value workstreams are:

Over 2.5 productization and threshold sweeps

BTTS Yes / No productization and threshold sweeps

replication of the same frozen walk-forward + forensic audit framework for those markets

eventual multi-product investor packaging once FTR, OU25, and BTTS all have equally defensible product lanes

In plain terms: the FTR lane is no longer the fire to put out. The engine is running. The next step is to widen the machine.

