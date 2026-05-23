# OU25 Walk-Forward Summary

## Title
Frozen OU25 walk-forward validation across 6 out-of-sample months

## Investor summary
The OU25 frozen branch family passed 6-month out-of-sample walk-forward validation and remained profitable at branch level across all five locked candidates.

However, the league audit shows that OU25 performance is not uniform across competitions. The edge is strongest when deployed selectively into the best-performing leagues and materially weaker in a small group of structurally poor leagues.

**Locked conclusion:**
- Main live OU25 branch: `ou25_band1_124_176`
- Premium / selective OU25 branch: `ou25_combined_topq_080`
- Control benchmark retained: `ou25_combined_baseline`
- Live deployment should be **league-filtered**, not league-agnostic

---

## Change log

### v2
- locked 6-month walk-forward branch leaderboard
- integrated league-level walk-forward audit
- added live deployment whitelist / blacklist
- added final live deployment recommendation

### v1
- branch-only walk-forward summary
- initial locked branch decision

---

## Purpose
This document records the out-of-sample walk-forward validation results for the locked OU25 frozen branches.

The goal was to determine which OU25 branch should be:
- retained as the baseline control,
- promoted as the main deploy candidate,
- promoted as the premium / selective branch,
- or deprioritised.

This validation is intended to sit alongside the existing FTR frozen walk-forward process without altering or weakening the FTR pipeline.

---

## Scope

### Walk-forward window
Validated across 6 monthly walk-forward periods.

### Branches tested
- `ou25_combined_baseline`
- `ou25_combined_topq_080`
- `ou25_mode_over_only`
- `ou25_band2_178_195`
- `ou25_band1_124_176`

### Metrics tracked
- total rows
- weighted hit rate
- weighted ROI
- weighted average odds
- profitable months
- losing months
- month-to-month stability
- bad month profile
- max drawdown

---

## Validation status

### Smoke-test status
Passed.

### Frozen branch comparison status
Passed.

### Forensic audit status
Passed.

### Walk-forward status
Passed across 6 months.

All five locked OU25 candidate branches remained profitable across the full walk-forward window.

### League audit status
Passed.

League-level walk-forward review confirms that the branch edge is real, but concentrated. The best deployment state is selective and league-aware.

---

## Final leaderboard

| branch | months_present | total_rows | weighted_hit | weighted_roi | weighted_avg_odds | profitable_months | losing_months | max_drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ou25_combined_topq_080 | 6 | 516 | 0.817829 | 0.428265 | 1.757015 | 6 | 0 | 0.0 |
| ou25_band1_124_176 | 6 | 829 | 0.820265 | 0.416374 | 1.737524 | 6 | 0 | 0.0 |
| ou25_band2_178_195 | 6 | 865 | 0.802312 | 0.400224 | 1.757276 | 6 | 0 | 0.0 |
| ou25_combined_baseline | 6 | 737 | 0.810041 | 0.395294 | 1.733983 | 6 | 0 | 0.0 |
| ou25_mode_over_only | 6 | 516 | 0.792636 | 0.361550 | 1.730155 | 6 | 0 | 0.0 |

---

## Interpretation

### 1) Best pure ROI branch
`ou25_combined_topq_080`

This branch produced the highest weighted walk-forward ROI:
- **ROI:** `0.428265`
- **Hit:** `0.817829`

It is the strongest premium / selective branch.
It sacrifices volume for quality and appears suitable as a higher-confidence filtered product lane.

### 2) Best all-round deployment branch
`ou25_band1_124_176`

This branch produced the best overall balance between:
- row volume,
- hit rate,
- ROI,
- and stability.

Key results:
- **Rows:** `829`
- **Hit:** `0.820265`
- **ROI:** `0.416374`

This is the strongest candidate for the default OU25 production branch.

### 3) Strong alternate branch
`ou25_band2_178_195`

This remained profitable and robust, with slightly more volume than `band1_124_176`, but weaker hit and ROI.

It remains a viable secondary branch but does not currently lead the pack.

### 4) Baseline control branch
`ou25_combined_baseline`

The baseline branch remained strong:
- profitable in all 6 months,
- good hit rate,
- good ROI,
- clean control benchmark.

It should remain in the repo as the reference baseline for future branch comparisons.

### 5) Over-only branch
`ou25_mode_over_only`

This branch underperformed the combined and widened-band variants in walk-forward.

It was still profitable across all months, but it is no longer the preferred main lane.

Current interpretation:
- not broken,
- not useless,
- but not the best production candidate.

---

## Stability read

### Month-level stability
All 5 branches:
- had **6 profitable months out of 6**
- had **0 losing months**
- showed **0.0 max drawdown** over the current walk-forward window

This is important.

The current OU25 branch family is not just backtest-positive.
It is now supported by a real out-of-sample walk-forward result.

### Worst-month profile
Worst monthly ROI by branch:

- `ou25_combined_topq_080`: `0.360628`
- `ou25_band1_124_176`: `0.324337`
- `ou25_band2_178_195`: `0.315276`
- `ou25_combined_baseline`: `0.330261`
- `ou25_mode_over_only`: `0.243176`

Even the worst month remained profitable for every branch.

---

## League audit read

The league audit materially changes the deployment picture.

At branch level, every OU25 candidate is profitable across the 6-month walk-forward window.
At league level, the edge is clearly clustered.

There are leagues where OU25 is consistently elite across branches, and there are leagues where OU25 is persistently weak or outright negative.

This means the correct live posture is:
- **branch selection first**
- **league filtering second**
- **no blind all-league deployment**

---

## League whitelist

These leagues showed the strongest repeatable OU25 walk-forward behaviour and should be treated as the primary live deployment pool.

### Core whitelist
- `England Championship`
- `Germany Bundesliga`
- `Italy Serie A`
- `Portugal Liga`
- `Champions League`
- `Europa League`

### Secondary whitelist
- `USA MLS`
- `Norway Eliteserien`
- `Netherlands Eredivisie`
- `Belgium Pro`

### Conditional / watchlist deploy only
These leagues have positive branches available, but the evidence is either lower-sample, more volatile, or mixed enough that they should be watched rather than treated as unconditional mainline leagues.

- `Japan J1`
- `France Ligue 1`
- `Europa Conference`
- `Brazil Serie A`
- `Spain La Liga`
- `England EFL League 1`

---

## League blacklist

These leagues should not currently be trusted for broad live OU25 deployment.

### Hard blacklist
- `England Premier League`
- `Scotland Premiership`

### Soft blacklist / avoid for now
- `England FA Cup`

Reason:
- repeated negative or unstable league-level walk-forward outcomes
- multiple branches breaking down in the same competition
- poor worst-month behaviour
- unacceptable live variance for current OU25 deployment standards

---

## Decision taken

### Main OU25 production candidate
**`ou25_band1_124_176`**

Reason:
- highest all-round quality,
- strongest hit/ROI balance,
- larger card than topq_080,
- better suited for main deployment.

### Premium / selective OU25 candidate
**`ou25_combined_topq_080`**

Reason:
- highest walk-forward ROI,
- best premium card profile,
- strongest choice when selectivity is preferred over volume.

### Control benchmark to retain
**`ou25_combined_baseline`**

Reason:
- stable benchmark,
- useful control branch,
- essential for future comparison.

### Branch deprioritised
**`ou25_mode_over_only`**

Reason:
- still profitable,
- but materially weaker than the current best combined branches,
- should not be the primary default OU25 lane.

---

## Final live deployment recommendation

### Recommended live default
Use **`ou25_band1_124_176`** as the main live OU25 branch, but only inside the whitelist / approved league set.

### Recommended premium lane
Use **`ou25_combined_topq_080`** as the premium filtered lane for users who want a tighter, higher-selectivity OU25 product.

### Do not deploy live as all-league OU25
Do **not** deploy OU25 as a universal all-league product yet.

The league audit shows this would dilute the edge and reintroduce avoidable failure states.

### Recommended live trust order
1. `ou25_band1_124_176` on core whitelist leagues
2. `ou25_combined_topq_080` on core whitelist leagues as premium lane
3. `ou25_combined_baseline` retained for control and monitoring
4. secondary whitelist leagues only after continued monitoring
5. blacklisted leagues excluded until materially improved by further walk-forward evidence

---

## Repo lock

The current locked OU25 decision is:
- **default branch:** `ou25_band1_124_176`
- **premium branch:** `ou25_combined_topq_080`
- **control branch:** `ou25_combined_baseline`
- **league-aware deployment:** required
- **hard avoid:** `England Premier League`, `Scotland Premiership`
- **soft avoid:** `England FA Cup`

---

## Caveats

1. This result is based on the current 6-month walk-forward window.
2. More months should still be generated and replayed when available.
3. Several leagues remain sample-sensitive and should not be overinterpreted.
4. League whitelist / blacklist status should be rechecked after each new walk-forward month.
5. This process is designed to coexist with the FTR walk-forward system, not replace it.

---

## Related files

### Comparison outputs
- `ou25_walkforward_comparison.csv`
- `ou25_walkforward_comparison.md`
- `ou25_walkforward_league_audit.csv`
- `ou25_walkforward_league_audit.md`

### Upstream runner / pipeline
- `run_frozen_walkforward.py`
- `run_ou25_walkforward.sh`
- `build_ou25_walkforward_comparison.py`
- `build_ou25_walkforward_league_audit.py`

### Prior validation
- OU25 frozen sweep outputs
- OU25 forensic audit outputs
- OU25 branch comparison leaderboard
- OU25 cumulative branch stats

---

## Next steps

### Immediate
- keep `ou25_band1_124_176` as main deploy candidate
- keep `ou25_combined_topq_080` as premium lane
- preserve `ou25_combined_baseline` as benchmark
- enforce the current league whitelist / blacklist in live OU25 deployment logic

### Next build items
- OU25 live whitelist file / config artifact
- OU25 month-by-month deployment dashboard
- extended walk-forward horizon
- automatic league promotion / demotion rules
- live deployment candidate report