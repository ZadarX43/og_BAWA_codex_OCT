# FTR Deploy Rulebook (Walk‑Forward Locked Reference)

> **Scope:** Full‑Time Result (FTR) productization for Odds Genius / BAWA.
>
> **Status:** Locked findings + deploy decision logic (two investor‑facing lanes).
>
> **Primary artifacts referenced:** Frozen month walk‑forward outputs (2024‑10 → 2025‑03), branch comparison + cumulative stats, forensic join audit, Poisson source audit.

---

## 0) Why this file exists

This file is the **single source of truth** for:

- what the FTR system is (and is not)
- the two validated FTR product lanes
- the branch architecture and the months audited
- the scripts and commands that produced the locked results
- the forensic interpretation (join integrity + leakage checks)
- the deployment decision logic you should use in production

This mirrors the **OU25/BTTS productization rulebooks** you’ve already locked.

---

## 1) Locked conclusion

The FTR system supports **two distinct frozen out‑of‑sample product lanes**:

- **Accuracy Lane** (high strike‑rate, lower odds): **best when upstream = IMP62**
- **ValueEV Lane** (higher odds + modeled edge): **best when upstream = IMP40**

The audited branch outputs show:

- clean fixture joins (no duplicate join inflation, no merge misses)
- filtered rows map cleanly back to their scored backtest universes
- Poisson FTR probability columns used by ValueEV are present and stable
- no evidence that post‑match outcome fields were used for selection or ranking  
  (post‑match fields can exist in exports for scoring; they must not drive gating)

---

## 2) Product lanes

### 2.1 Accuracy Lane (investor product A)

**Positioning:** disciplined favourite selection for strike‑rate and stability.

- Upstream: `IMP62`
- Gate philosophy: “safe leg”, moderate odds ceiling, strong confidence discipline
- Frozen tests: home/away‑only configuration used

**Representative hard constraints (from frozen gate runs):**
- `--ftr-max-od 1.85`
- `--top-q 0.7`
- `--ftr-home-away-only`

### 2.2 ValueEV Lane (investor product B)

**Positioning:** higher‑odds positive‑edge FTR product using Poisson‑normalized pick probabilities.

- Upstream: `IMP40`
- Gate philosophy: allow higher odds, require ValueEV edge, rank by `ftr_valueev_edge`

**Balanced vs Aggressive:**
- Balanced: `--ftr-valueev-edge-min 1.05`
- Aggressive: `--ftr-valueev-edge-min 1.08`
- Both used: `--ftr-valueev-od-min 1.8`

---

## 3) Branch architecture (locked)

Branches tested:

- `accuracy` → root `walkforward_frozen_accuracy/` → upstream `IMP62`
- `valueev_balanced` → root `walkforward_frozen_valueev_balanced/` → upstream `IMP40`
- `valueev_aggressive` → root `walkforward_frozen_valueev_aggressive/` → upstream `IMP40`

---

## 4) Audited month window (locked)

The current locked comparison window:

- 2024‑10
- 2024‑11
- 2024‑12
- 2025‑01
- 2025‑02
- 2025‑03

Six months, cross‑season (autumn → winter → early spring).

---

## 5) Scripts used (locked)

### Core generation
- `run_frozen_walkforward.py`  
  Frozen month walk‑forward runner (uses production ModelStore; no retraining).
- `bookie_allmarkets.py`  
  Generates the raw ALLMARKETS deployment universe for the target month.
- `backtest_deploy_csv.py`  
  Joins generated deployment rows to actual results; produces scored backtest CSVs.
- `apply_frozen_product_gates.py`  
  Applies product gates directly to scored backtest (bypasses old CLI path).

### Audit / aggregation
- `build_branch_comparison.py`
- `build_branch_cumulative_stats.py`
- `forensic_walkforward_audit.py`
- `poisson_source_audit.py`

---

## 6) Main output files (locked)

- `walkforward_branch_comparison_2024-10_to_2025-03.csv`
- `walkforward_branch_cumulative_stats_2024-10_to_2025-03.csv`
- `walkforward_forensic_audit.csv`
- `walkforward_fixture_spotcheck_samples.csv`
- `walkforward_poisson_source_audit.csv`

---

## 7) How the walk‑forward works (repeatable recipe)

1) **Generate monthly universe**  
   `bookie_allmarkets.py` writes ALLMARKETS using frozen production models.

2) **Score universe**  
   `backtest_deploy_csv.py` joins to truth and writes `__BACKTEST.csv`.

3) **Apply frozen gates**  
   `apply_frozen_product_gates.py` applies lane‑specific FTR gates.

4) **Archive per‑month artifacts**  
   Month folder receives gated CSVs + JSON summary.

5) **Aggregate across months**  
   Comparison + cumulative stats scripts compile the master story.

6) **Run forensics**  
   Join integrity + “used columns” leakage indicators + Poisson source checks.

---

## 8) Locked gate command patterns

### 8.1 Accuracy lane (IMP62 upstream)

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

### 8.2 ValueEV Balanced (IMP40 upstream)

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

### 8.3 ValueEV Aggressive (IMP40 upstream)

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

## 9) Hard metrics (locked snapshot)

### Month‑by‑Month Branch Comparison (2024‑10 → 2025‑03)

| Month   | Branch             | Rows | Hit     | ROI    | Avg Odds |
| ------- | ------------------ | ---: | -------:| ------:| -------: |
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

| Branch             | Months | Rows | Weighted Hit | Weighted ROI | Weighted Avg Odds |
| ------------------ | -----: | ---: | -----------: | -----------: | ----------------: |
| Accuracy           |      6 |  133 |       90.98% |       0.2159 |            1.3358 |
| ValueEV Balanced   |      6 |  244 |       97.13% |       1.1292 |            2.1925 |
| ValueEV Aggressive |      6 |  233 |       97.00% |       1.1292 |            2.1955 |

---

## 10) Forensic interpretation (what was proven)

### 10.1 Post‑match columns present vs used

Filtered exports contain post‑match fields (e.g., `home_team_goal_count`, `away_team_goal_count`, `correct`) **for scoring**.

Audit evidence indicates these fields were **not used** for selection/ranking in the gating logic.  
The audit distinguishes: “present in export” vs “used for selection”.

### 10.2 Fixture join integrity (locked)

Across audited branch‑months:

- merge misses: 0
- duplicate join rows: 0
- filtered rows missing from backtest: 0
- duplicate filtered fixtures: 0

### 10.3 Poisson source audit (ValueEV)

Across audited months:

- Poisson columns present and populated
- `null_p_rows = 0`
- probability mass diagnostics consistent with pre‑match Poisson + normalization
- no evidence of leakage

---

## 11) Production deployment decision logic

### 11.1 Key concept: FTR has **two products**, not one

In production you must treat these as separate “lanes”:

- `FTR_ACCURACY` (IMP62 universe + max‑od + TopQ + home/away restrictions)
- `FTR_VALUEEV_BALANCED` (IMP40 + od‑min + edge‑min + rank by edge)
- `FTR_VALUEEV_AGGRESSIVE` (same, stricter edge‑min)

### 11.2 Where do caches come from?

Caches must be built from the same distribution you are ranking against:

- usually a **rolling window** (e.g., last N weeks) *per league* and *per lane*
- or a **frozen backtest window** (for validation)
- do **not** silently mix seasons / model stores

In production, you typically compute per‑league thresholds on the **current run’s candidate universe** (ALLMARKETS output), so the selection remains stationary by league and by lane.

---

## 12) Next steps (upstream plugin mission)

You already did this for BTTS + OU25:
- `bookie_allmarkets.py`
- `deploy_gates.py`
- `deploy_rulebooks.py`
- `bookie_*_allmarkets` exporters and UI tags/reasons

Now do the same for FTR:

1) Define FTR deploy decision functions (one per lane)
2) Add lane‑specific caches (TopQ + edge thresholds)
3) Wire reasons for the UI (“ValueEV lane + edge≥1.05 + od≥1.80 + TopQ”)
4) Add regression guardrails: join integrity + used‑columns audit hooks

---

## Appendix: Investor one‑pagers (verbatim framing)

### Accuracy Lane
A higher‑strike‑rate FTR product optimized around disciplined favourite selection.

### ValueEV Lane
A higher‑odds, positive‑edge FTR product using Poisson‑normalized edge selection, with elite out‑of‑sample performance.

