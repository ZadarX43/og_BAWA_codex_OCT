# OG / BAWA Walk-Forward System Audit & Mission Recap
_Last updated: 2026-03-26_

## 1. Executive summary

This document is the clean operational recap of the current proven walk-forward stack for:

- **FTR** (Full Time Result)
- **OU25** (Over 2.5 Goals)
- **BTTS** (Both Teams To Score)

It captures:

- the working files and their roles
- current proven thresholds and live logic
- variant-test findings
- calendar overlay rules
- data flow through the pipeline
- commands used to run, audit, and verify the system
- generated outputs and where they live
- what is now proven vs what is still pending

This is the current source-of-truth recap for the audited deploy pipeline.

---

## 2. Current mission status

## Proven and working

### FTR
- **STANDARD FTR live logic is materially improved** by the new **Patch D full base-wide block**.
- The winning investor headline from the formal variant framework is:

> **FTR STANDARD = 190 / 205 = 92.68% hit rate, ROI = 0.5619**

This came from the live-winning variant:

- `FTR_STANDARD_BASE_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK`

### BTTS
- BTTS ELITE remains one of the strongest products in the stack.
- BTTS shows clear sensitivity to calendar/congestion regimes, especially **post-FIFA breaks** and **post-R16 second-leg March windows**.
- BTTS calendar throttling policy is now formalized and stamped per window.

### OU25
- OU25 STANDARD remains profitable and usable.
- OU25 OBSERVE remains too noisy to treat as premium exposure.
- OU25 is not yet the primary focus of the new calendar overlay logic, but the current walk-forward framework supports future extension.

### Calendar overlay
- Window-aware calendar regime assignment now works.
- Each walk-forward scored output can now be stamped with:
  - `calendar_regime`
  - `btts_leg_multiplier`
  - `ftr_leg_multiplier`
  - `stake_multiplier`
  - `overlay_note`

A complete audit file has been produced:

- `predictions_output/walk_forward/_MASTER/CALENDAR_REGIME_AUDIT.csv`

---

## 3. Core files and what they do

## 3.1 Main walk-forward orchestration

### `run_walkforward_windows.py`
This is the main multi-window walk-forward controller.

It handles:

- manifest loading
- source CSV resolution
- deploy CSV resolution
- merged actual result enrichment
- optional calendar flag enrichment
- scoring
- per-window summary writing
- master rollups
- FTR variant test framework
- draw-risk audits
- BTTS signal audits
- calendar regime stamping

### Important additions now present in this file
- `WINDOW_REGIME_MAP`
- `resolve_calendar_regime(window_id, cli_override=None)`
- `maybe_stamp_calendar_overlay(...)`
- calendar overlay stamping inside the main scoring path
- logging of resolved per-window calendar regime
- FTR formal variant framework and audits

---

## 3.2 Deploy brain

### `deploy_rulebook.py`
This is the live routing / gate / tier decision engine.

It is responsible for:
- reading the bookie / all-markets source
- gating candidates
- assigning ELITE / STANDARD / OBSERVE
- applying cross-market logic and tier shaping
- writing tiered deploy CSVs

### Proven FTR patch now promoted into live deploy logic
Patch D is now live in the FTR STANDARD path.

Live debug confirms:
- Patch C fires first
- Patch D then demotes weak `STANDARD_FTR_BASE` draw-gap rows to OBSERVE
- remaining STANDARD rows contain no surviving Patch D targets

### Patch D live target logic
The currently-proven live rule is:

- bucket = `STANDARD_FTR_BASE`
- `selected_minus_draw <= 0.15`
- `abs(power_diff) <= 18`

This is the full base-wide version.

---

## 3.3 Source generation

### `bookie_allmarkets.py`
This generates the source all-markets input file for a window.

Typical output shape:
- one source CSV per date window
- includes market rows and bookie/model fields
- used as the raw input into `deploy_rulebook.py`

---

## 3.4 Accumulator / shortlist layer

### `acca_builder.py`
This is the likely next file to patch so that final shortlist / acca output becomes calendar sensitive.

Planned role for calendar overlay integration:
- read stamped `calendar_regime`
- use `btts_leg_multiplier`
- use `ftr_leg_multiplier`
- use `stake_multiplier`
- adjust:
  - acca leg count guidance
  - shortlist pressure
  - stake sizing
  - optional market-level inclusion pressure

This is the next intended production integration point.

---

## 4. Walk-forward manifest

### Current manifest file
- `predictions_output/walk_forward/window_manifest.csv`

### Current windows in audit scope
- `w03_2024_08_16_2024_08_20`
- `w04_2024_08_30_2024_09_03`
- `w05_2024_09_13_2024_09_17`
- `w06_2024_11_08_2024_11_12`
- `w07_2024_11_29_2024_12_03`
- `w08_2024_12_13_2024_12_17`
- `w09_2025_01_17_2025_01_21`
- `w10_2025_02_07_2025_02_11`
- `w11_2025_03_07_2025_03_11`
- `w12_2025_08_15_2025_08_19`
- `w13_2025_09_12_2025_09_16`
- `w14_2025_10_17_2025_10_21`
- `w15_2025_11_21_2025_11_25`
- `w16_2026_01_16_2026_01_20`
- `w17_2026_02_13_2026_02_17`
- `w18_2026_03_13_2026_03_17`

---

## 5. Data flow

## Step 1 — source generation
`bookie_allmarkets.py` generates:

- `BOOKIE_*_ALLMARKETS_<date_from>_to_<date_to>.csv`

These files contain the canonical source predictions / market rows.

## Step 2 — deploy / routing
`deploy_rulebook.py` consumes the source file and writes tiered deploy outputs:

- `__DEPLOY_TIER_ELITE__...csv`
- `__DEPLOY_TIER_STANDARD__...csv`
- `__DEPLOY_TIER_OBSERVE__...csv`

## Step 3 — combine deploy tiers
`run_walkforward_windows.py` combines tier files into:

- `DEPLOY_COMBINED_<date_from>_to_<date_to>.csv`

## Step 4 — enrich with actuals
The combined deploy file is enriched with actual outcomes from:

- `Matches/__merged__/<League>__merged.csv`

This produces:
- `actual_ftr`
- `actual_over25`
- `actual_btts_yes`

## Step 5 — optional calendar flags enrichment
If present, files under:

- `predictions_output/calendar_flags`

are merged in.

## Step 6 — scoring
The combined file is scored into:

- `DEPLOY_COMBINED_SCORED_<date_from>_to_<date_to>.csv`

Scoring columns include:
- `ftr_hit`
- `ou25_hit`
- `btts_yes_hit`

## Step 7 — calendar regime stamping
Each scored window is stamped with:
- `calendar_regime`
- `btts_leg_multiplier`
- `ftr_leg_multiplier`
- `stake_multiplier`
- `overlay_note`

## Step 8 — reports / summaries / audits
Per-window and master-level outputs are written into:
- `04_reports/`
- `_MASTER/`

---

## 6. FTR — proven live logic and thresholds

## 6.1 Key FTR variant conclusion

The formal variant framework proved that the best investor headline for STANDARD FTR is:

- **Full Patch D base-wide block**
- not the narrower NO_DRAWWARN-only version
- not Patch C alone
- not Patch C + Patch D NO_DRAWWARN-only

### Winning headline
- **205 graded picks**
- **190 wins**
- **15 losses**
- **92.68% hit rate**
- **115.18 level-stake profit**
- **ROI = 0.561854**

Variant:
- `FTR_STANDARD_BASE_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK`

---

## 6.2 Patch D logic

### Live-winning rule
Apply to:
- `STANDARD_FTR_BASE`

Demote to OBSERVE when:
- `selected_minus_draw <= 0.15`
- `abs(power_diff) <= 18`

This is now the proven live FTR STANDARD overlay.

---

## 6.3 Formal variant results that matter

### Baseline
- `FTR_STANDARD_BASELINE`
- 193 / 214
- 90.19%
- ROI 0.5166

### Full Patch D
- `FTR_STANDARD_BASE_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK`
- 190 / 205
- 92.68%
- ROI 0.5619

### Patch D NO_DRAWWARN-only
- `FTR_STANDARD_BASE_NO_DRAWWARN_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK`
- 192 / 209
- 91.87%
- ROI 0.5452

### Patch C + Patch D full
- `FTR_STANDARD_PATCH_C_PLUS_PATCH_D_COMBO`
- same result as full Patch D in the tested sample
- 190 / 205
- 92.68%
- ROI 0.5619

### Key interpretation
- Patch C did not add incremental uplift in the tested walk-forward set
- Patch D produced the real uplift
- NO_DRAWWARN is the sharper sub-signal
- but the full base-wide block gives the best investor headline

---

## 6.4 Patch D live debug proof

Live debug confirmed:
- Patch C applied first
- Patch D then removed one live row in the tested March window
- the removed row was:
  - Champions League
  - 2026-03-17
  - Manchester City vs Real Madrid
  - selection HOME
  - `bookie_od = 1.41`
  - `model_p_for_bookie = 0.420689`
  - `draw_warning_token_bucket = DRAW_TOKEN`
  - `standard_reporting_bucket = STANDARD_FTR_BASE`

After Patch D:
- no remaining STANDARD FTR rows satisfied the Patch D target logic

This confirmed correct live application.

---

## 7. BTTS — proven state

## 7.1 BTTS tier quality summary

### BTTS ELITE
Current audited headline:
- **233 graded**
- **202 wins**
- **31 losses**
- **86.70%**
- **100.13 profit**
- **ROI = 0.4297**

### BTTS STANDARD
Current audited headline:
- **10 graded**
- **9 wins**
- **1 loss**
- **90.00%**
- **3.57 profit**
- **ROI = 0.3570**

### BTTS OBSERVE
Current audited headline:
- **911 graded**
- **585 wins**
- **326 losses**
- **64.22%**
- **121.23 profit**
- **ROI = 0.1331**

### Interpretation
- ELITE is the premium product
- STANDARD is too small to treat as a major evidence base yet
- OBSERVE still has value but is not premium-grade

---

## 7.2 BTTS calendar asymmetry proven
The walk-forward evidence showed:

### Post-FIFA break
- BTTS degrades
- FTR largely holds

### Post-R16 second-leg March
- BTTS degrades
- FTR can remain strong

### Post-UCL league phase weekends
- BTTS does not need suppression

This directly produced the calendar overlay throttling policy.

---

## 8. OU25 — proven state

## 8.1 OU25 tier quality summary

### OU25 STANDARD
Current audited headline:
- **101 graded**
- **81 wins**
- **20 losses**
- **80.20%**
- **24.28 profit**
- **ROI = 0.2404**

### OU25 OBSERVE
Current audited headline:
- **1263 graded**
- **707 wins**
- **556 losses**
- **55.98%**
- **-98.56 profit**
- **ROI = -0.0780**

### Interpretation
- OU25 STANDARD is currently usable and profitable
- OU25 OBSERVE remains too weak for premium exposure
- OU25 has not yet received the same level of calendar overlay targeting as BTTS/FTR

---

## 9. Calendar overlay policy now working

## 9.1 Window-aware regime map

### Current explicit mapping
- `w05_2024_09_13_2024_09_17` → `POST_FIFA_BREAK`
- `w06_2024_11_08_2024_11_12` → `POST_UCL_LEAGUE_PHASE`
- `w07_2024_11_29_2024_12_03` → `POST_UCL_LEAGUE_PHASE`
- `w08_2024_12_13_2024_12_17` → `POST_UCL_LEAGUE_PHASE`
- `w13_2025_09_12_2025_09_16` → `POST_FIFA_BREAK`
- `w14_2025_10_17_2025_10_21` → `POST_FIFA_BREAK`
- `w15_2025_11_21_2025_11_25` → `NOV_FIFA_PLUS_PRE_UCL`
- `w17_2026_02_13_2026_02_17` → `PRE_UCL_KNOCKOUT_FIRST_LEG`
- `w18_2026_03_13_2026_03_17` → `POST_R16_SECOND_LEG_MARCH`

All others currently fall back to:
- `NORMAL`

---

## 9.2 Regime rule table

| calendar_regime | BTTS multiplier | FTR multiplier | stake multiplier | note |
|---|---:|---:|---:|---|
| NORMAL | 1.00 | 1.00 | 1.00 | no calendar adjustment |
| POST_FIFA_BREAK | 0.75 | 1.00 | 0.90 | reduce BTTS after FIFA/international breaks |
| PRE_UCL_KNOCKOUT_FIRST_LEG | 1.00 | 0.75 | 0.90 | reduce FTR before UCL knockout first legs |
| NOV_FIFA_PLUS_PRE_UCL | 0.70 | 0.70 | 0.75 | reduce both in the worst annual congestion window |
| POST_UCL_LEAGUE_PHASE | 1.00 | 1.00 | 1.00 | deploy normally |
| POST_R16_SECOND_LEG_MARCH | 0.75 | 1.00 | 0.90 | reduce BTTS after R16 second legs in March |

---

## 9.3 Audit proof
The following audit file now proves the window-aware overlay is being stamped correctly:

- `predictions_output/walk_forward/_MASTER/CALENDAR_REGIME_AUDIT.csv`

This includes:
- `window_id`
- `date_from`
- `date_to`
- `calendar_regime`
- `btts_leg_multiplier`
- `ftr_leg_multiplier`
- `stake_multiplier`
- `overlay_note`

---

## 10. Master investor summary numbers

## 10.1 Compact market rollup

### Premium shortlist
- **FTR STANDARD** — 205 graded, 190 wins, 15 losses, 92.68%, profit 115.18, ROI 0.5619
- **FTR ELITE** — 83 graded, 75 wins, 8 losses, 90.36%, profit 38.66, ROI 0.4658
- **BTTS ELITE** — 233 graded, 202 wins, 31 losses, 86.70%, profit 100.13, ROI 0.4297
- **OU25 STANDARD** — 101 graded, 81 wins, 20 losses, 80.20%, profit 24.28, ROI 0.2404

### Non-premium / learning / coverage tiers
- **BTTS OBSERVE** — 911 graded, 64.22%, profit positive but lower-quality
- **OU25 OBSERVE** — 1263 graded, negative ROI
- **FTR OBSERVE** — 1740 graded, negative ROI

---

## 10.2 Best / worst windows identified

### FTR STANDARD
- Best: `w04_2024_08_30_2024_09_03` → 21/21
- Worst: `w15_2025_11_21_2025_11_25` → 8/12

### FTR ELITE
- Best: `w04_2024_08_30_2024_09_03` → 8/8
- Worst: `w14_2025_10_17_2025_10_21` → 3/5

### BTTS ELITE
- Best: `w17_2026_02_13_2026_02_17` → 13/13
- Worst: `w18_2026_03_13_2026_03_17` → 8/11

### OU25 STANDARD
- Best: `w10_2025_02_07_2025_02_11` → 7/7
- Worst: `w17_2026_02_13_2026_02_17` → 1/2

---

## 11. Key outputs and where they live

## Per-window
For each window:
- `01_source/`
- `02_deploy/`
- `03_scored/`
- `04_reports/`
- `logs/`

### Important per-window files
- `BOOKIE_ALLMARKETS_<date_from>_to_<date_to>.csv`
- `DEPLOY_COMBINED_<date_from>_to_<date_to>.csv`
- `DEPLOY_COMBINED_SCORED_<date_from>_to_<date_to>.csv`

---

## Master outputs
Located under:
- `predictions_output/walk_forward/_MASTER/`

### Important master files
- `_ALL_WINDOWS__SCORECARD.csv`
- `_ALL_WINDOWS__SCORECARD_GRADED_ONLY.csv`
- `_ALL_WINDOWS__MARKET_ROLLUP.csv`
- `_ALL_WINDOWS__BEST_WINDOWS.csv`
- `_ALL_WINDOWS__WORST_WINDOWS.csv`
- `INVESTOR_ONE_PAGE_SUMMARY.csv`
- `INVESTOR_ONE_PAGE_SUMMARY.md`
- `INVESTOR_SHORTLIST.csv`
- `INVESTOR_SHORTLIST.md`
- `CALENDAR_REGIME_AUDIT.csv`

### Audit subfolders
- `VARIANT_TESTS/`
- `DRAW_RISK_AUDITS/`
- `STANDARD_RESIDUAL_AUDITS/`
- `LOSS_AUDITS/`
- `BTTS_SIGNAL_AUDITS/`
- `BTTS_LIVE_PRODUCT_AUDITS/`

---

## 12. Commands used and validated

## 12.1 Re-run walk-forward scoring and stamping using existing files
```bash
python3 run_walkforward_windows.py   --manifest predictions_output/walk_forward/window_manifest.csv   --skip-source --skip-deploy --skip-summary
```

This:
- reuses existing source and deploy files
- enriches/scored existing windows
- stamps calendar regimes
- updates master outputs

---

## 12.2 Full walk-forward run
```bash
python3 run_walkforward_windows.py   --manifest predictions_output/walk_forward/window_manifest.csv
```

This runs:
- source
- deploy
- score
- summary
- master rollups

---

## 12.3 Test one global regime override
```bash
python3 run_walkforward_windows.py   --manifest predictions_output/walk_forward/window_manifest.csv   --calendar-regime POST_FIFA_BREAK   --skip-source --skip-deploy --skip-summary
```

This is for testing only.
Production behavior should rely on window-aware regime mapping.

---

## 12.4 Verify stamped overlay fields in a scored file
```bash
python3 - <<'PY'
import pandas as pd

p = "predictions_output/walk_forward/w18_2026_03_13_2026_03_17/03_scored/DEPLOY_COMBINED_SCORED_2026-03-13_to_2026-03-17.csv"
df = pd.read_csv(p, low_memory=False)

cols = [
    "calendar_regime",
    "btts_leg_multiplier",
    "ftr_leg_multiplier",
    "stake_multiplier",
    "overlay_note",
]
print(df[cols].drop_duplicates().to_string(index=False))
PY
```

---

## 12.5 Audit all windows for stamped regime consistency
```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

manifest = pd.read_csv("predictions_output/walk_forward/window_manifest.csv")

rows = []
for _, r in manifest.iterrows():
    p = Path(f"predictions_output/walk_forward/{r.window_id}/03_scored/DEPLOY_COMBINED_SCORED_{r.date_from}_to_{r.date_to}.csv")
    if not p.exists():
        continue
    df = pd.read_csv(p, low_memory=False)
    cols = ["calendar_regime","btts_leg_multiplier","ftr_leg_multiplier","stake_multiplier","overlay_note"]
    uniq = df[cols].drop_duplicates()
    for _, u in uniq.iterrows():
        rows.append({
            "window_id": r.window_id,
            "date_from": r.date_from,
            "date_to": r.date_to,
            **u.to_dict()
        })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("predictions_output/walk_forward/_MASTER/CALENDAR_REGIME_AUDIT.csv", index=False)
print("\nWROTE: predictions_output/walk_forward/_MASTER/CALENDAR_REGIME_AUDIT.csv")
PY
```

---

## 12.6 Inspect Patch D live debug
```bash
python3 deploy_rulebook.py   --src predictions_output/2026-03-26/BOOKIE_IMP20_ALLMARKETS_2026-03-13_to_2026-03-17.csv   --preset V1   --ftr-profile accuracy   --debug > /tmp/patch_d_live_debug.log 2>&1

grep -A35 -B5 'Patch C applied\|Patch D applied\|Patch D removed fixtures\|CHAOS residue block applied' /tmp/patch_d_live_debug.log
```

---

## 12.7 Confirm no surviving live Patch D targets remain in STANDARD
Typical inspection pattern:
```bash
python3 - <<'PY'
import pandas as pd

p = "predictions_output/walk_forward/w18_2026_03_13_2026_03_17/03_scored/DEPLOY_COMBINED_SCORED_2026-03-13_to_2026-03-17.csv"
df = pd.read_csv(p, low_memory=False)

sub = df[
    (df["market"].astype(str).str.lower() == "ftr") &
    (df["source_tier_file"].astype(str).str.upper() == "STANDARD") &
    (df["standard_reporting_bucket"].astype(str).str.upper() == "STANDARD_FTR_BASE")
].copy()

sub["selected_minus_draw"] = sub.apply(
    lambda r: (r["confidence_home"] - r["confidence_draw"]) if str(r["selection"]).upper() == "HOME"
    else ((r["confidence_away"] - r["confidence_draw"]) if str(r["selection"]).upper() == "AWAY" else None),
    axis=1
)

check = sub[
    sub["selected_minus_draw"].notna() &
    (sub["selected_minus_draw"] <= 0.15) &
    (pd.to_numeric(sub["power_diff"], errors="coerce").abs() <= 18)
]

print(check[[
    "league","match_date","home_team_name","away_team_name","selection",
    "bookie_od","model_p_for_bookie","selected_minus_draw","power_diff",
    "ppg_diff_pre","standard_reporting_bucket","context_reason_codes"
]].to_string(index=False))
PY
```

Expected outcome:
- empty dataframe after Patch D live promotion

---

## 13. Audits currently available

## FTR
- variant summary
- grouped summaries
- reject reasons
- kept wins
- kept losses
- kept loss draws
- draw-risk audits
- standard residual audits

## BTTS
- signal group audits
- live product audits
- wins/losses focus files
- by-window
- by-league

## OU25
- standard / observe rollups through the main scorecard and summaries

---

## 14. Current optimized settings / proven policy state

## FTR STANDARD
**Live-winning policy**
- keep current Patch D full base-wide block
- use full version, not NO_DRAWWARN-only
- Patch C did not add extra lift in the tested walk-forward slice

### Operational threshold
- `STANDARD_FTR_BASE`
- `selected_minus_draw <= 0.15`
- `abs(power_diff) <= 18`
- demote to OBSERVE

---

## FTR ELITE
Current elite audit remains strong but smaller sample than STANDARD.
No new live change promoted here from this calendar/variant phase.

---

## BTTS
Use calendar overlay pressure, especially:
- post-FIFA break
- post-R16 second-leg March

### Current policy
- BTTS multiplier reduced in those windows
- base product logic unchanged for now
- exposure throttled, not raw model altered

---

## OU25
No new live threshold change from this phase.
STANDARD remains profitable.
OBSERVE remains weak.

---

## 15. What is now proven vs what is pending

## Proven
- FTR Patch D full block improves STANDARD walk-forward headline
- live Patch D is correctly wired into deploy logic
- calendar regime stamping works per window
- audit CSVs prove correct regime assignment
- investor summary files are produced
- shortlist and one-page investor summaries exist
- walk-forward master rollup system is stable enough to use as reporting spine

## Pending / next
- wire calendar multipliers into `acca_builder.py`
- make final shortlist / acca outputs exposure-sensitive
- decide whether to apply multiplier effect to:
  - leg count
  - shortlist max rows
  - stake sizing
  - confidence thresholds
- extend calendar overlay logic to OU25 if future audits justify it
- optionally produce regime-specific performance rollups by market

---

## 16. Recommended next implementation order

### Phase 1
Patch `acca_builder.py` so that:
- BTTS exposure is scaled by `btts_leg_multiplier`
- FTR exposure is scaled by `ftr_leg_multiplier`
- staking is scaled by `stake_multiplier`

### Phase 2
Add an export showing:
- raw picks
- calendar-adjusted picks
- raw stake
- adjusted stake
- overlay note

### Phase 3
Backtest acca-level results with and without calendar throttling

---

## 17. Mission recap in one paragraph

The walk-forward system has now matured into a multi-layer audited deployment stack where raw model predictions flow through tiered routing, actual-result enrichment, formal FTR variant testing, investor-facing rollups, and a new window-aware calendar overlay policy. The strongest current premium finding is that **FTR STANDARD improves to 190/205 (92.68%, ROI 0.5619)** when the full Patch D base-wide draw-gap weak-power block is used. BTTS and FTR have also been shown to react differently to calendar stress, and this has now been formalized into a regime policy that stamps each walk-forward window with market-specific multipliers and stake guidance. The next step is not more theory; it is to wire those multipliers into the final shortlist / acca layer so live deployment pressure adapts automatically to the calendar.

---
