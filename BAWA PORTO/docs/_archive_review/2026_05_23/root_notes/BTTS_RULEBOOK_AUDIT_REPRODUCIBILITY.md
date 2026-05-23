# BTTS Rulebook Audit Reproducibility Guide

This document records how the final BTTS YES live policy was derived, so the full audit can be reproduced later.

## Objective
We wanted to answer three separate questions:

1. Is the new BTTS rulebook better or worse overall?
2. What does it look like when we exclude `OBSERVE` and evaluate only real deployable picks?
3. Which BTTS YES leagues should be:
   - kept live
   - baseline-only
   - blacklisted

---

## Golden rule for all future BTTS audits

### Do not evaluate live performance using OBSERVE
The live deployment set is:

- `ELITE`
- `STANDARD`

`OBSERVE` is for shadowing, tracking, and review only.

If `OBSERVE` is included in headline performance, it will dilute live accuracy and ROI and produce a false read.

---

## Source files used

Main scored files pattern:

```bash
predictions_output/walk_forward/w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv
```

These scored files contain the actual result columns required for truthful BTTS auditing, including:

- `actual_btts_yes`
- `btts_yes_hit`
- `deploy_tier`
- `bookie_pick`
- `bookie_od`
- `model_p_for_bookie`

---

## Important scoring rule

### Correct BTTS win logic
For BTTS markets:

- `YES` wins when `actual_btts_yes == 1`
- `NO` wins when `actual_btts_yes == 0`

That had to be corrected during the audit because one earlier pass incorrectly treated NO rows as zero-win rows.

### Correct graded rule
A BTTS row is graded when:

```python
actual_btts_yes is not null
```

---

## Coverage audit used first

Before evaluating the rulebook, we checked that each walk-forward window actually contained normal fixture breadth.

This was essential because `w139` looked broken at first, but it turned out to be an international-break / low-fixture regime window.

### Outcome
- `w137` normal breadth
- `w138` normal breadth
- `w139` low-fixture / international-break regime
- `w140` missing at time of check

This prevented a false conclusion that the source build or routing had failed.

---

## Final audit sequence

## Step 1 — 3-year master BTTS scored audit
Purpose:
- get broad all-tier BTTS picture
- inspect league profitability
- inspect tier mix
- inspect pick mix

This audit includes all tiers and is useful diagnostically, but **not** as the final deploy decision view.

### Calculation formulas
```python
graded = actual_btts_yes.notna()
```

```python
wins_yes = (bookie_pick == "YES") & (actual_btts_yes == 1)
wins_no  = (bookie_pick == "NO")  & (actual_btts_yes == 0)
wins = wins_yes | wins_no
losses = graded & ~wins
```

```python
profit_yes = bookie_od - 1.0 if YES wins else -1.0
profit_no  = bookie_od - 1.0 if NO wins  else -1.0
```

```python
hit_rate = wins / graded
roi = profit / graded
```

---

## Step 2 — BTTS gate suppressor audit
Purpose:
- find which gates are doing most of the suppression
- make sure the new BTTS YES logic is being driven by intended filters

The all-window suppressor ranking came out:

- `btts_yes_label_fail`
- `btts_yes_ge2_fail`
- `btts_yes_csmax_fail`
- `btts_yes_brazil_block`
- `btts_yes_fts_fail`
- `btts_yes_model_floor_fail`

This showed the live BTTS YES structure is mainly driven by:

- label quality first
- structural confirmation second
- explicit league block controls

---

## Step 3 — ELITE + STANDARD only audit
Purpose:
- remove OBSERVE distortion
- assess the actual production-ready BTTS system

### Filter used
```python
deploy_tier in ["ELITE", "STANDARD"]
```

### Result
This became the real headline audit:

- 2,357 graded
- 2,015 wins
- 342 losses
- 85.49% hit rate
- 43.42% ROI
- +1023.33 units

BTTS YES alone:

- 2,116 graded
- 1,847 wins
- 269 losses
- 87.29% hit rate
- 45.11% ROI

---

## Step 4 — BTTS YES by league, ELITE + STANDARD only
Purpose:
- isolate the actual BTTS YES weapon
- decide final league policy

### Exact command used
```bash
cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && python3 - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("predictions_output/walk_forward")
files = sorted(root.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))

frames = []

for fp in files:
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception as e:
        print("SKIP", fp, e)
        continue

    if df.empty:
        continue

    df["market"] = df.get("market", "").astype(str).str.lower().str.strip()
    df["bookie_pick"] = df.get("bookie_pick", df.get("selection", "")).astype(str).str.upper().str.strip()
    df["deploy_tier"] = df.get("deploy_tier", df.get("tier", "")).astype(str).str.upper().str.strip()
    df["league"] = df.get("league", "").astype(str).str.strip()

    sub = df[
        (df["market"] == "btts") &
        (df["bookie_pick"] == "YES") &
        (df["deploy_tier"].isin(["ELITE", "STANDARD"]))
    ].copy()

    if sub.empty:
        continue

    sub["bookie_od"] = pd.to_numeric(sub.get("bookie_od", pd.NA), errors="coerce")
    sub["model_p_for_bookie"] = pd.to_numeric(sub.get("model_p_for_bookie", pd.NA), errors="coerce")
    sub["actual_btts_yes"] = pd.to_numeric(sub.get("actual_btts_yes", pd.NA), errors="coerce")

    sub["graded"] = sub["actual_btts_yes"].notna().astype(int)
    sub["wins"] = ((sub["graded"] == 1) & (sub["actual_btts_yes"] == 1)).astype(int)
    sub["losses"] = ((sub["graded"] == 1) & (sub["actual_btts_yes"] == 0)).astype(int)

    sub["profit"] = 0.0
    sub.loc[sub["wins"] == 1, "profit"] = sub.loc[sub["wins"] == 1, "bookie_od"].fillna(0) - 1.0
    sub.loc[sub["losses"] == 1, "profit"] = -1.0

    frames.append(sub)

all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

by_league = (
    all_df.groupby("league", dropna=False)
    .agg(
        rows=("market", "size"),
        graded=("graded", "sum"),
        wins=("wins", "sum"),
        losses=("losses", "sum"),
        profit=("profit", "sum"),
        avg_bookie_od=("bookie_od", "mean"),
        avg_model_p=("model_p_for_bookie", "mean"),
    )
    .reset_index()
)

by_league["hit_rate"] = by_league["wins"] / by_league["graded"].replace(0, pd.NA)
by_league["roi"] = by_league["profit"] / by_league["graded"].replace(0, pd.NA)
by_league = by_league.sort_values(["roi", "graded"], ascending=[False, False])

print("\nBTTS YES ELITE+STANDARD ONLY BY LEAGUE")
print(by_league.to_string(index=False))

print("\nMIN 50 GRADED")
print(by_league[by_league["graded"] >= 50].to_string(index=False))

outdir = root / "_MASTER" / "BTTS_YES_ELITE_STANDARD_ONLY_BY_LEAGUE"
outdir.mkdir(parents=True, exist_ok=True)
by_league.to_csv(outdir / "BTTS_YES_ELITE_STANDARD_ONLY_BY_LEAGUE.csv", index=False)

print(f"\nWROTE: {outdir}")
PY
```

### Why this was the final deciding audit
Because it answered the exact production question:

> Which BTTS YES leagues are genuinely strong when we only include deployable rows?

---

## Decision framework used for final league policy

### Keep live
Generally required:
- clear positive ROI
- strong hit rate
- meaningful enough sample
- no evidence of being structurally toxic

### Provisional keep
Used when:
- sample is still small
- but returns are clean and positive

### Baseline-only
Used when:
- league can stay live under base logic
- but should not receive FTS rescue widening

### Hard blacklist
Used when:
- league is structurally poor
- or long-run profile is negative / unreliable
- or prior audit evidence was bad enough that we do not want it promoted live

---

## Final policy outputs created from the audit

### Hard blacklist
- Brazil Serie A
- Austria Bundesliga
- Australia A-League
- Czech First League
- Germany Bundesliga 2
- South Korea K League
- Denmark Superliga
- Swiss Super League

### Baseline-only watchlist
- USA MLS
- Turkey Super Lig
- Saudi Pro League
- England Championship
- Portugal Liga
- France Ligue 1
- Germany Bundesliga
- England Premier League

### FTS override allowed leagues
- Netherlands Eredivisie
- Europa Conference
- Japan J1
- Belgium Pro
- England FA Cup
- Europa League
- Champions League

---

## Exact code posture that was locked

The FTS override in `btts_yes_is_live()` and `btts_yes_reason()` must include:

```python
and league not in BTTS_YES_BASELINE_ONLY_WATCHLIST
```

This prevents rescue widening on watchlist leagues, while still allowing them to pass baseline logic if they qualify normally.

---

## Known audit pitfalls we hit and corrected

### 1. Counting OBSERVE as live
This made performance look much worse than the real deployable set.

### 2. Incorrect NO-row scoring
At one stage, BTTS NO rows were effectively being counted as zero-win rows. That had to be corrected.

### 3. Evaluating low-fixture windows as if they were normal
`w139` looked broken until source coverage showed it was a real low-fixture / international-break regime window.

### 4. Mixing pre-patch and post-patch logic
Some historical outputs appeared to include rows from leagues that should now be blocked. That is why future audits must be run on fully regenerated post-policy outputs only.

---

## Future audit checklist

For the next BTTS audit cycle:

1. Rebuild / regenerate the relevant walk-forward outputs.
2. Confirm source window coverage before judging anything.
3. Audit all BTTS first for diagnostics.
4. Audit `ELITE + STANDARD` only for live performance.
5. Audit `BTTS YES` by league within `ELITE + STANDARD`.
6. Do not change league policy from all-tier outputs.
7. Only promote a league if deployable BTTS YES evidence supports it.

---

## Output folders used in this cycle

Examples:

```bash
predictions_output/walk_forward/_MASTER/BTTS_3Y_RULEBOOK_AUDIT
predictions_output/walk_forward/_MASTER/BTTS_3Y_RULEBOOK_AUDIT_POSTPATCH_FIXED
predictions_output/walk_forward/_MASTER/BTTS_3Y_ELITE_STANDARD_ONLY_AUDIT
predictions_output/walk_forward/_MASTER/BTTS_YES_ELITE_STANDARD_ONLY_BY_LEAGUE
predictions_output/walk_forward/_MASTER/WINDOW_COVERAGE_AUDIT
```

---

## Recommended next step after this
Now that BTTS YES is locked, the next audit focus should be:

- OU25 coverage refinement
- OU25 live-vs-observe separation
- OU25 per-league deploy policy
- then combined deploy architecture alignment across:
  - FTR
  - BTTS
  - OU25

This closes the BTTS YES rulebook audit cycle in a reproducible way.
