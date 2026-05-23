# Final BTTS YES Live Rulebook League Policy

## Status
Locked from the current 3-year BTTS audit review.

## What counts as deployable
Only these tiers count as live deployment:

- `ELITE`
- `STANDARD`

`OBSERVE` is excluded from deployment evaluation and must not be included when judging live BTTS performance.

## Core result behind this policy
From the deployable set only (`ELITE + STANDARD`):

- 2,359 BTTS picks
- 2,357 graded
- 2,015 wins
- 342 losses
- 85.49% hit rate
- 43.42% ROI
- +1023.33 units

BTTS YES is the main engine:

- 2,117 rows
- 2,116 graded
- 1,847 wins
- 269 losses
- 87.29% hit rate
- 45.11% ROI

---

## Final BTTS YES league policy

### 1. KEEP LIVE — core approved leagues
These have strong deployable BTTS YES performance and are approved as live BTTS YES leagues.

- Japan J1
- Europa Conference
- Norway Eliteserien
- Belgium Pro
- Spain La Liga
- England FA Cup
- Netherlands Eredivisie
- USA MLS

### 2. KEEP LIVE — provisional / smaller-sample approvals
These are positive and deployable, but sample size is smaller, so they should be tagged as provisional until they build more volume.

- Europa League
- Champions League
- Italy Serie A
- France Ligue 1

### 3. BASELINE-ONLY / MONITOR
These should not get rescue widening beyond the baseline live logic until re-audited.

- England Premier League
- Germany Bundesliga
- France Ligue 1
- Portugal Liga
- England Championship
- Turkey Super Lig
- Saudi Pro League
- USA MLS

Note:
- `USA MLS` stays live because deployable results are strong, but it remains **baseline-only**.
- Baseline-only means: allow normal baseline pass logic, but do **not** allow FTS rescue widening.

### 4. BLOCK / HARD BLACKLIST
These should remain blocked for BTTS YES live deployment.

- Austria Bundesliga
- Australia A-League
- Czech First League
- Denmark Superliga
- South Korea K League
- Swiss Super League
- Brazil Serie A
- Germany Bundesliga 2

---

## Exact rulebook posture

### Hard blacklist
```python
BTTS_YES_HARD_BLACKLIST = {
    "Brazil Serie A",
    "Austria Bundesliga",
    "Australia A-League",
    "Czech First League",
    "Germany Bundesliga 2",
    "South Korea K League",
    "Denmark Superliga",
    "Swiss Super League",
}
```

### Baseline-only watchlist
```python
BTTS_YES_BASELINE_ONLY_WATCHLIST = {
    "USA MLS",
    "Turkey Super Lig",
    "Saudi Pro League",
    "England Championship",
    "Portugal Liga",
    "France Ligue 1",
    "Germany Bundesliga",
    "England Premier League",
}
```

### FTS override allowed leagues
```python
FTS_OVERRIDE_ALLOWED_LEAGUES = {
    "Netherlands Eredivisie",
    "Europa Conference",
    "Japan J1",
    "Belgium Pro",
    "England FA Cup",
    "Europa League",
    "Champions League",
}
```

### FTS override rule
FTS rescue is allowed only when all of the following are true:

- league is in `FTS_OVERRIDE_ALLOWED_LEAGUES`
- league is **not** in `BTTS_YES_BASELINE_ONLY_WATCHLIST`
- row is **not** in `BTTS_YES_HARD_BLACKLIST`
- baseline `final_live_pass == 0`
- failure is specifically the FTS veto path
- supporting flags remain intact:
  - `label_pass == 1`
  - `brazil_block == 0`
  - `ge2_pass == 1`
  - `model_floor_pass == 1`
  - `cs_pass == 1`
  - `double_blank_pass == 1`
  - `confirmation_pass == 1`

---

## Production interpretation

### What is live right now
The BTTS YES live model should be treated as:

- a **high-confidence deploy lane**
- driven mainly by:
  - Japan J1
  - Europa Conference
  - Belgium Pro
  - Norway Eliteserien
  - Netherlands Eredivisie
  - England FA Cup
  - Spain La Liga

### What is not allowed to distort evaluation
Do not include `OBSERVE` rows when judging:

- hit rate
- ROI
- profitability
- league keep / block decisions

### What still needs checking later
There is one implementation sanity item to keep on the list:

Some historical audit outputs still showed rows from leagues that are now in the hard blacklist. That strongly suggests some earlier windows or legacy files were generated before the final blacklist logic was fully locked. Future audits should always be run on the fully regenerated post-policy outputs only.

---

## Recommended next default for BTTS YES
Use this as the operating stance:

- **Live:** core keep-live + provisional keep-live leagues
- **Baseline-only:** watchlist leagues
- **Blocked:** hard blacklist leagues

That is the final BTTS YES rulebook league posture from this audit cycle.
