# Recovery Recap — 2026-04-14

This note captures the recovery work completed before the overnight rerun launched from:

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO`

The goal is to preserve:

- what broke
- what we checked
- what we found
- what we patched
- what thresholds / allowlists / market live lists are currently in force
- what is still pending after the overnight run

## 1. Problem Statement

After the latest model refresh / retrain work, walkforward quality had drifted away from the stronger prior BTTS / OU25 headline numbers.

The main investigation threads were:

- BTTS YES live collapse / lower hit rate vs earlier headline numbers
- OU25 OVER structural suppressors
- overlap / rescue logic between BTTS YES live and OU25 OVER live
- model regime mismatch across leagues

## 2. Key Files Investigated

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/bookie_allmarkets.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/deploy_rulebook.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/run_walkforward_windows.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/MODELSTORE_GOAL_REGIME_SPLIT.md`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/walk_forward/_MASTER/BTTS_SIGNAL_AUDITS/*`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/walk_forward/_MASTER/BTTS_LIVE_PRODUCT_AUDITS/*`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/walk_forward/_MASTER/OU25_3Y_RULEBOOK_AUDIT/*`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/walk_forward/_MASTER/OU25_GATE_AUDIT__SUMMARY.csv`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/walk_forward/_MASTER/OU25_GATE_AUDIT_BY_LEAGUE__SUMMARY.csv`

## 3. Core Findings

### 3.1 BTTS YES label gate was the main suppressor

In `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/deploy_rulebook.py`, BTTS YES live rows are gated to:

- `STRONG_YES`
- `VERY_STRONG_YES` non-Brazil

The main audit suppressor was:

- `btts_yes_label_fail`

This was traced to the live label gate, not to missing rows or a broken downstream filter.

### 3.2 BTTS signal collapse was real in many leagues

From the all-league BTTS signal / CS audit:

- `England Championship`: 100% `NEUTRAL`
- `England Premier League`: 100% `NEUTRAL`
- `Germany Bundesliga`: `NEUTRAL` + `WEAK_YES`, no `STRONG_YES`
- several defensive / lower-event leagues were structurally below the BTTS live gate

Healthy BTTS YES label-pass leagues were:

- `Japan J1`
- `Netherlands Eredivisie`
- `Europa Conference`
- `Norway Eliteserien`
- `Belgium Pro`
- `Brazil Serie A`
- `England FA Cup`
- `Germany Bundesliga 2`
- `Europa League`

### 3.3 BTTS label collapse was not solely caused by retrain

Important correction:

- `England Premier League` had the same collapse pattern despite not being part of the same retrained cohort story

Conclusion:

- this was not just “retrain broke BTTS”
- the BTTS live gate was always selective and structurally excluded lower-BTTS leagues
- prior stronger headline numbers were likely narrower / hotter cohorts than the full long-run walkforward

### 3.4 WEAK_YES was tested and should not be globally opened

All-window WEAK_YES shadow audit:

- rows: `2443`
- hit rate: `59.8%`
- ROI: `+1.5%`

Compared with live scored groups:

- `STRONG_YES`: `73.5%` hit, `+18.0%` ROI
- `VERY_STRONG_YES` non-Brazil: `78.0%` hit, `+30.7%` ROI

Conclusion:

- do **not** lower global BTTS thresholds
- `WEAK_YES` is too noisy globally

### 3.5 OU25 OVER -> BTTS YES rescue already existed

There was already a forward rescue lane in `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/deploy_rulebook.py`:

- `OU25 OVER` live overlap rescuing `BTTS YES` rows into `STANDARD`

This used:

- `OU25_OVER25_BTTS_RESCUE_CORE_ALLOWLIST`
- mirrored into:
  - `BTTS_YES_OU25_RESCUE_CORE_ALLOWLIST`

### 3.6 BTTS YES live -> OU25 OVER reverse rescue did not exist

This reverse lane did **not** exist before the patch.

We audited the reverse direction and found strong uplift:

- baseline OU25 OVER: `56.5%`, `-7.6% ROI`
- overlap with BTTS YES live: `67.9%`, `+10.1% ROI`
- hit-rate lift: `+11.4pp`

Top reverse-overlap leagues:

- `Belgium Pro`
- `Europa Conference`
- `Europa League`
- `Netherlands Eredivisie`
- `Germany Bundesliga 2`
- `Japan J1`
- `England FA Cup`
- `Norway Eliteserien`

### 3.7 Portugal Liga was blocked from live OU25 OVER

Before patch, Portugal Liga was in:

- `OU25_OVER25_BASELINE_ONLY_LEAGUES`

This prevented live deployment.

Portugal Liga also showed:

- positive baseline ROI in the OU25 OVER struct-fail analysis
- enough evidence to justify unlocking

## 4. Model Regime / Feature Regime Findings

From `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/MODELSTORE_GOAL_REGIME_SPLIT.md`:

- `25` leagues on `safe` schema (`20–24` features), created `2026-04-12`
- `4` leagues on `legacy` schema (`76–92` features), created `2026-02-12`

This mixed-regime estate was identified as a likely contributor to instability / inconsistency.

### 4.1 Safe schema leagues

- `Australia_A-League`
- `Australia_A_League`
- `Austria_Bundesliga`
- `Belgium_Pro`
- `Brazil_Serie_A`
- `Champions_League`
- `Czech_First_League`
- `Denmark_Superliga`
- `England_Championship`
- `England_EFL_League_1`
- `England_FA_Cup`
- `Europa_Conference`
- `Europa_League`
- `France_Ligue_1`
- `Germany_Bundesliga`
- `Germany_Bundesliga_2`
- `Italy_Serie_A`
- `Japan_J1`
- `Netherlands_Eredivisie`
- `Norway_Eliteserien`
- `Portugal_Liga`
- `Saudi_Pro_League`
- `South_Korea_K_League`
- `Swiss_Super_League`
- `Turkey_Super_Lig`

### 4.2 Legacy schema leagues

- `USA_MLS`
- `Spain_La_Liga`
- `England_Premier_League`
- `Scotland_Premiership`

### 4.3 Pending model / feature work

Still pending after this recovery pass:

- retrain remaining reduced-feature leagues into a unified regime
- likely target:
  - `GOAL_REGRESSOR_SCHEMA=legacy`
- snap / team enrichment for leagues still lacking the richer full feature path

## 5. BTTS Thresholds Confirmed

In `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/bookie_allmarkets.py`, BTTS side labels are assigned by probability:

- `VERY_STRONG_YES` if `p >= 0.70`
- `STRONG_YES` if `p >= 0.60`
- `WEAK_YES` if `p >= 0.55`
- `NEUTRAL` otherwise

We explicitly decided:

- keep these thresholds unchanged for now

Reason:

- WEAK_YES is not strong enough globally

## 6. OU25 Thresholds Confirmed

Current OU25 OVER tier policy in `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/deploy_rulebook.py`:

### 6.1 Tier 1 leagues

- `Netherlands Eredivisie`
- `Belgium Pro`
- `Champions League`
- `Europa League`
- `England FA Cup`
- `Portugal Liga`

### 6.2 Tier 2 leagues

- `Norway Eliteserien`
- `Spain La Liga`
- `Japan J1`
- `USA MLS`
- `Europa Conference`

### 6.3 Baseline-only leagues

- `England Championship`
- `Brazil Serie A`

### 6.4 Blacklist leagues

- `Austria Bundesliga`
- `Australia A-League`
- `Czech First League`
- `Denmark Superliga`
- `South Korea K League`
- `Swiss Super League`

### 6.5 Tier 1 rules

- `Netherlands Eredivisie`: `exp_goals_sum >= 3.40`
- `Belgium Pro`: `exp_goals_sum >= 3.40`
- `Champions League`: `exp_goals_sum >= 3.40`
- `Europa League`: `exp_goals_sum >= 2.80`
- `England FA Cup`: `bookie_lambda_total_fit >= 3.00`
- `Portugal Liga`: `exp_goals_sum >= 2.60`

### 6.6 Shared OU25 fallback thresholds

- `OU25_OVER25_TIER2_EG_MIN = 3.00`
- `OU25_OVER25_FALLBACK_P00_MAX = 0.06`
- `OU25_OVER25_FALLBACK_LAMBDA_MIN = 2.80`

### 6.7 OU25 structural gate components

Current OVER structural checks:

- `avg_btts_rate >= 0.50`
- `cs_over25_mass >= 0.15`
- `top3_over_count >= 2`

Optional one-sided veto:

- `one_sided_no_btts >= 1` and `avg_btts_rate <= 0.50`

## 7. BTTS / OU25 Allowlist Changes Applied

### 7.1 BTTS YES rescue via OU25 OVER live overlap

Expanded `OU25_OVER25_BTTS_RESCUE_CORE_ALLOWLIST` to include:

- `Belgium Pro`
- `Champions League`
- `England Premier League`
- `Europa Conference`
- `Europa League`
- `Norway Eliteserien`
- `South Korea K League`
- `Spain La Liga`
- `USA MLS`

This powers:

- `BTTS YES` rescue via live `OU25 OVER` overlap

### 7.2 Portugal Liga unlock

Applied:

- removed from `OU25_OVER25_BASELINE_ONLY_LEAGUES`
- added to `OU25_OVER25_TIER1_LEAGUES`
- added Tier 1 rule:
  - `exp_goals_sum >= 2.60`

### 7.3 New reverse rescue lane added

Added new allowlist:

- `OU25_OVER25_BTTS_YES_RESCUE_ALLOWLIST`

Leagues:

- `Norway Eliteserien`
- `Belgium Pro`
- `Europa Conference`
- `Europa League`
- `Netherlands Eredivisie`
- `Germany Bundesliga 2`
- `Japan J1`
- `England FA Cup`

Added reverse rescue behavior in `_enforce_ou25_live_policy(...)`:

- if BTTS YES is live on the same fixture
- and OU25 OVER row is in `OBSERVE`
- and league is allowlisted
- promote OU25 OVER row to `STANDARD`

Implementation details:

- uses `elite + standard` as BTTS live source
- rescue is `STANDARD` only
- de-dupes by:
  - `league`
  - `fixture_key`
  - `market`
  - `bookie_pick`
- stamps:
  - `OU25_OVER25_BTTS_YES_RESCUE`
  - `OU25_OVER25_BTTS_YES_RESCUE_DENYLIST`

## 8. Important Audit Lessons / Mistakes Avoided

### 8.1 Wrong artifact layer can mislead

We initially used scored candidate files for some shadow checks and hit false conclusions because:

- certain derived live feature columns were missing at that layer

We corrected this by checking the correct deploy / combined artifact layers.

### 8.2 BTTS WEAK_YES should not be promoted globally

This was tested explicitly and rejected.

### 8.3 OU25 struct gates are not “missing”; the earlier NaN result was a file-layer issue

Deploy outputs showed 98–100% non-null coverage for OU25 live structural columns.

## 9. Current Interpretation Before Overnight Run

### 9.1 BTTS

BTTS is not fundamentally broken.

The system is selective:

- `VERY_STRONG_YES` remains the highest-quality premium cohort
- `STRONG_YES` remains viable
- `WEAK_YES` remains blocked

The mismatch vs older headline numbers is likely a combination of:

- different cohort definitions
- longer honest walkforward horizon
- mixed model regime estate

### 9.2 OU25

OU25 appears operational, with:

- meaningful structural gating
- fallback support
- now an added reverse BTTS YES overlap rescue lane
- Portugal Liga unlocked from baseline-only restriction

## 10. Overnight Rerun Command

Separate output root was chosen to preserve baseline outputs:

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/walk_forward_ou25_btts_rescue_patch_2026_04_14`

Command launched:

```bash
cd "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO" && \
python3 run_walkforward_windows.py \
  --manifest walkforward_manifest_3y_thu_wed.csv \
  --base-outdir predictions_output/walk_forward_ou25_btts_rescue_patch_2026_04_14 \
  --predictions-dir predictions_output \
  --merged-dir "Matches/__merged__" \
  --calendar-flags-dir predictions_output/calendar_flags \
  --source-implied-min 20 \
  --bookie-extra-args '--leagues "Australia A-League,Austria Bundesliga,Belgium Pro,Brazil Serie A,Champions League,Czech First League,Denmark Superliga,England Championship,England EFL League 1,England FA Cup,England Premier League,England U21,Europa Conference,Europa League,France Ligue 1,Germany Bundesliga,Germany Bundesliga 2,Italy Serie A,Japan J1,Netherlands Eredivisie,Norway Eliteserien,Portugal Liga,Saudi Pro League,Scotland Premiership,South Korea K League,Spain La Liga,Sweden Allsvenskan,Swiss Super League,Turkey Super Lig,USA MLS" --markets ftr,btts,ou25 --ou25-implied-min 0.20 --btts-implied-min 0.20 --strict' \
  --deploy-extra-args '--preset V1 --ftr-profile accuracy --ftr-priority-ordering' \
  --score-candidates \
  --write-window-file-manifest \
  2>&1 | tee /tmp/run_walkforward_ou25_btts_rescue_patch_2026_04_14.log
```

## 11. Validation Completed Before Launch

Syntax / compile checks passed for:

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/bookie_allmarkets.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/deploy_rulebook.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/run_walkforward_windows.py`

## 12. Pending After Overnight Run

When the overnight run finishes, compare old vs new:

### 12.1 BTTS

- ELITE / STANDARD / OBSERVE
- BTTS YES live core / aggressive / combined
- whether BTTS rescue allowlist expansion materially improved `STANDARD` / combined live performance

### 12.2 OU25

- ELITE / STANDARD / OBSERVE
- whether Portugal Liga appears in live OU25 output
- whether reverse BTTS YES -> OU25 OVER rescue increases `STANDARD` volume and hit rate

### 12.3 Model regime cleanup still pending

- unify schema estate
- retrain reduced-feature leagues onto a single intended schema
- finish snap enrichment / team data coverage for leagues still missing richer feature paths

## 13. Recommended Next Notes After Run Completes

Add:

- final global tier summaries
- final BTTS live product summary
- final OU25 gate summary
- exact before / after diffs for:
  - BTTS rescue patch
  - Portugal Liga unlock
  - reverse OU25 rescue lane

