# Whole Product Estate Website Export Unlock

Date: 2026-05-25

## What Changed

The website export is no longer limited to the narrow OU25 Over / BTTS Yes evidence board when measuring product coverage.

`scripts/export_site_sqlite.py` now exports a unified table:

- `site_product_estate_signals`

It also exports policy-level product references:

- `site_product_policy_references`

This keeps the customer-facing state simple:

- `deploy`
- `watch`
- `avoid`

while preserving deeper evidence labels underneath for premium/audit views.

## Product Lanes Now Represented

Row-level product signals now include:

- Over 2.5
- BTTS Yes
- FTR
- Correct Score
- Team Goals 1.5
- BTTS No when a current board contains `BTTS / NO` rows

FTR is now represented in two separate layers:

- `ftr_production_tier`: old proven ELITE/STANDARD production rows from the deploy board
- `ftr_research_policy`: new FTR policy/attribution rows kept below the product layer as watch/avoid evidence

The goal evidence board can still contribute FTR context, but it is no longer allowed to act as FTR deploy authority. FTR deploy comes from the FTR production tier lane unless a future clean research pocket is explicitly promoted after drift checks.

BTTS No also has policy-reference rows even when no current fixture-level `BTTS / NO` candidates exist:

- `BTTS_NO_META_ELITE_LOCKED_BASELINE`
- `BTTS_NO_HYBRID_META_RESCUE_LOCKED_UEFA`
- `BTTS_NO_MLS_META_RESCUE_LOCKED`

## Verification Export

Temp export:

`/private/tmp/og_site_product_estate_ftr.sqlite`

Counts:

| Table | Rows |
| --- | ---: |
| `site_product_estate_signals` | 42,937 |
| `site_product_policy_references` | 14 |
| `site_goal_evidence_board` | 20,353 |
| `site_correct_score_context` | 14,877 |
| `site_tg15_context` | 20,340 |

Unified row-level customer states:

| State | Rows |
| --- | ---: |
| watch | 34,465 |
| avoid | 6,432 |
| deploy | 2,040 |

By product:

| Product | Total | Deploy | Watch | Avoid | Deploy % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct Score | 14,859 | 373 | 14,486 | 0 | 2.51% |
| Over 2.5 | 14,159 | 1,148 | 9,661 | 3,350 | 8.11% |
| BTTS Yes | 6,194 | 514 | 4,320 | 1,360 | 8.30% |
| Team Goals 1.5 | 4,088 | 0 | 4,088 | 0 | 0.00% |
| FTR | 3,637 | 5 | 1,910 | 1,722 | 0.14% |

FTR by segment:

| Segment | Deploy | Watch | Avoid |
| --- | ---: | ---: | ---: |
| `ftr_production_tier` | 5 | 0 | 0 |
| `ftr_live_context` | 0 | 5 | 0 |
| `ftr_research_policy` | 0 | 1,905 | 1,722 |

Current-window FTR production deploy rows:

| Tier | Rows | Reference hit rate |
| --- | ---: | ---: |
| ELITE | 3 | 92.64% |
| STANDARD | 2 | 83.36% |

BTTS No policy references:

| Policy | State | Rows | Hit Rate | ROI |
| --- | --- | ---: | ---: | ---: |
| `BTTS_NO_META_ELITE_LOCKED_BASELINE` | deploy | 2,712 | 90.15% | 73.35% |
| `BTTS_NO_HYBRID_META_RESCUE_LOCKED_UEFA` | deploy | 2 validation rows | n/a | n/a |
| `BTTS_NO_MLS_META_RESCUE_LOCKED` | deploy | 3 validation rows | n/a | n/a |

FTR policy references:

| Policy | State | Rows | Hit Rate | ROI / pick | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `FTR_ELITE` | deploy | n/a | 92.64% | n/a | locked historical production |
| `FTR_STANDARD` | deploy | n/a | 83.36% | n/a | locked historical production |
| `FTR_FAVORITE_FRAGILITY_WATCH` | avoid | 994 | 60.76% | -11.65% | caution confirmed not promotable |
| `FTR_DRAW_RESULT_WATCH_CANDIDATE` | avoid | 13 | 30.77% | -38.46% | caution confirmed not promotable |
| `FTR_DRAW_DANGER_BLOCK` | avoid | 48 | 27.08% | -43.60% | caution confirmed not promotable |
| `FTR_SEASONAL_CONTEXT_SUPPORT_CANDIDATE` | watch | 848 | 70.28% | +21.18% | promotion review ready with drift caution |
| `FTR_TARGET_LEAGUE_UNLOCK_CANDIDATE` | watch | 115 | 78.26% | +36.77% | strong fresh signal pending settlement |
| `FTR_FIXTURE_CONTEXT_SUPPORT_CANDIDATE` | watch | 501 | 72.26% | +19.03% | strong fresh signal pending settlement |
| `FTR_FULL_STACK_CS_SUPPORT_CANDIDATE` | watch | 977 | 72.06% | +23.94% | strong fresh signal pending settlement |
| `FTR_FULL_STACK_RESULT_CANDIDATE` | watch | 977 | 72.06% | +23.94% | strong fresh signal pending settlement |
| `FTR_TEAM_INTEL_SUPPORT_CANDIDATE` | watch | 120 | 68.33% | +18.29% | strong fresh signal pending settlement |

## Interpretation

The earlier apparent coverage collapse happened because the site was reading only the direction-locked OU25 Over / BTTS Yes board.

The new export makes the full estate visible:

- goal markets remain row-level deploy/watch/avoid;
- Correct Score is exposed as public top-3 plus premium/pro rule rows;
- TG15 is exposed as watch/context only until promotion;
- FTR has its old proven ELITE/STANDARD deploy lane restored as a production segment;
- FTR research remains separate, with draw danger, favorite fragility, and unresolved lean-mass shapes demoted to avoid;
- BTTS No is represented as a locked policy lane, with fixture rows to populate when current boards produce `BTTS / NO` candidates.

## Guardrails

- No production rulebook changes were made.
- TG15 remains support/context only, not deploy authority.
- FTR remains separate and is not mixed into OU25/BTTS logic.
- FTR research labels do not become customer deploy rows unless they pass future drift/promotion gates.
- `HOME_LEAN_MASS`, `AWAY_LEAN_MASS`, draw danger, and favorite fragility are hard-demoted unless a documented rescue policy is present.
- BTTS No policy references are not fabricated fixture rows.
- Correct Score deploy rows mean premium correct-score product rows, not guaranteed exact-score outcomes.

## Next

The webapp can now read `site_product_estate_signals` for a whole-product coverage view and `site_product_policy_references` for product proof cards.

The next UI step is to replace single-board coverage copy with product-lane tabs:

- Core Goals
- FTR
- Correct Score
- Team Goals
- BTTS No
