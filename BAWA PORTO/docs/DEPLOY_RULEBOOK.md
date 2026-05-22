# DEPLOY_RULEBOOK

## Purpose
Code-derived documentation for `deploy_rulebook.py`.

`deploy_rulebook.py` is the authoritative live-routing layer for Odds Genius. It reads an all-markets board from `bookie_allmarkets.py`, applies deterministic runtime vetoes plus market-specific policy gates, and writes deploy-tier outputs.

## Inputs
Primary input:
- `BOOKIE_IMP*_ALLMARKETS_<date_from>_to_<date_to>.csv`

Core columns consumed repeatedly:
- identity: `league`, `fixture_key`, `match_date`, `market`, `selection`, `bookie_pick`
- pricing / probability: `bookie_od`, `bookie_implied`, `bookie_implied_used`, `bookie_implied_novig`, `model_p_for_bookie`, `gap`, `gap_novig`
- FTR structure: `model_top_pick`, `ftr_margin`, `power_diff`, `pick_side_mass_top3`, `pick_side_margin_top3`
- BTTS structure: `signal_btts`, `p_home_fts`, `p_away_fts`, `p00_est`, `exp_goals_sum`
- OU25 structure: `signal_over25`, `ou25_policy_state`, `ou25_over_struct_pass`, `bookie_lambda_total_fit`, `exp_goals_sum`, `top3_over_count`
- overlay / context: `context_reason_codes`, `reason_codes`, `team_intel_*`, `meta_*`, `cs*`

Optional helper input:
- `deploy_gates.py` via `compute_deterministic_deploy_vetoes`
- learned draw-threshold JSONs and phase policy CSVs under `reports/`

## Outputs
Written next to the source file:
- preset summary CSV/markdown
- tiered CSVs: `ELITE`, `STANDARD`, `OBSERVE`
- optional raw/after-gates audit files
- optional BTTS / OU25 gate audit CSVs

Core output columns stamped or normalized:
- `deploy_tier`, `tier`
- `context_reason_codes`
- `reason_codes` when present
- `standard_reporting_bucket`
- market runtime fields such as `ou25_policy_state`
- product identity fields and profit-first markers where applicable

## Key Functions
Top-level routing:
- `apply_rulebook_close(...)`
- internal V1 path via `apply_rulebook_v1(...)` flow in `main()`
- `_build_standard_params(...)`
- `_coalesce_tier_cols(...)`
- `_stamp_product_identity(...)`

Reason stamping / identity:
- `_append_reason_token(...)`
- `_append_reason_code(...)`
- `_standard_reporting_bucket(...)`

OU25 live gating:
- `_stamp_ou25_runtime_fields(...)`
- `_enforce_ou25_live_policy(...)`
- `_ou25_struct_combo_mask(...)`
- `_ou25_top3_mask(...)`
- `_eligible_ou25_premium_rows(...)`
- `_eligible_ou25_under_standard_rows(...)`

BTTS live gating:
- `_stamp_btts_runtime_fields(...)`
- `_stamp_btts_yes_live_features(...)`
- `_stamp_btts_yes_double_blank_live_features(...)`
- `btts_yes_is_live(...)`
- `btts_yes_reason(...)`

FTR routing / rescue:
- `_derive_ftr_cs_overlay_fields(...)`
- `_stamp_cs_concentration_overlay_fields(...)`
- `_eligible_ftr_observe_rescue_rows(...)`
- `_eligible_ftr_cs_standard_promotion_rows(...)`
- `_apply_ftr_standard_shortprice_low_model_block(...)`
- `_apply_ftr_standard_base_draw_gap_le_0p15_and_weak_power_block(...)`
- `_apply_ftr_standard_chaos_residue_block(...)`

Gate audits / diagnostics:
- `_write_btts_gate_audit(...)`
- `_write_ou25_gate_audit(...)`
- `_run_gate_sanity_tests(...)`

## Gate / Veto Architecture
### Runtime veto layer
Before market routing, deterministic vetoes from `deploy_gates.py` can mark rows unsafe, e.g.
- `FTR_NO_SIDE_TRUE_CLOSE`
- `BTTS_YES_FTS_TOO_HIGH`
- `BTTS_YES_CS1_TRAP`
- `BTTS_NO_HIGH_GOALS_LOW_P00`

### FTR
Main ideas:
- implied probability gate
- gap / margin gate
- xG confirm
- PPG confirm
- glue / not-glue handling
- directional power sanity
- CS overlay rescue from `OBSERVE` to `STANDARD`
- later standard demotions for short-price / draw-trap / chaos residue

Important live blockers seen in current system:
- `FTR_ELITE_BLOCK_PICK_SIDE_MARGIN`
- `FTR_ELITE_BLOCK_SELECTION_MISMATCH`
- `FTR_ELITE_BLOCK_POWER`
- `FTR_NOT_GLUE_WARN`

### BTTS
Main ideas:
- split YES / NO lanes
- signal-label gating
- FTS and 0-0 / clean-sheet structural checks
- BTTS NO allowlist policy
- selective standard rescue lanes

### OU25
Main ideas:
- OVER25-only live posture
- review / shadow / UNDER rows pushed out of live tiers
- structural combo and top-3 score support
- league policy tiers
- BTTS-supported rescue lanes

Important OU25 blockers:
- `OU25_REVIEW_BLOCKED_LIVE`
- `OU25_OVER25_STRUCT_FAIL`
- `OU25_OVER25_BASELINE_OBSERVE`
- `OU25_OVER25_BLACKLIST`

## Tier Semantics
- `ELITE`: strongest deploy-ready rows after routing
- `STANDARD`: deployable but lower priority or rescued rows
- `OBSERVE`: non-deployable monitoring / shadow / blocked rows

Hard rule:
- `OBSERVE` rows are not deployable

## Reason Tokens Emitted
Representative tokens emitted directly by this file include:
- FTR: `FTR_CS_PROMOTED`, `FTR_HARD_NOT_GLUE_RESCUE_SHADOW`, `FTR_ELITE_BLOCK_PICK_SIDE_MARGIN`, `FTR_ELITE_BLOCK_SELECTION_MISMATCH`, `FTR_NOT_GLUE_WARN`, `FTR_DRAW_TOP_SOFT_AB`
- BTTS: `BTTS_NO_ALLOWLIST_ONLY`, `BTTS_STANDARD_RESCUE_OU25_PHASE7C`, `BTTS_STANDARD_RESCUE_GLOBAL`
- OU25: `OU25_REVIEW_BLOCKED_LIVE`, `OU25_TOP3_SOFT_AB`, `OU25_DIFFUSE_TOP3_RESCUE_SHADOW`, `OU25_DIFFUSE_TOP3_TIMING_SHADOW`, `OU25_DIFFUSE_TOP3_TIMING_STANDARD`, `OU25_EPL_REVIEW_SOFT_AB`, `OU25_OVER25_STRUCT_FAIL`, `OU25_PREMIUM_WHITELIST_STANDARD`
- generic: `DEMOTED_TO_OBSERVE`, `FALLBACK_STANDARD`, `TEAM_INTEL_CAUTION`, `TEAM_INTEL_UPLIFT`

See also `docs/REASON_TOKENS.md`.

## Protected Rules
- `deploy_rulebook.py` owns routing, tiers, vetoes, and gates.
- Value is additive only and must not override deploy gates.
- Live deploy must not consume `OBSERVE` rows.
- OU25 live posture is intentionally stricter than raw model output.
- Slip composition must not bypass routing decisions.

## Known Negative Controls
Designed fail-closed behaviors:
- OU25 review-league rows demoted from live tiers
- BTTS NO rows outside allowlists demoted to `OBSERVE`
- FTR rows with weak CS-side support demoted
- draw / chaos residue cleanup after standard routing

## Test Commands
Primary audit run:
```bash
./.venv/bin/python deploy_rulebook.py \
  --src predictions_output/<DATE>/BOOKIE_IMP20_ALLMARKETS_<FROM>_to_<TO>.csv \
  --preset V1 \
  --gate-mode preset \
  --ftr-profile accuracy \
  --debug \
  --tier-audit \
  --btts-gate-audit \
  --ou25-gate-audit
```

Soft-gates experiment:
```bash
./.venv/bin/python deploy_rulebook.py \
  --src predictions_output/<DATE>/BOOKIE_IMP20_ALLMARKETS_<FROM>_to_<TO>.csv \
  --preset V1_SOFT_GATES \
  --gate-mode preset \
  --ftr-profile accuracy \
  --debug \
  --tier-audit \
  --btts-gate-audit \
  --ou25-gate-audit
```

## Risks / TODOs
- The file is large and mixes routing, promotion, audit, and reporting responsibilities.
- Some legacy safety gates may now be stricter than necessary for the stronger model stack.
- Remaining live blockers worth revisiting later are now more structural than obviously stale.
- Future API-Football enrichment may provide stronger evidence for borderline FTR / OU25 cases.
