# REASON_TOKENS

## Purpose
Canonical reference for important reason tokens, labels, and routing markers seen across deploy generation.

This is not yet a complete machine-generated registry. It is a curated code-derived map from:
- `deploy_rulebook.py`
- `deploy_gates.py`
- `slip_formatter.py`
- `bookie_allmarkets.py`

## Main Token Columns
- `context_reason_codes`
- `reason_codes`
- `standard_reporting_bucket`
- `deploy_tier` / `tier`

## Token Families

### Tier / routing
- `DEMOTED_TO_OBSERVE`
- `FALLBACK_STANDARD`
- `STANDARD_FTR_CS_PROMOTED`
- `STANDARD_FTR_CS_PROMOTED_ALIGNED`
- `STANDARD_OU25`
- `STANDARD_BTTS`
- `STANDARD_OTHER`
- `ELITE`
- `OBSERVE`

### FTR blockers / promotions
- `FTR_CS_PROMOTED`
- `FTR_CS_PROMOTE_STANDARD`
- `FTR_CS_SHADOW_ONLY`
- `FTR_HARD_NOT_GLUE_RESCUE_SHADOW`
- `FTR_NOT_GLUE_WARN`
- `FTR_NOT_GLUE_SOFT_AB`
- `FTR_DRAW_RISK_WARN`
- `FTR_CHAOS_WARN`
- `FTR_ELITE_BLOCK_SELECTION_MISMATCH`
- `FTR_ELITE_BLOCK_POWER`
- `FTR_ELITE_BLOCK_PICK_SIDE_MARGIN`
- `FTR_ELITE_BLOCK_DOUBLE_CHAOS_WARN`
- `FTR_OBSERVE_RESCUE_STANDARD`
- `FTR_DRAW_TOP_SOFT_AB`

### OU25 blockers / promotions
- `OU25_PREMIUM_WHITELIST_STANDARD`
- `OU25_REVIEW_BLOCKED_LIVE`
- `OU25_SHADOW_BLOCKED_LIVE`
- `OU25_UNDER_BLOCKED_LIVE`
- `OU25_TOP3_SOFT_AB`
- `OU25_DIFFUSE_TOP3_RESCUE_SHADOW`
- `OU25_DIFFUSE_TOP3_TIMING_SHADOW`
- `OU25_DIFFUSE_TOP3_TIMING_STANDARD`
- `OU25_EPL_REVIEW_SOFT_AB`
- `OU25_OVER25_STRUCT_FAIL`
- `OU25_OVER25_BASELINE_OBSERVE`
- `OU25_OVER25_BLACKLIST`
- `OU25_OVER25_TIER1_PASS`
- `OU25_OVER25_TIER2_PASS`
- `OU25_OVER25_FALLBACK_PASS`
- `OU25_STANDARD_RESCUE_BTTS_PHASE7C`

### BTTS blockers / promotions
- `BTTS_NO_ALLOWLIST_ONLY`
- `BTTS_STANDARD_RESCUE_GLOBAL`
- `BTTS_STANDARD_RESCUE_OU25_OVER`
- `BTTS_STANDARD_RESCUE_OU25_PHASE7C`
- `BTTS_STANDARD_RESCUE_WEAK_YES`
- `BTTS_YES_BASELINE_LIVE`
- `BTTS_YES_NOT_LIVE`
- `BTTS_YES_HARD_BLACKLIST`
- `BTTS_META_ELITE`
- `BTTS_NO_META_ELITE`

### Team-intel / context overlays
- `TEAM_INTEL_CAUTION`
- `TEAM_INTEL_UPLIFT`
- `TEAM_INTEL_AVOID_IN_ACCA`
- `TEAM_HOME_GE2_POCKET`
- `TEAM_AWAY_GE2_POCKET`
- `TEAM_CS_VS_CS`
- `TEAM_SCORER_VS_SCORER`

### Deterministic vetoes from `deploy_gates.py`
- `FTR_NO_SIDE_TRUE_CLOSE`
- `FTR_NO_SIDE_CHAOS_MARGIN`
- `BTTS_YES_FTS_TOO_HIGH`
- `BTTS_YES_CS1_TRAP`
- `BTTS_NO_HIGH_GOALS_LOW_P00`
- `OVER25_P00_TOO_HIGH`
- `OVER25_DOWNGRADE_CS1_UNDERMAGNET`

### Signal / strength labels
These are often upstream labels rather than deploy reason codes, but they influence routing:
- `VERY_STRONG_OVER`, `STRONG_OVER`, `WEAK_OVER`
- `VERY_STRONG_UNDER`, `STRONG_UNDER`, `WEAK_UNDER`
- `VERY_STRONG_YES`, `STRONG_YES`, `WEAK_YES`
- `VERY_STRONG_NO`, `STRONG_NO`, `WEAK_NO`
- `SOFT_ALIGN`, `STRONG_ALIGN`

## How to Read Them
- Tokens in `context_reason_codes` describe why a row was kept, demoted, rescued, or tagged.
- Multiple tokens on one row are common and expected.
- A row can carry both positive and negative context, e.g. promoted by one lane and demoted later by another gate.
- Final deployability is determined by tier, not by any single token alone.

## Most Important Current Meanings
- `FTR_ELITE_BLOCK_PICK_SIDE_MARGIN`
  - selected FTR side lacks enough separation over competing side in top-3 correct-score support
- `OU25_OVER25_STRUCT_FAIL`
  - OU25 row failed structural live-policy requirements even if model / top3 support exists
- `OU25_DIFFUSE_TOP3_TIMING_SHADOW`
  - shadow-only Portugal/Spain refinement of the diffuse-top3 OU25 lane requiring supportive second-half acceleration and OU25 regime blend
- `OU25_DIFFUSE_TOP3_TIMING_STANDARD`
  - narrow STANDARD rescue for Portugal Liga and Spain La Liga when the diffuse-top3 shadow shape is confirmed by both second-half acceleration and OU25 regime blend
- `TEAM_INTEL_CAUTION`
  - cautionary overlay from team intelligence context
- `FTR_CS_PROMOTED`
  - FTR row rescued by correct-score support overlay
- `FTR_HARD_NOT_GLUE_RESCUE_SHADOW`
  - shadow-only FTR audit marker for strong side-aligned rows still carrying hard `not_glue` risk

## Test Commands
Quick token census in a tier file:
```bash
python3 - <<'PY'
import pandas as pd
p='predictions_output/<DATE>/<FILE>.csv'
df=pd.read_csv(p)
print(df['context_reason_codes'].fillna('').str.get_dummies(sep='|').sum().sort_values(ascending=False).head(50))
PY
```

## Risks / TODOs
- A full machine-generated registry would be better than a curated list.
- Some uppercase labels are signal tags, not deploy-routing tokens; keep that distinction clear.
- Future API-Football enrichment may justify adding new context token families rather than weakening existing ones blindly.
