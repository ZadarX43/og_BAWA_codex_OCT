# MODEL_OUTPUT_SCHEMA

Updated: `2026-05-04`

## Purpose

Document how routed deploy CSV fields are transformed into the first website-safe Odds Genius publishing layer.

## Source Contract

Primary source:
- newest true `predictions_output/**/DEPLOY_COMBINED_*.csv`

Explicitly excluded:
- `DEPLOY_COMBINED_SCORED_*.csv`

## Row Eligibility

### Premium export
- allowed tiers: `ELITE`, `STANDARD`
- blocked tier: `OBSERVE`

### Public export
- preferred tier: `ELITE`
- fallback tier: small `STANDARD` subset only if no `ELITE` rows exist

## Public JSON Schema

```json
{
  "fixture_id": "2026_03_29_Atletico_PR_Botafogo__FTR__HOME",
  "fixture_key": "2026_03_29_Atl_tico_PR_Botafogo",
  "kickoff_time": "2026-03-29",
  "league": "Brazil Serie A",
  "home_team": "Atlético PR",
  "away_team": "Botafogo",
  "market": "FTR",
  "pick": "HOME",
  "confidence_tier": "ELITE",
  "display_confidence": "High",
  "bookie_od": 2.5,
  "model_prob_display": "53%",
  "value_edge_display": "+13.2 pts",
  "short_reason": "Top-rated home result cleared live routing checks.",
  "is_free": true
}
```

## Premium JSON Schema

```json
{
  "fixture_id": "2026_03_29_Atletico_PR_Botafogo__FTR__HOME",
  "fixture_key": "2026_03_29_Atl_tico_PR_Botafogo",
  "kickoff_time": "2026-03-29",
  "league": "Brazil Serie A",
  "home_team": "Atlético PR",
  "away_team": "Botafogo",
  "market": "FTR",
  "pick": "HOME",
  "confidence_tier": "ELITE",
  "model_prob": 0.5324,
  "bookie_implied_prob": 0.4,
  "value_edge": 0.1324,
  "bookie_od": 2.5,
  "reason_tokens": ["DEPLOYABLE", "MARKET_FTR", "TIER_ELITE"],
  "human_reason": "Strong home-result signal with enough support to stay live.",
  "slip_role_hint": "anchor",
  "safe_for_small_acca_flag": true,
  "safe_for_large_acca_flag": true,
  "correct_score_shortlist": [
    {"scoreline": "1-0", "probability": 0.185},
    {"scoreline": "2-0", "probability": 0.11},
    {"scoreline": "2-1", "probability": 0.09}
  ],
  "premium_tier": "ELITE"
}
```

## Field Mapping

### Identity
- `fixture_id`
  - source: derived
  - rule: `<fixture_key>__<market>__<pick>`
- `fixture_key`
  - source: `fixture_key`
- `kickoff_time`
  - preferred source: `match_date`
- `league`
  - source: `league`
- `home_team`
  - preferred source: `home_team_name`
  - fallback: `home_team`
- `away_team`
  - preferred source: `away_team_name`
  - fallback: `away_team`

### Pick surface
- `market`
  - source: `market`
  - normalization: uppercase canonical values such as `FTR`, `BTTS`, `OU25`
- `pick`
  - preferred source: `selection`
  - fallback: `bookie_pick`
  - fallback: `pick`
- `confidence_tier`
  - source: `deploy_tier`
  - normalization: uppercase tier values
- `premium_tier`
  - source: `deploy_tier`

### Pricing and probability
- `bookie_od`
  - source: `bookie_od`
- `model_prob`
  - source: `model_p_for_bookie`
- `bookie_implied_prob`
  - preferred source: `bookie_implied`
  - fallback: `bookie_implied_novig`
- `value_edge`
  - preferred source: `gap`
  - fallback: `gap_novig`

### Public display derivations
- `display_confidence`
  - derived from `model_prob`
  - bucketed label only
- `model_prob_display`
  - derived from `model_prob`
  - rounded percentage string
- `value_edge_display`
  - derived from `value_edge`
  - rounded percentage-point string
- `short_reason`
  - derived from sanitized market/tier context
  - must not expose raw tokens

### Premium explanation fields
- `reason_tokens`
  - preferred source: `context_reason_codes`
  - fallback: `deploy_reason_codes`
  - then sanitized and filtered
- `human_reason`
  - derived from market, tier, and safe token surface
- `slip_role_hint`
  - derived from tier and market
- `safe_for_small_acca_flag`
  - derived from tier
- `safe_for_large_acca_flag`
  - derived from tier

### Correct score shortlist
- sources:
  - `cs1`, `cs1_p`
  - `cs2`, `cs2_p`
  - `cs3`, `cs3_p`
- output:
  - array of up to three objects
  - each object includes `scoreline` and rounded `probability`

## Fallback Strategy

When a mapped source field is missing:
- use the documented fallback column if one exists
- otherwise drop the row only if a critical identity/pick field is missing
- otherwise emit a safe placeholder and record the fallback in `publish_summary.json`

Critical fields:
- `fixture_key`
- `kickoff_time`
- `league`
- `home_team`
- `away_team`
- `market`
- `pick`
- `confidence_tier`

## Excluded Source Fields

The exporter must never pass through raw deploy CSV columns wholesale.

Examples of intentionally excluded source families:
- `context_reason_codes` raw string
- `deterministic_*`
- `learned_*`
- `meta_*`
- `support_*`
- `uefa_*`
- `*policy*`
- `*gate*`
- `*veto*`
- `*threshold*`
- `*lambda*`
- `*p00*`
- `*xg*`
- `__source_*`

## Output Files

- `frontend/public/data/public_predictions.json`
- `frontend/public/data/premium_predictions.json`
- `frontend/public/data/publish_summary.json`
- `reports/latest/PUBLISH_REPORT.md`
