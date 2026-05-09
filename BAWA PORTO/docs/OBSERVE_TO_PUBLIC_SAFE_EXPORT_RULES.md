# OBSERVE To Public-Safe Export Rules

## Purpose

Define how `OBSERVE` rows and weak model shape should be translated into website-safe fixture intelligence.

This document exists so the future exporter can:
- consume routed `OBSERVE` rows
- classify their meaning
- generate safe public labels
- generate safe summary text
- emit compact context tags
- avoid leaking internal deploy mechanics
- avoid presenting non-deployable rows as picks

This is a publishing-layer rulebook.
It does not alter:
- `deploy_rulebook.py`
- live gate logic
- live routing semantics

## Core Principle

`OBSERVE` is meaningful, but it is not deployable.

That means every `OBSERVE` export must satisfy both:

1. informational value
2. non-recommendation discipline

If a row fails either of those:
- do not publish it as `OBSERVE`
- instead demote it to:
  - `CONTEXT`
  - `MONITOR`
  - or `HIDDEN`

## Meaning Of OBSERVE In The Product

`OBSERVE` should mean:
- model or structure saw something worth watching
- the live rulebook did not approve deployment
- advanced users may still find the signal shape useful
- the platform should frame it as awareness, not action

Product translation:
- not a pick
- not a recommendation
- not dead data
- a monitored signal shape

## Allowed Product Language

### Safe lead-in vocabulary
- `Observed`
- `Model lean`
- `Shape`
- `Profile`
- `Context`
- `Monitoring note`
- `Signal state`

### Safe label patterns
- `Observed BTTS lean`
- `Observed scoring lean`
- `Observed home-side lean`
- `Observed away-side lean`
- `Observed draw-fragile profile`
- `Observed volatility warning`
- `Monitor scoring shape`

### Unsafe label patterns
- `Prediction`
- `Pick`
- `Bet`
- `Back this`
- `Banker`
- `Win call`
- `Strong recommendation`

## Export Classification Flow For OBSERVE Rows

For each routed `OBSERVE` row:

1. identify its market family
2. inspect safe token families
3. determine whether the row expresses:
   - weak directional lean
   - structural failure
   - volatility / fragility
   - shadow / blocked review state
4. translate into one of:
   - `OBSERVE`
   - `CONTEXT`
   - `MONITOR`
   - `HIDDEN`

### Promote to website `OBSERVE` if:
- there is a coherent safe label
- there is a useful summary that is not recommendation language
- there is enough fixture identity/context to make the row understandable

### Demote to `CONTEXT` if:
- the model shape is too weak to phrase as a lean
- the main value is the contextual warning
- the row is more about environment than market shape

### Demote to `MONITOR` if:
- there is no strong safe summary
- the row adds only weak completeness value
- the user may still care because the fixture is covered or followed

### Demote to `HIDDEN` if:
- the row is too internal
- the tokens are too route-specific
- the row depends on hidden research logic
- identity or field quality is poor

## OBSERVE Export Buckets

Recommended public-safe buckets:

### 1. Weak directional lean

Examples:
- weak BTTS YES / NO shape
- soft OU25 over lean
- home-side lean below deploy threshold
- away-side lean below deploy threshold

Safe label examples:
- `Observed BTTS lean`
- `Observed scoring lean`
- `Observed home-side lean`
- `Observed away-side lean`

### 2. Structural caution

Examples:
- over structure failed
- double-blank / FTS conflict
- top-3 shape too diffuse
- side support too weak

Safe label examples:
- `Observed structural caution`
- `Observed scoring friction`
- `Observed low-conviction shape`

### 3. Volatility / fragility

Examples:
- chaos warning
- draw-trap residue
- lineup instability
- derby instability
- heavy market conflict

Safe label examples:
- `Observed volatility warning`
- `Observed fragile fixture profile`
- `Observed draw-risk context`

### 4. Team-goals / scoring-shape context

Examples:
- home-side scoring tilt
- away-side scoring suppression
- muted total-goals profile
- elevated scoreline cluster

Safe label examples:
- `Observed home scoring shape`
- `Observed away scoring caution`
- `Observed goal-shape profile`

## Market-Specific Translation Rules

## FTR OBSERVE Rules

FTR OBSERVE rows often arise from:
- weak side support
- power mismatch blockers
- draw risk
- chaos residue
- glue / not-glue conflict

### Safe export behavior

If the row still expresses directional value:
- `Observed home-side lean`
- `Observed away-side lean`

If the row is mainly about instability:
- `Observed draw-risk context`
- `Observed fragile fixture profile`
- `Observed volatility warning`

### Safe summary examples
- `Observed home-side lean, but side support was not strong enough for deployment.`
- `Observed away-side lean, with enough shape to monitor but not enough stability to deploy.`
- `Fixture profile showed draw and chaos risk, so the row remained in observe state.`

### FTR tokens that should usually remain internal
- `FTR_ELITE_BLOCK_PICK_SIDE_MARGIN`
- `FTR_ELITE_BLOCK_SELECTION_MISMATCH`
- `FTR_ELITE_BLOCK_POWER`
- `FTR_CS_PROMOTED`
- `FTR_HARD_NOT_GLUE_RESCUE_SHADOW`

These may inform the bucket chosen, but they should not be emitted verbatim.

## BTTS OBSERVE Rules

BTTS OBSERVE rows often arise from:
- weak YES / NO signal
- FTS conflict
- 0-0 or clean-sheet risk
- allowlist / policy restrictions

### Safe export behavior

If the shape still leans attacking:
- `Observed BTTS lean`

If the shape is muted:
- `Observed low-event BTTS profile`

If the row is mostly blocked by fragility:
- `Observed BTTS caution`

### Safe summary examples
- `Observed BTTS lean based on attacking shape, but not enough stability for deployment.`
- `BTTS profile remained active to monitor, though defensive and fail-to-score risk kept it out of deploy.`
- `The row carried some BTTS shape, but low-event caution remained too strong for deployment.`

### BTTS tokens that should usually remain internal
- `BTTS_YES_NOT_LIVE`
- `BTTS_YES_HARD_BLACKLIST`
- `BTTS_NO_ALLOWLIST_ONLY`
- `BTTS_STANDARD_RESCUE_GLOBAL`
- `BTTS_STANDARD_RESCUE_OU25_PHASE7C`

## OU25 OBSERVE Rules

OU25 OBSERVE rows often arise from:
- structural fail
- review / shadow state
- top-3 diffusion
- blacklist policy
- weak support below live threshold

### Safe export behavior

If the row still leans high-event:
- `Observed scoring lean`
- `Observed over profile`

If the row is structurally weak:
- `Observed structural caution`

If the row is diffuse / uncertain:
- `Observed low-conviction goal profile`

### Safe summary examples
- `Goal-shape profile suggested elevated scoring potential, but structural support remained too weak for deployment.`
- `Observed over profile remained in monitoring state because the scoring shape was still too diffuse.`
- `Scoring conditions looked mildly positive, but not with enough structural confirmation to deploy.`

### OU25 tokens that should usually remain internal
- `OU25_REVIEW_BLOCKED_LIVE`
- `OU25_SHADOW_BLOCKED_LIVE`
- `OU25_OVER25_STRUCT_FAIL`
- `OU25_OVER25_BASELINE_OBSERVE`
- `OU25_DIFFUSE_TOP3_TIMING_SHADOW`

## Team-Goals / Secondary Market OBSERVE Rules

For later public-safe team-goals shapes:

Use:
- `Observed home scoring shape`
- `Observed away scoring shape`
- `Observed muted away output`
- `Observed dominant home profile`

Avoid:
- direct over-1.5 pick wording
- exact internal model phrasing

## Safe Context Tag Rules

Context tags should be:
- compact
- consistent
- non-internal
- useful for filtering later

Recommended public-safe tags:
- `attacking_shape`
- `defensive_instability`
- `low_event_risk`
- `high_event_potential`
- `draw_risk`
- `chaos_risk`
- `volatility`
- `weather_caution`
- `injury_caution`
- `lineup_pending`
- `goal_shape_positive`
- `goal_shape_muted`
- `home_side_lean`
- `away_side_lean`
- `not_deployable`

Avoid exporting internal tags such as:
- raw gate tokens
- route branch names
- threshold language
- bundle / model internals

## Summary Generation Rules

Every exported `OBSERVE` row should produce:

1. `signal_label`
2. `signal_strength`
3. `summary_text`
4. `context_tags`

### `signal_label`
- short
- user-facing
- not recommendation language

### `signal_strength`

For public-safe OBSERVE use:
- `medium`
  - coherent shape, strong enough to monitor closely
- `low`
  - weak shape or caution-led signal

Do not emit:
- `high` for `OBSERVE`

### `summary_text`

Should be:
- one sentence
- plain English
- no internal token names
- no exact threshold references

### `context_tags`

Should be:
- 1 to 4 tags max
- selected from the safe public tag set

## Translation Decision Table

### If row has weak directional shape and limited instability
- export as `OBSERVE`
- label with market-family lean

### If row has directional shape but significant fragility
- export as `OBSERVE`
- label with fragility-aware summary

### If row is mostly structural failure without user-facing lean
- export as `CONTEXT`

### If row is simply covered and mildly interesting
- export as `MONITOR`

### If row is heavily internal or token-led only
- export as `HIDDEN`

## Public-Safe Examples

### FTR Example

```json
{
  "publish_class": "OBSERVE",
  "signal_summary": {
    "signal_state": "observe",
    "market_family": "FTR",
    "signal_label": "Observed home-side lean",
    "signal_strength": "low",
    "summary_text": "Observed home-side lean, but side support was not strong enough for deployment.",
    "context_tags": ["home_side_lean", "not_deployable"]
  }
}
```

### BTTS Example

```json
{
  "publish_class": "OBSERVE",
  "signal_summary": {
    "signal_state": "observe",
    "market_family": "BTTS",
    "signal_label": "Observed BTTS lean",
    "signal_strength": "medium",
    "summary_text": "Observed BTTS lean based on attacking shape, but not enough stability for deployment.",
    "context_tags": ["attacking_shape", "defensive_instability", "not_deployable"]
  }
}
```

### OU25 Example

```json
{
  "publish_class": "OBSERVE",
  "signal_summary": {
    "signal_state": "observe",
    "market_family": "OU25",
    "signal_label": "Observed scoring lean",
    "signal_strength": "low",
    "summary_text": "Goal-shape profile suggested elevated scoring potential, but structural support remained too weak for deployment.",
    "context_tags": ["goal_shape_positive", "not_deployable"]
  }
}
```

## Fields That Must Never Leak From OBSERVE Translation

Do not publish:
- raw `context_reason_codes`
- raw `reason_codes`
- exact hidden thresholds
- exact internal probabilities that are not already approved
- internal route branch names
- private model path or artifact references
- provider-specific raw payloads

## Relationship To Notifications

`OBSERVE` exports are valid inputs to notifications, but not all should notify immediately.

### Good Telegram candidates
- followed fixture observe shift
- followed team observe update
- high-priority volatility / fragility observe state

### Better as digest
- low-strength observe shape
- routine monitor-plus-observe entries

## Validation Rules For OBSERVE Export

Each exported `OBSERVE` row must satisfy:
- `publish_class = "OBSERVE"`
- `signal_summary.signal_state = "observe"`
- `signal_summary.signal_label` present
- `signal_summary.summary_text` present
- no banned recommendation vocabulary
- no raw internal tokens emitted

Recommended banned words list:
- `pick`
- `prediction`
- `bet`
- `banker`
- `guaranteed`
- `lock`

## Exporter Implementation Guidance

The future exporter should implement:

1. token family inspection
2. market-family-specific translation
3. safe label generation
4. safe summary generation
5. safe tag generation
6. final class decision

The exporter should be deterministic and auditable.

## Immediate Follow-On Tasks

1. Task 55 — Followed Team / Fixture Notification Rules
2. Task 56 — Fixture Intelligence JSON Publisher
3. Task 57 — Fixture Intelligence Validator

## Summary

`OBSERVE` should become a public-safe awareness layer, not a shadow picks feed.

The rule is simple:
- translate signal shape
- keep the intelligence
- remove the route mechanics
- never pretend weak signal is a deployable recommendation
