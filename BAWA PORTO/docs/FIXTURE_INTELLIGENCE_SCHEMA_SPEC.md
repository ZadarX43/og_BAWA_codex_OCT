# Fixture Intelligence Schema Spec

## Purpose

Define the concrete website-safe JSON contract for the first fixture intelligence publishing layer.

This spec turns the publishing plan into implementation-ready schema guidance for:
- `DEPLOY`
- `OBSERVE`
- `CONTEXT`
- `MONITOR`

It is designed so a first exporter can:
- classify fixtures
- map safe fields
- generate website-safe JSON
- support followed-team / followed-fixture delivery later

This spec does not change:
- live routing logic
- `bookie_allmarkets.py`
- `deploy_rulebook.py`
- protected model internals

## Output Family

Recommended first artifact:
- `frontend/public/data/fixture_intelligence_public.json`

Optional future artifacts:
- `frontend/public/data/fixture_intelligence_premium.json`
- `frontend/public/data/fixture_follow_digest.json`
- `frontend/public/data/fixture_alert_feed.json`

The first exporter should target:
- `fixture_intelligence_public.json`

## Top-Level JSON Shape

Recommended top-level contract:

```json
{
  "generated_at": "2026-05-09T12:00:00Z",
  "source_run_id": "2026-05-09",
  "source_window": {
    "date_from": "2026-05-09",
    "date_to": "2026-05-11"
  },
  "coverage_summary": {
    "total_fixtures": 0,
    "deploy_count": 0,
    "observe_count": 0,
    "context_count": 0,
    "monitor_count": 0,
    "hidden_count": 0,
    "covered_leagues_count": 0
  },
  "fixtures": []
}
```

## Top-Level Metadata Fields

### `generated_at`
- ISO 8601 UTC timestamp
- required

### `source_run_id`
- operator-friendly publish lineage id
- required
- example:
  - `2026-05-09`
  - `weekend_2026-05-09`

### `source_window`
- optional but recommended
- includes:
  - `date_from`
  - `date_to`

### `coverage_summary`
- required
- counts by publish class
- useful for account dashboards, Telegram digests, and sanity checks

### `fixtures`
- required array
- each item follows the fixture schema below

## Per-Fixture Base Schema

Every published fixture should carry a shared base contract.

```json
{
  "fixture_id": "12345",
  "fixture_key": "eng_premierleague_arsenal_chelsea_2026-05-09",
  "publish_class": "DEPLOY",
  "coverage_status": "covered",
  "kickoff_time": "2026-05-09T15:00:00Z",
  "league": "Premier League",
  "league_logo_url": "https://...",
  "league_flag_url": "https://...",
  "home_team": "Arsenal",
  "home_team_logo_url": "https://...",
  "away_team": "Chelsea",
  "away_team_logo_url": "https://...",
  "logo_join_status": "FULL_MATCH",
  "odds_summary": {},
  "signal_summary": {},
  "context_summary": {},
  "follow_relevance": {},
  "updated_at": "2026-05-09T12:00:00Z"
}
```

## Base Fields

### Identity and timing
- `fixture_id`
  - string
  - optional if not reliably available
- `fixture_key`
  - string
  - required whenever canonical join exists
- `kickoff_time`
  - ISO 8601 UTC string
  - required

### Classification
- `publish_class`
  - required enum:
    - `DEPLOY`
    - `OBSERVE`
    - `CONTEXT`
    - `MONITOR`
- `coverage_status`
  - required enum:
    - `covered`
    - `follow_only`
    - `hidden`

### Competition and team identity
- `league`
- `league_logo_url`
- `league_flag_url`
- `home_team`
- `home_team_logo_url`
- `away_team`
- `away_team_logo_url`
- `logo_join_status`

Identity fields should reuse the same website-safe badge bridge logic already used by predictions.

### Structural sub-objects
- `odds_summary`
- `signal_summary`
- `context_summary`
- `follow_relevance`

### Meta
- `updated_at`
  - ISO 8601 UTC string
  - required

## Shared Sub-Objects

## `odds_summary`

Purpose:
- compact, safe, high-level bookmaker context

Contract:

```json
{
  "home_win_odds": 2.1,
  "draw_odds": 3.4,
  "away_win_odds": 3.5,
  "btts_yes_odds": 1.8,
  "over25_odds": 1.9,
  "odds_snapshot_status": "available"
}
```

Allowed fields:
- `home_win_odds`
- `draw_odds`
- `away_win_odds`
- `btts_yes_odds`
- `btts_no_odds`
- `over25_odds`
- `under25_odds`
- `home_team_over15_odds`
- `away_team_over15_odds`
- `odds_snapshot_status`

Notes:
- all odds fields optional
- `odds_snapshot_status` enum:
  - `available`
  - `partial`
  - `missing`

## `signal_summary`

Purpose:
- safe summary of model-derived deploy or observe shape

Contract:

```json
{
  "signal_state": "deploy",
  "market_family": "BTTS",
  "signal_label": "Observed BTTS lean",
  "signal_strength": "low",
  "confidence_tier": null,
  "deploy_pick": null,
  "premium_tier": null,
  "summary_text": "Observed BTTS lean based on attacking convergence and defensive instability.",
  "context_tags": ["attacking_convergence", "defensive_instability"]
}
```

Allowed fields:
- `signal_state`
  - enum:
    - `deploy`
    - `observe`
    - `context_only`
    - `monitor_only`
- `market_family`
  - examples:
    - `FTR`
    - `BTTS`
    - `OU25`
    - `TEAM_GOALS`
    - `CORRECT_SCORE`
- `signal_label`
  - safe public-facing label
- `signal_strength`
  - enum:
    - `high`
    - `medium`
    - `low`
    - `none`
- `confidence_tier`
  - allowed only for `DEPLOY`
  - examples:
    - `ELITE`
    - `STANDARD`
- `deploy_pick`
  - allowed only for `DEPLOY`
- `premium_tier`
  - optional
- `summary_text`
  - required when there is any signal or observe shape
- `context_tags`
  - safe, compressed tags

Important:
- `OBSERVE` and weak signals must not use recommendation language
- use:
  - `Observed BTTS lean`
  - `Observed home-side lean`
  - `Goal-shape profile suggests elevated scoring potential`

Avoid:
- `Prediction`
- `Pick`
- `Bet`

## `context_summary`

Purpose:
- compact environmental and contextual intelligence

Contract:

```json
{
  "weather_note": "Heavy rain risk before kickoff.",
  "injury_note": "Key attacker flagged as doubtful.",
  "lineup_note": "Lineups not confirmed yet.",
  "formation_note": "Expected back-four structure.",
  "market_movement_note": "Moderate away-price drift observed.",
  "fatigue_note": "Congestion risk after midweek fixture.",
  "form_note": "Home side scoring profile remains stable.",
  "h2h_note": "Recent meetings have leaned high-event."
}
```

Allowed fields:
- `weather_note`
- `injury_note`
- `lineup_note`
- `formation_note`
- `market_movement_note`
- `fatigue_note`
- `form_note`
- `h2h_note`
- `schedule_note`
- `volatility_note`

All fields optional.
All notes should be:
- brief
- normalized
- non-raw

## `follow_relevance`

Purpose:
- indicate whether the fixture is relevant for user watchlists later

Contract:

```json
{
  "team_follow_candidate": true,
  "fixture_follow_candidate": true,
  "league_follow_candidate": true,
  "market_follow_candidate": true,
  "notification_priority": "medium"
}
```

Allowed fields:
- `team_follow_candidate`
- `fixture_follow_candidate`
- `league_follow_candidate`
- `market_follow_candidate`
- `notification_priority`

`notification_priority` enum:
- `critical`
- `high`
- `medium`
- `low`
- `none`

## Class-Specific Rules

## `DEPLOY` Fixture Contract

`DEPLOY` fixtures must include:
- base schema
- `signal_summary.signal_state = "deploy"`
- `signal_summary.market_family`
- `signal_summary.confidence_tier`
- `signal_summary.deploy_pick`
- `signal_summary.signal_strength`
- `signal_summary.summary_text`

Optional:
- `signal_summary.premium_tier`
- `correct_score_summary`
- `value_edge_label`

Recommended extra object for deploy:

```json
{
  "deploy_summary": {
    "market": "BTTS",
    "pick": "YES",
    "confidence_tier": "ELITE",
    "bookie_od": 1.8,
    "value_edge_label": "positive"
  }
}
```

Allowed fields:
- `market`
- `pick`
- `confidence_tier`
- `bookie_od`
- `value_edge_label`

Do not expose:
- raw exact hidden thresholds
- model internals
- verbose reason-token dumps

## `OBSERVE` Fixture Contract

`OBSERVE` fixtures must include:
- base schema
- `signal_summary.signal_state = "observe"`
- `signal_summary.market_family`
- `signal_summary.signal_label`
- `signal_summary.signal_strength`
- `signal_summary.summary_text`

Optional:
- `context_tags`
- `context_summary`

Example:

```json
{
  "publish_class": "OBSERVE",
  "signal_summary": {
    "signal_state": "observe",
    "market_family": "OU25",
    "signal_label": "Observed scoring lean",
    "signal_strength": "low",
    "summary_text": "Goal-shape profile suggests elevated scoring potential, but not enough for deployment.",
    "context_tags": ["goal_shape_positive", "not_deployable"]
  }
}
```

## `CONTEXT` Fixture Contract

`CONTEXT` fixtures may have no meaningful model signal.

They must include:
- base schema
- `signal_summary.signal_state = "context_only"`
- `context_summary` with at least one note

Example:

```json
{
  "publish_class": "CONTEXT",
  "signal_summary": {
    "signal_state": "context_only",
    "signal_strength": "none",
    "summary_text": "No deployable edge. Context remains relevant."
  },
  "context_summary": {
    "weather_note": "Strong wind expected before kickoff.",
    "injury_note": "Multiple absences affecting the home midfield."
  }
}
```

## `MONITOR` Fixture Contract

`MONITOR` fixtures are the completeness layer.

They must include:
- base schema
- `signal_summary.signal_state = "monitor_only"`
- one concise monitor summary

Example:

```json
{
  "publish_class": "MONITOR",
  "signal_summary": {
    "signal_state": "monitor_only",
    "signal_strength": "none",
    "summary_text": "Covered fixture. No strong deploy signal at current state."
  }
}
```

They may also carry:
- odds summary
- one context note
- follow relevance

## Optional Helper Objects

## `correct_score_summary`

Allowed only if already considered public-safe.

Contract:

```json
{
  "scorelines": ["1-0", "2-1", "1-1"],
  "summary_text": "Top scoreline cluster remains home-leaning."
}
```

## `team_goals_summary`

Optional if team-goals shape is being surfaced.

Contract:

```json
{
  "home_scoring_shape": "positive",
  "away_scoring_shape": "muted",
  "summary_text": "Home side scoring shape remains stronger than away-side output."
}
```

## Field Provenance Table

### From deploy / routed output
- `fixture_id`
- `fixture_key`
- `kickoff_time`
- `league`
- team labels
- market family
- deploy pick
- confidence tier
- premium tier
- safe summary labels

### From logo/identity bridge
- `home_team_logo_url`
- `away_team_logo_url`
- `league_logo_url`
- `league_flag_url`
- `logo_join_status`

### From API-Football / enrichment after normalization
- kickoff identity confirmation
- lineup / formation note
- injury note
- market movement note
- odds summary
- fixture metadata

### From weather / other enrichments
- `weather_note`
- `fatigue_note`
- `schedule_note`

## Safe Vocabulary Rules

### Allowed for deploy
- `deployment`
- `signal`
- `edge`
- `confidence tier`

### Allowed for observe/context
- `observed lean`
- `shape`
- `profile`
- `context`
- `monitor`
- `elevated`
- `fragile`

### Avoid for observe/context
- `pick`
- `bet`
- `prediction`
- `banker`
- `lock`

## Exporter Expectations

The first exporter should:

1. ingest approved routed outputs and approved coverage inputs
2. classify each fixture into:
   - `DEPLOY`
   - `OBSERVE`
   - `CONTEXT`
   - `MONITOR`
   - or `HIDDEN`
3. map safe fields only
4. emit deterministic JSON
5. include coverage counts

### Important rule

If a fixture cannot be cleanly classified or lacks safe identity fields:
- demote to `HIDDEN`
- record it in operator reporting
- do not publish partial raw junk

## Validation Rules

Minimum validation rules for the future exporter:

- every published fixture must have:
  - `publish_class`
  - `kickoff_time`
  - `league`
  - `home_team`
  - `away_team`
  - `updated_at`
- `DEPLOY` requires:
  - market family
  - confidence tier
  - deploy pick
- `OBSERVE` requires:
  - safe summary text
  - no deploy-only wording
- no raw provider payload fragments
- no raw hidden model fields

## Immediate Follow-On Tasks

1. Task 54 — OBSERVE To Public-Safe Export Rules
2. Task 55 — Followed Team / Fixture Notification Rules
3. Task 56 — Fixture Intelligence JSON Publisher
4. Task 57 — Fixture Intelligence Validator

## Summary

This schema gives the platform a concrete publish contract for fixtures that are:
- deployable
- non-deployable but meaningful
- contextual only
- monitor-worthy

That is the bridge from:
- deploy-only board publishing

to:
- full fixture intelligence publishing
