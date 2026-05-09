# Followed Team And Fixture Notification Rules

## Purpose

Define how followed teams, fixtures, leagues, and markets should generate user-facing notifications.

This document specifies:
- follow types
- event sources
- notification triggers
- priority rules
- channel routing
- noise suppression rules
- how deploy, observe, context, and monitor classes should behave for followers

This is a delivery-layer ruleset.
It does not change:
- live deploy logic
- raw model routing
- the protected prediction spine

## Core Principle

A follow should create relevance, not noise.

If a user follows:
- a team
- a fixture
- a league
- a market

they should receive:
- only material updates
- in the right channel
- at the right urgency
- using safe intelligence language

The goal is:
- personalised awareness
- not alert spam

## Supported Follow Types

### 1. Team follow

User explicitly follows a club.

Examples:
- Arsenal
- Porto
- Inter Miami

Team follow should unlock:
- deploy alerts involving that team
- observe alerts involving that team
- injury / lineup / weather / volatility context
- team-centric digests

### 2. Fixture follow

User explicitly follows a match.

Examples:
- Arsenal vs Chelsea
- Barcelona vs Real Madrid

Fixture follow should unlock:
- pre-match context alerts
- observe/deploy transitions
- lineup/weather/news changes
- post-match result recap

### 3. League follow

User explicitly follows a competition.

Examples:
- Premier League
- Champions League
- MLS

League follow should unlock:
- slate digests
- notable deploy summaries
- major league-specific volatility notes

### 4. Market follow

User explicitly follows a market family.

Examples:
- BTTS
- OU25
- FTR
- Team Goals

Market follow should unlock:
- deploy alerts in that market
- observe / shape updates in that market
- digest summaries grouped by market

## Event Sources

Notifications should be generated from the published intelligence layer, not raw source payloads.

Primary event sources:
- deploy board publish
- observe-safe fixture intelligence publish
- context-safe fixture intelligence publish
- settlement / results publish

Optional later event sources:
- lineup confirmation ingest
- weather threshold crossing
- news/injury enrichment
- live-state updates

### Important boundary

Do not generate follower notifications directly from:
- raw API-Football JSON
- raw provider news dumps
- internal route tokens
- pre-normalized odds payloads

Always route through:
- normalized publish-safe intelligence events

## Notification Event Classes

Every follow notification should be classified as one of:

### A. `follow_deploy`

Triggered when a followed entity matches a deployable signal.

Examples:
- followed team appears in ELITE deploy
- followed fixture gets a STANDARD deploy row
- followed market gets a new premium signal

### B. `follow_observe`

Triggered when a followed entity matches a safe published `OBSERVE` entry.

Examples:
- followed fixture develops an observed BTTS lean
- followed team match enters volatility-led observe state

### C. `follow_context`

Triggered when the main value is contextual intelligence.

Examples:
- major injury on followed team
- severe weather on followed fixture
- lineup disruption
- market movement on followed fixture

### D. `follow_digest`

Triggered when updates should be bundled rather than pushed immediately.

Examples:
- daily team digest
- weekend followed league digest
- followed market digest
- results digest

### E. `follow_results`

Triggered after settlement.

Examples:
- followed fixture final result
- followed deploy result
- followed team recap

## Trigger Rules By Follow Type

## Team Follow Rules

### Immediate candidates
- elite deploy involving followed team
- severe weather affecting followed team fixture
- high-impact injury affecting followed team fixture
- major lineup disruption affecting followed team fixture
- high-priority observe shift involving followed team

### Digest candidates
- medium-priority observe summary
- daily team intelligence bundle
- weekly team results recap

### Website-only candidates
- low-priority monitor-only updates
- routine fixture listing
- archive details

## Fixture Follow Rules

Fixture follows are the highest-intent follow type.

### Immediate candidates
- fixture enters deploy state
- severe weather alert
- major injury / lineup alert
- sharp market movement
- high-priority volatility warning

### Digest candidates
- observe summary
- pre-match summary bundle
- post-match wrap-up

### Website-only candidates
- low-level completeness updates

## League Follow Rules

League follows should be more selective than team/fixture follows.

### Immediate candidates
- only elite deployments by default
- severe competition-wide event if relevant

### Digest candidates
- weekend slate digest
- daily league deploy summary
- league results summary

### Website-only candidates
- low-signal monitor traffic

## Market Follow Rules

### Immediate candidates
- elite or high-priority deploy in followed market
- critical observe/context event if directly tied to followed market

### Digest candidates
- grouped observe summaries for that market
- daily market digest

### Website-only candidates
- low-priority context notes

## Priority Model

Each notification candidate should be assigned:
- a follow relevance score
- a product priority
- a delivery urgency

### Priority levels
- `critical`
- `high`
- `medium`
- `low`
- `none`

### Suggested mapping

#### `critical`
- elite deploy on followed fixture
- major injury on followed team
- severe weather on followed fixture
- extreme market movement on followed fixture

#### `high`
- standard deploy on followed fixture
- observe volatility warning on followed team
- major lineup disruption

#### `medium`
- observe lean on followed market
- followed-team pre-match summary
- team-goals shape summary

#### `low`
- monitor-only note
- archive recap item
- low-conviction observe note

## Channel Routing Rules

## Telegram

Use for:
- critical immediate alerts
- selected high-priority alerts
- digests when user enabled them

Telegram should not receive:
- every low-priority observe row
- every covered fixture update

### Telegram defaults
- `critical` -> immediate
- `high` -> immediate if user selected immediate/mixed mode
- `medium` -> digest by default
- `low` -> website only or digest only

## Website account feed

Use for:
- complete preference-aware history
- low-priority updates
- follow monitoring feed
- archive browsing

Website is the safest home for:
- lower urgency intelligence
- full per-follow visibility

## Email later

Use for:
- morning digest
- weekend digest
- results digest

## Delivery Mode Interaction

Use the stored `alert_frequency_mode`:

### `immediate`
- send immediate Telegram for `critical` and `high`
- medium can still be batched if noisy

### `mixed`
- critical/high immediate
- medium/low batched into digest

### `digest_only`
- no immediate push except optional future override for critical
- everything collected into scheduled summaries

## Noise Suppression Rules

This is one of the most important sections.

### Rule 1: deduplicate by fixture + event family

Do not send multiple alerts that say effectively the same thing.

Example:
- injury update
- lineup caution
- volatility note

If they all resolve to the same practical event, prefer one consolidated message.

### Rule 2: one dominant alert per fixture state change

If a fixture moves:
- `MONITOR -> OBSERVE`
- `OBSERVE -> DEPLOY`

send the most important state-change message, not three separate ones.

### Rule 3: suppress low-priority alerts when a higher-priority alert already fired recently

Example:
- elite deploy fired
- do not also send a weak observe lean message for the same fixture shortly after

### Rule 4: avoid repeated market drift spam

Market movement must cross a meaningful threshold or change state before firing again.

### Rule 5: do not send non-signal intelligence if the user disabled it

This must respect:
- `allow_non_signal_intelligence`
- injury/weather/team-news toggles

## Consolidation Rules

When multiple relevant updates happen for one followed fixture, prefer a single consolidated summary message.

Example:

Instead of:
- injury alert
- weather alert
- observe alert

Prefer:
- `Fixture intelligence update`
  - injury concern
  - weather caution
  - observed BTTS lean

This is especially valuable for Telegram.

## Trigger Rules By Publish Class

## If fixture is `DEPLOY`

### Team follow
- immediate if alert class enabled

### Fixture follow
- immediate

### League follow
- immediate only if elite or if user chose broader deploy alerts

### Market follow
- immediate if matching market alerts enabled

## If fixture is `OBSERVE`

### Team follow
- immediate only if:
  - priority is high
  - or user explicitly opted into observe-style intelligence
- otherwise digest

### Fixture follow
- digest by default
- immediate if observe event is tagged `critical` or `high`

### League follow
- digest only

### Market follow
- digest by default

## If fixture is `CONTEXT`

### Team follow
- immediate only for:
  - major injury
  - severe weather
  - lineup shock
  - extreme market movement
- otherwise digest

### Fixture follow
- immediate for critical context
- digest for medium context

### League follow
- digest only

### Market follow
- usually website/digest only unless directly market-critical

## If fixture is `MONITOR`

Default:
- website only

Optional:
- digest inclusion for followed team / league / fixture

## User Preference Gating

A follow notification should only fire if all relevant gates pass.

### Required gates
- entity is followed
- channel is enabled
- alert class is enabled
- priority meets delivery threshold
- no recent suppression window conflict

### Optional gates
- quiet hours window
- pre-match time window
- allow-non-signal-intelligence

## Pre-Match Window Rules

Use `pre_match_window_minutes` to suppress stale or too-early alerts.

Recommended default behavior:
- do not send routine pre-match follow alerts outside the configured window
- exceptions:
  - major injury
  - severe weather
  - elite deploy

## Quiet Hours Rules

If quiet hours are configured:
- batch non-critical alerts until the next allowed window
- critical alerts may later override, but first implementation can simply delay everything

## Notification Payload Contract

Recommended internal payload shape:

```json
{
  "event_id": "follow_evt_123",
  "event_type": "follow_context",
  "priority": "high",
  "entity_type": "fixture",
  "entity_key": "arsenal_chelsea_2026-05-09",
  "channel_recommendation": "telegram_immediate",
  "headline": "Fixture intelligence update",
  "summary_lines": [
    "Major injury concern on the away side.",
    "Weather risk elevated before kickoff.",
    "Observed BTTS lean remains non-deployable."
  ],
  "fixture_id": "12345",
  "fixture_key": "eng_premierleague_arsenal_chelsea_2026-05-09",
  "updated_at": "2026-05-09T12:00:00Z"
}
```

This is an internal delivery object, not necessarily a public webpage object.

## Example Rules

### Example 1 — Followed team with elite deploy

If:
- user follows Arsenal
- Arsenal fixture gets ELITE BTTS
- `elite_alerts_enabled = true`
- `telegram_enabled = true`

Then:
- send immediate Telegram alert

### Example 2 — Followed fixture with heavy rain and lineup shock

If:
- user follows Arsenal vs Chelsea
- weather alert priority = critical
- lineup disruption priority = high

Then:
- send one consolidated Telegram update

### Example 3 — Followed market with weak observe lean

If:
- user follows `OU25`
- fixture publishes `OBSERVE` scoring lean
- priority = medium

Then:
- include in digest by default
- do not send immediate push unless user explicitly enables broader intelligence mode later

## Relationship To Future Features

These rules should later support:
- in-app push notifications
- saved watchlist dashboard
- daily intelligence timeline
- followed team pages
- premium fixture pages

## Immediate Follow-On Tasks

1. Task 56 — Fixture Intelligence JSON Publisher
2. Task 57 — Fixture Intelligence Validator
3. Task 58 — Notification Event Builder
4. Task 59 — Account Preference UX Refinement For Follows

## Summary

Follow notifications should make Odds Genius feel:
- selective
- aware
- personal
- intelligent

They should not feel:
- noisy
- repetitive
- generic

The rule is:

`follow creates relevance, not spam`
