# Intelligence Preferences And Alert Taxonomy Plan

## Purpose

Define the next product layer that turns Odds Genius from:
- a premium picks surface

into:
- a personalised football intelligence platform

This document covers:
- alert categories
- user preference schema
- Telegram delivery rules
- team / fixture / market follow model
- the boundary between deploy signals and intelligence alerts
- what belongs in free, premium, and future Pro tiers

## Core Product Shift

The key principle is:

Users should still receive value even when there are no deployable picks.

That means Odds Genius should operate across five linked layers:

1. Signals
   - deployable betting opportunities
2. Intelligence
   - context, risk, and pre-match insight
3. Monitoring
   - favourite teams, leagues, fixtures, and markets
4. Delivery
   - website, Telegram, and later email / app pushes
5. Preferences
   - user-controlled depth, timing, and alert scope

## Product Model

### 1. Deploy Signals

These remain:
- selective
- confidence-tiered
- higher-value
- premium-led

Examples:
- ELITE BTTS
- ELITE OU25
- STANDARD FTR
- premium value-edge deployment
- elite acca drop
- correct-score shortlist support

These are recommendation-adjacent outputs and should remain gated and deliberately limited.

### 2. Intelligence Alerts

These are broader informational outputs that may or may not imply a bet.

Examples:
- key player injury
- late team news
- lineup disruption
- rotation risk
- weather disruption
- market drift / line movement
- schedule congestion
- derby volatility
- scoring-shape bias
- fixture pressure / fragility

These alerts are valuable even when there is no deployable signal.

### 3. Monitoring Layer

Users should be able to follow:
- teams
- leagues
- fixtures
- markets
- later: player-event areas

This is important because a subscriber may want intelligence around:
- a club they support
- a league they trade
- a fixture they plan to bet manually
- a market family they specialise in

Even if Odds Genius publishes no premium signal, the platform can still provide intelligence.

### 4. Delivery Layer

The same event may be rendered differently by channel:

- website:
  - richer detail
  - cards
  - archived history
  - account preferences
- Telegram:
  - fast alerts
  - concise summaries
  - digests
- email later:
  - morning briefs
  - results summaries
- push later:
  - urgent live or pre-match notifications

### 5. Preferences Layer

Each user should be able to choose:
- what to follow
- what to ignore
- how often to be contacted
- which channel to use
- how much intelligence vs signal delivery they want

## Alert Taxonomy

Recommended high-level taxonomy:

### A. Signal Alerts

Purpose:
- deliver actionable deployable picks

Types:
- `elite_deployment`
- `strong_deployment` (future if introduced)
- `standard_deployment`
- `acca_drop`
- `value_edge_trigger`
- `correct_score_shortlist`

Default tone:
- direct
- selective
- concise
- premium

### B. Intelligence Alerts

Purpose:
- communicate important context that could affect outcome quality or market shape

Types:
- `injury_news`
- `lineup_disruption`
- `rotation_risk`
- `weather_alert`
- `market_movement`
- `referee_volatility`
- `schedule_congestion`
- `derby_instability`
- `fatigue_flag`
- `scoring_shape_shift`
- `fragility_warning`

Default tone:
- informational
- risk-aware
- non-hype

### C. Watchlist Alerts

Purpose:
- deliver updates about what the user has explicitly chosen to follow

Types:
- `followed_team_update`
- `followed_fixture_update`
- `followed_league_update`
- `followed_market_update`

Example:
- a user follows Arsenal and receives:
  - injury alert
  - weather impact
  - scoring-shape summary
  - elite deployment if one exists

### D. Digest Alerts

Purpose:
- bundle intelligence and reduce noise

Types:
- `daily_digest`
- `weekend_slate_digest`
- `results_digest`
- `followed_team_digest`
- `followed_fixture_summary`

### E. Future Live Alerts

Not for the first implementation, but the taxonomy should reserve space for:
- `live_pressure_spike`
- `live_xg_imbalance`
- `red_card_impact`
- `momentum_shift`
- `live_weather_change`

## Alert Severity And Priority

Every alert should carry a product-level priority, separate from betting tier:

- `critical`
  - high-impact late injury
  - major lineup shock
  - severe weather
  - elite deployment
- `high`
  - strong market movement
  - rotation warning
  - premium watchlist update
- `medium`
  - daily intelligence note
  - followed team pre-match insight
- `low`
  - digest items
  - archive / recap information

This lets delivery logic decide:
- immediate push to Telegram
- batch into digest
- website-only

## Preference Schema Direction

The current `notification_preferences` table should expand beyond the initial booleans.

Recommended new preference groups:

### Channel preferences
- `telegram_enabled`
- `email_enabled`
- `website_only_mode`

### Signal preferences
- `elite_alerts_enabled`
- `standard_alerts_enabled`
- `acca_alerts_enabled`
- `correct_score_alerts_enabled`

### Intelligence preferences
- `injury_alerts_enabled`
- `weather_alerts_enabled`
- `market_movement_alerts_enabled`
- `volatility_alerts_enabled`
- `team_news_alerts_enabled`

### Digest preferences
- `daily_digest_enabled`
- `results_digest_enabled`
- `weekend_slate_digest_enabled`

### Monitoring preferences
- `followed_teams_json`
- `followed_leagues_json`
- `followed_markets_json`
- `followed_fixtures_json`

### Delivery behavior
- `quiet_hours_json`
- `alert_frequency_mode`
- `pre_match_window_minutes`
- `allow_non_signal_intelligence`

Suggested `alert_frequency_mode` values:
- `immediate`
- `digest_only`
- `mixed`

## Team / Fixture / Market Follow Model

This is the retention engine.

### Teams

Users can follow a club even when there is no live deploy signal.

That unlocks:
- injury alerts
- team news
- weather changes
- tactical or scoring-shape summaries
- volatility warnings
- results summaries

This is especially important for:
- club supporters
- users with their own betting systems
- users who trade specific leagues or teams manually

### Fixtures

Users can follow a specific match.

That unlocks:
- pre-match intelligence summary
- lineup disruption alert
- weather change
- odds movement
- post-match result and grading

### Markets

Users can follow:
- BTTS
- OU25
- FTR
- team goals
- later player-event categories

That unlocks:
- targeted signal notifications
- targeted market-specific intelligence summaries

### Leagues

Users can follow:
- EPL
- La Liga
- MLS
- UCL
- etc.

That unlocks:
- weekend slate digests
- league-specific deployment summaries
- league volatility patterns

## Deploy Signal vs Intelligence Alert Boundary

This boundary must stay strict.

### A deploy signal is:
- routed by the live engine
- approved by model/gate logic
- recommendation-adjacent
- tiered
- eligible for premium board publication

### An intelligence alert is:
- informational
- contextual
- may affect outcome quality
- may exist without any deployable signal
- must not silently imply a pick when none exists

### Rule

No intelligence alert should masquerade as a deploy signal.

That means:
- no hidden recommendation language
- no “bet this now” framing if no signal exists
- no upgrade of contextual information into deployment without rulebook approval

## Telegram Delivery Rules

Telegram should stay:
- selective
- valuable
- not spammy

### Recommended default routing

Immediate Telegram:
- elite deployment
- acca drop
- major injury alert on followed team or followed fixture
- severe weather alert on followed fixture
- major market movement on followed fixture

Digest Telegram:
- daily summary
- weekend slate digest
- results digest
- followed team pre-match bundle

Website-only by default:
- lower-priority informational notes
- archive detail
- full rationale panels
- richer methodology context

### Telegram message design rules

Messages should be:
- short
- structured
- clearly labeled
- non-tipster in tone

Suggested format:

1. header
   - `ELITE DEPLOYMENT`
   - `TEAM INTELLIGENCE`
   - `WEATHER ALERT`
2. subject
   - fixture / team / league
3. summary
   - one to three short lines
4. optional CTA
   - `Open premium board`
   - `View fixture`
   - `See results`

## Free vs Premium vs Future Pro

### Free

Can receive:
- limited public signals
- delayed digests
- selected public results summaries
- light follow capability later

Free should not receive:
- full elite alerts
- premium board drops
- high-frequency personalised Telegram delivery

### Premium / OG Founder

Should receive:
- full deployable board
- elite and standard Telegram alerts
- acca drops
- results digest
- favourite team / fixture intelligence
- pre-match intelligence summaries
- key injury / weather / volatility alerts where relevant

This is the tier where “personalised intelligence feed” becomes real.

### Future Pro

Should expand into:
- deeper filtering
- richer fixture intelligence
- market-specific advanced alerts
- experimental live overlays
- player-event intelligence
- more granular timing and automation controls

## Legal And Positioning Advantage

This architecture is valuable because:

- not every message is a betting recommendation
- many messages are informational
- users can consume intelligence even without betting
- the platform becomes broader than “tips”

Positioning should remain:
- prediction intelligence
- informational analysis
- selective deployment
- monitored risk context

Avoid:
- guarantee language
- “banker” framing
- reckless urgency

## UX Principles

The system should feel:
- controlled
- selective
- intelligent
- personal

It should not feel:
- noisy
- spammy
- generic
- like a Telegram picks channel

Every user should feel:

`This platform understands what I care about.`

## Recommended Implementation Order

### Phase 1 — Preference foundation
- expand notification preference schema
- define alert taxonomy enums
- add account preference controls
- wire Telegram routing decisions

### Phase 2 — Follow system
- favourite teams
- favourite leagues
- favourite markets
- favourite fixtures

### Phase 3 — Intelligence summaries
- pre-match summaries
- injury / weather / volatility intelligence cards
- digest generation rules

### Phase 4 — Live / reactive layer
- live event monitoring
- real-time shifts
- in-play alerts

## Immediate Follow-On Tasks

1. Task 51 — Notification Preferences Schema Expansion
2. Task 52 — Account Preferences UX Spec
3. Task 53 — Telegram Alert Templates And Routing Rules
4. Task 54 — Followed Team / Fixture Data Model

## Summary

Odds Genius should not stop at:
- “what should I bet on?”

It should answer:
- what is happening around this match?
- what has changed?
- what matters?
- what should I monitor?
- where is the deployable edge, if one exists?

That is how the product becomes:

`a personalised football intelligence operating system`

instead of:

`a premium picks page`
