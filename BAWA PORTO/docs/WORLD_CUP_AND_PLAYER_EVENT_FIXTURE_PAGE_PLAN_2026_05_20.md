# World Cup + Player Event Fixture Page Plan

## Objective

Prepare the website publishing layer for World Cup + pre-season founder launch without changing production prediction routing.

The site should treat World Cup fixtures as first-class football intelligence pages, while being honest where tournament data differs from domestic league data.

## World Cup Coverage Contract

World Cup fixture pages should use the same compact fixture-brain shape as domestic competitions:

- fixture core
- goal market cards
- team context
- player context
- lineup context
- injury/sidelined shock context
- weather, venue, travel context
- H2H when genuinely available
- freshness and coverage status
- tier visibility

H2H must not be forced. For World Cup fixtures, H2H can be one of:

- `available`: real international H2H exists.
- `partial`: limited senior international H2H or tournament-history context exists.
- `missing`: no trustworthy H2H source is available.
- `unavailable_for_competition`: not enough history to support an H2H card.

Missing H2H should not block a World Cup fixture from being launch-ready.

## Current World Cup Data Position

The repo already has strong World Cup ingredients:

- historical API-Football normalized fixtures, lineups, events, player stats, and team stats for 2018/2022
- World Cup launch scaffold and research model matrix
- qualification context sidecars
- recent history and H2H sidecars
- venue/weather/travel scaffold
- player intelligence scaffold
- player power projection board
- 2026 player-event intelligence boards

Known gaps to fix before launch:

- `World_Cup__merged.csv` currently has no populated power-rating join columns after the standard patch step.
- World Cup odds synthesis does not follow the domestic `fd_odds_enriched_synth.csv` path; the adapter synthesizes under 2.5 directly and needs to be treated as an explicit World Cup source mode.
- Team/match tackles need a first-class website shortlist producer.
- H2H coverage needs a page-safe status rather than a hard expectation.

## Player Event Card Stack

Player event cards should sit immediately below the goal-market cards on fixture pages.

The public language should be watchlist/intelligence language, not bet-slip certainty:

- `Beta shortlist`
- `Preview list`
- `Lineup pending`
- `Confirmed lineup refresh`
- `Manual review`

Card families:

- Player Shots
- Shots On Target
- Player Tackles
- Player Fouls
- Player Fouled
- Bookings Watch
- Keeper Saves
- Key Passes
- Team / Match Tackles

Each card should carry:

- event family and label
- beta status
- lineup phase
- shortlist source scope
- top candidates
- score
- sample size / minutes sample where available
- short reason
- confidence label

Do not show missing families as empty confident cards. Show them as missing coverage or hide them behind a coverage note.

## Lineup Phase Model

Use this lifecycle for domestic and World Cup fixtures:

- `pre_tournament_projection`: World Cup-only early coverage before final matchday context.
- `pre_lineup_preview`: last starting XI, bench, player event rates, team event rates, injuries, and tactical role context.
- `lineup_pending`: fixture exists but shortlist context is not ready.
- `lineup_confirmed_refresh`: rebuilt after confirmed lineup automation, normally around T-60.
- `settled_audit_ready`: post-match state for proof and player-event audit.

The website should make the phase visible in the card chrome or metadata, not bury it in expert text.

## Tier Positioning

Standard/free view:

- fixture hero
- FTR, OU25, BTTS, TG1.5 market cards
- freshness and coverage state
- player-event availability teaser only

Founder/Premium:

- fixture context
- team context
- weather/travel
- H2H when available
- lineup summary
- injury/sidelined warnings

Pro:

- player-event cards
- player and team event shortlists
- richer role/context explanations

Pro+:

- audit dashboard
- model/source drilldowns
- downloadable compact intelligence
- raw compact debug where safe

## Automation Flow

Daily or pre-window:

1. Refresh local source data.
2. Build domestic and World Cup compact source sidecars.
3. Build pre-lineup player-event preview boards.
4. Compile fixture brain payloads.
5. Publish changed payloads only.
6. Smoke frontend, Worker, D1 index, and gated routes.

Matchday:

1. Refresh injuries and sidelined context.
2. Refresh fixture/team/player context.
3. At roughly T-60, fetch confirmed lineups.
4. Rebuild player-event fixture inputs.
5. Rebuild player-event cards.
6. Recompile and delta-publish fixture brain payloads.
7. Smoke fixture pages and premium gating.

Post-match:

1. Ingest results.
2. Settle public picks.
3. Update weekly results and archive.
4. Store player-event audit rows where available.

## Next Build Steps

1. Add a World Cup fixture completeness auditor that checks logos, goal-market outputs, weather/travel, H2H status, team/player context, player-event boards, lineups, injuries, and freshness.
2. Add a player-event card producer for team/match tackles.
3. Fix the World Cup power-rating join so tournament fixtures receive populated context where available.
4. Add an explicit World Cup adapter source flag for under 2.5 odds synthesis.
5. Wire `player_event_cards` into the fixture page below market cards and gate it to Pro+ surfaces according to the payload allowlist.
