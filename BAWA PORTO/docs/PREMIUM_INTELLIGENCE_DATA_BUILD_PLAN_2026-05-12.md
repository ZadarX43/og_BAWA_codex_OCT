# Premium Intelligence Data Build Plan

Updated: 2026-05-13

## Purpose

Build the premium website data layer once, then scale visibility by subscription tier.

The goal is to support the high-value Odds Genius product shape first:

- deep team intelligence
- deep squad/player intelligence
- fixture decision intelligence
- predicted and confirmed lineup intelligence
- player stats and ratings on lineup cards
- team match stats
- H2H/context cards
- premium alerts and operator review

Lower subscription tiers should receive fewer cards, shorter history, less export depth, and less alerting. They should not require a separate data model.

## Product Principle

The premium product is not “more picks.”

It is a football intelligence operating surface:

- what the model likes
- what the structure supports
- what the squad/player layer says
- what the lineup changes
- what the market confirms or contradicts
- what should be avoided

This is the product value that already showed up in weekend review: the website can flag caution that a raw acca workflow may miss.

## Existing Foundations

Already built or partially built:

- `team_intelligence`
- `player_intelligence`
- `fixture_decision_intelligence`
- `fixture_lineup_intelligence`
- `fixture_h2h_support`
- local SQLite export
- Cloudflare D1 test/production path
- Worker routes for current fixtures, fixture detail, and team detail

Important current truth:

- player ratings are already rich and include positions, rating bands, ranks, and UI summaries
- lineup payloads can be placeholders, predicted from last fixture, or confirmed provider lineups
- the data layer must preserve this status truth rather than pretending every fixture has confirmed lineups

## First Premium Curation Slice

Implement these tables first:

- `site_player_identity_map`
- `site_player_match_stats`
- `site_team_match_stats`
- `site_lineup_slots`
- `site_formation_slots`
- `site_fixture_market_intelligence`
- `site_player_event_shortlists`

The first five tables establish the player/team/lineup foundation. The market and event-shortlist tables establish the first premium product contract for fixture markets and player-event cards without touching live deploy routing.

## Launch Data Limits

The launch build should be deliberately smaller than the theoretical D1 ceiling.

Current working rule:

- default SQLite/D1 export is `active_site_latest_seasons`
- competitions are limited to the 22 currently published site competitions
- seasons are limited to the active published season per competition:
  - `2025/2026` for winter leagues
  - `2026` for calendar-year leagues already published that way
- historical match/player/team rows are not included by default
- deeper history remains behind `--include-history` and later high-tier/pro routes

Practical pre-launch targets:

- launch D1 under 500MB, so it also fits the D1 Free per-database ceiling during testing
- preferred launch target under 300MB once route payloads are finalized
- no route should depend on table scans
- primary page routes should read from cached route payload rows where possible
- raw premium tables remain available for audit and pro/deeper views, not first-render customer pages

First measured launch slice after applying this rule:

- SQLite size: ~99MB
- D1 SQL chunks: ~86MB across 22 chunks
- active fixtures: 156
- active teams: 388
- active squad rows: 388
- active player identity rows: 11,521
- cached fixture stat route payloads: 156
- cached team premium route payloads: 388

Current upstream gap exposed by the launch filter:

- active-window `site_player_match_stats`: 0
- active-window `site_team_match_stats`: 0
- active-window `site_lineup_slots`: 0
- active-window `team_lineup_snapshots`: 0

This is correct behavior for launch safety. It means the database is not silently filling the product with older historical rows. The next data job is to refresh/import current-season provider match stats and lineups for the 22 active competitions, then rerun this same exporter.

Current-site normalized import was added to close that gap from existing report-scoped API-Football outputs:

```bash
python3 scripts/api_football/import_current_site_normalized.py --write
python3 scripts/export_site_sqlite.py --output /private/tmp/odds_genius_current_stats_slice.sqlite
```

Measured after import:

- SQLite size: ~143MB
- D1 SQL chunks: ~132MB across 33 chunks
- `site_lineup_slots`: 12,063
- `site_player_match_stats`: 12,066
- `site_team_match_stats`: 26
- `team_lineup_snapshots`: 282
- `site_player_identity_map`: 15,190

Measured after adding the first market/event premium exporter slice:

- SQLite size: ~155MB
- D1 SQL chunks: ~144MB across 36 chunks
- active fixtures: 156
- `site_fixture_market_intelligence`: 1,092 rows
- `site_player_event_shortlists`: 2,618 rows
- cached fixture stat route payloads: 156
- cached team premium route payloads: 388
- schema version: 4

The player-event rows are intentionally marked as beta/manual-review shortlist intelligence. They are not priced probabilities, not deploy picks, and not permitted to override `deploy_rulebook.py`.

Current coverage truth:

- report-scoped current lineups/player stats exist for 14 of 22 active competitions
- current team stats exist only where `/fixtures/statistics` had already been fetched, mainly small EPL/Bundesliga settled windows
- no current lineups/player stats were found locally yet for Australia A-League, Austria Bundesliga, Denmark Superliga, Germany Bundesliga 2, Saudi Pro League, South Korea K League, Swiss Super League, or Turkey Super Lig
- those remaining gaps require a fresh current-context API-Football refresh, not a website/exporter rewrite

Why this matters:

- Cloudflare D1 Free currently has a 500MB per-database limit
- Workers Paid allows larger databases, but D1 still has a 10GB per-database hard ceiling
- D1 billing is driven by rows read/written and stored GB, so indexed and cached route payloads matter more than just total row count

Operational meaning:

- launch mode is small, active, current-season, and page-shaped
- pro/history mode can be built later from the same schema without forcing a frontend contract rewrite

### `site_player_identity_map`

Purpose:

- connect API player ids to Odds Genius player-rating ids
- keep lineup cards from attaching the wrong rating to the wrong player

Core fields:

- `player_key`
- `api_player_id`
- `rating_player_id`
- `name`
- `canonical_name`
- `club`
- `club_slug`
- `competition_key`
- `season`
- `position`
- `position_group`
- `rating_power`
- `rank_overall`
- `rank_position`
- `rank_club`
- `payload_json`

### `site_player_match_stats`

Purpose:

- fixture/player stats for premium player cards, post-match reports, and player-stat tabs

Source:

- `data_sources/api_football/normalized/match_player_stats.csv`

Core fields:

- fixture identity
- player identity
- team identity
- position and position group
- started/subbed flags
- minutes
- provider match rating
- goals, assists, shots, key passes
- tackles, duels, fouls, cards
- attached Odds Genius rating/rank payload where resolved

### `site_team_match_stats`

Purpose:

- factual team stats for fixture stats cards and post-match review

Source:

- `data_sources/api_football/normalized/match_team_stats.csv`

Core fields:

- possession
- shots
- shots on target
- corners
- fouls
- cards
- passes
- scoreline
- payload JSON for future-safe expansion

### `site_lineup_slots`

Purpose:

- one row per lineup player, confirmed or predicted
- attach player rating, rank, position, and pitch slot

Source:

- `data_sources/api_football/normalized/lineups.csv`
- `site_player_identity_map`
- formation slot resolver

Core fields:

- fixture/team/player identity
- formation
- starter or bench flag
- broad lineup position
- Odds Genius position group
- slot code
- pitch coordinates
- provider match rating where available
- OG player power
- club/position ranks
- payload JSON

### `site_formation_slots`

Purpose:

- stable pitch-coordinate templates for formations
- let confirmed lineups place players correctly instead of broadly by `G/D/M/F`

Core fields:

- formation
- slot code
- broad position
- line index
- slot index
- pitch x/y

### `site_fixture_market_intelligence`

Purpose:

- give Fixture Markets and Fixture Prediction pages a stable premium market contract
- expose the model-facing market posture without asking the frontend to recompute it
- support the initial paid-market hierarchy:
  - FTR
  - OU25
  - BTTS
  - team goals 1.5+
  - correct score
  - corners
  - cards

Source:

- `frontend/public/data/fixture_decision_intelligence/<fixture_key>.json`
- canonical `market_intelligence` and `market_suitability` objects from the fixture decision reconciler

Core fields:

- `fixture_key`
- `market_key`
- `market_family`
- `market_group`
- `market_label`
- `selection_label`
- `rank_role`
- `state`
- `alignment_score`
- `rating`
- `band`
- `model_lean`
- `confidence_band`
- `signal_state`
- support/caution counts
- public summary
- payload JSON

Product use:

- £20 can show the top market posture
- £49/£99 can show the richer market grid and reasoning cards
- £500 can receive route payloads and audit/export metadata later

### `site_player_event_shortlists`

Purpose:

- provide the first data contract for high-value player-event cards
- shortlist players for event families that subscribers care about:
  - shots on target 0.5 / 1.5
  - shots 1.5 / 2.5
  - tackles 0.5 / 1.5
  - fouls 0.5 / 1.5
  - player to be fouled 0.5
  - bookings
  - key passes
  - goalkeeper saves 1.5

Source:

- current-season active-site `site_player_match_stats`
- latest team lineup snapshots or confirmed fixture lineup slots
- `site_player_identity_map` for rating/rank attachment

Important boundary:

- this table is beta/manual-review shortlist intelligence
- it does not create odds, priced probabilities, slips, deploy rows, or automatic alerts
- every row carries `beta_status`, `confidence_label`, `source_lineup_status`, and a reason string

Core fields:

- fixture identity
- event key/family/label/threshold
- player/team identity
- home/away flag
- position and position group
- starter flag
- shortlist rank and score
- recent per-90 and average
- sample size and minutes sample
- OG rating power and ranks
- source lineup status
- beta/manual-review metadata
- payload JSON

## Next Premium Curation Slice

### `site_goal_combo_intelligence`

Purpose:

- expose goal-combo markets after the core market posture is stable:
  - FTR + BTTS
  - FTR + over 2.5
  - FTR + team goals 1.5+
  - over 2.5 + BTTS

This should use the existing model output and reconciler context, but remain a website premium layer until the production spine explicitly supports it.

### `site_fixture_h2h_stats`

Purpose:

- richer H2H cards:
  - last meetings
  - goals
  - BTTS
  - over/under
  - cards
  - venue split
  - direct sample vs historical same-team-pair fallback

This should be implemented as a separate H2H history/regime builder so it does not pollute the player/lineup curation slice.

## Worker Route Direction

Add page-shaped reads:

- `GET /api/site/fixtures/:fixture_key/stats`
- `GET /api/site/teams/:competition_key/:team_slug/premium`

These should read from D1 when available and keep static JSON fallback available on the frontend until parity is proven.

Cached route payload tables:

- `site_fixture_stats_payloads`
- `site_team_premium_payloads`

These are intentionally denormalized. They make the common website reads cheap while the normalized tables remain available for premium drilldown, audit, and later pro/API use.

## Subscription Tiering Principle

Build once, gate by depth:

- low tier: fixture verdict, compact lineup, limited team/player cards
- mid tier: full fixture intelligence, team stats, player drivers, market grid
- high tier: player match stats, full squad/rank cards, H2H regime cards, alerts, exports, API-style reads
- pro/B2B: bulk exports, historical depth, API access, custom alert rules

Do not fork the intelligence schema per tier.

## Implementation Order

1. Extend local SQLite schema.
2. Import the five first-slice premium tables.
3. Add D1 chunk export coverage.
4. Add Worker query helpers and routes.
5. Benchmark fixture/team premium reads.
6. Wire frontend cards progressively.
7. Add tier gating after the full card inventory exists.
