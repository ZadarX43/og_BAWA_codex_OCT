# Premium Intelligence Data Build Plan

Updated: 2026-05-12

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

## Next Premium Curation Slice

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

