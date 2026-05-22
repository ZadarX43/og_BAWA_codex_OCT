# PLAYER_EVENTS_API_FOOTBALL_PLAN

## Purpose
Design the first API-Football-backed pull plan for BAWA player-event research using:
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/docs/PLAYER_EVENTS_INPUT_SCHEMA.csv`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/docs/PLAYER_EVENTS_FRAMEWORK.md`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/docs/PLAYER_EVENTS_REPORT_TEMPLATE.md`

This plan is for research and beta only. It does not alter the production football deploy spine.

## Mission
Build a reliable pre-match and light live-context dataset for:
- player yellow cards
- player fouls committed

The first version should support:
- manual analyst review
- semi-automated markdown reports
- later scoring/ranking models

## Phase 1 Target Scope
### Target leagues
Phase 1 should focus on leagues where:
- lineup data is reliable
- referee identity is consistently published
- player card markets are actually listed
- foul / tackles / duel profiles are meaningful

Recommended first target leagues:
- England Premier League
- Spain La Liga
- Italy Serie A
- Germany Bundesliga
- France Ligue 1
- UEFA Champions League
- UEFA Europa League

Phase 1.5 expansion candidates:
- England Championship
- Netherlands Eredivisie
- Scotland Premiership
- Portugal Liga

Do not expand into every league at once. Add leagues only after market coverage and report usefulness are confirmed.

### Target sportsbooks
Primary sportsbook targets:
- `bet365`
- `Sky Bet`
- `Paddy Power`
- `William Hill`
- `Betfair Sportsbook`
- `Unibet`

Secondary / watchlist sportsbook targets:
- `BetVictor`
- `Coral`
- `Ladbrokes`
- `BetMGM`

Priority rule:
- only treat a sportsbook as primary if it regularly posts player card markets in the target leagues
- foul-committed markets are lower coverage and should be treated as additive, not required

### Target player-event markets
Phase 1 primary:
- player to be shown a card
- player over fouls committed
- team total cards

Phase 1 supporting:
- same-game combinations using card logic only for research
- team card environment / total match cards as context

Phase 2 candidates:
- player tackles
- player shots
- player shots on target

## Data Products
### Minimum usable output
One row per fixture-player candidate with enough context to rank yellow card and foul risk.

### Canonical input contract
Use:
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/docs/PLAYER_EVENTS_INPUT_SCHEMA.csv`

This schema is the staging contract between:
- raw API-Football normalized tables
- manual analyst notes
- eventual scoring/report scripts

### Pre-kickoff readiness gate
Use before generating high-confidence player-event boards:

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/docs/PLAYER_EVENTS_PRE_KICKOFF_READINESS_CHECKLIST.md`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/docs/PLAYER_EVENTS_PRE_KICKOFF_READINESS_FIELDS.csv`

The readiness gate answers whether a fixture has enough confirmed lineup, player profile, tactical matchup, fixture environment, and referee/discipline context to support a pre-kickoff player-event prediction.

## Source Mapping
### API-Football normalized tables already aligned with the wider API layer
From the existing API scaffolding:
- `fixtures_master.csv`
- `match_team_stats.csv`
- `match_events.csv`
- `match_player_stats.csv`
- `lineups.csv`
- `injuries.csv`
- `odds_prematch_long.csv`

### Mapping into `PLAYER_EVENTS_INPUT_SCHEMA.csv`
#### Fixture / context fields
- `fixture_key`:
  - from `fixtures_master.csv`
- `match_date`:
  - from `fixtures_master.csv`
- `competition`, `league`:
  - from `fixtures_master.csv`
- `home_team_name`, `away_team_name`, `venue`:
  - from `fixtures_master.csv`

#### Referee fields
- `referee_name`:
  - from fixture metadata where available
- `ref_cards_per_match`:
  - built from rolling historical referee aggregates
- `ref_foul_to_card_ratio`:
  - built from historical referee match stats
- `ref_dissent_strictness`, `ref_timewasting_strictness`:
  - initially manual / analyst-derived
  - later event-derived heuristics if event detail coverage is good enough

#### Market availability
- `market_yellow_cards_available`
- `market_fouls_available`
- `market_team_cards_available`
  - derived from prematch bookmaker market coverage in `odds_prematch_long.csv`

#### Player identity / role
- `team_name`, `player_name`, `player_team_side`
- `expected_start_flag`
- `expected_minutes`
- `position_group`
- `tactical_role`
- `likely_marking_assignment`
  - from `lineups.csv`, `players_master.csv`, and analyst enrichment

#### Discipline / defensive metrics
- `fouls_per90`
- `yellow_cards_per90`
- `tackles_per90`
- `interceptions_per90`
- `ground_duel_loss_rate`
- `aerial_duel_loss_rate`
- `dribbles_faced_per90`
- `minutes_last_3_matches`
- `days_rest`
  - mainly from rolling transforms over `match_player_stats.csv`

#### Team / tactical environment
- `team_avg_fouls`
- `team_avg_yellows`
- `opponent_possession_projection`
- `left_flank_dominance`
- `right_flank_dominance`
- `central_attack_dominance`
- `late_game_pressure_risk`
- `lead_protection_foul_risk`
- `trailing_frustration_risk`
  - built from:
    - team rolling stats
    - event timing distributions
    - matchup heuristics
    - analyst overlays

#### Odds fields
- `book_yellow_odds`
- `book_foul_line`
- `book_foul_over_odds`
  - from `odds_prematch_long.csv`
  - likely needs a bookmaker/market normalization layer for player-event mappings

## Required Derived Builders
### 1. Referee profile builder
New output suggestion:
- `data_sources/api_football/features/api_referee_profile_features.csv`

Should contain:
- referee rolling card rates
- foul-to-card ratio
- home vs away card tilt
- late-game card tendency

### 2. Player-event rolling builder
New output suggestion:
- `data_sources/api_football/features/api_player_event_features.csv`

Should contain:
- fouls per 90
- yellow cards per 90
- recent foul trend
- recent card trend
- defensive workload metrics
- likely role grouping

### 3. Player-event market coverage builder
New output suggestion:
- `reports/api_football/api_player_event_market_coverage_report.csv`

Should answer:
- which leagues have yellow card markets
- which leagues have foul lines
- which books post them reliably

### 4. Fixture-player staging builder
New output suggestion:
- `data_sources/api_football/features/player_events_fixture_input.csv`

This should be the machine-built version of the starter schema, ready for report generation.

## Pull Sequence
### Phase 1: Data foundation
1. fixtures
2. lineups
3. player stats
4. team stats
5. match events
6. prematch odds
7. injuries

### Phase 2: Derived research layer
1. referee profile features
2. player rolling event features
3. tactical exposure features
4. sportsbook market coverage report
5. fixture-player staging table

### Phase 3: Reporting layer
1. semi-auto yellow card report
2. semi-auto foul report
3. analyst review pass
4. post-match review capture

## Manual vs Automated Split
### Automated first
- fixture metadata
- player rolling discipline
- team discipline
- referee aggregates
- bookmaker market availability
- prematch player-event prices where available

### Analyst-supplied first
- likely marking assignment
- tactical role nuance
- dissent strictness
- time-wasting strictness
- psychological flags
- final dark-horse shortlist logic

This keeps the first version realistic and avoids pretending API data alone can replace football reading.

## Validation Rules
- do not score player-event fixtures where starting XI confidence is weak
- do not produce high-confidence yellow card reports without referee identity
- do not pretend foul markets are broadly available if coverage is sparse
- do not expand leagues before market coverage reporting says the market actually exists

## Recommended Deliverables
### Docs
- this plan
- target market coverage memo
- post-match review protocol

### Scripts
- fixture-player staging builder
- yellow card report generator
- later foul report generator

### Reports
- weekly player-event market coverage report
- fixture watchlist
- post-match hits/misses report

## Phase 1 Success Criteria
We can call phase 1 successful when:
- the target leagues are mapped cleanly into `PLAYER_EVENTS_INPUT_SCHEMA.csv`
- at least one repeatable yellow card report can be generated from structured input
- sportsbook market coverage is known league-by-league
- post-match review can compare prediction vs actual outcomes

## Next Immediate Build Steps
1. create target scope memo
2. build first manual/semi-auto yellow card report generator
3. design fixture-player staging output from API-Football normalized tables
4. run a small fixture sample through the full workflow
