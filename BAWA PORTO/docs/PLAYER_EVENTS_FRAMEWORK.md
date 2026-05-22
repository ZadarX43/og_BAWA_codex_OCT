# PLAYER_EVENTS_FRAMEWORK

## Purpose
Canonical framework for BAWA player-event research, beginning with:
- yellow cards
- fouls committed

This document turns the existing BAWA prompt and analyst notes into a production-aware research framework. It is intended to guide manual analysis, semi-automated reporting, and later API-Football-backed scoring workflows for greenlist / elite leagues where markets are liquid and actually offered.

## Status
Research and beta only.

This framework is not part of the protected production deploy spine yet. It must not change live football prediction routing in:
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/bookie_allmarkets.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/deploy_rulebook.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/slip_formatter.py`

## Initial Scope
### Event families
- player yellow cards
- player fouls committed

### Future candidate event families
- shots
- shots on target
- tackles
- cards + fouls combos
- first-half / second-half split events

### League scope
Start with:
- greenlist leagues
- elite leagues
- markets where bookmakers consistently price player-event lines

Do not expand blindly into leagues or sportsbooks with poor liquidity or inconsistent pricing.

## Core Principles
- Prioritize leagues and fixtures where the market exists and can actually be deployed.
- Treat lineup confirmation, tactical matchup context, and referee profile as first-class inputs.
- Separate pre-match logic from live / in-game adjustments.
- Use a weighted, explainable framework instead of opaque one-shot picks.
- Keep post-match review mandatory so the model can be refined from real outcomes.

## Research Boundary
### Production-safe now
- documenting the framework
- building data contracts
- creating manual or semi-automated reports
- pulling and storing research data for later analysis

### Not production-safe yet
- auto-routing player events into deploy products
- slip completion logic using player events
- value overlays that bypass live football deploy rules
- turning this into a live betting product without validation

## Pre-Match Framework
### 1. Fixture Context
Assess:
- rivalry / derby status
- relegation or title pressure
- knockout or elimination pressure
- expected aggression and tempo
- likely card environment

Useful examples:
- high-stakes matches
- low-margin games with tactical foul incentives
- matches where one side is expected to spend long periods defending

### 2. Referee Profile
Track:
- yellow cards per match
- foul-to-card ratio
- tolerance for dissent
- tolerance for time-wasting
- home / away tilt if evidence is strong

Referee data is a major weight driver for marginal card candidates.

### 3. Historical Player Discipline
For each player candidate, capture:
- fouls per 90
- yellow cards per 90
- minutes played
- booking efficiency
- recent form trend over the last 5 to 10 appearances

Roles with natural priority:
- defensive midfielders
- full-backs / wing-backs
- central defenders facing pace or transition stress
- aggressive pressing forwards

### 4. Tactical and Positional Matchups
Assess:
- formation vs formation interaction
- flank overloads
- isolated defenders
- central overloads
- transition exposure
- likely man-marking or suppression assignments

Common high-risk shapes:
- full-back vs fast direct winger
- defensive midfielder tasked with stopping counters
- overworked wide defender facing overlaps
- center-back forced into recovery defending in space

### 5. Team Discipline and Style
Capture:
- team fouls per match
- team cards per match
- pressing intensity
- transition vulnerability
- tactical foul tendency
- manager/style effect on aggression

### 6. Market Confirmation
Where available, use:
- player booking odds
- player foul lines
- team card lines

The market should confirm or challenge the internal view, not replace it.

## Live / In-Game Adjustment Layer
Use for matchday or in-play refinement:
- confirmed lineups
- final referee assignment
- role changes caused by absences or substitutions
- first-half foul accumulation
- repeated duels on one flank
- scoreline-driven tactical fouls
- late-game time-wasting / dissent risk

Important live triggers:
- player on 2+ fouls early
- defensive role shift after injury/substitution
- trailing team frustration
- protecting-a-lead tactical fouls

## Suggested Variable Families
### Player discipline
- `fouls_per90`
- `yellow_cards_per90`
- `booking_efficiency`
- `recent_foul_trend`
- `recent_card_trend`

### Tactical exposure
- `left_flank_dominance`
- `right_flank_dominance`
- `central_attack_dominance`
- `counterattack_threat`
- `isolated_defender_flag`
- `double_team_pressure_flag`

### Match context
- `match_stakes_score`
- `rivalry_flag`
- `late_game_pressure_risk`
- `trailing_frustration_risk`
- `lead_protection_foul_risk`

### Referee
- `ref_cards_per_match`
- `ref_foul_to_card_ratio`
- `ref_dissent_strictness`
- `ref_timewasting_strictness`

### Workload and fatigue
- `minutes_last_3_matches`
- `days_rest`
- `travel_fatigue_flag`
- `recent_injury_return_flag`

### Advanced performance
- `tackles_per90`
- `interceptions_per90`
- `ground_duel_loss_rate`
- `aerial_duel_loss_rate`
- `dribbles_faced_per90`
- `possession_lost_per90`
- `distance_covered`

## Suggested Weighting Model
Initial weighting for yellow card prediction:
- fixture context: 15%
- player historical discipline: 15%
- player performance / defensive actions: 15%
- referee profile: 15%
- formation / tactical analysis: 10%
- market odds: 10%
- psychological / personal factors: 10%
- environmental / scheduling: 5%
- team dynamics / managerial style: 5%

This is a starting point only. Recalibrate by league and market after post-match review.

## Booking Probability Index
For each player:
1. score each factor family on a standard scale
2. multiply by the assigned weight
3. sum into a `Booking Probability Index`
4. rank players within each fixture

Output tiers:
- high confidence
- moderate confidence
- low confidence

## Reporting Output Standard
Each fixture report should include:
- fixture context
- referee profile
- likely lineups / role notes
- team discipline context
- top booked-player candidates per team
- estimated team card range
- confidence notes
- live adjustment notes

Canonical template:
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/docs/PLAYER_EVENTS_REPORT_TEMPLATE.md`

## Data Sources
### Confirmed / preferred research sources
- API-Football for structured fixture, lineup, player, team, and referee data
- market odds feeds where available
- trusted visual/manual sources for contextual reading when structured data is weak

### Near-term expectation
This project will likely become a heavy API-Football pull for:
- greenlist leagues
- elite leagues
- deployable player-event markets

## Minimum Input Contract
Starter schema:
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/docs/PLAYER_EVENTS_INPUT_SCHEMA.csv`

This schema is intentionally broad enough for:
- manual entry
- spreadsheet workflows
- later API-Football ETL

## Validation Loop
After each match:
1. record actual booked players
2. record actual foul leaders
3. compare top-ranked players vs outcomes
4. identify misses caused by:
   - lineup changes
   - referee misspecification
   - poor tactical assumptions
   - over-weighted market influence
   - unmodeled in-game state changes
5. revise weights and rules

## Recommended Build Order
1. stabilize documentation
2. define data schema
3. ingest structured API-Football player-event inputs
4. build semi-automated report generator
5. validate on greenlist / elite leagues
6. only then consider productization

## Non-Negotiables
- do not mix this research logic into the live football deploy spine yet
- do not treat screenshots alone as a long-term data strategy
- do not expand to broad player-event deployment before market/liquidity validation
- do not skip post-match review

## Source Origin
This document is distilled from:
- the existing BAWA yellow card / foul prompt system
- learned match review notes
- target reporting examples
- live analyst workflow requirements
