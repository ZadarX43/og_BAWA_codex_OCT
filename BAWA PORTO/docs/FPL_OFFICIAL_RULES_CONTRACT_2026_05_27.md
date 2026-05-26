# FPL Official Rules Contract

Date: 2026-05-27

Source basis:
- Official Fantasy Premier League Help / Rules: `https://fantasy.premierleague.com/help/`
- User-supplied full rules extract in the product planning thread.

Purpose: make the Odds Genius Fantasy Intelligence system rule-complete before the UI starts saving season-long squads, picks, drafts, chips, and transfer plans.

## Rule Contract Now Encoded

The executable contract lives in:

`scripts/fpl/fpl_rules_engine.py`

It now covers:

- 15-player squad: 2 GK, 5 DEF, 5 MID, 3 FWD
- £100.0m initial budget
- maximum 3 players per Premier League club
- starting XI: 11 players, 1 GK, at least 3 DEF, at least 1 FWD
- captain and vice-captain must be different starting players
- four-player bench order validation
- automatic substitution formation protection
- captain fallback through vice-captain when captain does not play
- official selling-price rule: half of price rise, rounded down to nearest £0.1m
- free transfer cost: extra transfers cost -4 points
- stored free transfers capped at 5
- wildcard/free-hit transfer cost bypass
- wildcard/free-hit retained free-transfer count
- AFCON top-up to 5 free transfers
- one chip per Gameweek
- two uses per chip family across the season
- first-half and second-half chip windows
- Free Hit cannot be played in consecutive Gameweeks
- Free Hit and Wildcard cannot be cancelled once confirmed/played
- Bench Boost and Triple Captain can be cancelled before deadline
- playing/minutes, goals, assists, clean sheet, saves, penalties, cards, own goal, goals conceded, defensive contribution, and bonus-point scoring
- Bonus Points System event weights for future explainability/projection work
- official deadline labels for the current published season ruleset

## Product-State Fields We Must Persist

To hold a user's picks for the entire season and beyond, each account needs saved season state, not just a temporary pitch view.

Minimum account state:

- `manager_id`
- `season`
- `ruleset_name`
- `current_gameweek`
- `strategy_mode`
- `bank_tenths`
- `free_transfers`
- `squad_value_tenths`
- `chips_used`
- `last_confirmed_deadline`
- `favourite_club`, optional
- `mini_league_mode`, optional

Minimum squad slot state:

- `player_id`
- `web_name`
- `club`
- `position`
- `purchase_price_tenths`
- `current_price_tenths`
- `selling_price_tenths`
- `squad_role`: starter / bench
- `bench_order`
- `captain_flag`
- `vice_captain_flag`
- `locked_flag`
- `ignored_flag`

Minimum planning state:

- saved transfer plans
- transfer basket
- saved drafts
- watchlist
- bench shortlist
- chip plans
- deadline alerts
- weekly recommendations archive
- final submitted snapshot per Gameweek

## UI/UX Guardrails

The Fantasy page should never allow a user to create an illegal plan without clearly explaining why.

Required UI states:

- invalid squad size
- position-count breach
- club-limit breach
- over-budget squad
- invalid starting formation
- missing captain or vice-captain
- captain and vice-captain are the same player
- captain or vice-captain is benched
- bench order incomplete
- transfer exceeds bank
- transfer requires a hit
- transfer limit warning at 20 transfers
- chip unavailable in this Gameweek
- Free Hit attempted in consecutive Gameweeks
- first-half chip deadline approaching
- saved free transfers retained after Wildcard or Free Hit
- AFCON transfer top-up applied

## Navigation Model

For users, the product should feel like a deadline command centre:

1. Import squad by FPL team ID.
2. Choose strategy mode.
3. See pitch view and legal starting XI.
4. See captain, vice, bench, transfer, and chip recommendations.
5. Search players.
6. Add players to watchlist, bench shortlist, compare, or transfer basket.
7. Save draft or transfer plan.
8. Return later and continue from the saved season state.

## Tier Implications

Founder:
- general weekly FPL briefing
- top captains
- top transfer targets
- trap watch
- player search
- watchlist

Premium:
- import my squad
- personalised pitch view
- personalised transfer recommendation
- captain/vice recommendation
- bench order
- hit/no-hit logic
- wildcard pressure
- chip planner
- saved drafts
- strategy modes

Pro:
- deeper player-event and injury intelligence
- deadline risk alerts
- richer player comparison
- market-state enrichment

Pro+:
- audit/explainability
- model feature references
- downloadable decision payloads

## Open Rule Checks Before Launch

FPL deadlines and special-season rules can change. Before the 2026/27 launch, re-run a rules verification pass against the live official FPL Help/Rules and Bootstrap API.

Specific re-checks:

- exact 2026/27 deadline table
- chip windows
- AFCON or tournament-related transfer top-ups
- defensive contribution thresholds
- BPS event weights
- squad budget
- max stored free transfers

The engine is now versioned so a new ruleset can be added without corrupting old season saves.
