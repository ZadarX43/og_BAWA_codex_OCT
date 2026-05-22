# Player Events Pre-Kickoff Readiness Checklist

## Purpose

Define the required pre-kickoff data checks before generating high-confidence player-event predictions.

This is a research/beta gate for player events only. It does not change the protected production deploy spine.

Protected files that must not be affected by this checklist:

- `bookie_allmarkets.py`
- `deploy_rulebook.py`
- `slip_formatter.py`

## Readiness Verdicts

Use one of four verdicts per fixture:

- `GREEN_CONFIRMED`: suitable for highest-confidence beta player-event board.
- `AMBER_PARTIAL`: usable for watchlist only; some key data is missing.
- `RED_NO_LINEUPS`: do not generate player-event picks; lineups are missing.
- `RED_NO_PLAYER_PROFILES`: do not generate player-event picks; player profile data is insufficient.

## Minimum Snapshot Contract

Every pre-kickoff player-event payload should carry:

```json
{
  "fixture_id": "",
  "fixture_key": "",
  "kickoff_at": "",
  "capture_generated_at": "",
  "source_data_cutoff_at": "",
  "snapshot_phase": "confirmed_lineups_pre_kickoff",
  "pre_kickoff_eligible": true,
  "lineups_confirmed": true,
  "referee_confirmed": true
}
```

## 1. Confirmed Lineups

Status: required.

The run is not high-confidence without confirmed lineups.

Required fields:

- `starting_xi`
- `bench`
- `formation`
- `player_id`
- `team_id`
- `player_position`
- `lineup_status`
- `expected_minutes`
- `recent_starts`
- `recent_minutes`
- `substitution_risk`

Checklist:

- [ ] Starting XI exists for both teams.
- [ ] Bench exists for both teams.
- [ ] Formation exists for both teams.
- [ ] Every candidate has `player_id`, `team_id`, and player name.
- [ ] Every candidate has starting/substitute status.
- [ ] Expected minutes are available or derived.
- [ ] Recent starts/minutes are available.
- [ ] Substitution risk is available or derived.

Failure rule:

- If starting XI is missing, verdict is `RED_NO_LINEUPS`.
- If player IDs are missing, verdict cannot exceed `AMBER_PARTIAL`.

## 2. Player Event Profiles

Status: required.

Required per starter:

- `shots_per90`
- `sot_per90`
- `tackles_per90`
- `fouls_committed_per90`
- `fouls_drawn_per90`
- `cards_per90`
- `saves_per90` for goalkeepers
- `recent_5_event_trend`
- `recent_10_event_trend`
- `home_away_split`
- `starter_only_split`
- `opponent_strength_adjusted_rate`

Checklist:

- [ ] Shots/SOT profile exists for attacking candidates.
- [ ] Tackles profile exists for defensive candidates.
- [ ] Fouls committed/fouls drawn profile exists for contact candidates.
- [ ] Cards profile exists for booking candidates.
- [ ] Saves profile exists for starting goalkeepers.
- [ ] Recent 5/10 trend exists or is explicitly marked unavailable.
- [ ] Home/away split exists or is explicitly marked unavailable.
- [ ] Starter-only split exists or is explicitly marked unavailable.
- [ ] Opponent-adjusted profile exists or is explicitly marked unavailable.

Failure rule:

- If the profile has no per-90 history for a player, that player is `WATCH_ONLY`.
- If more than 40% of starters lack profile data, verdict is `RED_NO_PLAYER_PROFILES`.

## 3. Tactical Role + Matchup

Status: required for fouls, bookings, tackles, and high-confidence shots/SOT.

Required fields:

- `role_archetype`
- `contact_role_archetype`
- `opponent_pressure_channel`
- `direct_opponent_matchup`
- `likely_duel_zone`
- `team_attack_side_bias`
- `opposition_dribbler_runner_profile`
- `formation_matchup_label`

Role archetypes:

- `pressing_forward`
- `defensive_mid`
- `midfield_presser`
- `fullback_wingback_contact`
- `centre_back_under_pressure`
- `creator`
- `wide_dribbler`
- `keeper_under_siege`

Pressure channels:

- `left_flank`
- `right_flank`
- `central_transition`
- `box_pressure`
- `wide_isolation`
- `set_piece_pressure`
- `keeper_siege`

Checklist:

- [ ] Every candidate has a role archetype.
- [ ] Contact candidates have `contact_role_archetype`.
- [ ] Pressure channel exists for the fixture or player.
- [ ] Formation-vs-formation interaction is available.
- [ ] Direct opponent or likely duel zone is identified where possible.
- [ ] Team attack-side bias is available for shots/SOT/key-pass candidates.
- [ ] Opposition dribbler/runner profile is available for foul/card/tackle candidates.

Failure rule:

- Missing role archetype means candidate cannot be `HIGH`.
- Missing pressure channel means fouls/bookings cannot be higher than `WATCH_ONLY`.

## 4. Fixture Environment

Status: required.

Required fields:

- `og_ftr_pick`
- `og_ftr_model_prob`
- `og_btts_pick`
- `og_btts_model_prob`
- `og_ou25_pick`
- `og_ou25_model_prob`
- `cs_goal_mass`
- `expected_goals_environment`
- `team_goal_pressure`
- `pace_chaos_rating`
- `possession_projection`
- `defensive_workload`
- `game_state_stakes`
- `weather_context`

Checklist:

- [ ] OG FTR output exists.
- [ ] OG BTTS output exists.
- [ ] OG OU25 output exists.
- [ ] Correct Score Goal Mass Matrix exists.
- [ ] Expected goals environment exists.
- [ ] Team goal pressure exists.
- [ ] Pace/chaos rating exists.
- [ ] Possession expectation exists.
- [ ] Defensive workload exists.
- [ ] Game-state stakes are available or explicitly marked neutral.
- [ ] Weather is present when materially disruptive.

Failure rule:

- Missing OG model outputs means fixture cannot be `GREEN_CONFIRMED`.
- Missing CS goal mass means shots/SOT/keeper saves cannot be `HIGH`.

## 5. Referee + Discipline Context

Status: required for fouls/bookings; optional but useful for tackles.

Required fields:

- `referee_name`
- `ref_cards_per_match`
- `ref_fouls_called_per_match`
- `ref_foul_to_card_ratio`
- `ref_penalty_tendency`
- `ref_red_card_tendency`
- `ref_home_away_card_tilt`
- `team_fouls_for_against`
- `team_cards_for_against`
- `player_booking_efficiency`
- `live_market_card_line`

Checklist:

- [ ] Referee is confirmed.
- [ ] Referee cards per match exists.
- [ ] Referee fouls called per match exists.
- [ ] Foul-to-card ratio exists.
- [ ] Red-card/penalty tendency exists if available.
- [ ] Home/away card tilt exists if sample is sufficient.
- [ ] Team fouls/cards for and against exist.
- [ ] Player booking efficiency exists.
- [ ] Live-market card line exists where bookmaker coverage allows.

Failure rule:

- Missing referee means bookings cannot be higher than `AMBER_PARTIAL`.
- Missing player booking efficiency means booking candidate is `WATCH_ONLY`.

## Market-Specific Readiness

### Shots / SOT

Required:

- confirmed starter
- role archetype
- shots/SOT per 90
- expected minutes
- team attack pressure
- CS goal mass
- team attack-side bias

Verdict rule:

- `HIGH` only if confirmed starter + role + CS goal mass + player shot profile align.

### Tackles

Required:

- confirmed starter
- defensive role
- tackles per 90
- expected minutes
- opponent possession/attack channel
- defensive workload

Verdict rule:

- `HIGH` only if opponent pressure channel matches the player's role.

### Fouls Committed

Required:

- confirmed starter
- contact role archetype
- fouls committed per 90
- expected minutes
- opponent pressure channel
- team foul pace profile
- referee foul tempo

Verdict rule:

- `HIGH` only if role + foul profile + opponent pressure channel + referee/team foul context align.

### Bookings

Required:

- fouls candidate support
- player card history
- player booking efficiency
- referee strictness
- match stakes
- live-market card line if available

Verdict rule:

- Bookings are a contact cascade, not a standalone prediction.
- A player should not be `HIGH` for bookings unless they are also live for fouls, tackles, or repeated defensive duels.

### Keeper Saves

Required:

- confirmed starting goalkeeper
- opponent SOT pressure
- opponent goal pressure
- team defensive concession profile
- CS goal mass
- expected shot volume

Verdict rule:

- `HIGH` only if opponent SOT pressure and defensive workload both align.

## Readiness Score

Score each fixture from 0 to 100:

- confirmed lineup layer: 30
- player event profile layer: 20
- tactical role/matchup layer: 20
- fixture environment layer: 20
- referee/discipline layer: 10

Verdict:

- `GREEN_CONFIRMED`: 85-100
- `AMBER_PARTIAL`: 65-84
- `RED_NO_LINEUPS`: lineups missing, regardless of score
- `RED_NO_PLAYER_PROFILES`: player profiles missing, regardless of score

## Required Output Columns

Every generated player-event row should carry:

- `fixture_id`
- `fixture_key`
- `player_id`
- `team_id`
- `capture_generated_at`
- `source_data_cutoff_at`
- `fixture_kickoff_at`
- `snapshot_phase`
- `pre_kickoff_eligible`
- `readiness_verdict`
- `readiness_score`
- `missing_required_fields`
- `role_archetype`
- `contact_role_archetype`
- `pressure_channel`
- `market_readiness_note`

## Policy

- Do not publish player-event picks from `RED_*` fixtures.
- Do not treat `AMBER_PARTIAL` as a deploy-grade board.
- Keep fouls and bookings research-only until role/channel scoring is stable across multiple weekends.
- Persist IDs and timestamp fields before any player-event hit-rate claims are made publicly.
