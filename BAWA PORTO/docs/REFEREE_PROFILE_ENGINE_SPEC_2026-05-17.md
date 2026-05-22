# Referee Profile Engine Spec

Date: 2026-05-17
Status: beta / research support layer

## Purpose

Build a league-by-league referee profile system so player-event, bookings, fouls, cards, and open-play forecasts can condition on the assigned referee.

This is not part of the production deploy spine yet. It should feed the player-event beta stack and fixture intelligence overlays first.

## Existing Repo Hooks

Current partial implementations:

- `scripts/api_football/build_referee_profile_features.py`
- `scripts/player_events/build_referee_profiles.py`
- `scripts/player_events/build_player_cards_hazard_audit.py`
- `reports/2026-05-08/tactical_feature_registry/TACTICAL_FEATURE_REGISTRY.csv`

Existing feature family:

- `CARD_FOUL_ECOSYSTEM`

Existing fields already present in some outputs:

- `referee_name`
- `ref_cards_per_match`
- `ref_bookings_per_match`
- `ref_fouls_per_match`
- `ref_cards_per_foul`
- `ref_foul_to_card_ratio`
- `ref_late_cards_per_match`
- `ref_penalty_tendency`
- `ref_red_card_tendency`
- `ref_home_bias`
- `ref_strictness_score`
- `ref_leniency_band`
- `booking_pressure_with_ref`

## Upgrade Objective

Move from a simple strictness profile to a referee style profile:

```text
referee strictness
+ foul calling tempo
+ card conversion rate
+ late-game card tendency
+ penalty / VAR tendency
+ home-away tilt
+ open-play allowance
+ team interaction history
+ player-role interaction
```

## Required Output Per Referee

Recommended canonical output:

`data_sources/api_football/features/referee_profiles/referee_profiles__<LEAGUE_TAG>__<SEASON>.csv`

Core columns:

- `league`
- `league_tag`
- `season`
- `referee_name`
- `sample_matches`
- `sample_freshness_days`
- `cards_per_match_l20`
- `yellows_per_match_l20`
- `reds_per_match_l20`
- `fouls_per_match_l20`
- `foul_to_card_ratio_l20`
- `cards_per_foul_l20`
- `late_cards_per_match_l20`
- `first_half_cards_per_match_l20`
- `second_half_cards_per_match_l20`
- `penalties_per_match_l20`
- `var_penalty_events_l20`
- `home_cards_per_match_l20`
- `away_cards_per_match_l20`
- `home_card_tilt`
- `away_card_tilt`
- `open_play_foul_tolerance_score`
- `tactical_foul_punishment_score`
- `dissent_strictness_score`
- `timewasting_strictness_score`
- `strictness_score`
- `strictness_band`
- `profile_confidence`

## Team Interaction Output

Recommended canonical output:

`data_sources/api_football/features/referee_profiles/referee_team_interactions__<LEAGUE_TAG>__<SEASON>.csv`

Columns:

- `referee_name`
- `team_name`
- `matches_with_team`
- `team_cards_for_per_match`
- `team_cards_against_per_match`
- `team_fouls_for_per_match`
- `team_fouls_drawn_per_match`
- `team_penalties_for_per_match`
- `team_penalties_against_per_match`
- `ref_team_card_tilt`
- `ref_team_foul_tilt`
- `interaction_confidence`

This should answer: does this referee consistently punish this team, or does this team under/over-index under this referee?

## Fixture Overlay Output

Recommended fixture-level output:

`reports/latest/referee_overlay/referee_fixture_overlay_<window>.csv`

Columns:

- `fixture_key`
- `match_date`
- `league`
- `home_team_name`
- `away_team_name`
- `referee_name`
- `ref_profile_confidence`
- `ref_strictness_band`
- `expected_total_cards_ref_adjusted`
- `expected_total_fouls_ref_adjusted`
- `expected_first_half_cards_ref_adjusted`
- `expected_second_half_cards_ref_adjusted`
- `home_card_tilt`
- `away_card_tilt`
- `late_card_risk_flag`
- `penalty_risk_flag`
- `open_play_allowed_flag`
- `card_market_live_flag`
- `fouls_market_live_flag`
- `bookings_player_event_multiplier`

## Player Event Integration

For player bookings and fouls, the referee layer must not pick players by itself.

It should multiply or suppress the existing player-event cascade:

```text
player contact role
+ pressure channel
+ support context
+ game-state regime
+ referee conversion profile
- suppressors
```

High-confidence booking candidates need:

- player foul/contact support
- repeated duel/channel exposure
- referee card conversion support
- game-state heat
- no major suppressor

Referee-specific suppressors:

- high foul tolerance but low card conversion
- low first-half card rate
- low late-game card rate
- strong home/away tilt against the target side
- low sample confidence

## Live Match Triggers

The referee profile should adjust live reads:

- If ref has high foul tempo but low conversion: wait for repeated fouls before card entry.
- If ref has high card conversion: first foul plus repeated duel can be enough.
- If ref has high late-card tendency: protect-lead and chase-state cards get boosted after 65 minutes.
- If ref has penalty tendency: centre-backs, keepers, and box-clearance pressure gain caution risk.

## Build Plan

1. Standardize existing scripts into one canonical builder.
2. Add league coverage reporting: which covered leagues have referee samples, missing refs, and low-confidence refs.
3. Add team interaction joins.
4. Add first-half vs second-half card split.
5. Add open-play allowance proxy: fouls called per duel/contact-heavy fixture, where data allows.
6. Feed fixture overlay into player-event beta reports.
7. Backtest player-card candidate scoring with and without referee overlay.
8. Keep out of production deploy until audited.

## Lessons From EPL 2026-05-17 Live Test

The 3pm EPL player-event beta run showed that first-half state and repeated-foul nodes matter more than static reputation.

The referee layer should have helped answer:

- Which refs convert early repeated fouls to cards?
- Which refs allow midfield contact without cards?
- Which refs produce late protection/chasing cards?
- Which refs punish tactical fouls more than body-duel contact?

This should improve booking clusters, fouls committed, and live second-half card decisions.

