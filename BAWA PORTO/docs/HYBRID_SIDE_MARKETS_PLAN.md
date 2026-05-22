# HYBRID_SIDE_MARKETS_PLAN

## Purpose
Define the first side-market training sequence for the hybrid stack.

Priority side markets:
1. home goals over 1.5
2. away goals over 1.5
3. home fail to score
4. away fail to score
5. BTTS first half

## Why these markets first
These markets should benefit earliest from API context because they are more sensitive to:
- lineup strength
- player attack power
- missing scorers / creators
- shot pressure
- defensive absence severity
- goal timing style

## Core feature families
### Team rolling
- `home_goals_for_l5`
- `away_goals_for_l5`
- `home_goals_against_l5`
- `away_goals_against_l5`
- `home_scored_rate_l5`
- `away_scored_rate_l5`
- `home_conceded_rate_l5`
- `away_conceded_rate_l5`
- `combined_over25_rate_l5`
- `combined_btts_rate_l5`

### Shot / pressure
- `home_shots_l5`
- `away_shots_l5`
- `home_sot_l5`
- `away_sot_l5`
- `home_shots_inside_box_l5`
- `away_shots_inside_box_l5`
- `home_shot_accuracy_l5`
- `away_shot_accuracy_l5`
- `home_attack_pressure_index`
- `away_attack_pressure_index`
- `match_pressure_delta`

### XI / lineup
- `home_starting_xi_avg_rating_l5`
- `away_starting_xi_avg_rating_l5`
- `home_starting_xi_goals_per90_l5`
- `away_starting_xi_goals_per90_l5`
- `home_starting_xi_shots_per90_l5`
- `away_starting_xi_shots_per90_l5`
- `home_starting_xi_sot_per90_l5`
- `away_starting_xi_sot_per90_l5`
- `xi_goal_power_delta`
- `xi_shot_power_delta`
- `formation_mismatch_flag`

### Injury / absence
- `home_absence_severity_score`
- `away_absence_severity_score`
- `absence_severity_delta`
- `home_missing_goals_per90_l5`
- `away_missing_goals_per90_l5`
- `home_missing_assists_per90_l5`
- `away_missing_assists_per90_l5`
- `home_missing_tackles_per90_l5`
- `away_missing_tackles_per90_l5`

### Event timing / volatility
- `home_first_goal_rate_l10`
- `away_first_goal_rate_l10`
- `home_goal_after_75_rate_l10`
- `away_goal_after_75_rate_l10`
- `home_concede_after_75_rate_l10`
- `away_concede_after_75_rate_l10`
- `combined_chaos_index_l10`
- `combined_late_volatility_l10`

### Market priors
- `bookie_home_prob_norm`
- `bookie_away_prob_norm`
- `bookie_over25_prob_norm`
- `bookie_btts_yes_prob_norm`

## Target mapping
### Home goals over 1.5
Target:
- `target_home_goals_over15`

### Away goals over 1.5
Target:
- `target_away_goals_over15`

### Home fail to score
Target:
- `target_home_fts`

### Away fail to score
Target:
- `target_away_fts`

### BTTS first half
Target:
- `target_btts_first_half`

## Suggested training order
1. home goals over 1.5
2. away goals over 1.5
3. home FTS
4. away FTS
5. BTTS first half

## Evaluation priorities
For each market:
- AUC
- log loss
- Brier score
- accuracy at calibrated threshold
- top-decile precision
- deployable volume under protected filters

## Expected wins
### Home / away team goals O1.5
Most likely to benefit from:
- XI goal power
- absence severity
- shot pressure
- market expectation

### FTS
Most likely to benefit from:
- scored rate
- opposing defensive structure
- missing attackers / creators
- goalkeeper / defensive absences on the other side

### BTTS first half
Most likely to benefit from:
- first-goal rates
- HT over rates
- attacking XI quality
- early-goal event style

## Promotion rule
No side-market hybrid model is promoted unless it improves:
- ranking quality
- calibration
- deployable volume
- or false-positive suppression

without breaking current floors.
