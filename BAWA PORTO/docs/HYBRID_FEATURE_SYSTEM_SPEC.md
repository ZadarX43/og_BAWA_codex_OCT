# Hybrid Feature System Spec
Date: 2026-04-29
Repo: /Users/hughwade/Documents/Code/OG_master/BAWA PORTO
Status: Design locked for implementation planning

## 1. Purpose
This spec defines the next major upgrade to the hybrid football modeling system.

The goal is not just to add more API-Football features. The goal is to build a structured feature system that captures:
- team identity
- player-driven team strength
- tactical matchup interactions
- historical interaction regimes
- discipline and restraint
- referee influence

This is the layer that turns the current hybrid stack from a broad contextual model into a richer football intelligence system.

## 2. Design Principles
All new feature families must obey the following rules:
- pre-match only
- rolling or shifted only
- no current-match outcome leakage
- league-aware where appropriate
- lineup-aware where possible
- absence-aware where possible
- decomposable into explainable sub-components

Feature hierarchy:
- Tier 1 = team identity foundation
- Tier 2 = context and history
- Tier 3 = matchup interactions
- Tier 4 = referee profiles

Implementation rule:
- build stable atomic features first
- then build composites
- then build interactions
- then wire them into model subsets by market

## 3. Feature Family Map
## 3.1 Tier 1 — Team Identity
These are the foundation features. Everything downstream depends on these being clean.

Families:
- attack strength composite
- defensive strength composite
- midfield control composite
- wing strength composite
- conversion quality layer
- defensive restraint layer

### Attack strength composite
Purpose:
- summarize likely XI attacking output pre-match

Candidate raw inputs:
- `home_xi_goals_per90`
- `away_xi_goals_per90`
- `home_xi_shots_per90`
- `away_xi_shots_per90`
- `home_xi_sot_per90`
- `away_xi_sot_per90`
- `home_xi_key_passes_per90`
- `away_xi_key_passes_per90`
- `home_scored_rate_l5`
- `away_scored_rate_l5`
- `home_attack_pressure_index`
- `away_attack_pressure_index`
- `home_absence_attack_penalty`
- `away_absence_attack_penalty`
- `conversion_quality_home`
- `conversion_quality_away`

Created composites:
- `home_attack_strength`
- `away_attack_strength`
- `attack_strength_delta`

### Defensive strength composite
Purpose:
- summarize likely XI defensive suppression and ball-winning ability

Candidate raw inputs:
- `home_xi_tackles_per90`
- `away_xi_tackles_per90`
- `home_xi_interceptions_per90`
- `away_xi_interceptions_per90`
- `home_conceded_rate_l5`
- `away_conceded_rate_l5`
- `home_goals_against_l5`
- `away_goals_against_l5`
- `goals_against_suppression_home`
- `goals_against_suppression_away`
- `home_absence_defensive_penalty`
- `away_absence_defensive_penalty`
- `defensive_foul_rate_home`
- `defensive_foul_rate_away`
- `red_card_volatility_home`
- `red_card_volatility_away`

Created composites:
- `home_defensive_strength`
- `away_defensive_strength`
- `defensive_strength_delta`

### Midfield control composite
Purpose:
- estimate control, progression, and central stability

Candidate raw inputs:
- `home_xi_pass_accuracy`
- `away_xi_pass_accuracy`
- `home_xi_key_passes_per90`
- `away_xi_key_passes_per90`
- `home_possession_l5`
- `away_possession_l5`
- `home_passes_l5`
- `away_passes_l5`
- `home_pass_accuracy_l5`
- `away_pass_accuracy_l5`
- `ball_winning_rate_home`
- `ball_winning_rate_away`

Created composites:
- `home_midfield_control`
- `away_midfield_control`
- `midfield_control_delta`

### Wing strength composite
Purpose:
- capture flank threat and flank resistance

Candidate raw inputs:
- winger and fullback player rolling stats where position data permits
- `player_dribble_success_rate_l5`
- `player_key_passes_per90_l5` when available
- `player_cross_proxy_l5` if derivable later
- `player_duel_win_rate_l5`
- side-biased lineup or formation indicators
- `formation_attack_score`
- `formation_defence_score`

Created composites:
- `home_wing_strength`
- `away_wing_strength`
- `wing_strength_delta`
- later if feasible:
  - `home_left_wing_strength`
  - `home_right_wing_strength`
  - `away_left_wing_strength`
  - `away_right_wing_strength`

### Conversion quality layer
Purpose:
- estimate finishing quality and shot conversion efficiency

Candidate raw inputs:
- `goals_scored_l10`
- `shots_l10`
- `sot_l10`
- `home_xi_sot_per90`
- `away_xi_sot_per90`
- `home_xi_goals_per90`
- `away_xi_goals_per90`
- `home_scored_rate_l5`
- `away_scored_rate_l5`

Created features:
- `conversion_quality_home`
- `conversion_quality_away`
- `sot_conversion_home`
- `sot_conversion_away`
- `conversion_delta`

### Defensive restraint layer
Purpose:
- model disciplined defending versus reckless defending

Candidate raw inputs:
- `home_xi_fouls_committed_per90`
- `away_xi_fouls_committed_per90`
- `home_xi_cards_per90`
- `away_xi_cards_per90`
- `home_red_card_rate_l20`
- `away_red_card_rate_l20`
- team foul and card rolling features

Created features:
- `defensive_foul_rate_home`
- `defensive_foul_rate_away`
- `card_per_foul_home`
- `card_per_foul_away`
- `red_card_volatility_home`
- `red_card_volatility_away`
- `defensive_restraint_home`
- `defensive_restraint_away`
- `defensive_restraint_delta`

## 3.2 Tier 2 — Context and History
These families provide historical and regime context on top of the team identity layer.

Families:
- streak features, modernized and lineup-adjusted
- power ratings, absence-adjusted
- H2H regime features
- press versus press interactions

### Modernized streak features
Purpose:
- capture current regime and momentum without using naive raw streaks

Candidate families:
- scoring streaks
- conceding streaks
- over 2.5 streaks
- BTTS streaks
- first-to-score streaks
- first-to-concede streaks
- clean-sheet droughts
- lineup-adjusted versions weighted by current XI continuity

Created features:
- `home_scoring_streak_adj`
- `away_scoring_streak_adj`
- `home_conceding_streak_adj`
- `away_conceding_streak_adj`
- `home_over25_streak_adj`
- `away_over25_streak_adj`
- `home_btts_streak_adj`
- `away_btts_streak_adj`
- `xi_continuity_score_home`
- `xi_continuity_score_away`

### Absence-adjusted power ratings
Purpose:
- replace static team strength with strength adjusted for who is actually likely to play

Candidate raw inputs:
- existing power features
- team rolling form
- lineup quality
- absence severity
- schedule and rest context

Created features:
- `home_power_rating_hybrid`
- `away_power_rating_hybrid`
- `power_rating_delta_hybrid`
- `home_power_rating_xi_adj`
- `away_power_rating_xi_adj`

### H2H regime features
Purpose:
- model tactical interaction history rather than raw win-draw-loss counts

Candidate raw inputs:
- `h2h_avg_goals_last5`
- `h2h_btts_rate_last5`
- `h2h_avg_cards_last5`
- `h2h_avg_fouls_last5`
- `h2h_avg_possession_diff_last5`
- previous tactical interaction proxies

Created features:
- `h2h_goal_environment`
- `h2h_btts_regime`
- `h2h_booking_heat`
- `h2h_foul_intensity`
- `h2h_style_conflict`
- `h2h_referee_overlap`

### Press versus press interactions
Purpose:
- characterize tempo conflict, buildup stress, and chaos potential

Candidate raw inputs:
- existing press intensity features
- pressure indices
- defensive ball-winning rates
- pass accuracy and possession retention proxies

Created features:
- `home_press_vs_away_buildup`
- `away_press_vs_home_buildup`
- `pressed_vs_pressed`
- `press_mismatch_index`
- `buildup_resistance_home`
- `buildup_resistance_away`

## 3.3 Tier 3 — Matchup Interactions
This is where the system starts becoming materially stronger than a flat feature pile.

Families:
- home attack versus away defensive restraint
- away press versus home buildup resistance
- both teams chaos interaction
- both teams booking risk
- style conflict index

### Core interaction features
Created features:
- `home_attack_vs_away_defence_gap`
- `away_attack_vs_home_defence_gap`
- `home_attack_vs_away_restraint_gap`
- `away_attack_vs_home_restraint_gap`
- `home_press_vs_away_buildup_gap`
- `away_press_vs_home_buildup_gap`
- `both_teams_chaos_interaction`
- `both_teams_booking_risk`
- `style_conflict_index`
- `midfield_control_conflict_index`
- `wing_mismatch_index`

Construction logic:
- use deltas for additive relationships
- use products for compounding volatility relationships
- use absolute gaps for clash-style features

Examples:
- `both_teams_chaos_interaction = home_chaos_index * away_chaos_index`
- `style_conflict_index = abs(home_possession_l5 - away_possession_l5)`
- `home_attack_vs_away_defence_gap = home_attack_strength - away_defensive_strength`

## 3.4 Tier 4 — Referee Profile
Referee features are a distinct layer and should be treated as their own family.

Families:
- referee bookings tendency
- referee fouls and card conversion
- referee red-card tendency
- referee penalty tendency
- referee strictness and leniency bands
- referee matchup flags

### Referee profile features
Created features:
- `ref_bookings_per_match`
- `ref_fouls_per_match`
- `ref_cards_per_foul`
- `ref_red_card_tendency`
- `ref_penalty_tendency`
- `ref_strictness_score`
- `ref_home_bias`
- `ref_leniency_band`

### Referee matchup features
Created features:
- `aggressive_team_strict_ref_flag`
- `away_aggressive_team_strict_ref_flag`
- `combined_aggression_strict_ref_flag`
- `booking_pressure_with_ref`
- `foul_suppression_with_ref`

## 4. Build Sequence
## 4.1 Step 1 — Team identity composites
Build first:
- attack strength composites
- defensive strength composites
- midfield control composites
- wing strength composites

Reason:
- every later matchup, H2H, and referee interaction needs these clean identity vectors first

## 4.2 Step 2 — Conversion and defensive restraint
Build second:
- conversion quality layer
- SOT conversion features
- foul discipline features
- card per foul features
- red-card volatility

Reason:
- these sharpen both goal models and card models
- they also improve the realism of attack and defence composites

## 4.3 Step 3 — Matchup interactions
Build third:
- attack versus defence gaps
- press mismatch features
- chaos interactions
- booking risk interactions
- style conflict indices

Reason:
- interactions are only meaningful once component identity features exist

## 4.4 Step 4 — H2H regime features
Build fourth:
- tactical H2H regime families
- booking and foul heat in repeated fixtures
- H2H style conflict

Reason:
- useful in repeat fixtures and cup competitions
- should be layered carefully to avoid overfitting small samples

## 4.5 Step 5 — Referee profiles
Build fifth:
- rolling referee profiles
- strictness scores
- matchup flags against aggressive teams

Reason:
- unlocks cards, fouls, and player discipline markets
- may also add context to game tempo and under-style suppression

## 5. Market Mapping
## 5.1 Core markets
### FTR 1X2
Primary helpful families:
- power ratings, absence-adjusted
- midfield control composites
- attack and defence deltas
- lineup-adjusted streaks
- style conflict in balanced matches

Secondary helpful families:
- H2H tactical regime
- press mismatch

### BTTS Yes/No
Primary helpful families:
- attack strength composites
- defensive strength composites
- conversion quality
- matchup interaction gaps
- chaos interaction
- H2H goal environment

Secondary helpful families:
- defensive restraint
- press versus press

### OU25 Over/Under
Primary helpful families:
- attack strength composites
- conversion quality
- defensive strength
- both teams chaos interaction
- style conflict index
- H2H goal environment

Secondary helpful families:
- referee strictness as tempo suppression or disciplinary disruption context
- press mismatch

## 5.2 Side markets
### Home goals over 1.5
Primary helpful families:
- home attack strength
- away defensive strength
- home conversion quality
- away defensive restraint
- home_attack_vs_away_defence_gap

### Away goals over 1.5
Primary helpful families:
- away attack strength
- home defensive strength
- away conversion quality
- home defensive restraint
- away_attack_vs_home_defence_gap

### Fail to score
Primary helpful families:
- attack strength deficits
- defensive suppression
- conversion weakness
- press mismatch under pressure

### BTTS first half
Primary helpful families:
- first-goal timing
- early pressure
- attack strength
- chaos interaction
- H2H goal environment

## 5.3 Player and discipline markets
### Player shots and shots on target
Primary helpful families:
- attack strength composites
- wing strength
- midfield control
- press mismatch
- likely XI role concentration

### Fouls
Primary helpful families:
- defensive restraint
- both teams booking risk
- press conflict
- referee foul profile

### Bookings
Primary helpful families:
- defensive restraint
- card per foul
- red-card volatility
- referee strictness
- aggressive team plus strict ref flags

## 5.4 Live and in-play extensions later
Potential future beneficiaries:
- live OU25 state
- live BTTS state
- bookings remaining probability
- goal timing probability

These are downstream opportunities once the pre-match structural layer is stable.

## 6. Script Ownership Plan
## 6.1 Existing scripts to extend
### `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/scripts/api_football/build_team_rolling_features.py`
Should own:
- conversion quality atomic inputs
- defensive restraint atomic inputs
- streak modernization inputs
- build-up resistance inputs
- additional rolling scored and conceded context

### `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/scripts/api_football/build_player_rolling_features.py`
Should own:
- player attack, defence, midfield, and wing atomic signals
- player role grouped aggregates
- XI-ready player strength inputs

### `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/scripts/api_football/build_lineup_features.py`
Should own:
- XI attack composite assembly
- XI defensive composite assembly
- midfield control assembly
- wing strength assembly
- XI continuity features

### `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/scripts/api_football/build_injury_features.py`
Should own:
- attack and defence absence penalties
- absence severity refinements
- role-weighted absence penalties

### `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/scripts/api_football/build_event_features.py`
Should own:
- chaos features
- timing regime features
- red-card volatility
- H2H tactical interaction event-derived features if event-based history is used

### `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/scripts/api_football/build_enriched_fixture_features.py`
Should own:
- cross-family merging
- matchup interaction arithmetic
- final composite publication into the enriched feature layer

## 6.2 New scripts to create
### `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/scripts/api_football/build_team_identity_features.py`
Should own:
- attack strength composites
- defensive strength composites
- midfield control composites
- wing strength composites
- conversion quality composites
- defensive restraint composites

Reason:
- this deserves a dedicated script rather than overloading a single existing builder

### `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/scripts/api_football/build_matchup_interaction_features.py`
Should own:
- attack versus defence gap features
- press mismatch features
- chaos interaction features
- booking risk interaction features
- style conflict indices

### `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/scripts/api_football/build_h2h_regime_features.py`
Should own:
- H2H goal environment
- H2H BTTS regime
- H2H booking heat
- H2H foul intensity
- H2H style conflict
- referee overlap in H2H when available

### `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/scripts/api_football/build_referee_profile_features.py`
Should own:
- rolling referee profiles
- strictness scoring
- leniency bands
- referee matchup flags

## 6.3 Training scripts to update later
After feature builders exist, these training scripts should be updated to include the new families in controlled views:
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/train_hybrid_catboost.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/train_hybrid_xgb.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/train_hybrid_goal_mass.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/train_hybrid_side_markets.py`
- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/train_hybrid_ou25_tuned.py`

## 7. Feature View Strategy
The new families should not be dumped into one uncontrolled hybrid view.

Recommended controlled views:
- `baseline`
- `baseline_plus_identity`
- `baseline_plus_identity_plus_context`
- `baseline_plus_identity_plus_context_plus_matchups`
- `baseline_plus_identity_plus_context_plus_matchups_plus_ref`
- `api_identity_only`
- `api_interactions_only`

This preserves interpretability and makes ablation meaningful.

## 8. Calibration And Normalization Rules
These features should be normalized carefully.

Normalization guidance:
- per-league normalization for conversion and discipline metrics
- rolling-window stability checks for small samples
- clipped ratios where denominators are sparse
- explicit missingness flags for absent referee or H2H information
- avoid naive use of raw small-sample H2H counts

Key examples:
- `conversion_quality_home` should be normalized against league average conversion
- `ref_strictness_score` should be league-aware because league foul and card baselines vary materially
- `power_rating_hybrid` should remain comparable inside a competition window

## 9. Risks And Safeguards
Primary risks:
- feature explosion without clear ownership
- unstable small-sample H2H signals
- hidden leakage through improperly shifted H2H or referee data
- overfitting via too many interaction features
- role or position ambiguity in player-derived wing and midfield modeling

Safeguards:
- strict pre-match and shifted-only logic
- ablation-first rollout
- league-aware validation
- minimum sample thresholds for H2H and referee-derived profiles
- keep feature families modular and auditable

## 10. Recommended Implementation Order
Build in this exact order:
1. `build_team_identity_features.py`
2. extend `build_team_rolling_features.py` for conversion and restraint atomic inputs
3. extend `build_player_rolling_features.py` and `build_lineup_features.py` for role-aware identity inputs
4. `build_matchup_interaction_features.py`
5. `build_h2h_regime_features.py`
6. `build_referee_profile_features.py`
7. wire all of the above into `build_enriched_fixture_features.py`
8. add controlled feature views into the training scripts
9. rerun OU25, BTTS, team-goals, and FTS research grids
10. only then move into cards, fouls, and player markets

## 11. First Market Testing Order After Build
Recommended test order:
1. OU25
2. BTTS
3. home goals over 1.5
4. away goals over 1.5
5. FTS
6. bookings
7. fouls
8. player shots

Reason:
- OU25 and BTTS will reveal whether team identity and interaction features are truly adding football signal
- team-goal and FTS markets are already promising in the current hybrid stack
- bookings and fouls should wait until referee and restraint layers are live
- player markets should come after the broader tactical context is stable

## 12. Model Inventory This Feature System Enables
Core markets:
- FTR 1X2
- BTTS Yes/No
- OU25 Over/Under

Side markets:
- home goals over 1.5
- away goals over 1.5
- fail to score
- BTTS first half

Discipline and event markets:
- team bookings
- player bookings
- team fouls
- player fouls

Player performance markets later:
- player shots
- player shots on target
- player tackles
- player foul counts

Live extensions later:
- live OU25 state
- live BTTS state
- bookings remaining probability
- next-goal timing probability

## 13. Strategic Summary
This feature program is the next real moat-building stage.

The first hybrid phase proved:
- we can fetch the data
- we can normalize the data
- we can build safe pre-match tables
- we can train and scale hybrid models

The next phase should prove:
- we can encode real team identity
- we can encode tactical interactions
- we can encode discipline and officiating context
- we can route the right feature systems into the right markets

That is the point where the hybrid stack stops being merely broader and starts becoming meaningfully smarter.
