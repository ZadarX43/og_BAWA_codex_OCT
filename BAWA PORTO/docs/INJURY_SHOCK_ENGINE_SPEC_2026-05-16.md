# OG Injury Shock And Lineup Risk Engine

## Why This Exists

The Bayer Leverkusen vs Hamburger SV miss exposed a real structural blind spot:

- the historical model still saw Leverkusen through season-strength, xG, power, and goal-mass history
- the current match state had attacking absences, squad uncertainty, and end-of-cycle mood risk
- that difference between learned team strength and today's available team is exactly where deploy slips can get hurt

This layer is designed to detect:

```text
How different is today's team from the team the model learned from?
```

It is a pre-deploy warning layer first. It must not override `deploy_rulebook.py` until it has walk-forward evidence.

## Current Implementation

Module:

```text
injury_shock_engine.py
```

Inputs:

- normalized fixtures CSV
- normalized API-Football injuries CSV
- normalized match player stats CSV
- optional fixture context CSV for motivation / soft-volatility flags

Outputs:

- fixture-level injury shock CSV
- optional markdown warning board

Validated smoke command:

```bash
python3 injury_shock_engine.py \
  --fixtures-csv data_sources/api_football/normalized/fixtures_master__Germany_Bundesliga__2024.csv \
  --injuries-csv data_sources/api_football/normalized/injuries__Germany_Bundesliga__2024.csv \
  --player-stats-csv data_sources/api_football/normalized/match_player_stats__Germany_Bundesliga__2024.csv \
  --output-csv /private/tmp/og_injury_shock_bundesliga_2024.csv \
  --output-md /private/tmp/og_injury_shock_bundesliga_2024.md
```

Smoke result:

- rows: `308`
- deploy warning fixtures: `84`

## Output Columns

Core fixture identifiers:

- `fixture_id`
- `fixture_key`
- `league`
- `season`
- `match_date`
- `home_team_name`
- `away_team_name`

Team shock scores:

- `home_attack_absence_score`
- `away_attack_absence_score`
- `home_midfield_absence_score`
- `away_midfield_absence_score`
- `home_defence_absence_score`
- `away_defence_absence_score`
- `home_keeper_absence_score`
- `away_keeper_absence_score`
- `home_mobility_risk_score`
- `away_mobility_risk_score`
- `home_lineup_confidence_score`
- `away_lineup_confidence_score`
- `home_injury_news_severity`
- `away_injury_news_severity`

Market impact helpers:

- `goal_model_adjustment`
- `btts_adjustment`
- `ou25_adjustment`
- `ftr_volatility_adjustment`
- `deploy_warning_flag`
- `absence_edge_side`
- `warning_tokens`
- `home_absence_reasons`
- `away_absence_reasons`
- `context_flags`

## Warning Tokens

Initial token family:

- `HOME_ATTACK_SHOCK`
- `AWAY_ATTACK_SHOCK`
- `HOME_DEFENCE_SPINE_SHOCK`
- `AWAY_DEFENCE_SPINE_SHOCK`
- `MOTIVATION_VOLATILITY`
- `REQUIRE_LINEUP_CONFIRMATION`

These should eventually feed:

- Team Intelligence caution layers
- Fixture Intelligence caution layers
- Player Event readiness
- pre-slip manual review
- public-safe fixture decision copy

They should not become hard deploy vetoes until proved.

## Market Interpretation

Attack shock:

- downgrade `OU25 OVER25`
- downgrade team goals 1.5
- downgrade `BTTS YES` when one side is specifically weakened
- increase FTR volatility when the favourite is affected

Defensive spine shock:

- boost BTTS/Over danger if only the defence is weakened
- increase FTR volatility if the favourite is missing CB/keeper/DM spine
- downgrade clean-sheet confidence

Mobility / late-test risk:

- reduce lineup confidence
- require confirmed lineup before deploy
- downgrade player events that rely on minutes, pace, wide threat, or repeated high-intensity actions

Motivation volatility:

- downgrade dominant favourite certainty
- increase FTR variance
- treat handicap and clean-sheet signals with caution
- monitor end-of-season, title-won, cup-final-ahead, manager-exit, relegation-safe, farewell, and rotation contexts

## Context CSV Contract

Optional `--context-csv` columns:

```text
fixture_key,title_won_flag,cup_final_ahead_flag,manager_exit_noise_flag,europe_secured_flag,relegation_safe_flag,must_win_flag,rotation_risk_flag,farewell_match_flag,context_note
```

Use this for soft states that API injuries cannot know:

- title already won
- Europe already secured
- relegation safety
- cup final ahead
- manager exit noise
- star transfer noise
- farewell match
- last home game
- must-win pressure
- known rotation pressure

## Integration Position

Target future order:

```text
model prediction
-> injury/news shock engine
-> fixture/team/player intelligence reconciliation
-> deploy_rulebook.py
-> slip_formatter.py
-> website publish
```

Current safe order:

```text
model prediction
-> deploy_rulebook.py
-> injury/news shock engine sidecar
-> manual deploy review / report
-> future backtest before gate promotion
```

This is intentional. We do not mutate protected deploy routing without evidence.

## Evidence Plan

Backtest questions:

1. When `ATTACK_SHOCK` appears against `OU25 OVER25`, does hit rate fall?
2. When `DEFENCE_SPINE_SHOCK` appears, does BTTS/Over improve or does FTR volatility dominate?
3. Does `REQUIRE_LINEUP_CONFIRMATION` save more slips than it blocks?
4. Are warnings more valuable for favourites than underdogs?
5. Does the warning layer improve ELITE/STANDARD deployed picks, or only OBSERVE avoidance?

Promotion path:

1. Build warning board for historical API-covered leagues.
2. Join to settled deploy outputs.
3. Compare hit rate by market and token.
4. Compare base deploy vs deploy excluding shock fixtures.
5. Only then propose `deploy_rulebook.py` caution/veto integration.

## Known Limitations

- API-Football injury rows can lack reliable position/function detail, so function labels are inferred from recent player stats.
- Some players can be misclassified when the provider reports broad positions only.
- Soft news/motivation context requires a separate context feed or manual CSV until external content ingestion is productized.
- Current scores are warning features, not calibrated probabilities.
- This remains research / pre-deploy intelligence until walk-forward proof exists.

## Next Build Queue

1. Add fixture/player IDs and timestamp discipline to all injury shock outputs.
2. Create a weekend `injury_shock_context_flags.csv` from news/previews.
3. Add injury shock join into `fixture_decision_reconciler.py` as public-safe caution tokens.
4. Build `scripts/score_injury_shock_vs_deploy_results.py`.
5. Backtest token-level market impact for `FTR`, `BTTS`, `OU25`, and team goals 1.5.
6. Consider hard rule candidates only after proof, for example:

```text
No ELITE upgrade when favourite has ATTACK_SHOCK and lineup is unconfirmed.
No OU25 elite deploy when key striker/creator missing and team-goals support weakens.
Require confirmed lineup when two or more expected attacking starters are absent/doubtful.
```
