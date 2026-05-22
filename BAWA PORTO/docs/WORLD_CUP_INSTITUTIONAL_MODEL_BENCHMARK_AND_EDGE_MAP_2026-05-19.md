# World Cup Institutional Model Benchmark and Edge Map

Date: 2026-05-19
Scope: research-only World Cup feature planning for Odds Genius / BAWA PORTO.

## Why this matters

Institutional World Cup forecasts have usually tried to solve a tournament-winner problem:

- estimate each team's macro strength
- simulate the bracket many times
- publish a champion forecast

Odds Genius is solving a different and more commercially useful problem:

- select individual match markets
- avoid fragile fixture contexts
- protect against late pre-match shocks
- deploy only when the evidence stack is clean enough

That means institutional models are useful as a benchmark, but not the full edge.

## External Benchmark Signals

Known institutional/model families:

- Monte Carlo tournament simulations
- Poisson expected-goal models
- Elo/FIFA/team-strength ratings
- player-level quality and availability
- bookmaker odds priors
- machine-learning classifiers
- bracket path difficulty

Useful public examples to keep in view:

- Joachim Klement / investment strategy style World Cup forecasting
- Goldman Sachs 2018 World Cup model
- broader 2018 investment-bank prediction attempts

These models help define the baseline feature families we should not ignore.

## Odds Genius Differentiated Edge

The likely advantage is not merely predicting the champion. It is combining macro strength with deploy selectivity.

Candidate edge layers:

- lineup volatility
- injury/suspension shock
- player rating delta against expected XI
- team power rating momentum
- tactical mismatch and fouling pressure
- referee behaviour
- weather and venue conditions
- travel and rest fatigue
- match-state incentives
- draw/stalemate risk
- market structure traps
- late sentiment/news movement
- competition-stage pressure

For World Cup group-stage betting, the strongest initial edge candidates are:

- team-strength prior
- squad/player-quality prior
- injuries and likely XI disruption
- tactical pace/shot profile
- weather/venue/travel adjustment
- market-price disagreement
- draw/stalemate shape

## Current Estate Position

Already built:

- FootyStats World Cup audit for 2006, 2010, 2014, 2018, 2022
- World Cup research foundation
- API-Football World Cup coverage audit
- FootyStats/API fixture bridge for 2010, 2014, 2018, 2022
- `Matches/__merged__/World_Cup__merged.csv` adapter for priced 2018 and 2022 rows
- 2026 launch scaffold from API-Football fixtures plus historical team priors
- 2026 macro prior engine with optional external/Kaggle prior joins
- 2026 player/squad intelligence scaffold
- 2026 qualification-context scaffold for confederation, route, travel/climate, volatility and market-risk priors
- FootyStats World Cup qualification foundation across 2018, 2022, and 2026 cycles where supplied
- Fjelstul World Cup archive historical priors for teams, stages, referees, goals, and card style through 2018
- 2026 fixture-level research feature matrix joining macro, player scaffold, qualification, and historical World Cup priors
- FootyStats additions context sidecar from regional tournaments, friendlies, and extra qualification files

Known limitations:

- 2006, 2010, and 2014 FootyStats rows have no usable prices in the current drop.
- API-Football currently exposes 72 World Cup 2026 group-stage fixtures, not the full 104-match tournament.
- 2026 odds are pending because API-Football historical odds retrieval is window-limited.
- 2026 player intelligence is pending until squads, injuries, and player endpoint data are refreshed and timestamped.
- Fjelstul archive coverage ends at 2018 and is CC-BY-SA 4.0, so derived product packaging needs attribution/license review.
- FootyStats 2026 qualifier aggregates cover 45 of 48 scheduled teams; the three misses are host auto-qualifiers Canada, Mexico, and USA.
- Additions player ratings can contain source outliers above a 0-10 scale; the additions builder repairs these with `ratings_total_overall / appearances_overall` where valid, caps borderline repaired values just above 10, and emits an anomaly audit.

## Feature Build Direction

The next World Cup feature families should be added as sidecars first:

1. `world_cup_team_strength_prior`
   - historical World Cup PPG
   - goal difference per match
   - stage reach
   - recent tournament recency weighting
   - confederation/region indicator

2. `world_cup_market_prior`
   - FTR, BTTS, OU25, TG1.5 odds where available
   - implied probability and no-vig transforms
   - market disagreement against model probability

3. `world_cup_player_quality_prior`
   - squad player ratings
   - expected XI quality
   - attacking/defensive player absence weight
   - goalkeeper/centre-back/striker shock flags

4. `world_cup_context_risk`
   - group-stage match number
   - knockout flag
   - qualification incentive
   - draw-suffices flag
   - rotation risk
   - host/venue/travel/rest adjustment

5. `world_cup_deploy_safety`
   - FTR block/caution band
   - draw/stalemate risk
   - injury shock
   - lineup uncertainty
   - market trap flag
   - low-confidence fixture identity flag

6. `world_cup_qualification_context`
   - confederation
   - qualification route
   - playoff/intercontinental-playoff flag
   - qualification group position and PPG
   - qualification goal difference per match
   - route verification flag
   - travel/climate complexity prior
   - confederation volatility prior
   - market efficiency risk prior

## Macro Prior Engine

Current research scaffold:

- `scripts/build_world_cup_macro_prior_engine.py`
- `data_sources/footystats_world_cup/macro_prior_engine/world_cup_team_macro_strength_2026.csv`
- `data_sources/footystats_world_cup/macro_prior_engine/world_cup_2026_macro_prior_fixture_matrix.csv`
- `data_sources/footystats_world_cup/macro_prior_engine/world_cup_2026_macro_probability_board.csv`
- `data_sources/footystats_world_cup/macro_prior_engine/world_cup_external_prior_template.csv`

The first pass uses only pre-2026 World Cup priors and current API-Football fixture schedule data. It can be rerun with external priors:

```bash
python3 scripts/build_world_cup_macro_prior_engine.py \
  --external-priors data_sources/footystats_world_cup/macro_prior_engine/world_cup_external_prior_template.csv \
  --outdir data_sources/footystats_world_cup/macro_prior_engine
```

Supported external/Kaggle-style columns include:

- `team_slug` or `team_name`
- `fifa_rank`
- `fifa_points`
- `elo` or `elo_rating`
- `squad_market_value_eur`
- `squad_avg_rating`
- `expected_xi_avg_rating`
- `domestic_player_rating`
- `gdp_per_capita`
- `population`
- `mean_temperature_c`

This is the baseline layer that CatBoost/XGB/goal-mass systems can consume before lineup, injury, weather, referee, and market-structure filters are joined.

## Qualification Context Scaffold

Current research scaffold:

- `scripts/build_world_cup_qualification_context_scaffold.py`
- `scripts/build_world_cup_footystats_qualification_foundation.py`
- `data_sources/footystats_world_cup/qualification_foundation/world_cup_2026_qualification_override_from_footystats.csv`
- `data_sources/footystats_world_cup/qualification_context_2026_footystats_enriched/world_cup_2026_team_qualification_context.csv`
- `data_sources/footystats_world_cup/qualification_context_2026_footystats_enriched/world_cup_2026_fixture_qualification_context_matrix.csv`
- `data_sources/footystats_world_cup/qualification_context_2026_footystats_enriched/world_cup_2026_model_ready_with_qualification_context.csv`
- `data_sources/footystats_world_cup/qualification_context_2026_footystats_enriched/world_cup_qualification_context_override_template.csv`

The scaffold uses confederation priors, host auto-qualification, and FootyStats qualifier aggregate team stats. Detailed route fields are still intentionally left as override/template fields until a fully verified qualification-route dataset is joined.

Current FootyStats qualifier coverage:

- Teams with qualifier aggregate stats: 45 / 48
- Fixtures with both sides covered: 63 / 72
- Fixtures with at least one side covered: 72 / 72
- Host auto-qualifier misses: Canada, Mexico, USA

Supported verified override fields include:

- `qualification_route`
- `qualification_route_verified_flag`
- `qualification_group`
- `qualification_position`
- `qualification_points`
- `qualification_goal_diff`
- `qualification_playoff_flag`
- `nations_league_playoff_context_flag`
- `intercontinental_playoff_flag`
- `qualifier_matches_played`
- `qualifier_ppg`
- `qualifier_goal_diff_per_match`
- `qualifier_goals_for_per_match`
- `qualifier_goals_against_per_match`

## Fjelstul Historical Priors

Current research scaffold:

- `scripts/build_world_cup_fjelstul_historical_priors.py`
- `data_sources/footystats_world_cup/fjelstul_historical_priors/world_cup_fjelstul_team_historical_priors.csv`
- `data_sources/footystats_world_cup/fjelstul_historical_priors/world_cup_2026_team_historical_prior_sidecar.csv`
- `data_sources/footystats_world_cup/fjelstul_historical_priors/world_cup_fjelstul_stage_event_priors.csv`
- `data_sources/footystats_world_cup/fjelstul_historical_priors/world_cup_fjelstul_referee_style_priors.csv`

Current Fjelstul coverage:

- Historical World Cup team priors joined to 2026 field: 42 / 48
- Recent 2010-2018 team priors joined to 2026 field: 34 / 48
- Stage/event prior rows: 14
- Referee style prior rows: 380

Use this as a World Cup-specific macro context layer, not as current squad intelligence. It is especially useful for stage priors, tournament-history deltas, referee/card priors, and group-stage QA.

## Research Feature Matrix

Current combined sidecar:

- `scripts/build_world_cup_additions_context_sidecar.py`
- `scripts/build_world_cup_recent_history_and_h2h_sidecar.py`
- `scripts/build_world_cup_venue_weather_travel_scaffold.py`
- `scripts/build_world_cup_2026_research_feature_matrix.py`
- `scripts/build_world_cup_modelling_target_harness.py`
- `data_sources/footystats_world_cup/additions_context_2026/world_cup_additions_context_sidecar.csv`
- `data_sources/footystats_world_cup/additions_context_2026/world_cup_additions_player_rating_anomalies.csv`
- `data_sources/footystats_world_cup/recent_history_h2h_2026/world_cup_2026_fixture_recent_history_sidecar.csv`
- `data_sources/footystats_world_cup/recent_history_h2h_2026/world_cup_2026_local_h2h_sidecar.csv`
- `data_sources/footystats_world_cup/recent_history_h2h_2026/world_cup_api_h2h_manifest.csv`
- `data_sources/footystats_world_cup/venue_weather_travel_2026/world_cup_2026_fixture_venue_travel_weather_scaffold.csv`
- `data_sources/footystats_world_cup/research_feature_matrix_2026/world_cup_2026_research_feature_matrix.csv`
- `data_sources/footystats_world_cup/research_feature_matrix_2026/world_cup_2026_research_feature_readiness_counts.csv`
- `data_sources/footystats_world_cup/research_feature_matrix_2026/world_cup_2026_research_feature_readiness_v2_counts.csv`
- `data_sources/footystats_world_cup/modelling_target_harness/world_cup_historical_targets_wide.csv`
- `data_sources/footystats_world_cup/modelling_target_harness/world_cup_historical_targets_long.csv`
- `data_sources/footystats_world_cup/modelling_target_harness/world_cup_2026_ablation_readiness.csv`

Current fixture coverage:

- Fixtures: 72
- Both-side FootyStats qualifier coverage: 63 / 72
- Both-side Fjelstul historical coverage: 55 / 72
- Both-side Fjelstul recent 2010-2018 coverage: 35 / 72
- Both-side additions context coverage: 72 / 72
- Both-side additions player context coverage: 72 / 72
- Both-side recent-history coverage: 72 / 72
- Local FootyStats H2H available: 15 / 72
- API-Football H2H fetch manifest rows: 72 / 72
- Venue/weather/travel scaffold known venue: 62 / 72
- Venue assignment pending: 10 / 72

Current target harness:

- Historical labelled World Cup fixtures: 128 across 2018 and 2022
- Long target rows: 768 across FTR, BTTS, OU25, home TG1.5, away TG1.5, any-team TG1.5
- Market baseline hit rates: FTR favourite 57.0%, BTTS market pick 53.9%, OU25 market pick 50.8%
- 2026 ablation groups emitted: macro only, macro plus qualifier, macro plus additions, macro plus Fjelstul history, full stack
- Current full-stack feature group: 369 features, 343 numeric, 89.0% mean numeric non-null coverage

This is the clean fixture-level sidecar for CatBoost/XGB/goal-mass experiments before adding verified final-squad, injury, likely-XI, weather, referee appointment, and market-price windows.

The honest modelling interpretation is important:

- macro-only and macro plus Fjelstul can be compared against the current 2018/2022 historical World Cup adapter
- qualifier, additions, recent-history, H2H, venue, travel, and full-stack accuracy tests need backbuilt 2018/2022 time-safe sidecars before being treated as real ablation results
- venue/weather/travel currently contains venue geography and climate/travel proxies only; `weather_snapshot_status` remains pending until prematch weather API pulls are available
- official 2026 squads, injuries, lineups, and final team sheets remain the later truth layer when API coverage matures

## Research Model Runner and Historical Backbuild

Current research scripts:

- `scripts/run_world_cup_research_model_ablation.py`
- `scripts/build_world_cup_historical_full_stack_backbuild.py`
- `scripts/run_world_cup_trainer_native_research.py`

Current output roots:

- `data_sources/footystats_world_cup/model_ablation_runs/leak_safe_2018_train_2022_test/`
- `data_sources/footystats_world_cup/model_ablation_runs/full_stack_backbuilt_2018_train_2022_test/`
- `data_sources/footystats_world_cup/historical_full_stack_backbuild/`

Runner posture:

- canonical input only: `Matches/__merged__/World_Cup__merged.csv`
- train/test split: 2018 World Cup train, 2022 World Cup test
- no ModelStore writes
- no production trainer use
- no deploy policy changes
- CatBoost/XGBoost are supported by the runner but were not installed locally during this smoke run
- local engines used: LightGBM, sklearn HistGradientBoosting, logistic regression, plus Poisson goal-mass baseline

Leak-safe run highlights on the 2022 holdout:

- FTR: best observed was macro plus Fjelstul via LightGBM at 59.4%
- BTTS: macro plus Fjelstul improved sharply versus macro-only, with LightGBM at 64.1% and sklearn HGB at 62.5%
- OU25: macro-only logistic and macro plus Fjelstul logistic both reached 62.5%, but tree models were weaker
- Home TG1.5: macro-only logistic reached 68.8%
- Away TG1.5: macro plus Fjelstul sklearn HGB reached 70.3%
- Any-team TG1.5: macro-only logistic reached 73.4%

Backbuilt full-stack sidecar coverage:

- recent-history both sides: 128 / 128
- qualifier-history both sides: 113 / 128
- all-prior additions history both sides: 128 / 128
- local FootyStats H2H available: 53 / 128
- historical venue known: 128 / 128

The qualifier blocker was mostly source-scope, not source-absence. The historical backbuild now scans the full `FOOTYSTATS_DROP` international match estate, not only `World Cup Additions`.

Trainer-native runner:

- imports `train_markets.build_features`
- imports `train_markets._derive_targets`
- imports `train_markets._best_f1_threshold`
- uses CatBoost/XGBoost when installed
- writes research CSVs only
- does not write `ModelStore`

Recommended command in the CatBoost/XGBoost training environment:

```bash
python3 scripts/build_world_cup_historical_full_stack_backbuild.py \
  --source-mode both \
  --footystats-drop /Users/hughwade/Desktop/FOOTYSTATS_DROP \
  --outdir data_sources/footystats_world_cup/historical_full_stack_backbuild

python3 scripts/run_world_cup_trainer_native_research.py \
  --engines catboost,xgboost \
  --groups trainer_native_macro,trainer_native_full_stack_backbuilt \
  --markets ftr,btts,over25,home_ge2,away_ge2,any_team_ge2 \
  --iterations 500 \
  --threads 4 \
  --outdir data_sources/footystats_world_cup/trainer_native_research_runs/catboost_xgb_2018_train_2022_test
```

This is the first serious World Cup model-run path because it keeps the World Cup harness research-only while using the same feature construction and target conventions as the existing core training script.

## Research Boundary

Do not promote World Cup sidecars into production gates until:

- the fields are timestamp-safe
- fixture identity joins are audited
- training uses only `Matches/__merged__/World_Cup__merged.csv`
- any additional source files are documented by source timestamp and coverage
- the sidecar improves retained hit rate or ROI without unacceptable missed-winner cost

The right product posture is:

- use institutional methods as baseline priors
- use Odds Genius intelligence layers as selective deployment filters
- report coverage explicitly
- deploy only trusted coverage, not maximum coverage
