# Odds Genius System Brain

Repo: `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO`
Updated: `2026-05-04`
Purpose: canonical system-brain note for Codex, future chat threads, and the website / webapp build handoff.

## 1. System Mission
Odds Genius / BAWA PORTO is a football prediction and deploy system built around:
- strong pre-match canonical data
- market-specific predictive models
- deterministic deploy routing
- conservative slip formatting
- auditable walk-forward and backtest outputs

The current production-safe core is the goal-model estate:
- `FTR`
- `BTTS`
- `OU25`
- `CS` overlays / support structure

Player events, weather, and newer research lanes remain separate from the production spine unless explicitly promoted.

## 2. Current State Summary
### Mature / production-strong
- `FTR`, `BTTS`, and `OU25` are the mature goal-model markets.
- `deploy_rulebook.py` is the live source of truth for routing and tiering.
- `slip_formatter.py` is thin and should stay thin.
- goal-model outputs are now proving themselves in live deploy and curated acca use.

### Research-active
- player-events probability engine proof (`tackles` Phase 1 first)
- PPG-dominance / larger-acca reverse engineering
- website / webapp orchestration and automation handoff

### User-reported live notes to preserve
These are not formal repo-computed metrics; they are live operating notes worth carrying forward:
- BTTS elite list: `1` short of a full win on the latest weekend
- OU25 elite list: `3` short on the latest weekend
- curated larger odds acca landed at roughly `29/1` (`£10 -> £296`)
- `slip_formatter` top `FTR + BTTS` landed at roughly `£10 -> £100`
- larger-odds style selections have now landed across `3/4` recent weeks

## 3. Protected Production Spine
Treat these files as the protected production chain:
- `footystats_drop_ingest.py`
- `etl_press_intensity.py`
- `build_merged.py`
- `patch_merge_add_streaks.py`
- `team_ratings.py`
- `patch_merge_add_power_ratings.py`
- `make_fd_odds_enriched_synth.py`
- `patch_merge_add_synth_odds.py`
- `pipeline_qa_gate.py`
- `bookie_allmarkets.py`
- `deploy_rulebook.py`
- `slip_formatter.py`

Rules:
- never train from raw season CSVs
- canonical training input is merged data only
- never predict before integrity passes
- `deploy_rulebook.py` owns routing, tiers, vetoes, and gates
- `slip_formatter.py` must not add filler, rescue, or prediction logic
- `OBSERVE` rows are non-deployable

## 4. Canonical Pipeline Flow
### Data refresh / canonical build
1. `footystats_drop_ingest.py`
2. `etl_press_intensity.py`
3. `build_merged.py --recursive --rolling-press`
4. `patch_merge_add_streaks.py`
5. `team_ratings.py`
6. `patch_merge_add_power_ratings.py`
7. `make_fd_odds_enriched_synth.py --emit-ou25-novig`
8. `patch_merge_add_synth_odds.py`
9. `pipeline_qa_gate.py`

Hard stop:
- if integrity fails, do not run `bookie_allmarkets.py`

### Prediction / deploy / product flow
1. `bookie_allmarkets.py`
2. `deploy_rulebook.py`
3. `slip_formatter.py`
4. optional audit / walk-forward scoring / slip audit

### Historical validation flow
1. canonical merged inputs
2. train / refresh models
3. `bookie_allmarkets.py --strict` where appropriate
4. `run_walkforward_windows.py`
5. `run_slip_walkforward_audit.py`
6. per-market / per-tier / per-window audit outputs under `reports/` and `predictions_output/`

## 5. Core Script Map
### Data + enrichment
- `footystats_drop_ingest.py`: raw FootyStats ingest
- `etl_press_intensity.py`: press-intensity enrichment
- `build_merged.py`: canonical merged dataset builder
- `patch_merge_add_streaks.py`: streak features
- `team_ratings.py`: team rating generation
- `patch_merge_add_power_ratings.py`: merged power-rating patch
- `make_fd_odds_enriched_synth.py`: conservative synth odds construction
- `patch_merge_add_synth_odds.py`: merged synth patch
- `pipeline_qa_gate.py`: integrity and schema gate

### Training / model build
Primary trainers:
- `train_markets.py`
- `train_investor_leagues_v2.py`

Specialized / supporting trainers:
- `train_hybrid_xgb.py`
- `train_hybrid_catboost.py`
- `train_hybrid_goal_mass.py`
- `train_hybrid_ou25_tuned.py`
- `train_hybrid_side_markets.py`
- `train_ftr_with_team_stats.py`
- `train_draw_classifier.py`
- `retrain_all.py`
- `retrain_btts_expanded.py`
- `retrain_over25_expanded.py`
- `retrain_secondary_goal_markets_all.py`
- `train_poison_leg_model.py`

Training guardrails:
- use one trainer path intentionally per run
- verify `ModelStore/` outputs after training
- do not mix ad hoc artifact naming schemes casually

### Prediction board + routing
- `bookie_allmarkets.py`: builds the all-markets candidate pool
- `deploy_rulebook.py`: deterministic live routing and tiering
- `build_correct_score_deploy.py`: correct-score deploy support
- `meta_to_allmarkets.py`: metadata / board augmentation helper

### Product formatting
- `slip_formatter.py`: ranked boards, family summary, singles, doubles, trebles, yankees, heinz, fixed accas

### Walk-forward / audit / proof
- `run_walkforward_windows.py`: windowed historical generation and scoring
- `run_slip_walkforward_audit.py`: ranked-board / slip audit over walk-forward outputs
- `build_correct_score_backtest_summary.py`: CS backtest summary
- `scripts/player_events/proof/build_tackles_nb_proof.py`: Phase 1 player-events proof runner

## 6. Markets In Scope
### Core production markets
- `FTR`
- `BTTS`
- `OU25`
- `CS` support / overlays

### Supporting / derived families seen in output estate
- team-goals style lanes (`home_ge2`, `away_ge2`, `home_ge3`, `away_ge3`, `FTS` families)
- Poisson / lambda support for goal-mass logic
- correct-score concentration and alignment overlays

### Research / beta
- player events
- weather
- newer side markets / acca research overlays unless explicitly promoted

## 7. Key Modeling Ideas
### Goal-model logic
The goal-model stack is no longer just a rulebook. It combines:
- canonical merged features
- market-specific models
- goal-mass / Poisson support
- correct-score top-3 structure
- deploy rulebook gating

### Value edge logic
Common edge families across the live goal stack:
- `model_p_for_bookie` versus `bookie_implied` / `bookie_implied_novig`
- `gap` / `gap_novig`
- `ftr_margin`
- `power_diff`
- `pick_side_mass_top3`
- `pick_side_margin_top3`
- `signal_btts`
- `signal_over25`
- `exp_goals_sum`
- `bookie_lambda_total_fit`
- correct-score alignment / fragility / banker structure

### Human curation logic now worth preserving
A separate profitable layer is emerging from manual / curated larger acca work, especially around:
- `PPG dominance`
- H2H history
- home-team recent goals scored / conceded
- away-team recent goals scored / conceded
- FootyStats overview texture
- preview-text context such as Sports Mole notes

This is not yet formalized into the core production engine, but it is now important enough to preserve and later reverse engineer.

## 8. Tier Logic
Canonical deploy tiers:
- `ELITE`: strongest deploy-ready rows
- `STANDARD`: deployable but lower-priority / rescued rows
- `OBSERVE`: non-deployable monitoring / shadow rows

Important conventions:
- `OBSERVE` is never live deployable
- `ELITE` and `STANDARD` feed live boards and slips
- `standard_reporting_bucket` carries lane-level subgroup identity
- `profit_first_keep` is optional metadata, not the routing authority

Examples of known standard families / buckets in the estate:
- `STANDARD_FTR_CS_PROMOTED_ALIGNED`
- `STANDARD_FTR_BASE`
- `STANDARD_BTTS`
- `STANDARD_OU25`
- combo / rescue families for FTR and BTTS

## 9. Output File Structure
### Canonical inputs
- merged training inputs:
  - `Matches/__merged__/<LEAGUE_TAG>__merged.csv`
- enriched league files:
  - `Matches/<League>/fd_odds_enriched.csv`
  - `Matches/<League>/fd_odds_enriched_synth.csv`

### Models
- `ModelStore/<LeagueTag>/...`
- goal ensembles, calibrators, thresholds, feature manifests, market metrics

### Daily / run outputs
- `predictions_output/<RUN_DATE>/...`

Typical prediction / deploy outputs:
- `BOOKIE_IMP*_ALLMARKETS_<FROM>_to_<TO>.csv`
- `...__DEPLOY_PRESET_V1.csv`
- `...__DEPLOY_PRESET_V1.md`
- `...__DEPLOY_TIER_ELITE__...csv`
- `...__DEPLOY_TIER_STANDARD__...csv`
- `...__DEPLOY_TIER_OBSERVE__...csv`
- gate audit CSVs
- debug logs

### Walk-forward estate
Typical window structure under `predictions_output/<RUN_ID>/wXXX_<window>/`:
- `01_source/`
- `02_deploy/`
- `03_scored/`

Common files:
- `DEPLOY_CANDIDATES_RAW.csv`
- `DEPLOY_CANDIDATES_AFTER_GATES.csv`
- `DEPLOY_COMBINED_<FROM>_to_<TO>.csv`
- scored versions of deploy outputs
- per-window board / gate audit files

### Slip / board outputs
Often under `reports/<DATE>/...` or walk-forward slip folders:
- `ranked_board_*.csv`
- `ranked_board_ftr_*.csv`
- `ranked_board_btts_*.csv`
- `ranked_board_ou25_*.csv`
- `ranked_board_family_summary_*.csv`
- `slips_singles_*.csv`
- `slips_doubles_*.csv`
- `slips_trebles_*.csv`
- `slips_yankee_*.csv`
- `slips_heinz_*.csv`
- `slips_acca03_*.csv`, `slips_acca04_*.csv`, etc.

### Reports / audits
- `reports/<DATE>/...`
- `reports/player_events/...`
- `reports/player_events/proof/...`
- `reports/.../SLIP_WALKFORWARD_AUDIT__<RUN_ID>/...`

## 10. Key Metrics To Carry Forward
### Goal-model / deploy metrics
- hit rate by market
- hit rate by deploy tier
- hit rate by `standard_reporting_bucket`
- top-board precision
- acca family hit rate / ROI
- walk-forward Brier / log-loss where available
- per-league coverage and quality

### Value / structure metrics
- model minus bookie gap
- no-vig edge
- correct-score support mass
- draw / chaos flags
- Poisson coherence for goal-side and team-goals logic

### Player-events proof metrics
Current proof discipline for Track A focuses on:
- MAE on expected count
- Brier score
- log-loss
- ECE / calibration
- top-decile precision
- coverage
- leak audit pass/fail

## 11. Current Product Positioning
### What is strong enough to talk about confidently
- goal-model markets are the mature commercial core
- deploy routing and slip formatting are auditable and structured
- larger curated acca logic is now worth formal preservation and later codification

### What is not yet ready to claim as fully mature
- player-events probability engine
- cards hazard modeling
- full public website automation until the pipeline handoff is explicitly wired

## 12. Expected Automated State
Target operational architecture:
1. refresh canonical source data safely
2. rebuild / patch merged inputs
3. QA gate passes
4. generate all-markets board
5. route via deploy rulebook
6. emit ranked boards and slip products
7. publish selected outputs to a website / webapp layer
8. archive artifacts to GitHub-backed repo history and/or report folders

## 13. Website / Webapp Handoff Requirements
The future website / webapp thread should assume:
- the prediction pipeline already exists and should not be re-invented in the frontend
- the frontend should consume generated outputs, not create betting logic itself
- GitHub should be used for code / version control
- Cloudflare is the expected web delivery / automation edge

### Website connection points to plan around
Likely feed surfaces:
- latest `BOOKIE_IMP*_ALLMARKETS_*.csv`
- latest deploy-tier CSVs
- ranked boards from `slip_formatter.py`
- summarized report markdown / CSV artifacts
- future API or scheduled-export layer derived from those outputs

### Frontend responsibilities
- surface latest run metadata
- show deploy tiers cleanly
- expose ranked boards / market boards
- display confidence / reason-token / context summaries
- separate production picks from research outputs
- preserve auditability: every displayed pick should be traceable back to a generated file

### Backend / orchestration responsibilities
- scheduled data refresh
- scheduled prediction generation
- scheduled deploy routing
- export normalization for web consumption
- artifact versioning and fallback to last good run if a new run fails QA

## 14. Future Thread Use
This document is intended to let a fresh Codex thread quickly orient around:
- what the production spine is
- what must not be broken
- what the mature markets are
- where outputs live
- what the webapp should connect to
- what remains research versus production

Use this as the first orientation document before asking Codex to work on:
- website build
- Cloudflare / GitHub deployment wiring
- export normalization
- product presentation layers
- automation around prediction output publishing
