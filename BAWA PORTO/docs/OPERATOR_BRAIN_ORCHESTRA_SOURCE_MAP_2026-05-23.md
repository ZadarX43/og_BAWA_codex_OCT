# Operator / Brain / Orchestra Source Map - 2026-05-23

Purpose: identify which root-level support files should be folded into the formal Odds Genius operating layers before the next cleanup or automation pass.

This map does not change model behavior or deploy routing. It records ownership so production-adjacent helpers stop living as invisible local files.

## Layer Definitions

- Operator: runs the safe order, checks integrity, blocks unsafe prediction/deploy work, and exposes repeatable commands.
- Brain: builds reusable football intelligence state from model outputs, merged data, ratings, context, injuries, weather, player/team signals, and fixture history.
- Orchestra: turns Brain outputs into website, Worker, R2/D1, Telegram, summaries, alerts, public proof, and user-facing tier payloads.

## First Commit Group: Core Support Modules

These files are imported by production or production-adjacent scripts and should be tracked before more archive work:

- `constants.py`: canonical feature allowlists, league thresholds, and shared settings used by training/inference helpers.
- `prediction_overlay.py`: shared prediction overlay/support logic used by all-markets generation and downstream audits.
- `prediction_report.py`: lightweight report helper used by historical prediction scripts.
- `signal_layers.py`: reusable signal layer helpers imported by all-markets and overlay paths.
- `streaks_module.py`: streak/context helper imported by merged-build and all-markets paths.
- `og_model_paths.py`: ModelStore path resolution helper imported by all-markets and packaging helpers.
- `deploy_gates.py`: deploy gate helper imported by deploy rulebook variants and deploy tooling.
- `deploy_presets.py`: season preset/cap helper used by rulebook filtering tooling.
- `manifest_and_calendar_flags.py`: calendar/manifest context helper for fixture windows.
- `odds_synth.py`: odds synthesis helper used in the odds-enrichment family.
- `correct_score_product.py`: correct-score product helper used by correct-score build scripts.

## First Commit Group: Config / Policy Contracts

These files define policy/config contracts rather than generated reports:

- `btts_league_policy.json`
- `btts_no_live_allowlist.json`
- `ou25_league_policy.json`
- `over25_deploy_policy.json`
- `over25_from_ou25_policy.json`
- `under25_btts_no_support_allowlist.json`
- `market_proxy_league_config.json`
- `market_proxy_promotion_config.json`
- `market_cs_proxy_promotion_config.json`
- `ratings_publish_sources.sample.json`
- `pyproject.toml`
- `mypy.ini`

## Next Commit Group: Brain Intelligence Modules

These should be tracked in the next group after a compile/import check:

- `injury_shock_engine.py`: injury-to-market impact sidecar for FTR, BTTS, and OU25 support/contradiction.
- `team_rating_engine.py`: team rating helper for fixture intelligence and team context cards.
- `travel_fatigue.py`: travel/context helper for fixture support layers.
- `uefa_context.py`: UEFA/world/international competition context helper.
- `weather_data.py`: weather soft-context helper.
- `seasonal_market_ledger.py`: market ledger/history helper.

Do not commit yet:

- `player_prop_models.py`: currently appears to be a filename-only stub.
- `player_usage_profiles.py`: currently appears to be a filename-only stub.

## Already Tracked Orchestra / Website Spine

These files already exist in tracked history and are the current website/publish orchestration layer:

- `fixture_decision_reconciler.py`
- `fixture_h2h_support_engine.py`
- `fixture_lineup_intelligence_engine.py`
- `fixture_preview_generator.py`
- `publish_fixture_intelligence.py`
- `publish_predictions.py`
- `grade_weekend_results.py`
- `validate_fixture_intelligence.py`
- `validate_public_export.py`
- `validate_weekly_results.py`
- `validate_live_results_feed.py`
- `build_covered_fixture_universe.py`
- `validate_covered_fixture_universe.py`
- `build_league_coverage_audit.py`
- `validate_league_coverage_audit.py`
- `build_league_remediation_artifact.py`
- `validate_league_remediation_artifact.py`

## Explicitly Not Folded Yet

- `game_context.py`: currently compiles but appears to contain only a filename expression; leave untracked until reviewed.
- `player_prop_models.py` and `player_usage_profiles.py`: filename-only stubs; leave untracked until replaced by real player-event Brain modules.
- `train_markets.py`, `train_investor_leagues_v2.py`, and other trainer files: imported and important, but should be committed in a training-specific group, not mixed into operator/brain/orchestra.
- Research/backtest/audit harnesses: archive/review by family after checking current docs/runbooks.
- Shell runners other than `deploy_weekend_runner.sh` and `rebuild_all_merged.sh`: review before tracking; many are old frozen/sandbox runners.

## Recommended Integration Direction

1. Operator owns command sequencing and hard stops.
2. Brain owns compact per-fixture intelligence contracts and sidecars.
3. Orchestra owns publish/export, R2/D1 deltas, site smoke checks, Telegram-ready summaries, and user tier payloads.
4. Model/deploy behavior remains owned by the production spine, especially `bookie_allmarkets.py`, `deploy_rulebook.py`, and `slip_formatter.py`.
