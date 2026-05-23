# Odds Genius Troubleshooting Guide - 2026-05-23

## Purpose

This guide gives a fast route from symptom to files, commands, and likely causes.

Use this alongside `docs/ODDS_GENIUS_SYSTEM_OPERATING_MAP_2026-05-23.md`.

## First Checks

Run these before changing code:

```bash
git status --short
git status --short -- footystats_drop_ingest.py etl_press_intensity.py build_merged.py patch_merge_add_streaks.py team_ratings.py patch_merge_add_power_ratings.py make_fd_odds_enriched_synth.py patch_merge_add_synth_odds.py pipeline_qa_gate.py bookie_allmarkets.py deploy_rulebook.py slip_formatter.py
```

Expected:

- Production spine clean unless the current task explicitly touches it.
- Generated payload families should not dirty Git.

## Symptom: Fresh Data Arrived But Predictions Look Old

Check:

- Did FootyStats ingest run?
- Did merged files update under `Matches/__merged__/`?
- Did `pipeline_qa_gate.py` pass?
- Did `bookie_allmarkets.py` run after the fresh data?
- Does `frontend/public/data/publish_summary.json` point at the expected run?

Files:

- `FOOTYSTATS_DROP/`
- `Matches/__merged__/`
- `frontend/public/data/publish_summary.json`
- `predictions_output/`

Likely fix:

- Rerun the safe operating order from data refresh through deploy.

## Symptom: Website Fixture Page Shows Old or Wrong Market Cards

Check:

- Static contract files:
  - `frontend/public/data/fixture_intelligence_public.json`
  - `frontend/public/data/public_predictions.json`
  - `frontend/public/data/premium_predictions.json`
- Brain payload:
  - `build/site_publish/current/payloads/fixtures/`
- Worker route:
  - D1 index row points at the right R2 object.

Likely causes:

- Static frontend snapshot updated but R2/D1 was not uploaded.
- R2 payload uploaded but D1 index points at the previous hash/path.
- Fixture card is falling back to static data while Worker route is stale.

Useful commands:

```bash
python3 scripts/site_publish/orchestrator.py --from-date YYYY-MM-DD --to-date YYYY-MM-DD
python3 scripts/cloudflare_preview_readiness.py
```

## Symptom: FTR / BTTS / OU25 Card Disagrees With Published Pick

Check:

- Is the card reading actual all-market model output or contextual support?
- Is route/audit split present?
- Is the published deploy route from `deploy_rulebook.py` being displayed separately from caution/audit context?

Files:

- `fixture_decision_reconciler.py`
- `scripts/audit_fixture_decision_route_conflicts.py`
- `frontend/public/data/fixture_intelligence_public.json`
- `frontend/public/data/premium_predictions.json`

Rule:

- Do not change `deploy_rulebook.py` until the conflict is proven to be a routing/gate problem rather than a display/reconciler problem.

## Symptom: Player / Team / H2H Intelligence Missing

Check source coverage:

- API-football normalized/local source files
- site SQLite export
- fixture brain compiler output
- R2 payload for the fixture

Files/scripts:

- `scripts/api_football/refresh_current_context_overlay_window.py`
- `scripts/export_site_sqlite.py`
- `scripts/site_publish/fixture_brain_compiler.py`
- `scripts/build_goal_market_overlay_support_audit.py`
- `scripts/build_injury_shock_market_impact_sidecar.py`

Likely causes:

- API-football source data missing for that league/window.
- H2H history is thin and fallback copy should show.
- Player-event cards did not compile because lineup/player source was absent.
- R2/D1 publish did not include the latest compact payload.

## Symptom: Injury Shock Report Lists Players But No Market Impact

Check:

- Player position/rating is available.
- Injury/sidelined source joined to team/player ratings.
- Market impact sidecar ran after fresh player/team data existed.

Scripts:

- `scripts/build_injury_shock_coverage_scan.py`
- `scripts/build_injury_shock_market_impact_sidecar.py`
- `scripts/api_football/fetch_injuries.py`
- `scripts/api_football/fetch_sidelined.py`
- `scripts/api_football/build_injury_features.py`

Expected output:

- FTR impact
- BTTS impact
- OU25 impact
- player rating / role / position
- confidence or coverage caveat

## Symptom: Results Page Is Stale or Picks Stay Pending

Check:

- Settlement source has final scores.
- Published pick IDs match fixture/result IDs.
- `weekly_results.json` and `results_archive.json` were regenerated.
- Results page uses the latest static contract or Worker route.

Files:

- `scripts/settle_published_results.py`
- `frontend/public/data/weekly_results.json`
- `frontend/public/data/results_archive.json`
- `frontend/public/data/live_results_feed.json`

Important:

- Settlement should be deterministic. Reruns should not dirty files only because the clock changed.

## Symptom: Checkout / Account Smoke Fails

Check Worker env secrets:

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `PREMIUM_TOKEN_SECRET`
- `AUTH_MAGIC_LINK_SECRET`
- `AUTH_SESSION_SECRET`
- `RESEND_API_KEY`
- `AUTH_EMAIL_FROM`

Files:

- `worker/wrangler.toml`
- `worker/src/index.js`
- `worker/test_worker_local.js`

Commands:

```bash
python3 scripts/smoke_live_worker_account.py
```

Likely causes:

- Test Worker env missing secrets.
- Stripe webhook secret belongs to wrong endpoint/env.
- Resend sender/domain mismatch.
- Session secret differs between issue/restore flows.

## Symptom: Cloudflare Preview Readiness Fails

Check:

- Pages preview URL
- Worker URL/env
- D1 binding/index
- R2 bucket binding
- static contract files
- checkout/auth secrets

Command:

```bash
python3 scripts/cloudflare_preview_readiness.py
```

Likely causes:

- Frontend deployed but Worker/D1/R2 not updated.
- R2 payload uploaded but D1 delta not applied.
- Test env missing account/checkout secrets.
- Static contract points at a different publish run than the R2 manifest.

## Symptom: Repo Gets Dirty After Normal Work

Expected ignored/generated areas:

- `data_sources/api_football/normalized/`
- `reports/latest/`
- `frontend/public/data/fixture_decision_intelligence/`
- `frontend/public/data/fixture_lineup_intelligence/`
- `frontend/public/data/fixture_h2h_support/`
- `frontend/public/data/player_intelligence/`
- `frontend/public/data/team_intelligence/`
- `build/`
- `worker/site-data/`
- `*.bak_*`
- generated `*_audit.csv`, `*_comparison.csv`, `*_manifest.csv`, `*_summary.csv`, `*_report.csv`

If these show dirty, check `.gitignore` and whether the files were already tracked before the ignore rule.

Use `git rm --cached`, not file deletion, when untracking generated files.

## Symptom: World Cup Fixture Missing Context

Check:

- Competition mapping exists.
- World Cup data foundation ran.
- API-football/FootyStats bridge exists for fixture/team names.
- H2H fallback is allowed where history is thin.
- Venue/weather/travel scaffold exists.

Scripts:

- `scripts/build_world_cup_data_foundation.py`
- `scripts/build_world_cup_api_footystats_bridge.py`
- `scripts/build_world_cup_recent_history_and_h2h_sidecar.py`
- `scripts/build_world_cup_venue_weather_travel_scaffold.py`
- `scripts/report_world_cup_2026_coverage_gaps.py`

## Safe Patch Checklist

Before patching:

1. Identify which layer owns the problem.
2. Confirm whether the production spine is involved.
3. Prefer website/reconciler/publish fixes over deploy-rule changes when the issue is display/data flow.
4. Run the smallest relevant smoke check.
5. Commit small, named batches.

After patching:

1. Check protected spine status.
2. Run syntax/compile checks for changed Python/JS.
3. Run site/Worker/readiness smoke where relevant.
4. Record any remaining caveat in docs or final notes.
