# WEBSITE_FULL_HANDOFF_2026-05-12

Updated: `2026-05-12`

## Purpose

This is the full website handoff file for starting a fresh Codex website-build thread with minimal context loss.

It is intended to preserve:

- website architecture
- current product framing
- frontend behavior
- intelligence layers
- backend/Worker/Stripe direction
- Cloudflare/GitHub/Wrangler deployment flow
- publish/data workflows
- key docs and where they live
- what is done
- what is still missing
- what must not be confused with the protected football prediction spine

Read this file first in any fresh website thread.

## Core Product Frame

Odds Genius is **not** a tipster site and should not be built like one.

Canonical framing:

> Odds Genius is a judgement system, not a stats container.

And:

> Odds Genius analyses football fixtures through team strength, squad quality, player roles, lineup structure, historical matchup context and market alignment. Each match is converted into a public-safe decision read showing the signal, the supporting evidence, the key cautions and the strongest aligned markets.

The UI should feel like:

- structured football intelligence
- selective decision support
- premium, calm, analytical

It should not feel like:

- raw CSV dump
- sportsbook chaos
- “here are some picks”

## Critical Repo / System Boundary

### Repo root for website work

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO`

### Git root

- `/Users/hughwade/Documents/Code/OG_master`

### Protected production spine

These are not casual website-edit files:

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

Website work should avoid changing deploy/prediction behavior unless explicitly requested.

## The Website In One View

Architecture:

```text
Local football pipeline
  -> safe publish/export layer
  -> committed/static JSON artifacts
  -> frontend HTML/CSS/JS
  -> GitHub
  -> Cloudflare Pages
  -> Cloudflare Worker for secure/backend routes
  -> Stripe for billing and subscription lifecycle
```

## Current Live / Deploy Surfaces

### Cloudflare Pages

Canonical public production site:

- [https://og-bawa-codex-oct.pages.dev/](https://og-bawa-codex-oct.pages.dev/)

Branch behavior:

- `main` = production
- `dev` = preview

Important:

- Cloudflare Pages only publishes what is committed and pushed
- local code changes do not affect the live site until `main` is updated

### Worker

Secure backend boundary:

- [https://odds-genius-worker.hughcwade.workers.dev](https://odds-genius-worker.hughcwade.workers.dev)
- health:
  - [https://odds-genius-worker.hughcwade.workers.dev/health](https://odds-genius-worker.hughcwade.workers.dev/health)

## Current Frontend File Map

Main frontend pages:

- [frontend/index.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/index.html)
- [frontend/matches.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/matches.html)
- [frontend/live.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/live.html)
- [frontend/results.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/results.html)
- [frontend/competitions.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/competitions.html)
- [frontend/teams.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/teams.html)
- [frontend/dashboard.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/dashboard.html)
- [frontend/premium.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/premium.html)
- [frontend/account.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/account.html)
- [frontend/fixture.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/fixture.html)
- [frontend/pricing.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/pricing.html)
- [frontend/methodology.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/methodology.html)
- [frontend/onboarding.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/onboarding.html)
- [frontend/internal-review.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/internal-review.html)

Core frontend logic:

- [frontend/assets/app.js](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/assets/app.js)
- [frontend/assets/styles.css](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/assets/styles.css)
- [frontend/assets/config.js](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/assets/config.js)

## Frontend Runtime Boundary

The browser must not:

- run model logic
- run deploy logic
- expose raw provider CSVs
- expose formulas
- expose internal debug reasons unless intentionally building a private layer

The browser should:

- read approved publish-safe JSON
- render judgement layers
- degrade gracefully when optional data is missing

## Publish-Safe Data Roots The Frontend Uses

### Existing general site JSON

- `frontend/public/data/public_predictions.json`
- `frontend/public/data/premium_predictions.json`
- `frontend/public/data/publish_summary.json`
- `frontend/public/data/weekly_results.json`
- `frontend/public/data/results_archive.json` (next archive UI direction)
- `frontend/public/data/fixture_intelligence_public.json`
- `frontend/public/data/covered_fixture_universe.json`

### Team intelligence

- `frontend/public/data/team_intelligence/team_ratings_index.json`
- `frontend/public/data/team_intelligence/competitions/<competition_key>__<season_start>/<season_end>.json`

### Player intelligence

- `frontend/public/data/player_intelligence/club_squad_ratings.json`
- `frontend/public/data/player_intelligence/clubs/<competition_key>/<year>/<season>/<club_slug>.json`
- some yearly-only directory shapes also exist for certain leagues

### Fixture lineup intelligence

- `frontend/public/data/fixture_lineup_intelligence/index.json`
- `frontend/public/data/fixture_lineup_intelligence/<fixture_key>.json`

### Fixture H2H support

- `frontend/public/data/fixture_h2h_support/index.json`
- `frontend/public/data/fixture_h2h_support/<fixture_key>.json`

### Fixture decision intelligence

- `frontend/public/data/fixture_decision_intelligence/index.json`
- `frontend/public/data/fixture_decision_intelligence/<fixture_key>.json`

## Intelligence Stack That Now Exists

### 1. Team intelligence

Built and publish-safe.

Ratings include:

- `OG Power Rating`
- `Attack Flow`
- `Defensive Lock`
- `Goal Heat`
- `BTTS Pressure`
- `Over 2.5 Heat`
- `Control Rating`
- `First Strike`
- `Corner Pressure`
- `Card Heat`
- `Chaos Rating`
- `Home Fortress`
- `Away Threat`

Also includes:

- `profile_tags`
- `summary`
- `market_tendencies`
- `home_profile`
- `away_profile`
- `timing_profile`
- `rating_explanations`

Engine / spec:

- [team_rating_engine.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/team_rating_engine.py)
- [docs/TEAM_PLAYER_RATINGS_ENGINE_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/TEAM_PLAYER_RATINGS_ENGINE_SPEC.md)

### 2. Player / squad intelligence

Built and publish-safe.

Ratings include:

- `OG Player Power`
- `Goal Threat`
- `Creative Spark`
- `Midfield Engine`
- `Defensive Lock`
- `Pressing Heat`
- `Ball Progression`
- `Aerial Dominance`
- `Discipline Risk`
- `Booking Heat`
- `Goalkeeper Shield`

Also includes:

- leaders
- ranks
- role mix
- featured players
- player UI summary fields

Engine / spec:

- [player_rating_engine.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/player_rating_engine.py)
- [docs/TEAM_PLAYER_RATINGS_ENGINE_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/TEAM_PLAYER_RATINGS_ENGINE_SPEC.md)

### 3. Fixture lineup intelligence

Built and publish-safe.

Outputs include:

- `home_units`
- `away_units`
- `home_lineup_profiles`
- `away_lineup_profiles`
- `player_matchups`
- `key_mismatches`

Unit families include:

- `Attack Unit`
- `Midfield Control`
- `Defensive Unit`
- `Wide Threat`
- `Central Threat`
- `Discipline Risk`

Engine:

- [fixture_lineup_intelligence_engine.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/fixture_lineup_intelligence_engine.py)

### 4. Fixture H2H support

Built and publish-safe as a supporting layer.

Not the same thing as power ratings.

It is a separate matchup-history/context layer.

Outputs include:

- `sample_size`
- `goal_environment`
- `btts_regime`
- `booking_heat`
- `foul_intensity`
- `style_conflict_index`
- `home_win_rate`
- `draw_rate`
- `over25_rate`
- `summary`

Engine:

- [fixture_h2h_support_engine.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/fixture_h2h_support_engine.py)

### 5. Fixture decision intelligence

This is the central judgement layer.

Outputs include:

- `primary_signal`
- `signal_state`
- `agreement_score`
- `confidence_band`
- `supporting_layers`
- `caution_layers`
- `profile_tags`
- `profile_narrative`
- `team_faceoff_summary`
- `unit_battle_summary`
- `key_player_drivers`
- `key_mismatches`
- `h2h_context`
- `market_suitability`
- `market_intelligence`
- `watchlist`
- `public_safe_summary`
- `internal_reason_tokens`
- `preview`

Engine / specs:

- [fixture_decision_reconciler.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/fixture_decision_reconciler.py)
- [fixture_preview_generator.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/fixture_preview_generator.py)
- [docs/FIXTURE_DECISION_RECONCILER_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/FIXTURE_DECISION_RECONCILER_SPEC.md)
- [docs/AGREEMENT_SCORE_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/AGREEMENT_SCORE_SPEC.md)
- [docs/PUBLIC_EXPLANATION_STYLE_GUIDE.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/PUBLIC_EXPLANATION_STYLE_GUIDE.md)
- [docs/intelligence_schema.json](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/intelligence_schema.json)

## Canonical Product Hierarchy

This is the intended product spine:

1. `Team Intelligence`
2. `Squad / Player Intelligence`
3. `Fixture Intelligence`
4. `Lineup Intelligence`
5. `Unit Intelligence`
6. `Player Matchups`
7. `H2H / Context`
8. `Market Decision Layer`

This is the company/product story too.

## Page-Purpose Architecture

### Teams page

Question answered:

- `How strong is this team?`
- `Which players create that strength?`

Belongs on Teams:

- team headline ratings
- team summary
- profile tags
- timing profile
- home/away profile
- market tendencies
- squad leaders
- role mix
- featured players

Does not belong on Teams:

- full deploy verdict framing
- aggressive market picks as the main story
- raw provider stat dumps

### Fixture page

Question answered:

- `What is the signal today?`
- `Do the structural layers support it?`

Belongs on Fixtures:

- top verdict
- agreement stack
- caution framing
- team face-off
- unit battle
- player drivers
- market suitability
- lineup/formation layer
- H2H as supporting context

Does not belong on Fixtures:

- raw CSV style tables as the main event
- every rating dumped without purpose
- hidden formulas

### H2H

Question answered:

- `Does historical matchup context support the current read?`

H2H is supporting-only, not lead content.

## Current Frontend Behavior By Surface

### Teams

Current site can consume:

- `team_intelligence`
- `player_intelligence`

Key behavior:

- team overview + intelligence tab can render ratings and team summaries
- squad/player modules can render leaders, snapshots, and featured players

### Fixtures

Current site can consume:

- `fixture_decision_intelligence`
- `fixture_lineup_intelligence`
- `fixture_h2h_support`
- `team_intelligence` / `player_intelligence` fallbacks

Important behavior:

- fixture hero is now reconciler-led
- H2H/decision surface is not assembled ad hoc anymore
- lineup/player layers degrade gracefully when not published
- player drivers can fall back to squad intelligence if lineup intelligence is absent

## Active-Site Publish Estate State As Of 2026-05-12

This is the currently important shipped state after the latest completion pass.

### Active competitions

The live site publish estate is trimmed to the `22` active site competitions and latest seasons.

### Current fixture window

- `156` current fixture keys

### Coverage counts

- `388` active team rows
- `388` active club squad rows
- `156` decision payloads
- `156` lineup payloads
- `156` H2H payloads

### Important truth

That does **not** mean all upstream lineup/H2H layers are truly populated.

Actual current-window reality:

- lineup real coverage remains partial
- H2H direct current coverage remains partial

So the publish layer now emits placeholders where needed, rather than leaving gaps.

Reference report:

- [docs/ACTIVE_SITE_INTELLIGENCE_PUBLISH_ESTATE_COMPLETION_2026-05-12.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/ACTIVE_SITE_INTELLIGENCE_PUBLISH_ESTATE_COMPLETION_2026-05-12.md)

## Preview Object / Natural Language Layer

Built server-side:

- [fixture_preview_generator.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/fixture_preview_generator.py)

The `preview` object includes:

- `headline`
- `short_summary`
- `market_summary`
- `caution_line`
- `telegram_summary`
- `premium_summary`

Rules:

- deterministic templates first
- no raw provider stats
- no formulas
- no in-browser LLM generation

Future optional enhancement:

- cheap server-side GPT generation for summaries/Telegram copy only
- never in-browser
- derived from already-safe fixture decision objects

## Worker / Backend State

### Worker files

- [worker/src/index.js](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/src/index.js)
- [worker/src/auth.js](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/src/auth.js)
- [worker/src/subscriber_store.js](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/src/subscriber_store.js)
- [worker/src/account_store.js](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/src/account_store.js)
- [worker/wrangler.toml](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/wrangler.toml)
- [worker/wrangler.example.toml](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/wrangler.example.toml)
- [worker/README_WORKER.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/README_WORKER.md)
- [worker/DEPLOY_WORKER.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/DEPLOY_WORKER.md)

### Worker current responsibility

The Worker is the secure backend boundary for:

- Stripe checkout
- Stripe webhook verification
- subscriber-state persistence
- premium token / session flow
- protected premium prediction delivery
- account state
- Telegram link flow
- some widget/reference routes

### Important Worker routes

- `GET /health`
- `GET /api/account/state`
- `GET /api/account/alerts`
- `POST /api/account/preferences`
- `POST /api/account/alerts/refresh`
- `POST /api/account/alerts/dispatch`
- `POST /api/account/telegram/link/start`
- `POST /api/account/telegram/link/complete`
- `POST /api/account/telegram/test-alert`
- `POST /api/account/telegram/fixture-alert`
- `GET /api/widgets/football/standings`
- `GET /api/widgets/football/fixture-lookup`
- `POST /api/telegram/webhook`
- `POST /api/stripe/checkout`
- `POST /api/premium/token`
- `POST /api/stripe/portal`
- `POST /api/stripe/webhook`
- `GET /api/premium/predictions`
- auth magic-link/session routes

### Worker storage / bindings

Current or planned:

- `SUBSCRIBER_STATE` KV
- `ACCOUNT_DB` D1 (optional but increasingly real)

Migrations exist under:

- `worker/migrations/`

### Wrangler / local runtime

Worker package:

- [worker/package.json](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/package.json)

Run local harness:

```bash
node worker/test_worker_local.js
```

## Stripe State

Stripe is partially wired, not fully finished.

Important doc:

- [STRIPE_SETUP.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/STRIPE_SETUP.md)

Planned offer:

- Founding Member Plan
- `£20/month`

Current route direction:

- `POST /api/stripe/checkout`
- `POST /api/stripe/portal`
- `POST /api/stripe/webhook`

Never commit:

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`

## Cloudflare / GitHub / Deployment Flow

Primary docs:

- [CLOUDFLARE_PAGES.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/CLOUDFLARE_PAGES.md)
- [WEBSITE_GIT_CLOUDFLARE_MASTER_RUNBOOK.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/WEBSITE_GIT_CLOUDFLARE_MASTER_RUNBOOK.md)
- [DEPLOYMENT_RUNBOOK.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/DEPLOYMENT_RUNBOOK.md)

Canonical rules:

- `main` = production Pages branch
- `dev` = preview Pages branch
- no build step for static frontend v1
- publish directory = `frontend`

Pages only updates when:

- JSON is regenerated
- files are committed
- branch is pushed

## Core Publish / Validation Workflow

### General website publish

Run:

```bash
python3 publish_predictions.py
python3 validate_public_export.py
python3 validate_fixture_intelligence.py
python3 validate_covered_fixture_universe.py
python3 scripts/smoke_frontend_static.py
```

### Team / player / fixture intelligence publish

Important scripts:

- [team_rating_engine.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/team_rating_engine.py)
- [player_rating_engine.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/player_rating_engine.py)
- [fixture_lineup_intelligence_engine.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/fixture_lineup_intelligence_engine.py)
- [fixture_h2h_support_engine.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/fixture_h2h_support_engine.py)
- [fixture_decision_reconciler.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/fixture_decision_reconciler.py)
- [fixture_preview_generator.py](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/fixture_preview_generator.py)

Config:

- [ratings_publish_sources.json](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/ratings_publish_sources.json)
- [ratings_publish_sources.sample.json](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/ratings_publish_sources.sample.json)

## High-Value Markdown Docs To Know

### First read

1. [AGENTS.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/AGENTS.md)
2. [docs/ACTIVE_SITE_INTELLIGENCE_PUBLISH_ESTATE_COMPLETION_2026-05-12.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/ACTIVE_SITE_INTELLIGENCE_PUBLISH_ESTATE_COMPLETION_2026-05-12.md)
3. [README_FRONTEND.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/README_FRONTEND.md)
4. [WEBSITE_GIT_CLOUDFLARE_MASTER_RUNBOOK.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/WEBSITE_GIT_CLOUDFLARE_MASTER_RUNBOOK.md)
5. [docs/TEAM_PLAYER_RATINGS_ENGINE_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/TEAM_PLAYER_RATINGS_ENGINE_SPEC.md)
6. [docs/FIXTURE_DECISION_RECONCILER_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/FIXTURE_DECISION_RECONCILER_SPEC.md)
7. [docs/intelligence_schema.json](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/intelligence_schema.json)

### Frontend / architecture / UX

- [docs/NAVIGATION_AND_PAGE_ARCHITECTURE_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/NAVIGATION_AND_PAGE_ARCHITECTURE_SPEC.md)
- [docs/SOFASCORE_INSPIRED_INFORMATION_ARCHITECTURE_PLAN.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/SOFASCORE_INSPIRED_INFORMATION_ARCHITECTURE_PLAN.md)
- [docs/CLARITY_REPLACING_CHAOS_BRAND_SYSTEM.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/CLARITY_REPLACING_CHAOS_BRAND_SYSTEM.md)
- [docs/CLARITY_UX_APPLICATION_PLAN.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/CLARITY_UX_APPLICATION_PLAN.md)
- [docs/WEBSITE_REMAINING_WORKSTACK.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/WEBSITE_REMAINING_WORKSTACK.md)
- [docs/WEBSITE_10K_USER_ARCHITECTURE_PLAN.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/WEBSITE_10K_USER_ARCHITECTURE_PLAN.md)

### Intelligence / ratings / decision system

- [docs/TEAM_PLAYER_RATINGS_ENGINE_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/TEAM_PLAYER_RATINGS_ENGINE_SPEC.md)
- [docs/FIXTURE_DECISION_RECONCILER_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/FIXTURE_DECISION_RECONCILER_SPEC.md)
- [docs/AGREEMENT_SCORE_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/AGREEMENT_SCORE_SPEC.md)
- [docs/PUBLIC_EXPLANATION_STYLE_GUIDE.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/PUBLIC_EXPLANATION_STYLE_GUIDE.md)
- [docs/REASON_TOKENS.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/REASON_TOKENS.md)
- [docs/FIXTURE_INTELLIGENCE_SCHEMA_SPEC.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/FIXTURE_INTELLIGENCE_SCHEMA_SPEC.md)

### Worker / auth / account / backend

- [WORKER_BACKEND_PLAN.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/WORKER_BACKEND_PLAN.md)
- [worker/README_WORKER.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/README_WORKER.md)
- [worker/DEPLOY_WORKER.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/DEPLOY_WORKER.md)
- [STRIPE_SETUP.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/STRIPE_SETUP.md)
- [docs/REAL_AUTH_MAGIC_LINK_PLAN.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/REAL_AUTH_MAGIC_LINK_PLAN.md)
- [docs/SESSION_REGISTRY_SCHEMA_AND_WORKER_CONTRACT.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/SESSION_REGISTRY_SCHEMA_AND_WORKER_CONTRACT.md)
- [docs/D1_USER_AND_TELEGRAM_LINKING_SCHEMA.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/D1_USER_AND_TELEGRAM_LINKING_SCHEMA.md)

### Operational deployment

- [CLOUDFLARE_PAGES.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/CLOUDFLARE_PAGES.md)
- [CLOUDFLARE_PAGES_CONNECT_STEPS.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/CLOUDFLARE_PAGES_CONNECT_STEPS.md)
- [DEPLOYMENT_RUNBOOK.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/DEPLOYMENT_RUNBOOK.md)
- [LAUNCH_PREDEPLOY_CHECKLIST.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/LAUNCH_PREDEPLOY_CHECKLIST.md)

### System maps / doc indexes

- [docs/SYSTEM_MAP.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/SYSTEM_MAP.md)
- [docs/CURRENT_SYSTEM_INTEGRAL_FILE_MAP_2026-05-12.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/CURRENT_SYSTEM_INTEGRAL_FILE_MAP_2026-05-12.md)
- [ROOT_DOC_INDEX.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/ROOT_DOC_INDEX.md)
- [docs/ROOT_MARKDOWN_INDEX_SUMMARY_2026-05-12.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/ROOT_MARKDOWN_INDEX_SUMMARY_2026-05-12.md)

## Current Known Truths / Caveats

### 1. Intelligence is built, but some layers still rely on fallback logic

The system no longer “breaks” when optional layers are missing, but not every fixture has rich upstream lineup or H2H data.

### 2. Placeholder coverage is intentional

If lineup/H2H upstream data is absent, the site should say so cleanly and fall back to:

- team intelligence
- squad/player intelligence
- decision intelligence

### 3. Repo cleanup is not required for website progress

The repo is messy, but website build work can continue without full repo reorganization.

### 4. Production updates require `main`

If the site at `og-bawa-codex-oct.pages.dev` does not change, check:

- whether data/code were committed
- whether `main` was updated
- whether Cloudflare Pages finished rebuilding

## Current Build Priorities

### Immediate next website work

1. make placeholder lineup/H2H states feel deliberate
2. make player-driver fallback clearer on fixture pages
3. make best / secondary / weak / avoid market intelligence more visible
4. improve first-view prominence of intelligence on team and fixture pages

### Next after that

1. richer preview consumption for Telegram/comms/export
2. stronger WATCHLIST product treatment
3. premium/free tier separation across surfaces
4. richer formation interactions:
   - click states
   - deeper hover cards
   - lane overlays

### Later

1. post-match audit layer
2. expected vs actual structure
3. caution accuracy
4. model won/lost explanation
5. B2B / API exposure of deeper layers

## How To Start A Fresh Codex Website Thread

In a fresh thread, the first instruction should roughly be:

1. read `AGENTS.md`
2. read this handoff file
3. read the publish-estate completion report
4. inspect `frontend/assets/app.js`, `frontend/assets/styles.css`, and the publish data roots
5. do not reopen repo-tidy work unless explicitly requested
6. continue website productization, not prediction-pipeline rewrites

Suggested first local verification steps:

```bash
python3 scripts/smoke_frontend_static.py
cd frontend && python3 -m http.server 8000
```

Then inspect:

- `teams.html`
- `fixture.html`
- current published `fixture_decision_intelligence`
- current published `fixture_lineup_intelligence`
- current published `fixture_h2h_support`

## Final Principle

Do not let the website become “more data.”

The value is not:

- raw goals
- raw xG
- raw provider tables

The value is:

- structural judgement
- support vs caution
- clear reasoning
- graceful handling of uncertainty

The site should always feel like:

- a football intelligence terminal

not:

- a spreadsheet with branding

