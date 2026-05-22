# Website Launch Plan - 2026-06-01

## Purpose

This document defines what remains before the Odds Genius website can be launched publicly on `2026-06-01`.

It separates:

- already built product foundations
- launch-critical gaps
- pre-launch data and proof work
- premium tier readiness
- deferred post-launch work

It does not change the protected football prediction spine.

## Current Website State

The website is no longer a prototype page set. It already has:

- public homepage
- matches / live / predictions / results pages
- competitions and teams pages
- fixture intelligence page
- premium, pricing, account, onboarding, methodology, dashboard pages
- public and premium prediction JSON feeds
- live results feed JSON
- weekly results and results archive surfaces
- fixture decision intelligence payloads
- fixture lineup intelligence payloads
- fixture H2H support payloads
- team intelligence payloads
- player intelligence payloads
- external content payloads for team news and fixture media
- weather and space-weather UI prototype work
- Cloudflare Pages frontend deployment path
- Cloudflare Worker backend boundary
- Stripe / auth / premium gate foundations

The launch problem is not “build the website from scratch.”

The launch problem is:

- make the proof layer credible
- make data payloads controlled
- make the premium intelligence contract consistent
- make account and payment paths clean
- make fixture pages composed and deduplicated
- make the operating runbooks clear enough to repeat every weekend

## Launch-Critical Workstreams

## 1. Public Proof And Results

Status: partially built.

Already exists:

- `frontend/public/data/live_results_feed.json`
- `frontend/public/data/weekly_results.json`
- `frontend/public/data/results_archive.json`
- `frontend/results.html`
- `docs/PUBLIC_RESULTS_FEED_BUILD_NOTES_2026-05-14.md`
- `docs/RESULTS_SETTLEMENT_AND_ARCHIVE_PLAN.md`

Still needed before launch:

- automated grading from published picks to settled results
- stable win/loss/void fields per pick
- public hit-rate rollups by market, tier, league, and week
- clear pending vs settled state
- no manual editing of proof metrics
- visible recent-week performance feed
- launch-safe visual treatment for winners and losers

Launch decision:

- Results page must be credible enough for public users before paid launch.
- This is the highest-priority launch workstream.

## 2. Prediction Publish Discipline

Status: mostly built, needs one clean launch wrapper.

Already exists:

- public prediction board
- premium prediction board
- deploy output to website-safe JSON
- weekend summary pack:
  - `docs/weekend_prediction_summary_pack_2026_05_15_to_2026_05_19`
- stable labels:
  - `FULL_DEPLOY_BOARD_MAY14_19`
  - `FORWARD_DEPLOY_BOARD_MAY15_19`
  - `INTEL_SUPPORTED_DEPLOY_SUBSET_MAY15_19`
  - `TG15_CS_MASS_SUPPORT_REVIEW_MAY15_19`
  - `OBSERVE_RESEARCH_ONLY`

Still needed before launch:

- one repeatable command/runbook for:
  - refresh data
  - run predictions
  - run deploy
  - run site-safe export
  - run result smoke checks
  - publish to preview/prod
- board freshness metadata visible on site
- explicit “last updated” and “next refresh” text
- public copy that explains ELITE / STANDARD / OBSERVE without overpromising

Launch decision:

- `OBSERVE` can be shown only as watch/research context, never as a deploy result.
- Public page should show fewer, clearer picks rather than full internal board density.

## 3. Premium Intelligence Contract

Status: data contract exists, route/product consistency still needs tightening.

Already exists:

- `site_fixture_market_intelligence`
- `site_player_event_shortlists`
- `site_player_identity_map`
- `site_player_match_stats`
- `site_team_match_stats`
- `site_lineup_slots`
- `site_formation_slots`
- Barcelona vs Real Madrid full demo fixture
- player/team/fixture intelligence docs and exporters

Still needed before launch:

- decide exactly what £20, £49, £99, and £500 see
- hide pro/audit metadata from public and lower tiers
- render player-event shortlists as beta/intelligence cards, not priced prop tips
- expose TG1.5 / goal-combo research only when labelled correctly
- make Fixture Stats / Markets / Lineups pages consume the premium payloads consistently
- ensure missing payloads degrade gracefully

Launch tier split:

- £20: core fixture read, top market posture, predicted lineups, public proof.
- £49: player-event shortlists, event cards, richer fixture intelligence.
- £99: deeper player/team stat profiles, combo markets, goal-shape reasoning.
- £500: pro/audit metadata, alert routing, operational state, downloadable route payloads.

Launch decision:

- Premium product is not “more picks.”
- Premium product is the intelligence behind the read.

## 4. Fixture Page Composition

Status: strong, but still needs deduplication/polish.

Already improved:

- Barca vs Real Madrid demo fixture
- YouTube highlights embed
- weather card
- space-weather card
- fixture hero
- market/lineup/context intelligence layers

Still needed before launch:

- remove duplicate chip wallpaper across tabs
- keep Form tab focused on team form, not league-context repetition
- Markets tab should own full ranked market posture
- Prediction tab should avoid repeating Markets content
- Lineups tab should avoid redundant provider lineup tables where pitch/bench already show the same players
- ensure weather badge alignment and card balance are final
- verify all key tabs on desktop and mobile

Launch decision:

- Demo fixture should be the visual standard for the product.
- It needs to feel like a finished research surface, not stacked internal notes.

## 5. External Content

Status: initial payloads exist, ingestion still early.

Already exists:

- fixture media payload for Barca vs Real Madrid
- team news payloads for Barcelona and Real Madrid
- external content source registry
- YouTube embed proof of concept
- RSS/source planning docs

Still needed before launch:

- robust RSS ingestion into `news_signals`
- team-tag matching for club news
- source attribution and links
- News tab on fixture page
- News tab on team page
- no article scraping/copying beyond safe headline/link/summary treatment

Launch decision:

- External content should be an intelligence signal feed, not a news dump.
- Use external sources as raw inputs; Odds Genius supplies interpretation.

## 6. Weather And Environmental Context

Status: demo/weather cards built, launch data coverage incomplete.

Already exists:

- weather card design
- condition badge vocabulary
- weather context docs
- space weather experimental presentation

Still needed before launch:

- weather payloads for all active fixture venues
- compact weather chip in fixture hero metadata
- weather data freshness fields
- clear “soft context only” wording
- avoid making environmental layers look like hard prediction drivers

Launch decision:

- Weather is a differentiator, but must stay credible and soft-scored.

## 7. Player Event Predictions

Status: beta/research.

Already exists:

- player-event data contract
- player-event shortlist exporter
- hit-rate band board
- interaction shadow board
- weekend player-event plan

Still needed before launch:

- current lineup/squad refresh discipline
- pre-kickoff snapshot fields on payloads
- market-by-market player-event cards:
  - shots on target
  - shots
  - tackles
  - fouls
  - player fouled
  - bookings
  - key passes
  - goalkeeper saves
- avoid public prop language unless odds/prices are available

Launch decision:

- Player events are a £49/£99 product hook, but should launch as “shortlists” and “event intelligence,” not final priced betting picks.

## 8. Account, Billing, And Auth

Status: foundations built, polish still needed.

Already exists:

- Stripe Checkout
- Worker premium gate
- magic-link auth
- session-cookie premium unlock
- account page foundations
- pricing page

Still needed before launch:

- billing self-service path
- subscription management UX
- clean post-checkout redirect
- clean auth/session banners
- account state clarity:
  - signed out
  - free
  - active premium
  - expired
  - payment issue
- production smoke test of paywall route

Launch decision:

- Do not launch paid traffic until checkout, unlock, session restore, and account messaging are boringly reliable.

## 9. Data Footprint And Hosting

Status: measured; needs active policy.

Current public data audit:

- public data files: `13,576`
- JSON files: `13,570`
- size: `107.3 MB`
- largest groups:
  - player intelligence: `56.0 MB`
  - team intelligence: `46.7 MB`
  - fixture decision intelligence: `1.3 MB`

Premium SQLite/D1 measured slice:

- with premium market/event exporter: about `155 MB`
- D1 SQL chunks: about `144 MB`
- schema version: `4`

Still needed before launch:

- active-season/current-competition restriction policy
- no accidental historical-data publishing
- route payload caching policy
- static-vs-D1 split
- cost estimate for:
  - one demo fixture fully populated
  - one weekend fully populated
  - a normal month of operation

Launch decision:

- Launch should use compact current-season active-competition payloads.
- Deep historical/pro export remains later-tier work.

## 10. Deployment And Operations

Status: deployment path known; promotion discipline needs logging.

Known production behavior:

- Cloudflare Pages production is `main`
- `dev` is preview
- local changes do not affect production until committed, pushed, and promoted to `main`

Still needed before launch:

- write one `SITE_DEPLOYMENT_RUNBOOK.md`
- define preview smoke checks
- define production promotion checks
- define rollback approach
- confirm Cloudflare Pages project settings
- confirm Worker environment bindings
- confirm production/preview data roots

Launch decision:

- No more ambiguous “if production watches dev/main” wording. The docs state `main = production`, `dev = preview`.

## Pre-Launch Priority Order

## Week 1: 2026-05-15 to 2026-05-19

1. Settle and publish weekend/MLS results into the website proof feed.
2. Run data audit on current prediction and premium payload size.
3. Run player-event predictions for Bundesliga/EPL weekend test windows.
4. Capture Sportsmole previews and score against OG goal mass.
5. Continue fixture page deduplication and results-feed UI polish.

## Week 2: 2026-05-20 to 2026-05-26

1. Automate grading pipeline and archive outputs.
2. Finalize launch-tier visibility rules.
3. Polish pricing, account, checkout, and premium unlock UX.
4. Build/validate team and fixture news tabs.
5. Run site data footprint audit after all payloads are included.

## Launch Week: 2026-05-27 to 2026-06-01

1. Freeze launch data policy.
2. Smoke test all pages on preview.
3. Test checkout/account/premium unlock.
4. Test Results page and archive rollups.
5. Test fixture page demo and active fixtures.
6. Promote `dev` to `main`.
7. Verify Cloudflare production site.

## Launch Blockers

These must be resolved before paid launch:

- results grading must be repeatable and not manually fabricated
- production deploy path must be documented and verified
- account/checkout/premium unlock must work cleanly
- pricing tiers must match what the UI actually exposes
- fixture pages must not leak internal audit metadata to public tiers
- public proof page must clearly separate pending and settled picks

## Non-Blockers For June 1

These can ship after launch:

- full mobile app
- push notifications
- full Telegram account linking
- deep historical archives
- B2B/API tier
- fully priced player-prop markets
- full live reactive in-play system

## Next Immediate Action

The next build should be:

1. data audit on currently published picks and payloads
2. public results settlement pipeline hardening
3. player-event weekend run
4. website results/premium fixture UI polish

This order keeps launch tied to proof, trust, and paid-product clarity rather than adding more raw content.
