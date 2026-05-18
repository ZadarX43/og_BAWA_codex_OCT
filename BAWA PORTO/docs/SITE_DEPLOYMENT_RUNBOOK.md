# Site Deployment Runbook

Updated: `2026-05-18`

## Purpose

Prepare the Odds Genius website for the World Cup + pre-season Founder Early Access launch without changing model generation, deploy routing, or the protected football prediction spine.

Launch framing:

> OG Founder Early Access: Football Prediction Intelligence System, World Cup + Pre-Season Edition.

## Hard Boundaries

- Do not edit `deploy_rulebook.py` for website polish.
- Do not edit `bookie_allmarkets.py` for website copy or tier work.
- Do not edit `slip_formatter.py` to rescue product gaps.
- Do not publish `OBSERVE` rows as deployable results.
- Do not let value overlays override hard deploy gates.
- Weather and player-event cards remain context or beta intelligence until explicitly promoted.

## Launch Checklist

### 1. Public Proof

- Publish settled rows with `result_status`: `pending`, `won`, `lost`, or `void`.
- Keep FTR, BTTS, OU25, and TG1.5 metrics separate.
- Show pending rows separately from settled rows.
- Render winner, loser, pending, and void states visibly on `results.html`.
- Preserve deploy and observe/research summaries as separate proof layers.

Settlement command:

```bash
python3 scripts/publish_results_proof.py
```

Expected outputs:

- `frontend/public/data/weekly_results.json`
- `frontend/public/data/results_archive.json`
- `reports/latest/RESULTS_SETTLEMENT_REPORT.md`
- `reports/latest/RESULTS_PAGE_SMOKE_REPORT.md`
- `reports/latest/RESULTS_PUBLISH_RUN_REPORT.md`

This command runs the settlement script and then smoke-checks the public Results page contract. Use the lower-level command only when debugging settlement itself:

```bash
python3 scripts/settle_published_results.py
```

### 2. Homepage

- First viewport should say `Sports Prediction Intelligence System` and `Football v0.12`.
- Show launch model families: FTR, BTTS, Over 2.5, TG1.5, goal-market combos.
- Show player-event beta families: shots, shots on target, tackles, fouls, player fouled, key passes, goalkeeper saves, corners, bookings.
- Link quickly to matches, results, and Founder Early Access pricing.

### 3. Fixture Pages

- Under each fixture hero, show compact cards for FTR, OU25, BTTS, and TG1.5.
- Each card should include model/support percentage where publish-safe, bookmaker odds where available, lean/state, and caution state.
- Tabs own their domains: Prediction for the read, Markets for ranked market posture, Lineups for squad state, H2H for history, News for external signals.
- Missing data should render as pending or unavailable, never as empty broken UI.

### 4. Tier Visibility

- Free: limited public board, public proof, public methodology.
- Founder: discounted early access to core premium fixture intelligence, market cards, fixture reads, results archive, and selected beta previews.
- Premium: standard paid version of core premium fixture intelligence after Founder closes.
- Pro: player-event intelligence cards, deeper team/player intelligence, combos, and expert panels.
- Pro+: audit dashboards, advanced filters, downloadable intelligence, and operational coverage views.
- Syndicate and B2B/API remain future products.
- Field-level allowlists are documented in `docs/PREMIUM_PAYLOAD_ALLOWLIST.md` and enforced by the Worker premium route.

### 5. Account And Checkout

- Smoke-test Stripe checkout.
- Confirm post-checkout redirect.
- Confirm session restore after reload.
- Confirm account states: signed out, free, founder, premium, expired, payment issue.
- Confirm billing self-service route.
- Confirm premium route gating and upgrade prompts.

Worker smoke command:

```bash
node worker/test_worker_local.js
```

This harness covers Stripe Checkout, billing portal creation, session restore, premium route gating, inactive/payment-issue lockout, and tiered payload allowlists.

### 6. Freshness Metadata

- Show `generated_at` or equivalent publish timestamp.
- Show next refresh where available.
- Show data coverage status where available.
- Confirm stale or missing data degrades calmly.

Publish smoke command:

```bash
python3 scripts/site_publish_smoke.py
```

Expected report:

- `reports/latest/SITE_PUBLISH_SMOKE_REPORT.md`

Cloudflare preview readiness command:

```bash
python3 scripts/cloudflare_preview_readiness.py
```

Expected report:

- `reports/latest/CLOUDFLARE_PREVIEW_READINESS_REPORT.md`

This command checks the local proof/data bundle, Worker config, live Worker health/env/routes, signed-out premium/portal locks, Stripe test Checkout session creation, public Pages routes, public proof JSON reachability, and whether preview `weekly_results.json` matches the local bundle.

## Website Data Footprint

Measured locally on `2026-05-18` with `du -sh frontend/public/data frontend/public/data/*`.

Current total: approximately `118M`.

Largest payloads:

- `frontend/public/data/player_intelligence`: `56M`
- `frontend/public/data/team_intelligence`: `55M`
- `frontend/public/data/api_football_logo_asset_manifest.json`: `1.8M`
- `frontend/public/data/fixture_decision_intelligence`: `1.7M`
- `frontend/public/data/fixture_lineup_intelligence`: `728K`
- `frontend/public/data/fixture_h2h_support`: `704K`
- `frontend/public/data/fixture_intelligence_public.json`: `420K`

Decision point:

- The current footprint is manageable for static hosting, but player/team intelligence dominate payload size.
- Before expanding current-season payload coverage, audit whether Cloudflare Pages and any paid tier limits cover the intended monthly bandwidth and file count.
- If the £20/month Cloudflare data tier removes practical pressure around current-season payload hosting, it is likely worth it for launch reliability.

## Smoke-Test Flow

One-command local launch smoke:

```bash
python3 scripts/run_launch_smoke.py
```

Expected report:

- `reports/latest/LAUNCH_SMOKE_REPORT.md`

Run locally from the repo root:

```bash
python3 -m http.server 4173 --directory frontend
```

Then check:

- `http://localhost:4173/index.html`
- `http://localhost:4173/results.html`
- `http://localhost:4173/matches.html`
- one fixture page with `fixture.html?fixture=<fixture_key>`
- `http://localhost:4173/pricing.html`
- `http://localhost:4173/account.html`

Minimum checks:

- Homepage first viewport explains what the product is, what it predicts, why to trust it, and what can be bought.
- Results page shows won/lost/pending/void states with clear visual treatment.
- Results market split does not blend FTR, BTTS, OU25, and TG1.5.
- Fixture page shows FTR, OU25, BTTS, and TG1.5 market cards under the hero.
- Pricing page shows Founder, Premium, Pro, and Pro+ visibility rules.
- Mobile viewport has no clipped primary content or unusable tap targets.

## Publish Order

1. Confirm data integrity has passed.
2. Confirm predictions and deploy outputs were produced through the protected pipeline.
3. Export website-safe JSON only.
4. Run `python3 scripts/publish_results_proof.py` after fixtures finish.
5. Rebuild or refresh static frontend data.
6. Run `python3 scripts/run_launch_smoke.py`.
7. Run local browser smoke checks.
8. Review `reports/latest/LAUNCH_SMOKE_REPORT.md`.
9. Push to preview branch.
10. Let Cloudflare Pages redeploy preview.
11. Run `python3 scripts/cloudflare_preview_readiness.py`.
12. Promote to production only after checkout/account/results/preview readiness checks pass.

Promotion blockers:

- Worker `/health` missing required env or routes.
- Signed-out premium or billing portal routes fail to lock.
- Stripe test Checkout route cannot create a test session.
- Public Pages routes or required JSON files are unreachable.
- Preview `weekly_results.json` does not match the local settled proof bundle.

Warnings, not blockers:

- TG1.5 has no current weekly rollup while odds/support remain pending.
- Data footprint is below hard limits but should still be monitored before expanding player/team payloads.

## Open Launch Gaps

- Official/team news ingestion path.
- Cloudflare preview smoke after Worker env vars are wired.
- Automation scheduler wiring for settlement, export, and post-publish smoke checks.
