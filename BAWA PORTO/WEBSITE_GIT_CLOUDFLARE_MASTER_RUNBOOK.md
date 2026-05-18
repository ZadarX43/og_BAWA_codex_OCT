# WEBSITE_GIT_CLOUDFLARE_MASTER_RUNBOOK

Updated: `2026-05-04`

## Purpose

This is the single master handover document for the Odds Genius website layer.

It explains:

- what the live website is
- what is static vs secure
- how Git is set up
- how Cloudflare Pages is set up
- how the Cloudflare Worker is set up
- how Stripe fits in
- what has already been proven live
- what must never be committed
- how to resume work safely later

This document is intentionally broader than the smaller focused docs.
It is the "one file to read first" reference.

## Current Live Surfaces

### Static frontend

- Production Pages URL: [https://og-bawa-codex-oct.pages.dev](https://og-bawa-codex-oct.pages.dev)

Canonical live pages:

- `/`
- `/predictions`
- `/premium`
- `/results`
- `/pricing`
- `/methodology`
- `/account`

Static data currently served from Pages:

- `/public/data/public_predictions.json`
- `/public/data/premium_predictions.json`
- `/public/data/publish_summary.json`
- `/public/data/weekly_results.json`

### Secure backend

- Worker URL: [https://odds-genius-worker.hughcwade.workers.dev](https://odds-genius-worker.hughcwade.workers.dev)
- Health URL: [https://odds-genius-worker.hughcwade.workers.dev/health](https://odds-genius-worker.hughcwade.workers.dev/health)

## What The Product Is

Odds Genius is not positioned as a tipster site.

The current product framing is:

- evidence-first football market intelligence
- pricing inefficiency detection
- selective deployment only when model edge survives structural, volatility, and stability checks

Core positioning line:

> Odds Genius is not built to predict every game. It is built to identify when the market is wrong.

## Architecture In One View

```text
Local model engine
  -> approved JSON export layer
  -> committed static frontend data
  -> GitHub repo
  -> Cloudflare Pages
  -> Cloudflare Worker for secure premium and billing routes
  -> Stripe for checkout + webhook subscription events
```

## Non-Negotiable Boundaries

### Website

- The frontend does not run model logic.
- The frontend only reads approved exported JSON.
- The frontend must not recreate deployment logic.
- Premium must stay locked by default unless the Worker verifies access.

### Security

- Never commit secrets.
- Never put Stripe secrets in frontend files.
- Never put Worker secrets in markdown docs.
- Never describe static premium JSON as the final secure architecture.

### Production pipeline

Per `AGENTS.md`, do not casually change the protected production spine or model logic while doing website work.

## Repository Structure That Matters For The Website

Repo root for the web product work:

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO`

Git root:

- `/Users/hughwade/Documents/Code/OG_master`

Important website paths:

- `frontend/`
- `frontend/assets/`
- `frontend/public/data/`
- `.github/workflows/`
- `worker/`

Important export / validation scripts:

- `publish_predictions.py`
- `validate_public_export.py`
- `grade_weekend_results.py`
- `validate_weekly_results.py`
- `scripts/smoke_frontend_static.py`

## What Each Website Page Does

### `/`

Homepage.

Purpose:

- explain the product quickly
- show proof above the fold
- route users to predictions, proof, or premium

Current positioning:

- evidence-first market intelligence
- ROI / signal count / validation credibility
- proof-led rather than entertainment-led

### `/predictions`

Public board.

Purpose:

- show free public picks
- prove the product outputs are real
- create a natural path into premium

Data source:

- `frontend/public/data/public_predictions.json`

### `/premium`

Locked premium board.

Purpose:

- show the stronger offer
- keep access locked by default
- fetch protected Worker premium data only with valid token

Important:

- no token = locked
- `?demo=1` remains internal preview mode only

### `/results`

Proof page.

Purpose:

- show settled outcomes
- show current hit-rate window
- reinforce that predictions and results are independently published

Data source:

- `frontend/public/data/weekly_results.json`

### `/pricing`

Offer page.

Purpose:

- explain Free vs Founding Member
- present the `£20/month` premium offer
- route upgrade flow to the Worker checkout route

### `/methodology`

Trust page.

Purpose:

- explain that the browser does not run model logic
- explain that the site displays approved generated outputs
- reinforce value / structure / stability / deployment filtering

### `/account`

Account and subscription placeholder.

Purpose:

- checkout success landing page
- future home for subscription management
- debug-only token storage tools behind `?debug=1`

## Static Data Files

The frontend currently depends on:

- `frontend/public/data/public_predictions.json`
- `frontend/public/data/premium_predictions.json`
- `frontend/public/data/publish_summary.json`
- `frontend/public/data/weekly_results.json`

## Export And Validation Flow

### Publish predictions

Default:

```bash
python3 publish_predictions.py
```

Explicit source:

```bash
python3 publish_predictions.py --src predictions_output/<RUN>/02_deploy/DEPLOY_COMBINED_<FROM>_to_<TO>.csv
```

Outputs:

- `frontend/public/data/public_predictions.json`
- `frontend/public/data/premium_predictions.json`
- `frontend/public/data/publish_summary.json`
- `reports/latest/PUBLISH_REPORT.md`

### Publish weekly results

Default:

```bash
python3 grade_weekend_results.py
```

Explicit source:

```bash
python3 grade_weekend_results.py --src predictions_output/<RUN>/03_scored/DEPLOY_COMBINED_SCORED_<FROM>_to_<TO>.csv
```

Output:

- `frontend/public/data/weekly_results.json`

### Required validations

Run:

```bash
python3 validate_public_export.py
python3 validate_weekly_results.py
python3 scripts/smoke_frontend_static.py
node --check frontend/assets/app.js
node worker/test_worker_local.js
```

## Git Setup

### Local git root

- `/Users/hughwade/Documents/Code/OG_master`

### Remote

- `origin = https://github.com/ZadarX43/og_BAWA_codex_OCT.git`

### Working branch model

- `dev` = active preview branch
- `main` = later production branch for Pages production deploys

### Current reality

The website work was intentionally committed as a curated slice only.
The wider BAWA PORTO warehouse contains many unrelated tracked and untracked research files.

Do not mass-add the repo.

Use explicit staging only.

### Safe first-class website/deploy paths

- `BAWA PORTO/frontend/`
- `BAWA PORTO/worker/`
- `BAWA PORTO/.github/`
- selected docs
- selected export / validation scripts

### Never casually commit

- `.env`
- `.env.*`
- `.venv/`
- `predictions_output/`
- `Matches/`
- `Teams/`
- `Players/`
- `ModelStore/`
- logs
- temp outputs
- zip archives
- local Worker secrets/config
- warehouse backups

### Important local caveat

`worker/wrangler.toml` is a local operational file.
It contains real KV ids and public deployment vars.
Treat committing it as optional and deliberate, not automatic.

## Cloudflare Pages Setup

### Repo connection

- GitHub repository: `ZadarX43/og_BAWA_codex_OCT`

### Settings that worked

- Framework preset: `None`
- Preview branch: `dev`
- Production branch: `main`
- Build command: blank
- Root directory: `BAWA PORTO`
- Build output directory: `frontend`

Fallback if Cloudflare pathing ever needs it:

- Root directory: repo root
- Build output directory: `BAWA PORTO/frontend`

### Important Pages rule

Cloudflare Pages only serves committed files.

That means:

- run export first
- validate
- commit JSON changes
- push `dev`
- let Pages redeploy

Cloudflare does not run the local model engine.

## Frontend Worker Integration

Frontend config file:

- `frontend/assets/config.js`

Current behavior:

```js
window.OG_CONFIG = {
  WORKER_API_BASE: "https://odds-genius-worker.hughcwade.workers.dev"
};
```

This is safe because it is public origin config only, not a secret.

### Current frontend behavior

- pricing CTA calls Worker checkout when configured
- premium page fetches Worker premium route only with token
- no token means locked state
- `?demo=1` still exists for internal preview
- `?debug=1` reveals account debug/token tools

## Cloudflare Worker Setup

### Purpose

The Worker is the secure boundary for:

- Stripe checkout session creation
- Stripe webhook verification
- subscriber-state persistence
- premium token issuance scaffolding
- protected premium prediction delivery

### Live Worker routes

- `GET /health`
- `POST /api/stripe/checkout`
- `POST /api/premium/token`
- `POST /api/stripe/portal`
- `POST /api/stripe/webhook`
- `GET /api/premium/predictions`

### Required Worker binding

- `SUBSCRIBER_STATE`

### Live KV ids in the local deployed config

- live id: `82c0190b2e5a43e1aff70eebb70c4388`
- preview id: `a152e737280f4eb6a1d7946410a0ba45`

### Worker vars

- `STRIPE_PRICE_ID`
- `SITE_URL`
- `PREMIUM_DATA_SOURCE`

### Worker secrets

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `PREMIUM_TOKEN_SECRET`

Never commit the secret values.

## Stripe Setup

### Current commercial offer

- Founding Member
- `£20/month`

### Current live price id in local deploy config

- `price_1TTRvkDoSY9qcu1woHJ5iKSZ`

This is not a secret.

### Stripe responsibilities

- Checkout session creation
- subscription lifecycle events
- webhook-driven membership state

### Webhook endpoint

- `https://odds-genius-worker.hughcwade.workers.dev/api/stripe/webhook`

### Subscribed events

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`

## Premium Access Flow

Current secure path:

1. user clicks premium upgrade
2. frontend calls Worker checkout route
3. Stripe Checkout completes
4. Stripe webhook hits Worker
5. Worker verifies signature
6. Worker writes `SUBSCRIBER_STATE`
7. active subscriber can obtain a short-lived developer/test token
8. premium route verifies token and subscriber state
9. Worker returns allowlisted premium rows only

## What Has Been Proven Live Already

The following have been tested successfully:

- Cloudflare Pages deploy
- Worker deploy
- Worker `/health`
- Stripe Checkout session creation
- Stripe webhook delivery to Worker
- KV persistence of subscriber state
- active subscription status persistence
- premium token issuance
- protected premium endpoint access with valid token
- rejection of invalid/old token after secret rotation

## Known Security And Product Boundaries

### True today

- premium route is protected server-side
- frontend defaults premium to locked
- old exposed premium token was invalidated by secret rotation

### Still not final production auth

- token issuance route is still scaffolding
- public users should not ultimately paste tokens manually
- final auth should move to magic-link or authenticated session issuance

### Static premium JSON warning

`frontend/public/data/premium_predictions.json` still exists as a committed preview artifact.

It is acceptable for preview and publish workflow reasons.
It is not the final secure premium architecture.

Long term, premium delivery should come from the Worker only.

## Known Git / Repo Risks

- there is a very large warehouse/research estate in this repo
- there are many unrelated untracked files
- do not use wildcard staging at repo root
- do not commit whole directories casually
- always use explicit `git add` paths for website/deploy work

## Current Documentation Map

This file is the master summary.

Focused supporting docs still matter:

- `README_FRONTEND.md`
- `DEPLOYMENT_RUNBOOK.md`
- `CLOUDFLARE_PAGES.md`
- `worker/README_WORKER.md`
- `worker/DEPLOY_WORKER.md`
- `STRIPE_SETUP.md`
- `PREMIUM_ACCESS_PLAN.md`
- `PREMIUM_AUTH_PLAN.md`
- `WORKER_BACKEND_PLAN.md`
- `WORKER_SUBSCRIPTION_STATE.md`
- `LAUNCH_PREDEPLOY_CHECKLIST.md`

## How To Resume Work Later

Read this file first, then:

1. confirm git branch and working tree
2. confirm which live surface is being changed:
   - static frontend
   - Worker
   - Stripe flow
   - export pipeline
3. run validations before any deploy-facing commit
4. use `dev` for preview pushes
5. let Cloudflare Pages redeploy from `dev`
6. only promote to `main` intentionally

## Recommended Resume Checklist

### If changing frontend only

```bash
python3 validate_public_export.py
python3 validate_weekly_results.py
python3 scripts/smoke_frontend_static.py
node --check frontend/assets/app.js
```

### If changing Worker

```bash
node --check worker/src/index.js
node --check worker/src/auth.js
node --check worker/src/subscriber_store.js
node worker/test_worker_local.js
```

### If pushing a Pages update

```bash
git status --short
git add <explicit website files only>
git commit -m "<intentional message>"
git push origin dev
```

Then verify the live preview at:

- [https://og-bawa-codex-oct.pages.dev](https://og-bawa-codex-oct.pages.dev)

## Practical Summary

The current stack is:

- a local football prediction engine
- a safe JSON export layer
- a live static Cloudflare Pages product
- a live Cloudflare Worker secure backend
- a live Stripe-backed subscription skeleton

It is not "finished".
But it is real, deployable, and resumable.

If picking this up later, treat this as the operational truth:

> The frontend sells and displays the product.
> The Worker protects premium access.
> Stripe proves entitlement.
> Git and Cloudflare only publish what has been explicitly validated and committed.
