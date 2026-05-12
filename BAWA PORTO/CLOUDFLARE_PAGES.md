# CLOUDFLARE_PAGES

Updated: `2026-05-04`

## Purpose

This document explains how to connect the Odds Genius static frontend scaffold to Cloudflare Pages without exposing secrets or moving model logic into the frontend.

## Static V1 Deployment Shape

Current system flow:

1. local model engine
2. safe export
3. frontend static files
4. validation
5. GitHub
6. Cloudflare Pages
7. Cloudflare Worker for secure premium/backend routes

Current Cloudflare Pages assumptions:

- production branch: `main`
- preview branch: `dev`
- build command: none
- build output directory depends on whether Root directory is `BAWA PORTO` or repo root
- environment variables required for static v1: none

## GitHub Repo Connection

In Cloudflare Pages:

1. Create a new Pages project
2. Connect the GitHub repository
3. Select the Odds Genius repository
4. Use:
   - production branch: `main`
   - preview branch support from `dev`

## Build Settings

Use these settings:

- Framework preset: `None`
- Build command: leave blank
- Preferred Root directory: `BAWA PORTO`
- Preferred Build output directory: `frontend`
- Fallback Root directory: repository root
- Fallback Build output directory: `BAWA PORTO/frontend`

Why:

- the site is static HTML/CSS/JS
- published JSON already lives inside `frontend/public/data/`
- the app lives inside the `BAWA PORTO` subfolder, not at repository root
- no Node build step is required for v1

## What Cloudflare Serves

Cloudflare Pages should serve:

- `BAWA PORTO/frontend/index.html`
- `BAWA PORTO/frontend/predictions.html`
- `BAWA PORTO/frontend/premium.html`
- `BAWA PORTO/frontend/results.html`
- `BAWA PORTO/frontend/pricing.html`
- `BAWA PORTO/frontend/methodology.html`
- `BAWA PORTO/frontend/account.html`
- `BAWA PORTO/frontend/assets/*`
- `BAWA PORTO/frontend/public/data/*`

## Preview And Production Behavior

### `dev`

Use `dev` for preview deploys.

Recommended flow:

1. regenerate predictions locally
2. validate public export
3. run frontend smoke
4. commit to `dev`
5. push `dev`
6. review Cloudflare preview

### `main`

Use `main` for production deploys.

Recommended flow:

1. review preview on `dev`
2. merge approved changes to `main`
3. push `main`
4. Cloudflare Pages publishes production

## Environment Variables

### Static v1

No environment variables are required.

Do not add:

- API keys
- model paths
- secret tokens
- private deploy thresholds

### Future Stripe / backend variables

If later stages add Stripe or a backend:

- keep secrets only in Cloudflare Pages / Workers environment settings
- never commit secrets to the repo
- never expose secret values in frontend JavaScript
- never put secret values in published JSON

Likely future secret examples:

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- backend auth or API service credentials

Those belong in protected deployment settings, not in static files.

## Worker Pairing

Cloudflare Pages and the Worker should be treated as separate deployment surfaces:

- Pages serves the static site
- Worker serves secure backend routes

Current Worker route family:

- `GET /health`
- `POST /api/stripe/checkout`
- `POST /api/premium/token`
- `POST /api/stripe/portal`
- `POST /api/stripe/webhook`
- `GET /api/premium/predictions`

Worker setup details live in:

- [worker/DEPLOY_WORKER.md](/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/worker/DEPLOY_WORKER.md:1)
- [CLOUDFLARE_PAGES_CONNECT_STEPS.md](/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/CLOUDFLARE_PAGES_CONNECT_STEPS.md:1)

## Future Frontend To Worker Direction

The frontend should later call Worker routes for secure premium flows instead of relying on static premium JSON.

Likely sequence:

1. pricing or premium CTA calls Worker checkout route
2. Stripe + webhook write subscriber state
3. verified session or magic-link flow obtains token
4. frontend requests Worker premium route with verified access

## Operational Warning

Cloudflare Pages only publishes what is committed.

That means:

- predictions must be regenerated before commit
- validated JSON must be committed if the site is expected to update
- Cloudflare does not run the model engine

## Recommended Next Step After Pages Connection

After Pages is connected:

1. confirm `dev` preview deploy works
2. confirm `main` deploy works
3. optionally add a small README note or badge with preview URL
4. later decide whether published JSON should stay committed or move to a backend delivery layer
