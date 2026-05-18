# CLOUDFLARE_PAGES_CONNECT_STEPS

Updated: `2026-05-04`

## Purpose

Exact Cloudflare Pages connection steps for the pushed `dev` branch of the Odds Genius web deployment scaffold.

## Repo Readiness

Confirmed in repo:

- `dev` branch exists on `origin`
- `frontend/index.html` exists
- `frontend/assets/config.js` exists
- `frontend/public/data/*.json` exists
- GitHub workflows exist
- frontend and Cloudflare docs exist

Repository:

- `ZadarX43/og_BAWA_codex_OCT`

## Important Path Detail

The static app does not live at repo root.

It lives under:

- `BAWA PORTO/frontend`

That means Cloudflare Pages must be pointed at the subfolder correctly.

## Recommended Cloudflare Pages Settings

### Option A

Recommended if Cloudflare accepts a root directory cleanly.

- Repository: `ZadarX43/og_BAWA_codex_OCT`
- Production branch: `main`
- Preview branch: `dev`
- Framework preset: `None`
- Root directory: `BAWA PORTO`
- Build command: empty
- Build output directory: `frontend`

Why this is the preferred setup:

- it keeps the Pages project scoped to the app folder
- it avoids needing `BAWA PORTO/frontend` as the output path

### Option B

Fallback if you prefer repository root or Cloudflare root-directory behavior is awkward.

- Repository: `ZadarX43/og_BAWA_codex_OCT`
- Production branch: `main`
- Preview branch: `dev`
- Framework preset: `None`
- Root directory: repository root
- Build command: empty
- Build output directory: `BAWA PORTO/frontend`

## Recommended First Attempt

Use Option A first:

- Root directory: `BAWA PORTO`
- Build output directory: `frontend`

## What Pages Should Serve

Pages should serve:

- `index.html`
- `predictions.html`
- `premium.html`
- `results.html`
- `pricing.html`
- `methodology.html`
- `account.html`
- `assets/*`
- `public/data/*`

## No Pages Secrets Needed For Static Preview

For the first static preview deploy:

- no environment variables required
- no Stripe secrets in Pages
- no Worker secrets in Pages

## Worker Note

The frontend already includes:

- `frontend/assets/config.js`

That file can later point the frontend at the deployed Worker base URL without adding secrets into the static site.

## First Preview Verification

After Cloudflare connects and deploys `dev`, verify:

1. homepage loads
2. predictions page loads
3. premium page stays locked by default
4. results page loads
5. `public/data/public_predictions.json` is reachable
6. `public/data/weekly_results.json` is reachable

## If The Deploy Fails

Most likely cause:

- wrong root directory vs output directory combination because of the `BAWA PORTO` subfolder

Fix:

- switch between Option A and Option B
- do not change frontend code first
