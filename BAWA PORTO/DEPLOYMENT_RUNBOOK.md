# DEPLOYMENT_RUNBOOK

Updated: `2026-05-04`

## Purpose

Operational runbook for the static Odds Genius website deployment path.

This runbook does not change model logic.
It assumes the model engine remains local and the site consumes published JSON.

## Deployment Chain

1. local model engine
2. safe export
3. frontend static layer
4. validation
5. GitHub
6. Cloudflare Pages
7. Cloudflare Worker secure backend

## Step 1 — Regenerate Predictions

Use the default latest-source publish:

```bash
python3 publish_predictions.py
```

Or use the one-command wrapper:

```bash
./scripts/publish_latest_deploy.sh
```

Or use an explicit deploy source:

```bash
python3 publish_predictions.py --src predictions_output/<RUN>/02_deploy/DEPLOY_COMBINED_<FROM>_to_<TO>.csv
```

Wrapper with explicit source:

```bash
./scripts/publish_latest_deploy.sh --src predictions_output/<RUN>/02_deploy/DEPLOY_COMBINED_<FROM>_to_<TO>.csv
```

Outputs:

- `frontend/public/data/public_predictions.json`
- `frontend/public/data/premium_predictions.json`
- `frontend/public/data/publish_summary.json`
- `reports/latest/PUBLISH_REPORT.md`

Optional results publish after settled outcomes exist:

```bash
python3 grade_weekend_results.py
```

Or use an explicit scored source:

```bash
python3 grade_weekend_results.py --src predictions_output/<RUN>/03_scored/DEPLOY_COMBINED_SCORED_<FROM>_to_<TO>.csv
```

Results output:

- `frontend/public/data/weekly_results.json`

Optional public live proof feed publish after an audit pack exists:

```bash
python3 scripts/build_public_results_feed.py
```

Live proof feed output:

- `frontend/public/data/live_results_feed.json`
- `docs/PUBLIC_RESULTS_FEED_BUILD_NOTES_2026-05-14.md`

## Step 2 — Validate Export

Run:

```bash
python3 validate_public_export.py
python3 validate_weekly_results.py
python3 validate_live_results_feed.py
```

`validate_public_export.py` must pass before deployment.
If `weekly_results.json` exists, `validate_weekly_results.py` should also pass before deployment.
If `live_results_feed.json` exists, `validate_live_results_feed.py` should pass before deployment.

## Step 3 — Frontend Static Smoke

Run:

```bash
python3 scripts/smoke_frontend_static.py
```

This verifies:

- page files exist
- core assets exist
- published JSON exists
- asset references resolve
- forbidden private/internal terms are not present in visible frontend content

## Step 4 — Preview Locally

Run:

```bash
cd frontend
python3 -m http.server 8000
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/predictions.html`
- `http://127.0.0.1:8000/premium.html`
- `http://127.0.0.1:8000/results.html`
- `http://127.0.0.1:8000/pricing.html`
- `http://127.0.0.1:8000/methodology.html`
- `http://127.0.0.1:8000/account.html`

## Step 5 — Commit To `dev`

After export, validation, and preview pass:

```bash
git add frontend/public/data reports/latest frontend .github README_FRONTEND.md CLOUDFLARE_PAGES.md DEPLOYMENT_RUNBOOK.md scripts/smoke_frontend_static.py publish_predictions.py validate_public_export.py grade_weekend_results.py validate_weekly_results.py
git commit -m "Prepare static frontend deploy"
```

## Step 6 — Push To GitHub

Push preview branch:

```bash
git push origin dev
```

## Step 7 — Cloudflare Preview

Cloudflare Pages is expected to be connected to the GitHub repo with:

- production branch: `main`
- preview branch: `dev`
- build command: none
- build output directory: `frontend`

After pushing `dev`:

1. wait for Cloudflare preview deploy
2. open preview URL
3. confirm the pages and published JSON load correctly

## Step 8 — Promote To Production

After preview approval:

```bash
git checkout main
git merge dev
git push origin main
```

Cloudflare Pages then publishes production from `main`.

If the local worktree is dirty and `dev` is known to be exactly the preview revision approved for production, an equivalent non-checkout promotion is:

```bash
git push origin dev:main
```

Use this only when `origin/main` is the immediate production predecessor of `dev` or after confirming the exact branch relationship. This avoids dragging unrelated local working-tree changes across branches.

## Step 9 — Prepare Worker Deployment

Use:

- [worker/DEPLOY_WORKER.md](/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/worker/DEPLOY_WORKER.md:1)
- [worker/wrangler.example.toml](/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/worker/wrangler.example.toml:1)

Key Worker resources:

- `SUBSCRIBER_STATE` KV namespace
- Worker secrets:
  - `STRIPE_SECRET_KEY`
  - `STRIPE_WEBHOOK_SECRET`
  - `PREMIUM_TOKEN_SECRET`
- Worker vars:
  - `STRIPE_PRICE_ID`
  - `SITE_URL`
  - `PREMIUM_DATA_SOURCE`

## Step 10 — Test Worker Locally

Run:

```bash
node worker/test_worker_local.js
```

## Step 11 — Deploy Worker Manually

Only after local harness and static validations pass:

```bash
cd worker
wrangler deploy
```

Then test:

```bash
curl https://<worker-subdomain>/health
```

## Step 12 — Pre-Deploy Operator Checklist

Before live Cloudflare and Stripe wiring, use:

- [LAUNCH_PREDEPLOY_CHECKLIST.md](/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/LAUNCH_PREDEPLOY_CHECKLIST.md:1)

This checklist covers:

- git branch safety
- commit boundaries
- Cloudflare Pages + Worker setup
- KV namespace creation
- Worker vars and secrets
- Stripe test product, price, and webhook setup
- deployed Worker health checks
- checkout, webhook, and protected premium route testing
- rollback planning

## Non-Negotiable Safety Rules

- never commit secrets
- never commit model binaries because of website work
- never expose raw routing logic in frontend files
- never bypass `validate_public_export.py`
- never bypass `validate_weekly_results.py` when publishing results proof
- never treat the frontend as a place to recreate model logic
- never weaken Worker auth to make premium access “easier”

## Quick Checklist

- publish completed
- results published when available
- export validation passed
- results validation passed when `weekly_results.json` is present
- frontend smoke passed
- local preview checked
- Worker local harness passed
- `dev` pushed
- Cloudflare preview checked
- Worker `/health` checked after deploy
- Stripe test checkout flow checked
- webhook flow checked
- protected premium route checked
- merged to `main` only after review
