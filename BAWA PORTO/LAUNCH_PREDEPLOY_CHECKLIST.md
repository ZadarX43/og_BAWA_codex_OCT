# LAUNCH_PREDEPLOY_CHECKLIST

Updated: `2026-05-04`

## Purpose

Final operator checklist before live Cloudflare and Stripe wiring for Odds Genius.

This checklist is for launch prep only.
It does not deploy automatically and it does not authorize committing secrets.

## 1. Git Branch State

Before doing anything live:

- confirm you are on the intended branch
- confirm the branch has a valid `HEAD`
- confirm local changes match what you intend to push
- confirm you are not mixing unrelated work into the deploy branch

Recommended checks:

```bash
git status
git branch --show-current
git log --oneline -5
```

If git does not report a valid `HEAD`, stop and fix repo state before deployment.

## 2. Files Safe To Commit

Safe to commit:

- frontend static files
- `frontend/public/data/public_predictions.json`
- `frontend/public/data/premium_predictions.json`
- `frontend/public/data/publish_summary.json`
- `frontend/public/data/weekly_results.json`
- `reports/latest/*`
- Worker source files
- docs and runbooks
- GitHub workflow files
- smoke and validation scripts

## 3. Files Never To Commit

Never commit:

- `.env` files
- API keys
- Stripe secrets
- webhook secrets
- premium token secret values
- model binaries
- raw source data dumps
- private thresholds or gate formulas
- ad hoc local config containing secrets

Examples:

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `PREMIUM_TOKEN_SECRET`

## 4. Cloudflare Pages Setup

Confirm:

- Pages project created
- GitHub repo connected
- production branch set to `main`
- preview branch flow uses `dev`
- build command blank
- output directory `frontend`

## 5. Cloudflare Worker Setup

Confirm:

- Worker project prepared from `worker/wrangler.example.toml`
- Worker name chosen
- Worker deploy target understood
- no secrets placed in Wrangler file

## 6. KV Namespace Setup

Create and bind:

- `SUBSCRIBER_STATE`

Commands:

```bash
cd worker
wrangler kv namespace create SUBSCRIBER_STATE
wrangler kv namespace create SUBSCRIBER_STATE --preview
```

Then place the returned ids into Wrangler config.

## 7. Worker Env Vars And Secrets

Secrets to set in Cloudflare Worker secret management:

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `PREMIUM_TOKEN_SECRET`

Vars to set in Wrangler config:

- `STRIPE_PRICE_ID`
- `SITE_URL`
- `PREMIUM_DATA_SOURCE`

## 8. Stripe Test Product And Price

Before live checkout testing:

- create test product for Founding Member plan
- create recurring monthly test price
- copy resulting Stripe test price id into `STRIPE_PRICE_ID`

Recommended meaning:

- Founding Member Plan
- `£20/month`

## 9. Stripe Webhook Endpoint Setup

Create a Stripe test webhook endpoint pointing to:

- `POST /api/stripe/webhook`

Capture the signing secret and store it only as:

- `STRIPE_WEBHOOK_SECRET`

Recommended event subscriptions:

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`

## 10. Local Validation Commands

Run all of these before pushing:

```bash
python3 validate_public_export.py
python3 validate_weekly_results.py
python3 scripts/smoke_frontend_static.py
node --check frontend/assets/app.js
node --check worker/src/index.js
node worker/test_worker_local.js
```

## 11. Deployed Health Test

After Worker deploy:

```bash
curl https://<worker-subdomain>/health
```

Confirm:

- `200`
- expected route list
- env summary looks sensible without exposing secrets

## 12. Test Checkout Flow

In test mode:

1. configure real Worker base in frontend config
2. click upgrade CTA from pricing page
3. confirm redirect to Stripe Checkout
4. complete test checkout
5. confirm return to `account.html?checkout=success`

## 13. Test Webhook Event Flow

Confirm:

1. Stripe sends webhook
2. Worker verifies signature
3. `SUBSCRIBER_STATE` record is written
4. customer and subscription ids match expected test data

## 14. Test Protected Premium Endpoint

Confirm all of these:

1. no token returns `401`
2. expired token returns `401`
3. inactive subscriber returns `401`
4. valid token + active subscriber returns `200`
5. response only contains premium allowlist fields

## 15. Rollback Plan

If something breaks:

Pages rollback options:

- revert deploy branch commit
- push fixed `dev`
- if needed, stop promoting to `main`

Worker rollback options:

- redeploy previous known-good Worker code
- temporarily disable frontend Worker base config
- revert Worker route changes if premium path is unstable

Stripe rollback posture:

- keep test mode until end-to-end flow is proven
- do not switch to live price or live secret values until test flow passes

## 16. Manual Launch Sequence

Manual operator flow:

1. push `dev`
2. connect Cloudflare Pages
3. deploy Worker
4. add Stripe test price
5. add Worker secrets
6. run first real checkout test

## Final Rule

Do not go live on premium access until:

- checkout works
- webhook writes subscriber state
- protected premium route passes with verified token
- frontend Worker handoff works
- rollback path is understood
