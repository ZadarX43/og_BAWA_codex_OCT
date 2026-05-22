# DEPLOY_WORKER

Updated: `2026-05-04`

## Purpose

Deployment runbook for the Odds Genius Cloudflare Worker layer.

This document covers setup only.
It does not deploy automatically and does not place real secrets in the repo.

## What The Worker Is For

The Worker is the secure backend boundary for:

- Stripe Checkout session creation
- Stripe webhook verification
- subscriber-state persistence
- premium token issuance scaffolding
- protected premium prediction delivery

## Files In Scope

- `worker/src/index.js`
- `worker/src/auth.js`
- `worker/src/subscriber_store.js`
- `worker/wrangler.example.toml`
- `worker/test_worker_local.js`

## Step 1 — Prepare Wrangler Config

Use `worker/wrangler.example.toml` as the template.

Recommended flow:

1. copy it to a real local config that is not used for secrets
2. keep secrets out of the file
3. use Cloudflare secret management for secret values

Example:

```bash
cp worker/wrangler.example.toml worker/wrangler.toml
```

## Step 2 — Create KV Namespace

Create the subscriber-state KV namespace:

```bash
cd worker
wrangler kv namespace create SUBSCRIBER_STATE
```

For preview or non-production environments you may also want:

```bash
cd worker
wrangler kv namespace create SUBSCRIBER_STATE --preview
```

Take the returned namespace id and place it into the Wrangler config under:

```toml
[[kv_namespaces]]
binding = "SUBSCRIBER_STATE"
id = "replace-with-real-kv-namespace-id"
```

## Step 3 — Set Worker Vars

Set non-secret vars in `wrangler.toml`:

- `STRIPE_PRICE_ID`
- `SITE_URL`
- `PREMIUM_DATA_SOURCE`

Recommended meanings:

- `STRIPE_PRICE_ID`
  - the real Stripe recurring price id for the founding plan
- `SITE_URL`
  - the frontend site base URL
- `PREMIUM_DATA_SOURCE`
  - a fetchable premium JSON URL or protected source path

## Step 4 — Set Worker Secrets

Never place these in git-tracked files:

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `PREMIUM_TOKEN_SECRET`

Set them with Wrangler:

```bash
cd worker
wrangler secret put STRIPE_SECRET_KEY
wrangler secret put STRIPE_WEBHOOK_SECRET
wrangler secret put PREMIUM_TOKEN_SECRET
```

For the `site_data_test` Worker environment, set the same required launch-smoke
secrets against that environment as well. Cloudflare Worker environments do not
expose secret values back through Wrangler, so these must be re-entered rather
than copied from the primary Worker:

```bash
cd worker
wrangler secret put STRIPE_SECRET_KEY --env site_data_test
wrangler secret put STRIPE_WEBHOOK_SECRET --env site_data_test
wrangler secret put PREMIUM_TOKEN_SECRET --env site_data_test
wrangler secret put AUTH_MAGIC_LINK_SECRET --env site_data_test
wrangler secret put AUTH_SESSION_SECRET --env site_data_test
wrangler secret put RESEND_API_KEY --env site_data_test
wrangler secret put AUTH_EMAIL_FROM --env site_data_test
```

After setting them, verify the test Worker against the active Pages preview:

```bash
python3 scripts/cloudflare_preview_readiness.py \
  --worker-url https://odds-genius-worker-site-data-test.hughcwade.workers.dev \
  --site-url <active-pages-preview-url> \
  --skip-local-publish
```

## Step 5 — Run Local Harness Before Deploy

Run:

```bash
node worker/test_worker_local.js
```

This currently proves:

- valid subscriber state can obtain a developer/test token
- protected premium route returns `200` with valid token
- allowlist filtering is enforced
- missing token returns `401`
- expired token returns `401`
- inactive subscriber returns `401`

## Step 6 — Deploy Worker

Once Wrangler is configured:

```bash
cd worker
wrangler deploy
```

Do this only after local checks pass.

## Step 7 — Test Health Route

After deployment:

```bash
curl https://<worker-subdomain>/health
```

Expected:

- `200`
- JSON response listing current routes

## Step 8 — Test Protected Premium Route

Local harness proves route logic before Cloudflare wiring.

After real deployment, test in stages:

1. confirm `/health`
2. confirm webhook binding and env configuration
3. confirm developer/test token route if intentionally enabled
4. confirm `GET /api/premium/predictions` returns `401` without token
5. confirm verified token returns protected data only

## Step 9 — Frontend Integration Direction

Later frontend integration should call Worker routes, not rely on public static premium JSON for real paid access.

Likely future frontend flow:

1. user upgrades
2. frontend calls `POST /api/stripe/checkout`
3. Stripe completes payment
4. webhook writes subscriber state
5. verified session or magic-link flow obtains token
6. frontend calls `GET /api/premium/predictions`

## Important Deployment Notes

- Worker runtime cannot rely on local filesystem reads
- `PREMIUM_DATA_SOURCE` must point to something fetchable by the Worker
- long term, KV, R2, or protected asset fetch is more robust than public static premium JSON
- the current token issuance route is for controlled testing only and should later be replaced by verified magic-link or authenticated session issuance

## Minimal Launch Checklist

- Wrangler config created
- `SUBSCRIBER_STATE` namespace created
- secrets set in Cloudflare
- vars set in Wrangler
- local Worker harness passed
- `/health` tested after deploy
- protected premium route tested with and without token
