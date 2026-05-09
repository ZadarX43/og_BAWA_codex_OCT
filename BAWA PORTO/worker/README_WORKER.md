# README_WORKER

Updated: `2026-05-04`

## Purpose

This folder is the Cloudflare Worker scaffold for future secure billing and premium delivery.

Current status:

- checkout route wired
- webhook route wired
- premium token route wired for developer/test issuance
- premium route is token-protected and schema-filtered
- magic-link request, verify, session, and logout now have a real first implementation
- optional D1-backed account-state mirroring now exists for users, subscriptions, Telegram links, preferences, and auth events
- portal route remains a placeholder
- transactional email delivery still requires provider secrets before production auth is fully live

## Planned Responsibilities

- create Stripe Checkout sessions
- create Stripe Customer Portal sessions
- receive Stripe webhooks
- return premium predictions only after secure access checks

## Current Routes

- `GET /health`
- `GET /api/account/state`
- `POST /api/account/telegram/link/start`
- `POST /api/account/telegram/link/complete`
- `POST /api/telegram/webhook`
- `POST /api/stripe/checkout`
- `POST /api/premium/token`
- `POST /api/stripe/portal`
- `POST /api/stripe/webhook`
- `GET /api/premium/predictions`

Each route currently returns safe JSON that explains it is not wired yet.

Exception:

- `POST /api/stripe/checkout` now attempts to create a real Stripe Checkout Session when the required env vars are present
- `POST /api/stripe/webhook` now verifies Stripe signatures and persists subscriber-state records when the required secret and binding are present
- `POST /api/premium/token` now issues a signed 7-day premium token after verified subscriber-state lookup
- `GET /api/premium/predictions` now verifies a signed v1 premium token, edge-caches the sanitized shared premium payload, and returns only the approved premium schema
- auth route skeletons now exist for magic-link request, verify, session, and logout so production auth can replace public token handling incrementally

## Required Future Environment Variables

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `SITE_URL`
- `PREMIUM_DATA_SOURCE`
- `STRIPE_PRICE_ID`
- `PREMIUM_TOKEN_SECRET`
- `AUTH_MAGIC_LINK_SECRET`
- `AUTH_SESSION_SECRET` (optional but recommended)
- `RESEND_API_KEY`
- `AUTH_EMAIL_FROM`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_BOT_USERNAME` (optional)
- `TELEGRAM_WEBHOOK_SECRET`

Do not store real values in this repo.

Required binding for webhook persistence:

- `SUBSCRIBER_STATE`

Optional binding for account/product state:

- `ACCOUNT_DB` (D1)

## Local Notes

This scaffold is syntax-checkable with:

```bash
node --check worker/src/index.js
```

Later, once Wrangler is installed and configured, a local Worker dev loop can be added.

Local harness:

```bash
node worker/test_worker_local.js
```

Current harness coverage:

- active subscriber can obtain a developer/test token and access protected premium route
- active subscriber can request a magic link, verify it, receive a session cookie, and open the premium route without a bearer token
- session-authenticated account state can mirror into the optional D1 layer
- Telegram link flow can issue a one-time code and complete a D1-backed account link when `ACCOUNT_DB` is present
- Telegram bot webhook can consume `/start oglink_CODE` and complete the D1-backed account link when bot env vars are present
- protected route returns only allowlisted premium fields
- missing token returns `401`
- expired token returns `401`
- inactive subscriber cannot obtain or use premium access

## Checkout Route

`POST /api/stripe/checkout` now:

- reads `STRIPE_SECRET_KEY`
- reads `SITE_URL`
- reads `STRIPE_PRICE_ID`
- creates a Stripe Checkout Session in `subscription` mode
- returns `{ "ok": true, "url": "..." }` on success

Redirects:

- success URL: `SITE_URL/account.html?checkout=success`
- cancel URL: `SITE_URL/pricing.html?checkout=cancelled`

Optional request JSON fields:

- `email`
- `reference`

Those are only used as optional hints for the Checkout Session and do not provide auth.

## Webhook Route

`POST /api/stripe/webhook` now:

- reads `STRIPE_WEBHOOK_SECRET`
- verifies the `stripe-signature` header using HMAC SHA-256
- accepts these event types:
  - `checkout.session.completed`
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
- writes a simple subscriber-state record into `SUBSCRIBER_STATE`

Current persisted record contract:

- `customer_id`
- `subscription_id`
- `status`
- `price_id`
- `current_period_end`
- `updated_at`

This route still does not unlock premium delivery by itself.

## Premium Token Route

`POST /api/premium/token` now:

- accepts JSON body with:
  - `customer_id`
  - `subscription_id`
- reads `SUBSCRIBER_STATE`
- requires matching subscriber state
- requires `active` or `trialing` status
- issues a signed token using `PREMIUM_TOKEN_SECRET`
- sets expiry to 7 days

Important warning:

- this route is developer/test scaffolding only
- final public issuance should use magic-link email verification or authenticated session checks

## Premium Route

`GET /api/premium/predictions` now:

- first attempts secure session-cookie verification
- falls back to transitional premium token verification during migration
- checks for a premium token in `Authorization: Bearer ...` or `og_premium_token` cookie form
- verifies a v1 HMAC-signed token payload containing:
  - `customer_id`
  - `subscription_id`
  - `exp`
- requires an active or trialing matching subscriber-state record in `SUBSCRIBER_STATE`
- loads premium prediction JSON from `PREMIUM_DATA_SOURCE`
- returns only the approved premium allowlist plus minimal metadata

Current state:

- access remains denied unless token verification and KV state both pass
- premium data source must currently be an absolute URL or same-origin path
- frontend-only entitlement is still not trusted

Current response metadata:

- `generated_at` when available from the source payload
- `subscriber_customer_id`
- `auth_mode`
- `count`

## Auth Routes

`POST /api/auth/magic-link/request` now:

- validates email input
- rate limits by IP and email cooldown keys in KV
- looks up active/trialing subscriber state by normalized email
- writes a short-lived one-time token into KV
- sends a verification email through Resend when the address is eligible
- always returns a generic success response for eligible/non-eligible addresses

`GET /api/auth/magic-link/verify` now:

- consumes the one-time KV token
- confirms matching subscriber state is still active or trialing
- issues a signed `og_premium_session` cookie
- redirects back to `account.html` with a friendly auth state

`GET /api/auth/session` now:

- reports current session-backed premium state when a valid session cookie is present
- falls back to transitional token-backed auth during migration

`POST /api/auth/logout` now:

- clears both the session cookie and transitional premium token cookie

## Account State And Telegram Routes

`GET /api/account/state`:

- requires a verified premium session
- returns session-linked account state
- includes D1-backed user/subscription/Telegram/preferences data when `ACCOUNT_DB` is configured
- falls back to session-only metadata when D1 is not yet bound

`POST /api/account/telegram/link/start`:

- requires a verified premium session
- requires `ACCOUNT_DB` plus `SUBSCRIBER_STATE`
- generates a short-lived one-time code
- stores the code in KV with TTL
- returns a Telegram bot deep link when `TELEGRAM_BOT_USERNAME` is configured

`POST /api/account/telegram/link/complete`:

- accepts `code` and Telegram identity fields
- consumes the one-time KV code
- writes the linked Telegram identity into D1
- can still be used directly by trusted internal tooling if needed

`POST /api/telegram/webhook`:

- expects Telegram Bot API webhook updates
- requires `TELEGRAM_BOT_TOKEN`
- requires `TELEGRAM_WEBHOOK_SECRET`
- validates the `x-telegram-bot-api-secret-token` header
- accepts `/start oglink_CODE` or a pasted six-character link code
- completes the same D1/KV link flow as the direct completion endpoint
- replies in-chat with success or retry guidance
- enables Telegram notifications in preferences

## D1 Migration

Initial schema lives at:

- `worker/migrations/0001_account_state.sql`

Suggested setup once you create the D1 database in Cloudflare:

```bash
cd worker
wrangler d1 create odds-genius-account-state
wrangler d1 migrations apply ACCOUNT_DB
```

Then add the returned `database_id` to:

- `worker/wrangler.toml`

or your production environment override.

Current source strategy note:

- Worker runtime cannot rely on local filesystem reads
- final deployment should use KV, R2, or a protected static asset fetch strategy when direct same-origin fetch is not suitable

## Architecture Boundary

Static frontend pages may describe premium access, but real premium protection must happen here or in an equivalent backend layer.

Until that is implemented:

- `premium.html` remains a locked preview
- checkout links are placeholders
- premium JSON is not secure access
