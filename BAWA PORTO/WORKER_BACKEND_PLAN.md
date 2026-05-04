# WORKER_BACKEND_PLAN

Updated: `2026-05-04`

## Purpose

Define the first secure-backend direction for Odds Genius premium access, Stripe billing flows, and protected premium prediction delivery.

## Why This Layer Exists

Static v1 is useful for product proof, but it is not secure subscriber enforcement.

The Worker layer is intended to become the boundary for:

- Stripe Checkout session creation
- Stripe Customer Portal session creation
- Stripe webhook handling
- authenticated premium prediction delivery

## Scope For This Scaffold

This task creates only a safe placeholder scaffold.

It does not yet provide:

- real Stripe API calls
- real auth
- real session verification
- real premium data protection
- premium entitlement checks

## Target Routes

- `GET /health`
- `POST /api/stripe/checkout`
- `POST /api/stripe/portal`
- `POST /api/stripe/webhook`
- `GET /api/premium/predictions`

## Route Intent

### `GET /health`

Simple health probe for Cloudflare deployment readiness.

### `POST /api/stripe/checkout`

Later:

- validate request
- verify plan
- create Stripe Checkout Session using server-side secret
- return Checkout URL

### `POST /api/stripe/portal`

Later:

- verify signed-in subscriber
- create Stripe Customer Portal session
- return Portal URL

### `POST /api/stripe/webhook`

Later:

- verify Stripe signature
- handle subscription lifecycle events
- update subscriber access state

### `GET /api/premium/predictions`

Later:

- verify authenticated premium access
- fetch premium board from protected source
- return safe premium payload only to entitled users

## Required Future Environment Variables

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `SITE_URL`
- `STRIPE_PRICE_ID`
- `PREMIUM_DATA_SOURCE`

Required binding for webhook persistence:

- `SUBSCRIBER_STATE`

Notes:

- no real values belong in git
- these must live in Cloudflare Worker environment settings later

## Premium Data Direction

Do not move `frontend/public/data/premium_predictions.json` yet.

Short term:

- it remains a controlled preview artifact for static v1

Later:

- premium data should be served through the Worker after access checks
- the Worker may read from KV, R2, D1, durable source storage, or another protected backend source

## Recommended Implementation Phases

### Phase 1

Placeholder Worker routes and docs.

### Phase 2

Stripe Checkout and Customer Portal session creation using protected secrets.

### Phase 3

Webhook verification and subscriber-state persistence.

### Phase 4

Protected premium data route with real auth/session gating.

## Security Reminder

The secure boundary must be server-side.

Do not rely on:

- hidden frontend links
- query params
- client-side flags
- obscured static JSON paths

Those are product-preview tools only, not access control.
