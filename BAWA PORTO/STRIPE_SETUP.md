# STRIPE_SETUP

Updated: `2026-05-04`

## Purpose

Document the planned Stripe integration path for Odds Genius while keeping static v1 secret-free.

## Current Status

Stripe is partially wired.

This repo currently supports:

- premium offer presentation
- locked premium UI
- Worker-backed Checkout Session route scaffold
- Worker-backed webhook verification scaffold
- account placeholder flow

This repo does not yet support:

- real Customer Portal sessions
- authenticated subscription enforcement
- frontend auth-backed entitlement checks

## Founding Plan

Planned first offer:

- product name: Founding Member Plan
- price: `£20/month`
- billing cadence: monthly

## Planned Stripe Checkout Flow

Target shape:

1. user clicks upgrade CTA on `premium.html` or `pricing.html`
2. frontend calls backend or Cloudflare Worker endpoint
3. Worker creates Stripe Checkout Session
4. Worker returns Checkout URL
5. browser redirects to Stripe Checkout
6. Stripe returns user to success or cancel URL
7. later backend verification decides entitlement

Important:

- Checkout Session creation must not happen in static frontend code
- secret keys must never be embedded in JavaScript

## Planned Stripe Customer Portal Flow

Target shape:

1. authenticated subscriber opens account page
2. frontend requests a Portal Session from backend or Worker
3. backend creates Stripe Customer Portal session
4. backend returns Portal URL
5. browser redirects user to Stripe portal

Use cases:

- update payment method
- cancel subscription
- review billing details

## Secrets And Environment Variables

Do not commit any secrets.

Future secret examples:

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`

Required non-secret config for checkout:

- `SITE_URL`
- `STRIPE_PRICE_ID`
- `PREMIUM_DATA_SOURCE`

These must live only in protected deployment environment settings such as:

- Cloudflare Workers environment variables
- Cloudflare Pages environment settings if a server-side function layer is added
- any later backend hosting secret store

Never place secrets in:

- git-tracked frontend files
- published JSON
- markdown docs with real values

## Premium Data Delivery Warning

`frontend/public/data/premium_predictions.json` is acceptable only as a controlled preview artifact during static v1.

It is not the final secure architecture.

Final premium delivery should move to:

- authenticated backend endpoint
- or authenticated Cloudflare Worker endpoint

That delivery layer should verify subscriber access before returning premium board data.

## Suggested Later Endpoints

Examples for a later secure layer:

- `POST /api/stripe/checkout`
- `POST /api/stripe/portal`
- `POST /api/stripe/webhook`
- `GET /api/premium/predictions`

## Current Checkout Route

The Worker now includes:

- `POST /api/stripe/checkout`

Current behavior:

- uses `STRIPE_SECRET_KEY`
- uses `SITE_URL`
- uses `STRIPE_PRICE_ID`
- creates a subscription-mode Stripe Checkout Session
- returns `{ "ok": true, "url": "..." }` when successful

Current redirects:

- success URL: `SITE_URL/account.html?checkout=success`
- cancel URL: `SITE_URL/pricing.html?checkout=cancelled`

This route still does not provide entitlement by itself.

## Webhook Notes

Webhook processing belongs server-side only.

Likely events later:

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`

Webhook verification must use the Stripe webhook secret in protected environment configuration only.

Current Worker webhook scaffold:

- verifies Stripe signatures with `STRIPE_WEBHOOK_SECRET`
- accepts core subscription lifecycle events
- persists a simple subscriber-state record into the Worker store binding

Required binding for the current scaffold:

- `SUBSCRIBER_STATE`

Important boundary:

- persisted subscriber state is now the future source for entitlement checks
- premium predictions must still remain locked until a real auth/session layer uses that state

## Static V1 Summary

For now:

- pricing page defines the offer
- premium page stays locked by default
- account page explains the future billing home
- checkout actions remain placeholders

That keeps the product direction visible without faking secure access before the backend exists.
