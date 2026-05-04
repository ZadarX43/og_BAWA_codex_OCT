# PREMIUM_AUTH_PLAN

Updated: `2026-05-04`

## Purpose

Define the minimum secure entitlement check for Odds Genius before the Worker is allowed to return premium prediction data.

## Current Rule

`GET /api/premium/predictions` must fail closed until the Worker can verify both:

1. a valid premium access credential
2. an active subscriber state record

## What This Task Adds

- a Worker auth scaffold
- a real v1 `verifyPremiumAccess(request, env)` verifier
- a fail-closed premium route boundary

It does not yet add:

- user accounts
- full login flows
- premium data delivery

## Recommended Simplest V1

Use a signed premium access token issued only after verified Stripe subscription state exists.

Shape:

1. Stripe Checkout completes
2. Stripe webhook writes subscriber state
3. a developer/test issuance route can mint a short-lived signed token after verified subscriber lookup
4. `GET /api/premium/predictions` verifies the token
5. the Worker checks subscriber state for active entitlement
6. only then can premium data be returned

Why this is the best first version:

- smaller than full account infrastructure
- server-side entitlement remains authoritative
- avoids pretending that frontend-only gating is real security

## Future Auth Options

### Option A

Magic link login tied to Stripe customer email.

Pros:

- simple founder-friendly UX
- no password system
- can map directly to subscriber email

Tradeoffs:

- still needs secure token issuance and email delivery
- email identity and Stripe customer mapping must be handled carefully

### Option B

Stripe Customer Portal or account-session based access.

Pros:

- closer to billing ownership
- good for subscription management flows

Tradeoffs:

- still needs application-side session handling
- Stripe billing state alone is not a full app-auth system

### Option C

Clerk, Supabase, or Auth0 later.

Pros:

- full auth stack available
- easier long-term account management

Tradeoffs:

- heavier than needed for the first premium entitlement version
- more moving parts before premium delivery is proven

## Token V1 Format

This scaffold uses a signed HMAC token, not a full JWT library dependency.

Shape:

- `<base64url(json_payload)>.<base64url(hmac_sha256_signature)>`

Required payload fields:

- `customer_id`
- `subscription_id`
- `exp`

Verification rules:

1. extract token from `Authorization: Bearer ...` or `og_premium_token`
2. decode payload
3. verify HMAC SHA-256 signature with `PREMIUM_TOKEN_SECRET`
4. reject expired tokens
5. read `SUBSCRIBER_STATE`
6. require matching `customer_id`
7. require matching `subscription_id`
8. require subscription status of `active` or `trialing`

## Token Issuance Scaffold

Current developer/test route:

- `POST /api/premium/token`

Request body:

- `customer_id`
- `subscription_id`

Current issuance rules:

1. load subscriber state from `SUBSCRIBER_STATE`
2. require matching `customer_id`
3. require matching `subscription_id`
4. require status of `active` or `trialing`
5. sign token with `PREMIUM_TOKEN_SECRET`
6. set expiry to 7 days

Important warning:

- this is only scaffolding for controlled testing
- final public issuance should require magic-link email verification or authenticated session issuance

## Required Future Binding And Secrets

Current bindings and config already in play:

- `SUBSCRIBER_STATE`
- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `STRIPE_PRICE_ID`
- `SITE_URL`
- `PREMIUM_DATA_SOURCE`

Recommended next auth secret:

- `PREMIUM_TOKEN_SECRET`

This must live only in protected Worker environment settings.

## Fail-Closed Requirement

Until token issuance is live:

- no token means no premium access
- invalid token means no premium access
- missing subscriber state means no premium access
- inactive subscriber state means no premium access

## Next Implementation Step

After this scaffold, build:

1. protected premium route response after successful verification
2. final issuance flow with magic-link email verification or authenticated session
3. frontend/backend handoff for sending the token securely
