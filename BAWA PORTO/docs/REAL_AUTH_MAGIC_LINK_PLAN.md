# Real Auth / Magic-Link Plan

## Purpose
This document defines how Odds Genius should move from developer/test premium token handling to a real customer authentication flow based on magic-link or lightweight session auth.

The aim is not to build a heavy account system.
The aim is to make premium access feel like a real product while preserving the current core architecture:

- predictions remain offline and published
- premium board remains a shared protected artifact
- Stripe remains billing truth
- Worker remains the entitlement gate

## Current State

### What already exists
- Stripe Checkout creates subscriptions
- Stripe webhook writes subscriber-state records to `SUBSCRIBER_STATE`
- Worker verifies a signed premium token
- Worker verifies that token against subscriber state
- Premium route is protected and now edge-cached

### Transitional UX that still exists
- `POST /api/premium/token` is developer/test scaffolding
- frontend stores `og_premium_token` locally
- account page still supports debug-only token pasting behind `?debug=1`

Current account flow references:
- [frontend/assets/app.js](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/assets/app.js:1166)
- [worker/src/auth.js](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/src/auth.js:1)
- [worker/src/index.js](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/src/index.js:477)

### What is good about the current baseline
- entitlement is already separated from prediction generation
- subscriber truth already exists
- protected premium route already exists
- Worker token contract is already defined and tested

### What is not acceptable long term
- manual token handling
- developer-style token issuance as a user-facing flow
- public dependence on localStorage token pasting

## Core Product Principle
Users should never feel like they are handling tokens.

They should feel like:
1. they paid
2. they verified identity
3. they are signed in
4. premium unlocks automatically

## Recommended End State

### Customer flow
1. Customer completes Stripe Checkout
2. Stripe webhook persists subscriber state
3. Customer lands on account page
4. Customer verifies email via magic link
5. Worker creates a signed short-lived premium session/token
6. Session is stored securely in cookie form
7. Frontend calls premium route normally
8. Worker verifies session and subscriber entitlement

### Core architectural rule
Magic-link auth identifies the user.
Subscriber-state entitlement authorizes the premium board.

Do not combine those into one opaque product assumption.

Identity and entitlement should stay conceptually separate even if the user experiences them as one flow.

## Why Magic Link Is The Right First Auth Model

### Pros
- low-friction for users
- no password storage burden
- matches a premium content product well
- easy to connect to Stripe email identity
- compatible with session-cookie flow

### Why not full traditional auth first
- too much account surface area too early
- more complexity than Odds Genius needs right now
- distracts from the real task: unlock premium board cleanly

## Recommended Auth Model

## Phase 1: Verified email -> signed premium session

### Flow
1. User enters email on account page or follows a post-checkout route
2. System sends a magic link
3. User clicks magic link
4. Worker validates the one-time login token
5. Worker creates a signed session token or secure premium session cookie
6. Frontend uses cookie-backed session automatically

### Session payload should identify
- customer_id
- subscription_id
- email or canonical identity reference
- exp
- optional session version

### Important
Do not make the browser manually store this as a visible user token.
Move toward secure cookie-backed handling.

## Session Strategy

### Recommended near-term
Use an HTTP-only secure cookie for the premium session.

Why:
- cleaner user experience
- less exposed than localStorage
- better fit for a real sign-in flow

Cookie example concept:
- `og_premium_session`

### Session TTL
Recommended:
- short-to-medium lived session
- renewable by re-verification or re-auth path

Good starting direction:
- 1 to 7 days depending on friction tolerance

### Why not permanent login
- simpler revocation and cleaner entitlement handling
- easier to reason about if subscriber state changes

## Identity Source of Truth

### Immediate source
Stripe customer email plus subscriber-state records

### Required rule
A magic link should only unlock access if:
- email maps to a known subscriber path
- subscriber state is active or trialing

### Mapping options
Short term:
- use subscriber record email when present
- lookup by customer/subscription association already written from webhook flow

Later:
- maintain a small identity table for account normalization if needed

## Storage Responsibilities

### KV
Keep using KV for:
- subscriber-state entitlement lookup
- customer/subscription mapping
- lightweight auth-related lookup material if needed temporarily

### D1
Introduce D1 later only if needed for:
- session audit history
- identity normalization
- support/admin account lookup
- magic-link issuance logs

Do not introduce D1 just to make the first magic-link flow work if KV + signed sessions are sufficient.

## Routes To Add

### Likely new Worker routes
- `POST /api/auth/magic-link/request`
- `GET /api/auth/magic-link/verify`
- `POST /api/auth/logout`
- optional `GET /api/auth/session`

### Responsibilities

#### `POST /api/auth/magic-link/request`
- accept email
- validate basic request shape
- rate limit aggressively
- generate one-time auth token
- send magic-link email

#### `GET /api/auth/magic-link/verify`
- validate one-time token
- resolve subscriber identity
- confirm active entitlement
- set secure session cookie
- redirect user to account or premium page

#### `POST /api/auth/logout`
- clear session cookie

#### `GET /api/auth/session`
- optional lightweight session check for frontend UX
- return session/entitlement status without exposing secrets

## One-Time Magic-Link Token Design

### Requirements
- single use
- short expiry
- signed or stored server-side with verification
- resistant to replay

### Good implementation choices
Option A:
- signed token + nonce persisted server-side

Option B:
- opaque one-time token stored in KV with expiry

Recommended first implementation:
- opaque one-time token in KV with short TTL

Why:
- easier single-use invalidation
- simpler replay protection
- no ambiguity around whether a link has already been consumed

## Email Delivery

### Requirement
The system must send the magic link via a reliable transactional email path.

This plan intentionally does not lock the delivery provider yet.

Possible delivery providers later:
- Postmark
- Resend
- SendGrid
- other transactional mail providers

### Product requirement
Auth delivery must be:
- reliable
- fast
- not treated as marketing email

## Entitlement Check Rules

Magic-link verification alone is not enough.

The system must still verify:
- subscriber-state exists
- subscription is active or trialing
- customer/subscription identity matches the verified session

This keeps the existing strong entitlement model intact.

## Frontend Changes Required

### Account page
Replace debug-era public posture with:
- email verification prompt
- signed-in state
- membership state
- premium access status
- logout action

Keep `?debug=1` internal tooling hidden and isolated until fully retired.

### Premium page
Premium page should:
- try the Worker route with session cookie automatically
- show locked state if not verified
- show clear “verify email / sign in” path

### Pricing page
Pricing CTA should remain checkout-first.
Post-checkout should flow naturally into sign-in / magic-link verification.

## Security Requirements

### Required
- HTTP-only secure cookies
- one-time magic-link tokens
- short token expiry
- session expiry
- rate limiting on auth request route
- replay protection
- audit logging for auth events

### Strongly recommended
- bind magic-link completion to normalized email identity
- log invalid/replayed verification attempts
- reject excessive auth requests per IP/email pair

## Abuse Protection

### Rate limit
- magic-link request route
- magic-link verification route
- session introspection route if exposed

### Defend against
- auth spam
- email enumeration
- token replay
- brute-force verification attempts

### UX rule
Error responses should not leak whether a given email is subscribed unless product policy explicitly allows it.

## Monitoring Requirements

Track:
- magic-link request count
- email send success/failure
- verification success/failure
- session creation success/failure
- premium unlock success after auth
- repeated invalid token attempts

## Migration Plan

## Phase A: Design and route scaffolding
- define auth request route
- define verification route
- define session cookie format
- keep existing premium token verification untouched

## Phase B: Internal dual mode
- allow both debug token flow and magic-link flow
- keep debug flow behind `?debug=1` only
- test real sign-in path without removing fallback yet

## Phase C: Public launch of real auth
- move premium access to session-first flow
- keep debug tooling internal only
- update account and premium UI copy

## Phase D: Retire transitional UX
- remove public-facing token references
- keep only internal operator/debug paths if still necessary

## What Should Not Change

### Do not change
- offline prediction generation
- publish/export flow
- premium board as a shared published artifact
- subscriber-state as entitlement authority
- Worker as the premium gate

### Do not become
- a heavyweight user-account platform
- a password product
- a per-user prediction engine

## Recommended First Implementation

The clean first production auth version should be:

1. Stripe Checkout writes subscriber state through webhook
2. User requests a magic link from account page
3. Worker sends one-time email link
4. Worker verifies link and sets secure session cookie
5. Premium page uses cookie-backed session automatically
6. Worker still checks subscriber entitlement before returning premium board

This gives Odds Genius a real user identity flow without disturbing the core product architecture.

## Explicit Recommendation
Odds Genius should not expose “token management” to customers.

It should feel like a premium intelligence product with:
- checkout
- email verification
- automatic unlock

That is the right bridge from the current controlled-test token model to a real subscriber product.
