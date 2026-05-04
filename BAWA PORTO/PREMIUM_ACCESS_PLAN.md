# PREMIUM_ACCESS_PLAN

Updated: `2026-05-04`

## Purpose

Define the first safe premium-access product structure for Odds Genius without pretending that static files are secure entitlement enforcement.

## Current Static V1 Posture

Static v1 is a preview layer only.

It may:

- show locked premium cards
- show the premium offer and upgrade CTA
- explain what premium access is intended to include
- provide a placeholder checkout path for product flow review

It must not:

- claim that static files are secure subscriber gating
- treat `frontend/public/data/premium_predictions.json` as the final architecture
- expose secrets
- expose backend-only billing logic in frontend code

## Product Intent

Premium is the paid board for subscribers who want:

- the full deployable board
- ELITE picks
- richer explanation
- shortlist context
- slip-role hints and published safety flags where allowed

## Founding Offer

Initial pricing direction:

- Founding Member Plan
- `£20/month`
- locked while subscribed
- early access cohort

## Recommended Access Phases

### Phase 1

Static preview only.

- public board is live
- premium page is locked by default
- checkout is placeholder-only
- account page is placeholder-only

### Phase 2

Stripe Checkout + secure delivery.

- user clicks upgrade CTA
- frontend hands off to backend or Cloudflare Worker
- backend creates Stripe Checkout Session
- success and cancel URLs return to the site
- subscriber status is stored or verified server-side

### Phase 3

Authenticated premium access.

- premium JSON moves behind authenticated backend or Worker delivery
- frontend requests premium data only after verified access
- subscriber session decides whether premium board can be returned
- Worker should return only the approved premium schema, never raw source payloads
- frontend may use a non-secret Worker API base config, but must never hold secrets

### Phase 4

Stripe Customer Portal.

- signed-in user opens manage subscription flow
- backend creates Customer Portal session
- user can update payment method, cancel, or manage plan state

## Architecture Boundary

Future secure premium access should look like:

1. frontend upgrade CTA
2. backend or Cloudflare Worker
3. Stripe Checkout / Stripe Portal
4. authenticated session or token verification
5. protected premium data response

Not like:

1. static page
2. public premium JSON file
3. CSS or client-side hiding

## Data Boundary

Public export policy remains unchanged.

Important rule:

- premium website data may exist for controlled preview during static v1
- final paid access must move premium delivery behind authenticated infrastructure
- secure Worker delivery should re-allowlist premium fields even after auth passes

## Operational Reminder

Until secure gating exists:

- premium.html should default to locked state
- any demo reveal should be explicit and internal-only
- no one should describe static v1 as secure premium access
- any local developer/test token storage must be treated as scaffolding, not public auth
