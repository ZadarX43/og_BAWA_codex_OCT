# Odds Genius Website 10k User Architecture Plan

## Purpose
This document defines the target web architecture for Odds Genius as the product moves from a working commercial shell into a production-grade system capable of serving up to 10,000 logged-in users without coupling live user traffic to model execution.

The governing principle is simple:

- predictions stay offline
- model inference never runs inside user request paths
- website boards are published artifacts
- Cloudflare delivers content at the edge
- entitlement and auth stay lightweight

This is the correct architecture for Odds Genius because the product is a prediction intelligence system, not a real-time compute app.

## Non-Negotiable Product Architecture

### Keep
- Offline model engine
- Export -> publish -> edge delivery workflow
- Public board as static JSON
- Premium board as precomputed JSON behind Worker entitlement
- Cloudflare Pages for website delivery
- Cloudflare Worker for premium gating and account actions
- Stripe-backed subscription state
- KV-backed lightweight subscriber lookup

### Do not do
- Do not serve model inference live to end users
- Do not move toward a VPS-first monolith for the core product
- Do not mix user login traffic with model compute traffic
- Do not make premium board access depend on heavy mutable database reads
- Do not let website sessions become a trigger for prediction generation

## Current Baseline

### What already exists
- Static frontend on Cloudflare Pages
- Public predictions served from `frontend/public/data/public_predictions.json`
- Premium predictions published to `frontend/public/data/premium_predictions.json`
- Cloudflare Worker routes for:
  - `/health`
  - `/api/stripe/checkout`
  - `/api/premium/token`
  - `/api/stripe/portal`
  - `/api/stripe/webhook`
  - `/api/premium/predictions`
- Stripe subscription checkout and webhook entitlement updates
- `SUBSCRIBER_STATE` KV namespace for lightweight subscriber-state storage
- Local publish bridge from model CSV output into website-safe JSON
- One-command publish helper:
  - `scripts/publish_latest_deploy.sh`

### What the baseline proves
- Public and premium boards can be published from offline model outputs
- Premium board can be protected behind Worker entitlement
- Subscriber state can be updated through Stripe webhook events
- The website can scale better than a live-compute system because board content is prebuilt

## 10k User Target Architecture

### High-level stack
1. Offline model engine
2. Publish/export layer
3. Static artifact delivery via Cloudflare Pages
4. Lightweight auth and entitlement on Cloudflare Workers
5. Stripe for billing and subscription truth
6. KV for subscriber-state lookup
7. Optional D1 later for user/account metadata and audit reporting

### Request model

#### Public traffic
- User requests Pages HTML, CSS, JS, and public JSON
- Cloudflare edge serves static content directly
- No Worker hop required for normal public board reads

#### Premium traffic
- User loads frontend from Pages
- Frontend calls Worker for premium access
- Worker verifies identity/session and entitlement
- Worker returns premium JSON payload or a cached premium board response
- Worker does not trigger model logic

#### Publish model
- Models run locally or in controlled offline jobs
- Deploy CSV output is generated offline
- `publish_predictions.py` transforms routed deploy output into website-safe JSON
- Publish validations run
- Updated JSON is committed and deployed to Pages

## Performance Goals

### User-facing goals
- Public page loads should remain CDN-fast
- Premium board unlock should feel like a gated read, not a computation request
- No page should stall on backend model execution
- Logged-in volume should not degrade board speed as long as reads stay mostly static

### Operational goals
- Website should survive traffic spikes around weekend publish windows
- Premium endpoint should remain cheap enough to serve burst traffic
- Webhook processing should be idempotent and low-latency

## Current Bottlenecks

### 1. Premium board fetch path is functional but not fully edge-optimized
Current state:
- Worker can fetch premium data from `PREMIUM_DATA_SOURCE`

Risk:
- If every premium request forces a fresh fetch path without caching discipline, Worker traffic grows more expensive than necessary

Required action:
- Add explicit cache behavior for the premium board response
- Prefer edge caching with short freshness and clean invalidation by publish cycle

### 2. Auth model is still transitional
Current state:
- Worker token scaffolding exists and has already been used for controlled testing

Risk:
- Manual or developer-style token handling should not be the long-term member experience

Required action:
- Replace test-oriented token flow with real user identity issuance
- Prefer magic-link or lightweight authenticated session flow

### 3. Subscriber storage is good for entitlement, not full account product depth
Current state:
- KV stores subscriber state effectively

Risk:
- KV is not the right place to grow into rich profile/history/features data

Required action:
- Keep KV for entitlement lookups
- Introduce D1 later only for website/account metadata if needed

### 4. Observability is still light
Current state:
- Core functionality is proven

Risk:
- Failures could go unnoticed under real user scale

Required action:
- Add structured Worker logs
- Track route latency and error rates
- Add webhook failure visibility
- Monitor publish freshness

### 5. Publish remains operator-driven
Current state:
- Publish bridge exists and now has a wrapper

Risk:
- The system works, but operational consistency still depends on careful human use

Required action:
- Harden the publish workflow into an explicit operational routine with validation and post-publish checks

## Storage Responsibilities

### KV
Use KV for:
- subscriber entitlement state
- customer-to-subscription lookup
- subscription-to-record lookup
- cheap, fast authorization reads
- temporary publish metadata if required

Do not use KV for:
- rich user profile systems
- long-form audit history
- large analytics collections
- relational reporting

### D1
Introduce D1 only when the website needs:
- account metadata
- audit tables
- publish history registry
- founder cohort tracking
- support/admin lookups
- operational dashboards with relational queries

Do not introduce D1 yet for:
- premium board content
- prediction generation
- per-request market reads

### External database
Do not introduce an external database unless one of these becomes true:
- D1 proves insufficient for product/account complexity
- you need deep reporting workloads separated from user traffic
- B2B/API products require dedicated data services

The default near-term answer is:
- KV now
- D1 later if the website/account layer genuinely needs it
- no external DB until the product scale or commercial scope requires it

## Caching Strategy

## Premium board caching
Required design:
- Premium board remains a published artifact
- Worker verifies entitlement
- Worker returns cached premium content instead of refetching expensively on every request

Recommended behavior:
- cache premium board at the edge
- use short, explicit cache headers
- keep entitlement check separate from heavy content generation

Target model:
1. User requests premium board
2. Worker verifies auth/session and entitlement
3. Worker serves cached premium payload or fetches published artifact once and caches it

### Public board caching
- Public board should remain static JSON on Pages
- Let Cloudflare CDN do most of the work
- Publish cycle naturally refreshes the board

### HTML/asset caching
- Cache JS/CSS aggressively when fingerprinted or versioned
- Keep HTML fresh enough to reflect the latest board references and conversion copy

## Auth and Session Strategy

### Immediate direction
- Remove manual/dev token feel from the public product
- Move toward magic-link or clean session-based access

### Recommended architecture
Phase 1:
- Stripe remains billing truth
- Worker continues entitlement checks
- Add lightweight identity issuance

Phase 2:
- Introduce magic-link login or equivalent lightweight auth
- Tie user identity to subscriber state
- Replace token-paste UX entirely

Phase 3:
- Add account session lifecycle
- Add subscription management UX
- Add secure re-entry path for premium users without developer-style flows

### Auth principles
- logged in should mean identity is known
- entitlement should mean board access is allowed
- neither should trigger model computation

## Monitoring and Observability

### Required monitoring
- Worker route response times
- Worker error rate by route
- Stripe webhook success/failure counts
- Premium endpoint latency
- Premium endpoint response status distribution
- Publish freshness
- Premium board fetch/cache health

### Logging
Add structured logs for:
- checkout creation
- webhook acceptance
- entitlement updates
- premium fetch denials
- premium fetch successes
- session/auth failures

### Alert candidates
- webhook failure spikes
- `/api/premium/predictions` latency degradation
- repeated malformed or invalid auth attempts
- publish artifacts missing or stale

## Abuse Protection

### Needed controls
- rate limit premium token/session issuance
- rate limit premium data endpoint by identity and IP
- apply basic bot protection where appropriate
- keep cache-control disciplined so static reads stay cheap

### Why this matters
Without rate limiting, a light Worker can still become noisy under scraping or brute-force behavior.

The goal is not heavyweight user policing. The goal is to keep premium gating cheap, predictable, and resistant to obvious abuse.

## Publish Pipeline At Scale

### Core rule
Publishing remains an offline operational workflow.

### Recommended production flow
1. Offline models generate routed deploy CSV
2. Publish command transforms routed deploy output into website JSON
3. Validation runs automatically
4. JSON artifacts are committed/deployed
5. Pages redeploy serves new public board
6. Worker premium endpoint serves the new premium board

### Required publish guarantees
- publish source file must be explicit or safely auto-discovered
- validations must fail loudly
- artifact counts and tier mix should be summarized
- publish reports should be retained

### Current operator baseline
- `scripts/publish_latest_deploy.sh`

### Next workflow hardening steps
- Add explicit publish timestamp metadata
- Add board freshness display for operators
- Add a pre-push checklist for weekend board releases
- Add a tiny post-publish smoke check script against live Pages and Worker routes

## Recommended Phasing

### Phase A: Harden current architecture
- Keep Pages + Worker + KV + Stripe
- Add premium board edge caching
- Add better Worker logging
- Add rate limiting
- Remove any remaining dev-facing auth language from live user flow

### Phase B: Clean user identity
- Introduce magic-link or session auth
- Tie subscriber state to user identity cleanly
- Remove manual token workflow from customer-facing experience

### Phase C: Add account-grade metadata
- Introduce D1 only if account/profile/support/admin features require it
- Keep board delivery static and cached

### Phase D: Product expansion
- Slip auditor
- richer founder/pro tiers
- advanced diagnostics
- B2B/API surfaces

None of these later phases should break the core rule:
offline prediction generation, published boards, edge delivery.

## Explicit Recommendation
The right Odds Genius web architecture is:

- offline model engine
- static board publishing
- Cloudflare edge delivery
- Worker-based entitlement
- Stripe-backed subscription state
- lightweight user identity layer

This can become a strong 10,000-user system because it scales reads, not compute.

The path forward is not to make the site smarter at request time.
It is to make the published artifact, entitlement layer, and account flow cleaner, faster, and more observable.

## Action Checklist

### Now
- Keep predictions offline and published
- Keep premium board as published JSON behind Worker
- Add premium edge caching plan
- Add Worker route logging plan
- Add rate limiting plan

### Next
- Replace manual/dev-style auth UX with magic-link or session auth
- Add post-publish live smoke checks
- Add board freshness and publish metadata visibility

### Later
- Add D1 for website/account metadata if needed
- Add higher-tier plan support
- Add richer support/admin tools

## Final Rule
Odds Genius should scale like a publishing and entitlement system, not like a request-time prediction engine.
