# Premium Endpoint Caching Plan

## Purpose
This document defines the caching strategy for `GET /api/premium/predictions` so Odds Genius can serve premium board traffic efficiently at scale without coupling user reads to repeated origin fetches.

The goal is not to cache authentication.
The goal is to cache the published premium board payload while keeping entitlement checks lightweight and correct.

## Current Baseline

### Current request path
The current Worker flow is:

1. Frontend calls `GET /api/premium/predictions`
2. Worker verifies the premium token and subscriber state
3. Worker calls `fetchPremiumSource()`
4. Worker fetches `PREMIUM_DATA_SOURCE`
5. Worker sanitizes the response and returns it to the caller

Current implementation location:
- [worker/src/index.js](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/worker/src/index.js:515)

### What is good about the current baseline
- Predictions remain offline and published
- Premium access is gated by Worker entitlement
- Premium payload is schema-filtered before return
- No model computation runs in the request path

### What is missing
- No explicit edge caching strategy for the premium payload
- Worker may refetch the premium JSON unnecessarily under load
- No explicit cache invalidation model tied to the publish cycle
- No observability around cache hit/miss behavior

## Core Principle
Premium board content should be cached.
Subscriber entitlement should not.

That means:
- identity/session verification remains live and lightweight
- entitlement verification remains live and lightweight
- premium board payload becomes an edge-cached published artifact

## Target Behavior

### Desired flow
1. User requests premium board
2. Worker verifies auth/session and subscriber state
3. Worker serves cached premium board payload if available
4. If cache is cold or expired, Worker fetches the published premium JSON once
5. Worker sanitizes and stores the response in cache
6. Subsequent entitled users are served from the cache

This preserves security while reducing repeated premium-source fetches.

## What Should Be Cached

### Cacheable
- Sanitized premium predictions payload
- Optional `generated_at` / publish metadata
- Optional publish summary metadata if later attached to premium responses

### Not cacheable per user
- Auth/session validity
- Subscriber entitlement result
- Token issuance response
- Stripe webhook writes
- Checkout or portal responses

## Recommended Caching Layer

### Primary recommendation
Use Worker edge caching for the premium board payload.

Best current fit:
- Cloudflare Worker cache for the published premium board response
- A cache key tied to the premium source URL

Why:
- no need to introduce a new storage dependency yet
- response payload is identical across entitled users
- invalidation can follow publish freshness rules
- keeps request-time work small

## Recommended Implementation Model

### Worker behavior
Split the current premium route into two conceptual steps:

1. `verifyPremiumAccess()`
2. `loadCachedPremiumBoard()`

The second step should:
- check cache first
- fall back to fetching `PREMIUM_DATA_SOURCE`
- sanitize rows
- write sanitized payload to cache
- return cached/sanitized result

### Important design choice
Cache the sanitized payload, not the raw upstream response.

Why:
- guarantees the cached object already conforms to the premium schema allowlist
- avoids repeated sanitization cost under load
- reduces risk of accidentally returning upstream-only fields later

## Cache Key Strategy

### Default key
Use a shared cache key for the current premium source, for example:
- `premium-board::<PREMIUM_DATA_SOURCE>`

### Better future key
Use a key that includes publish freshness if available:
- `premium-board::<PREMIUM_DATA_SOURCE>::<generated_at>`

This lets new publishes naturally rotate the cache without blunt cache purges.

If `generated_at` is absent, fallback to:
- `premium-board::<PREMIUM_DATA_SOURCE>`

## Freshness Policy

### Recommended TTL
Use a short, explicit TTL.

Good starting policy:
- Edge freshness: 5 to 15 minutes
- Browser caching: minimal or private depending on auth delivery

Why:
- premium board does not change minute-by-minute
- publish cadence is operational, not continuous real-time
- short TTL gives safety while still offloading repeated reads

### Recommended response behavior
Worker response should clearly set:
- edge caching policy
- browser caching policy

Recommended direction:
- edge cache allowed
- browser cache conservative

This avoids stale board issues on the client while still letting Cloudflare absorb traffic.

## Invalidation Strategy

### Phase 1: Time-based
Use TTL-only caching initially.

This is acceptable because:
- board publishes are discrete events
- traffic scale is still modest
- operational complexity stays low

### Phase 2: Publish-aware
When publish workflow matures, include publish metadata in the cache key:
- `generated_at`
- publish timestamp
- source file fingerprint

That gives soft automatic invalidation when the board changes.

### Phase 3: Explicit purge hook
If needed later, the publish workflow can trigger a known cache-rotation signal or purge event after successful board deployment.

Not required yet.

## Frontend Considerations

### Frontend should not change the model
The frontend still calls:
- `GET /api/premium/predictions`

No direct premium JSON fetch should happen from the browser when the Worker-protected path is intended.

### Frontend caching
Frontend should treat premium data as:
- protected
- requestable
- not something to aggressively persist locally unless the auth model later explicitly supports it

## Security Model

### Keep live
- token/session verification
- KV subscriber-state verification

### Cache only shared board content
The cached payload must not contain user-specific secrets or user-tailored content.

That is why this approach is safe:
- all entitled users receive the same published premium board
- access decision is still performed before payload return

## Monitoring Plan

Add route-level observability for:
- cache hit
- cache miss
- upstream premium source fetch success
- upstream premium source fetch failure
- route latency
- premium source response size

### Suggested log fields
- `route`
- `cache_status`
- `source_url`
- `generated_at`
- `row_count`
- `subscriber_status`
- `latency_ms`

## Abuse and Rate Limiting

Caching reduces load, but it does not replace abuse controls.

Add or plan:
- rate limiting on premium endpoint by IP and/or identity
- rate limiting on token/session issuance
- detection for repeated invalid auth attempts

Why:
- premium endpoint is still protected content
- cacheable content can still be hammered if auth attempts are cheap

## Failure Behavior

### If premium source fetch fails and cache exists
Preferred behavior:
- serve the last valid cached payload for a limited stale window
- emit warning logs

This protects the user experience during brief source hiccups.

### If premium source fetch fails and no cache exists
Return the current error contract with clear operator-facing logging.

## Publish Workflow Relationship

This caching plan assumes:
- predictions are still published offline
- `publish_predictions.py` remains the exporter
- premium board remains a static artifact deployed to Pages

The cache exists to reduce repeated reads of that artifact.
It must never become a substitute for the publish pipeline.

## Implementation Phases

### Phase A: Low-risk caching
- Add edge cache lookup/write around sanitized premium payload
- Add cache status logging
- Keep TTL-only freshness

### Phase B: Publish-aware rotation
- Include publish freshness in cache key if available
- Add response metadata for board freshness

### Phase C: Resilience hardening
- stale-if-error behavior
- explicit operator-facing cache diagnostics

## Recommended First Implementation

The first safe implementation should do exactly this:

1. Verify token/session and KV entitlement
2. Check cache for a sanitized premium board payload
3. If hit, return it
4. If miss, fetch `PREMIUM_DATA_SOURCE`
5. Sanitize the payload
6. Store the sanitized payload in edge cache with short TTL
7. Return the sanitized payload

No new DB.
No live compute.
No user-specific cached content.

## Explicit Recommendation
Odds Genius should scale premium reads like a protected static asset, not like a dynamic personalized API.

The Worker should remain:
- an entitlement gate
- a payload sanitizer
- a cache-aware edge response layer

That is the right path to serving premium traffic for 10,000 users while keeping model execution fully offline.
