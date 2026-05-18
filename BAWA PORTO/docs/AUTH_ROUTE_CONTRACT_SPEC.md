# Auth Route Contract Spec

## Purpose
This document defines the route-level API contract for the first real Odds Genius auth flow.

It is designed to replace public reliance on developer/test premium token handling with:
- magic-link request
- magic-link verification
- secure session cookie
- premium access through session-backed Worker entitlement

This spec is intentionally narrow.
It focuses on the first production-safe auth surface, not a full account platform.

## Core Principles

- Stripe remains billing truth
- `SUBSCRIBER_STATE` remains entitlement authority
- premium board remains a shared published artifact
- auth identifies the user
- entitlement authorizes premium access
- no route should trigger model inference

## Session Model

### Public user experience
Users should experience:
1. checkout
2. email verification
3. automatic premium unlock

They should not experience:
- token pasting
- manual API key handling
- developer-facing auth language

### Session transport
Use secure cookie-backed session handling.

Recommended cookie:
- `og_premium_session`

Recommended cookie properties:
- `HttpOnly`
- `Secure`
- `SameSite=Lax`
- path `/`
- explicit expiry

## Route Set

### Required routes
- `POST /api/auth/magic-link/request`
- `GET /api/auth/magic-link/verify`
- `POST /api/auth/logout`
- `GET /api/auth/session`

### Existing route kept
- `GET /api/premium/predictions`

This route should continue to work, but it should eventually be able to authorize through session cookie flow instead of developer/test token flow only.

## 1. POST /api/auth/magic-link/request

### Purpose
Accept an email address and request a one-time sign-in link.

### Request body
```json
{
  "email": "user@example.com"
}
```

### Validation rules
- `email` must be a string
- trimmed
- lowercased for matching
- basic email format validation

### Required backend behavior
- rate limit by IP
- rate limit by normalized email
- do not leak whether the email is subscribed
- create a one-time auth token with short TTL
- persist token record server-side
- send verification link via transactional email

### Success response
Always return a generic success-style response.

```json
{
  "ok": true,
  "status": "magic_link_requested",
  "message": "If the address is eligible, a sign-in link has been sent."
}
```

### Failure response
Use only for malformed request or server-side send failure.

Malformed request:
```json
{
  "ok": false,
  "status": "request_error",
  "message": "A valid email address is required."
}
```

Rate-limited:
```json
{
  "ok": false,
  "status": "rate_limited",
  "message": "Too many requests. Try again later."
}
```

Email provider failure:
```json
{
  "ok": false,
  "status": "email_send_failed",
  "message": "Unable to send sign-in link right now."
}
```

## 2. GET /api/auth/magic-link/verify

### Purpose
Validate the one-time magic-link token and establish a signed session cookie.

### Request shape
Token should be provided by query parameter.

Example:
`/api/auth/magic-link/verify?token=<opaque_token>`

### Validation rules
- token must exist
- token must be known
- token must not be expired
- token must not be previously consumed

### Required backend behavior
- resolve normalized email identity from token
- resolve matching subscriber identity
- confirm subscriber state is active or trialing
- mint secure session cookie
- mark magic-link token as consumed
- redirect to account or premium page

### Success behavior
Recommended:
- HTTP `302` or `303`
- set `og_premium_session`
- redirect to:
  - `/account.html?auth=success`
  - or `/premium.html?auth=success`

### Failure behavior
Redirect to a friendly error state rather than returning raw JSON in normal browser flow.

Recommended examples:
- `/account.html?auth=invalid`
- `/account.html?auth=expired`
- `/account.html?auth=inactive`

### Optional JSON failure mode
If later needed for API callers:
```json
{
  "ok": false,
  "status": "invalid_magic_link",
  "message": "This sign-in link is invalid or has expired."
}
```

## 3. POST /api/auth/logout

### Purpose
Clear the active premium session cookie.

### Request body
No body required.

### Success response
```json
{
  "ok": true,
  "status": "logged_out"
}
```

### Required behavior
- clear `og_premium_session`
- return success even if no session existed

## 4. GET /api/auth/session

### Purpose
Let the frontend check whether the user currently has a valid session and premium entitlement.

### Request body
None.

### Input source
- secure cookie

### Success response
```json
{
  "ok": true,
  "authenticated": true,
  "entitled": true,
  "email_hint": "u***@example.com",
  "customer_id": "cus_123",
  "subscription_id": "sub_123",
  "subscription_status": "active"
}
```

### Unauthenticated response
```json
{
  "ok": true,
  "authenticated": false,
  "entitled": false
}
```

### Notes
- `email_hint` should be masked
- do not expose sensitive session internals
- this route exists for frontend UX, not for trust decisions outside Worker auth

## 5. GET /api/premium/predictions

### Purpose
Return the protected premium board.

### Current state
Already live and protected by signed premium token verification.

### Migration target
This route should accept either:
- secure session cookie
- or transitional signed premium token during migration

### Long-term target
Session-backed auth should become primary.
Developer/test token flow should become internal-only.

### Success response
Current response shape should remain stable:
```json
{
  "ok": true,
  "generated_at": "2026-05-08T12:00:00.000Z",
  "subscriber_customer_id": "cus_123",
  "count": 28,
  "predictions": []
}
```

### Cache behavior
- entitlement/auth remains live
- shared premium payload may be edge-cached
- route may return:
  - `x-og-premium-cache: miss`
  - `x-og-premium-cache: hit`
  - `x-og-premium-cache: bypass`

## Token and Session Internals

## Magic-link token
Recommended implementation:
- opaque token
- stored server-side
- single use
- short TTL

Recommended record fields:
- `token`
- `email`
- `created_at`
- `expires_at`
- `consumed_at`
- `request_ip`
- optional `request_user_agent`

## Session token
Recommended payload:
- `customer_id`
- `subscription_id`
- `email`
- `exp`
- optional `session_version`

Recommended delivery:
- signed session cookie

## Storage Responsibilities

### KV
Good first fit for:
- magic-link token records
- small session support lookups if needed
- subscriber-state resolution

### D1
Introduce only when needed for:
- account audit history
- admin/support lookup
- richer session auditing
- identity normalization tables

## Error Taxonomy

### Suggested statuses
- `magic_link_requested`
- `request_error`
- `rate_limited`
- `email_send_failed`
- `invalid_magic_link`
- `expired_magic_link`
- `consumed_magic_link`
- `inactive_subscription`
- `authenticated`
- `logged_out`
- `missing_session`
- `invalid_session`
- `expired_session`

## Security Requirements

### Required
- no user-facing token handling
- generic magic-link request success responses
- single-use verification tokens
- secure cookie-backed session
- rate limiting
- replay protection
- audit logging

### Strongly recommended
- masked email hints only
- short magic-link TTL
- short-to-medium session TTL
- server-side revoke path if needed later

## Monitoring Fields

Track for auth routes:
- route
- ip
- normalized_email_hash or safe identity handle
- request outcome
- email send outcome
- verification outcome
- session creation outcome
- entitlement outcome
- latency_ms

## Migration Notes

### During transition
Allow:
- session-backed auth
- debug token flow behind internal/debug mode only

### After public rollout
Keep:
- session-backed auth

Retire from public UX:
- manual token storage
- token paste forms
- token-oriented copy

## Recommended First Build Order

1. `POST /api/auth/magic-link/request`
2. `GET /api/auth/magic-link/verify`
3. secure session cookie issuance
4. `GET /api/auth/session`
5. adapt `GET /api/premium/predictions` to accept session-backed auth path
6. `POST /api/auth/logout`

## Final Rule
The auth contract should make Odds Genius feel like a premium intelligence platform:
- pay
- verify email
- unlock

Not:
- pay
- obtain token
- paste token
- debug access manually
