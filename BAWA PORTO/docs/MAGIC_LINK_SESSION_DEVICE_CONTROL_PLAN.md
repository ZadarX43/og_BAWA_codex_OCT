# Magic-Link Session & Device Control Plan

## Goal

Keep the current magic-link authentication model, but strengthen it with session and device controls so premium access feels intentional, harder to share casually, and commercially safer.

This is not a password-auth migration.
It is a control layer on top of the existing:

- magic-link verification
- signed session cookie
- D1-backed account state
- premium entitlement checks

## Product Position

Odds Genius should continue to feel:

- calm
- low-friction
- selective
- modern

It should not force users into:

- password creation
- password resets
- credential management overhead

Magic-link auth stays the default because it fits the current product tone better.

## What Magic-Link Already Gives Us

- no reusable password to share
- sign-in depends on access to the user email account
- device/browser session persists after verification
- lower support burden than password auth

## What Magic-Link Does Not Solve Alone

By itself, magic-link auth does not fully prevent:

- forwarding a sign-in email
- signing into multiple devices intentionally
- long-lived premium session reuse across devices

So the anti-sharing layer must come from session/device controls, not from passwords alone.

## Core Outcome

Build a session/device model where:

- a user can still sign in cleanly with email
- premium access feels tied to known devices/sessions
- account sharing becomes more inconvenient and more visible
- sensitive actions can require light re-verification

## Phase 1 — Session Registry

Add a D1-backed session registry for authenticated account sessions.

Each session should store:

- `session_id`
- `user_id`
- `created_at`
- `last_seen_at`
- `expires_at`
- `revoked_at`
- `ip_hash`
- `user_agent_hash`
- `device_label`
- `is_current`
- `is_primary`

### Notes

- store hashes or safe metadata where possible, not raw high-risk identifiers unless required
- `device_label` should be product-safe, e.g.:
  - `Hugh's MacBook Pro`
  - `iPhone Safari`
  - `Chrome on Windows`

## Phase 2 — Active Session Limit

Define a bounded active-session policy.

Suggested first policy:

- allow up to `3` active sessions per account
- if a fourth session is created:
  - revoke the oldest non-primary session
  - or force the user to manage sessions before completing sign-in

Recommended first release:

- silent bounded rotation for oldest non-primary session

This keeps friction low while making casual account spreading harder.

## Phase 3 — Session Management UI

Add an account section:

- `Signed-in devices`

This should show:

- current device
- recent devices/sessions
- last active time
- primary session marker
- revoke action

Example actions:

- `This device`
- `Make primary`
- `Sign out other devices`
- `Revoke`

## Phase 4 — Primary Device Behavior

Add optional product logic for a preferred primary session/device.

This should not fully block secondary sessions at first.

Instead:

- primary session gets the smoothest premium continuity
- secondary sessions can still work, but are more likely to:
  - lose older sessions
  - require re-verification
  - have reduced persistence

This creates soft anti-sharing pressure without overly punishing legitimate users.

## Phase 5 — Re-Verification For Sensitive Actions

Require fresh verification for higher-risk actions such as:

- changing email
- linking or relinking Telegram
- changing billing/subscription ownership
- signing in on a new device after active-session cap is reached
- promoting a new primary device

This should use:

- short re-verification via email magic link

Not:

- password prompt

## Phase 6 — Session Risk Rules

Add lightweight session heuristics for account protection.

Examples:

- unusual number of new devices in a short window
- repeated sign-ins from multiple device types rapidly
- primary device replaced too often
- frequent re-linking patterns

Response options:

- soft flag only
- require re-verification
- revoke oldest sessions
- temporarily prevent new premium session creation

## Recommended UX Copy

Keep the language calm and non-accusatory.

Use:

- `Signed-in devices`
- `Current session`
- `Recent sessions`
- `Verify this device`
- `Sign out other devices`
- `This helps protect your premium access`

Avoid:

- `suspicious activity`
- `account abuse`
- `fraud`
- `unauthorised device`

unless the risk level truly requires stronger language.

## Suggested Account Surface

Inside `Account`, add:

- `Account`
- `Settings`
- `Preferences`
- `Language`
- `Billing`
- `Devices`
- `Help`

The new `Devices` section becomes the user-facing home of this system.

## Data / Backend Work

### Additions

- D1 session table
- Worker session issue/revoke helpers
- current-session resolver
- session rotation rules
- revoke-other-sessions action
- device-summary serializer for account UI

### Worker Routes

Suggested routes:

- `GET /api/account/sessions`
- `POST /api/account/sessions/revoke`
- `POST /api/account/sessions/revoke-others`
- `POST /api/account/sessions/make-primary`
- `POST /api/account/sessions/reverify`

## Initial Commercial Policy

Recommended first live policy:

- magic link stays the only auth method
- up to `3` active sessions
- one optional `primary` session/device
- signed-in devices visible in account
- user can revoke older sessions manually
- sensitive actions require fresh verification

This is strong enough for an early premium product without becoming over-engineered.

## Why This Is Better Than Adding Passwords Now

Passwords would add:

- reset flow
- password storage complexity
- more support burden
- more friction
- more sharable credentials

Session/device controls solve the more relevant premium problem directly:

- access continuity
- premium control
- anti-sharing pressure
- cleaner account trust

## Deliverables

### Phase A

- D1 session schema
- Worker session registry
- account `Devices` view
- revoke current/other sessions support

### Phase B

- primary device/session behavior
- bounded active session cap
- re-verification for sensitive actions

### Phase C

- soft risk heuristics
- better session visibility
- support/admin audit surfaces later if needed

## Recommended Next Task

**Task 78 — Session Registry Schema + Worker Contract**

That should define:

- D1 schema
- cookie/session mapping model
- session rotation rules
- device labeling rules
- Worker API contracts

That is the correct technical bridge from this product plan into implementation.
