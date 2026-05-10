# Session Registry Schema And Worker Contract

## Purpose

Define the implementation contract for the magic-link session/device control layer.

This is the technical follow-on from:

- [MAGIC_LINK_SESSION_DEVICE_CONTROL_PLAN.md](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/docs/MAGIC_LINK_SESSION_DEVICE_CONTROL_PLAN.md)

It covers:

- D1 schema for sessions
- cookie/session mapping
- active-session cap rules
- revoke and make-primary flows
- account `Devices` API contract

## Scope

This work sits inside the existing auth model:

- email magic-link verification
- signed session cookie
- D1-backed account state
- Worker-authenticated premium access

It does **not** introduce:

- password auth
- password reset flows
- password storage

## Core Model

The browser session cookie should point to a tracked session record in D1.

That session record becomes the authoritative object for:

- current signed-in device/session
- recent device history
- revocation
- primary device preference
- active session limits

## D1 Schema

### Table: `account_sessions`

Suggested columns:

- `id TEXT PRIMARY KEY`
- `user_id TEXT NOT NULL`
- `session_token_hash TEXT NOT NULL UNIQUE`
- `device_label TEXT`
- `user_agent_hash TEXT`
- `ip_hash TEXT`
- `session_kind TEXT NOT NULL DEFAULT 'browser'`
- `is_primary INTEGER NOT NULL DEFAULT 0`
- `is_revoked INTEGER NOT NULL DEFAULT 0`
- `issued_at TEXT NOT NULL`
- `last_seen_at TEXT NOT NULL`
- `expires_at TEXT NOT NULL`
- `revoked_at TEXT`
- `revoke_reason TEXT`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

### Recommended indexes

- `INDEX idx_account_sessions_user_id ON account_sessions(user_id)`
- `INDEX idx_account_sessions_user_revoked ON account_sessions(user_id, is_revoked)`
- `INDEX idx_account_sessions_user_primary ON account_sessions(user_id, is_primary)`
- `INDEX idx_account_sessions_expires_at ON account_sessions(expires_at)`

## Cookie / Session Mapping

### Current state

The signed cookie today effectively carries authenticated session state for the Worker.

### Target state

The cookie should resolve to one D1 session record.

Recommended model:

1. issue a random `session_id`
2. issue a random opaque `session_token`
3. store only `session_token_hash` in D1
4. sign the cookie payload with:
   - `session_id`
   - `session_token`
   - issued timestamp

### Validation flow

On each authenticated request:

1. verify cookie signature
2. read `session_id`
3. look up session row
4. compare stored `session_token_hash`
5. check:
   - not revoked
   - not expired
   - belongs to active user
6. refresh `last_seen_at`

If any check fails:

- clear cookie
- treat as signed out

## Device Labeling Rules

Device labels are for user-facing account management, not deep fingerprinting.

Suggested derivation:

- browser family
- operating system family
- optional “current device” marker

Examples:

- `Safari on iPhone`
- `Chrome on Mac`
- `Firefox on Windows`

Avoid overly invasive fingerprinting.

## Active Session Cap

### Default policy

- max `3` active non-revoked sessions per account

### Definition

An active session is:

- not revoked
- not expired

### On new session creation

1. load all active sessions for user
2. if active count is below cap:
   - create session normally
3. if active count meets or exceeds cap:
   - revoke oldest non-primary session first
   - if only primary remains plus cap is still exceeded, revoke oldest remaining non-current session

### First release behavior

Prefer silent bounded rotation over blocking sign-in.

That keeps UX smooth while reducing spread.

## Primary Session Rules

### Meaning

`is_primary = 1` means:

- preferred continuity session/device
- last to be rotated out automatically

### Constraints

- only one primary session per user at a time

### On make-primary

1. verify request comes from valid signed-in session
2. verify target session belongs to same user
3. unset existing primary session
4. set target session as primary
5. update timestamps

## Revoke Flow

### Revoke one session

Used when the user chooses:

- `Sign out this device`
- `Revoke` on a specific device

Flow:

1. verify current session
2. verify target session belongs to same user
3. mark:
   - `is_revoked = 1`
   - `revoked_at = now`
   - `revoke_reason = 'user_revoked'`
4. if target is current session:
   - clear cookie immediately in response

### Revoke other sessions

Used when the user chooses:

- `Sign out other devices`

Flow:

1. verify current session
2. revoke all user sessions except current session
3. preserve current session

## Re-Verification Flow

Sensitive actions should require fresh verification rather than a password.

Examples:

- make another session primary
- revoke many sessions
- change email
- alter future billing ownership

### Model

1. user initiates sensitive action
2. Worker issues short-lived verification intent
3. email magic link confirms intent
4. Worker completes action after verification

This can reuse the existing magic-link infrastructure with an added `intent_type`.

## Worker API Contract

### `GET /api/account/sessions`

Returns current and recent sessions for the signed-in user.

#### Response

```json
{
  "status": "account_sessions_loaded",
  "sessions": [
    {
      "id": "sess_123",
      "device_label": "Safari on iPhone",
      "is_current": true,
      "is_primary": true,
      "issued_at": "2026-05-10T09:12:00Z",
      "last_seen_at": "2026-05-10T10:02:00Z",
      "expires_at": "2026-06-09T09:12:00Z",
      "session_kind": "browser"
    }
  ]
}
```

### `POST /api/account/sessions/revoke`

Revokes one session.

#### Request

```json
{
  "session_id": "sess_123"
}
```

#### Response

```json
{
  "status": "account_session_revoked",
  "revoked_session_id": "sess_123"
}
```

### `POST /api/account/sessions/revoke-others`

Revokes all sessions for the user except the current one.

#### Response

```json
{
  "status": "account_other_sessions_revoked",
  "revoked_count": 2
}
```

### `POST /api/account/sessions/make-primary`

Promotes a session to primary.

#### Request

```json
{
  "session_id": "sess_123"
}
```

#### Response

```json
{
  "status": "account_session_primary_updated",
  "primary_session_id": "sess_123"
}
```

### `POST /api/account/sessions/reverify`

Starts a short re-verification flow for a sensitive action.

#### Request

```json
{
  "intent_type": "make_primary",
  "session_id": "sess_123"
}
```

#### Response

```json
{
  "status": "account_session_reverify_sent"
}
```

## Account Devices UI Contract

Inside the account shell, add:

- `Devices`

The UI should show:

- current device
- primary device marker
- recent activity
- revoke action
- sign out others action

Suggested labels:

- `Current device`
- `Primary`
- `Last active`
- `Make primary`
- `Revoke`
- `Sign out other devices`

## Session Lifetime

Recommended first-release defaults:

- cookie/session lifetime: `30 days`
- refresh `last_seen_at` on authenticated use
- optional inactivity expiry later

Future refinement:

- shorter persistence for non-primary sessions
- longer continuity for primary session

## Failure Handling

### If session row is missing

- clear cookie
- treat as signed out

### If session is revoked

- clear cookie
- show calm signed-out state

### If session exceeds active policy

- rotate older session out
- do not block normal sign-in in first release

## Logging / Audit

Track the following events:

- session_created
- session_validated
- session_revoked
- session_primary_changed
- session_rotated_due_to_cap
- session_reverify_sent

This should be light operational logging, not a heavy admin system yet.

## Recommended Build Order

### Step 1

- add `account_sessions` D1 table
- issue tracked sessions on magic-link verify
- validate session against D1 on authenticated requests

### Step 2

- add account `Devices` read endpoint
- add revoke and revoke-others actions

### Step 3

- add primary-session support
- add bounded active-session rotation

### Step 4

- add re-verification intents for sensitive actions

## Non-Goals For This Phase

- password auth
- SSO
- biometric auth
- full fraud detection system
- admin console

## Recommended Next Task

**Task 79 — D1 Session Table + Worker Session Issuance**

That should implement:

- migration for `account_sessions`
- session creation on magic-link verify
- session lookup on authenticated requests
- revoke/expiry-safe cookie clearing
