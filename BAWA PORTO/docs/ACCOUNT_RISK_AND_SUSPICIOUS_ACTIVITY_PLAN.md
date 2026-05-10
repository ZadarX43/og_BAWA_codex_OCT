# Account Risk And Suspicious Activity Plan

## Purpose

Define the next backend control layer after session and device management.

This document explains how Odds Genius should:

- identify an account as one coherent backend entity
- track device, session, IP, and billing-linked activity safely
- flag unusual patterns for review
- suspend or reinstate accounts when needed
- support internal admin notes without turning the product into a hostile auth experience

This is not a fraud-platform build.
It is a calm, premium-access control layer for a subscription intelligence product.

## Core Product Position

User-facing experience should stay:

- low-friction
- calm
- non-accusatory
- email-magic-link first

Backend controls should still let us:

- detect casual sharing
- detect suspicious device churn
- detect unusual session spread
- track review outcomes
- suspend access if necessary

## Scope

This plan covers:

- backend account identity model
- session / IP / device heuristics
- review flags
- suspension / reinstatement flows
- admin notes and billing-risk notes

This plan does not introduce:

- password auth
- invasive fingerprinting
- full KYC
- raw payment-card storage

## Product Truth

Odds Genius needs one internal account identity that joins together:

- user record
- subscription record
- notification preferences
- Telegram link state
- session/device state
- auth events
- risk flags
- admin notes

That lets us reason about the account as a whole instead of only reacting to one session at a time.

## Backend Account Identity Model

## Central Account Object

The effective backend account identity should be built around:

- `users`
- `subscriptions`
- `notification_preferences`
- `telegram_links`
- `account_sessions`
- `auth_events`

To that, add a risk layer:

- `account_risk_state`
- `account_risk_flags`
- `account_admin_notes`

## Identity Join Keys

Preferred stable joins:

- `user_id`
- `email_normalized`
- `stripe_customer_id`
- `stripe_subscription_id`

Secondary operational joins:

- `telegram_user_id`
- `telegram_chat_id`
- active `session_id`

## Risk-Oriented Account Summary

Each account should be reducible into one backend summary view:

- `user_id`
- verified email
- subscription status
- linked Telegram status
- active session count
- recent session count
- primary device/session
- recent IP-hash diversity
- recent device-label diversity
- current risk level
- current account status
- open flags count
- last review outcome

## Data Safety Principle

Store what helps us make account decisions.
Avoid storing raw sensitive information unless truly required.

Preferred:

- hashed IP hint
- hashed user-agent hint
- derived device label
- Stripe customer/subscription references
- internal admin notes

Avoid:

- raw payment card details
- unnecessary geolocation precision
- invasive browser fingerprinting

## Recommended New Tables

## `account_risk_state`

One row per user.
This is the current backend risk posture for the account.

Suggested fields:

- `user_id TEXT PRIMARY KEY`
- `account_status TEXT NOT NULL DEFAULT 'active'`
- `risk_level TEXT NOT NULL DEFAULT 'low'`
- `risk_score INTEGER NOT NULL DEFAULT 0`
- `review_status TEXT NOT NULL DEFAULT 'clear'`
- `last_risk_event_at TEXT`
- `last_reviewed_at TEXT`
- `last_reviewed_by TEXT`
- `suspended_at TEXT`
- `suspension_reason TEXT`
- `reinstated_at TEXT`
- `reinstatement_reason TEXT`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

### Suggested enums

`account_status`

- `active`
- `restricted`
- `suspended`
- `closed`

`risk_level`

- `low`
- `medium`
- `high`
- `critical`

`review_status`

- `clear`
- `watch`
- `manual_review`
- `restricted`
- `suspended`

## `account_risk_flags`

Append-only or mostly append-only event log for specific concerns.

Suggested fields:

- `id TEXT PRIMARY KEY`
- `user_id TEXT NOT NULL`
- `flag_type TEXT NOT NULL`
- `severity TEXT NOT NULL`
- `flag_status TEXT NOT NULL DEFAULT 'open'`
- `source TEXT NOT NULL`
- `summary TEXT NOT NULL`
- `evidence_json TEXT`
- `opened_at TEXT NOT NULL`
- `resolved_at TEXT`
- `resolved_by TEXT`
- `resolution_note TEXT`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

### Example `flag_type`

- `high_device_churn`
- `high_ip_spread`
- `rapid_session_rotation`
- `shared_access_pattern`
- `billing_identity_mismatch`
- `telegram_relink_pattern`
- `manual_support_concern`

### Example `source`

- `session_heuristic`
- `billing_review`
- `telegram_review`
- `support_manual`
- `admin_manual`

## `account_admin_notes`

Internal operator notes for support, billing context, and review decisions.

Suggested fields:

- `id TEXT PRIMARY KEY`
- `user_id TEXT NOT NULL`
- `note_type TEXT NOT NULL`
- `visibility TEXT NOT NULL DEFAULT 'internal'`
- `content TEXT NOT NULL`
- `author_id TEXT`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

### Example `note_type`

- `billing_note`
- `risk_note`
- `support_note`
- `reinstatement_note`
- `telegram_note`

Important:

- these are internal notes only
- they should never leak to the customer-facing account shell

## Session / IP / Device Risk Heuristics

## Heuristic Philosophy

Do not suspend on one weak signal.
Accumulate evidence across:

- sessions
- device labels
- IP-hash spread
- timing
- subscription/account ownership context

## Heuristic Set

### 1. High device churn

Trigger idea:

- more than `4` distinct device sessions created in `7 days`

Interpretation:

- normal for a few legitimate devices
- suspicious if repeated and paired with session rotation

Recommended effect:

- open `medium` flag
- increase risk score

### 2. High IP spread

Trigger idea:

- many distinct IP hashes across a short window
- especially if paired with multiple device labels

Interpretation:

- could be travel, mobile switching, VPNs
- should not auto-suspend alone

Recommended effect:

- open `watch` flag first
- escalate only when combined with device/session spread

### 3. Rapid session rotation

Trigger idea:

- repeated sign-ins causing frequent revoke/reissue patterns
- especially after active-session cap enforcement

Interpretation:

- often a strong sharing indicator

Recommended effect:

- open `medium` or `high` flag
- consider temporary new-session restriction

### 4. Shared access pattern

Trigger idea:

- device types and IP spread imply multiple people using the same premium account concurrently

Examples:

- iPhone Safari, Chrome Windows, Chrome Mac, Edge Windows all active within a short period
- current session constantly changes primary device

Recommended effect:

- move review status to `manual_review`
- allow operator decision before suspension

### 5. Telegram relink churn

Trigger idea:

- repeated Telegram relinking or chat-id switching in a short time

Interpretation:

- possible sharing or account handoff

Recommended effect:

- open `medium` flag
- optionally require re-verification before future relinking

### 6. Billing identity mismatch

Trigger idea:

- support-side evidence or repeated email/subscription mismatch behavior

Interpretation:

- not necessarily malicious
- but important for entitlement ownership review

Recommended effect:

- admin note + manual review flag

## Risk Scoring

Use a simple additive score first.

Example only:

- high device churn: `+20`
- high IP spread: `+15`
- rapid session rotation: `+25`
- repeated revoke/recreate pattern: `+20`
- Telegram relink churn: `+15`
- manual support concern: `+20`
- confirmed false positive resolution: `-20`
- verified reinstatement: reset toward baseline

### Suggested risk bands

- `0-19`: low
- `20-39`: medium
- `40-59`: high
- `60+`: critical

The score should inform review priority, not replace judgment.

## Review Flags

## Flag Lifecycle

Each flag should move through:

- `open`
- `investigating`
- `resolved`
- `dismissed`

## Resolution Examples

- `false_positive_travel`
- `legitimate_multi_device_use`
- `sharing_confirmed`
- `billing_owner_verified`
- `support_override`

## Review Queue Purpose

The review queue should answer:

- which accounts need attention now
- what evidence caused the concern
- what changed recently
- what action, if any, has already been taken

## Suspend / Reinstate Flows

## Suspension Principles

Suspension should be:

- rare
- evidence-based
- reversible
- logged clearly

Do not suspend on one weak heuristic.
Prefer:

- flag
- review
- restrict
- suspend if needed

## Soft Restriction State

Before full suspension, support a `restricted` state.

Effects:

- current session may remain active briefly
- new session creation can be blocked
- Telegram delivery can be paused
- account is prompted to re-verify if needed

This is useful when confidence is not yet high enough for full suspension.

## Full Suspension State

Effects:

- session access denied
- premium endpoints blocked
- queued Telegram alerts halted
- account shell shows a calm support message

Recommended user-facing tone:

- calm
- non-accusatory
- support-oriented

Example:

`We paused premium access on this account while we review account security. Please contact support if this looks wrong.`

## Reinstatement

Reinstatement should require:

- review note
- explicit reinstatement reason
- optional flag cleanup or status downgrade

Recommended recorded fields:

- `reinstated_at`
- `reinstatement_reason`
- `reviewed_by`

## Admin Notes / Billing-Risk Notes Surface

## Why This Matters

Support and billing context often explains patterns that heuristics alone cannot.

Examples:

- legitimate travel
- customer changed email
- household usage clarified
- manual goodwill reinstatement
- known subscription ownership edge case

## Admin Surface Goals

Internal account admin view should eventually show:

- account identity summary
- subscription summary
- session/device summary
- open risk flags
- auth event timeline
- internal notes
- current account status
- quick actions:
  - restrict
  - suspend
  - reinstate
  - revoke all sessions

## Billing Notes

This should be about account/billing context, not storing card secrets.

Good examples:

- `Subscriber confirmed billing ownership via support on 2026-05-10`
- `Refund dispute under review`
- `Customer requested email ownership correction`
- `Stripe subscription restored after failed webhook delay`

Avoid storing:

- full card details
- unnecessary personal financial data

## Suspicious Activity Timeline

Eventually, admin review should be able to pull one timeline across:

- magic-link requests
- magic-link verifies
- session creation
- session revoke events
- make-primary events
- Telegram link changes
- premium access failures
- review flag openings/resolutions
- manual admin actions

This can mostly be built on top of:

- `auth_events`
- `account_sessions`
- `account_risk_flags`
- `account_admin_notes`

## Recommended Phase Order

### Phase 1

- write this plan
- keep device actions live
- continue logging auth/device actions

### Phase 2

- add `account_risk_state`
- add `account_risk_flags`
- add `account_admin_notes`

### Phase 3

- compute first heuristics from:
  - active session count
  - recent session count
  - IP-hash diversity
  - device-label diversity
  - relink churn

### Phase 4

- add internal review/admin surface
- add restrict/suspend/reinstate actions

### Phase 5

- add customer-facing restricted/suspended messaging
- add support handoff flow

## Immediate Recommendation

The next implementation after device actions should not be automatic suspension.

It should be:

- risk-state schema
- flag/event recording
- admin review readiness

That gives us:

- visibility first
- enforcement second

Which is the right order for a premium intelligence product.
