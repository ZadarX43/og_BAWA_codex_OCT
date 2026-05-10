# Internal Account Review Surface Plan

## Purpose

Define the first internal/admin review surface for Odds Genius accounts.

This sits on top of:

- magic-link auth
- subscription-linked account identity
- session/device controls
- account risk state
- risk flags
- admin notes

The goal is to give internal operators a calm, evidence-led workspace for reviewing accounts without exposing internal risk language to customers.

## Product Boundary

This surface is:

- internal only
- admin / support / operator facing
- separate from the member account shell

This surface is not:

- a public support page
- a customer settings page
- a security scare-screen

## Core Outcome

An internal operator should be able to answer:

- who this account belongs to
- whether billing is active
- what devices and sessions are active
- whether unusual patterns are present
- what notes already exist
- what action has already been taken
- whether to leave the account alone, restrict it, suspend it, or reinstate it

## Scope

This plan defines the first internal/admin view for:

- account summary
- open flags
- auth/session timeline
- admin notes
- restrict / suspend / reinstate actions

## Internal UX Principle

This should feel like:

- a review desk
- a support console
- an account operations surface

It should not feel like:

- a fraud panic board
- a noisy security dashboard
- an overbuilt SOC interface

The tone should stay:

- calm
- evidence-led
- legible
- operational

## Primary Use Cases

### 1. Routine review

An operator checks a normal account to confirm:

- subscription is active
- sessions look ordinary
- no open flags need action

### 2. Suspicious session spread review

An operator reviews an account flagged for:

- high device spread
- unusual session churn
- repeated primary-device changes

### 3. Billing-linked review

An operator checks:

- ownership questions
- refund/dispute context
- subscription mismatch context

### 4. Reinstatement review

An operator restores access after:

- false positive
- support resolution
- clarified ownership

## Surface Structure

The internal review surface should have five primary sections:

1. Account summary
2. Open flags
3. Auth and session timeline
4. Internal notes
5. Actions

## 1. Account Summary

This is the top summary layer.

It should show:

- `user_id`
- verified email
- subscription status
- Stripe customer reference
- Stripe subscription reference
- Telegram link status
- account status
- risk level
- risk score
- review status
- open flags count
- active session count
- primary device
- last risk event timestamp

### Recommended layout

Top header:

- email / account identity
- membership status chip
- account status chip
- risk level chip

Right-side key facts:

- active sessions
- primary device
- last reviewed
- last risk event

## 2. Open Flags

This is the operator’s main review queue inside the account.

Each flag card should show:

- `flag_type`
- severity
- status
- source
- summary
- opened time
- linked evidence snapshot

### Evidence snapshot examples

- active session count
- distinct device labels
- last three device labels
- recent IP-hash diversity
- trigger source
- related action timestamp

### Flag actions

Per flag:

- `Mark investigating`
- `Resolve`
- `Dismiss`
- `Add note`

## 3. Auth / Session Timeline

This is the chronological evidence layer.

It should combine:

- magic-link requested
- magic-link verified
- session created
- session revoked
- revoke others
- make primary
- Telegram link completed
- premium access failures
- risk flagged events
- restriction / suspension / reinstatement events

### Timeline row fields

- timestamp
- event type
- short summary
- device label if relevant
- IP or user-agent hint if relevant
- operator note link if relevant

### Timeline purpose

The operator should be able to see:

- when the account changed
- whether current flags make sense
- whether behavior is escalating or stabilizing

## 4. Internal Notes

This section should surface:

- billing notes
- risk notes
- support notes
- reinstatement notes
- Telegram notes

### Note fields

- type
- author
- created time
- updated time
- content

### Note creation

Operators should be able to add notes like:

- `Billing owner verified through support`
- `Customer travelling between UK and Portugal`
- `Household usage clarified; no suspension needed`
- `Repeated Telegram relinking under review`

Important:

- notes are internal only
- notes must never render in the customer-facing account area

## 5. Actions

This is the operator control panel for account status decisions.

## Restrict

Purpose:

- soft control before full suspension

Effects:

- block new session creation if desired
- optionally pause Telegram delivery
- keep current account visible in review state

When to use:

- unusual activity needs verification
- evidence is meaningful but not final

## Suspend

Purpose:

- fully pause access

Effects:

- premium session access denied
- premium endpoints blocked
- alert delivery halted
- account status becomes `suspended`
- suspension reason recorded

When to use:

- repeated strong evidence of sharing or abuse
- clear ownership or billing-risk issue
- manual operator decision

## Reinstate

Purpose:

- restore access after review

Effects:

- account status returns to `active`
- reinstatement reason recorded
- last reviewed metadata updated

When to use:

- false positive
- customer clarification accepted
- support case resolved

## Action Logging Rules

Every internal action should create:

- auth or admin event
- updated risk state where appropriate
- optional flag resolution
- optional admin note

No silent operator actions.

## Suggested Backend Contract

This doc does not require final implementation names, but the first internal layer will likely need:

- `GET /internal/accounts/:user_id`
- `GET /internal/accounts/:user_id/timeline`
- `POST /internal/accounts/:user_id/notes`
- `POST /internal/accounts/:user_id/restrict`
- `POST /internal/accounts/:user_id/suspend`
- `POST /internal/accounts/:user_id/reinstate`
- `POST /internal/accounts/:user_id/flags/:flag_id/resolve`
- `POST /internal/accounts/:user_id/flags/:flag_id/dismiss`

These should be internal/admin protected only.

## Data Inputs

The internal review surface should draw from:

- `users`
- `subscriptions`
- `telegram_links`
- `notification_preferences`
- `account_sessions`
- `auth_events`
- `account_risk_state`
- `account_risk_flags`
- `account_admin_notes`

## Customer-Facing Separation

This matters a lot.

Customer-facing account UX should still say things like:

- `Signed-in devices`
- `Current session`
- `Sign out other devices`

The internal review surface can use:

- `manual review`
- `restricted`
- `suspended`
- `risk flag`

Do not mix those two vocabularies.

## Permissions Model

The first version should assume only trusted internal operators can access this surface.

Later, role separation can be added:

- support reviewer
- billing reviewer
- risk reviewer
- admin

For now, the important point is:

- no public route
- no member-visible exposure
- clear audit trail for operator actions

## Recommended First Release Layout

### Top summary bar

- email
- subscription status
- account status
- risk level
- review status

### Left column

- account summary
- open flags

### Main column

- auth/session timeline

### Right column

- notes
- quick actions

On mobile/internal narrow view:

- summary
- flags
- timeline
- notes
- actions

## Decision Rules

The operator should default to:

1. read the summary
2. inspect open flags
3. confirm the timeline supports the concern
4. read existing notes
5. choose the lightest safe action

This reinforces:

- evidence first
- action second

## First Release Recommendations

### Build now

- internal account summary serializer
- open flag list
- auth/session timeline serializer
- admin notes read/write model
- restrict / suspend / reinstate action contract

### Defer slightly

- bulk review tools
- batch moderation queues
- advanced search across all accounts
- geolocation dashboards

## Operational Notes

This surface is where future account-risk work should converge.

That includes:

- suspicious activity review
- billing-owner context
- subscription disputes
- Telegram misuse patterns
- reinstatement history

If done well, it becomes:

- the internal source of truth for account trust decisions

## Immediate Recommendation

After this plan, the next implementation move should be:

**Task 85 — Internal Account Review Schema + Read APIs**

That should build:

- summary read model
- flags read model
- notes read/write model
- timeline serializer

Before adding live restrict/suspend/reinstate controls.
