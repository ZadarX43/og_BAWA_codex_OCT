# WORKER_SUBSCRIPTION_STATE

Updated: `2026-05-04`

## Purpose

Define the first subscriber-state contract for the Odds Genius Worker.

This layer exists so Stripe webhook events can become authoritative membership state before premium delivery is unlocked.

## Current Scope

This scaffold covers:

- Stripe webhook signature verification
- event parsing for core subscription lifecycle events
- a simple persistence contract for subscriber state

It does not yet cover:

- frontend auth
- signed-in sessions
- premium entitlement checks on live user requests
- Customer Portal wiring

## Events In Scope

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`

## Record Contract

Subscriber state records should include:

- `customer_id`
- `subscription_id`
- `status`
- `price_id`
- `current_period_end`
- `updated_at`

Additional useful fields in this scaffold:

- `checkout_session_id`
- `email`
- `source_event_id`
- `source_event_type`
- `source_event_created`

## Store Shape

The scaffold uses a simple KV-style binding contract named:

- `SUBSCRIBER_STATE`

Expected capabilities:

- `get(key)`
- `put(key, value)`

## Key Strategy

The Worker writes two records for easier future lookup:

1. canonical subscription record
   - key: `subscription:<subscription_id>`
2. customer index record
   - key: `customer:<customer_id>`

The customer index record points to the latest known subscription state for that customer.

## Status Semantics

Examples:

- `checkout_completed`
- `trialing`
- `active`
- `past_due`
- `canceled`
- `unpaid`

Webhook subscription events should be treated as more authoritative than checkout completion alone.

## Required Binding And Secrets

- `SUBSCRIBER_STATE` binding
- `STRIPE_WEBHOOK_SECRET`

Related config still used elsewhere:

- `STRIPE_SECRET_KEY`
- `STRIPE_PRICE_ID`
- `SITE_URL`
- `PREMIUM_DATA_SOURCE`

## Security Reminder

Do not unlock premium predictions off checkout success alone.

Correct order:

1. Stripe Checkout session created
2. Stripe webhook verified
3. subscriber state persisted
4. later entitlement checks use stored state
5. only then should premium delivery be unlocked
