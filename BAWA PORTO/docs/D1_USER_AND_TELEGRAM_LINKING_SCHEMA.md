# D1 User And Telegram Linking Schema

## Purpose

Define the first D1-backed application schema for:
- website user identity
- Stripe-linked subscriber records
- Telegram account linking
- notification preferences
- lightweight account/product settings

This is not intended to replace:
- `SUBSCRIBER_STATE` KV for entitlement checks
- Worker session verification

It is intended to add product-state storage that KV is not ideal for long term.

## Core Principle

Stripe remains billing truth.
KV remains fast entitlement truth.
D1 becomes the user/application data layer.

Telegram should be attached to the web account, not become the billing or auth authority.

## What D1 Should Store

### User identity
- user profile row
- canonical email
- timestamps
- status flags

### Subscriber linkage
- Stripe customer id
- Stripe subscription id
- last known subscription status mirror

### Telegram linkage
- Telegram user id
- Telegram username
- Telegram chat id if needed
- verified link status
- linked_at / revoked_at

### Preferences
- favourite leagues
- favourite markets
- notification choices
- saved product view settings

### Audit trail
- magic-link event logs
- Telegram link events
- optional settings change logs

## What Should Stay In KV

Keep in KV:
- `SUBSCRIBER_STATE`
- magic-link one-time tokens
- short TTL rate-limit keys
- fast session-adjacent lightweight lookup material if needed

Why:
- fast edge reads
- simple entitlement checks
- no need to move latency-sensitive auth checks into SQL immediately

## Recommended Schema

## 1. `users`

Purpose:
- canonical web-account identity row

Suggested columns:
- `id` TEXT PRIMARY KEY
- `email` TEXT NOT NULL UNIQUE
- `email_normalized` TEXT NOT NULL UNIQUE
- `email_verified_at` TEXT
- `created_at` TEXT NOT NULL
- `updated_at` TEXT NOT NULL
- `account_status` TEXT NOT NULL

Suggested statuses:
- `active`
- `disabled`
- `pending`

## 2. `subscriptions`

Purpose:
- app-side view of Stripe-linked account membership

Suggested columns:
- `id` TEXT PRIMARY KEY
- `user_id` TEXT NOT NULL
- `stripe_customer_id` TEXT NOT NULL UNIQUE
- `stripe_subscription_id` TEXT NOT NULL UNIQUE
- `subscription_status` TEXT NOT NULL
- `price_id` TEXT
- `current_period_end` TEXT
- `created_at` TEXT NOT NULL
- `updated_at` TEXT NOT NULL

Foreign key:
- `user_id -> users.id`

Important:
- this mirrors Stripe/subscriber truth for product use
- it does not replace Stripe as billing authority

## 3. `telegram_links`

Purpose:
- map a website user to a Telegram identity

Suggested columns:
- `id` TEXT PRIMARY KEY
- `user_id` TEXT NOT NULL
- `telegram_user_id` TEXT NOT NULL
- `telegram_username` TEXT
- `telegram_chat_id` TEXT
- `link_status` TEXT NOT NULL
- `linked_at` TEXT
- `revoked_at` TEXT
- `created_at` TEXT NOT NULL
- `updated_at` TEXT NOT NULL

Suggested statuses:
- `pending`
- `linked`
- `revoked`

Constraint recommendation:
- unique on `telegram_user_id`
- one active link per `user_id`

## 4. `notification_preferences`

Purpose:
- store product delivery choices

Suggested columns:
- `id` TEXT PRIMARY KEY
- `user_id` TEXT NOT NULL UNIQUE
- `email_enabled` INTEGER NOT NULL
- `telegram_enabled` INTEGER NOT NULL
- `elite_alerts_enabled` INTEGER NOT NULL
- `acca_alerts_enabled` INTEGER NOT NULL
- `results_digest_enabled` INTEGER NOT NULL
- `favourite_markets_json` TEXT
- `favourite_leagues_json` TEXT
- `updated_at` TEXT NOT NULL

Foreign key:
- `user_id -> users.id`

## 5. `auth_events`

Purpose:
- lightweight audit trail

Suggested columns:
- `id` TEXT PRIMARY KEY
- `user_id` TEXT
- `email_normalized` TEXT
- `event_type` TEXT NOT NULL
- `ip_hint` TEXT
- `user_agent_hint` TEXT
- `created_at` TEXT NOT NULL
- `metadata_json` TEXT

Example event types:
- `magic_link_requested`
- `magic_link_verified`
- `logout`
- `telegram_link_started`
- `telegram_link_completed`

## Telegram Linking Flow

Recommended flow:
1. signed-in web user clicks `Link Telegram`
2. Worker generates one-time linking code
3. user opens Telegram bot
4. bot asks for or receives code
5. Worker verifies code
6. D1 writes `telegram_links`
7. user can now receive premium notifications/posts on Telegram

## How Telegram Should Be Used

### Phase 1
Telegram as a comms channel:
- alerts
- elite deployments
- acca posts
- daily summaries

### Phase 2
Telegram as a linked premium surface:
- verified paid users only
- private bot or channel access
- selective premium comms

## Relationship Between Tables

Recommended model:
- one `users` row
- one active `subscriptions` row per active membership
- zero or one active `telegram_links` row
- one `notification_preferences` row
- many `auth_events`

## Why D1 Is The Right Next Layer

Because these are relational product concerns:
- one user
- one subscription
- optional Telegram identity
- preferences
- audit history

KV is fine for edge auth and entitlement, but not ideal as the long-term source of truth for user/app state.

## What Not To Put In D1 Yet
- raw model outputs
- large prediction archives
- live inference state
- heavy analytics event streams

Keep D1 focused on account/product state first.

## Integration Boundaries

### Worker auth
- session verification can still stay KV + signed token based

### Stripe
- webhook continues writing KV entitlement
- later it can also upsert D1 subscription mirror rows

### Frontend account page
- reads session state from Worker
- later reads richer account/Telegram/preferences state from Worker-backed D1 routes

## Recommended Next Implementation Step

Build the initial D1 schema and migration for:
- `users`
- `subscriptions`
- `telegram_links`
- `notification_preferences`
- `auth_events`

Then wire only the smallest production-safe read/write paths:
1. user upsert on successful auth
2. subscription mirror upsert from webhook
3. Telegram link placeholder route contracts
