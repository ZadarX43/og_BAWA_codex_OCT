# Platform State And Next Backend Work

## What Is Live Now

### Public website
- Home
- Predictions
- Results
- Premium
- Pricing
- Methodology
- Account

### Commercial and auth stack
- Stripe Checkout
- Stripe webhook entitlement persistence
- Cloudflare Worker premium gate
- Magic-link email auth
- Session-cookie premium unlock
- Premium page opens without manual token handling
- Public/premium board split is real

### Publishing and data bridge
- local model output -> website-safe JSON export
- public board publish works
- premium board publish works
- weekend board is live
- logo/badge bridge is live
- premium payload now preserves badge fields

### Protected premium delivery
- Worker verifies entitlement
- premium payload is edge-cached
- session auth is now the main path
- transitional token fallback still exists, but should eventually be retired

## What Is Not Built Yet

### User/account platform
- no real user database yet
- no subscription management UI
- no billing self-service UX beyond Stripe primitives
- no saved preferences/favourites
- no notification preferences
- no Telegram linking
- no full historical results archive per user-facing filter set

### Operational product layer
- no automated "finished games -> graded results board" pipeline yet
- no live ROI/win-rate archive updates after settlement
- no notifications system
- no fixture detail intelligence pages yet
- no search/filter-heavy results archive yet

## Do We Need A Database Now

Not a heavy one yet.

### Best shape

#### KV for:
- subscriber entitlement
- magic-link tokens
- lightweight session/state lookup
- small auth/rate-limit state

#### D1 for next-step product data:
- user identity record
- user preferences
- Telegram link mapping
- notification settings
- auth audit trail
- results archive / pick settlement metadata if richer filtering is needed

#### External DB:
- not needed yet unless the product outgrows Cloudflare-native simplicity

### Recommendation
- today: KV is enough for auth + entitlement
- next backend phase: add D1 for user/app data

## How To Think About Telegram

There are two different roles.

### 1. Telegram as comms channel
- send premium alerts
- send daily elite picks
- send acca posts
- send reminders / market alerts

### 2. Telegram as access-linked product surface
- user links Telegram account to Odds Genius account
- only verified paid users receive private premium bot/channel access
- possibly unlock a private channel/group automatically

### Best recommendation
- start with Telegram as a comms layer, not the core source of truth
- keep website auth + Stripe + Worker as the real authority
- later add Telegram linking as an optional connected channel

### Best future model
- website account is primary identity
- Stripe decides paid status
- Worker/KV/D1 decide entitlement
- Telegram is just an attached delivery surface

## How To Link Premium Account With Telegram Pro Access

Recommended flow:
1. user signs into Odds Genius on web
2. account page shows `Link Telegram`
3. user clicks deep link to your bot
4. bot gives one-time code
5. website confirms code
6. D1 stores Telegram mapping
7. Telegram channel/bot messages only go to entitled users

Do not make Telegram the billing or auth truth.
Use it as a connected surface.

## What Needs To Happen For Results After Matches Finish

This is the next important operational bridge.

### Target flow
1. weekend board is published
2. matches finish Monday/Tuesday
3. settled outcomes are ingested
4. picks are graded
5. wins/losses/voids are written to results JSON
6. results page updates:
   - settled count
   - hit rate
   - ROI
   - wins/losses
   - chart series
7. premium/public boards can move old settled rows into archive/results views

So yes: an automated grading + archive pipeline is needed.

## What That Pipeline Should Produce
- `weekly_results.json`
- results archive JSON/CSV
- per-pick settlement fields:
  - fixture
  - market
  - pick
  - odds
  - tier
  - result
  - won/lost/void
  - profit_units
  - settled_at
- rollups:
  - hit rate
  - ROI
  - streaks
  - by market
  - by tier
  - by league
  - by week/window

## What To Build Next On Backend
1. automated grading pipeline for finished picks
2. D1 user/app database
3. Telegram account-link model
4. notification settings + delivery framework
5. session-only auth cleanup after confidence grows

## Recommendation On Priorities

### Immediate backend priorities
1. automate `published picks -> settled results board`
2. add D1 for user/app state
3. keep Telegram as optional linked comms, not core auth

### Not yet priority
- big VPS architecture
- heavy monolith
- real-time model serving
- overbuilt user system

## Best Summary

What exists now is enough for a paid MVP.

What is needed next is not more pages. It is:
- results automation
- user/app state storage
- linked delivery channels like Telegram
- archive depth and transparency

The platform is now real enough that the next backend work should focus on operations, retention, and proof, not just unlocking access.

## Next Planned Docs
- Task 46 — Results Settlement & Archive Plan
- Task 47 — D1 User + Telegram Linking Schema
