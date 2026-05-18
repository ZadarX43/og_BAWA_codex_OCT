# Account Flow UX Spec

## Purpose
This document turns the auth route contract into the actual customer-facing journey for Odds Genius.

It defines what the customer should experience across:
- checkout success
- account page
- verify-email flow
- premium unlock
- signed-in state
- logout
- subscription-management placeholder states

The objective is to remove the remaining developer/test feel and replace it with a clean premium-product experience.

## Core UX Principle
The customer should feel:
- I subscribed
- I verified my email
- I am recognized
- premium is unlocked

The customer should not feel:
- I need to manage a token
- I need to understand backend state
- I am interacting with scaffolding

## Product Truth To Preserve

- Stripe is the billing truth
- webhook-backed subscriber state is the entitlement truth
- premium board is a shared published artifact
- Worker is the gate between public and premium

The UX should make this feel simple even though the underlying system is structured.

## Entry Points

### 1. Checkout success return
Current return path:
- `/account.html?checkout=success`

### 2. Direct account visit
- `/account.html`

### 3. Premium page visit while not authenticated
- `/premium.html`

### 4. Magic-link verify redirect
- `/account.html?auth=success`
- or `/premium.html?auth=success`

## Journey Overview

## Stage A: Post-checkout handoff

### User state
- customer has completed Stripe Checkout
- webhook may already have written subscriber state
- user has not yet verified identity in the website session

### Account page goal
Turn “payment completed” into “email verification requested”.

### Recommended account message
Headline:
- `Your membership is almost ready.`

Body:
- `Verify your email to unlock premium board access on this device.`

Primary CTA:
- `Send sign-in link`

Secondary CTA:
- `Open pricing`

### Status indicators
Show:
- Checkout: complete
- Identity: not verified yet
- Premium access: pending email verification

Do not show:
- token language
- Worker/debug scaffolding

## Stage B: Magic-link request

### User action
User enters email and requests sign-in link.

### UX requirements
- email field
- clear button label
- non-scary confirmation state

### Request form copy
Field label:
- `Email`

Helper text:
- `Use the same email you used for checkout.`

Button:
- `Send sign-in link`

### Success state
Message:
- `If the address is eligible, a sign-in link has been sent.`

This avoids leaking subscriber existence.

### Error states
- invalid email
- rate limited
- temporary send failure

Recommended tone:
- concise
- calm
- not technical

## Stage C: Email verification

### User action
User clicks magic link from email.

### Backend effect
- verification token checked
- entitlement checked
- secure session cookie set
- user redirected back into product

### Preferred redirect destination
Default:
- `/account.html?auth=success`

Optional later optimization:
- redirect back to the originally intended page if premium was requested first

## Stage D: Verified account state

### User state
- session exists
- entitlement exists
- premium can be accessed

### Account page goal
Confirm success and show a clean premium-ready state.

### Recommended account hero
Headline:
- `Premium access unlocked.`

Body:
- `This device is verified for your membership. Open the premium board or manage your subscription when management tools go live.`

Primary CTA:
- `Open premium board`

Secondary CTA:
- `Go to results`

### Status indicators
Show:
- Membership: active
- Access: verified
- Device/session: signed in

Optional:
- masked email hint

Do not show:
- raw customer id
- subscription id
- token presence

## Stage E: Premium page unlock

### Not signed in / not verified
Premium page should show:
- locked board preview
- clear upgrade or sign-in path

If unsubscribed:
- primary CTA: `Unlock founding membership`

If subscribed but not verified:
- primary CTA: `Verify email to unlock`

### Signed in and entitled
Premium page should:
- load premium board automatically
- not ask for any token
- optionally confirm with a small line like:
  - `Verified premium access active`

### Premium page copy rule
The locked state should feel commercial.
The unlocked state should feel operational.

## Stage F: Signed-in session state

### Account page should show
- membership active
- premium access active
- email verified
- logout action
- subscription management coming soon

### Optional later additions
- last verified time
- linked email hint
- board freshness metadata

## Stage G: Logout

### User action
User clicks logout.

### UX behavior
- clear session cookie
- confirm signed out
- premium page returns to locked state

### Recommended post-logout message
- `You have been signed out from this device.`

### Account page after logout
Headline:
- `Verify your email to unlock premium access again.`

Do not show old “token missing” style messaging.

## Subscription Management Placeholder State

### Immediate requirement
We do not yet have full customer portal/account management UX.

### What account page should say
Use calm, product-grade copy:
- `Subscription management is coming soon.`
- `Your premium access remains controlled by your active membership state.`

### Optional CTA
- `Open premium board`

### Do not say
- placeholder
- developer/test
- scaffolding
- Worker-aware

That language should remain internal/debug only.

## Page-by-Page UX Spec

## Account page

### Anonymous / no session
Show:
- email verification form
- premium access explanation
- checkout-success acknowledgement if present

### Verified / entitled
Show:
- premium unlocked state
- open premium CTA
- logout CTA
- subscription management placeholder

### Debug mode only
If `?debug=1`, internal token/testing tools may still render temporarily.
They must remain hidden from standard user flow.

## Premium page

### Anonymous
- locked premium board preview
- CTA to pricing

### Paid but not verified
- locked premium board preview
- CTA to verify email

### Verified and entitled
- full premium board

## Pricing page

### Before checkout
- commercial pricing story
- founder CTA

### After checkout return
User should be nudged into verification, not re-sold immediately.

## Results page

No special auth behavior needed.
This remains a trust and proof surface.

## UX Copy Recommendations

## Checkout success
- `Your membership is almost ready.`
- `Verify your email to unlock premium board access on this device.`

## Verification request success
- `If the address is eligible, a sign-in link has been sent.`

## Verified success
- `Premium access unlocked.`
- `This device is verified for your membership.`

## Signed-out state
- `You have been signed out from this device.`

## Subscription placeholder
- `Subscription management is coming soon.`

## State Matrix

### 1. Not subscribed
- account: invite checkout
- premium: locked + upgrade CTA

### 2. Subscribed, not verified
- account: request magic link
- premium: locked + verify CTA

### 3. Verified and entitled
- account: unlocked + logout + management placeholder
- premium: unlocked board

### 4. Verified but entitlement inactive
- account: explain membership inactive
- premium: locked + renew/reactivate CTA

## Error-State UX

### Invalid link
- `This sign-in link is invalid or has expired. Request a new one.`

### Expired link
- `This sign-in link has expired. Request a new one.`

### Inactive subscription
- `Your membership is not currently active. Premium access is locked until your subscription is active again.`

### Temporary auth/system issue
- `We could not complete sign-in right now. Try again in a moment.`

Avoid technical messages in normal user states.

## UX Implementation Priorities

### Priority 1
- account page verification flow
- signed-in state
- logout state
- premium page session-aware lock/unlock behavior

### Priority 2
- redirect users to intended destination after verification
- cleaner success/failure banners

### Priority 3
- fuller subscription management UI when portal/account routes are ready

## Migration Guidance

### During transition
- keep debug token flow behind `?debug=1`
- standard flow becomes email verification

### After rollout
- remove public references to token storage
- keep internal debug capability only if still needed operationally

## Final Rule
The account flow should make Odds Genius feel like a premium intelligence platform:
- paid membership
- verified identity
- automatic unlock

Not:
- backend test harness
- token workflow
- developer-controlled access ritual
