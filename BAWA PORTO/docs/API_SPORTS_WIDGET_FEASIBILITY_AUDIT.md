# API-SPORTS Widget Feasibility Audit

## Purpose

Decide whether, where, and how API-SPORTS widgets should be used inside Odds Genius.

This audit exists to prevent two bad outcomes:

- rebuilding the core Odds Genius experience around third-party widgets
- ignoring useful widget utilities that could accelerate specific reference surfaces

The goal is:

- keep the main Odds Genius product custom
- borrow widget utility where it genuinely improves speed, coverage, or reference depth
- preserve the `clarity replacing chaos` product philosophy

## Current Product Position

Odds Genius is now:

- structurally strong
- philosophically clear
- commercially plausible

But it is not yet fully finished-feeling as a polished intelligence workspace.

The main product still needs:

- a fuller fixture-first visual pass on homepage and dashboard
- a visibly first-class `PASS / NO EDGE` state across the feed
- richer `CONTEXT / MONITOR` intelligence
- deeper results/proof automation
- more complete billing/self-service polish

That means widgets should be evaluated as:

- selective accelerators

not:

- the foundation of the main product experience

## What API-SPORTS Clearly Offers

As of late 2025 and early 2026, API-SPORTS publicly offers:

- Widgets v3
- a browser-based Widget Builder
- dynamic linking/targeting between widgets
- theme options and CSS customization
- multi-sport support
- a documented caching/proxy pattern to avoid exposing the API key directly

This makes widgets useful for:

- live-score/reference modules
- standings
- squads
- lineups
- injuries
- rapid UI prototyping

It does not make them automatically right for:

- the main Odds Genius dashboard
- the premium intelligence feed
- the emotional/UX core of the product

## Product Stance

Odds Genius should:

- copy Sofascore’s hierarchy
- borrow useful utility modules where needed
- keep the core experience custom, branded, and intelligence-led

The right framing is:

- Sofascore-like orientation
- Odds Genius interpretation

not:

- embedded third-party widgets as the main product

## Decision Summary

### Use widgets here

Safe and potentially high-value widget surfaces:

- standings blocks
- squad side-panels
- lineup side-panels
- injury/reference side-panels
- lightweight live-score reference modules
- internal/admin quick-reference tools
- temporary prototype surfaces used to test navigation or interaction ideas

These are good widget targets because they are:

- reference-heavy
- lower-risk from a brand perspective
- not the deepest part of the premium moat

### Do not use widgets here

Keep these custom:

- homepage core feed
- premium dashboard core experience
- personalised intelligence stream
- `DEPLOY / OBSERVE / MONITOR / PASS` feed logic
- decision-companion UX
- calm onboarding flow
- `My Feed / My Alerts`
- premium fixture interpretation layer
- results/proof storytelling surfaces

These should stay custom because they are where Odds Genius differentiates through:

- pacing
- interpretation
- restraint
- premium tone
- trust

### Prototype first here

The safest first prototype targets are:

1. standings block on fixture detail or league context surfaces
2. lineup / squad / injury side-panel on fixture pages
3. lightweight live-score/reference tab inside fixture detail

Recommended first prototype:

- standings on fixture detail or a side context rail

Why:

- useful immediately
- low brand risk
- easy to evaluate
- naturally secondary to the main intelligence layer

## UX Fit Assessment

### Strong fit

Widgets fit when the job is:

- quick orientation
- reference lookup
- secondary factual context
- support detail beside a custom intelligence surface

### Weak fit

Widgets fit poorly when the job is:

- interpretation
- premium positioning
- calm action pacing
- selective decision support
- emotionally disciplined UX

Odds Genius wins on:

- what it means
- why it matters
- when not to act

Widgets mainly help with:

- what happened
- what is scheduled
- who is available
- where the baseline stats sit

## Security And Key Handling

Using widgets with a visible client-side API key is not acceptable for the core product.

Minimum acceptable rule:

- no production widget should ship with an exposed unrestricted API key

### Safe delivery options

#### Option A — Cloudflare proxy/cache

Use the existing Cloudflare stack to:

- proxy widget/API requests
- inject the secret server-side
- cache by endpoint/path/query
- restrict origin behavior
- keep the key out of the client

Pros:

- aligned with the current stack
- fewer moving parts
- better operational consistency
- easier to govern from one platform

Cons:

- requires custom implementation and cache design

#### Option B — BunnyCDN pull-zone approach

Use BunnyCDN as documented by API-SPORTS to:

- add the auth header at the edge
- cache endpoint combinations
- hide the key from the widget code

Pros:

- explicitly documented by API-SPORTS
- fast to stand up
- useful if widget traffic becomes large and repetitive

Cons:

- adds another platform
- adds another operational/security surface
- weaker stack coherence than staying on Cloudflare first

### Recommendation

Start with:

- Cloudflare-first proxy/cache design

Only introduce BunnyCDN if:

- widget traffic becomes heavy enough to justify it
- media/image optimization needs a separate delivery layer
- Cloudflare implementation friction clearly outweighs the extra platform cost

## Operational Recommendation

### Phase 1

- do not rebuild the website around widgets
- keep homepage/dashboard/feed custom
- design one safe prototype surface

### Phase 2

- proxy/cache safely through Cloudflare
- test one widget utility module
- judge:
  - visual fit
  - brand fit
  - performance
  - cache efficiency
  - API cost behavior

### Phase 3

If the prototype works:

- expand only to adjacent utility/reference surfaces

If it does not:

- keep the interaction lessons
- discard the embed
- continue with custom UI

## Recommended Next Task

The clean implementation follow-on is:

### Task 89 — Widget Utility Prototype Plan

Define:

- one prototype surface
- the exact data path
- Cloudflare proxy/cache design
- success criteria
- fallback if the widget feels off-brand

Recommended prototype:

- fixture-detail standings or lineup/reference side-panel

## Sources

- [API-SPORTS widgets page](https://api-sports.io/widgets)
- [Introducing the API-SPORTS Widget Builder](https://www.api-football.com/news/post/introducing-the-api-sports-widget-builder)
- [How to create a sports website in just a few minutes using widgets](https://www.api-football.com/news/post/how-to-create-a-sports-website-in-just-a-few-minutes-using-widgets)
- [How To Optimize Widgets, Cache And Security Tutorial](https://www.api-football.com/news/post/how-to-optimize-widgets-cache-and-security-tutorial)
