# Widget Utility Prototype Plan

## Purpose

Define the first safe API-SPORTS widget prototype for Odds Genius.

This plan exists to answer:

- which single widget surface we should prototype first
- how it should be framed so it fits Odds Genius branding
- how the request/key path should work safely
- how we will judge whether it is good enough to keep

This is not a plan to widgetize the core product.

It is a plan to test one utility/reference module inside a custom Odds Genius shell.

## Core Decision

The first widget prototype should be:

- a fixture-detail standings block

not:

- the homepage feed
- the premium dashboard
- the main personalised intelligence stream

## Why This Prototype Wins

Fixture-detail standings are the safest first test because they are:

- clearly secondary
- reference-heavy
- useful immediately
- low-risk from a brand perspective
- easy to compare against a custom-built alternative later

They also fit the current product shape well:

- fixture pages already exist
- fixture pages already carry custom intelligence framing
- a standings block can sit beside or below the custom interpretation layer without weakening the Odds Genius tone

## Prototype Surface

### Exact surface

Place the prototype on:

- [frontend/fixture.html](/Users/hughwade/Documents/Code/OG_master/BAWA%20PORTO/frontend/fixture.html:1)

Recommended placement:

- a right-side or lower “League table” reference block inside fixture detail

Not above:

- action state
- why this reached deployment
- why we may be wrong
- decision companion

The widget must remain:

- support context

not:

- the hero of the page

## UI Framing Rule

The widget should not appear as a raw embed dropped into the page.

It should sit inside a branded Odds Genius wrapper:

- custom section heading
- custom explanatory microcopy
- custom spacing and card container
- calm background treatment
- clear distinction between:
  - `Interpretation`
  - `Reference`

Recommended label:

- `League table`

Recommended support copy:

- `Reference context for this fixture. Use this as orientation, not as the decision layer.`

## Overlay / Branding Strategy

Yes, Codex should treat the widget as something we can frame and soften with our own shell.

That means:

- do not rely on the raw widget appearance alone
- build a branded wrapper around it
- harmonize spacing, border, headings, and supporting copy
- treat the widget like a factual panel inside the Odds Genius desk

What we can realistically control well:

- surrounding layout
- section labels
- supporting interpretation copy
- container styles
- hide/show behavior
- placement in the fixture-detail hierarchy

What we should not depend on too heavily:

- deep internal DOM restyling that could break when API-SPORTS updates widget internals

So the design stance is:

- wrapper-first branding
- light widget theming
- no brittle over-customization

## Exact Data / Request Path

The prototype should not expose a direct client-side API key.

### Recommended path

1. fixture page determines the relevant competition / league context
2. widget requests go through a Cloudflare-controlled proxy/cache layer
3. proxy injects the upstream secret server-side
4. proxy returns cacheable widget/API responses
5. frontend loads the widget through the safe proxied path

### Recommendation

Use:

- Cloudflare-first proxy/cache

Do not start with:

- direct client-side API key exposure

Use BunnyCDN only if later needed for:

- much heavier widget traffic
- more aggressive endpoint-specific cache rules
- media/image optimization beyond what Cloudflare is already doing

## Proxy / Cache Shape

### Proposed path family

Recommended example:

- `/api/widgets/football/...`

This should:

- map only to the endpoints/widgets we allow
- inject the API key server-side
- reject unknown upstream paths
- apply cache by normalized query/path

### Initial cache posture

Suggested starting cache policy:

- standings: 300 to 900 seconds
- league metadata: hours
- media/logo references: longer CDN cache

This is intentionally calmer than:

- live-score widgets
- fixture event feeds

because standings do not need hyper-short refresh intervals for the first prototype

## Success Criteria

The prototype is successful if all are true:

1. it feels visually compatible with the fixture page
2. it does not interrupt or dilute the custom intelligence hierarchy
3. it improves orientation value for the user
4. it does not expose the API key
5. it performs acceptably through the proxy/cache layer
6. it does not materially complicate the product stack

### Product questions to answer

- Does this make the fixture page more useful without making it noisier?
- Does it feel like an Odds Genius desk module or a pasted third-party block?
- Does the branded wrapper soften the widget enough?
- Does the user still understand that Odds Genius owns the interpretation layer?

## Failure Conditions

The prototype should be rejected if:

- it feels off-brand
- it visually dominates the custom fixture page
- it introduces key/security discomfort
- it is fragile to style/DOM changes
- it encourages moving more core surfaces to widgets by convenience alone

## Fallback If It Feels Off-Brand

If the standings widget fails the brand test:

- keep the proxy/cache learnings
- keep the section placement learnings
- replace the widget with a custom standings/reference module later

That is still a successful prototype outcome because it clarifies:

- what should stay custom
- what reference density the page can carry comfortably

## Why Not Start With Lineups / Injuries First

Those are attractive, but slightly riskier as the first test because:

- lineup/injury modules can affect the emotional reading of the fixture more directly
- they sit closer to the custom intelligence layer
- they are more likely to invite over-dependence on third-party presentation

Standings are cleaner as the first test because they are:

- simpler
- calmer
- more obviously reference-only

## Recommended Implementation Order

### Phase 1 — Controlled prototype

- add one branded standings utility block to fixture detail
- wire it through Cloudflare-safe proxy/cache
- evaluate visual fit and product usefulness

### Phase 2 — Review

If standings work:

- consider a second prototype:
  - lineup side-panel
  - injury/context side-panel

If standings do not work:

- stop there
- keep main surfaces fully custom

## Next Build Move

The implementation follow-on after this plan should be:

### Task 90 — Fixture Detail Standings Prototype

That should build:

- the branded wrapper on fixture detail
- the Cloudflare-safe proxy/cache path
- the first live standings prototype
- a review checklist for fit, performance, and brand tone
