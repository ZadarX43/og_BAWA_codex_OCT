# DESIGN_MERGE_PLAN

Updated: `2026-05-05`

## Purpose

This document maps the external Odds Genius concept screens into a realistic implementation plan for the live site.

It is not a rebuild spec for the full product.
It is a merge plan:

- what visual language should carry over
- which concept should influence which live page
- what to keep out of the current conversion funnel
- what order to build things in later

## Source Concept Set Reviewed

### Design system source

- [/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/precision_intelligence/DESIGN.md](/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/precision_intelligence/DESIGN.md)

### Product brief source

- [/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_product_brief_summary.md](/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_product_brief_summary.md)

### Concept pages

- [/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_market_intelligence_landing_page/code.html](/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_market_intelligence_landing_page/code.html)
- [/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_premium_dashboard/code.html](/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_premium_dashboard/code.html)
- [/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_performance_proof_results/code.html](/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_performance_proof_results/code.html)
- [/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_membership_plans/code.html](/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_membership_plans/code.html)
- [/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_bet_slip_intelligence_audit/code.html](/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_bet_slip_intelligence_audit/code.html)
- [/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_dashboard_with_agent_preview/code.html](/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_dashboard_with_agent_preview/code.html)
- [/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_global_intelligence_command_vr/code.html](/Users/hughwade/Downloads/stitch_odds_genius_intelligence_platform/odds_genius_global_intelligence_command_vr/code.html)

## Core Design Direction To Adopt

The strongest common direction is:

- institutional
- dark-mode
- precision-led
- compact
- high-signal
- technical rather than “sportsbook flashy”

This means the live site should feel more like:

- a quant dashboard
- a market intelligence terminal
- a deployment monitor

And less like:

- a tipster brand
- a sports news site
- a gambling promo funnel

## Design System To Keep

From `DESIGN.md`, these should become the standing visual rules:

### Palette

- deep charcoal / slate base
- precision green / emerald for positive signal and primary actions
- yellow for caution / partial agreement
- red for conflict / suppression

### Type

- `Inter` for body and interface
- `Space Grotesk` for labels, data strings, stats, percentages, and system tags

### Surface model

- tonal layers
- low-contrast borders
- minimal shadow use
- glow only on active data or signal states

### Density

- keep layouts compact
- reduce decorative empty space
- prioritize visible proof, metrics, state, and routing cues

## Product Positioning The Design Must Support

The UI should reinforce this message:

> Odds Genius is a pricing inefficiency detection engine for football markets.

The design should always visually support:

- selectivity
- proof
- signal quality
- deployment discipline
- professional seriousness

## Concept Ranking

### Strongest concepts for the live core product

1. `odds_genius_market_intelligence_landing_page`
2. `odds_genius_premium_dashboard`
3. `odds_genius_performance_proof_results`
4. `odds_genius_membership_plans`

### Strong concepts for future modules

5. `odds_genius_bet_slip_intelligence_audit`
6. `odds_genius_dashboard_with_agent_preview`

### Concept to park for now

7. `odds_genius_global_intelligence_command_vr`

## What To Use From Each Concept

## 1. Homepage

### Primary source

- `odds_genius_market_intelligence_landing_page`

### What to borrow

- headline treatment
- proof strip directly below hero
- dark institutional hero background
- high-confidence CTA pairings
- “engineered / technical superiority” block pattern
- strong rectangular sections instead of airy editorial layouts

### What to adapt

- keep current live conversion copy, not the weaker placeholder text from the concept
- remove fake institutional client logos unless real
- keep homepage centered on product proof, not abstract enterprise language

### What to avoid

- overuse of generic “analyst/trust badge” motifs
- too much empty vertical hero space

### Desired homepage structure

1. tight hero
2. hard proof strip
3. “why this wins”
4. decision-layer / moat explainer
5. sample board preview
6. premium CTA band

## 2. Predictions Page

### Primary source

- `odds_genius_premium_dashboard`

### What to borrow

- table-plus-card hybrid layout
- stronger market chips
- clearer status/state column logic
- denser data hierarchy
- confidence presented as a right-edge bar or compact score

### What to adapt

- predictions page is public, so it should feel like the premium dashboard’s lighter sibling
- use alignment/partial/conflict language carefully if the live export supports it
- preserve current public JSON schema and frontend logic

### What to avoid

- fake rows
- invented markets or states not backed by current exports
- overcomplicated side rail if the public board needs speed first

### Desired predictions hierarchy

1. pick
2. edge
3. odds / model probability / implied probability
4. confidence tier
5. supporting reasons

## 3. Premium Page

### Primary sources

- `odds_genius_premium_dashboard`
- `odds_genius_dashboard_with_agent_preview`

### What to borrow

- the stronger premium board visual language
- institutional dashboard shell
- control-layer tease
- status color blocks for signal state explanation

### What to adapt

- current premium page remains locked by default
- use locked preview cards and blurred board blocks
- keep Worker-protected access framing
- make “Agent Access” a future tier, not a live distraction in the main premium funnel

### What to avoid

- exposing advanced controls before they exist
- letting future agent product confuse the core £20/month offer

### Desired premium flow

1. lock hero
2. “what premium unlocks”
3. premium proof block
4. locked board preview
5. CTA to pricing / checkout

## 4. Results / Proof Page

### Primary source

- `odds_genius_performance_proof_results`

### What to borrow

- big proof headline
- metric cards at top
- walk-forward chart section
- historical settled table
- subtle tier tabs / filters

### What to adapt

- current data may not support every visual from the concept yet
- use real available weekly proof first
- reserve multi-period charts for later if backend data is not in export form yet

### What to avoid

- fabricating back-history visuals
- implying fully interactive proof tooling if only a subset is live

### Desired results structure

1. proof headline
2. hit-rate / settled / pending cards
3. weekly proof dashboard
4. wins / misses / market breakdown
5. methodology note about independent results publication

## 5. Pricing Page

### Primary source

- `odds_genius_membership_plans`

### What to borrow

- two-column plan comparison
- strong highlighted founding card
- technical comparison matrix
- long-form feature explanation below the cards

### What to adapt

- keep current conversion copy and product truth
- `£20/month` or `£20/mo` remains the live founding offer
- avoid fake API / syndicate promises on the core public pricing page

### What to avoid

- overselling features not yet live
- adding enterprise tier clutter into the main public pricing funnel

### Desired pricing structure

1. hero
2. Free vs Founding Member cards
3. comparison matrix
4. proof / trust band
5. hard closing CTA

## 6. Methodology Page

### Primary source

- live copy plus the structural seriousness of `odds_genius_premium_dashboard`

### What to borrow

- compact label style
- structured information blocks
- explicit system-state terminology

### What to adapt

- methodology should explain the real architecture:
  - generated exports
  - no browser-side model logic
  - multi-model checks
  - structure / value / stability filters

### Desired methodology structure

1. what the frontend is
2. what the engine does
3. what deployment means
4. why some signals never publish
5. final market-wrong positioning line

## 7. Account Page

### Primary source

- current live page with a cleaner premium shell influence from the dashboard concepts

### What to borrow

- top-level tone from premium dashboard
- compact status blocks
- cleaner member-state UI

### What to avoid

- visible developer scaffolding for public users
- exposing test/auth internals unless `?debug=1`

## 8. Slip Auditor

### Primary source

- `odds_genius_bet_slip_intelligence_audit`

### Current recommendation

Do not merge into the current core nav yet.

Treat this as:

- a future module
- an upsell surface
- a separate product experiment

### Why

It is strong, but it introduces a second product story:

- current live story = pricing edge and premium board
- slip auditor story = post-analysis and swap logic

Both are valuable, but they should not compete on the homepage right now.

## 9. Agent Access

### Primary source

- `odds_genius_dashboard_with_agent_preview`

### Current recommendation

Use this as a design language reference only.

Good for future:

- Agent tier
- syndicate layer
- advanced threshold controls
- raw volatility / stability panels

Not for immediate public funnel:

- too advanced
- risks confusing the main paid offer

## 10. Spatial / VR

### Primary source

- `odds_genius_global_intelligence_command_vr`

### Current recommendation

Park this for now.

Do not use as the homepage direction.

### Why

- visually powerful
- too cinematic for the current conversion path
- eats space
- lowers clarity
- feels like a concept feature, not the commercial spine

### Future placement

Possible later page:

- `/vr`
- `/spatial`
- `/command`

Use only when:

- the core pricing/proof/product loop is already strong
- the globe can support real live data value

## Live Site Merge Strategy

The right merge strategy is not:

- recreate every concept screen exactly

The right merge strategy is:

- adopt one shared design system
- map each concept to the page where it is strongest
- preserve the current product truth and backend reality

## Recommended Page Mapping

### Live site page -> design source

- `index.html` -> market intelligence landing page
- `predictions.html` -> premium dashboard lite
- `premium.html` -> premium dashboard + locked control layer
- `results.html` -> performance proof results
- `pricing.html` -> membership plans
- `methodology.html` -> live copy + system dashboard structure
- future `slip-auditor.html` -> bet slip intelligence audit
- future `agent.html` -> dashboard with agent preview
- future `vr.html` -> global intelligence command VR

## Implementation Rules

When this gets built later:

- do not change model logic
- do not change Worker auth/security
- do not fake data fields not available in exports
- do not let the visuals outrun the real product
- do not add decorative complexity that hurts scan speed

## Build Priorities

### Priority 1

Apply the design system consistently:

- color variables
- typography
- border and card treatment
- button language
- signal-state colors

### Priority 2

Refit the core commercial pages:

- homepage
- predictions
- premium
- results
- pricing

### Priority 3

Refine member / methodology surfaces.

### Priority 4

Introduce future modules only after the core funnel converts.

## Recommended Next Build Passes

### Pass A

Design system unification:

- bring the live site closer to `DESIGN.md`

### Pass B

Homepage and premium/dashboard shell merge.

### Pass C

Results and pricing merge.

### Pass D

Future module exploration:

- slip auditor
- agent access
- spatial command

## Strategic Conclusion

The concept work is valuable because it confirms the product should feel like:

- institutional
- data-dense
- selective
- technical
- premium

The strongest commercial direction is not the VR concept.
It is the combination of:

- landing page proof
- premium dashboard state language
- results dashboard credibility
- pricing matrix clarity

If the live site follows this merge plan, Odds Genius will feel less like a polished betting site and more like what it actually wants to be:

> a deployment engine for football market mispricing.
