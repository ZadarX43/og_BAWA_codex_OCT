# LIVE_FRONTEND_REFACTOR_SPEC

Updated: `2026-05-05`

## Purpose

This document converts the design merge plan into a practical implementation spec for the live Odds Genius frontend.

It is not a backend plan.
It is not a model plan.
It is a page-by-page frontend refactor brief for the current static site and Worker-aware premium flow.

## Product Truth To Preserve

Before changing visuals, keep these realities fixed:

- no model logic in the browser
- no export policy changes
- no Stripe/backend auth weakening
- premium remains locked by default
- `?demo=1` remains internal premium preview mode
- `?debug=1` remains the only place for public-hidden debug/token scaffolding
- frontend continues to consume approved exported JSON
- Worker remains the secure premium boundary

## What You Liked And What That Means

Based on the review, the strongest reusable ingredients are:

### Keep and amplify

- membership / pricing page structure
- performance proof graph treatment
- emerald / precision-green glow language
- compact institutional dashboard surfaces
- darker slate / charcoal foundation
- crisp technical labels and status chips

### Do not copy directly

- placeholder enterprise copy
- fake “financial intelligence” wording when it drifts too far from football
- fake customer/trust logos
- fake controls or fake data views not supported by the real product
- VR globe on the homepage

### Working rule

Use the concepts for:

- visual structure
- hierarchy
- motion tone
- component patterns

Do not use them as:

- final copy
- fake feature promises
- invented data schema

## Design Language To Apply

### Color system

Use the current live design system as the base, but push it toward the stronger concept language:

- background: deep charcoal / slate
- primary signal: glowing emerald green
- caution: muted yellow
- conflict: muted red
- cards: low-contrast tonal surfaces
- borders: thin, cool, technical lines

### Glow behavior

Use emerald glow selectively on:

- primary CTAs
- positive ROI and edge indicators
- active nav states
- selected chips and active tabs
- proof highlights

Do not use glow on:

- every card
- full page backgrounds
- paragraph text
- large decorative areas

### Typography

Preferred stack:

- primary UI/body/headings: `Inter`
- numeric/data/caps labels: `Space Grotesk`

### Tone

The site should feel like:

- a signal deployment terminal
- a proof-led subscription product

Not like:

- a sportsbook
- a trading simulator parody
- a gambling promo site

## File-By-File Refactor Scope

## 1. `frontend/assets/styles.css`

### Goal

This is the main visual system upgrade file.

### Add or refine

- stronger color tokens for emerald glow states
- clearer card surface tiers
- stronger tab and nav active states
- graph panel styling
- pricing comparison table styling
- state-chip styling:
  - deploy
  - caution
  - conflict
- denser table styles
- premium lock overlay styles
- dashboard shell utilities

### Specifically introduce

- a reusable proof-chart container style
- reusable metric-card style
- reusable technical label style
- reusable comparison-table style
- reusable “locked preview” blur/fade treatment

### Avoid

- adding too many one-off page-specific utility classes
- overcomplicating the responsive system

## 2. `frontend/assets/app.js`

### Goal

Keep logic stable, but improve structure and rendering output where needed.

### Allowed improvements

- cleaner markup generation for prediction cards
- clearer class naming in generated UI
- stronger public/premium/proof section rendering
- optional support for better results stats grouping if already derivable from existing JSON

### Do not change

- Worker API security behavior
- premium lock defaults
- export assumptions
- JSON schema expectations

### Highest-value rendering upgrades

- prediction cards should visually prioritize:
  1. pick
  2. value edge
  3. odds / confidence
- premium preview cards should feel closer to the premium dashboard concepts
- results rendering should support a more “proof dashboard” visual shell

## 3. `frontend/index.html`

### Goal

Turn the homepage into the strongest commercial entry point.

### Borrow from concept

Source:

- market intelligence landing page

### Structure target

1. sharp hero
2. proof strip
3. “why this wins”
4. decision-layer / moat block
5. sample board preview
6. premium CTA section

### Visual directives

- remove any leftover editorial softness
- use stronger grid structure
- use tighter vertical spacing
- use more dashboard-like proof blocks
- keep the glowing green CTA treatment

### Copy treatment

Keep current live positioning copy as truth.
Only use concept structure, not concept wording.

## 4. `frontend/predictions.html`

### Goal

Move the public board closer to a lite version of the premium dashboard.

### Borrow from concept

Source:

- premium dashboard

### Structure target

1. top summary strip
2. filter / board header area
3. denser prediction cards or table-card hybrids
4. clear market/tier chips
5. optional “why this deploys” note area

### Key visual priorities

- bigger pick text
- stronger green edge chip
- clearer spacing between:
  - pick
  - edge
  - odds
  - confidence

### Avoid

- overcomplicated side rails on mobile
- fake filter controls with no live logic behind them

## 5. `frontend/premium.html`

### Goal

Make the premium page feel like the premium dashboard is real and valuable even before unlock.

### Borrow from concept

Sources:

- premium dashboard
- dashboard with agent preview

### Structure target

1. locked hero
2. premium proof block
3. “what you unlock”
4. locked premium board preview
5. premium CTA band

### Visual treatment

- use premium dashboard color discipline
- use stronger state cards:
  - alignment
  - partial
  - conflict
- blur / lock deeper board layers
- use a more “system access” feel for the locked state

### Copy rule

No visible dev/test wording in the main premium copy.

## 6. `frontend/results.html`

### Goal

This should become the most convincing trust page on the site.

### Borrow from concept

Source:

- performance proof results

### Highest-priority carry-over

- the graph treatment

This is one of the strongest parts of the concept set.

### Structure target

1. proof hero
2. metric cards
3. graph / validation panel
4. settled history / recent proof list
5. market/tier breakdowns

### Graph rule

Even if the initial graph is simplified, it should visually feel:

- institutional
- clean
- low-noise
- performance-led

### Important note

If real current exports do not support a full historical curve, use a visually strong but honest reduced version.
Do not fake back-history.

## 7. `frontend/pricing.html`

### Goal

This page should borrow the strongest parts of the membership concept almost directly at the layout level.

### Borrow from concept

Source:

- membership plans

### Highest-priority carry-over

- two-card plan comparison
- highlighted founding-member plan
- lower comparison matrix
- high-trust technical framing

### Structure target

1. hero
2. Free vs Founding Member cards
3. protocol / feature comparison band
4. technical matrix
5. lower CTA section

### Copy rule

Keep current real product offer.
Do not inherit fake “API access” / “global 40+ markets” claims unless true and intentionally launched.

### Visual note

This is the page where the glowing green highlighted card can work hardest.

## 8. `frontend/methodology.html`

### Goal

Methodology should feel more like system documentation than marketing prose.

### Borrow from concepts

Indirectly from:

- premium dashboard
- proof dashboard

### Structure target

1. engine summary
2. export architecture explanation
3. decision engine explanation
4. value / structure / stability checks
5. final positioning line

### Visual treatment

- label-heavy
- compact
- technical blocks
- maybe small diagram cards or stacked “decision stages”

## 9. `frontend/account.html`

### Goal

Make account feel more product-grade and less like a visible scaffold.

### Structure target

1. account / membership summary
2. subscription status placeholder
3. manage subscription coming soon
4. hidden debug layer only when `?debug=1`

### Visual treatment

- more premium dashboard polish
- tighter panels
- visible member state card

## Component Refactor Priorities

## A. Metric cards

Need one shared style for:

- ROI
- hit rate
- signal count
- settled / pending
- protected floors

Desired traits:

- large number
- compact label
- subtle accent bar or glow
- stronger alignment to the proof dashboard concept

## B. Status chips

Create a shared visual grammar for:

- Alignment / Deploy
- Partial / Caution
- Conflict / Suppress
- Premium / Strong / Standard

These should become recurring product-language components.

## C. Graph panels

The results page graph look should become a reusable panel style.

Use for:

- walk-forward validation
- proof trend cards
- possible future results visualizations

## D. Comparison matrix

Pricing’s technical matrix is worth standardizing.

Could later support:

- Free vs Premium
- Premium vs Agent
- public vs protected access explanation

## E. Locked board preview

Premium needs a stronger locked-preview component:

- partial visibility
- blur or fade
- lock icon or restricted overlay
- CTA anchor nearby

## Mobile Refactor Rules

### Keep

- dark, premium feel
- strong CTAs
- visible proof

### Reduce

- extra vertical padding
- oversized hero dead space
- wide table assumptions

### Ensure

- pick and edge visible quickly
- proof cards stack cleanly
- pricing cards stay readable
- chart section remains coherent
- nav remains usable without feeling stuffed

## What Not To Build Yet

Do not pull these forward into the live core shell yet:

- VR / spatial homepage
- fake market heatmaps
- fake live enterprise status chrome
- Agent Access controls as if they are public features
- Slip Auditor as a top-nav core page

These are future modules, not current conversion spine.

## Suggested Delivery Order

### Phase 1

Design-system strengthening:

- `styles.css`
- shared card/chip/table/chart treatments

### Phase 2

Commercial pages:

- `index.html`
- `pricing.html`
- `premium.html`

### Phase 3

Trust and board pages:

- `results.html`
- `predictions.html`

### Phase 4

Support pages:

- `methodology.html`
- `account.html`

## Best Immediate Wins

If only a few changes are made first, prioritize:

1. pricing page layout merge from the membership concept
2. results page graph and proof-card styling
3. premium page locked dashboard styling
4. stronger emerald signal-glow system in shared CSS

## Final Guidance

The useful value in these designs is not the exact copy.
It is the product posture.

The best merge path is:

- keep the current truthful product messaging
- upgrade the visual shell to the stronger institutional dashboard language
- use the graph, pricing, and premium-board concepts as the highest-value imports

In short:

> Keep the real product.
> Borrow the better shell.
> Throw away the fake copy.
