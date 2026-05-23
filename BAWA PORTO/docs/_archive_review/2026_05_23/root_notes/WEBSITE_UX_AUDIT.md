# WEBSITE_UX_AUDIT

Updated: `2026-05-04`

## Scope

Audit target:

- Live site: `https://og-bawa-codex-oct.pages.dev`
- Pages reviewed:
  - `/`
  - `/predictions`
  - `/premium`
  - `/results`
  - `/pricing`

Reference design language reviewed from provided Odds Genius screenshots:

- dark navy/charcoal backdrop
- teal-to-mint glow CTAs
- premium sports-tech feel
- compact, high-signal layout
- product surfaces like Bet Checker / Acca Builder / Co-Pilot

Important product note from founder:

- the 3D globe / VR-style fixture explorer should not be the homepage hero right now
- it is space-heavy and should be parked to a future dedicated `VR` page or research/demo surface

## Executive Read

The live scaffold is operational and trustworthy from a systems perspective, but it does not yet feel like the premium Odds Genius product shown in the screenshots.

The current live site is:

- clearer than the older concept build
- structurally safer
- easier to ship

But it is also:

- too soft and editorial in tone
- too beige/light compared with the existing Odds Genius brand language
- too generic in premium conversion flow
- not forceful enough above the fold about proof, edge, and paid value

The biggest UX gap is not functionality. It is product posture.

Right now the site reads like:

- "clean prototype for a data product"

It needs to read like:

- "serious, high-signal football intelligence platform with a paid edge"

## 1. First Impression

Current impression:

- calm
- readable
- competent
- safe

But not:

- sharp
- premium
- sports-native
- urgent

The screenshots suggest a much stronger brand language:

- dark, focused, almost terminal-meets-trading-desk
- glowing CTA energy
- denser information surfaces
- a more expensive-looking product

The live site currently feels more like a polished report than a sports intelligence product.

## 2. Homepage Clarity

What works:

- headline is understandable
- navigation is simple
- board counts and source window add some legitimacy

What is weak:

- the hero does not immediately show what a bettor gets today
- there is not enough "why this matters now" above the fold
- the methodology tone arrives too early
- there is not enough direct proof or sample outcome in the hero area

The homepage should answer, in the first screen:

1. What is Odds Genius?
2. Why trust it?
3. What do I get free?
4. What do I get if I pay?
5. What should I click next?

Current hero gets `1` partially and `5` partially, but under-delivers on `2`, `3`, and `4`.

## 3. Trust / Proof Above The Fold

This is the biggest homepage weakness.

The site says:

- model-backed
- walk-forward tested

But it does not visually prove it fast enough.

What is missing above the fold:

- recent win-rate summary
- settled picks count
- current premium/public board counts in a tighter proof strip
- "last updated" confidence signal
- one or two sample picks with real numbers

Recommendation:

- add a compact proof ribbon directly under the hero
- include:
  - latest publish time
  - current public picks
  - current premium picks
  - settled hit rate
  - audit-safe "last graded window"

## 4. Predictions Board Readability

Current strengths:

- data is present
- card and table formats exist
- fields are understandable

Current weaknesses:

- cards are a little too airy for betting use
- hierarchy between market, pick, tier, edge, and odds is not strong enough
- public board does not yet feel fast to scan
- the current styling is elegant but not high-performance

The screenshots suggest a more productized information density:

- tighter modules
- stronger chip system
- obvious action hierarchy
- visually grouped signal components

Recommendation:

- make the predictions page feel more like a board and less like a magazine layout
- prioritize:
  - fixture
  - market
  - pick
  - tier
  - edge
  - odds
  - short reason

Immediate UI upgrades:

- stronger market chips
- stronger ELITE / STANDARD color treatment
- reduce text blocks
- introduce clearer scan lanes
- add sticky filter/sort bar later

## 5. Premium Lock Conversion

Current strengths:

- premium is clearly locked by default
- Worker/token logic is honest
- demo mode is separated

Current weaknesses:

- locked state is informative but not persuasive enough
- conversion copy is still product-scaffold language
- too much emphasis remains on implementation state
- not enough emphasis on subscriber value

The premium page should sell:

- full board access
- ELITE picks
- richer explanations
- correct-score shortlist
- slip-role hints
- evidence-first edge

Right now it explains the mechanics more than the outcome.

Recommendation:

- replace part of the lock page copy with a sharper paid proposition
- show 1 locked premium board preview block
- show 3 concrete reasons to upgrade
- show one strong CTA
- move most developer/test language deeper into account or internal mode

## 6. Pricing Clarity

Current strengths:

- `£20/month` is simple
- free vs founding member split is understandable

Current weaknesses:

- pricing page still reads like a staging page
- the founding offer is not framed with enough confidence
- there is not enough "who this is for"
- not enough "what exactly unlocks"

Recommendation:

- make pricing page more decisive
- add:
  - "For serious weekend bettors"
  - "Full deployable board"
  - "ELITE picks and richer edge context"
  - "Cancel anytime"
  - "Early founding member pricing"

The page needs to reduce hesitation and feel like a real offer, not a placeholder.

## 7. Results / Proof Credibility

Current strengths:

- results page exists
- structured weekly proof is present
- by-market and by-tier summaries are useful

Current weaknesses:

- the visual treatment is too similar to every other page
- proof should feel more scoreboard-like
- settled vs pending needs stronger emphasis
- the page should feel like evidence, not just data output

Recommendation:

- lead with a proof dashboard block
- visually separate:
  - settled
  - pending
  - hit rate
  - featured wins
  - misses
- use a more performance-led layout

This page is one of the most commercially important pages because it converts skepticism into trust.

## 8. Mobile Layout

The current CSS does include mobile breakpoints, which is good.

What still likely needs improvement:

- top nav becomes crowded quickly
- large hero copy may dominate too much vertical space
- tables will still feel heavy on smaller phones
- premium and pricing surfaces need tighter vertical pacing

Likely mobile issues:

- too much top padding before payoff
- insufficient sticky CTA behavior
- scan density drops too much

Recommendation:

- compress hero on mobile
- move proof strip higher
- reduce card padding slightly
- consider a segmented mobile nav or drawer later
- convert some table-heavy sections to stacked cards on narrow widths

## 9. Typography / Spacing

This is one of the clearest mismatches with the screenshots.

Current live site:

- serif-led
- editorial
- warm
- spacious

Screenshot reference:

- sans-serif
- tighter
- darker
- more technical
- more premium sports dashboard

The current font and spacing system make the product feel more like:

- newsletter intelligence

Instead of:

- live football edge platform

Recommendation:

- move away from the Georgia / newspaper feel
- adopt a sharper sans pairing
- tighten spacing rhythm
- increase contrast on headings and chips
- use the teal glow treatment from the current Odds Genius design as the signature accent

## 10. Brand Feel Vs Premium AI Sports Product

This is the biggest design conclusion.

The live site currently does not inherit enough of the brand energy already present in the screenshots.

What the screenshots do well:

- premium dark mode
- futuristic but still useful
- teal gradient CTA system
- strong dashboard surfaces
- good visual fit for AI + betting + intelligence

What should be carried forward:

- dark navy / charcoal base
- teal-to-mint accent gradient
- slightly luminous CTA/buttons
- compact surface cards
- more "signal terminal" energy

What should not be carried forward to the homepage right now:

- the VR globe as primary hero

Recommendation:

- keep the color system and premium feel
- drop the globe from the main homepage experience
- park it to:
  - `/vr`
  - or a future "fixture explorer" product page

## Immediate Design Fixes

- shift the whole live site from warm light editorial to dark premium sports-tech
- replace serif typography with sharper sans typography
- add a homepage proof strip directly under hero
- tighten predictions board spacing and hierarchy
- reduce explanatory filler on premium page
- strengthen pricing page offer framing
- make results page feel like a proof dashboard
- keep the globe out of the core homepage funnel

## Immediate Copy Fixes

- reduce scaffold/dev wording on customer-facing pages
- replace "prepared" or "placeholder" language with product language where safe
- sharpen homepage subhead to emphasize:
  - model-backed edge
  - live board
  - proof
  - paid unlock
- premium page should sell benefits before mechanics
- pricing page should speak to bettor outcome, not just access plumbing

## Layout Changes

- homepage:
  - hero left: value proposition + CTA
  - hero right: live proof / sample board snapshot
  - proof strip below
- predictions:
  - denser board module
  - stronger filters/chips
- premium:
  - single high-converting locked hero
  - benefit stack
  - one primary upgrade CTA
- results:
  - scoreboard top block
  - featured proof section
- pricing:
  - one strong paid card, one simpler free card

## Component Upgrades

- top navigation should feel more branded and product-grade
- buttons should use the teal gradient/glow language
- chips need more visual meaning by tier and market
- proof stats should have a stronger dashboard style
- locked premium cards should look aspirational, not generic
- card surfaces should feel tighter and more expensive

## Mobile Fixes

- compress hero copy and button stack
- shorten nav treatment
- convert wider comparison/table surfaces into stacked cards
- reduce padding and dead space
- keep CTA visible earlier
- ensure proof metrics appear above heavy explanatory sections

## Conversion Improvements

- homepage must prove value faster
- premium page must sell outcome more aggressively
- pricing page must feel like a real offer, not a future one
- results page should be used as a conversion tool
- add stronger internal link flow:
  - homepage -> predictions
  - homepage -> results
  - predictions -> premium
  - premium -> pricing
  - pricing -> checkout

## Recommended Next Build Order

### Task 29 — Premium Design Upgrade Pass

Priority order:

1. Brand/design system refit to dark premium Odds Genius style
2. Homepage hero + proof strip rebuild
3. Predictions board layout upgrade
4. Premium lock conversion redesign
5. Results page proof dashboard treatment
6. Pricing page conversion upgrade
7. Mobile tightening pass

## Bottom Line

The live site is operationally strong but visually underpowered for the brand you already have.

The right move is not to make it prettier in a generic way.
The right move is to make it feel unmistakably like Odds Genius:

- darker
- sharper
- more premium
- more proof-led
- more conversion-focused

And the VR globe should be removed from the main homepage funnel for now and parked to a future dedicated page.
