# Navigation And Page Architecture Spec

## Purpose

Define the stable navigation and page architecture for Odds Genius so the website feels like one football intelligence system rather than a set of separate surfaces.

This spec borrows:

- OneFootball's entity-page consistency
- Sofascore's fixture-first orientation
- Odds Genius' calm intelligence philosophy

It does not recommend:

- publisher-style clutter
- social/video sprawl
- sportsbook-style urgency
- generic content-feed architecture

## Core Product Rule

Odds Genius should feel like:

- a football intelligence workspace

not:

- a news publisher
- a sportsbook shell
- a prediction spreadsheet

The architecture must answer quickly:

- where am I
- what kind of page is this
- what matters first
- how deep can I go

## Navigation Model

Odds Genius should use a stable two-level navigation model:

1. global navigation
2. entity/page sub-navigation

This is the main structural lesson from the reference products.

## Global Navigation

Recommended global navigation:

- Matches
- Live
- Results
- Competitions
- Teams
- Dashboard
- Premium
- Account

### Global nav roles

`Matches`

- broad fixture-first browsing surface
- grouped by date, league, and live state
- non-personal default football desk

`Live`

- active live-state surface
- fewer fixtures
- higher urgency language
- scoreboard + live intelligence orientation

`Results`

- settled outcomes
- proof/archive/reporting
- public trust layer

`Competitions`

- competition landing pages
- league tables
- grouped fixtures
- competition-level intelligence context

`Teams`

- team landing pages
- team fixtures/results/squad/form/intelligence

`Dashboard`

- signed-in personalised desk
- follows
- alerts
- ranked intelligence feed
- account-shaped delivery behavior

`Premium`

- premium board and premium positioning
- gated value/intelligence surfaces

`Account`

- membership
- devices
- Telegram
- preferences
- onboarding

## Page Ownership

The product needs clearer page boundaries.

### Matches owns

- broad discovery
- grouped fixtures
- competition-first scanning
- live and upcoming orientation
- calm non-personal browsing

### Dashboard owns

- personal relevance
- saved follows
- alert posture
- why something interrupted
- why something stayed website-only
- selective intelligence delivery

### Fixture owns

- one fixture at a time
- decision framing
- reference layers
- lineups/table/stats/form context
- direct explanation of `DEPLOY / OBSERVE / MONITOR / PASS`

The rule is:

- `Matches` helps you browse
- `Dashboard` helps you prioritise
- `Fixture` helps you interpret

## Fixture Page Model

The fixture page should become the main deep-reading surface.

### Fixture header

The header should always show:

- home team
- away team
- kickoff / live / final state
- league
- `DEPLOY / OBSERVE / MONITOR / PASS`
- one-line intelligence summary

### Fixture primary tabs

Recommended fixture tabs:

- Overview
- Intelligence
- Lineups
- Table
- Stats
- Form
- Context

### Fixture tab roles

`Overview`

- immediate orientation
- fixture state
- market family
- quick published summary
- top-level metadata

`Intelligence`

- primary Odds Genius layer
- why deploy / why pass
- caution notes
- context tags
- market shape
- Telegram relevance
- decision-companion framing

`Lineups`

- confirmed lineups
- formations
- team-shape reference
- squad-side support context

`Table`

- league position
- points/games/form orientation
- competition context

`Stats`

- factual support layer
- score/odds/reference metrics
- match-state or market-state signals

`Form`

- recent match rhythm
- last-five style support
- nearby fixture context if fuller team form is not yet available

`Context`

- why this matches the user
- related fixtures
- broader support/risk environment

### Fixture page rule

`Intelligence` must be the main tab conceptually, even if `Overview` appears first.

The page is not a stats shell with an intelligence add-on.

It is:

- an intelligence page with reference tabs beside it

## Team Page Model

Recommended team page tabs:

- Overview
- Fixtures
- Results
- Squad
- Form
- Intelligence

### Team page rule

Skip these for now:

- News
- Transfers

unless they later become useful to prediction context.

The team page should stay product-led, not publisher-led.

## Competition Page Model

Recommended competition page sections/tabs:

- Overview
- Fixtures
- Table
- Form
- Intelligence

### Competition page role

Competition pages should help users:

- orient inside one league
- scan the current slate
- understand table position
- see current league context

They should not become giant content hubs.

## Card And Section Rhythm

Across Matches, Dashboard, Team, Competition, and Fixture pages:

- use consistent card rhythm
- use stable headers
- keep information blocks modular
- let the user descend gradually into depth

Borrow directly:

- competition-grouped fixture sections
- consistent entity page tabs
- modular stacked fixture sections
- card-based scan rhythm

Avoid directly:

- giant footers
- social clutter
- video-led filler
- off-mission editorial overload

## Immediate Product Implications

### Near-term

1. keep the main Odds Genius experience custom
2. make fixture pages use a stable tab model
3. make team/competition pages the next entity layer after fixture
4. preserve the distinction between `Matches` and `Dashboard`

### Later

1. implement the global nav fully
2. build team pages
3. build competition pages
4. tighten live-state surface around `Live`

## Decision Summary

Odds Genius should adopt:

- fixture-first browsing
- consistent entity-page tabs
- stable global nav
- stable page-level subnav
- intelligence-first fixture depth

Odds Genius should not adopt:

- publisher sprawl
- off-mission media modules
- dense social/footer clutter
- stats-first product framing

The correct product merge is:

- OneFootball entity structure
- Sofascore fixture orientation
- Odds Genius intelligence interpretation
