# League Gap Remediation Plan

## Purpose

Define the next remediation layer after the league coverage audit.

This plan turns coverage findings into explicit action paths for weak leagues.

It should answer, for each weak or incomplete league:

- is the issue a true model blind spot?
- is it an `ALLMARKETS` / intake miss?
- is it a routed output bottleneck?
- is it an `OBSERVE` suppression problem?
- is it mainly an overlay/context gap?
- should the league target:
  - `full_coverage`
  - `partial_coverage`
  - `context_only`
  - or remain hidden for now

## Why This Exists

The coverage audit gave us the truth map.

We now know:

- which leagues are fully alive
- which leagues are only partially alive
- which leagues are context-supported but weak
- which leagues are genuine blind spots

The next step is not “fix everything.”

It is to define:

- which leagues are worth recovery
- what kind of recovery they need
- what success looks like

## Core Remediation Types

Each weak league should be assigned one dominant remediation type.

### 1. Model Expansion

Use when:

- the league lacks meaningful market outputs
- and the core problem is weak or missing model estate

Examples:

- no usable `FTR`
- no usable `BTTS`
- no usable `OU25`
- no stable FootyStats-era goal base

Typical fixes:

- onboard or retrain league-specific goal models
- restore missing market-family training estate
- widen allowed model families for the league

### 2. Routing Expansion

Use when:

- model output exists upstream
- but too little survives into routed `OBSERVE` or `DEPLOY`

Examples:

- league appears in upstream coverage
- but too many fixtures die before routed publication
- only one market family survives

Typical fixes:

- inspect pre-`ALLMARKETS` fixture loss
- inspect routed-market survival by family
- widen safe `OBSERVE` publication first
- reduce over-strict suppression where justified

### 3. Ingestion Repair

Use when:

- the league should be present
- but fixtures do not reach `ALLMARKETS`
- or bookmaker / source joins are failing

Examples:

- blind-spot league with expected bookie support
- fixtures present in source universe but absent from emitted market files

Typical fixes:

- repair bookmaker intake
- repair team/fixture joins
- repair source harmonisation
- restore pre-routing market rows

### 4. Overlay / Context Enhancement

Use when:

- routed model output is thin
- but contextual intelligence can still create product value

Examples:

- league has historical overlay support
- league has current odds/injuries/lineups
- but few or no routed signals survive

Typical fixes:

- strengthen `CONTEXT`
- improve non-routed notes
- support followed-team intelligence
- make non-pick monitoring genuinely useful

### 5. Deliberate Deprioritisation

Use when:

- a league is costly to recover
- low subscriber relevance
- low fixture importance
- weak data estate
- weak operator leverage

This is a valid decision.

Not every league needs immediate recovery.

## Two Main Tracks

## Track A — Blind-Spot Recovery

This track is for leagues that the audit classifies as real weak points or blind spots.

Current examples:

- `Germany Bundesliga 2`
- `Turkey Super Lig`

For these leagues, we must answer:

- do we have usable FootyStats-era goal models?
- do we have any stable `FTR` / `BTTS` / `OU25` market estate?
- do we have bookmaker coverage that should be feeding `ALLMARKETS`?
- do we have API-Football overlay support or not?
- can we at least raise safe `OBSERVE` or `CONTEXT`?
- if not, should the league stay hidden until repaired?

### Blind-Spot Decision Matrix

For each blind-spot league:

1. If model estate exists but routing is absent:
- classify as ingestion/routing repair

2. If no model estate but overlay support exists:
- classify as context-first recovery

3. If no model estate and no overlay support:
- classify as true blind spot
- likely keep hidden until a proper foundation exists

### Germany Bundesliga 2

Current read:

- no routed fixtures
- no routed `FTR`
- no routed `BTTS`
- no routed `OU25`
- no historical overlay support in current repo
- no current-window overlay support detected

Likely remediation type:

- `model expansion` or `ingestion repair`

Likely target state:

- first `context_only` if overlay support can be added cheaply
- otherwise remain hidden until at least `OBSERVE` or safe context becomes meaningful

### Turkey Super Lig

Current read:

- same broad pattern as Bundesliga 2
- no routed model output in active window
- no overlay support detected

Likely remediation type:

- `model expansion` or `ingestion repair`

Likely target state:

- probably remain hidden until there is a more credible recovery path

## Track B — Partial-Coverage Strengthening

This track is for leagues already alive in some form, but clearly incomplete.

Current examples:

- `Portugal Liga`
- `Belgium Pro`
- leagues with routed `OBSERVE` but limited market breadth
- leagues with only one strong market family

Questions:

- can we widen from `FTR` only into `BTTS` / `OU25`?
- can we improve routed fixture yield?
- can we reduce the non-routed tail?
- can we make `CONTEXT` richer where routing still fails?
- can followed users still receive useful intelligence even without deploys?

### Portugal Liga

Current read:

- routed fixtures exist
- routed `FTR` exists
- `BTTS` / `OU25` absent in active window
- historical overlay support exists
- current-window overlay support exists
- large non-routed tail remains

Likely remediation type:

- `routing expansion`
- plus `overlay/context enhancement`

Why Portugal is the best first target:

- not blind
- already produces some routed output
- already has context support
- already has live overlay support
- recovery should be easier than starting from zero

Target state:

- move from `partial_coverage` toward stronger multi-market routed presence

### Belgium Pro

Current read:

- healthy routing
- deploys exist
- but market-family breadth is narrow in this window
- overlay support is strong

Likely remediation type:

- `routing expansion`

Target state:

- widen market-family expression
- reduce dependence on one routed shape

### England Premier League / Italy Serie A / similar partials

These are not weak in the same way as blind spots.

They are partial because:

- routing exists
- some market families exist
- deploy count may be zero in-window

These should generally be treated as:

- healthy but selective
- not urgent recovery candidates unless subscriber demand says otherwise

## Ranking Framework

Do not remediate every weak league at once.

Rank leagues by:

### 1. Subscriber relevance

- how important is this league to user demand?
- do followed-team / followed-league features make this more valuable?

### 2. Fixture volume

- how many fixtures does the league contribute?
- how often will improvements surface to users?

### 3. Existing model estate

- do we already have usable FootyStats-era goal-market models?
- do we already have routed market rows in some windows?

### 4. Ease of recovery

- can we recover via routing?
- or does the league require full model onboarding?

### 5. Overlay support

- can the league at least become rich `CONTEXT` even before routed signals improve?

## Recommended Priority Order

### Priority 1 — Portugal Liga

Why:

- partial already
- routed `FTR` exists
- overlay support exists
- context lane is alive
- easiest visible product uplift

Recommended path:

- audit why `BTTS` / `OU25` are not surviving
- inspect pre-`ALLMARKETS` losses for non-routed fixtures
- strengthen follow-driven context output

### Priority 2 — Germany Bundesliga 2

Why:

- clear truth signal from audit
- explicit blind spot
- important to decide whether to recover or hide

Recommended path:

- audit whether any upstream model estate exists
- audit bookmaker intake
- decide whether first milestone is `context_only` or full hidden status

### Priority 3 — Turkey Super Lig

Why:

- same blind-spot pattern
- likely lower leverage than Portugal
- still needs explicit decision

Recommended path:

- same audit as Bundesliga 2
- likely deprioritise unless data estate is easier than expected

## Per-League Remediation Output Contract

Recommended future artifact:

- `league_remediation_plan.json`

Each weak league should carry:

```json
{
  "league": "Portugal Liga",
  "current_classification": "partial_coverage",
  "target_classification": "full_coverage",
  "primary_issue": "routing_expansion",
  "secondary_issue": "overlay_context_enhancement",
  "market_gaps": {
    "ftr": false,
    "btts": true,
    "ou25": true
  },
  "recommended_actions": [
    "Audit why BTTS rows do not survive routed publish",
    "Audit why OU25 rows do not survive routed publish",
    "Strengthen CONTEXT messaging for non-routed fixtures"
  ],
  "priority_rank": 1
}
```

## Important Constraint

Do not confuse:

- selective no-deploy weekends

with:

- broken leagues

Some partial leagues are healthy but simply conservative.

This remediation plan should focus on:

- true structural weakness
- avoidable routing loss
- meaningful model/coverage gaps

## Immediate Next Actions

After this plan, the next implementation layer should be:

1. build a machine-readable remediation artifact
2. produce a ranked weak-league table
3. start with:
   - `Portugal Liga`
   - `Germany Bundesliga 2`
   - `Turkey Super Lig`

## Success Condition

This task is successful when every weak league has:

- a current coverage diagnosis
- a target state
- a primary remediation type
- a ranked recovery priority
- and a clear decision whether to recover, context-support, or hide

That becomes the control panel for expanding the platform intelligently rather than randomly.
