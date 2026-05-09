# League Coverage Gap Audit

## Purpose

Define the source-of-truth audit for league coverage across the Odds Genius routing, intelligence, and overlay stack.

This audit exists to answer, for each league in an active publish window:

- do we have `FTR` model output?
- do we have `BTTS` model output?
- do we have `OU25` model output?
- do fixtures reach `ALLMARKETS`?
- do fixtures reach routed `OBSERVE`?
- do fixtures reach routed `DEPLOY`?
- do we have historical overlay support?
- do we have current-window odds?
- do we have current-window injuries?
- do we have current-window lineups?
- do we have current-window team stats?
- do we have current-window player stats?
- is the league:
  - `full_coverage`
  - `partial_coverage`
  - `context_only`
  - `blind_spot`

This should remove guesswork and make league weaknesses visible before they become frontend quality problems.

## Why This Audit Matters

The fixture-intelligence layer has exposed an important truth:

- some fixtures are fully routed into `DEPLOY` or `OBSERVE`
- some fixtures never reach routed output but still have usable context
- some leagues appear to have only partial market coverage
- some leagues may be true blind spots

Without a league audit, these states are easy to confuse.

This audit must distinguish:

1. no routed output for this fixture
2. no routed output for this league in this market
3. no usable context estate for this league
4. incomplete current-window overlay refresh
5. true model / ingestion blind spot

## Core Classification States

### `full_coverage`

Use when a league has all of the following in the active window:

- fixtures enter `ALLMARKETS`
- at least one of:
  - routed `DEPLOY`
  - routed `OBSERVE`
- meaningful market presence across the intended lane
- usable overlay support or current-window context sources

This does not require every fixture to become a deployable pick.
It means the league is operationally alive in the system.

### `partial_coverage`

Use when a league has meaningful routing or overlay support, but coverage is incomplete.

Examples:

- league has some `ALLMARKETS` presence but many fixtures fail pre-routing
- league has `FTR` rows but weak or missing `BTTS` / `OU25`
- league has historical overlays but thin current-window enrichments
- some fixtures become `OBSERVE`, others fall into `CONTEXT`

### `context_only`

Use when a league has no routed output for the relevant fixtures in the active window, but still has enough safe context to publish non-routed intelligence.

Examples:

- no live routed rows
- but historical overlay support exists
- or current-window odds / injuries exist
- or approved fixture metadata allows safe `CONTEXT` cards

### `blind_spot`

Use when a league is effectively uncovered in the active system.

Examples:

- no `ALLMARKETS` presence
- no routed `OBSERVE`
- no routed `DEPLOY`
- no historical overlay estate
- no meaningful current-window overlay sources
- only bare fixture identity can be produced

This state should be treated as a product and pipeline gap, not just a frontend issue.

## Audit Inputs

The audit should be built from existing artifacts, not manual judgment.

### Routed publish sources

- current-window `BOOKIE_*ALLMARKETS*.csv`
- routed tier files:
  - `__DEPLOY_TIER_ELITE__`
  - `__DEPLOY_TIER_STANDARD__`
  - `__DEPLOY_TIER_OBSERVE__`

### Upstream non-routed sources

- `PRE_ALLMARKETS_FIXTURE_LOSS_REPORT_<from>_to_<to>.csv`
- `PRE_ALLMARKETS_FIXTURE_LOSS_DETAILS_<from>_to_<to>.csv`

### Intelligence artifacts

- `frontend/public/data/covered_fixture_universe.json`
- `frontend/public/data/fixture_intelligence_public.json`
- `reports/latest/COVERED_FIXTURE_UNIVERSE_REPORT.md`
- `reports/latest/FIXTURE_INTELLIGENCE_REPORT.md`

### Historical overlay estate

Use whichever approved overlay families exist in the repo for the league:

- team rolling / form profiles
- injury overlays
- lineup-shape overlays
- goal-environment overlays
- referee / matchup overlays where available

### Current-window overlay refresh outputs

- `reports/latest/api_current_context_overlay_window/CURRENT_CONTEXT_OVERLAY_SUMMARY.json`
- normalized current-window odds
- normalized current-window injuries
- normalized current-window lineups
- normalized current-window team stats
- normalized current-window player stats
- normalized current-window events

## Per-League Audit Contract

The audit should produce one record per league per active window.

Recommended artifact:

- `frontend/public/data/league_coverage_audit.json`

Recommended record shape:

```json
{
  "league": "Portugal Liga",
  "window_from": "2026-05-09",
  "window_to": "2026-05-11",
  "fixture_counts": {
    "covered_total": 9,
    "routed_total": 2,
    "deploy_total": 0,
    "observe_total": 2,
    "context_total": 7,
    "monitor_total": 0,
    "hidden_total": 0
  },
  "market_output": {
    "ftr_present": true,
    "btts_present": false,
    "ou25_present": false
  },
  "routing_presence": {
    "allmarkets_present": true,
    "observe_present": true,
    "deploy_present": false,
    "loss_report_presence": true
  },
  "overlay_support": {
    "historical_overlay": true,
    "current_odds": true,
    "current_injuries": false,
    "current_lineups": false,
    "current_team_stats": false,
    "current_player_stats": false
  },
  "classification": "partial_coverage",
  "notes": [
    "League has covered fixtures in the active window.",
    "No deployable picks routed in this window.",
    "Historical overlay support exists, so non-routed fixtures can still publish CONTEXT."
  ]
}
```

## Required Metrics

For each league, compute:

### Fixture counts

- total covered fixtures
- routed fixtures
- non-routed fixtures
- published `DEPLOY`
- published `OBSERVE`
- published `CONTEXT`
- published `MONITOR`
- unpublished / hidden if applicable

### Market presence

For routed rows only:

- `FTR` present yes/no
- `BTTS` present yes/no
- `OU25` present yes/no

Optional deeper counts:

- routed `FTR` fixture count
- routed `BTTS` fixture count
- routed `OU25` fixture count

### Routing presence

- did fixtures enter `ALLMARKETS`
- did fixtures reach routed `OBSERVE`
- did fixtures reach routed `DEPLOY`
- did the loss report capture dropped fixtures

### Overlay support

- historical overlay estate present
- current odds present
- current injuries present
- current lineups present
- current team stats present
- current player stats present
- current events present

## Classification Logic

Recommended first-pass classification order:

### `blind_spot`

Assign if all are true:

- `allmarkets_present = false`
- `observe_present = false`
- `deploy_present = false`
- `historical_overlay = false`
- current odds/injuries/lineups/team/player support is effectively absent

### `context_only`

Assign if:

- `deploy_total = 0`
- `observe_total = 0`
- `context_total > 0`
- and some overlay or fixture-intelligence support exists

### `partial_coverage`

Assign if:

- some routed output exists, but not enough to treat the league as healthy
- or only one market family is consistently present
- or routed coverage exists but a large non-routed tail remains

### `full_coverage`

Assign if:

- the league has stable routed presence in the active window
- multiple intended market families are alive
- non-routed tail is limited or context lane is merely supplemental

## Important Distinctions

This audit must not collapse these situations together:

### Case A: fixture-level miss

A single fixture may fail routing in a league that is otherwise healthy.

That should not downgrade the whole league to blind spot.

### Case B: market-family weakness

A league may have:

- routed `FTR`
- but missing `BTTS`
- and missing `OU25`

That is partial coverage, not full coverage.

### Case C: context-supported but unrouted

A league may have no routed rows in the active window, but still have:

- historical overlays
- current odds
- current injuries

That is `context_only`, not blind spot.

### Case D: true model / ingestion blind spot

A league with no routed rows and no useful context support should be marked clearly as a blind spot.

## Initial Read On Current Findings

Based on the current window explored during Task 60 and Task 61:

### Germany Bundesliga 2

Likely state:

- no routed rows for sampled fixtures
- no strong historical overlay estate in current repo
- current context cards mostly coverage-gap language

Working classification:

- likely `blind_spot` or lower-end `context_only`

### Portugal Liga

Likely state:

- no routed rows for sampled fixtures in this window
- but historical overlay support exists for at least some fixtures
- some `CONTEXT` cards carry real historical notes

Working classification:

- likely `context_only` or `partial_coverage`

This is exactly why the audit is needed: the system should stop relying on informal interpretation of card output.

## Output Use Cases

This audit should drive:

- internal league coverage decisions
- model expansion priorities
- frontend trust messaging
- alert routing confidence
- follow-system reliability
- future app readiness

It should also support:

- “do not expose this league publicly yet”
- “context only”
- “premium signal-capable”

## Recommended Follow-On Implementation

After this audit spec, the next implementation should be:

1. build `league_coverage_audit.json`
2. generate `LEAGUE_COVERAGE_AUDIT_REPORT.md`
3. surface league coverage class in internal reports
4. optionally surface a hidden internal-only dashboard

## Success Condition

This task is successful when, for any league in a live window, we can answer with evidence:

- what routed market output exists
- what did not route
- what context still exists
- whether the league is healthy, partial, context-only, or blind

That becomes the operating map for the next model and intelligence expansion phase.
