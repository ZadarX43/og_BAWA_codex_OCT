# Covered Fixture Universe Builder Spec

## Purpose

Define the implementation-ready builder spec for the covered fixture universe.

This is the first concrete backend step after the covered fixture `CONTEXT` / `MONITOR` plan.

Its job is to answer one question cleanly and deterministically:

### Which upcoming fixtures belong to the Odds Genius covered intelligence universe for the active publish window?

This builder does not:
- generate picks
- generate deploy tiers
- generate observe translations
- publish frontend JSON directly

It produces the canonical non-routed fixture intake set that later layers can enrich, classify, and publish.

## Why This Builder Exists

The routed exporter only sees fixtures that already entered:
- `ELITE`
- `STANDARD`
- `OBSERVE`

That is not enough for the broader intelligence system.

We need a separate builder that:
- identifies all upcoming fixtures in covered leagues
- normalizes fixture identity
- joins safe availability markers
- flags routed vs non-routed fixtures
- gives downstream exporters a complete universe to work from

Without this builder:
- `CONTEXT` and `MONITOR` have no stable intake
- followed teams can disappear when no routed row exists
- the platform remains dependent on routed output visibility

## Output Target

Recommended first builder artifact:

- `frontend/public/data/covered_fixture_universe.json`

Recommended operator/debug artifact:

- `reports/latest/COVERED_FIXTURE_UNIVERSE_REPORT.md`

Optional intermediate operator CSV:

- `reports/latest/covered_fixture_universe.csv`

The builder should be deterministic and rerunnable.

## Core Responsibilities

The covered fixture universe builder must:

1. identify all upcoming fixtures in the active publish window
2. restrict to leagues currently inside the Odds Genius covered set
3. normalize fixture identity to the same canonical join style used by the website and routed exporters
4. attach logo / badge bridge fields
5. attach routed-presence markers
6. attach source availability markers for:
   - FootyStats-era goal / shape base
   - API-Football fixtures / odds / injuries / lineups / stats
7. assign an initial universe status:
   - `routed`
   - `non_routed`
   - `unsupported`
   - `hidden`

This builder should stop before:
- final `CONTEXT` / `MONITOR` classification
- final reasoning generation
- Telegram routing

## Canonical Inputs

## 1. Fixture identity base

Primary:
- `data_sources/api_football/normalized/fixtures_master*.csv`

Role:
- competition
- fixture id
- kickoff time
- home / away teams
- date-window candidate universe

This is the preferred primary source for upcoming fixture completeness.

## 2. Routed presence references

Primary:
- routed tier files:
  - `__DEPLOY_TIER_ELITE__`
  - `__DEPLOY_TIER_STANDARD__`
  - `__DEPLOY_TIER_OBSERVE__`
- existing:
  - `frontend/public/data/fixture_intelligence_public.json`

Role:
- detect whether a fixture is already handled by the routed exporter
- prevent duplicate or conflicting primary ownership

## 3. Stable goal / shape base availability

Primary candidates:
- current all-markets files
- approved support / candidate files
- FootyStats-era stable shape families
- `DEPLOY_CANDIDATES_RAW.csv`
- `DEPLOY_CANDIDATES_AFTER_GATES.csv`

Role:
- mark whether a fixture has usable model-shape or market-shape coverage
- this should be an availability marker at builder stage, not a publish decision yet

## 4. API-Football overlay availability

Primary:
- `odds_prematch_long*.csv`
- `injuries*.csv`
- `lineups*.csv`
- `match_team_stats*.csv`
- later:
  - `match_player_stats*.csv`
  - `match_events*.csv`

Role:
- mark what enrichment families are available for each fixture

## Covered League Set

The builder needs a canonical covered-league registry.

Near-term acceptable sources:
- a manually maintained allowlist derived from current production leagues
- the same operational league set used for the live publish window

Longer-term preferred source:
- explicit config or manifest such as:
  - `configs/covered_leagues.json`
  - or a validated competition manifest

The builder should not infer league coverage from random file presence alone.

## Publish Window Rules

The builder must support the same active window concept used by the live publish path.

### Required window inputs

- `date_from`
- `date_to`

These should usually be derived from:
- the selected current deploy/all-markets run
- or explicit operator args

### Recommended CLI behavior

Support:
- `--src <deploy/allmarkets file>` to inherit window from the current run
- optional explicit:
  - `--date-from`
  - `--date-to`

### Important rule

The builder should never quietly scan the whole historical fixture base.

It must always operate inside a declared current publish window.

## Canonical Join Keys

The builder should preserve / generate:

- `fixture_id` where available
- `fixture_key` as canonical website join key
- `league`
- `kickoff_time`
- `home_team`
- `away_team`

It should reuse the same normalization logic already established by:
- `publish_predictions.py`
- `publish_fixture_intelligence.py`
- the badge/logo join layer

If identity cannot be normalized cleanly:
- mark as `hidden`
- report it

## Universe Record Shape

Recommended per-fixture builder record:

```json
{
  "fixture_id": "12345",
  "fixture_key": "2026_05_09_Arsenal_Chelsea",
  "kickoff_time": "2026-05-09T15:00:00Z",
  "league": "England Premier League",
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "league_logo_url": "https://...",
  "league_flag_url": "https://...",
  "home_team_logo_url": "https://...",
  "away_team_logo_url": "https://...",
  "logo_join_status": "FULL_MATCH",
  "coverage_status": "covered",
  "routing_status": "non_routed",
  "source_availability": {
    "fixtures_master": true,
    "routed_deploy": false,
    "routed_observe": false,
    "goal_shape_base": true,
    "prematch_odds": true,
    "injuries": true,
    "lineups": false,
    "team_stats": true,
    "player_stats": false,
    "match_events": false
  },
  "follow_candidates": {
    "team_follow_candidate": true,
    "fixture_follow_candidate": true,
    "league_follow_candidate": true
  },
  "updated_at": "2026-05-09T12:00:00Z"
}
```

## Top-Level Output Shape

Recommended:

```json
{
  "generated_at": "2026-05-09T12:00:00Z",
  "source_run_id": "2026-05-09",
  "source_window": {
    "date_from": "2026-05-09",
    "date_to": "2026-05-11"
  },
  "coverage_summary": {
    "total_fixtures": 0,
    "routed_count": 0,
    "non_routed_count": 0,
    "hidden_count": 0,
    "covered_leagues_count": 0
  },
  "fixtures": []
}
```

## Routing Status Rules

Each fixture should get one of:

### `routed`

Meaning:
- fixture already appears in routed `DEPLOY` or routed `OBSERVE`

Use:
- downstream intelligence merger can treat routed layer as primary owner

### `non_routed`

Meaning:
- covered fixture
- valid identity
- not already present in routed fixture-intelligence output
- candidate for later `CONTEXT` or `MONITOR`

### `unsupported`

Meaning:
- fixture exists but not inside current covered competition policy

Normally exclude from final public-safe artifact.

### `hidden`

Meaning:
- malformed
- missing identity
- unsafe join quality
- unusable for website-safe pipeline

## Source Availability Markers

The builder should not yet derive final context or signal text.

Instead it should record whether each enrichment family is available.

Recommended booleans:
- `fixtures_master`
- `routed_deploy`
- `routed_observe`
- `goal_shape_base`
- `prematch_odds`
- `injuries`
- `lineups`
- `team_stats`
- `player_stats`
- `match_events`

This gives downstream logic a clean basis for:
- `CONTEXT`
- `MONITOR`
- follow relevance
- premium intelligence depth

## Follow Candidate Rules

At builder stage, these can be simple:

- `team_follow_candidate = true`
- `fixture_follow_candidate = true`
- `league_follow_candidate = true`

for every covered fixture with valid identity

The point is:
- if the fixture is in the covered universe, it is inherently a candidate for follow systems later

## Exclusion Rules

The builder must exclude:
- malformed fixtures
- unsupported leagues
- duplicate fixture keys
- stale fixtures outside window
- obviously broken joins

If duplicate fixture keys appear:
- keep the best identity-complete row
- log the collision in the report

## Relationship To Future CONTEXT/MONITOR Classifier

This builder is not the classifier.

It should only produce the universe.

Downstream classifier will decide:
- `CONTEXT`
- `MONITOR`
- `HIDDEN`

based on:
- source availability
- overlay presence
- weak-shape availability
- followed-user relevance
- context materiality

## Builder Reporting

The operator report should include:

- source window
- covered leagues count
- total covered fixtures
- routed fixtures count
- non-routed fixtures count
- hidden fixtures count
- logo join counts
- source availability coverage counts
- top missing enrichment families by league

This report matters because it will immediately show:
- where coverage is thin
- where API overlays are missing
- where follow value is weak

## Recommended CLI

Recommended script shape:

```bash
python3 build_covered_fixture_universe.py --src predictions_output/<RUN>/<DEPLOY_PRESET_OR_ALLMARKETS_FILE>.csv
```

Optional:

```bash
python3 build_covered_fixture_universe.py --date-from 2026-05-09 --date-to 2026-05-11
```

## Validation Rules

Minimum:

- every fixture has:
  - `fixture_key`
  - `kickoff_time`
  - `league`
  - `home_team`
  - `away_team`
  - `routing_status`
  - `source_availability`
- no duplicate `fixture_key`
- no fixture outside window
- only covered leagues remain in final output
- no raw provider payload fragments

## Safe Implementation Order

1. implement builder as standalone helper
2. validate the resulting universe counts and identity joins
3. only after that, build the `CONTEXT` / `MONITOR` classifier on top

This keeps the system auditable and avoids mixing:
- intake problems
- enrichment problems
- publish-language problems

## Summary

Task 58 defines the canonical intake layer for all covered upcoming fixtures.

It is the first real implementation bridge from:
- routed fixture intelligence

to:
- full covered-league intelligence publishing

Once this builder exists, the next layer can finally decide:
- which non-routed fixtures deserve `CONTEXT`
- which deserve `MONITOR`
- and how followed users get value even when the engine produces no pick at all
