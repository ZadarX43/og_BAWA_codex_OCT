# Pre-Kickoff Snapshot Discipline - 2026-05-13

## Purpose

Verify whether the weekend intelligence layers can be treated as genuine pre-kickoff evidence.

This matters because the weekend scoring review showed promising signals, but we must separate:

- true pre-match intelligence
- late-published/current-site intelligence
- post-match discovery analysis

Without that split, we risk fooling ourselves into thinking the system knew something before kickoff when the payload was actually generated later.

## Audit Command

```bash
python3 scripts/audit_pre_kickoff_intelligence_snapshots.py
```

Outputs:

```text
reports/latest/pre_kickoff_snapshot_audit/
```

Key files:

```text
SUMMARY.md
summary.json
pre_kickoff_snapshot_audit_rows.csv
```

## Current Audit Result

Fixture feed:

```text
rows: 156
feed generated at: 2026-05-10T17:28:49Z
pre-kickoff rows: 34
post-kickoff rows: 122
```

By publish class:

```text
DEPLOY:  6 pre-kickoff, 20 post-kickoff
OBSERVE: 23 pre-kickoff, 83 post-kickoff
CONTEXT: 5 pre-kickoff, 19 post-kickoff
```

Provider kickoff matching:

```text
provider kickoff matched: 128
site feed kickoff used: 28
```

Decision / lineup / H2H payload audit after metadata backfill:

```text
fixture_decision_intelligence: 156 post-kickoff/backfill
fixture_lineup_intelligence: 156 post-kickoff/backfill
fixture_h2h_support: 156 post-kickoff/backfill
```

## What This Means

The weekend intelligence scoring is valid as a post-match discovery exercise.

It is not yet valid as proof that every intelligence warning existed before kickoff.

The public fixture feed and deeper payloads now carry explicit generation/cutoff metadata. Current decision, lineup, H2H, and demo external fixture payloads were metadata-backfilled on May 13, 2026, so they are auditable as post-kickoff/backfill snapshots rather than proof of pre-kickoff availability.

## Required Data Contract

Every publish-safe intelligence payload should carry:

```json
{
  "capture_generated_at": "2026-05-09T10:30:00Z",
  "source_data_cutoff_at": "2026-05-09T10:25:00Z",
  "fixture_kickoff_at": "2026-05-09T15:00:00Z",
  "pre_kickoff_eligible": true,
  "snapshot_phase": "pre_kickoff"
}
```

Recommended `snapshot_phase` values:

```text
early_pre_match
pre_kickoff
lineup_confirmed
live
post_match
backfill
```

## Application Rule

For deploy/review analysis:

```text
Only use rows where pre_kickoff_eligible = true
```

For post-match product pages:

```text
post_match and backfill rows are allowed, but must not be used to claim pre-match predictive evidence
```

## Next Build

The website publish/export layer now emits these timestamp fields into:

```text
frontend/public/data/fixture_intelligence_public.json
frontend/public/data/fixture_decision_intelligence/<fixture_key>.json
frontend/public/data/fixture_lineup_intelligence/<fixture_key>.json
frontend/public/data/fixture_h2h_support/<fixture_key>.json
frontend/public/data/external_content/fixture_media/<fixture_key>.json
```

The SQLite/D1 exporter also promotes the same fields into first-class columns on:

```text
fixtures
fixture_decisions
fixture_lineups
fixture_h2h
```

Backfill command:

```bash
python3 scripts/apply_snapshot_metadata_to_publish_estate.py
```

Then rerun:

```bash
python3 scripts/audit_pre_kickoff_intelligence_snapshots.py
python3 scripts/score_weekend_intelligence_audit.py --fetch-api-results --sleep-seconds 0 --daily-cap 75000
```

The deep GPT analysis should then be run twice:

1. all settled rows, for discovery
2. pre-kickoff eligible rows only, for evidence

That second file is the one that can influence future deploy policy.
