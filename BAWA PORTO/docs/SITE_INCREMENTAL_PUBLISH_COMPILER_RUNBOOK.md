# Site Incremental Publish Compiler Runbook

Updated: `2026-05-19`

## Purpose

Use the MacBook/local build as the source-data and calculation layer, then publish only compact changed website artifacts.

This avoids pushing raw player, team, fixture, lineup, event, and match-stat source tables to Cloudflare every time.

## Boundary

Local only:

- API-Football normalized source rows
- raw player match stats
- raw team match stats
- raw match events
- lineup slots and formation slots
- player-event shortlist source rows
- fixture market source rows

Website publish layer:

- compact fixture page payloads
- compact team page payloads
- result payloads
- D1 index/cache rows needed by Worker routes
- manifest, changed manifest, upload plan, changed-only D1 SQL

## Run Order

Build or refresh the local site SQLite artifact:

```bash
python3 scripts/export_site_sqlite.py
```

Compile incremental publish artifacts:

```bash
python3 scripts/publish_compiler.py
```

Outputs:

```text
build/site_publish/current/manifest.json
build/site_publish/current/changed_manifest.json
build/site_publish/current/upload_plan.json
build/site_publish/current/d1_changed_index.sql
build/site_publish/current/payloads/
```

## Expected No-Op Behavior

Running the compiler twice without changing source data should produce:

```json
{
  "objects_changed": 0,
  "objects_changed_bytes": 0,
  "d1_rows_changed": 0
}
```

That is the main guardrail. If a no-op rebuild produces hundreds of changed objects, check for unstable timestamps or non-deterministic payload ordering before publishing.

## Changed-Only Publishing

Use `upload_plan.json` as the object upload list. It contains only changed object payloads and their target object keys.

Then apply the changed D1 rows only:

```bash
wrangler d1 execute odds-genius-site-data-test --remote --file build/site_publish/current/d1_changed_index.sql
```

Promote to production only after the test Worker and test D1 route checks pass:

```bash
wrangler d1 execute odds-genius-site-data --remote --file build/site_publish/current/d1_changed_index.sql
```

## Smoke Checks

Use a cache-busted fixture detail route:

```bash
curl -fsS "https://odds-genius-worker-site-data-test.hughcwade.workers.dev/api/site/fixtures/2026_05_09_Stuttgart_Bayer_Leverkusen?cb=publish-compiler"
```

Use a fixture with stats coverage:

```bash
curl -fsS "https://odds-genius-worker-site-data-test.hughcwade.workers.dev/api/site/fixtures/2026_05_10_Cremonese_Pisa/stats?cb=publish-compiler"
```

Production checks use the same paths on:

```text
https://odds-genius-worker.hughcwade.workers.dev
```

## Notes

- `scripts/publish_compiler.py` does not train models and does not touch deploy routing.
- The compiler reads `build/site_data/odds_genius.sqlite`.
- Source/evidence tables are excluded from the publish layer by design.
- R2/static object upload can be automated later from `upload_plan.json` without changing the compiler contract.
