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

After the fresh data/model/deploy overlay systems have produced their outputs, run the central site publish orchestrator:

```bash
python3 scripts/site_publish/orchestrator.py --from-date 2026-05-22 --to-date 2026-05-26
```

The orchestrator runs the publish-safe order:

1. settle public proof/result files
2. rebuild `build/site_data/odds_genius.sqlite`
3. inventory upstream coverage for the target window
4. recalculate injury shock impact against `FTR`, `BTTS`, and `OU25`
5. compile compact fixture-brain payloads
6. build local summary dry-run drafts from `summary_inputs`
7. compile compact publish artifacts from SQLite plus fixture brain
8. audit upcoming fixture page completeness

Upload/apply can be added to the same run when ready:

```bash
python3 scripts/site_publish/orchestrator.py --from-date 2026-05-22 --to-date 2026-05-26 --run-r2-upload --apply-d1
```

For manual/debug runs, the core individual commands are:

```bash
python3 scripts/export_site_sqlite.py
python3 scripts/build_injury_shock_market_impact_sidecar.py
python3 scripts/site_publish/fixture_brain_compiler.py
python3 scripts/build_fixture_summary_dry_run.py
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
- The fixture brain is compiled before `scripts/publish_compiler.py`, so fixture payloads can include the latest H2H/team/player/weather/injury intelligence.
- `scripts/build_fixture_summary_dry_run.py` reads only `summary_inputs`; it is a local GPT-summary contract smoke and does not call the OpenAI API.
- Source/evidence tables are excluded from the publish layer by design.
- R2/static object upload can be automated later from `upload_plan.json` without changing the compiler contract.
