# Site SQLite/D1 Export And Coverage Runbook

Updated: 2026-05-12

## Purpose

This is the operational runbook for rebuilding the Odds Genius website data layer from the current publish estate, checking coverage, checking size, and preparing Cloudflare D1 import chunks.

Use this when we need to rerun or verify:

- the local `odds_genius.sqlite` website database
- the D1-compatible SQL chunk export
- the active-site premium data slice
- fixture, team, lineup, player-stat, and team-stat coverage
- launch-size limits before pushing data into D1

This exists so the SQLite/D1 work is not remembered only through terminal output or ignored report files.

## Core Rule

The launch export is active-site and latest-season only.

Default export scope:

- the 22 competitions currently published to the website
- the active published season for each competition
- the current fixture window
- page-shaped route payloads for fixture and team pages
- current-season lineup/player/team stat rows only where local normalized provider data exists

Do not use historical rows to make the launch product look artificially complete.

Historical/pro-depth data must stay behind explicit history mode or future high-tier routes.

## Current Launch Limits

Working launch limits:

- preferred launch SQLite/D1 size: under 300MB
- hard prelaunch ceiling: under 500MB
- no customer page route should depend on large table scans
- static JSON fallback should remain available until Worker/D1 parity is proven after a full publish cycle

Why:

- D1 Free has a 500MB per-database limit during testing.
- Workers Paid gives more room, but launch should not need it.
- D1 cost and speed depend on rows read/written, not only stored bytes.
- Cached page-shaped routes are the right website contract.

## Inputs

Static website publish estate:

```text
frontend/public/data/team_intelligence/
frontend/public/data/player_intelligence/
frontend/public/data/fixture_decision_intelligence/
frontend/public/data/fixture_lineup_intelligence/
frontend/public/data/fixture_h2h_support/
```

Canonical current API-Football normalized store:

```text
data_sources/api_football/normalized/
```

Report-scoped normalized sources that can be merged into the canonical store:

```text
reports/**/normalized/
```

Important scripts:

```text
scripts/api_football/import_current_site_normalized.py
scripts/export_site_sqlite.py
scripts/export_site_d1_chunks.py
scripts/benchmark_site_sqlite.py
scripts/benchmark_site_worker_routes.py
scripts/smoke_frontend_static.py
```

## Step 1: Import Current-Site Normalized Provider Files

Always dry-run first:

```bash
python3 scripts/api_football/import_current_site_normalized.py
```

Then write the canonical normalized files:

```bash
python3 scripts/api_football/import_current_site_normalized.py --write
```

This merges report-scoped current API-Football files into:

```text
data_sources/api_football/normalized/
```

It also writes:

```text
data_sources/api_football/normalized/current_site_normalized_import_manifest.json
reports/latest/CURRENT_SITE_NORMALIZED_IMPORT_REPORT.md
```

Important: `reports/latest/...` is not a safe memory layer if reports are ignored or not staged. Preserve important coverage numbers in versioned docs when they matter.

Latest measured import summary:

```text
active league tags: 22
records inspected: 88
lineups: 12,063 rows across 14 sourced files
match_player_stats: 12,066 rows across 14 sourced files
match_team_stats: 26 rows across 6 sourced files
fixtures_master: 416 rows across 19 sourced files
```

Coverage truth after this import:

- current lineups/player stats exist locally for 14 of 22 active competitions
- current team stats are still thin because `/fixtures/statistics` was only present for a small settled window
- the missing competitions need a fresh current-context API-Football refresh, not an exporter rewrite

Missing current lineups/player stats locally as of this run:

```text
Australia_A_League
Austria_Bundesliga
Denmark_Superliga
Germany_Bundesliga_2
Saudi_Pro_League
South_Korea_K_League
Switzerland_Super_League
Turkey_Super_Lig
```

## Step 2: Export The Launch SQLite Database

Default active-site export:

```bash
python3 scripts/export_site_sqlite.py --output /private/tmp/odds_genius_current_stats_slice.sqlite
```

The default mode is the launch mode. It does not include deep historical rows.

Only use this when explicitly testing historical/pro scope:

```bash
python3 scripts/export_site_sqlite.py --include-history --output /private/tmp/odds_genius_history_test.sqlite
```

Latest launch export baseline after the current-site normalized import:

```json
{
  "club_squads": 388,
  "fixture_decisions": 156,
  "fixture_h2h": 156,
  "fixture_lineups": 156,
  "fixtures": 156,
  "site_fixture_stats_payloads": 156,
  "site_formation_slots": 198,
  "site_lineup_slots": 12063,
  "site_player_identity_map": 15190,
  "site_player_match_stats": 12066,
  "site_team_match_stats": 26,
  "site_team_premium_payloads": 388,
  "team_intelligence": 388,
  "team_lineup_snapshots": 282,
  "size_bytes": 142766080
}
```

Human size:

```text
SQLite: about 143MB
```

Good launch signs:

- `fixtures` is 156
- `fixture_decisions` is 156
- `fixture_lineups` is 156
- `fixture_h2h` is 156
- `team_intelligence` is 388
- `club_squads` is 388
- `site_player_match_stats` and `site_lineup_slots` are non-zero after current normalized import
- SQLite remains under 300MB preferred, and under 500MB hard ceiling

Stop signs:

- launch SQLite exceeds 500MB
- fixture/decision/lineup/H2H counts drift unexpectedly
- `site_player_match_stats` or `site_lineup_slots` returns to 0 after an import that was expected to carry current provider files
- active competition data is silently backfilled from old historical rows
- compact route payload rows disappear. D1 exports intentionally skip source/evidence detail tables by default, not the site payloads.

## Step 3: Export D1 SQL Chunks

For incremental publishing, prefer the publish compiler before doing a full D1 chunk replacement:

```bash
python3 scripts/publish_compiler.py
```

This writes compact payload objects, `upload_plan.json`, and changed-only D1 SQL:

```text
build/site_publish/current/d1_changed_index.sql
```

Use the full D1 chunk export below when rebuilding or replacing the whole site-data database.

Create D1-compatible chunks from the SQLite file:

```bash
python3 scripts/export_site_d1_chunks.py --db /private/tmp/odds_genius_current_stats_slice.sqlite --output-dir /private/tmp/og_d1_current_stats_chunks --max-bytes 4194304
```

Default D1 exports insert compact route payloads and skip the heavy source/evidence detail tables:

- `site_player_identity_map`
- `site_player_match_stats`
- `site_team_match_stats`
- `site_match_events`
- `site_lineup_slots`
- `site_formation_slots`
- `site_fixture_market_intelligence`
- `site_player_event_shortlists`

The local Mac build calculates player, team, fixture, H2H, market, and player-event surfaces before export. Cloudflare D1 receives compact precomputed site payloads such as `site_fixture_stats_payloads` and `site_team_premium_payloads`.

Only include source/evidence detail rows for an explicit audit export:

```bash
python3 scripts/export_site_d1_chunks.py --db /private/tmp/odds_genius_current_stats_slice.sqlite --output-dir /private/tmp/og_d1_with_source_tables --include-source-tables
```

Latest compact D1 chunk baseline:

```text
chunks: 12
total SQL bytes: 42,544,322
human size: about 43MB
```

The first chunk contains schema. Later chunks contain inserts.

Use a test D1 database before production. The production binding is currently named:

```text
SITE_DATA_DB
```

Worker config locations:

```text
worker/wrangler.toml
worker/wrangler.example.toml
```

Typical D1 import shape:

```bash
cd worker
npx wrangler d1 execute odds-genius-site-data-test --remote --file /private/tmp/og_d1_current_stats_chunks/0000.sql
npx wrangler d1 execute odds-genius-site-data-test --remote --file /private/tmp/og_d1_current_stats_chunks/0001.sql
```

For all chunks, run the same command in chunk order. Do not import to production D1 until the test database verifies route parity.

## Step 4: Benchmark Local SQLite Page Reads

Run:

```bash
python3 scripts/benchmark_site_sqlite.py --db /private/tmp/odds_genius_current_stats_slice.sqlite --iterations 50
```

Latest measured local read shape:

```text
current fixtures: median about 0.165ms, p95 about 0.216ms
fixture detail: median about 0.057ms, p95 about 0.082ms
team lookup: median about 0.378ms, p95 about 0.485ms
```

These are local SQLite timings, not Cloudflare D1 timings. They prove the schema and page-shaped query shape are not inherently heavy.

## Step 5: Verify Worker Routes After D1 Import

Worker route contract:

```text
GET /api/site/fixtures/current?limit=200
GET /api/site/fixtures/:fixture_key
GET /api/site/teams/:competition_key/:team_slug
```

Production Worker URL:

```text
https://odds-genius-worker.hughcwade.workers.dev
```

Cached route benchmark:

```bash
python3 scripts/benchmark_site_worker_routes.py --iterations 30 --warmup 3
```

Decision gate:

- Worker-side fixture detail p95 under 75ms
- Worker-side current fixtures p95 under 75ms
- Worker-side team detail p95 under 100ms
- cache hit ratio above 90% during normal browsing
- response shape still allows static JSON fallback

Previous D1 route test passed this gate on the smaller 33MB estate. The current 143MB premium slice still needs test-D1 import and route benchmarking before production replacement.

## Step 6: Smoke Checks

Run:

```bash
python3 scripts/smoke_frontend_static.py
python3 -m py_compile scripts/api_football/import_current_site_normalized.py scripts/export_site_sqlite.py scripts/export_site_d1_chunks.py scripts/publish_compiler.py scripts/benchmark_site_sqlite.py scripts/benchmark_site_worker_routes.py
```

Optional file-size check:

```bash
du -h /private/tmp/odds_genius_current_stats_slice.sqlite
du -sh /private/tmp/og_d1_current_stats_chunks
```

## What The Counts Mean

`fixture_decisions = 156`, `fixture_lineups = 156`, and `fixture_h2h = 156` means the website has one canonical payload per current fixture key.

It does not mean every upstream layer is rich.

Lineup/H2H payloads can still be:

- confirmed provider lineup
- predicted from last fixture
- unavailable placeholder
- same-team-pair H2H fallback
- H2H unavailable placeholder

The product should preserve that truth in the UI.

`site_player_match_stats` and `site_team_match_stats` are factual provider stat layers. If they are thin, the correct fix is upstream provider refresh, not UI invention and not historical fill.

## Why We Built It This Way

This gives Odds Genius the premium data foundation without locking us into one database provider.

The frontend reads stable page-shaped Worker routes. Behind that route contract, the storage can remain D1 for launch or move later to Turso/Hetzner without rebuilding fixture and team pages.

It also protects launch discipline:

- active competitions only
- current season only
- compact cached payloads first
- deeper history reserved for high-tier/pro products
- static JSON fallback still available

## Related Docs

```text
docs/SITE_SQLITE_DATA_LAYER_PLAN.md
docs/PREMIUM_INTELLIGENCE_DATA_BUILD_PLAN_2026-05-12.md
docs/ACTIVE_SITE_INTELLIGENCE_PUBLISH_ESTATE_COMPLETION_2026-05-12.md
worker/README_WORKER.md
worker/DEPLOY_WORKER.md
```
