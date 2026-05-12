# Site SQLite Data Layer Plan

Updated: 2026-05-12

## Purpose

Create a low-risk bridge from the current publish-safe JSON estate to a database-backed website data layer.

The goal is to avoid a post-launch rewrite if Odds Genius scales quickly. The frontend can keep its current product shape while the data access path moves behind a small Worker API backed by SQLite-compatible storage.

## What Exists Now

Local exporter:

```bash
python3 scripts/export_site_sqlite.py
```

The default export is active-site scoped: current published fixture competitions plus the latest team/player season for each active competition. Full historical export is still available when explicitly needed:

```bash
python3 scripts/export_site_sqlite.py --include-history
```

Default output:

```text
build/site_data/odds_genius.sqlite
```

Benchmark:

```bash
python3 scripts/benchmark_site_sqlite.py --iterations 500
```

Worker query contract:

```text
worker/src/site_data_store.js
```

Optional Worker routes, activated only when a future `SITE_DATA_DB` binding exists:

```text
GET /api/site/fixtures/current
GET /api/site/fixtures/:fixture_key
GET /api/site/teams/:competition_key/:team_slug
```

D1 chunk exporter:

```bash
python3 scripts/export_site_d1_chunks.py
```

Cached route benchmark:

```bash
python3 scripts/benchmark_site_worker_routes.py --iterations 30 --warmup 3
```

Player publish pruning helper:

```bash
python3 scripts/prune_player_intelligence_active_site.py
```

## Current Export Result

Latest local export from the current publish estate:

```json
{
  "fixtures": 156,
  "fixture_decisions": 156,
  "fixture_lineups": 156,
  "fixture_h2h": 156,
  "team_intelligence": 388,
  "club_squads": 388,
  "team_lineup_snapshots": 287,
  "size_bytes": 33255424,
  "elapsed_ms": 805
}
```

The database is about 33 MB after trimming team/player intelligence to the active-site latest-season footprint and preserving rich team-intelligence bundle payloads.

The stale broad player publish estate was the real size problem:

```text
before prune: player_intelligence about 1.8G
after prune:  player_intelligence about 84M
```

The pruned player estate contains:

```json
{
  "active_competitions": 22,
  "club_squad_ratings": 388,
  "player_ratings": 11526,
  "kept_club_files": 388
}
```

## Benchmark Result

Local SQLite benchmark over 500 iterations:

```json
{
  "fixture_detail": {
    "median_ms": 0.218,
    "p95_ms": 0.398
  },
  "current_fixtures": {
    "median_ms": 0.2,
    "p95_ms": 0.277
  },
  "team_lookup": {
    "median_ms": 0.44,
    "p95_ms": 0.526
  }
}
```

These are local timings, not Cloudflare D1/Turso/Hetzner timings. They are still useful because they prove the schema and page-shaped reads are not inherently heavy.

## D1 Test Result

Test database:

```text
odds-genius-site-data
database_id: 2a21d853-9d3e-4c6e-bc03-a2e138c96020
region: WEUR
remote size: 33.29 MB
```

Test Worker:

```text
https://odds-genius-worker-site-data-test.hughcwade.workers.dev
```

The isolated test database remains available as:

```text
odds-genius-site-data-test
database_id: 2e03996d-257f-434a-8599-623c35bce4aa
```

Imported row counts:

```json
{
  "fixtures": 156,
  "fixture_decisions": 156,
  "fixture_lineups": 156,
  "fixture_h2h": 156,
  "team_intelligence": 388,
  "club_squads": 388,
  "team_lineup_snapshots": 287
}
```

Real Worker route benchmark over 30 requests per route:

```json
{
  "current_fixtures": {
    "worker_elapsed_median_ms": 13.5,
    "worker_elapsed_p95_ms": 21,
    "network_total_median_ms": 281.51,
    "network_total_p95_ms": 761.01
  },
  "fixture_detail": {
    "worker_elapsed_median_ms": 20,
    "worker_elapsed_p95_ms": 28,
    "network_total_median_ms": 335.37,
    "network_total_p95_ms": 1491.55
  },
  "team_detail": {
    "worker_elapsed_median_ms": 46,
    "worker_elapsed_p95_ms": 83,
    "network_total_median_ms": 541.46,
    "network_total_p95_ms": 1243.72,
    "lineup_snapshot_all_present": true
  }
}
```

This passes the D1 decision gate for Worker-side read time. The higher network totals are outside-Worker request latency and should be handled with normal Worker/CDN caching.

The team route now resolves provider-ish lineup snapshot keys such as `club_brugge_kv` from site-facing slugs such as `club_brugge`. That keeps the team response complete, although a future alias table would make this cheaper than scanning competition snapshots.

## Frontend API Fallback

The frontend is wired to prefer the site-data Worker routes through:

```js
window.OG_CONFIG.SITE_DATA_API_BASE
```

Current default:

```text
https://odds-genius-worker.hughcwade.workers.dev
```

Fallback behavior:

- current fixture rows: Worker `/api/site/fixtures/current?limit=200`, then `fixture_intelligence_public.json`
- fixture detail: Worker `/api/site/fixtures/:fixture_key`, then static decision/lineup/H2H payloads
- team detail: Worker `/api/site/teams/:competition_key/:team_slug`, then static team/player payloads

The static team fallback now uses the real competition bundle shape:

```text
team_intelligence/competitions/<competition_key>__<season>.json
```

## Cached Route Test

After enabling Worker edge caching for `/api/site/...`, the cached benchmark returned `x-og-site-cache: HIT` for all measured requests.

Production Worker promotion completed:

```text
https://odds-genius-worker.hughcwade.workers.dev
SITE_DATA_DB -> odds-genius-site-data
database_id: 2a21d853-9d3e-4c6e-bc03-a2e138c96020
version: 8096da1f-4ce6-425a-b4eb-b5eb551d168e
```

Production route verification:

```json
{
  "health_has_site_data_db": true,
  "fixture_detail": {
    "decision": true,
    "lineup": true,
    "h2h": true
  },
  "team_detail": {
    "team": "Club Brugge",
    "og_power_rating": 77,
    "squad": true,
    "lineup_snapshot": true
  }
}
```

Latest cached run:

```json
{
  "current_fixtures": {
    "cache_hit_requests": 10,
    "network_total_median_ms": 572.36,
    "network_total_p95_ms": 779.1,
    "bytes_median": 374906
  },
  "fixture_detail": {
    "cache_hit_requests": 10,
    "network_total_median_ms": 237.26,
    "network_total_p95_ms": 417.46,
    "bytes_median": 40767
  },
  "team_detail": {
    "cache_hit_requests": 10,
    "network_total_median_ms": 248.53,
    "network_total_p95_ms": 286.08,
    "bytes_median": 112548
  }
}
```

The current-fixtures route now returns full fixture payloads so it can replace the static fixture feed safely. Browser compression will reduce transfer size compared with this Python benchmark, but this route is the one to watch if the fixture window grows.

## Schema Shape

The export uses narrow lookup tables with full payload JSON preserved:

```text
fixtures
fixture_decisions
fixture_lineups
fixture_h2h
team_intelligence
club_squads
team_lineup_snapshots
metadata
```

This gives us both:

- indexed SQL lookups for speed
- full existing payload compatibility for the frontend/Worker response shape

## Scaling Direction

Recommended production path:

1. Keep Cloudflare Pages as the frontend.
2. Serve page-shaped reads through the Cloudflare Worker.
3. Put this schema into Cloudflare D1 first.
4. Add Worker/CDN caching on the three site API routes.
5. Keep static JSON as fallback until parity is proven.

Why this avoids rework:

- the frontend reads from stable page-shaped API routes
- the storage backend can move from D1 to Turso or Hetzner later without changing page code
- publish scripts already have a single database export target

## Provider Decision Gate

Do not pick infrastructure by instinct. Pick it after testing this exported shape on candidate platforms.

Decision criteria:

- fixture detail p95 under 75 ms from the Worker
- current fixtures p95 under 75 ms from the Worker
- team detail p95 under 100 ms from the Worker
- cache hit ratio above 90% during normal browsing
- database import can complete safely during publish
- backup/restore is boring and repeatable

## Next Implementation Step

Treat D1 as good enough for launch unless a broader load test disproves it.

Immediate next work:

- keep the active-site player prune in the publish workflow
- consider a dedicated team alias table for cheaper snapshot joins
- promote the frontend config change through Git/Cloudflare Pages so the public Pages site uses the production Worker route
- keep static JSON fallback until production page parity has been observed for a full publish cycle
