# External Content Intelligence Plan

Updated: 2026-05-13

## Purpose

Add legally safer external context to Odds Genius without turning the product into a news site.

The site should use outside content as evidence, media, or context, then let Odds Genius provide the interpretation:

- fixture highlights
- injury and team-news signals
- manager quote context
- weather overlays
- experimental environmental volatility overlays
- post-match reaction/sentiment summaries

## Product Rule

Odds Genius is not competing with BBC, Sky, ESPN, YouTube, or club websites.

External sources provide raw information.

Odds Genius turns that information into football intelligence.

## Safe Usage Model

### YouTube highlights

Use public YouTube embeds through the YouTube player.

Rules:

- embed only
- do not download videos
- do not rehost thumbnails or video files unless separately licensed
- prefer official channels from leagues, clubs, competitions, and broadcasters
- keep player controls and attribution intact

Current implementation:

- source registry: `frontend/public/data/external_content/source_registry.json`
- fixture media index: `frontend/public/data/external_content/fixture_media/index.json`
- fixture media payloads: `frontend/public/data/external_content/fixture_media/<fixture_key>.json`
- context builder: `scripts/external_content/build_fixture_context_signals.py`

### RSS and article sources

Use RSS/news as a signal input, not as copied article content.

Public display should normally include:

- headline
- publisher
- timestamp
- outbound link
- short Odds Genius interpretation written by us

Avoid:

- full article reproduction
- large quoted extracts
- provider-owned images unless explicitly licensed
- misleading source presentation

Good public UI:

```text
Arsenal creativity watch
Odegaard is reported doubtful. If confirmed, the squad model reduces central creation trust and raises BTTS caution.
Source: BBC Sport
```

### Weather

Weather belongs in the interpretation layer:

- pace suppression
- fatigue amplification
- crossing efficiency
- card volatility
- shot accuracy suppression
- upset volatility

Production note:

Open-Meteo is useful, but its free API is not the default commercial-production answer unless the correct terms/plan are confirmed.

### Space weather

This can be a distinctive Odds Genius layer, but it must stay experimental.

Public wording should be:

```text
Experimental environmental volatility
Solar/weather context is monitored as a weak volatility overlay. It does not force a prediction.
```

Avoid:

```text
Solar storms make this pick stronger.
```

## Data Contract

### `external_content/source_registry.json`

Purpose:

One canonical public-safe registry of sources, usage modes, and rules.

Core fields:

- `source_id`
- `provider`
- `usage_mode`
- `terms_url`
- `notes`

### `external_content/fixture_media/index.json`

Purpose:

Small lookup index for fixtures with known media payloads.

Core fields:

- `fixture_key`
- `media_count`
- `primary_type`
- `source_id`

### `external_content/fixture_media/<fixture_key>.json`

Purpose:

Per-fixture external content payload.

Current fields:

- `fixture_key`
- `updated_at`
- `media`
- `news_signals`
- `weather_signals`
- `space_weather_signals`
- `sentiment_signals`

Media item fields:

- `content_id`
- `type`
- `source_id`
- `provider`
- `title`
- `heading`
- `summary`
- `source_url`
- `embed_url`
- `usage_mode`
- `rights_note`
- `priority`

## Website Behavior

Fixture pages now try to load:

```text
frontend/public/data/external_content/fixture_media/<fixture_key>.json
```

If a YouTube embed exists, the hero media module renders from that payload.

If weather or space-weather context exists, the fixture page renders a Weather Forecast card above the highlights module.

If no payload exists, nothing is shown.

There is currently a fallback for the Barcelona vs Real Madrid demo fixture so the hero remains stable while the external-content layer is rolled out.

## Database / Worker Direction

The database tables now used for the first slice are:

- `site_external_sources`
- `site_fixture_external_content`
- `site_fixture_context_payloads`

News, weather, space-weather, media, and sentiment items are stored as typed rows in `site_fixture_external_content` and cached into one page-shaped context payload per fixture. Separate typed tables can still be added later if query pressure justifies them.

Suggested route:

```text
GET /api/site/fixtures/:fixture_key/context
```

Route payload should be page-shaped:

- top media embed
- source-backed signal cards
- weather context
- post-match reaction context
- source audit metadata

## Subscription Fit

### £20

- official/embedded highlights where available
- top source-backed context cards
- basic weather note

### £49

- team news intelligence
- injury/lineup rumour translation
- player-event impact notes
- weather adjustment cards

### £99

- richer market impact from external signals
- combo-market context
- player shortlist adjustments
- source confidence scoring

### £500

- source audit metadata
- alert routing
- downloadable route payloads
- operational source freshness and stale-data warnings

## Current Build Slice Completed

The first implementation slice now exists:

```bash
python3 scripts/external_content/build_fixture_context_signals.py --demo-barca
python3 scripts/export_site_sqlite.py
python3 scripts/export_site_d1_chunks.py
```

The builder supports:

- RSS/Atom ingestion through `--rss-url`
- headline/link/source-only matching into `news_signals`
- demo weather context for `2026_05_10_FC_Barcelona_Real_Madrid`
- demo space-weather monitoring for the same 24-hour fixture window

The Barcelona demo fixture currently carries:

- one YouTube highlight embed
- one Weather Forecast card
- one Space Weather monitor card
- four source-linked news signals:
  - FC Barcelona official first-team news source page
  - Real Madrid official first-team news source page
  - Sky Sports football news watch source page
  - BBC Sport football RSS monitor source

SQLite/D1 export now includes those context items in:

- `site_fixture_external_content`
- `site_fixture_context_payloads`

## News Feed Implementation Update

Added:

- `frontend/public/data/external_content/news_sources.json`
- `frontend/public/data/external_content/team_news/<team_slug>.json`
- `frontend/public/data/external_content/team_news/index.json`
- Fixture `News` tab rendering from `news_signals`
- Team `News` tab rendering from `team_news/<team_slug>.json`

Current demo payloads:

- `2026_05_10_FC_Barcelona_Real_Madrid`: 4 news signals
- `fc_barcelona`: 3 news signals
- `real_madrid`: 3 news signals

Official club websites are currently treated as source-page links, not scraped feeds. BBC Sport is configured as an RSS-capable source, but broad RSS matching must stay conservative because generic terms such as `real`, `club`, `city`, and `united` create false positives. The matcher now excludes those generic team tokens.

Run the controlled demo seed without network RSS:

```bash
python3 scripts/external_content/build_fixture_context_signals.py \
  --source-config /private/tmp/nonexistent-news-sources.json \
  --demo-barca \
  --demo-barca-news
```

Run configured RSS ingestion when network access and source policy are approved:

```bash
python3 scripts/external_content/build_fixture_context_signals.py --demo-barca --demo-barca-news
```

Then export:

```bash
python3 scripts/export_site_sqlite.py
python3 scripts/export_site_d1_chunks.py
```

Current measured D1/static impact after adding the demo news signals:

- `site_fixture_external_content`: 7 rows
- `site_fixture_context_payloads`: 1 cached fixture context payload
- D1 chunk total: ~144.9MB
- SQLite: ~156.4MB

## Next Build Slice

1. Add a small RSS ingestion script that writes headline/link/source-only signal payloads.
2. Add weather context payloads for active fixture venues.
3. Export those payloads into SQLite/D1.
4. Add Worker route `GET /api/site/fixtures/:fixture_key/context`.
5. Add a Fixture Context tab section:
   - media
   - team news signals
   - weather signal
   - source audit footer

## Guardrail

This must remain a website intelligence layer.

Do not edit:

- `deploy_rulebook.py`
- `bookie_allmarkets.py`
- `slip_formatter.py`
- protected production spine files
