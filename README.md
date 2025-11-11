# Odds Genius — Globe Fixtures UI

A lightweight, client-side web app that visualizes football fixtures on a 3D globe and pairs each geo-marker with a rich, right-hand insight panel. Left/right keys (and clicks) move between fixtures; the selected marker is highlighted with a soft ring pulse at ground level. CSV data drives everything (fixtures, geo, logos, stats).

## TL;DR

* **Data in:** `docs/data/fixtures.csv` (or `.tsv`).
* **Runs from:** static files (no server required).
* **Globe:** [three-globe] with robust ESM→CDN fallbacks.
* **Navigate:** click markers or press ← / →.
* **Logos:** `home_logo_url`, `away_logo_url` fields in CSV.
* **Highlight:** larger marker + subtle ground pulse ring.
* **No build step required.** Open `docs/index.html` in a web server.

---

## Features

* 🌍 **Accurate geo-mapping** — uses three-globe’s `pointsData`, linked to `latitude`/`longitude`.
* 🧭 **Left/Right navigation** — camera flies to next fixture; panel updates; marker selection syncs.
* 🏷️ **Club crests** — top badges pull logos from CSV (fallback to initials).
* ✨ **Visual clarity** — tuned bloom, subtle pulse ring at surface; markers hug the sphere.
* 📈 **Useful stats** — tidy xG edge, full-time prediction confidence, PPG momentum.
* 🧩 **Zero back-end** — CSV is parsed client-side (PapaParse).
* 🧱 **Resilient loading** — tries local ESM, then esm.sh, then UMD CDNs; helpful on-screen error if not found.

---

## Tech Stack

* **Three.js** for WebGL rendering
* **three-globe** for geo projections/marker placement
* **EffectComposer + FXAA + Bloom** for polish
* **PapaParse** for CSV parsing
* Vanilla JS modules, no bundler needed

---

## Data schema (CSV)

> Save as `docs/data/fixtures.csv`. Missing fields are handled gracefully.

| Field                            | Type         | Notes                                                            |
| -------------------------------- | ------------ | ---------------------------------------------------------------- |
| `fixture_id`                     | string       | Unique id for selection/click sync                               |
| `home_team`, `away_team`         | string       | Team names                                                       |
| `home_logo_url`, `away_logo_url` | string (URL) | Optional club crest URLs                                         |
| `date_utc`                       | ISO datetime | Ex: `2025-11-28T17:45:00Z`                                       |
| `competition`                    | string       | League/competition                                               |
| `stadium`,`city`,`country`       | string       | For panel context (+ fallback geocode)                           |
| `latitude`,`longitude`           | number       | Decimal degrees; if missing we try a stadium fallback map        |
| `predicted_winner`               | string       | e.g., `Lazio`                                                    |
| `confidence_ftr`                 | number (0–1) | Win confidence                                                   |
| `xg_home`,`xg_away`              | number       | Expected goals                                                   |
| `ppg_home`,`ppg_away`            | number       | Points per game                                                  |
| `over25_prob`,`btts_prob`        | number (0–1) | Market snapshot                                                  |
| `key_players_shots`              | string       | `Name\|detail;Name\|detail` (escaped pipe keeps table alignment) |
| `key_players_bookings`           | string       | Same format as shots                                             |
| `key_players_tackles`            | string       | Same format as shots                                             |

**Example:**

```csv
fixture_id,home_team,away_team,home_logo_url,away_logo_url,date_utc,competition,stadium,city,country,latitude,longitude,predicted_winner,confidence_ftr,xg_home,xg_away,ppg_home,ppg_away,over25_prob,btts_prob,key_players_shots,key_players_bookings,key_players_tackles
abc123,Lazio,Celtic,https://..../lazio.png,https://..../celtic.png,2025-11-28T17:45:00Z,UEFA Champions League,Stadio Olimpico (Roma),Rome,Italy,41.9339,12.4545,Lazio,0.61,1.8,1.0,1.8,0.3,0.58,0.56,"Pedro|Shots/90 1.93;Gustav|Shots/90 1.85","Nicolò|Cards/90 1.97","Nicolò|Tackles/90 3.28"
```

---

## Project layout

```
docs/
  index.html
  styles.css
  app.module.js
  data/
    fixtures.csv
  vendor/
    three-globe.module.js      # (optional local ESM)
    three-globe.min.js         # (optional local UMD)
    OrbitControls.js
    EffectComposer.js
    RenderPass.js
    ShaderPass.js
    UnrealBloomPass.js
    FXAAShader.js
    CopyShader.js
    LuminosityHighPassShader.js
```

> If you don’t include local `three-globe` files, the app will automatically try esm.sh or CDNs.

---

## Quick start

1. **Clone or drop files** into a folder.
2. Put your **fixtures** at `docs/data/fixtures.csv`.
3. Serve the `docs` folder:

   ```bash
   # any static server works
   npx serve docs
   # or
   python3 -m http.server -d docs 8080
   ```
4. Open `http://localhost:PORT`.

---

## Controls

* **Click** a marker → focus fixture.
* **Left/Right** arrow keys → previous/next fixture (camera and panel sync).
* Scroll/drag to orbit; auto-rotate is enabled; pan is disabled (cleaner UX).

---

## How it works

* **Load CSV** → parse to normalized fixture objects.
* **Bind to globe** → `globe.pointsData(fixtures)` uses lat/lng to position markers at surface altitude (`~0.02` + confidence lift).
* **Selection** → the active fixture sets `__active=true` (larger marker) and draws a subtle **pulse ring** at ground level.
* **Panel** → renders prediction, xG edge, PPG momentum, player lists, market snapshot.
* **Logos** → top badges render `<img>` if URLs exist; fall back to 2–3-letter initials.

---

## Styling hooks

* Crest badges:

  ```css
  .badge img { width:100%; height:100%; object-fit:contain; display:block; }
  .badge.has-logo { background: rgba(255,255,255,.08); }
  ```
* Adjust bloom feel in `app.module.js` (strength/threshold) if you tweak colors.

---

## Design for scale (hundreds of fixtures)

* Points are instanced; the pulse ring is a single animated item – light to render.
* Avoid world-space text labels for every point; we use `pointLabel` tooltips.
* **Next step for high density:** add **zoom-based clustering**:
  * At higher altitude: cluster markers per city/region (`{ lat,lng,count }` with a “+N” marker).
  * As you zoom in: decompose cluster to individual fixtures.
* If you need filtering (date, league, country), filter the `fixtures` array then call `globe.pointsData(filtered)` and refresh the panel selection logic.

---

## Troubleshooting

* **Globe fails to load**: you’ll see a visible error telling you to add a local copy of `three-globe` in `docs/vendor/…`. Otherwise the loader tries esm.sh and jsDelivr/unpkg automatically.
* **No fixtures rendered**: the CSV path or headers are wrong. Check dev console for parse errors. Verify `fixture_id`, `latitude`, `longitude` exist.
* **Logos missing**: ensure `http(s)` URLs and no CORS blocks.
* **Markers look “off”**: we keep altitude tiny and ring anchored at `0.02`. Don’t set huge pointAltitudes.

---

## License & credits

* **three.js** / **three-globe** under their respective open-source licenses.
* UI/logic © Odds Genius (you).

---

## Roadmap / TODO

* [ ] Zoom-based clustering for dense fixture sets.
* [ ] Filters: date range, competition, country.
* [ ] Pin on selection: keep selected marker always on top with slight glow.
* [ ] Accumulator builder / slip audit integration (OCR pipeline).
* [ ] Persist last viewed fixture (localStorage).
* [ ] Add “jump to today / this weekend” quick nav.

---

## Maintainers’ cheatsheet

* **Change auto-rotate**: `controls.autoRotateSpeed`.
* **Marker size**: `globe.pointRadius(d => d.__active ? 0.65 : 0.22)`.
* **Ring style**: `globe.ringColor(() => 'rgba(102,227,210,0.85)')`, period/speed in `updateHighlight()`.
* **Bloom feel**: tweak `new UnrealBloomPass(..., strength, radius, threshold)`.

---

This should be everything Codex needs to recognize structure, run locally, and know where to add features (clustering, filters, etc.).
