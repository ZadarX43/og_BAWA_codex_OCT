# Site Data Footprint Audit

Generated: 2026-05-14T13:20:58+00:00
Public data root: `frontend/public/data`

## Total Static Public Data

- Files: 13576
- JSON files: 13570
- Size: 107.3 MB (112532027 bytes)

## Group Breakdown

| Group | Files | JSON | Size |
| --- | ---: | ---: | ---: |
| player_intelligence | 82 | 80 | 56.0 MB |
| team_intelligence | 13004 | 13001 | 46.7 MB |
| logo_assets | 1 | 1 | 1.7 MB |
| fixture_decision_intelligence | 157 | 157 | 1.3 MB |
| fixture_lineup_intelligence | 157 | 157 | 297.0 KB |
| public_core | 6 | 6 | 244.9 KB |
| fixture_h2h_support | 157 | 157 | 220.9 KB |
| external_content | 7 | 7 | 18.1 KB |
| site_data | 0 | 0 | 0 B |
| weather_context | 0 | 0 | 0 B |

## Largest Files By Group

### player_intelligence

- `frontend/public/data/player_intelligence/club_squad_ratings.json` — 24.4 MB
- `frontend/public/data/player_intelligence/player_ratings.json` — 24.2 MB
- `frontend/public/data/player_intelligence/player_ratings.csv` — 1.5 MB
- `frontend/public/data/player_intelligence/clubs/brazil_serie_a/2026/vit_ria.json` — 115.5 KB
- `frontend/public/data/player_intelligence/clubs/brazil_serie_a/2026/remo.json` — 111.9 KB

### team_intelligence

- `frontend/public/data/team_intelligence/competitions/usa_leagues_cup__2023.json` — 173.1 KB
- `frontend/public/data/team_intelligence/competitions/usa_leagues_cup__2024.json` — 173.0 KB
- `frontend/public/data/team_intelligence/competitions/usa_leagues_cup__2025.json` — 132.6 KB
- `frontend/public/data/team_intelligence/team_ratings_index.json` — 115.6 KB
- `frontend/public/data/team_intelligence/competitions/usa_mls__2025.json` — 112.8 KB

### logo_assets

- `frontend/public/data/api_football_logo_asset_manifest.json` — 1.7 MB

### fixture_decision_intelligence

- `frontend/public/data/fixture_decision_intelligence/index.json` — 69.4 KB
- `frontend/public/data/fixture_decision_intelligence/2026_05_10_Vasco_da_Gama_Atl_tico_PR.json` — 10.8 KB
- `frontend/public/data/fixture_decision_intelligence/2026_05_09_RAAL_La_Louvi_re_Cercle_Brugge.json` — 10.7 KB
- `frontend/public/data/fixture_decision_intelligence/2026_05_10_Portland_Timbers_Sporting_KC.json` — 10.6 KB
- `frontend/public/data/fixture_decision_intelligence/2026_05_10_Atl_tico_Mineiro_Botafogo.json` — 10.5 KB

### fixture_lineup_intelligence

- `frontend/public/data/fixture_lineup_intelligence/index.json` — 78.2 KB
- `frontend/public/data/fixture_lineup_intelligence/2026_05_10_Crystal_Palace_Everton.json` — 14.6 KB
- `frontend/public/data/fixture_lineup_intelligence/2026_05_10_Burnley_Aston_Villa.json` — 14.5 KB
- `frontend/public/data/fixture_lineup_intelligence/2026_05_09_New_England_Revolution_Philadelphia_Union.json` — 3.0 KB
- `frontend/public/data/fixture_lineup_intelligence/2026_05_10_SJ_Earthquakes_Vancouver_Whitecaps.json` — 3.0 KB

### public_core

- `frontend/public/data/live_results_feed.json` — 178.6 KB
- `frontend/public/data/premium_predictions.json` — 39.3 KB
- `frontend/public/data/results_archive.json` — 14.2 KB
- `frontend/public/data/weekly_results.json` — 5.4 KB
- `frontend/public/data/public_predictions.json` — 5.3 KB

### fixture_h2h_support

- `frontend/public/data/fixture_h2h_support/index.json` — 78.0 KB
- `frontend/public/data/fixture_h2h_support/2026_05_09_New_England_Revolution_Philadelphia_Union.json` — 1.0 KB
- `frontend/public/data/fixture_h2h_support/2026_05_10_Fortuna_Sittard_PEC_Zwolle.json` — 1015 B
- `frontend/public/data/fixture_h2h_support/2026_05_09_Manchester_City_Brentford.json` — 1009 B
- `frontend/public/data/fixture_h2h_support/2026_05_09_Stevenage_Stockport_County.json` — 1008 B

### external_content

- `frontend/public/data/external_content/fixture_media/2026_05_10_FC_Barcelona_Real_Madrid.json` — 6.7 KB
- `frontend/public/data/external_content/source_registry.json` — 3.4 KB
- `frontend/public/data/external_content/team_news/fc_barcelona.json` — 2.8 KB
- `frontend/public/data/external_content/team_news/real_madrid.json` — 2.8 KB
- `frontend/public/data/external_content/news_sources.json` — 2.0 KB

## Hosting Notes

- Keep public proof/results as compact JSON.
- Keep current-season active competition data in D1/KV-backed route payloads.
- Reserve deep historical rows, full event logs, and downloadable payloads for higher tiers.
