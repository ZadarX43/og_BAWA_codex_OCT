# Team And Player Ratings Engine Spec

## Objective
Build a publish-safe Odds Genius intelligence layer that converts raw season statistics into proprietary team and player ratings for the website and app.

The public product should consume derived ratings, profile tags, and short explanations.
It should not expose raw provider columns or internal weighting formulas.

## Scope
This spec covers:
- `team_rating_engine.py`
- `player_rating_engine.py`
- a shared source configuration contract
- website-ready JSON and CSV exports

This spec does not yet cover:
- H2H control integration
- formation pitch rendering
- post-match player impact scoring

Fixture lineup unit ratings are now reserved as the next publish-safe layer:
- `frontend/public/data/fixture_lineup_intelligence/<fixture_key>.json`
- `frontend/public/data/fixture_h2h_support/<fixture_key>.json`
- optional frontend consumption only
- no raw lineup provider payloads exposed directly in the browser

## Source Boundary
Raw provider files remain hidden inputs only.

Expected source families:
- FootyStats-style team season CSV
- FootyStats/API-Football style player season CSV
- optional lineup / H2H / fixture files later

Protected production files such as `team_ratings.py` remain separate from this publish-safe layer.

## Output Philosophy
The website should read proprietary intelligence objects, not raw CSV rows.

### Team output example
```json
{
  "team": "Arsenal",
  "competition": "Premier League",
  "season": "2025/2026",
  "ratings": {
    "og_power_rating": 95,
    "attack_flow_rating": 86,
    "defensive_lock_rating": 89,
    "goal_heat_rating": 58,
    "btts_pressure_rating": 48,
    "over25_heat_rating": 57,
    "control_rating": 82,
    "first_strike_rating": 86,
    "corner_pressure_rating": 78,
    "card_heat_rating": 18,
    "chaos_rating": 24,
    "home_fortress_rating": 94,
    "away_threat_rating": 82
  },
  "profile_tags": [
    "Elite Control Side",
    "Low Chaos",
    "Strong First Goal"
  ]
}
```

### Player output example
```json
{
  "name": "Bukayo Saka",
  "club": "Arsenal",
  "competition": "Premier League",
  "season": "2025/2026",
  "position_group": "winger",
  "ratings": {
    "og_player_power": 91,
    "goal_threat": 84,
    "creative_spark": 92,
    "pressing_heat": 78,
    "ball_progression": 86,
    "discipline_risk": 31,
    "booking_heat": 22
  },
  "ranks": {
    "league_overall_rank": 14,
    "position_rank": 3,
    "club_rank": 2
  },
  "tags": [
    "Elite Creator",
    "Wide Goal Threat",
    "High Usage Attacker"
  ]
}
```

## Rating Bands
Use one shared language system across team, player, fixture, and market surfaces.

| Score | Label | Meaning |
| --- | --- | --- |
| 90–100 | Elite | Top-tier signal |
| 80–89 | Strong | Very positive |
| 70–79 | Positive | Useful edge |
| 55–69 | Mixed | Needs opponent/context |
| 40–54 | Weak | Limited trust |
| 0–39 | Red Flag | Avoid or danger |

## Team Ratings Phase 1
Phase 1 team ratings:
- `og_power_rating`
- `attack_flow_rating`
- `defensive_lock_rating`
- `goal_heat_rating`
- `btts_pressure_rating`
- `over25_heat_rating`
- `control_rating`
- `first_strike_rating`
- `corner_pressure_rating`
- `card_heat_rating`
- `chaos_rating`
- `home_fortress_rating`
- `away_threat_rating`

### Team normalization
- cohort = competition + season
- score every underlying stat league-relatively
- use percentile-style scaling to `0–100`
- invert negative indicators where appropriate
- apply a confidence multiplier using matches played

### Team confidence
- low sample teams should be pulled toward neutral `50`
- neutralization formula:
  - `adjusted = 50 + ((raw - 50) * confidence_multiplier)`

## Player Ratings Phase 1
Phase 1 player ratings:
- `og_player_power`
- `goal_threat`
- `shot_threat`
- `xg_threat`
- `creative_spark`
- `xa_threat`
- `midfield_engine`
- `defensive_lock`
- `pressing_heat`
- `ball_progression`
- `aerial_dominance`
- `goalkeeper_shield`
- `discipline_risk`
- `booking_heat`

### Player normalization
- cohort = competition + season
- position-aware weighting
- use per-90 stats where possible
- use league-relative and position-relative ranking outputs
- apply minutes confidence adjustment

### Minutes confidence bands
| Minutes | Label |
| --- | --- |
| 0–299 | Very low sample |
| 300–599 | Low sample |
| 600–899 | Medium sample |
| 900–1499 | Trusted |
| 1500+ | Strong sample |

## Publish Outputs
Default public output root:
- `frontend/public/data/`

### Team outputs
- `frontend/public/data/team_intelligence/team_ratings.csv`
- `frontend/public/data/team_intelligence/team_ratings_index.json`
- `frontend/public/data/team_intelligence/competitions/<competition_key>__<season>.json`
- `frontend/public/data/team_intelligence/teams/<competition_key>/<season>/<team_slug>.json`

### Player outputs
- `frontend/public/data/player_intelligence/player_ratings.csv`
- `frontend/public/data/player_intelligence/player_ratings.json`
- `frontend/public/data/player_intelligence/club_squad_ratings.json`
- `frontend/public/data/player_intelligence/clubs/<competition_key>/<season>/<club_slug>.json`

## Frontend Consumption Contract
The frontend should consume publish-safe intelligence objects only.
It must not read raw provider columns directly.

### Final frontend data paths
Core static exports:
- `frontend/public/data/public_predictions.json`
- `frontend/public/data/premium_predictions.json`
- `frontend/public/data/publish_summary.json`
- `frontend/public/data/fixture_intelligence_public.json`
- `frontend/public/data/weekly_results.json`

Team intelligence layer:
- `frontend/public/data/team_intelligence/team_ratings_index.json`
- `frontend/public/data/team_intelligence/competitions/<competition_key>__<season>.json`
- `frontend/public/data/team_intelligence/teams/<competition_key>/<season>/<team_slug>.json`

Player intelligence layer:
- `frontend/public/data/player_intelligence/club_squad_ratings.json`
- `frontend/public/data/player_intelligence/clubs/<competition_key>/<season>/<club_slug>.json`
- `frontend/public/data/player_intelligence/player_ratings.json`

Fixture lineup intelligence layer:
- `frontend/public/data/fixture_lineup_intelligence/<fixture_key>.json`

Fixture H2H support layer:
- `frontend/public/data/fixture_h2h_support/<fixture_key>.json`
- `frontend/public/data/fixture_h2h_support/index.json`

Fixture decision intelligence layer:
- `frontend/public/data/fixture_decision_intelligence/<fixture_key>.json`
- `frontend/public/data/fixture_decision_intelligence/index.json`

### Required index fields for frontend routing
`team_ratings_index.json` entries must expose:
- `team`
- `team_slug`
- `competition`
- `competition_key`
- `season`
- `headline_rating`
- `headline_band`
- `profile_tags`

`club_squad_ratings.json` entries must expose:
- `club`
- `club_slug`
- `competition`
- `competition_key`
- `season`
- `leaders`
- `players`

### Team detail contract
Per-team detail JSON is the canonical team desk intelligence object.
The frontend may safely render:
- `summary.headline`
- `summary.profile`
- `summary.primary_strengths`
- `summary.main_caution`
- `profile_tags`
- `ratings`
- `rating_bands`
- `rating_explanations`
- `market_tendencies`
- `home_profile`
- `away_profile`
- `timing_profile`
- `sample_confidence`

### Club squad contract
Per-club squad JSON is the canonical player-intelligence object for team desks.
The frontend may safely render:
- `club`
- `club_slug`
- `competition`
- `competition_key`
- `season`
- `leaders`
- `players`

Each player object may safely expose:
- `name`
- `surname`
- `position`
- `position_group`
- `minutes_confidence`
- `ratings`
- `rating_bands`
- `rating_explanations`
- `ranks`
- `tags`
- `ui`

### Fixture lineup intelligence contract
Per-fixture lineup intelligence JSON is the canonical optional unit/mismatch layer for fixture lineups.
The frontend may safely render:
- `fixture`
- `home_team`
- `away_team`
- `home_formation`
- `away_formation`
- `home_units`
- `away_units`
- `key_mismatches`
- `player_matchups`
- `home_lineup_profiles`
- `away_lineup_profiles`
- `home_resolution`
- `away_resolution`
- `formation_context`

### Fallback behavior
- Team desks must continue working when team/player intelligence JSON is absent.
- Current-window fixture-linked rendering remains the fallback source.
- Intelligence files enrich the team experience; they do not replace the protected production publish layer.

## Source Config Contract
Use a config JSON so file paths can be managed separately from the engine logic.

Example shape:
```json
{
  "teams": [
    {
      "source": "New Additions/example/league-teams.csv",
      "competition_name": "USA MLS",
      "competition_key": "usa_mls",
      "season": "2024"
    }
  ],
  "players": [
    {
      "source": "New Additions/example/league-players.csv",
      "competition_name": "USA MLS",
      "competition_key": "usa_mls",
      "season": "2024"
    }
  ]
}
```

Recommended live filename when source paths are ready:
- `ratings_publish_sources.json`

Keep that file path/config separate from the public website exports.

## Public-Safe Rules
- never publish raw provider rows directly
- never expose internal formulas or exact feature weights in public JSON
- expose short human explanations, tags, bands, and derived scores only
- keep optional raw-debug exports internal or disabled by default

## Recommended Build Order
1. Implement `team_rating_engine.py`
2. Add source config support
3. Export website-ready team JSON and CSV
4. Implement `player_rating_engine.py`
5. Export player ratings and club squad groupings
6. Later add lineup / unit / mismatch engines for fixture pages
