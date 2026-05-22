from __future__ import annotations

from .paths import ensure_dirs
from .normalize_fixtures_master import build_stub as fixtures_master
from .normalize_match_team_stats import build_stub as match_team_stats
from .normalize_match_events import build_stub as match_events
from .normalize_match_player_stats import build_stub as match_player_stats
from .normalize_lineups import build_stub as lineups
from .normalize_injuries import build_stub as injuries
from .normalize_odds_prematch_long import build_stub as odds_prematch_long
from .normalize_odds_live_long import build_stub as odds_live_long
from .build_team_rolling_features import build_stub as api_team_rolling_features
from .build_player_rolling_features import build_stub as api_player_rolling_features
from .build_lineup_features import build_stub as api_lineup_features
from .build_injury_features import build_stub as api_injury_features
from .build_event_features import build_stub as api_event_features
from .build_odds_features import build_stub as api_odds_features
from .build_live_features import build_stub as api_live_features
from .build_enriched_fixture_features import build_stub as api_enriched_fixture_features


def main() -> None:
    ensure_dirs()
    builders = [
        fixtures_master, match_team_stats, match_events, match_player_stats,
        lineups, injuries, odds_prematch_long, odds_live_long,
        api_team_rolling_features, api_player_rolling_features,
        api_lineup_features, api_injury_features, api_event_features,
        api_odds_features, api_live_features, api_enriched_fixture_features,
    ]
    for builder in builders:
        df = builder()
        print(f"STUB READY: {builder.__name__} rows={len(df)}")


if __name__ == '__main__':
    main()
