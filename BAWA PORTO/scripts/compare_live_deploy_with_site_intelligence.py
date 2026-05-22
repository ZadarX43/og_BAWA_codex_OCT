#!/usr/bin/env python3
"""Compare a live deploy folder with publish-safe site intelligence.

Research/reporting only. This does not alter deploy routing.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from build_goal_market_signal_matrix import btts_signal, ftr_signal, ou25_signal, shape_flags, state_from_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_DIR = ROOT / "predictions_output" / "2026-05-13_mls_may14_live_imp20" / "02_deploy"
DEFAULT_API_ROOT = ROOT / "reports" / "latest" / "api_current_context_overlay_window_mls_2026_05_13_to_2026_05_14"
DEFAULT_OUTDIR = ROOT / "reports" / "latest" / "live_model_intelligence_compare"
TEAM_INTEL_ROOT = ROOT / "frontend" / "public" / "data" / "team_intelligence" / "competitions"
SQUAD_INTEL_PATH = ROOT / "frontend" / "public" / "data" / "player_intelligence" / "club_squad_ratings.json"

LEAGUE_TO_COMPETITION = {
    "Australia A-League": "australia_a_league",
    "Austria Bundesliga": "austria_bundesliga",
    "Belgium Pro": "belgium_pro",
    "Brazil Serie A": "brazil_serie_a",
    "Czech First League": "czech_first_league",
    "Denmark Superliga": "denmark_superliga",
    "England Championship": "england_championship",
    "England EFL League 1": "england_efl_league_1",
    "England FA Cup": "england_fa_cup",
    "England Premier League": "england_premier_league",
    "France Ligue 1": "france_ligue_1",
    "Germany Bundesliga": "germany_bundesliga",
    "Germany Bundesliga 2": "germany_bundesliga_2",
    "Italy Serie A": "italy_serie_a",
    "Japan J1": "japan_j1",
    "Netherlands Eredivisie": "netherlands_eredivisie",
    "Norway Eliteserien": "norway_eliteserien",
    "Portugal Liga": "portugal_liga",
    "Saudi Pro League": "saudi_pro_league",
    "Scotland Premiership": "scotland_premiership",
    "South Korea K League": "south_korea_k_league",
    "Spain La Liga": "spain_la_liga",
    "Swiss Super League": "swiss_super_league",
    "Turkey Super Lig": "turkey_super_lig",
    "USA MLS": "usa_mls",
}

ALIASES = {
    "cf montreal": "montreal impact",
    "montreal": "montreal impact",
    "orlando city sc": "orlando city",
    "new york red bulls": "new york rb",
    "san jose earthquakes": "sj earthquakes",
    "sporting kansas city": "sporting kc",
    "los angeles galaxy": "la galaxy",
    "minnesota united fc": "minnesota united",
}

METRIC_SHORTS = {
    "goal_heat_rating": "goal_heat",
    "btts_pressure_rating": "btts_pressure",
    "attack_flow_rating": "attack_flow",
    "defensive_lock_rating": "defensive_lock",
    "first_strike_rating": "first_strike",
    "chaos_rating": "chaos",
}


def norm_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return ALIASES.get(text, text)


def pair_key(home: Any, away: Any) -> tuple[str, str]:
    return (norm_name(home), norm_name(away))


def competition_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text in LEAGUE_TO_COMPETITION:
        return LEAGUE_TO_COMPETITION[text]
    return norm_name(text).replace(" ", "_")


def load_deploy_rows(deploy_dir: Path) -> pd.DataFrame:
    frames = []
    for tier in ("ELITE", "STANDARD", "OBSERVE"):
        matches = sorted(deploy_dir.glob(f"*DEPLOY_TIER_{tier}*.csv"))
        for path in matches:
            df = pd.read_csv(path)
            df["deploy_file_tier"] = tier
            df["deploy_source_file"] = path.name
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No DEPLOY_TIER csv files found in {deploy_dir}")
    return pd.concat(frames, ignore_index=True)


def load_team_intelligence() -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(TEAM_INTEL_ROOT.glob("**/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        comp = str(payload.get("competition_key") or "")
        for team in payload.get("teams") or []:
            for key in (team.get("team"), team.get("team_slug")):
                norm = norm_name(key)
                if not norm:
                    continue
                lookup[(comp, norm)] = team
                lookup.setdefault(("", norm), team)
    return lookup


def load_squad_intelligence() -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(SQUAD_INTEL_PATH.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for club in payload:
        comp = str(club.get("competition_key") or "")
        players = club.get("players") or []
        summary = {
            "club": club.get("club"),
            "players": players,
            "top_goal": top_players(players, "goal_threat"),
            "top_creative": top_players(players, "creative_spark"),
            "top_defensive": top_players(players, "defensive_lock"),
            "top_booking": top_players(players, "booking_heat"),
        }
        for key in (club.get("club"), club.get("club_slug")):
            norm = norm_name(key)
            if not norm:
                continue
            lookup[(comp, norm)] = summary
            lookup.setdefault(("", norm), summary)
    return lookup


def top_players(players: list[dict[str, Any]], metric: str, limit: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(players, key=lambda p: float((p.get("ratings") or {}).get(metric) or 0), reverse=True)
    return [
        {
            "name": player.get("name"),
            "position": player.get("position"),
            "score": (player.get("ratings") or {}).get(metric),
            "power": (player.get("ratings") or {}).get("og_player_power"),
        }
        for player in ranked[:limit]
    ]


def load_provider_features(api_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    fixture_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    enriched_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for fixtures_path in sorted((api_root / "normalized").glob("fixtures_master__*__*.csv")):
        fixtures = pd.read_csv(fixtures_path)
        for row in fixtures.itertuples(index=False):
            fixture_lookup[pair_key(row.home_team_name, row.away_team_name)] = row._asdict()
    for enriched_path in sorted((api_root / "features").glob("api_enriched_fixture_features__*__*.csv")):
        enriched = pd.read_csv(enriched_path)
        for row in enriched.itertuples(index=False):
            enriched_lookup[pair_key(row.home_team_name, row.away_team_name)] = row._asdict()
    return fixture_lookup, enriched_lookup


def team_metrics(home: dict[str, Any] | None, away: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    home_ratings = (home or {}).get("ratings") or {}
    away_ratings = (away or {}).get("ratings") or {}
    values: dict[str, dict[str, float]] = {}
    for long_key, short in METRIC_SHORTS.items():
        home_value = float(home_ratings.get(long_key) or 50)
        away_value = float(away_ratings.get(long_key) or 50)
        values[short] = {"home": home_value, "away": away_value, "delta": home_value - away_value}
    return values


def support_tokens(row: pd.Series) -> set[str]:
    raw = " ".join(str(row.get(col) or "") for col in ("context_reason_codes", "team_intel_overlay_reason", "deterministic_adjust_reason"))
    return {token.strip().upper() for token in re.split(r"[|,;\s]+", raw) if token.strip()}


def team_goals_signal(metrics: dict[str, dict[str, float]], squad: dict[str, Any], side: str) -> dict[str, Any]:
    own = "home" if side == "home" else "away"
    opp = "away" if side == "home" else "home"
    threat_key = "home_goal_threat_top" if side == "home" else "away_goal_threat_top"
    threat = float(squad.get(threat_key) or 55)
    attack_vs_lock = metrics["attack_flow"][own] - metrics["defensive_lock"][opp]
    score = (
        50.0
        + attack_vs_lock * 0.45
        + (metrics["goal_heat"][own] - 60.0) * 0.25
        + (metrics["first_strike"][own] - 60.0) * 0.20
        + (metrics["chaos"][opp] - 50.0) * 0.15
        + (threat - 65.0) * 0.20
    )
    score = max(0.0, min(100.0, score))
    state = state_from_score(score)
    return {
        "market": f"{'HOME' if side == 'home' else 'AWAY'}_TEAM_GOALS_15",
        "signal_pick": "OVER15" if state == "BOOST" else "UNDER15" if state == "AVOID" else "WATCH",
        "signal_score": round(score, 2),
        "signal_state": state,
    }


def pick_for_market(row: pd.Series) -> tuple[str, str]:
    market = str(row.get("market") or "").upper()
    selection = str(row.get("selection") or row.get("bookie_pick") or "").upper()
    if market == "BTTS":
        return "BTTS", "YES" if "YES" in selection else "NO" if "NO" in selection else selection
    if market == "OU25":
        return "OU25", "OVER25" if "OVER" in selection else "UNDER25" if "UNDER" in selection else selection
    if market == "FTR":
        return "FTR", selection
    if market in {"TG25", "TEAM_GOALS"}:
        return market, selection
    return market, selection


def signal_alignment(model_market: str, model_pick: str, signals: dict[str, dict[str, Any]]) -> tuple[str, str, float, str]:
    signal = signals.get(model_market)
    if not signal:
        return "not_scored", "", 0.0, ""
    signal_pick = str(signal.get("signal_pick") or "").upper()
    state = str(signal.get("signal_state") or "").upper()
    score = float(signal.get("signal_score") or 0)
    if signal_pick in {"WATCH", "NO_EDGE"}:
        return "review", signal_pick, score, state
    if signal_pick == model_pick:
        return "supports_model", signal_pick, score, state
    return "conflicts_model", signal_pick, score, state


def short_player_list(players: list[dict[str, Any]]) -> str:
    return "; ".join(f"{p.get('name')} {p.get('score')}" for p in players if p.get("name"))


def build_report(deploy_rows: pd.DataFrame, api_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    team_lookup = load_team_intelligence()
    squad_lookup = load_squad_intelligence()
    fixture_lookup, enriched_lookup = load_provider_features(api_root)

    rows: list[dict[str, Any]] = []
    fixture_cards: dict[str, dict[str, Any]] = {}
    for _, row in deploy_rows.iterrows():
        home_name = row.get("home_team_name")
        away_name = row.get("away_team_name")
        key = pair_key(home_name, away_name)
        comp = competition_key(row.get("league"))
        home_team = team_lookup.get((comp, key[0])) or team_lookup.get(("", key[0]))
        away_team = team_lookup.get((comp, key[1])) or team_lookup.get(("", key[1]))
        home_squad = squad_lookup.get((comp, key[0])) or squad_lookup.get(("", key[0])) or {}
        away_squad = squad_lookup.get((comp, key[1])) or squad_lookup.get(("", key[1])) or {}
        provider = fixture_lookup.get(key, {})
        enriched = enriched_lookup.get(key, {})
        metrics = team_metrics(home_team, away_team)
        squad_summary = {
            "home_goal_threat_top": max([float(p.get("score") or 0) for p in home_squad.get("top_goal", [])] or [55]),
            "away_goal_threat_top": max([float(p.get("score") or 0) for p in away_squad.get("top_goal", [])] or [55]),
        }
        tokens = support_tokens(row)
        signals = {
            "FTR": ftr_signal(metrics),
            "OU25": ou25_signal(metrics, tokens),
            "BTTS": btts_signal(metrics, tokens),
            "HOME_TEAM_GOALS_15": team_goals_signal(metrics, squad_summary, "home"),
            "AWAY_TEAM_GOALS_15": team_goals_signal(metrics, squad_summary, "away"),
        }
        model_market, model_pick = pick_for_market(row)
        alignment, signal_pick, signal_score, signal_state = signal_alignment(model_market, model_pick, signals)
        shape = ";".join(shape_flags(metrics))
        base = {
            "fixture_key": row.get("fixture_key"),
            "match_date": row.get("match_date"),
            "league": row.get("league"),
            "competition_key": comp,
            "home_team": home_name,
            "away_team": away_name,
            "deploy_tier": row.get("deploy_tier") or row.get("deploy_file_tier"),
            "market": model_market,
            "model_pick": model_pick,
            "model_prob": row.get("model_p_for_bookie"),
            "model_prob_xgb": row.get("model_p_for_bookie_xgb") or row.get("model_p_for_bookie_xgb_btts") or row.get("model_p_for_bookie_xgb_ou25"),
            "bookie_implied_novig": row.get("bookie_implied_novig"),
            "value_edge": row.get("value_edge"),
            "team_intel_overlay_action": row.get("team_intel_overlay_action"),
            "team_intel_overlay_fit_score": row.get("team_intel_overlay_market_fit_score"),
            "team_intel_overlay_reason": row.get("team_intel_overlay_reason"),
            "site_signal_alignment": alignment,
            "site_signal_pick": signal_pick,
            "site_signal_score": signal_score,
            "site_signal_state": signal_state,
            "shape_flags": shape,
            "provider_fixture_id": provider.get("fixture_id"),
            "provider_kickoff_ts_utc": provider.get("kickoff_ts_utc"),
            "referee_name": provider.get("referee_name"),
            "home_injuries": enriched.get("home_injured_players_count"),
            "away_injuries": enriched.get("away_injured_players_count"),
            "home_absence_severity": enriched.get("home_absence_severity_score"),
            "away_absence_severity": enriched.get("away_absence_severity_score"),
            "bookie_over25_prob_norm": enriched.get("bookie_over25_prob_norm"),
            "bookie_btts_yes_prob_norm": enriched.get("bookie_btts_yes_prob_norm"),
            "home_goal_drivers": short_player_list(home_squad.get("top_goal", [])),
            "away_goal_drivers": short_player_list(away_squad.get("top_goal", [])),
            "home_creative_drivers": short_player_list(home_squad.get("top_creative", [])),
            "away_creative_drivers": short_player_list(away_squad.get("top_creative", [])),
        }
        for short, values in metrics.items():
            base[f"home_{short}"] = values["home"]
            base[f"away_{short}"] = values["away"]
            base[f"{short}_delta"] = values["delta"]
        for market_key, signal in signals.items():
            base[f"{market_key.lower()}_signal_pick"] = signal.get("signal_pick")
            base[f"{market_key.lower()}_signal_state"] = signal.get("signal_state")
            base[f"{market_key.lower()}_signal_score"] = signal.get("signal_score")
        rows.append(base)

        fixture_key = str(row.get("fixture_key") or "")
        if fixture_key not in fixture_cards:
            fixture_cards[fixture_key] = {
                **{k: base[k] for k in ("fixture_key", "match_date", "league", "competition_key", "home_team", "away_team", "provider_fixture_id", "provider_kickoff_ts_utc", "referee_name", "home_injuries", "away_injuries", "home_goal_drivers", "away_goal_drivers", "shape_flags")},
                "team_signal_ftr": signals["FTR"]["signal_pick"],
                "team_signal_ftr_score": signals["FTR"]["signal_score"],
                "team_signal_ou25": signals["OU25"]["signal_pick"],
                "team_signal_ou25_score": signals["OU25"]["signal_score"],
                "team_signal_btts": signals["BTTS"]["signal_pick"],
                "team_signal_btts_score": signals["BTTS"]["signal_score"],
                "home_team_goals_15": signals["HOME_TEAM_GOALS_15"]["signal_pick"],
                "home_team_goals_15_score": signals["HOME_TEAM_GOALS_15"]["signal_score"],
                "away_team_goals_15": signals["AWAY_TEAM_GOALS_15"]["signal_pick"],
                "away_team_goals_15_score": signals["AWAY_TEAM_GOALS_15"]["signal_score"],
            }

    scored = pd.DataFrame(rows)
    cards = pd.DataFrame(fixture_cards.values())
    summary = (
        scored.groupby(["deploy_tier", "market", "site_signal_alignment"], dropna=False)
        .agg(rows=("fixture_key", "count"), avg_signal_score=("site_signal_score", "mean"), avg_model_prob=("model_prob", "mean"))
        .reset_index()
        .assign(avg_signal_score=lambda df: df["avg_signal_score"].round(1), avg_model_prob=lambda df: df["avg_model_prob"].round(3))
        .sort_values(["deploy_tier", "market", "site_signal_alignment"])
    )
    return scored, cards, summary


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, [c for c in columns if c in df.columns]].copy()
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join("---" for _ in view.columns) + " |"]
    for item in view.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in item) + " |")
    return "\n".join(lines)


def write_summary(scored: pd.DataFrame, cards: pd.DataFrame, alignment_summary: pd.DataFrame, api_root: Path) -> str:
    conflicts = scored[scored["site_signal_alignment"].eq("conflicts_model")].copy()
    supports = scored[scored["site_signal_alignment"].eq("supports_model")].copy()
    high_review = conflicts[conflicts["deploy_tier"].isin(["ELITE", "STANDARD"])].copy()
    api_label = str(api_root.resolve().relative_to(ROOT)) if api_root.resolve().is_relative_to(ROOT) else str(api_root)
    return "\n".join(
        [
            "# Live Model vs Site Intelligence",
            "",
            "Research-only comparison for the live deploy board. This does not alter deploy routing.",
            "",
            "## Coverage",
            "",
            f"- Deploy rows compared: {len(scored)}",
            f"- Unique fixtures: {cards['fixture_key'].nunique() if not cards.empty else 0}",
            f"- Fresh API root: `{api_label}`",
            "- Provider context is included when a matching API overlay exists.",
            "- Missing provider fields mean the row is using team/squad intelligence and model-board context only.",
            "",
            "## Alignment Summary",
            "",
            markdown_table(alignment_summary, ["deploy_tier", "market", "site_signal_alignment", "rows", "avg_signal_score", "avg_model_prob"]),
            "",
            "## High-Review Rows",
            "",
            markdown_table(
                high_review,
                [
                    "deploy_tier",
                    "market",
                    "home_team",
                    "away_team",
                    "model_pick",
                    "site_signal_pick",
                    "site_signal_score",
                    "shape_flags",
                    "team_intel_overlay_action",
                    "team_intel_overlay_reason",
                ],
            ),
            "",
            "## Supported Rows",
            "",
            markdown_table(
                supports.head(20),
                ["deploy_tier", "market", "home_team", "away_team", "model_pick", "site_signal_pick", "site_signal_score", "shape_flags"],
            ),
            "",
            "## Fixture Intelligence Cards",
            "",
            markdown_table(
                cards,
                [
                    "home_team",
                    "away_team",
                    "team_signal_ftr",
                    "team_signal_ou25",
                    "team_signal_btts",
                    "home_team_goals_15",
                    "away_team_goals_15",
                    "home_injuries",
                    "away_injuries",
                    "shape_flags",
                ],
            ),
            "",
            "## Read",
            "",
            "- Treat `supports_model` as a confidence enhancer only.",
            "- Treat `conflicts_model` as the review queue before slips/accas.",
            "- Any confirmed-lineup or player-event upgrade needs a second refresh closer to kickoff.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-dir", type=Path, default=DEFAULT_DEPLOY_DIR)
    parser.add_argument("--api-root", type=Path, default=DEFAULT_API_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deploy_rows = load_deploy_rows(args.deploy_dir)
    scored, cards, alignment_summary = build_report(deploy_rows, args.api_root)
    args.outdir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.outdir / "live_model_intelligence_rows.csv", index=False)
    cards.to_csv(args.outdir / "live_fixture_intelligence_cards.csv", index=False)
    alignment_summary.to_csv(args.outdir / "live_alignment_summary.csv", index=False)
    outdir_label = str(args.outdir.resolve().relative_to(ROOT)) if args.outdir.resolve().is_relative_to(ROOT) else str(args.outdir)
    summary = {
        "deploy_rows": int(len(scored)),
        "unique_fixtures": int(scored["fixture_key"].nunique()),
        "alignment_counts": dict(Counter(scored["site_signal_alignment"])),
        "outputs": outdir_label,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.outdir / "SUMMARY.md").write_text(write_summary(scored, cards, alignment_summary, args.api_root), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
