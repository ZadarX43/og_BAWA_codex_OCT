#!/usr/bin/env python3
"""Build the recurring all-league injury shock coverage report.

This is a reporting and audit adapter. It does not mutate prediction outputs,
deploy gates, or `deploy_rulebook.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from injury_shock_engine import (  # noqa: E402
    build_context_index,
    build_injury_shock_board,
    clamp,
    context_volatility_score,
    dedupe_absence_sources,
    player_recent_metrics,
    read_optional_csv,
    safe_float,
    score_absence,
)


DEFAULT_OUTDIR = ROOT / "reports/latest/injury_shock_coverage_scan"
DEFAULT_ADMIN_JSON = ROOT / "frontend/public/data/internal/injury_shock_admin_dashboard.json"
DEFAULT_DOC = ROOT / "docs/INJURY_SHOCK_COVERAGE_SCAN_RUNBOOK.md"
DEFAULT_CONTEXT = ROOT / "docs/INJURY_SHOCK_CONTEXT_FLAGS_TEMPLATE.csv"
DEFAULT_RESULTS_FEED = ROOT / "frontend/public/data/live_results_feed.json"
DEFAULT_PLAYER_RATINGS = ROOT / "frontend/public/data/player_intelligence/player_ratings.csv"
COMPETITION_KEY_ALIASES = {
    "Australia_A_League": {"australia_a_league"},
    "Austria_Bundesliga": {"austria_bundesliga"},
    "Belgium_Pro": {"belgium_pro"},
    "Brazil_Serie_A": {"brazil_serie_a"},
    "Denmark_Superliga": {"denmark_superliga"},
    "England_Premier_League": {"england_premier_league", "premier_league"},
    "England_Championship": {"england_championship", "championship"},
    "England_EFL_League_1": {"england_efl_league_1", "league_one"},
    "France_Ligue_1": {"france_ligue_1", "ligue_1"},
    "Germany_Bundesliga": {"germany_bundesliga", "bundesliga"},
    "Germany_Bundesliga_2": {"germany_bundesliga_2", "bundesliga_2"},
    "Italy_Serie_A": {"italy_serie_a", "serie_a"},
    "Netherlands_Eredivisie": {"netherlands_eredivisie", "eredivisie"},
    "Norway_Eliteserien": {"norway_eliteserien"},
    "Portugal_Liga": {"portugal_liga", "primeira_liga"},
    "Saudi_Pro_League": {"saudi_pro_league"},
    "Scotland_Premiership": {"scotland_premiership"},
    "South_Korea_K_League": {"south_korea_k_league"},
    "Spain_La_Liga": {"spain_la_liga", "la_liga"},
    "Switzerland_Super_League": {"swiss_super_league", "switzerland_super_league"},
    "Turkey_Super_Lig": {"turkey_super_lig", "super_lig"},
    "USA_MLS": {"usa_mls", "mls"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_team(value: Any) -> str:
    text = norm_text(value)
    text = re.sub(r"\bsj\b", "san jose", text)
    text = re.sub(r"\bnyc\b", "new york city", text)
    text = re.sub(r"\bny\b", "new york", text)
    replacements = {
        "sj": "san jose",
        "ny": "new york",
        "nyc": "new york city",
        "monchengladbach": "borussia monchengladbach",
        "m gladbach": "borussia monchengladbach",
        "st louis": "saint louis",
        "st louis city": "saint louis city",
        "salt lake": "real salt lake",
        "ny red bulls": "new york red bulls",
    }
    text = replacements.get(text, text)
    parts = [part for part in text.split() if part not in {"fc", "cf", "sc", "afc", "club"}]
    return " ".join(parts)


def fixture_join_key(match_date: Any, home: Any, away: Any) -> str:
    date_text = str(match_date or "").strip()[:10].replace("-", "_")
    return f"{date_text}_{canonical_team(home)}_{canonical_team(away)}"


def name_tokens(value: Any) -> list[str]:
    return [token for token in norm_text(value).split() if token]


def initial_surname_key(value: Any) -> tuple[str, str]:
    tokens = name_tokens(value)
    if len(tokens) < 2:
        return "", ""
    initials = "".join(token[:1] for token in tokens[:-1] if token)
    surname = tokens[-1]
    return initials, surname


def surname_key(value: Any) -> str:
    tokens = name_tokens(value)
    return tokens[-1] if tokens else ""


def unique_rating_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    signatures = {
        (
            norm_text(row.get("name")),
            canonical_team(row.get("club")),
            norm_text(row.get("competition_key")),
            str(row.get("season") or ""),
        )
        for row in rows
    }
    if len(signatures) == 1:
        return rows[0]
    return rows[0] if len(rows) == 1 else {}


def parse_tag(path: Path, prefix: str) -> tuple[str, int] | None:
    match = re.match(rf"{re.escape(prefix)}__(.+)__(\d{{4}})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def file_age_hours(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return round((datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600.0, 2)


def freshness_label(age_hours: float | None) -> str:
    if age_hours is None:
        return "MISSING"
    if age_hours <= 72:
        return "GREEN"
    if age_hours <= 168:
        return "AMBER"
    return "STALE"


def csv_row_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
    except OSError:
        return 0


def latest_published_ts(injuries: pd.DataFrame) -> str:
    if injuries.empty or "published_ts_utc" not in injuries.columns:
        return ""
    ts = pd.to_datetime(injuries["published_ts_utc"], errors="coerce", utc=True).dropna()
    if ts.empty:
        return ""
    return ts.max().replace(microsecond=0).isoformat()


def source_freshness_status(files: list[Path | None], latest_injury_ts: str, injuries_row_count: int = 0) -> str:
    labels = [freshness_label(file_age_hours(path)) for path in files]
    if "MISSING" in labels:
        return "RED_MISSING_SOURCE"
    if "STALE" in labels:
        return "RED_STALE_SOURCE"
    if "AMBER" in labels:
        return "AMBER_SOURCE_AGE"
    required_counts = [csv_row_count(path) for path in files if path is not None]
    if required_counts and min(required_counts) == 0:
        return "RED_EMPTY_TRUSTED_SOURCE"
    if latest_injury_ts:
        return "GREEN"
    if injuries_row_count == 0:
        return "RED_EMPTY_TRUSTED_SOURCE"
    return "AMBER_NO_PROVIDER_TIMESTAMP"


def discover_fixture_sets(normalized_roots: list[Path]) -> list[dict[str, Any]]:
    assets: dict[tuple[str, int], dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for root in normalized_roots:
        if not root.exists():
            continue
        for prefix, field in [
            ("fixtures_master", "fixtures_csv"),
            ("injuries", "injuries_csv"),
            ("sidelined", "sidelined_csv"),
            ("match_player_stats", "player_stats_csv"),
            ("lineups", "lineups_csv"),
        ]:
            for path in root.glob(f"{prefix}__*.csv"):
                parsed = parse_tag(path, prefix)
                if parsed is not None:
                    assets[parsed][field].append(path)
    latest_by_league: dict[str, dict[str, Any]] = {}
    for (league_tag, season), grouped in assets.items():
        if not grouped.get("fixtures_csv"):
            continue
        fixtures_csv = choose_source(grouped.get("fixtures_csv", []), prefer_rows=True)
        injuries_csv = choose_source(grouped.get("injuries_csv", []), prefer_rows=True)
        player_stats_csv = choose_source(grouped.get("match_player_stats_csv", []) or grouped.get("player_stats_csv", []), prefer_rows=True)
        lineups_csv = choose_source(grouped.get("lineups_csv", []), prefer_rows=True)
        sidelined_csv = choose_source(grouped.get("sidelined_csv", []), prefer_rows=True)
        item = {
            "league_tag": league_tag,
            "season": season,
            "root": fixtures_csv.parent if fixtures_csv else grouped["fixtures_csv"][0].parent,
            "fixtures_csv": fixtures_csv,
            "injuries_csv": injuries_csv,
            "player_stats_csv": player_stats_csv,
            "lineups_csv": lineups_csv,
            "sidelined_csv": sidelined_csv,
            "source_candidates": {key: [str(path) for path in paths] for key, paths in grouped.items()},
        }
        current = latest_by_league.get(item["league_tag"])
        if current is None or int(item["season"]) > int(current["season"]):
            latest_by_league[item["league_tag"]] = item
        elif current is not None and int(item["season"]) == int(current["season"]):
            if item["fixtures_csv"] and current["fixtures_csv"] and item["fixtures_csv"].stat().st_mtime > current["fixtures_csv"].stat().st_mtime:
                latest_by_league[item["league_tag"]] = item
    return sorted(latest_by_league.values(), key=lambda item: item["league_tag"])


def choose_source(paths: list[Path], prefer_rows: bool = False) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    trusted = [path for path in existing if valid_report_normalized_root(path.parent)]
    candidates = trusted or existing
    if prefer_rows:
        with_rows = [(csv_row_count(path), path.stat().st_mtime, path) for path in candidates]
        with_rows.sort(key=lambda item: (item[0] > 0, item[1]), reverse=True)
        if with_rows[0][0] > 0:
            return with_rows[0][2]
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_player_ratings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "by_name_club": {},
            "by_name_comp": {},
            "by_unique_name": {},
            "by_initial_surname_club": {},
            "by_initial_surname_comp": {},
            "by_surname_club": {},
        }
    ratings = pd.read_csv(path, low_memory=False)
    by_name_club: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_name_comp: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_initial_surname_club: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_initial_surname_comp: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_surname_club: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for _, row in ratings.iterrows():
        rec = row.to_dict()
        name = norm_text(row.get("name"))
        club = canonical_team(row.get("club"))
        comp = norm_text(row.get("competition_key"))
        initials, surname = initial_surname_key(row.get("name"))
        if name and club:
            by_name_club[(name, club)].append(rec)
        if name and comp:
            by_name_comp[(name, comp)].append(rec)
        if name:
            by_name[name].append(rec)
        if initials and surname and club:
            by_initial_surname_club[(initials, surname, club)].append(rec)
        if initials and surname and comp:
            by_initial_surname_comp[(initials, surname, comp)].append(rec)
        if surname and club:
            by_surname_club[(surname, club)].append(rec)
    unique = {name: rows[0] for name, rows in by_name.items() if len({canonical_team(row.get("club")) for row in rows}) == 1}
    return {
        "by_name_club": by_name_club,
        "by_name_comp": by_name_comp,
        "by_unique_name": unique,
        "by_initial_surname_club": by_initial_surname_club,
        "by_initial_surname_comp": by_initial_surname_comp,
        "by_surname_club": by_surname_club,
    }


def lookup_player_rating(player_ratings: dict[str, Any], player_name: Any, team_name: Any, league_tag: str | None = None) -> tuple[dict[str, Any], str]:
    name = norm_text(player_name)
    club = canonical_team(team_name)
    if not name:
        return {}, "MISSING_PLAYER_RATING"
    rating = unique_rating_row(player_ratings.get("by_name_club", {}).get((name, club), []))
    if rating:
        return rating, "OK_PLAYER_RATING_NAME_CLUB"
    for comp in COMPETITION_KEY_ALIASES.get(str(league_tag or ""), {norm_text(league_tag)}):
        rows = player_ratings.get("by_name_comp", {}).get((name, norm_text(comp)), [])
        if len(rows) == 1:
            return rows[0], "OK_PLAYER_RATING_NAME_COMP"
    rating = player_ratings.get("by_unique_name", {}).get(name)
    if rating:
        return rating, "OK_PLAYER_RATING_UNIQUE_NAME"
    initials, surname = initial_surname_key(player_name)
    if initials and surname:
        rating = unique_rating_row(player_ratings.get("by_initial_surname_club", {}).get((initials, surname, club), []))
        if rating:
            return rating, "OK_PLAYER_RATING_INITIAL_SURNAME_CLUB"
    surname = surname_key(player_name)
    tokens = name_tokens(player_name)
    surname_only_safe = len(tokens) == 1 or (len(tokens) == 2 and len(tokens[0]) == 1)
    if surname and club and surname_only_safe:
        rating = unique_rating_row(player_ratings.get("by_surname_club", {}).get((surname, club), []))
        if rating:
            return rating, "OK_PLAYER_RATING_SURNAME_CLUB_UNIQUE"
    return {}, "MISSING_PLAYER_RATING"


def rating_role_from_position_group(rating: dict[str, Any]) -> str:
    position = norm_text(rating.get("position_group"))
    if position in {"forward", "attacker", "striker", "winger"}:
        return "attacker"
    if position in {"midfield", "midfielder"}:
        return "midfielder"
    if position in {"defence", "defense", "defender", "fullback", "centre back", "center back"}:
        return "defender"
    if position in {"goalkeeper", "keeper"}:
        return "goalkeeper"
    return "unknown"


def rating_absence_scores(rating: dict[str, Any], state: str) -> dict[str, Any]:
    if not rating:
        return {}
    role = rating_role_from_position_group(rating)
    power = clamp(safe_float(rating.get("og_player_power"), 55.0), 0.0, 100.0) / 100.0
    goal = clamp(safe_float(rating.get("goal_threat")), 0.0, 100.0) / 100.0
    shot = clamp(safe_float(rating.get("shot_threat")), 0.0, 100.0) / 100.0
    xg = clamp(safe_float(rating.get("xg_threat")), 0.0, 100.0) / 100.0
    creative = clamp(safe_float(rating.get("creative_spark")), 0.0, 100.0) / 100.0
    midfield = clamp(safe_float(rating.get("midfield_engine")), 0.0, 100.0) / 100.0
    defence = clamp(safe_float(rating.get("defensive_lock")), 0.0, 100.0) / 100.0
    keeper = clamp(safe_float(rating.get("goalkeeper_shield")), 0.0, 100.0) / 100.0
    state_mult = 0.72 if state == "doubt_or_late_test" else 0.55 if state == "reported" else 1.0
    impact = max(0.45, power) * state_mult
    if role == "attacker":
        attack = 14.0 * impact + 10.0 * max(goal, shot, xg) * state_mult + 6.0 * creative * state_mult
        function = "primary_goal_threat" if max(goal, shot, xg) >= 0.60 else "creator_or_wide_forward" if creative >= 0.60 else "attacking_depth"
        return {"role": role, "function": function, "attack_score": attack, "midfield_score": 0.0, "defence_score": 0.0, "keeper_score": 0.0}
    if role == "midfielder":
        mid = 13.0 * impact + 10.0 * midfield * state_mult
        attack = 7.0 * creative * state_mult
        function = "midfield_ball_winner" if midfield >= 0.62 else "central_creator" if creative >= 0.62 else "midfield_rotation"
        return {"role": role, "function": function, "attack_score": attack, "midfield_score": mid, "defence_score": 0.0, "keeper_score": 0.0}
    if role == "defender":
        def_score = 13.0 * impact + 10.0 * defence * state_mult
        function = "defensive_duel_anchor" if defence >= 0.60 else "defensive_depth"
        return {"role": role, "function": function, "attack_score": 0.0, "midfield_score": 0.0, "defence_score": def_score, "keeper_score": 0.0}
    if role == "goalkeeper":
        keeper_score = 18.0 * impact + 14.0 * keeper * state_mult
        return {"role": role, "function": "goalkeeper", "attack_score": 0.0, "midfield_score": 0.0, "defence_score": 8.0 * impact, "keeper_score": keeper_score}
    return {}


def apply_player_impact_overlay(board: pd.DataFrame, player_rows: pd.DataFrame) -> pd.DataFrame:
    if board.empty or player_rows.empty or "fixture_id" not in player_rows.columns:
        return board
    out = board.copy()
    grouped = (
        player_rows.groupby(["fixture_id", "team_side"], dropna=False)
        .agg(
            impact_attack=("attack_score", "sum"),
            impact_midfield=("midfield_score", "sum"),
            impact_defence=("defence_score", "sum"),
            impact_keeper=("keeper_score", "sum"),
            impact_mobility=("mobility_score", "sum"),
            impact_importance=("importance_weight_used", "sum"),
            impact_rows=("player_name", "count"),
            impact_reasons=(
                "player_name",
                lambda s: "; ".join(
                    player_rows.loc[s.index, ["player_name", "reason"]]
                    .astype(str)
                    .apply(lambda row: f"{row['player_name']} ({row['reason']})", axis=1)
                    .head(8)
                    .tolist()
                ),
            ),
        )
        .reset_index()
    )
    for side in ["home", "away"]:
        side_rows = grouped[grouped["team_side"].astype(str).eq(side)].copy()
        if side_rows.empty:
            continue
        side_rows = side_rows.rename(
            columns={
                "impact_attack": f"{side}_player_impact_attack_score",
                "impact_midfield": f"{side}_player_impact_midfield_score",
                "impact_defence": f"{side}_player_impact_defence_score",
                "impact_keeper": f"{side}_player_impact_keeper_score",
                "impact_mobility": f"{side}_player_impact_mobility_score",
                "impact_importance": f"{side}_player_impact_importance_sum",
                "impact_rows": f"{side}_player_impact_rows",
                "impact_reasons": f"{side}_player_impact_reasons",
            }
        )
        out = out.merge(side_rows.drop(columns=["team_side"]), on="fixture_id", how="left")
        for base, impact_col in [
            (f"{side}_attack_absence_score", f"{side}_player_impact_attack_score"),
            (f"{side}_midfield_absence_score", f"{side}_player_impact_midfield_score"),
            (f"{side}_defence_absence_score", f"{side}_player_impact_defence_score"),
            (f"{side}_keeper_absence_score", f"{side}_player_impact_keeper_score"),
            (f"{side}_mobility_risk_score", f"{side}_player_impact_mobility_score"),
        ]:
            if base in out.columns and impact_col in out.columns:
                out[base] = pd.concat(
                    [pd.to_numeric(out[base], errors="coerce").fillna(0.0), pd.to_numeric(out[impact_col], errors="coerce").fillna(0.0)],
                    axis=1,
                ).max(axis=1)
        severity_col = f"{side}_injury_news_severity"
        importance_col = f"{side}_player_impact_importance_sum"
        rows_col = f"{side}_player_impact_rows"
        if severity_col in out.columns and importance_col in out.columns:
            row_counts = pd.to_numeric(out[rows_col], errors="coerce").fillna(0.0) if rows_col in out.columns else pd.Series(0.0, index=out.index)
            out[severity_col] = pd.concat(
                [
                    pd.to_numeric(out[severity_col], errors="coerce").fillna(0.0),
                    (pd.to_numeric(out[importance_col], errors="coerce").fillna(0.0) * 10.0)
                    + (row_counts * 2.0),
                ],
                axis=1,
            ).max(axis=1)
    max_attack = out[[col for col in ["home_attack_absence_score", "away_attack_absence_score"] if col in out.columns]].max(axis=1)
    max_severity = out[[col for col in ["home_injury_news_severity", "away_injury_news_severity"] if col in out.columns]].max(axis=1)
    max_mobility = out[[col for col in ["home_mobility_risk_score", "away_mobility_risk_score"] if col in out.columns]].max(axis=1)
    overlay_warning = max_attack.ge(16.0) | max_severity.ge(18.0) | max_mobility.ge(14.0)
    out["injury_presence_review_flag"] = overlay_warning.astype(int)
    if "deploy_warning_flag" in out.columns:
        out["deploy_warning_flag"] = (
            pd.to_numeric(out["deploy_warning_flag"], errors="coerce").fillna(0).astype(int).eq(1) | overlay_warning
        ).astype(int)
    out["warning_tokens"] = out["warning_tokens"].fillna("").astype(str) if "warning_tokens" in out.columns else ""
    out.loc[overlay_warning & ~out["warning_tokens"].str.contains("PLAYER_IMPACT_REVIEW", na=False), "warning_tokens"] = (
        out.loc[overlay_warning & ~out["warning_tokens"].str.contains("PLAYER_IMPACT_REVIEW", na=False), "warning_tokens"].str.strip("|")
        + "|PLAYER_IMPACT_REVIEW"
    ).str.strip("|")
    return out


def result_items(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for window in payload.get("windows", []) or []:
        for item in window.get("items", []) or []:
            rows.append(item)
    return pd.DataFrame(rows)


def expected_lineup_map(lineups: pd.DataFrame, fixture_id: int, team_id: int) -> set[int]:
    if lineups.empty:
        return set()
    sub = lineups[(pd.to_numeric(lineups.get("fixture_id"), errors="coerce") == fixture_id) & (pd.to_numeric(lineups.get("team_id"), errors="coerce") == team_id)]
    if sub.empty:
        return set()
    starters = sub[pd.to_numeric(sub.get("is_starting_xi"), errors="coerce").fillna(0).eq(1)]
    return {int(safe_float(value, -1)) for value in starters.get("player_id", []) if safe_float(value, -1) >= 0}


def build_sidelined_summary(sidelined: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if sidelined.empty or "player_id" not in sidelined.columns:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for player_id, group in sidelined.groupby(pd.to_numeric(sidelined["player_id"], errors="coerce"), dropna=True):
        if pd.isna(player_id):
            continue
        open_flag = pd.to_numeric(group.get("is_open_absence", 0), errors="coerce").fillna(0).astype(int)
        out[int(player_id)] = {
            "sidelined_history_count": int(len(group)),
            "sidelined_open_absence_flag": int(open_flag.max() if not open_flag.empty else 0),
            "sidelined_reason_history": "|".join(
                group.get("reason", pd.Series(dtype=str)).dropna().astype(str).replace("", pd.NA).dropna().head(5).tolist()
            ),
        }
    return out


def build_player_impact_rows(
    fixtures_csv: Path,
    injuries_csv: Path | None,
    sidelined_csv: Path | None,
    player_stats_csv: Path,
    lineups_csv: Path | None,
    player_ratings: dict[str, Any],
    context_csv: Path | None,
    league_tag: str | None = None,
) -> pd.DataFrame:
    if not fixtures_csv.exists() or not player_stats_csv.exists():
        return pd.DataFrame()
    fixtures = pd.read_csv(fixtures_csv, low_memory=False)
    injuries = read_optional_csv(str(injuries_csv))
    sidelined = read_optional_csv(str(sidelined_csv))
    player_stats = pd.read_csv(player_stats_csv, low_memory=False)
    lineups = read_optional_csv(str(lineups_csv))
    sidelined_by_player = build_sidelined_summary(sidelined)
    context_by_fixture = build_context_index(str(context_csv) if context_csv and context_csv.exists() else None)
    if injuries.empty:
        return pd.DataFrame()
    injuries = dedupe_absence_sources(injuries)

    fixtures["kickoff_ts_utc"] = pd.to_datetime(fixtures.get("kickoff_ts_utc"), errors="coerce", utc=True)
    player_stats = player_stats.merge(fixtures[["fixture_id", "kickoff_ts_utc"]], on="fixture_id", how="left")
    player_stats = player_stats.sort_values(["kickoff_ts_utc", "fixture_id", "player_id"]).reset_index(drop=True)

    stats_by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for _, row in player_stats.iterrows():
        stats_by_fixture[int(row["fixture_id"])].append(row.to_dict())

    injuries_by_fixture = {int(fid): df.copy() for fid, df in injuries.groupby("fixture_id")} if not injuries.empty else {}
    history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    last_expected_starters: dict[int, set[int]] = defaultdict(set)
    rows: list[dict[str, Any]] = []

    for _, fixture in fixtures.sort_values(["kickoff_ts_utc", "fixture_id"]).iterrows():
        fixture_id = int(fixture["fixture_id"])
        fx_inj = injuries_by_fixture.get(fixture_id, pd.DataFrame(columns=injuries.columns))
        kickoff = pd.to_datetime(fixture.get("kickoff_ts_utc"), errors="coerce", utc=True)
        context_score, context_flags = context_volatility_score(context_by_fixture.get(str(fixture.get("fixture_key")), {}))

        for _, injury in fx_inj.iterrows():
            player_id = int(safe_float(injury.get("player_id"), -1))
            team_id = int(safe_float(injury.get("team_id"), -1))
            side = "home" if team_id == int(fixture.get("home_team_id")) else "away" if team_id == int(fixture.get("away_team_id")) else "unknown"
            team_name = fixture.get("home_team_name") if side == "home" else fixture.get("away_team_name") if side == "away" else ""
            metrics = player_recent_metrics(history.get(player_id, []))
            scored = score_absence(injury, metrics)
            rating, rating_join_status = lookup_player_rating(player_ratings, injury.get("player_name"), team_name, league_tag)
            rating_scored = rating_absence_scores(rating, str(scored.get("state") or "reported"))
            if rating_scored and str(scored.get("function")) == "unknown_function":
                scored.update(rating_scored)
            elif rating_scored:
                for score_key in ["attack_score", "midfield_score", "defence_score", "keeper_score"]:
                    scored[score_key] = max(safe_float(scored.get(score_key)), safe_float(rating_scored.get(score_key)))
            rating_importance_weight = round(clamp(safe_float(rating.get("og_player_power"), 0.0), 0.0, 100.0) / 100.0, 3) if rating else None
            if safe_float(metrics.get("minutes_l10_total")) > 0:
                importance_join_status = "OK_PLAYER_HISTORY"
                importance_source = "player_history"
            elif rating:
                importance_join_status = "OK_PLAYER_RATING_FALLBACK"
                importance_source = "player_rating"
            else:
                importance_join_status = "MISSING_PLAYER_HISTORY"
                importance_source = "missing"
            first_seen_value = injury.get("availability_first_seen_ts_utc") or injury.get("published_ts_utc")
            published = pd.to_datetime(first_seen_value, errors="coerce", utc=True)
            known_hours = None
            if pd.notna(published) and pd.notna(kickoff):
                known_hours = round((kickoff - published).total_seconds() / 3600.0, 2)
            expected_starters = last_expected_starters.get(team_id, set())
            current_starters = expected_lineup_map(lineups, fixture_id, team_id)
            sidelined_summary = sidelined_by_player.get(player_id, {})
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "fixture_key": fixture.get("fixture_key"),
                    "fixture_join_key": fixture_join_key(fixture.get("match_date"), fixture.get("home_team_name"), fixture.get("away_team_name")),
                    "league": fixture.get("league"),
                    "season": fixture.get("season"),
                    "match_date": fixture.get("match_date"),
                    "kickoff_ts_utc": fixture.get("kickoff_ts_utc"),
                    "team_id": team_id,
                    "team_name": team_name,
                    "team_side": side,
                    "player_id": player_id,
                    "player_name": injury.get("player_name"),
                    "absence_type": injury.get("absence_type"),
                    "reason": injury.get("reason"),
                    "status": injury.get("status"),
                    "published_ts_utc": injury.get("published_ts_utc"),
                    "source_scope": injury.get("source_scope", ""),
                    "availability_first_seen_ts_utc": injury.get("availability_first_seen_ts_utc", ""),
                    "availability_first_seen_source_scope": injury.get("availability_first_seen_source_scope", ""),
                    "fixture_only_late_confirmation_flag": int(safe_float(injury.get("fixture_only_late_confirmation_flag"), 0)),
                    "known_hours_before_kickoff": known_hours,
                    "known_since_midweek_flag": int(known_hours is not None and known_hours >= 48.0),
                    "sidelined_history_count": int(sidelined_summary.get("sidelined_history_count", 0)),
                    "sidelined_open_absence_flag": int(sidelined_summary.get("sidelined_open_absence_flag", 0)),
                    "sidelined_reason_history": sidelined_summary.get("sidelined_reason_history", ""),
                    "inferred_position": scored.get("role"),
                    "structural_function": scored.get("function"),
                    "absence_state": scored.get("state"),
                    "importance_weight": scored.get("impact_weight"),
                    "rating_importance_weight": rating_importance_weight,
                    "importance_weight_used": max(safe_float(scored.get("impact_weight")), safe_float(rating_importance_weight)),
                    "importance_source": importance_source,
                    "minutes_l10_total": round(safe_float(metrics.get("minutes_l10_total")), 2),
                    "starter_rate_l10": round(safe_float(metrics.get("starter_rate_l10")), 3),
                    "goals_per90_l5": round(safe_float(metrics.get("goals_per90_l5")), 3),
                    "assists_per90_l5": round(safe_float(metrics.get("assists_per90_l5")), 3),
                    "shots_per90_l5": round(safe_float(metrics.get("shots_per90_l5")), 3),
                    "tackles_per90_l5": round(safe_float(metrics.get("tackles_per90_l5")), 3),
                    "og_player_power": safe_float(rating.get("og_player_power"), None) if rating else None,
                    "matched_rating_name": rating.get("name", "") if rating else "",
                    "matched_rating_club": rating.get("club", "") if rating else "",
                    "matched_rating_competition_key": rating.get("competition_key", "") if rating else "",
                    "player_rating_join_status": rating_join_status,
                    "player_importance_join_status": importance_join_status,
                    "expected_xi_absence_flag": int(player_id in expected_starters),
                    "current_confirmed_lineup_absence_flag": int(bool(current_starters) and player_id not in current_starters),
                    "attack_score": round(safe_float(scored.get("attack_score")), 2),
                    "midfield_score": round(safe_float(scored.get("midfield_score")), 2),
                    "defence_score": round(safe_float(scored.get("defence_score")), 2),
                    "keeper_score": round(safe_float(scored.get("keeper_score")), 2),
                    "mobility_score": round(safe_float(scored.get("mobility_score")), 2),
                    "context_volatility_score": round(context_score, 2),
                    "context_flags": "|".join(context_flags),
                }
            )

        for rec in stats_by_fixture.get(fixture_id, []):
            history[int(rec["player_id"])].append(rec)

        if not lineups.empty:
            for team_id in [int(fixture.get("home_team_id")), int(fixture.get("away_team_id"))]:
                starters = expected_lineup_map(lineups, fixture_id, team_id)
                if starters:
                    last_expected_starters[team_id] = starters

    return pd.DataFrame(rows)


def load_fixture_board(item: dict[str, Any], outdir: Path, context_csv: Path | None) -> pd.DataFrame:
    temp_csv = outdir / "league_boards" / f"INJURY_SHOCK_BOARD__{item['league_tag']}__{item['season']}.csv"
    board = build_injury_shock_board(
        fixtures_csv=str(item["fixtures_csv"]),
        injuries_csv=str(item["injuries_csv"]),
        player_stats_csv=str(item["player_stats_csv"]),
        context_csv=str(context_csv) if context_csv and context_csv.exists() else None,
        output_csv=str(temp_csv),
    )
    if not board.empty:
        board["fixture_join_key"] = [
            fixture_join_key(row.get("match_date"), row.get("home_team_name"), row.get("away_team_name"))
            for _, row in board.iterrows()
        ]
    return board


def build_backtest_link(fixture_board: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if fixture_board.empty or results.empty or "fixture_key" not in results.columns:
        return pd.DataFrame()
    keep_cols = [
        "source_window",
        "source_type",
        "fixture_key",
        "league",
        "market",
        "pick",
        "tier",
        "publish_class",
        "result_status",
        "score",
        "site_signal_alignment",
        "site_signal_state",
        "home_team",
        "away_team",
        "kickoff_time",
    ]
    keep = results[[col for col in keep_cols if col in results.columns]].copy()
    keep["fixture_join_key"] = [
        fixture_join_key(row.get("kickoff_time"), row.get("home_team"), row.get("away_team"))
        for _, row in keep.iterrows()
    ]
    board = fixture_board.copy()
    if "fixture_join_key" not in board.columns:
        board["fixture_join_key"] = [
            fixture_join_key(row.get("match_date"), row.get("home_team_name"), row.get("away_team_name"))
            for _, row in board.iterrows()
        ]
    joined = keep.merge(
        board[
            [
                "fixture_join_key",
                "fixture_key",
                "deploy_warning_flag",
                "warning_tokens",
                "home_attack_absence_score",
                "away_attack_absence_score",
                "home_defence_absence_score",
                "away_defence_absence_score",
                "motivation_volatility_score",
                "goal_model_adjustment",
                "btts_adjustment",
                "ou25_adjustment",
                "ftr_volatility_adjustment",
            ]
        ].rename(columns={"fixture_key": "injury_shock_fixture_key"}),
        on="fixture_join_key",
        how="left",
    )
    joined["injury_shock_join_status"] = joined["deploy_warning_flag"].notna().map({True: "JOINED", False: "NO_MATCH"})
    settled = joined[joined.get("result_status", "").astype(str).str.lower().isin(["won", "lost", "cashed", "void"])].copy()
    return settled


def build_league_coverage_row(item: dict[str, Any], board: pd.DataFrame, player_rows: pd.DataFrame) -> dict[str, Any]:
    injuries = read_optional_csv(str(item["injuries_csv"]))
    latest_ts = latest_published_ts(injuries)
    injuries_count = csv_row_count(item["injuries_csv"])
    sidelined_count = csv_row_count(item.get("sidelined_csv"))
    source_status = source_freshness_status(
        [item["fixtures_csv"], item["injuries_csv"], item["player_stats_csv"], item["lineups_csv"]],
        latest_ts,
        injuries_count,
    )
    return {
        "league_tag": item["league_tag"],
        "season": item["season"],
        "source_root": str(item["root"]),
        "fixtures_csv": str(item["fixtures_csv"]) if item.get("fixtures_csv") else "",
        "injuries_csv": str(item["injuries_csv"]) if item.get("injuries_csv") else "",
        "sidelined_csv": str(item["sidelined_csv"]) if item.get("sidelined_csv") else "",
        "player_stats_csv": str(item["player_stats_csv"]) if item.get("player_stats_csv") else "",
        "lineups_csv": str(item["lineups_csv"]) if item.get("lineups_csv") else "",
        "fixtures_file_age_hours": file_age_hours(item["fixtures_csv"]),
        "injuries_file_age_hours": file_age_hours(item["injuries_csv"]),
        "sidelined_file_age_hours": file_age_hours(item.get("sidelined_csv")),
        "player_stats_file_age_hours": file_age_hours(item["player_stats_csv"]),
        "lineups_file_age_hours": file_age_hours(item["lineups_csv"]),
        "fixtures_row_count": csv_row_count(item["fixtures_csv"]),
        "injuries_row_count": injuries_count,
        "sidelined_row_count": sidelined_count,
        "player_stats_row_count": csv_row_count(item["player_stats_csv"]),
        "lineups_row_count": csv_row_count(item["lineups_csv"]),
        "latest_injury_published_ts_utc": latest_ts,
        "source_freshness_status": source_status,
        "fixtures_scored": int(len(board)),
        "injury_rows": int(len(player_rows)),
        "warning_fixtures": int(pd.to_numeric(board.get("deploy_warning_flag", 0), errors="coerce").fillna(0).sum()) if not board.empty else 0,
        "expected_xi_absence_rows": int(pd.to_numeric(player_rows.get("expected_xi_absence_flag", 0), errors="coerce").fillna(0).sum()) if not player_rows.empty else 0,
        "missing_player_history_rows": int(player_rows.get("player_importance_join_status", pd.Series(dtype=str)).astype(str).eq("MISSING_PLAYER_HISTORY").sum()) if not player_rows.empty else 0,
        "missing_player_rating_rows": int(player_rows.get("player_rating_join_status", pd.Series(dtype=str)).astype(str).eq("MISSING_PLAYER_RATING").sum()) if not player_rows.empty else 0,
        "fixture_only_late_confirmation_rows": int(pd.to_numeric(player_rows.get("fixture_only_late_confirmation_flag", 0), errors="coerce").fillna(0).sum()) if not player_rows.empty else 0,
        "known_since_midweek_rows": int(pd.to_numeric(player_rows.get("known_since_midweek_flag", 0), errors="coerce").fillna(0).sum()) if not player_rows.empty else 0,
    }


def admin_payload(fixture_board: pd.DataFrame, player_rows: pd.DataFrame, coverage: pd.DataFrame, backtest: pd.DataFrame) -> dict[str, Any]:
    warnings = fixture_board[pd.to_numeric(fixture_board.get("deploy_warning_flag", 0), errors="coerce").fillna(0).eq(1)].copy() if not fixture_board.empty else pd.DataFrame()
    if not warnings.empty:
        warnings["admin_sort"] = warnings[["home_injury_news_severity", "away_injury_news_severity", "motivation_volatility_score", "ftr_volatility_adjustment"]].max(axis=1)
    top_warnings = warnings.sort_values("admin_sort", ascending=False).head(40).drop(columns=["admin_sort"], errors="ignore").to_dict("records") if not warnings.empty else []
    sort_weight = "importance_weight_used" if "importance_weight_used" in player_rows.columns else "importance_weight"
    top_players = player_rows.sort_values(["expected_xi_absence_flag", sort_weight], ascending=[False, False]).head(80).to_dict("records") if not player_rows.empty else []
    return {
        "generated_at": utc_now(),
        "contract_version": 1,
        "research_only": True,
        "summary": {
            "leagues": int(len(coverage)),
            "fixtures_scored": int(len(fixture_board)),
            "warning_fixtures": int(len(warnings)),
            "injury_rows": int(len(player_rows)),
            "expected_xi_absence_rows": int(pd.to_numeric(player_rows.get("expected_xi_absence_flag", 0), errors="coerce").fillna(0).sum()) if not player_rows.empty else 0,
            "settled_backtest_rows": int(len(backtest)),
        },
        "coverage_by_league": coverage.to_dict("records") if not coverage.empty else [],
        "top_fixture_warnings": top_warnings,
        "top_player_impacts": top_players,
        "backtest_preview": backtest.head(80).to_dict("records") if not backtest.empty else [],
        "notes": [
            "This payload is an admin/research view only.",
            "Warning tokens do not override deploy gates until walk-forward proof exists.",
            "Missing player-history joins should be treated as a data-quality issue before relying on the warning.",
        ],
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def write_markdown(
    output_md: Path,
    fixture_board: pd.DataFrame,
    player_rows: pd.DataFrame,
    coverage: pd.DataFrame,
    backtest: pd.DataFrame,
) -> None:
    warnings = fixture_board[pd.to_numeric(fixture_board.get("deploy_warning_flag", 0), errors="coerce").fillna(0).eq(1)].copy() if not fixture_board.empty else pd.DataFrame()
    lines = [
        "# Injury Shock Coverage Scan",
        "",
        f"- generated_at: `{utc_now()}`",
        "- research_only: `true`",
        "- deploy_policy_changed: `false`",
        "",
        "## Summary",
        "",
        f"- leagues_scanned: `{len(coverage)}`",
        f"- fixtures_scored: `{len(fixture_board)}`",
        f"- injury_player_rows: `{len(player_rows)}`",
        f"- warning_fixtures: `{len(warnings)}`",
        f"- settled_backtest_rows: `{len(backtest)}`",
        "",
        "## League Coverage",
        "",
    ]
    if coverage.empty:
        lines.append("No league coverage rows produced.")
    else:
        for _, row in coverage.sort_values(["source_freshness_status", "league_tag"]).iterrows():
            lines.append(
                f"- `{row['league_tag']}` {row['season']}: freshness=`{row['source_freshness_status']}`, "
                f"fixtures={row['fixtures_scored']}, injuries={row['injury_rows']}, warnings={row['warning_fixtures']}, "
                f"expected_xi_absences={row['expected_xi_absence_rows']}, "
                f"midweek_known={row.get('known_since_midweek_rows', 0)}, fixture_only_late={row.get('fixture_only_late_confirmation_rows', 0)}"
            )
    lines.extend(["", "## Top Fixture Warnings", ""])
    if warnings.empty:
        lines.append("No fixture warning rows surfaced.")
    else:
        warnings["admin_sort"] = warnings[["home_injury_news_severity", "away_injury_news_severity", "motivation_volatility_score", "ftr_volatility_adjustment"]].max(axis=1)
        for _, row in warnings.sort_values("admin_sort", ascending=False).head(25).iterrows():
            lines.append(f"### {row['home_team_name']} vs {row['away_team_name']}")
            lines.append(f"- fixture_key: `{row['fixture_key']}`")
            lines.append(f"- warning_tokens: `{row['warning_tokens']}`")
            lines.append(
                f"- scores: home_attack={row['home_attack_absence_score']}, away_attack={row['away_attack_absence_score']}, "
                f"home_def={row['home_defence_absence_score']}, away_def={row['away_defence_absence_score']}, motivation={row['motivation_volatility_score']}"
            )
            lines.append(f"- home_absence_reasons: `{row.get('home_absence_reasons', '')}`")
            lines.append(f"- away_absence_reasons: `{row.get('away_absence_reasons', '')}`")
            lines.append("")
    lines.extend(["## Top Player Impact Rows", ""])
    if player_rows.empty:
        lines.append("No player impact rows produced.")
    else:
        sort_weight = "importance_weight_used" if "importance_weight_used" in player_rows.columns else "importance_weight"
        for _, row in player_rows.sort_values(["expected_xi_absence_flag", sort_weight], ascending=[False, False]).head(30).iterrows():
            lines.append(
                f"- `{row['league']}` {row['team_name']} | {row['player_name']} | function=`{row['structural_function']}` | "
                f"importance={row.get(sort_weight, row['importance_weight'])} | source=`{row.get('importance_source', '')}` | "
                f"expected_xi_absent={row['expected_xi_absence_flag']} | known_hours={row['known_hours_before_kickoff']} | "
                f"scope=`{row.get('source_scope', '')}` | first_seen=`{row.get('availability_first_seen_ts_utc', '')}` | "
                f"rating_join=`{row['player_rating_join_status']}`"
            )
    lines.extend(["", "## Player Rating Join Quality", ""])
    if player_rows.empty or "player_rating_join_status" not in player_rows.columns:
        lines.append("No player rating join rows produced.")
    else:
        for status, count in player_rows["player_rating_join_status"].astype(str).value_counts().items():
            lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Backtest Join", ""])
    if backtest.empty:
        lines.append("No settled deploy/result rows joined in this scan.")
    else:
        settled = backtest[backtest["injury_shock_join_status"].eq("JOINED")].copy()
        shock = settled[pd.to_numeric(settled.get("deploy_warning_flag", 0), errors="coerce").fillna(0).eq(1)]
        losses = shock[shock.get("result_status", "").astype(str).str.lower().eq("lost")]
        lines.append(f"- joined_settled_rows: `{len(settled)}`")
        lines.append(f"- shock_warning_settled_rows: `{len(shock)}`")
        lines.append(f"- shock_warning_losses: `{len(losses)}`")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_runbook(path: Path) -> None:
    lines = [
        "# Injury Shock Coverage Scan Runbook",
        "",
        "Purpose: recurring pre-deploy injury, lineup, and motivation risk radar across covered leagues.",
        "",
        "Standard command:",
        "",
        "```bash",
        "python3 scripts/build_injury_shock_coverage_scan.py \\",
        "  --outdir reports/latest/injury_shock_coverage_scan \\",
        "  --admin-json frontend/public/data/internal/injury_shock_admin_dashboard.json",
        "```",
        "",
        "Required outputs:",
        "",
        "- `INJURY_SHOCK_COVERAGE_SCAN.csv`",
        "- `INJURY_SHOCK_PLAYER_IMPACT.csv`",
        "- `INJURY_SHOCK_PLAYER_RATING_JOIN_AUDIT.csv`",
        "- `INJURY_SHOCK_LEAGUE_COVERAGE.csv`",
        "- `INJURY_SHOCK_BACKTEST_LINK.csv`",
        "- `INJURY_SHOCK_COVERAGE_SCAN.md`",
        "- `frontend/public/data/internal/injury_shock_admin_dashboard.json`",
        "",
        "Guardrail: this is a reporting layer only. It must not mutate `deploy_rulebook.py` or promote vetoes without walk-forward proof.",
        "",
        "Fresh source refresh command when leagues are `RED_MISSING_SOURCE`:",
        "",
        "```bash",
        "python3 scripts/api_football/refresh_current_context_overlay_window.py \\",
        "  --from-date YYYY-MM-DD \\",
        "  --to-date YYYY-MM-DD \\",
        "  --outdir reports/latest/api_current_context_overlay_window_YYYY_MM_DD_to_YYYY_MM_DD",
        "```",
        "",
        "Structured injury availability refresh command for the Injury Shock engine:",
        "",
        "```bash",
        "python3 scripts/api_football/refresh_current_injury_lineup_window.py \\",
        "  --from-date YYYY-MM-DD \\",
        "  --to-date YYYY-MM-DD \\",
        "  --injury-query-scopes fixture,league_season,league_date,team_season \\",
        "  --include-sidelined \\",
        "  --outdir reports/latest/api_current_injury_lineup_window_YYYY_MM_DD_to_YYYY_MM_DD",
        "```",
        "",
        "Interpretation:",
        "",
        "- `fixture` injury scope is late match confirmation.",
        "- `league_season`, `league_date`, and `team_season` are the early-warning availability layer.",
        "- `availability_first_seen_ts_utc` comes from the persistent local seen registry.",
        "- `fixture_only_late_confirmation_flag=1` means OG only learned it through fixture-level confirmation.",
        "",
        "When a large refresh is split into multiple league batches, rebuild the combined manifest before scanning:",
        "",
        "```bash",
        "python3 scripts/api_football/build_current_injury_lineup_window_manifest.py \\",
        "  --outdir reports/latest/api_current_injury_lineup_window_YYYY_MM_DD_to_YYYY_MM_DD \\",
        "  --from-date YYYY-MM-DD \\",
        "  --to-date YYYY-MM-DD \\",
        "  --injury-query-scopes fixture,league_season,league_date,team_season \\",
        "  --include-sidelined \\",
        "  --seen-registry-csv data_sources/api_football/availability_seen_registry.csv",
        "```",
        "",
        "The refresh runner also rebuilds this combined manifest automatically at the end of each completed run.",
        "",
        "The scan only trusts `reports/latest/**/normalized` folders whose parent has `CURRENT_CONTEXT_OVERLAY_MANIFEST.json`.",
        "It also trusts the lighter `CURRENT_INJURY_LINEUP_WINDOW_MANIFEST.json` produced by `refresh_current_injury_lineup_window.py`.",
        "Smoke-test folders are ignored so partial test pulls cannot contaminate production coverage.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def valid_report_normalized_root(path: Path) -> bool:
    parent = path.parent
    if "smoke" in str(parent).lower():
        return False
    if (parent / "CURRENT_CONTEXT_OVERLAY_MANIFEST.json").exists():
        return True
    if (parent / "CURRENT_INJURY_LINEUP_WINDOW_MANIFEST.json").exists():
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all-league injury shock coverage report.")
    parser.add_argument("--normalized-root", default=str(ROOT / "data_sources/api_football/normalized"))
    parser.add_argument("--include-reports-latest", action="store_true", default=True)
    parser.add_argument("--context-csv", default=str(DEFAULT_CONTEXT))
    parser.add_argument("--player-ratings-csv", default=str(DEFAULT_PLAYER_RATINGS))
    parser.add_argument("--results-feed-json", default=str(DEFAULT_RESULTS_FEED))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--admin-json", default=str(DEFAULT_ADMIN_JSON))
    parser.add_argument("--runbook-md", default=str(DEFAULT_DOC))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "league_boards").mkdir(parents=True, exist_ok=True)

    roots = [Path(args.normalized_root)]
    if args.include_reports_latest:
        roots.extend(
            root
            for root in sorted((ROOT / "reports/latest").glob("**/normalized"))
            if valid_report_normalized_root(root)
        )

    context_csv = Path(args.context_csv) if args.context_csv else None
    player_ratings = load_player_ratings(Path(args.player_ratings_csv))
    fixture_sets = discover_fixture_sets(roots)
    fixture_boards: list[pd.DataFrame] = []
    player_boards: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    for item in fixture_sets:
        if item.get("player_stats_csv") is None or not item["player_stats_csv"].exists():
            empty_board = pd.DataFrame()
            empty_players = pd.DataFrame()
        else:
            empty_board = load_fixture_board(item, outdir, context_csv)
            empty_players = build_player_impact_rows(
                fixtures_csv=item["fixtures_csv"],
                injuries_csv=item["injuries_csv"],
                sidelined_csv=item.get("sidelined_csv"),
                player_stats_csv=item["player_stats_csv"],
                lineups_csv=item["lineups_csv"],
                player_ratings=player_ratings,
                context_csv=context_csv,
                league_tag=item["league_tag"],
            )
            empty_board = apply_player_impact_overlay(empty_board, empty_players)
        if not empty_board.empty:
            empty_board["league_tag"] = item["league_tag"]
            empty_board["source_root"] = str(item["root"])
            empty_board["injury_source_csv"] = str(item["injuries_csv"]) if item.get("injuries_csv") else ""
            empty_board["sidelined_source_csv"] = str(item["sidelined_csv"]) if item.get("sidelined_csv") else ""
            empty_board["player_stats_source_csv"] = str(item["player_stats_csv"]) if item.get("player_stats_csv") else ""
            fixture_boards.append(empty_board)
        if not empty_players.empty:
            empty_players["league_tag"] = item["league_tag"]
            empty_players["source_root"] = str(item["root"])
            empty_players["injury_source_csv"] = str(item["injuries_csv"]) if item.get("injuries_csv") else ""
            empty_players["sidelined_source_csv"] = str(item["sidelined_csv"]) if item.get("sidelined_csv") else ""
            empty_players["player_stats_source_csv"] = str(item["player_stats_csv"]) if item.get("player_stats_csv") else ""
            player_boards.append(empty_players)
        coverage_rows.append(build_league_coverage_row(item, empty_board, empty_players))

    fixture_board = pd.concat(fixture_boards, ignore_index=True, sort=False) if fixture_boards else pd.DataFrame()
    player_rows = pd.concat(player_boards, ignore_index=True, sort=False) if player_boards else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    results = result_items(Path(args.results_feed_json))
    backtest = build_backtest_link(fixture_board, results)

    fixture_csv = outdir / "INJURY_SHOCK_COVERAGE_SCAN.csv"
    player_csv = outdir / "INJURY_SHOCK_PLAYER_IMPACT.csv"
    player_rating_audit_csv = outdir / "INJURY_SHOCK_PLAYER_RATING_JOIN_AUDIT.csv"
    coverage_csv = outdir / "INJURY_SHOCK_LEAGUE_COVERAGE.csv"
    backtest_csv = outdir / "INJURY_SHOCK_BACKTEST_LINK.csv"
    output_md = outdir / "INJURY_SHOCK_COVERAGE_SCAN.md"
    admin_json = Path(args.admin_json)

    fixture_board.to_csv(fixture_csv, index=False)
    player_rows.to_csv(player_csv, index=False)
    player_rating_audit_cols = [
        "league_tag",
        "season",
        "match_date",
        "team_name",
        "player_name",
        "player_rating_join_status",
        "source_scope",
        "availability_first_seen_ts_utc",
        "availability_first_seen_source_scope",
        "fixture_only_late_confirmation_flag",
        "known_since_midweek_flag",
        "sidelined_history_count",
        "sidelined_open_absence_flag",
        "matched_rating_name",
        "matched_rating_club",
        "matched_rating_competition_key",
        "og_player_power",
        "importance_source",
        "player_importance_join_status",
        "source_root",
    ]
    if player_rows.empty:
        pd.DataFrame(columns=player_rating_audit_cols).to_csv(player_rating_audit_csv, index=False)
    else:
        player_rows[[col for col in player_rating_audit_cols if col in player_rows.columns]].to_csv(player_rating_audit_csv, index=False)
    coverage.to_csv(coverage_csv, index=False)
    backtest.to_csv(backtest_csv, index=False)
    write_markdown(output_md, fixture_board, player_rows, coverage, backtest)
    admin_json.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(admin_payload(fixture_board, player_rows, coverage, backtest))
    admin_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    build_runbook(Path(args.runbook_md))

    print(f"WROTE: {fixture_csv} rows={len(fixture_board)}")
    print(f"WROTE: {player_csv} rows={len(player_rows)}")
    print(f"WROTE: {player_rating_audit_csv} rows={len(player_rows)}")
    print(f"WROTE: {coverage_csv} rows={len(coverage)}")
    print(f"WROTE: {backtest_csv} rows={len(backtest)}")
    print(f"WROTE: {output_md}")
    print(f"WROTE: {admin_json}")


if __name__ == "__main__":
    main()
