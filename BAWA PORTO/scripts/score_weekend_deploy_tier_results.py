#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.client import APIFootballClient


DEFAULT_DEPLOY_DIR = ROOT / "predictions_output/2026-05-14_imp20_may14_to_may19/02_deploy"
DEFAULT_OUTDIR = ROOT / "reports/latest/weekend_deploy_tier_scoring_2026_05_14_to_2026_05_19"

TIER_FILES = {
    "ELITE": "*__DEPLOY_TIER_ELITE__*.csv",
    "STANDARD": "*__DEPLOY_TIER_STANDARD__*.csv",
    "OBSERVE": "*__DEPLOY_TIER_OBSERVE__*.csv",
}
SUPPORTED_MARKETS = {"FTR", "BTTS", "OU25"}
FINAL_STATUSES = {"FT", "AET", "PEN"}

LEAGUE_TAG_HINTS = {
    "Australia A-League": "Australia_A_League",
    "Austria Bundesliga": "Austria_Bundesliga",
    "Belgium Pro": "Belgium_Pro",
    "Brazil Serie A": "Brazil_Serie_A",
    "Denmark Superliga": "Denmark_Superliga",
    "England Championship": "England_Championship",
    "England EFL League 1": "England_EFL_League_1",
    "England FA Cup": "England_FA_Cup",
    "England Premier League": "England_Premier_League",
    "France Ligue 1": "France_Ligue_1",
    "Germany Bundesliga": "Germany_Bundesliga",
    "Germany Bundesliga 2": "Germany_Bundesliga_2",
    "Italy Serie A": "Italy_Serie_A",
    "Netherlands Eredivisie": "Netherlands_Eredivisie",
    "Norway Eliteserien": "Norway_Eliteserien",
    "Portugal Liga": "Portugal_Liga",
    "Saudi Pro League": "Saudi_Pro_League",
    "Scotland Premiership": "Scotland_Premiership",
    "South Korea K League": "South_Korea_K_League",
    "Spain La Liga": "Spain_La_Liga",
    "Switzerland Super League": "Switzerland_Super_League",
    "Turkey Super Lig": "Turkey_Super_Lig",
    "USA MLS": "USA_MLS",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_team_text(value: Any) -> str:
    text = f" {str(value or '').lower()} "
    replacements = {
        " sj earthquakes ": " san jose earthquakes ",
        " sporting kc ": " sporting kansas city ",
        " la galaxy ": " los angeles galaxy ",
        " lafc ": " los angeles fc ",
        " los angeles fc ": " los angeles fc ",
        " new york rb ": " new york red bulls ",
        " montreal impact ": " cf montreal ",
        " inter miami cf ": " inter miami ",
        " fc cincinnati ": " cincinnati ",
        " rsl ": " real salt lake ",
        " st. louis ": " st louis ",
        " st louis city ": " st louis ",
        " krc genk ": " genk ",
        " angers sco ": " angers ",
        " mainz 05 ": " mainz ",
        " 1. fsv mainz 05 ": " mainz ",
        " 1. fc koln ": " koln ",
        " rb leipzig ": " leipzig ",
        " bayern munich ": " bayern munchen ",
        " m'gladbach ": " monchengladbach ",
        " borussia m gladbach ": " monchengladbach ",
        " borussia monchengladbach ": " monchengladbach ",
        " bayer 04 leverkusen ": " bayer leverkusen ",
        " stade brestois 29 ": " brest ",
        " paris saint germain ": " psg ",
        " paris sg ": " psg ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def token_set(value: Any) -> set[str]:
    aliases = {
        "1": "",
        "1899": "",
        "afc": "",
        "athletic": "",
        "borussia": "",
        "cf": "",
        "city": "",
        "club": "",
        "fc": "",
        "sc": "",
        "stade": "",
        "sv": "",
        "the": "",
        "united": "",
    }
    tokens = set()
    for token in norm(canonical_team_text(value)).split("_"):
        mapped = aliases.get(token, token)
        if mapped:
            tokens.add(mapped)
    return tokens


def token_score(left: Any, right: Any) -> float:
    lset = token_set(left)
    rset = token_set(right)
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / max(1, min(len(lset), len(rset)))


def parse_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[:10]


def date_delta_days(left: Any, right: Any) -> int | None:
    left_date = parse_date(left)
    right_date = parse_date(right)
    if not left_date or not right_date:
        return None
    try:
        ldt = datetime.fromisoformat(left_date).date()
        rdt = datetime.fromisoformat(right_date).date()
    except ValueError:
        return None
    return abs((ldt - rdt).days)


def is_final_status(value: Any) -> bool:
    return str(value or "").strip().upper() in FINAL_STATUSES


def market_key(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    if text in {"OVER25", "UNDER25", "OVER_25", "UNDER_25", "OVER_2_5", "UNDER_2_5"}:
        return "OU25"
    if text in {"BOTH_TEAMS_TO_SCORE"}:
        return "BTTS"
    return text


def normalize_pick(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    aliases = {
        "HOME_WIN": "HOME",
        "AWAY_WIN": "AWAY",
        "OVER_25": "OVER25",
        "UNDER_25": "UNDER25",
        "OVER_2_5": "OVER25",
        "UNDER_2_5": "UNDER25",
    }
    return aliases.get(text, text)


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def actual_outcome(market: str, home_goals: int | None, away_goals: int | None) -> str:
    if home_goals is None or away_goals is None:
        return ""
    market = market_key(market)
    if market == "FTR":
        if home_goals > away_goals:
            return "HOME"
        if away_goals > home_goals:
            return "AWAY"
        return "DRAW"
    if market == "BTTS":
        return "YES" if home_goals > 0 and away_goals > 0 else "NO"
    if market == "OU25":
        return "OVER25" if home_goals + away_goals > 2 else "UNDER25"
    return ""


def score_status(market: str, pick: str, home_goals: int | None, away_goals: int | None) -> str:
    actual = actual_outcome(market, home_goals, away_goals)
    if not actual:
        return "missing_actual"
    return "won" if normalize_pick(pick) == actual else "lost"


def profit_units(status: str, odds: float | None) -> float | None:
    if odds is None:
        return None
    if status == "won":
        return round(odds - 1.0, 4)
    if status == "lost":
        return -1.0
    return None


def read_tier_rows(deploy_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for tier, pattern in TIER_FILES.items():
        matches = sorted(deploy_dir.glob(pattern))
        if not matches:
            continue
        path = matches[-1]
        frame = pd.read_csv(path)
        frame["score_tier"] = tier
        frame["score_source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No tier files found in {deploy_dir}")
    rows = pd.concat(frames, ignore_index=True)
    rows["score_market"] = rows["market"].map(market_key)
    rows = rows[rows["score_market"].isin(SUPPORTED_MARKETS)].copy()
    rows["score_pick"] = rows["selection"].fillna(rows.get("bookie_pick", "")).map(normalize_pick)
    rows["score_match_date"] = rows["match_date"].map(parse_date)
    return rows


def root_priority(path: Path) -> tuple[int, int, str]:
    text = str(path)
    if "_final" in text:
        freshness = 0
    elif "reports/latest" in text:
        freshness = 1
    else:
        freshness = 2
    return (freshness, len(text), text)


def candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for base in [
        ROOT / "reports/latest",
        ROOT / "data_sources/api_football",
    ]:
        if not base.exists():
            continue
        roots.extend(path for path in base.rglob("normalized") if path.is_dir())
    # Prefer explicit final snapshots over live/HT refresh snapshots and older data_sources.
    return sorted(set(roots), key=root_priority)


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_fixture_records() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    fixtures: list[dict[str, Any]] = []
    scores: dict[int, dict[str, Any]] = {}
    seen_fixture_paths: set[Path] = set()

    for root in candidate_roots():
        for fixture_path in root.glob("fixtures_master__*.csv"):
            if fixture_path in seen_fixture_paths:
                continue
            seen_fixture_paths.add(fixture_path)
            fixture_df = read_csv_safe(fixture_path)
            if fixture_df.empty or "fixture_id" not in fixture_df.columns:
                continue
            league_tag = fixture_path.name.replace("fixtures_master__", "").rsplit("__", 1)[0]
            status_by_fixture = {
                int(row["fixture_id"]): str(row.get("status") or "")
                for row in fixture_df.to_dict("records")
                if to_int(row.get("fixture_id")) is not None
            }
            stat_path = root / fixture_path.name.replace("fixtures_master__", "match_team_stats__")
            stat_df = read_csv_safe(stat_path)
            if not stat_df.empty and {"fixture_id", "is_home", "goals_for"}.issubset(stat_df.columns):
                for fixture_id, group in stat_df.groupby("fixture_id"):
                    home = group[group["is_home"].astype(str).isin(["1", "True", "true"])]
                    away = group[group["is_home"].astype(str).isin(["0", "False", "false"])]
                    if not home.empty and not away.empty:
                        fixture_id_int = int(fixture_id)
                        status = status_by_fixture.get(fixture_id_int, "")
                        existing = scores.get(fixture_id_int)
                        if existing and existing.get("status") == "FT" and status != "FT":
                            continue
                        scores[fixture_id_int] = {
                            "home_goals": int(home.iloc[0]["goals_for"]),
                            "away_goals": int(away.iloc[0]["goals_for"]),
                            "status": status,
                            "source_root": str(root),
                        }
            for row in fixture_df.to_dict("records"):
                fixtures.append(
                    {
                        "fixture_id": to_int(row.get("fixture_id")),
                        "league_id": to_int(row.get("league_id")),
                        "season": to_int(row.get("season")),
                        "league": row.get("league"),
                        "league_tag": league_tag,
                        "match_date": parse_date(row.get("match_date") or row.get("kickoff_ts_utc")),
                        "home_team_name": row.get("home_team_name"),
                        "away_team_name": row.get("away_team_name"),
                        "status": row.get("status"),
                        "source_root": str(root),
                    }
                )
    return fixtures, scores


def match_fixture(row: pd.Series, fixtures: list[dict[str, Any]]) -> dict[str, Any] | None:
    date = row["score_match_date"]
    league_hint = LEAGUE_TAG_HINTS.get(str(row.get("league") or ""), "")
    best: dict[str, Any] | None = None
    best_score = 0.0
    best_status_score = -1
    for fx in fixtures:
        if fx.get("match_date") != date:
            continue
        if league_hint and fx.get("league_tag") != league_hint and norm(row.get("league")) not in norm(fx.get("league")):
            continue
        home_score = token_score(row.get("home_team_name"), fx.get("home_team_name"))
        away_score = token_score(row.get("away_team_name"), fx.get("away_team_name"))
        combined = min(home_score, away_score)
        status_score = 2 if str(fx.get("status") or "") == "FT" else 1 if str(fx.get("status") or "") in {"AET", "PEN"} else 0
        if combined > best_score or (combined == best_score and status_score > best_status_score):
            best = fx
            best_score = combined
            best_status_score = status_score
    if best_score >= 0.66:
        out = dict(best or {})
        out["fixture_match_score"] = round(best_score, 4)
        return out
    return None


def result_from_api_item(item: dict[str, Any]) -> dict[str, Any]:
    fixture = item.get("fixture") or {}
    teams = item.get("teams") or {}
    goals = item.get("goals") or {}
    league = item.get("league") or {}
    status = fixture.get("status") or {}
    return {
        "fixture_id": to_int(fixture.get("id")),
        "league_id": to_int(league.get("id")),
        "season": to_int(league.get("season")),
        "league": league.get("name"),
        "match_date": parse_date(fixture.get("date")),
        "home_team_name": (teams.get("home") or {}).get("name"),
        "away_team_name": (teams.get("away") or {}).get("name"),
        "home_goals": to_int(goals.get("home")),
        "away_goals": to_int(goals.get("away")),
        "status": status.get("short"),
    }


def fetch_api_results(
    rows: pd.DataFrame,
    fixture_matches: list[dict[str, Any] | None],
    args: argparse.Namespace,
    outdir: Path,
) -> list[dict[str, Any]]:
    pairs: set[tuple[int, int]] = set()
    for fx in fixture_matches:
        if not fx:
            continue
        league_id = fx.get("league_id")
        season = fx.get("season")
        if league_id and season:
            pairs.add((int(league_id), int(season)))

    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    raw_payloads: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for league_id, season in sorted(pairs):
        payload = client.get_json(
            "/fixtures",
            {
                "league": league_id,
                "season": season,
                "from": args.from_date,
                "to": args.to_date,
                "timezone": args.timezone,
            },
        )
        raw_payloads.append({"league_id": league_id, "season": season, "payload": payload})
        for item in payload.get("response") or []:
            results.append(result_from_api_item(item))

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "raw_provider_results.json").write_text(json.dumps(raw_payloads, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def match_api_result(row: pd.Series, fx: dict[str, Any] | None, api_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if fx and fx.get("fixture_id"):
        for result in api_results:
            if result.get("fixture_id") == fx.get("fixture_id"):
                return result
    best: dict[str, Any] | None = None
    best_score = 0.0
    best_date_delta = 999
    date = row["score_match_date"]
    league_id = fx.get("league_id") if fx else None
    for result in api_results:
        delta = date_delta_days(result.get("match_date"), date)
        if delta is None or delta > 1:
            continue
        if league_id and result.get("league_id") != league_id:
            continue
        combined = min(
            token_score(row.get("home_team_name"), result.get("home_team_name")),
            token_score(row.get("away_team_name"), result.get("away_team_name")),
        )
        if combined > best_score or (combined == best_score and delta < best_date_delta):
            best = result
            best_score = combined
            best_date_delta = delta
    if best_score >= 0.66:
        out = dict(best or {})
        out["api_match_score"] = round(best_score, 4)
        out["api_date_delta_days"] = best_date_delta
        return out
    return None


def summarize(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda pair: tuple(str(part) for part in pair[0])):
        settled = [row for row in items if row.get("result_status") in {"won", "lost"}]
        wins = sum(1 for row in settled if row.get("result_status") == "won")
        losses = sum(1 for row in settled if row.get("result_status") == "lost")
        profits = [row.get("profit_units") for row in settled if row.get("profit_units") not in (None, "")]
        block = {field: key[idx] for idx, field in enumerate(group_fields)}
        block.update(
            {
                "rows": len(items),
                "settled": len(settled),
                "missing_actual": sum(1 for row in items if row.get("result_status") == "missing_actual"),
                "wins": wins,
                "losses": losses,
                "hit_rate": round(wins / len(settled), 4) if settled else None,
                "profit_units": round(float(sum(profits)), 4) if profits else None,
                "roi": round(float(sum(profits)) / len(profits), 4) if profits else None,
            }
        )
        out.append(block)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Weekend Deploy Tier Scoring",
        "",
        f"Generated: {summary['generated_at']}",
        f"Window: {summary['from_date']} to {summary['to_date']}",
        f"Deploy folder: `{summary['deploy_dir']}`",
        "",
        "## Overall",
        "",
        f"- Rows: {summary['overall']['rows']}",
        f"- Settled: {summary['overall']['settled']}",
        f"- Wins: {summary['overall']['wins']}",
        f"- Losses: {summary['overall']['losses']}",
        f"- Missing actuals: {summary['overall']['missing_actual']}",
        f"- Hit rate: {summary['overall']['hit_rate']}",
        "",
        "## Tier x Market",
        "",
    ]
    for row in summary["by_tier_market"]:
        lines.append(
            f"- {row['score_tier']} {row['score_market']}: "
            f"{row['wins']}/{row['settled']} hit_rate={row['hit_rate']} missing={row['missing_actual']}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- ELITE, STANDARD, and OBSERVE are scored separately.",
            "- OBSERVE rows are research/audit only and should not be blended into live deploy hit rates.",
            "- Missing actuals are usually fixtures not yet settled, not covered by the local provider snapshot, or unmatched aliases.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ELITE/STANDARD/OBSERVE deploy tier rows by market.")
    parser.add_argument("--deploy-dir", type=Path, default=DEFAULT_DEPLOY_DIR)
    parser.add_argument("--from-date", default="2026-05-14")
    parser.add_argument("--to-date", default="2026-05-19")
    parser.add_argument("--timezone", default="Europe/London")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fetch-api-results", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--daily-cap", type=int, default=75000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deploy_dir = args.deploy_dir if args.deploy_dir.is_absolute() else ROOT / args.deploy_dir
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir

    deploy_rows = read_tier_rows(deploy_dir)
    fixtures, local_scores = load_fixture_records()
    fixture_matches = [match_fixture(row, fixtures) for _, row in deploy_rows.iterrows()]
    api_results: list[dict[str, Any]] = []
    if args.fetch_api_results:
        api_results = fetch_api_results(deploy_rows, fixture_matches, args, outdir)

    scored_rows: list[dict[str, Any]] = []
    for idx, row in deploy_rows.iterrows():
        fx = fixture_matches[idx]
        home_goals = away_goals = None
        actual_source = ""
        provider_status = ""
        provider_fixture_id = fx.get("fixture_id") if fx else None
        provider_league_id = fx.get("league_id") if fx else None
        provider_season = fx.get("season") if fx else None
        fixture_match_score = fx.get("fixture_match_score") if fx else None
        api_match_score = None
        api_date_delta_days = None

        if fx and fx.get("fixture_id") in local_scores:
            local = local_scores[int(fx["fixture_id"])]
            home_goals, away_goals = local.get("home_goals"), local.get("away_goals")
            actual_source = "local_match_team_stats"
            provider_status = str(local.get("status") or fx.get("status") or "")

        if (home_goals is None or away_goals is None) and api_results:
            result = match_api_result(row, fx, api_results)
            if result:
                home_goals = result.get("home_goals")
                away_goals = result.get("away_goals")
                provider_status = str(result.get("status") or "")
                provider_fixture_id = result.get("fixture_id") or provider_fixture_id
                provider_league_id = result.get("league_id") or provider_league_id
                provider_season = result.get("season") or provider_season
                api_match_score = result.get("api_match_score")
                api_date_delta_days = result.get("api_date_delta_days")
                actual_source = "api_football_fixture_result"

        if provider_status and not is_final_status(provider_status):
            home_goals = None
            away_goals = None

        market = str(row.get("score_market") or "")
        pick = str(row.get("score_pick") or "")
        result_status = score_status(market, pick, home_goals, away_goals)
        odds = to_float(row.get("bookie_od") or row.get("odds_ft_over25") or row.get("odds_btts_yes") or row.get("od_home"))
        scored_rows.append(
            {
                "fixture_key": row.get("fixture_key"),
                "match_date": row.get("score_match_date"),
                "league": row.get("league"),
                "home_team_name": row.get("home_team_name"),
                "away_team_name": row.get("away_team_name"),
                "score_tier": row.get("score_tier"),
                "score_market": market,
                "pick": pick,
                "actual": actual_outcome(market, home_goals, away_goals),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result_status": result_status,
                "provider_status": provider_status,
                "actual_source": actual_source,
                "provider_fixture_id": provider_fixture_id,
                "provider_league_id": provider_league_id,
                "provider_season": provider_season,
                "fixture_match_score": fixture_match_score,
                "api_match_score": api_match_score,
                "api_date_delta_days": api_date_delta_days,
                "bookie_od": odds,
                "value_edge": to_float(row.get("value_edge")),
                "profit_units": profit_units(result_status, odds),
                "model_prob": to_float(row.get("p_pick") or row.get("model_p_for_bookie")),
                "model_prob_xgb": to_float(row.get("model_p_for_bookie_xgb") or row.get("model_p_for_bookie_xgb_btts") or row.get("model_p_for_bookie_xgb_ou25")),
                "score_source_file": row.get("score_source_file"),
            }
        )

    fieldnames = [
        "fixture_key",
        "match_date",
        "league",
        "home_team_name",
        "away_team_name",
        "score_tier",
        "score_market",
        "pick",
        "actual",
        "home_goals",
        "away_goals",
        "result_status",
        "provider_status",
        "actual_source",
        "provider_fixture_id",
        "provider_league_id",
        "provider_season",
        "fixture_match_score",
        "api_match_score",
        "api_date_delta_days",
        "bookie_od",
        "value_edge",
        "profit_units",
        "model_prob",
        "model_prob_xgb",
        "score_source_file",
    ]
    write_csv(outdir / "DEPLOY_TIER_SCORE_ROWS.csv", scored_rows, fieldnames)

    summary = {
        "generated_at": utc_now(),
        "from_date": args.from_date,
        "to_date": args.to_date,
        "deploy_dir": str(deploy_dir.relative_to(ROOT)),
        "fetch_api_results": bool(args.fetch_api_results),
        "api_result_rows": len(api_results),
        "overall": summarize(scored_rows, [])[0],
        "by_tier": summarize(scored_rows, ["score_tier"]),
        "by_market": summarize(scored_rows, ["score_market"]),
        "by_tier_market": summarize(scored_rows, ["score_tier", "score_market"]),
        "by_league_tier_market": summarize(scored_rows, ["league", "score_tier", "score_market"]),
        "outputs": {
            "rows_csv": str((outdir / "DEPLOY_TIER_SCORE_ROWS.csv").relative_to(ROOT)),
            "summary_json": str((outdir / "summary.json").relative_to(ROOT)),
            "summary_md": str((outdir / "SUMMARY.md").relative_to(ROOT)),
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (outdir / "SUMMARY.md").write_text(markdown(summary), encoding="utf-8")

    print(f"Rows: {summary['overall']['rows']}")
    print(f"Settled: {summary['overall']['settled']}")
    print(f"Wins/losses: {summary['overall']['wins']} / {summary['overall']['losses']}")
    print(f"Hit rate: {summary['overall']['hit_rate']}")
    print(f"Missing actuals: {summary['overall']['missing_actual']}")
    print(f"Outputs: {outdir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
