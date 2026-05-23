#!/usr/bin/env python3
"""Build a targeted API-Football provider-season refresh manifest.

Research-only helper. It converts low trusted-identity coverage and
calendar-year source gaps into provider-season API refresh rows that can be
reviewed before spending API budget.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCORED_ROOT = Path("predictions_output/hybrid_shadow_walkforward_2026_05_01_parity_rebuild")
DEFAULT_IDENTITY_SUMMARY = Path(
    "reports/latest/walkforward_intelligence_benchmark_intel_trusted_coverage_alias_lift_2026_05_18/"
    "00_fixture_identity_map/fixture_identity_league_summary.csv"
)
DEFAULT_SCOPE_WORKLIST = Path("reports/latest/fixture_identity_alias_suggestions_2026_05_18/fixture_identity_scope_worklist.csv")
DEFAULT_NORMALIZED_DIR = Path("data_sources/api_football/normalized")
DEFAULT_RAW_DIR = Path("data_sources/api_football/raw")
DEFAULT_OUT = Path("reports/latest/api_football_targeted_source_refresh_manifest_2026_05_18.csv")

FALLBACK_LEAGUE_IDS = {
    "Austria_Bundesliga": 218,
    "Belgium_Pro": 144,
    "Brazil_Serie_A": 71,
    "Champions_League": 2,
    "Czech_First_League": 345,
    "Denmark_Superliga": 119,
    "England_Championship": 40,
    "England_EFL_League_1": 41,
    "England_FA_Cup": 45,
    "England_Premier_League": 39,
    "Europa_Conference": 848,
    "Europa_League": 3,
    "France_Ligue_1": 61,
    "Germany_Bundesliga": 78,
    "Germany_Bundesliga_2": 79,
    "Italy_Serie_A": 135,
    "Japan_J1": 98,
    "Netherlands_Eredivisie": 88,
    "Norway_Eliteserien": 103,
    "Portugal_Liga": 94,
    "Scotland_Premiership": 179,
    "South_Korea_K_League": 292,
    "Spain_La_Liga": 140,
    "Swiss_Super_League": 207,
    "Turkey_Super_Lig": 203,
    "USA_MLS": 253,
}

WINTER_LEAGUES = {
    "Austria_Bundesliga",
    "Belgium_Pro",
    "Champions_League",
    "Czech_First_League",
    "Denmark_Superliga",
    "England_Championship",
    "England_EFL_League_1",
    "England_FA_Cup",
    "England_Premier_League",
    "Europa_Conference",
    "Europa_League",
    "France_Ligue_1",
    "Germany_Bundesliga",
    "Germany_Bundesliga_2",
    "Italy_Serie_A",
    "Netherlands_Eredivisie",
    "Portugal_Liga",
    "Scotland_Premiership",
    "Spain_La_Liga",
    "Swiss_Super_League",
    "Turkey_Super_Lig",
}


def league_tag(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def fixture_count_raw(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            count += len(payload.get("response", []) or [])
    return count


def load_league_id_map(normalized_dir: Path) -> dict[str, int]:
    mapping = dict(FALLBACK_LEAGUE_IDS)
    for path in sorted(normalized_dir.glob("fixtures_master__*.csv")):
        frame = safe_read_csv(path)
        if frame.empty or "league_id" not in frame.columns:
            continue
        stem = path.name.replace("fixtures_master__", "").replace(".csv", "")
        parts = stem.rsplit("__", 1)
        tag = parts[0]
        values = pd.to_numeric(frame["league_id"], errors="coerce").dropna()
        if values.empty:
            continue
        mapping[tag] = int(values.iloc[0])
    return mapping


def load_scored_dates(scored_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    wanted = {"league", "match_date", "fixture_key"}
    for path in sorted(scored_root.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv")):
        frame = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
        frame["match_date_dt"] = pd.to_datetime(frame.get("match_date"), errors="coerce")
        frame["calendar_year"] = frame["match_date_dt"].dt.year.astype("Int64")
        frame["league_tag"] = frame.get("league", "").map(league_tag)
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["league", "league_tag", "calendar_year", "sample_dates", "required_unique_fixtures"])
    df = pd.concat(rows, ignore_index=True, sort=False).dropna(subset=["calendar_year"])
    grouped = []
    for (league, tag, year), group in df.groupby(["league", "league_tag", "calendar_year"], dropna=False):
        dates = sorted(group["match_date_dt"].dt.strftime("%Y-%m-%d").dropna().unique().tolist())
        grouped.append(
            {
                "league": league,
                "league_tag": tag,
                "calendar_year": int(year),
                "scored_min_date": dates[0] if dates else "",
                "scored_max_date": dates[-1] if dates else "",
                "scored_sample_dates": ",".join(dates[:12]),
                "required_scored_rows": int(len(group)),
                "required_unique_fixtures": int(group["fixture_key"].nunique()),
            }
        )
    return pd.DataFrame(grouped)


def parse_dates(text: Any) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for item in str(text or "").split(","):
        parsed = pd.to_datetime(item.strip(), errors="coerce")
        if not pd.isna(parsed):
            dates.append(parsed)
    return dates


def int_value(value: Any, default: int = 0) -> int:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return int(parsed)


def float_value(value: Any, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return float(parsed)


def provider_seasons_for(tag: str, calendar_year: int, dates: list[pd.Timestamp]) -> list[int]:
    if tag not in WINTER_LEAGUES:
        return [calendar_year]
    if not dates:
        if calendar_year >= 2026:
            return [calendar_year - 1]
        return [calendar_year]
    months = sorted({int(date.month) for date in dates})
    if months and max(months) <= 6:
        return [calendar_year - 1]
    if months and min(months) >= 7:
        return [calendar_year]
    return [calendar_year - 1, calendar_year]


def row_reason(scope_issue: str, usable_rate: float, api_fixtures: int, existing_raw: int) -> str:
    reasons: list[str] = []
    if "NO_API" in scope_issue:
        reasons.append("NO_LOCAL_API_SOURCE")
    if "SCORED_DATES_NOT_IN_API" in scope_issue:
        reasons.append("CALENDAR_DATES_MISSING_FROM_API_SOURCE")
    if usable_rate == 0:
        reasons.append("ZERO_TRUSTED_COVERAGE")
    elif usable_rate < 0.6:
        reasons.append("LOW_TRUSTED_COVERAGE")
    if api_fixtures > 0 and existing_raw > 0 and api_fixtures <= max(15, existing_raw):
        reasons.append("POSSIBLY_STALE_PARTIAL_SOURCE")
    return "|".join(dict.fromkeys(reasons)) or "TARGETED_REFRESH"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", default=str(DEFAULT_SCORED_ROOT))
    parser.add_argument("--identity-league-summary", default=str(DEFAULT_IDENTITY_SUMMARY))
    parser.add_argument("--scope-worklist", default=str(DEFAULT_SCOPE_WORKLIST))
    parser.add_argument("--normalized-dir", default=str(DEFAULT_NORMALIZED_DIR))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--max-usable-rate", type=float, default=0.60)
    parser.add_argument("--min-fixtures", type=int, default=25)
    parser.add_argument(
        "--include-low-coverage-only",
        action="store_true",
        help="Also include low trusted-coverage rows that do not have an explicit source gap.",
    )
    parser.add_argument("--target-leagues", default="", help="Optional comma-separated league tags to restrict refresh.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    identity = safe_read_csv(Path(args.identity_league_summary))
    scope = safe_read_csv(Path(args.scope_worklist))
    scored_dates = load_scored_dates(Path(args.scored_root))
    league_ids = load_league_id_map(Path(args.normalized_dir))

    if identity.empty:
        raise SystemExit(f"No identity summary rows found: {args.identity_league_summary}")
    identity["league_tag"] = identity["league"].map(league_tag)
    identity["calendar_year"] = pd.to_numeric(identity["calendar_year"], errors="coerce").astype("Int64")
    identity["fixtures"] = pd.to_numeric(identity["fixtures"], errors="coerce").fillna(0).astype(int)
    identity["usable_rate"] = pd.to_numeric(identity["usable_rate"], errors="coerce").fillna(0.0)

    candidates = identity[
        identity["fixtures"].ge(args.min_fixtures)
        & identity["usable_rate"].le(args.max_usable_rate)
    ].copy()
    if args.target_leagues.strip():
        target = {league_tag(item) for item in args.target_leagues.split(",") if item.strip()}
        candidates = candidates[candidates["league_tag"].isin(target)].copy()

    if not scope.empty:
        scope["league_tag"] = scope["league_tag"].map(league_tag)
        scope["calendar_year"] = pd.to_numeric(scope["calendar_year"], errors="coerce").astype("Int64")
        scope_keep = scope[
            ["league_tag", "calendar_year", "scope_issue", "scored_fixtures", "api_fixtures", "sample_dates"]
        ].copy()
        candidates = candidates.merge(scope_keep, on=["league_tag", "calendar_year"], how="left")
    else:
        candidates["scope_issue"] = ""
        candidates["scored_fixtures"] = candidates["fixtures"]
        candidates["api_fixtures"] = 0
        candidates["sample_dates"] = ""

    if not args.include_low_coverage_only:
        source_gap = candidates["scope_issue"].astype("string").fillna("").str.contains(
            "NO_API|SCORED_DATES_NOT_IN_API",
            regex=True,
        )
        zero_coverage = pd.to_numeric(candidates["usable_rate"], errors="coerce").fillna(0.0).eq(0.0)
        candidates = candidates[source_gap | zero_coverage].copy()

    candidates = candidates.merge(
        scored_dates[
            [
                "league_tag",
                "calendar_year",
                "scored_min_date",
                "scored_max_date",
                "scored_sample_dates",
                "required_scored_rows",
                "required_unique_fixtures",
            ]
        ],
        on=["league_tag", "calendar_year"],
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        tag = str(row["league_tag"])
        calendar_year = int(row["calendar_year"])
        sample_dates = parse_dates(row.get("sample_dates"))
        if not sample_dates:
            sample_dates = parse_dates(row.get("scored_sample_dates"))
        provider_seasons = provider_seasons_for(tag, calendar_year, sample_dates)
        league_id = league_ids.get(tag)
        for provider_season in provider_seasons:
            raw_path = Path(args.raw_dir) / f"fixtures__league_{league_id}__season_{provider_season}__fixtures.jsonl"
            existing_raw = fixture_count_raw(raw_path) if league_id is not None else 0
            rows.append(
                {
                    "manifest_status": "READY" if league_id is not None else "MISSING_LEAGUE_ID",
                    "league": row.get("league", tag),
                    "league_tag": tag,
                    "league_id": league_id,
                    "season": provider_season,
                    "calendar_year": calendar_year,
                    "required_scored_rows": int_value(row.get("required_scored_rows"), int_value(row.get("fixtures"))),
                    "required_unique_fixtures": int_value(row.get("required_unique_fixtures"), int_value(row.get("fixtures"))),
                    "identity_usable_rate": float_value(row.get("usable_rate")),
                    "identity_usable_fixtures": int_value(row.get("usable_fixtures")),
                    "identity_total_fixtures": int_value(row.get("fixtures")),
                    "scope_issue": str(row.get("scope_issue") or ""),
                    "scope_api_fixtures": int_value(row.get("api_fixtures")),
                    "existing_raw_fixture_count": existing_raw,
                    "scored_min_date": row.get("scored_min_date", ""),
                    "scored_max_date": row.get("scored_max_date", ""),
                    "refresh_reason": row_reason(
                        str(row.get("scope_issue") or ""),
                        float_value(row.get("usable_rate")),
                        int_value(row.get("api_fixtures")),
                        existing_raw,
                    ),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        group_cols = ["manifest_status", "league", "league_tag", "league_id", "season"]
        out = (
            out.groupby(group_cols, dropna=False)
            .agg(
                calendar_years=("calendar_year", lambda s: ",".join(str(int(v)) for v in sorted(set(s)))),
                required_scored_rows=("required_scored_rows", "sum"),
                required_unique_fixtures=("required_unique_fixtures", "sum"),
                identity_usable_rate=("identity_usable_rate", "min"),
                identity_usable_fixtures=("identity_usable_fixtures", "sum"),
                identity_total_fixtures=("identity_total_fixtures", "sum"),
                scope_issue=("scope_issue", lambda s: "|".join(sorted({str(v) for v in s if str(v) and str(v) != "nan"}))),
                scope_api_fixtures=("scope_api_fixtures", "max"),
                existing_raw_fixture_count=("existing_raw_fixture_count", "max"),
                scored_min_date=("scored_min_date", "min"),
                scored_max_date=("scored_max_date", "max"),
                refresh_reason=("refresh_reason", lambda s: "|".join(sorted({part for v in s for part in str(v).split("|") if part}))),
            )
            .reset_index()
            .sort_values(["manifest_status", "season", "league_tag"])
            .reset_index(drop=True)
        )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    ready = int(out["manifest_status"].eq("READY").sum()) if not out.empty else 0
    print(f"Rows: {len(out)}")
    print(f"Ready rows: {ready}")
    print(f"Output: {output}")
    if not out.empty:
        print(out[["manifest_status", "league_tag", "season", "calendar_years", "refresh_reason"]].head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
