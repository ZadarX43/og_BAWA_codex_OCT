#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ======================================================================================
# Window generation
# ======================================================================================
WEEKDAY_NAME_TO_NUM = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def parse_date(text: str) -> date:
    return datetime.strptime(str(text).strip(), "%Y-%m-%d").date()


def daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    date_from: str
    date_to: str
    window_type: str
    season_label: str



def infer_season_label(d: date) -> str:
    # Football season style: Aug 2024 -> 2024_2025
    if d.month >= 7:
        return f"{d.year}_{d.year + 1}"
    return f"{d.year - 1}_{d.year}"



def first_weekday_on_or_after(start: date, weekday_num: int) -> date:
    delta = (weekday_num - start.weekday()) % 7
    return start + timedelta(days=delta)



def generate_fixed_windows(
    *,
    start_date: date,
    end_date: date,
    anchor_weekday: int,
    span_days: int,
    prefix: str,
    window_type: str,
) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    anchor = first_weekday_on_or_after(start_date, anchor_weekday)

    while anchor <= end_date:
        d_from = anchor
        d_to = anchor + timedelta(days=span_days - 1)
        if d_to > end_date:
            break
        season_label = infer_season_label(d_from)
        window_id = f"{prefix}_{d_from:%Y_%m_%d}_{d_to:%Y_%m_%d}"
        windows.append(
            WindowSpec(
                window_id=window_id,
                date_from=d_from.isoformat(),
                date_to=d_to.isoformat(),
                window_type=window_type,
                season_label=season_label,
            )
        )
        anchor += timedelta(days=7)

    return windows



def build_manifest_dataframe(
    *,
    start_date: date,
    end_date: date,
    include_weekend: bool,
    include_midweek_uefa: bool,
) -> pd.DataFrame:
    windows: list[WindowSpec] = []

    if include_weekend:
        windows.extend(
            generate_fixed_windows(
                start_date=start_date,
                end_date=end_date,
                anchor_weekday=WEEKDAY_NAME_TO_NUM["friday"],
                span_days=5,
                prefix="wf",
                window_type="weekend_fri_tue",
            )
        )

    if include_midweek_uefa:
        windows.extend(
            generate_fixed_windows(
                start_date=start_date,
                end_date=end_date,
                anchor_weekday=WEEKDAY_NAME_TO_NUM["tuesday"],
                span_days=3,
                prefix="uefa",
                window_type="midweek_tue_thu",
            )
        )

    df = pd.DataFrame([w.__dict__ for w in windows])
    if df.empty:
        return df

    df = df.sort_values(["date_from", "date_to", "window_type", "window_id"]).reset_index(drop=True)
    return df


# ======================================================================================
# Calendar flagging
# ======================================================================================
EURO_COMPETITION_KEYWORDS = [
    "champions league",
    "europa league",
    "conference league",
    "uefa",
]

DOMESTIC_CUP_KEYWORDS = [
    "fa cup",
    "efl cup",
    "carabao cup",
    "copa del rey",
    "coppa italia",
    "dfb pokal",
    "coupe de france",
    "knvb beker",
    "taça de portugal",
    "taça de portugal placard",
    "j.league cup",
    "emperor",
]



def month_name_from_num(m: int) -> str:
    return calendar.month_name[int(m)] if pd.notna(m) and 1 <= int(m) <= 12 else ""



def compute_festive_period_flag(date_from: pd.Series, date_to: pd.Series) -> pd.Series:
    starts = pd.to_datetime(date_from, errors="coerce")
    ends = pd.to_datetime(date_to, errors="coerce")

    def _is_festive(row_start: pd.Timestamp, row_end: pd.Timestamp) -> bool:
        if pd.isna(row_start) or pd.isna(row_end):
            return False
        cur = row_start.date()
        end = row_end.date()
        while cur <= end:
            if (cur.month == 12 and cur.day >= 20) or (cur.month == 1 and cur.day <= 7):
                return True
            cur += timedelta(days=1)
        return False

    return pd.Series(
        [_is_festive(s, e) for s, e in zip(starts, ends)],
        index=date_from.index,
        dtype="bool",
    )



def compute_month_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt_from = pd.to_datetime(out["date_from"], errors="coerce")
    dt_to = pd.to_datetime(out["date_to"], errors="coerce")

    out["year"] = dt_from.dt.year
    out["month"] = dt_from.dt.month
    out["month_name"] = out["month"].map(month_name_from_num)
    out["iso_week"] = dt_from.dt.isocalendar().week.astype("Int64")
    out["quarter"] = dt_from.dt.quarter.astype("Int64")
    out["is_weekend_window"] = dt_from.dt.weekday.eq(4)
    out["is_midweek_window"] = dt_from.dt.weekday.isin([1, 2])
    out["is_festive_period"] = compute_festive_period_flag(out["date_from"], out["date_to"])
    out["is_august"] = out["month"].eq(8)
    out["is_september"] = out["month"].eq(9)
    out["is_october"] = out["month"].eq(10)
    out["is_november"] = out["month"].eq(11)
    out["is_december"] = out["month"].eq(12)
    out["is_january"] = out["month"].eq(1)
    out["is_february"] = out["month"].eq(2)
    out["is_march"] = out["month"].eq(3)
    out["days_in_window"] = (dt_to - dt_from).dt.days.add(1).astype("Int64")
    return out



def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower()



def classify_competition_flags(series: pd.Series) -> pd.DataFrame:
    txt = normalize_text_series(series)
    is_euro = pd.Series(False, index=series.index)
    for kw in EURO_COMPETITION_KEYWORDS:
        is_euro = is_euro | txt.str.contains(kw, regex=False)

    is_domestic_cup = pd.Series(False, index=series.index)
    for kw in DOMESTIC_CUP_KEYWORDS:
        is_domestic_cup = is_domestic_cup | txt.str.contains(kw, regex=False)

    return pd.DataFrame(
        {
            "competition_text_norm": txt,
            "is_euro_competition": is_euro,
            "is_domestic_cup": is_domestic_cup,
        }
    )



def add_calendar_flags_to_scored_df(
    scored_df: pd.DataFrame,
    *,
    uefa_competitions_path: str | None = None,
    domestic_cups_path: str | None = None,
    lookback_days: int = 4,
    lookahead_days: int = 4,
) -> pd.DataFrame:
    """
    Enrich a scored deploy dataframe with calendar regime fields.

    This function is intentionally conservative. It works in two modes:
    1. Baseline mode without external fixtures: month / festive / weekend / midweek flags only.
    2. Overlay mode with optional competition fixture CSVs for Europe/domestic cup timing.

    Expected optional fixture CSV columns:
      - match_date
      - home_team_name
      - away_team_name
      - league OR competition
    """
    out = scored_df.copy()
    if out.empty:
        return out

    out["match_date"] = pd.to_datetime(out.get("match_date", pd.Series(pd.NaT, index=out.index)), errors="coerce")
    out["month"] = out["match_date"].dt.month
    out["month_name"] = out["month"].map(month_name_from_num)
    out["year"] = out["match_date"].dt.year
    out["iso_week"] = out["match_date"].dt.isocalendar().week.astype("Int64")
    out["weekday_num"] = out["match_date"].dt.weekday
    out["weekday_name"] = out["match_date"].dt.day_name()
    out["is_weekend_match"] = out["weekday_num"].isin([4, 5, 6, 0])
    out["is_midweek_match"] = out["weekday_num"].isin([1, 2, 3])
    out["is_festive_period"] = out["match_date"].dt.month.eq(12) & out["match_date"].dt.day.ge(20)
    out["is_festive_period"] = out["is_festive_period"] | (out["match_date"].dt.month.eq(1) & out["match_date"].dt.day.le(7))

    out["uefa_competition_in_window"] = False
    out["domestic_cup_in_window"] = False
    out["team_played_europe_recently"] = False
    out["team_plays_europe_soon"] = False
    out["team_europe_overlap_flag"] = False
    out["team_played_domestic_cup_recently"] = False
    out["team_plays_domestic_cup_soon"] = False

    def _load_fixture_file(path_str: str | None) -> pd.DataFrame:
        if not path_str:
            return pd.DataFrame()
        p = Path(path_str)
        if not p.exists():
            return pd.DataFrame()
        fx = pd.read_csv(p, low_memory=False)
        if fx.empty:
            return fx
        fx["match_date"] = pd.to_datetime(fx.get("match_date", pd.Series(pd.NaT, index=fx.index)), errors="coerce")
        fx["home_team_name"] = fx.get("home_team_name", pd.Series("", index=fx.index)).astype("string").fillna("").str.strip()
        fx["away_team_name"] = fx.get("away_team_name", pd.Series("", index=fx.index)).astype("string").fillna("").str.strip()

        comp_col = "league" if "league" in fx.columns else "competition" if "competition" in fx.columns else None
        if comp_col is None:
            fx["competition_name"] = ""
        else:
            fx["competition_name"] = fx[comp_col].astype("string").fillna("").str.strip()
        return fx

    euro_fx = _load_fixture_file(uefa_competitions_path)
    cup_fx = _load_fixture_file(domestic_cups_path)

    def _enrich_overlap_flags(target_df: pd.DataFrame, event_df: pd.DataFrame, *, prefix: str) -> None:
        if event_df.empty:
            return

        event_df = event_df.copy()
        event_df["is_same_day_event"] = True

        team_events = pd.concat(
            [
                event_df[["match_date", "home_team_name", "competition_name"]].rename(columns={"home_team_name": "team_name"}),
                event_df[["match_date", "away_team_name", "competition_name"]].rename(columns={"away_team_name": "team_name"}),
            ],
            ignore_index=True,
        )
        team_events["team_name"] = team_events["team_name"].astype("string").fillna("").str.strip()
        team_events = team_events.loc[team_events["team_name"].ne("")].copy()

        all_window_hits = []
        recent_hits = []
        soon_hits = []

        for idx, row in target_df.iterrows():
            md = row.get("match_date")
            if pd.isna(md):
                all_window_hits.append(False)
                recent_hits.append(False)
                soon_hits.append(False)
                continue

            teams = [
                str(row.get("home_team_name", "")).strip(),
                str(row.get("away_team_name", "")).strip(),
            ]
            teams = [t for t in teams if t]
            if not teams:
                all_window_hits.append(False)
                recent_hits.append(False)
                soon_hits.append(False)
                continue

            sub = team_events.loc[team_events["team_name"].isin(teams)].copy()
            if sub.empty:
                all_window_hits.append(False)
                recent_hits.append(False)
                soon_hits.append(False)
                continue

            delta_days = (sub["match_date"] - md).dt.days
            all_window_hits.append(delta_days.eq(0).any())
            recent_hits.append(delta_days.between(-lookback_days, -1).any())
            soon_hits.append(delta_days.between(1, lookahead_days).any())

        target_df[f"{prefix}_in_window"] = pd.Series(all_window_hits, index=target_df.index, dtype="bool")
        target_df[f"team_played_{prefix}_recently"] = pd.Series(recent_hits, index=target_df.index, dtype="bool")
        target_df[f"team_plays_{prefix}_soon"] = pd.Series(soon_hits, index=target_df.index, dtype="bool")

    _enrich_overlap_flags(out, euro_fx, prefix="uefa_competition")
    _enrich_overlap_flags(out, cup_fx, prefix="domestic_cup")

    if "team_played_uefa_competition_recently" in out.columns:
        out["team_played_europe_recently"] = out["team_played_uefa_competition_recently"]
    if "team_plays_uefa_competition_soon" in out.columns:
        out["team_plays_europe_soon"] = out["team_plays_uefa_competition_soon"]
    if "uefa_competition_in_window" in out.columns:
        out["uefa_competition_in_window"] = out["uefa_competition_in_window"].fillna(False)

    out["team_europe_overlap_flag"] = (
        out["uefa_competition_in_window"].fillna(False)
        | out["team_played_europe_recently"].fillna(False)
        | out["team_plays_europe_soon"].fillna(False)
    )

    out["calendar_regime_tag"] = np.select(
        [
            out["is_festive_period"].fillna(False),
            out["team_europe_overlap_flag"].fillna(False),
            out["domestic_cup_in_window"].fillna(False),
            out["is_midweek_match"].fillna(False),
        ],
        [
            "FESTIVE_PERIOD",
            "EUROPE_OVERLAP",
            "DOMESTIC_CUP_PERIOD",
            "MIDWEEK_NON_EURO",
        ],
        default="NORMAL_WINDOW",
    )

    return out


# ======================================================================================
# CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate walk-forward manifest and/or add calendar flags.")

    sub = ap.add_subparsers(dest="command", required=True)

    ap_manifest = sub.add_parser("make-manifest", help="Build a window manifest CSV.")
    ap_manifest.add_argument("--start-date", required=True)
    ap_manifest.add_argument("--end-date", required=True)
    ap_manifest.add_argument("--out", required=True)
    ap_manifest.add_argument("--no-weekend", action="store_true")
    ap_manifest.add_argument("--include-midweek-uefa", action="store_true")

    ap_flags = sub.add_parser("add-calendar-flags", help="Add calendar regime flags to a scored CSV.")
    ap_flags.add_argument("--src", required=True)
    ap_flags.add_argument("--out", required=True)
    ap_flags.add_argument("--uefa-fixtures", default="")
    ap_flags.add_argument("--domestic-cup-fixtures", default="")
    ap_flags.add_argument("--lookback-days", type=int, default=4)
    ap_flags.add_argument("--lookahead-days", type=int, default=4)

    return ap.parse_args()



def cmd_make_manifest(args: argparse.Namespace) -> None:
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    df = build_manifest_dataframe(
        start_date=start_date,
        end_date=end_date,
        include_weekend=not args.no_weekend,
        include_midweek_uefa=bool(args.include_midweek_uefa),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote manifest: {out_path}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df.head(10).to_string(index=False))



def cmd_add_calendar_flags(args: argparse.Namespace) -> None:
    src = Path(args.src)
    out = Path(args.out)
    df = pd.read_csv(src, low_memory=False)
    flagged = add_calendar_flags_to_scored_df(
        df,
        uefa_competitions_path=args.uefa_fixtures or None,
        domestic_cups_path=args.domestic_cup_fixtures or None,
        lookback_days=int(args.lookback_days),
        lookahead_days=int(args.lookahead_days),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(out, index=False)
    print(f"Wrote flagged file: {out}")
    print(f"Rows: {len(flagged)}")



def main() -> None:
    args = parse_args()
    if args.command == "make-manifest":
        cmd_make_manifest(args)
        return
    if args.command == "add-calendar-flags":
        cmd_add_calendar_flags(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
