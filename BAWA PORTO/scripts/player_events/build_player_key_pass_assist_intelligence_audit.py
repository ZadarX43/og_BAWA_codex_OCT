#!/usr/bin/env python3
"""Audit key-pass and assist intelligence from API-Football player events.

Research-only. This builds proof tables for player key-pass bands and assist
watch cells using lagged player form only. It does not create priced odds,
deploy picks, slips, or production routing changes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.team_name_map import normalize_team_name
from scripts.build_player_event_live_interaction_features import norm_text


DEFAULT_ACTUAL_ROOT = ROOT / "data_sources" / "api_football" / "normalized"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_key_pass_assist_intelligence_audit"

MARKETS = {
    "KEY_PASSES_0_5_WATCH": ("passes_key", 1, 0.62),
    "KEY_PASSES_1_5_WATCH": ("passes_key", 2, 0.35),
    "ASSIST_0_5_WATCH": ("assists", 1, 0.10),
}

SCORE_THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85]


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def parse_file_tag(path: Path) -> tuple[str, int] | None:
    match = re.match(r"match_player_stats__(.+)__(\d{4})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def read_selected(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in cols if col in header.columns]
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def load_player_events(actual_root: Path, seasons: set[int] | None) -> pd.DataFrame:
    stats_cols = [
        "fixture_id",
        "team_id",
        "player_id",
        "player_name",
        "position",
        "minutes",
        "started_flag",
        "shots_total",
        "shots_on_target",
        "passes_key",
        "assists",
    ]
    fixture_cols = [
        "fixture_id",
        "fixture_key",
        "match_date",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "status",
    ]
    frames: list[pd.DataFrame] = []
    for stats_path in sorted(actual_root.glob("match_player_stats__*.csv")):
        parsed = parse_file_tag(stats_path)
        if parsed is None:
            continue
        league_tag, season_tag = parsed
        if seasons and season_tag not in seasons:
            continue
        fixtures_path = actual_root / f"fixtures_master__{league_tag}__{season_tag}.csv"
        stats = read_selected(stats_path, stats_cols)
        fixtures = read_selected(fixtures_path, fixture_cols)
        if stats.empty or fixtures.empty:
            continue
        merged = stats.merge(fixtures, on="fixture_id", how="left")
        merged["league_tag"] = league_tag
        merged["season_tag"] = season_tag
        for col in ["fixture_id", "team_id", "home_team_id", "away_team_id", "player_id"]:
            if col in merged.columns:
                merged[col] = num(merged[col]).astype("Int64")
        merged["team_name"] = np.where(
            merged["team_id"].eq(merged["home_team_id"]),
            merged["home_team_name"],
            merged["away_team_name"],
        )
        merged["player_team_side"] = np.where(merged["team_id"].eq(merged["home_team_id"]), "HOME", "AWAY")
        merged["match_date"] = pd.to_datetime(merged["match_date"], errors="coerce")
        merged["match_month"] = merged["match_date"].dt.strftime("%Y-%m")
        merged["league"] = merged["league_tag"].astype(str).str.replace("_", " ", regex=False)
        merged["team_name_norm"] = [
            norm_text(normalize_team_name(team, league_tag)) for team in merged["team_name"]
        ]
        merged["player_name_norm"] = merged["player_name"].map(norm_text)
        merged["position_group"] = merged.get("position", "").map(position_group)
        for col in ["minutes", "started_flag", "shots_total", "shots_on_target", "passes_key", "assists"]:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = num(merged[col]).fillna(0.0)
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out[out["match_date"].notna()].copy()
    return out.sort_values(["league_tag", "player_id", "match_date", "fixture_id"]).reset_index(drop=True)


def position_group(value: Any) -> str:
    text = str(value or "").upper().strip()
    if text == "G":
        return "Goalkeeper"
    if text == "D":
        return "Defender"
    if text == "M":
        return "Midfielder"
    if text == "F":
        return "Forward"
    return "Unknown"


def add_lagged_form(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["league_tag", "player_id"]
    grouped = out.groupby(group_cols, dropna=False)
    out["prior_apps"] = grouped.cumcount()
    for col in ["passes_key", "assists", "minutes", "started_flag", "shots_total", "shots_on_target"]:
        shifted = grouped[col].shift()
        out[f"{col}_l5"] = shifted.groupby([out["league_tag"], out["player_id"]]).rolling(5, min_periods=3).mean().reset_index(level=[0, 1], drop=True)
        out[f"{col}_l8"] = shifted.groupby([out["league_tag"], out["player_id"]]).rolling(8, min_periods=4).mean().reset_index(level=[0, 1], drop=True)
    out["key_pass_1plus_l8_rate"] = (
        grouped["passes_key"]
        .shift()
        .ge(1)
        .astype(float)
        .groupby([out["league_tag"], out["player_id"]])
        .rolling(8, min_periods=4)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    out["key_pass_2plus_l8_rate"] = (
        grouped["passes_key"]
        .shift()
        .ge(2)
        .astype(float)
        .groupby([out["league_tag"], out["player_id"]])
        .rolling(8, min_periods=4)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    out["assist_l8_rate"] = (
        grouped["assists"]
        .shift()
        .ge(1)
        .astype(float)
        .groupby([out["league_tag"], out["player_id"]])
        .rolling(8, min_periods=4)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    return out


def pct_rank_by_league_season(df: pd.DataFrame, col: str) -> pd.Series:
    filled = num(df[col]).fillna(-1)
    return filled.groupby([df["league_tag"], df["season_tag"]]).rank(pct=True)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "passes_key_l5",
        "passes_key_l8",
        "key_pass_1plus_l8_rate",
        "key_pass_2plus_l8_rate",
        "assist_l8_rate",
        "minutes_l5",
        "started_flag_l5",
        "shots_total_l8",
        "shots_on_target_l8",
    ]:
        out[f"{col}_pct"] = pct_rank_by_league_season(out, col)
    creator_position_boost = out["position_group"].map({"Midfielder": 0.06, "Forward": 0.05, "Defender": -0.02}).fillna(0.0)
    out["key_pass_intel_score"] = (
        0.30 * out["passes_key_l8_pct"]
        + 0.24 * out["key_pass_1plus_l8_rate_pct"]
        + 0.16 * out["key_pass_2plus_l8_rate_pct"]
        + 0.12 * out["minutes_l5_pct"]
        + 0.10 * out["started_flag_l5_pct"]
        + 0.08 * out["shots_total_l8_pct"]
        + creator_position_boost
    ).clip(0.0, 1.0)
    out["assist_intel_score"] = (
        0.26 * out["assist_l8_rate_pct"]
        + 0.22 * out["passes_key_l8_pct"]
        + 0.18 * out["key_pass_1plus_l8_rate_pct"]
        + 0.12 * out["shots_on_target_l8_pct"]
        + 0.12 * out["minutes_l5_pct"]
        + 0.10 * out["started_flag_l5_pct"]
        + creator_position_boost
    ).clip(0.0, 1.0)
    out["creator_context_label"] = np.select(
        [
            out["key_pass_intel_score"].ge(0.85),
            out["key_pass_intel_score"].ge(0.75),
            out["key_pass_intel_score"].ge(0.65),
        ],
        ["CREATOR_CORE", "CREATOR_READY", "CREATOR_WATCH"],
        default="CREATOR_OBSERVE",
    )
    return out


def classify_status(market: str, rows: int, hit_rate: float, baseline: float, lift: float, stable_month_share: float) -> str:
    if rows < 80 or pd.isna(hit_rate):
        return "WATCH"
    if market == "KEY_PASSES_0_5_WATCH":
        if hit_rate >= 0.62 and lift >= 0.18 and stable_month_share >= 0.60:
            return "CORE_WATCH"
        if hit_rate >= 0.56 and lift >= 0.12:
            return "RESEARCH_READY"
        if lift >= 0.08:
            return "WATCH"
    if market == "KEY_PASSES_1_5_WATCH":
        if hit_rate >= 0.32 and lift >= 0.14 and stable_month_share >= 0.15:
            return "CORE_WATCH"
        if hit_rate >= 0.28 and lift >= 0.10:
            return "RESEARCH_READY"
        if lift >= 0.06:
            return "WATCH"
    if market == "ASSIST_0_5_WATCH":
        if hit_rate >= 0.16 and lift >= 0.04 and stable_month_share >= 0.60:
            return "RESEARCH_READY"
        if lift >= 0.025:
            return "WATCH"
    return "DO_NOT_USE"


def score_market_cells(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = scored[
        scored["prior_apps"].ge(5)
        & scored["minutes_l5"].ge(35)
        & scored["position_group"].isin(["Midfielder", "Forward", "Defender"])
    ].copy()
    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    for market, (stat_col, threshold, target_rate) in MARKETS.items():
        score_col = "assist_intel_score" if market == "ASSIST_0_5_WATCH" else "key_pass_intel_score"
        eligible[f"{market}_hit"] = num(eligible[stat_col]).ge(threshold).astype(float)
        baseline = float(eligible[f"{market}_hit"].mean()) if not eligible.empty else np.nan
        for score_t in SCORE_THRESHOLDS:
            cell = eligible[eligible[score_col].ge(score_t)].copy()
            if cell.empty:
                continue
            monthly = (
                cell.groupby("match_month", dropna=False)[f"{market}_hit"]
                .agg(rows="size", hits="sum", hit_rate="mean")
                .reset_index()
            )
            stable_month_share = float(monthly.loc[monthly["rows"].ge(15), "hit_rate"].ge(target_rate).mean()) if not monthly.empty else np.nan
            rows.append(
                {
                    "market": market,
                    "cell": f"SCORE_GE_{score_t:.2f}",
                    "rows": int(len(cell)),
                    "hits": int(cell[f"{market}_hit"].sum()),
                    "hit_rate": float(cell[f"{market}_hit"].mean()),
                    "baseline_hit_rate": baseline,
                    "lift": float(cell[f"{market}_hit"].mean() - baseline),
                    "stable_month_share": stable_month_share,
                    "recommended_status": classify_status(
                        market,
                        int(len(cell)),
                        float(cell[f"{market}_hit"].mean()),
                        baseline,
                        float(cell[f"{market}_hit"].mean() - baseline),
                        stable_month_share,
                    ),
                }
            )
            for _, month_row in monthly.iterrows():
                monthly_rows.append(
                    {
                        "market": market,
                        "cell": f"SCORE_GE_{score_t:.2f}",
                        "match_month": month_row["match_month"],
                        "rows": int(month_row["rows"]),
                        "hits": int(month_row["hits"]),
                        "hit_rate": float(month_row["hit_rate"]),
                    }
                )
    summary = pd.DataFrame(rows).sort_values(["market", "cell"]).reset_index(drop=True)
    monthly_summary = pd.DataFrame(monthly_rows)
    return summary, monthly_summary


def score_league_position_cells(scored: pd.DataFrame) -> pd.DataFrame:
    eligible = scored[
        scored["prior_apps"].ge(5)
        & scored["minutes_l5"].ge(35)
        & scored["key_pass_intel_score"].ge(0.75)
        & scored["position_group"].isin(["Midfielder", "Forward", "Defender"])
    ].copy()
    if eligible.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for market, (stat_col, threshold, _target_rate) in MARKETS.items():
        eligible[f"{market}_hit"] = num(eligible[stat_col]).ge(threshold).astype(float)
        for key, group in eligible.groupby(["league", "position_group"], dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            rows.append(
                {
                    "market": market,
                    "league": key[0],
                    "position_group": key[1],
                    "rows": int(len(group)),
                    "hits": int(group[f"{market}_hit"].sum()),
                    "hit_rate": float(group[f"{market}_hit"].mean()),
                    "mean_key_pass_score": float(group["key_pass_intel_score"].mean()),
                    "mean_assist_score": float(group["assist_intel_score"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["market", "hit_rate", "rows"], ascending=[True, False, False]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            work[col] = work[col].astype("string").fillna("")
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in work.columns) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, scored: pd.DataFrame, summary: pd.DataFrame, league_position: pd.DataFrame) -> None:
    lines = [
        "# Player Key Pass / Assist Intelligence Audit",
        "",
        "Research-only audit using lagged API-Football player-event form.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Assist is treated as a watch/intelligence signal because it is low-frequency and teammate-finish dependent.",
        "",
        "## Overall",
        f"- scored rows: `{len(scored)}`",
        f"- leagues: `{scored['league'].nunique() if not scored.empty else 0}`",
        f"- players: `{scored['player_id'].nunique() if not scored.empty else 0}`",
        "",
        "## Market Threshold Cells",
        markdown_table(summary, max_rows=80),
        "",
        "## League / Position Cells at Score >= 0.75",
        markdown_table(league_position, max_rows=100),
        "",
        "## Interpretation",
        "- Key passes are the primary product candidate here.",
        "- Assist watch should be shown as creator involvement / assist threat, not as a strong binary prop.",
        "- Next step is a live shadow board that labels current expected starters with key-pass / assist-watch context.",
    ]
    (outdir / "PLAYER_KEY_PASS_ASSIST_INTELLIGENCE_AUDIT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-root", type=Path, default=DEFAULT_ACTUAL_ROOT)
    parser.add_argument("--season", type=int, action="append", default=[])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    seasons = set(args.season) if args.season else {2022, 2023, 2024}
    raw = load_player_events(args.actual_root, seasons)
    if raw.empty:
        raise SystemExit(f"No player-event actuals found under {args.actual_root}")
    scored = add_scores(add_lagged_form(raw))
    summary, monthly = score_market_cells(scored)
    league_position = score_league_position_cells(scored)

    scored.to_csv(args.outdir / "PLAYER_KEY_PASS_ASSIST_INTELLIGENCE_SCORED_ROWS.csv", index=False)
    summary.to_csv(args.outdir / "PLAYER_KEY_PASS_ASSIST_INTELLIGENCE_THRESHOLD_CELLS.csv", index=False)
    monthly.to_csv(args.outdir / "PLAYER_KEY_PASS_ASSIST_INTELLIGENCE_MONTHLY_STABILITY.csv", index=False)
    league_position.to_csv(args.outdir / "PLAYER_KEY_PASS_ASSIST_INTELLIGENCE_LEAGUE_POSITION_CELLS.csv", index=False)
    write_report(args.outdir, scored, summary, league_position)

    print(f"WROTE {args.outdir}")
    print(f"scored_rows={len(scored)} leagues={scored['league'].nunique()} players={scored['player_id'].nunique()}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
