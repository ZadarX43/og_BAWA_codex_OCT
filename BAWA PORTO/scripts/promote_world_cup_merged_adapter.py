#!/usr/bin/env python3
"""QA and promote the World Cup research matrix to canonical merged format.

This is a World Cup-specific adapter, not the normal FootyStats drop ingest.
It writes `Matches/__merged__/World_Cup__merged.csv` only after validating that
the promoted rows have usable priced market coverage and no duplicate fixtures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data_sources/footystats_world_cup/model_matrix/world_cup_research_model_matrix.csv")
DEFAULT_OUTPUT = Path("Matches/__merged__/World_Cup__merged.csv")
DEFAULT_REPORT_DIR = Path("reports/latest/world_cup_merged_adapter_qa_2026_05_19")

FTR_ODDS = ["odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win"]
BTTS_ODDS = ["odds_btts_yes", "odds_btts_no"]
OU25_ODDS = ["odds_ft_over25", "odds_ft_under25"]
BASE_REQUIRED = ["fixture_key", "match_date", "home_team_name", "away_team_name", "status"]
LABEL_COLUMNS = ["home_team_goal_count", "away_team_goal_count"]


def valid_odds(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").where(lambda s: s > 1.01)


def synth_under25_from_over25(over25: pd.Series) -> pd.Series:
    over = pd.to_numeric(over25, errors="coerce")
    p_over = 1.0 / over
    p_under = 1.0 - p_over
    p_under = p_under.where((p_under > 0.01) & (p_under < 0.99))
    return 1.0 / p_under


def label_ftr(row: pd.Series) -> str:
    h = pd.to_numeric(row.get("home_team_goal_count"), errors="coerce")
    a = pd.to_numeric(row.get("away_team_goal_count"), errors="coerce")
    if pd.isna(h) or pd.isna(a):
        return ""
    if h > a:
        return "HOME"
    if a > h:
        return "AWAY"
    return "DRAW"


def prep_matrix(src: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = src.copy()
    df["league"] = "World Cup"
    df["league_tag"] = "World_Cup"
    df["status"] = df["status"].fillna("complete")
    df["match_date"] = pd.to_datetime(df.get("match_date", df.get("kickoff_dt")), errors="coerce").dt.date.astype(str)

    for col in FTR_ODDS + BTTS_ODDS + ["odds_ft_over25", "odds_ft_over15"]:
        if col in df.columns:
            df[col] = valid_odds(df[col])

    df["odds_ft_under25"] = synth_under25_from_over25(df["odds_ft_over25"])
    df["odds_ft_under25_source"] = np.where(
        df["odds_ft_under25"].notna(),
        "SYNTH_FROM_OVER25_COMPLEMENT_RESEARCH",
        "",
    )

    df["world_cup_training_scope"] = np.where(
        df[FTR_ODDS + BTTS_ODDS + OU25_ODDS].notna().all(axis=1),
        "PRICED_MARKET_READY",
        "UNPRICED_RESEARCH_ONLY",
    )
    df["actual_ftr_label"] = df.apply(label_ftr, axis=1)
    df["actual_btts_label"] = np.where(
        (pd.to_numeric(df["home_team_goal_count"], errors="coerce") > 0)
        & (pd.to_numeric(df["away_team_goal_count"], errors="coerce") > 0),
        1,
        0,
    )
    df["actual_over25_label"] = np.where(pd.to_numeric(df["total_goal_count"], errors="coerce") > 2.5, 1, 0)
    promoted = df[df["world_cup_training_scope"].eq("PRICED_MARKET_READY")].copy()
    return df, promoted


def qa_table(full: pd.DataFrame, promoted: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season, group in full.groupby("season", dropna=False):
        promo = promoted[promoted["season"].eq(season)]
        rows.append(
            {
                "season": int(season),
                "source_rows": len(group),
                "promoted_rows": len(promo),
                "priced_ready_rate": float(len(promo) / len(group)) if len(group) else 0.0,
                "ftr_valid_rows": int(group[FTR_ODDS].notna().all(axis=1).sum()),
                "btts_valid_rows": int(group[BTTS_ODDS].notna().all(axis=1).sum()),
                "ou25_valid_rows": int(group[OU25_ODDS].notna().all(axis=1).sum()) if all(c in group.columns for c in OU25_ODDS) else 0,
                "api_join_rate": float(pd.to_numeric(group.get("api_fixture_joined_flag", 0), errors="coerce").fillna(0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("season")


def validate(promoted: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if promoted.empty:
        problems.append("No promoted rows after valid-odds filter.")
        return problems
    for col in BASE_REQUIRED + LABEL_COLUMNS + FTR_ODDS + BTTS_ODDS + OU25_ODDS:
        if col not in promoted.columns:
            problems.append(f"Missing required column: {col}")
        elif promoted[col].isna().any():
            problems.append(f"Required column has nulls: {col}")
    duplicated = int(promoted["fixture_key"].duplicated().sum()) if "fixture_key" in promoted.columns else 0
    if duplicated:
        problems.append(f"Duplicate fixture_key rows: {duplicated}")
    if len(promoted) < 100:
        problems.append(f"Promoted rows too low for useful trainer smoke: {len(promoted)} < 100")
    invalid_status = ~promoted["status"].astype(str).str.lower().isin({"complete", "ft", "finished", "match finished"})
    if invalid_status.any():
        problems.append(f"Non-complete status rows: {int(invalid_status.sum())}")
    return problems


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            view[col] = view[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in view.columns) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--no-write", action="store_true", help="Run QA but do not write the merged output.")
    args = parser.parse_args()

    src = pd.read_csv(args.input, low_memory=False)
    full, promoted = prep_matrix(src)
    qa = qa_table(full, promoted)
    problems = validate(promoted)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    qa.to_csv(report_dir / "world_cup_merged_adapter_qa_by_season.csv", index=False)
    promoted.to_csv(report_dir / "World_Cup__merged_candidate.csv", index=False)
    excluded = full[~full.index.isin(promoted.index)].copy()
    excluded.to_csv(report_dir / "world_cup_unpriced_research_only_rows.csv", index=False)

    output = Path(args.output)
    wrote = False
    if not problems and not args.no_write:
        output.parent.mkdir(parents=True, exist_ok=True)
        promoted.to_csv(output, index=False)
        wrote = True

    summary = [
        "# World Cup Merged Adapter QA",
        "",
        f"Input: `{args.input}`",
        f"Output: `{output}`",
        f"Wrote merged output: `{wrote}`",
        "",
        "## QA By Season",
        markdown_table(qa),
        "",
        "## Gate Result",
        "`PASS`" if not problems else "`FAIL`",
        "",
    ]
    if problems:
        summary.extend(["Problems:"] + [f"- {problem}" for problem in problems] + [""])
    summary.extend(
        [
            "## Policy",
            "- Promoted rows require valid FTR, BTTS, Over 2.5, and synthesized Under 2.5 odds.",
            "- `odds_ft_under25` is research-only synthetic odds from the Over 2.5 complement because FootyStats World Cup exports do not include an under-2.5 price.",
            "- 2006, 2010, and 2014 remain available as unpriced research-only rows but are excluded from the canonical merged trainer input.",
            "- No player/team tournament aggregate performance columns are promoted.",
            "",
            "## Outputs",
            f"- `{report_dir / 'world_cup_merged_adapter_qa_by_season.csv'}`",
            f"- `{report_dir / 'World_Cup__merged_candidate.csv'}`",
            f"- `{report_dir / 'world_cup_unpriced_research_only_rows.csv'}`",
        ]
    )
    (report_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Gate: {'PASS' if not problems else 'FAIL'}")
    print(f"Promoted rows: {len(promoted)} / {len(full)}")
    print(f"Wrote merged output: {wrote}")
    print(f"Report: {report_dir}")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
