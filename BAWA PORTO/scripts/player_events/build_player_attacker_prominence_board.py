#!/usr/bin/env python3
"""Rank attacker prominence for player shots/SOT intelligence.

Research-only board for strikers, centre-forwards, wide forwards, and wingers.
It combines current live shadow labels with historical repeatability/top-slice
evidence where available. No priced odds or deploy artifacts are written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COPY_PACK = ROOT / "reports" / "2026-05-07" / "player_event_dashboard_copy_pack" / "PLAYER_EVENT_DASHBOARD_COPY_PACK.csv"
DEFAULT_REPEATABILITY = (
    ROOT / "reports" / "2026-05-06" / "player_event_threshold_stability_audit" / "player_event_player_repeatability.csv"
)
DEFAULT_TOP_SLICES = (
    ROOT / "reports" / "2026-05-06" / "player_event_threshold_stability_audit" / "player_event_top_slice_stability.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_attacker_prominence_board"

ATTACK_ROLE_TERMS = ("striker", "centre", "center", "forward", "winger")
ATTACK_MARKETS = {"Player Shots 1.5+", "Player Shots 2.5+", "Player SOT 0.5+"}


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def is_attacker(row: pd.Series) -> bool:
    text = f"{row.get('position_group', '')} {row.get('tactical_role', '')}".lower()
    return any(term in text for term in ATTACK_ROLE_TERMS)


def score_row(row: pd.Series) -> float:
    score = 0.0
    if row.get("product_confidence_label") == "CORE WATCH":
        score += 3.0
    if row.get("product_confidence_label") == "FORM + MATCHUP":
        score += 2.0
    if "ROLE EDGE" in str(row.get("product_context_label", "")):
        score += 1.0
    if "LINEUP WATCH" not in str(row.get("product_context_label", "")):
        score += 0.5
    prob_text = str(row.get("display_probability_band", "")).replace("%", "")
    try:
        score += float(prob_text) / 100.0
    except ValueError:
        pass
    return score


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-pack", type=Path, default=DEFAULT_COPY_PACK)
    parser.add_argument("--repeatability", type=Path, default=DEFAULT_REPEATABILITY)
    parser.add_argument("--top-slices", type=Path, default=DEFAULT_TOP_SLICES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.copy_pack.exists():
        raise SystemExit(f"Missing copy pack: {args.copy_pack}")
    current = pd.read_csv(args.copy_pack, low_memory=False)
    current = current[current["market_display"].isin(ATTACK_MARKETS)].copy()
    current = current[current.apply(is_attacker, axis=1)].copy()
    current["prominence_points"] = current.apply(score_row, axis=1)

    repeatability = pd.read_csv(args.repeatability, low_memory=False) if args.repeatability.exists() else pd.DataFrame()
    if not repeatability.empty:
        repeatability = repeatability[repeatability["market"].astype(str).isin(["shots", "shots_ge2", "shots_ge3", "shots_on_target"])].copy()
        repeatability_summary = (
            repeatability.groupby(["team_name", "player_name"], dropna=False)
            .agg(
                repeatability_rows=("rows", "sum"),
                repeatability_best_hit_rate=("hit_rate", "max"),
                repeatability_markets=("market", lambda s: "|".join(sorted(set(map(str, s))))),
            )
            .reset_index()
        )
    else:
        repeatability_summary = pd.DataFrame(columns=["team_name", "player_name"])

    top_slices = pd.read_csv(args.top_slices, low_memory=False) if args.top_slices.exists() else pd.DataFrame()
    if not top_slices.empty:
        top_slices = top_slices[top_slices["market"].astype(str).isin(["shots", "shots_ge2", "shots_ge3", "shots_on_target"])].copy()
        top_slice_summary = (
            top_slices.groupby("market", dropna=False)
            .agg(best_top_slice_hit_rate=("hit_rate", "max"), best_stable_month_share=("stable_month_share", "max"))
            .reset_index()
        )
    else:
        top_slice_summary = pd.DataFrame()

    board = (
        current.groupby(
            [
                "match_date",
                "league",
                "fixture_key",
                "home_team_name",
                "away_team_name",
                "team_name",
                "player_name",
                "position_group",
                "tactical_role",
            ],
            dropna=False,
        )
        .agg(
            live_watch_rows=("market_display", "size"),
            live_markets=("market_display", lambda s: " | ".join(sorted(set(map(str, s))))),
            best_product_label=("product_confidence_label", lambda s: "CORE WATCH" if "CORE WATCH" in set(s) else "FORM + MATCHUP"),
            context_labels=("product_context_label", lambda s: " | ".join(sorted(set(map(str, s))))),
            prominence_score=("prominence_points", "sum"),
            min_expected_minutes=("expected_minutes", "min"),
        )
        .reset_index()
    )
    if not repeatability_summary.empty:
        board = board.merge(repeatability_summary, on=["team_name", "player_name"], how="left")
    board["repeatability_rows"] = num(board.get("repeatability_rows", pd.Series(np.nan, index=board.index))).fillna(0)
    board["repeatability_best_hit_rate"] = num(board.get("repeatability_best_hit_rate", pd.Series(np.nan, index=board.index)))
    board["prominence_score"] = board["prominence_score"] + (board["repeatability_rows"].clip(upper=50) / 50.0)
    board["prominence_band"] = np.select(
        [
            board["prominence_score"].ge(7.0),
            board["prominence_score"].ge(4.5),
            board["prominence_score"].ge(2.5),
        ],
        ["ATTACKER_CORE", "ATTACKER_STRONG", "ATTACKER_WATCH"],
        default="ATTACKER_INFO",
    )
    board = board.sort_values(["prominence_score", "live_watch_rows"], ascending=[False, False])

    board.to_csv(args.outdir / "PLAYER_ATTACKER_PROMINENCE_BOARD.csv", index=False)
    top_slice_summary.to_csv(args.outdir / "PLAYER_ATTACKER_PROMINENCE_TOP_SLICE_REFERENCE.csv", index=False)
    lines = [
        "# Player Attacker Prominence Board",
        "",
        "Research-only board for attacker shots/SOT prominence.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "",
        "## Overall",
        f"- attacker rows: `{len(board)}`",
        f"- fixtures: `{board['fixture_key'].nunique() if not board.empty else 0}`",
        f"- players: `{board['player_name'].nunique() if not board.empty else 0}`",
        "",
        "## Prominence Bands",
        markdown_table(board.groupby("prominence_band", dropna=False).size().reset_index(name="rows")),
        "",
        "## Top Attackers",
        markdown_table(
            board[
                [
                    "match_date",
                    "league",
                    "player_name",
                    "team_name",
                    "live_markets",
                    "prominence_band",
                    "prominence_score",
                    "repeatability_rows",
                    "repeatability_best_hit_rate",
                ]
            ],
            max_rows=40,
        ),
    ]
    (args.outdir / "PLAYER_ATTACKER_PROMINENCE_BOARD.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE {args.outdir}")
    print(f"attacker_rows={len(board)}")


if __name__ == "__main__":
    main()
