from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONFIDENCE_MAP = {
    "LOW": 0.35,
    "MEDIUM": 0.55,
    "HIGH": 0.75,
}


def _risk_focus_for_role(role: str) -> str:
    role_text = str(role or "")
    if role_text == "Holding midfielder":
        return "missing DM"
    if role_text == "Wide defender / wing-back":
        return "missing full-back"
    if role_text == "Centre-back enforcer":
        return "missing CB duel anchor"
    return "no core structural flag"


def _normalize_board(df: pd.DataFrame, board_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["board_name"] = board_name
    if "review_family" not in out.columns:
        out["review_family"] = out.get("source_family", "UNSET")
    if "score" not in out.columns:
        out["score"] = pd.to_numeric(out.get("market_score", 0.0), errors="coerce").fillna(0.0)
    if "market_hit_rate" not in out.columns:
        out["market_hit_rate"] = (
            out.get("market_confidence", pd.Series("", index=out.index))
            .astype(str)
            .str.upper()
            .map(CONFIDENCE_MAP)
            .fillna(0.0)
        )
    if "role_hit_rate" not in out.columns:
        out["role_hit_rate"] = out["market_hit_rate"]
    if "fixture_family" not in out.columns:
        out["fixture_family"] = out.get("formation_matchup_label", "UNSET")
    if "opponent_striker_profile" not in out.columns:
        out["opponent_striker_profile"] = "UNSET"
    out["prematch_risk_focus"] = out.get("tactical_role", pd.Series("UNSET", index=out.index)).astype(str).map(_risk_focus_for_role)
    return out


def build_scaffold(master_csv: str, bookings_csv: str, team_weekend_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    master = _normalize_board(pd.read_csv(master_csv, low_memory=False), "MASTER_SPECIALIST")
    bookings = _normalize_board(pd.read_csv(bookings_csv, low_memory=False), "BOOKINGS_SUPER_ELITE")
    team_weekend = _normalize_board(pd.read_csv(team_weekend_csv, low_memory=False), "TEAM_SPECIFIC_WEEKEND")

    combined = pd.concat([df for df in [master, bookings, team_weekend] if not df.empty], ignore_index=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if combined.empty:
        combined.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Player Market 3Y Walkforward Scaffold\n\nNo rows matched.\n")
        return combined

    combined["match_date"] = pd.to_datetime(combined.get("match_date"), errors="coerce")
    audit = (
        combined.groupby(
            ["market", "review_family", "tactical_role", "fixture_family", "prematch_risk_focus"],
            dropna=False,
        )
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", pd.Series.nunique),
            teams=("team_name", pd.Series.nunique),
            boards=("board_name", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique()))),
            avg_market_hit_rate=("market_hit_rate", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_role_hit_rate=("role_hit_rate", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_score=("score", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            date_from=("match_date", "min"),
            date_to=("match_date", "max"),
        )
        .reset_index()
        .sort_values(["avg_market_hit_rate", "avg_role_hit_rate", "rows"], ascending=[False, False, False])
    )
    audit.to_csv(output_csv, index=False)

    cb = combined[
        combined.get("tactical_role", pd.Series("", index=combined.index)).astype(str).eq("Centre-back enforcer")
        & combined.get("opponent_striker_profile", pd.Series("UNSET", index=combined.index)).astype(str).ne("UNSET")
    ].copy()
    cb_sub = pd.DataFrame()
    if not cb.empty:
        cb_sub = (
            cb.groupby(["opponent_striker_profile", "market", "review_family"], dropna=False)
            .agg(
                rows=("fixture_key", "size"),
                fixtures=("fixture_key", pd.Series.nunique),
                avg_market_hit_rate=("market_hit_rate", lambda s: pd.to_numeric(s, errors="coerce").mean()),
                avg_score=("score", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            )
            .reset_index()
            .sort_values(["avg_market_hit_rate", "avg_score"], ascending=[False, False])
        )

    lines = [
        "# Player Market 3Y Walkforward Scaffold",
        "",
        "- Beta scaffold only: this is the first generalized walkforward shape across the player-market system, not a final backtest.",
        f"- total_rows={len(combined)} | total_fixtures={combined['fixture_key'].nunique()} | markets={combined['market'].astype(str).nunique()}",
        "",
        "## Core Grid",
    ]
    for market, sub in audit.groupby("market", sort=False):
        lines.append(f"### {market}")
        for _, row in sub.head(8).iterrows():
            date_from = pd.to_datetime(row["date_from"]).date() if pd.notna(row["date_from"]) else "NA"
            date_to = pd.to_datetime(row["date_to"]).date() if pd.notna(row["date_to"]) else "NA"
            lines.append(
                f"- {row['review_family']} | {row['tactical_role']} | {row['fixture_family']} | risk={row['prematch_risk_focus']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={row['avg_market_hit_rate']:.3f} | role_hit={row['avg_role_hit_rate']:.3f} | avg_score={row['avg_score']:.2f} | boards={row['boards']} | window={date_from}->{date_to}"
            )
        lines.append("")

    lines.append("## CB Subtype Appendix")
    if cb_sub.empty:
        lines.append("- No centre-back subtype rows matched.")
    else:
        for subtype, sub in cb_sub.groupby("opponent_striker_profile", sort=False):
            lines.append(f"### {subtype}")
            for _, row in sub.iterrows():
                lines.append(
                    f"- {row['market']} | {row['review_family']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={row['avg_market_hit_rate']:.3f} | avg_score={row['avg_score']:.2f}"
                )
            lines.append("")

    Path(output_md).write_text("\n".join(lines) + "\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a first generalized player-market 3-year walkforward scaffold, with a CB subtype appendix.")
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--bookings-csv", required=True)
    parser.add_argument("--team-weekend-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_scaffold(args.master_csv, args.bookings_csv, args.team_weekend_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
