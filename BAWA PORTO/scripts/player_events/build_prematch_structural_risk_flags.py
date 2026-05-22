from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _focuses(sub: pd.DataFrame) -> list[str]:
    roles = set(sub.get("tactical_role", pd.Series("", index=sub.index)).astype(str))
    out: list[str] = []
    if "Holding midfielder" in roles:
        out.append("missing DM")
    if "Wide defender / wing-back" in roles:
        out.append("missing full-back")
    if "Centre-back enforcer" in roles:
        out.append("missing CB duel anchor")
    return out


def build_flags(master_csv: str, team_csv: str, matchup_csv: str, output_md: str) -> None:
    boards = []
    for path, label in [
        (master_csv, "MASTER_WEEKEND"),
        (team_csv, "TEAM_SPECIFIC"),
        (matchup_csv, "MATCHUP_TOP_SHEET"),
    ]:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            if not df.empty:
                df = df.copy()
                df["source_board_label"] = label
                boards.append(df)
    combined = pd.concat(boards, ignore_index=True) if boards else pd.DataFrame()

    lines = ["# Pre-Match Structural Risk Flags", ""]
    if combined.empty:
        lines.append("No rows matched.")
        Path(output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(output_md).write_text("\n".join(lines) + "\n")
        return

    lines.append("- Beta review only: use these as pre-kickoff structure checks, not automatic market overrides.")
    lines.append("")

    fixture_rank = []
    for fixture_key, sub in combined.groupby("fixture_key", sort=False):
        focuses = _focuses(sub)
        if not focuses:
            continue
        fixture_rank.append(
            {
                "fixture_key": fixture_key,
                "home_team_name": sub.iloc[0]["home_team_name"],
                "away_team_name": sub.iloc[0]["away_team_name"],
                "focuses": focuses,
                "rows": len(sub),
                "top_score": pd.to_numeric(sub.get("summary_priority", sub.get("fixture_priority", sub.get("market_score", 0.0))), errors="coerce").fillna(0.0).max(),
            }
        )
    fixture_df = pd.DataFrame(fixture_rank).sort_values(["top_score", "rows"], ascending=[False, False]) if fixture_rank else pd.DataFrame()

    if fixture_df.empty:
        lines.append("No structural risk fixtures surfaced.")
    else:
        for _, row in fixture_df.iterrows():
            sub = combined[combined["fixture_key"] == row["fixture_key"]].copy()
            lines.append(f"## {row['fixture_key']}")
            lines.append(f"- {row['home_team_name']} vs {row['away_team_name']} | focus={', '.join(row['focuses'])}")
            focus_rows = sub[
                sub.get("tactical_role", pd.Series("", index=sub.index)).astype(str).isin(
                    ["Holding midfielder", "Wide defender / wing-back", "Centre-back enforcer"]
                )
            ].copy()
            for _, r in focus_rows.head(6).iterrows():
                lines.append(
                    f"- {r['player_name']} ({r['team_name']}) | {r['market']} | {r['tactical_role']} | board={r.get('source_board_label','UNSET')}"
                )
            lines.append("")

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny pre-match structural risk flags markdown across the broader player-market sheets.")
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--team-csv", required=True)
    parser.add_argument("--matchup-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_flags(args.master_csv, args.team_csv, args.matchup_csv, args.output_md)
    print(f"WROTE: {args.output_md}")
