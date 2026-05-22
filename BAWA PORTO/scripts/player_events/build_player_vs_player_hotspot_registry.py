from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MIN_ROWS = 2


def build_registry(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Player-vs-Player Hotspot Registry\n\nNo rows matched.\n")
        return df

    grouped = (
        df.groupby(
            [
                "review_family",
                "tactical_role",
                "opponent_flank_profile",
                "player_vs_player_matchup_tag",
            ],
            as_index=False,
        )
        .agg(
            rows=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            teams=("team_name", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique())[:8])),
            markets=("market", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique()))),
            avg_market_hit=("market_hit_rate", "mean"),
            avg_role_hit=("role_hit_rate", "mean"),
            avg_score=("score", "mean"),
            manual_override_rows=("manual_side_override_active", "sum") if "manual_side_override_active" in df.columns else ("player_name", lambda s: 0),
        )
        .sort_values(["avg_market_hit", "rows", "avg_score"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    grouped["registry_tier"] = "WATCH"
    grouped.loc[grouped["rows"].ge(MIN_ROWS), "registry_tier"] = "REPEATING"
    grouped.loc[(grouped["rows"].ge(MIN_ROWS)) & (grouped["avg_market_hit"].ge(0.75)), "registry_tier"] = "STRONG_REPEATING"
    grouped.to_csv(output_csv, index=False)

    lines = ["# Player-vs-Player Hotspot Registry", "", f"- rows: {len(grouped)}", f"- minimum repeat threshold: {MIN_ROWS} rows", ""]
    for matchup_tag in [
        "WINGER_VS_FULLBACK_ISOLATION",
        "ADVANCED_8S_VS_HOLDING_MID",
        "STRIKER_VS_FRONT_FOOT_CB",
        "BOX_MIDFIELD_VS_DM_COLLISION",
        "HIGH_8S_VS_SINGLE_PIVOT",
    ]:
        sub = grouped[grouped["player_vs_player_matchup_tag"].eq(matchup_tag)].head(10)
        lines.append(f"## {matchup_tag}")
        if sub.empty:
            lines.append("No rows matched.")
            lines.append("")
            continue
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['review_family']} | {row['tactical_role']} | {row['opponent_flank_profile']}: tier={row['registry_tier']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_market_hit={row['avg_market_hit']:.3f} | markets={row['markets']} | teams={row['teams']}"
            )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a player-vs-player hotspot registry from the team-specific weekend sheet.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_registry(args.input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
