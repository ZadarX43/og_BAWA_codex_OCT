from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FLANK_TAGS = {"WINGER_VS_FULLBACK_ISOLATION", "WINGER_SWITCH_VS_FULLBACK"}


def build_helper(shortlist_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(shortlist_csv, low_memory=False)
    if df.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Manual-Side Hotspot Helper\n\nNo shortlist rows matched.\n")
        return df
    flank = df[df["player_vs_player_matchup_tag"].isin(FLANK_TAGS)].copy()
    flank = flank[(flank["manual_side_override_active"].fillna(0).astype(int) == 0)].copy()
    if flank.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        flank.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Manual-Side Hotspot Helper\n\nNo shortlist fixtures currently need manual side confirmation.\n")
        return flank
    flank["override_priority"] = (
        pd.to_numeric(flank["cascade_strength"], errors="coerce").fillna(0.0)
        + 15.0 * pd.to_numeric(flank["market_hit_rate"], errors="coerce").fillna(0.0)
        + 8.0 * pd.to_numeric(flank["role_hit_rate"], errors="coerce").fillna(0.0)
    )
    out = flank.sort_values(["override_priority", "fixture_key", "team_name"], ascending=[False, True, True]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    lines = ["# Manual-Side Hotspot Helper", "", f"- fixtures: {out['fixture_key'].nunique()} | rows: {len(out)}", ""]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(f"- {first['home_team_name']} vs {first['away_team_name']} | cascade_strength={first['cascade_strength']:.1f}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['player_name']} ({row['team_name']}) | {row['market']} | {row['tactical_role']} | matchup={row['player_vs_player_matchup_tag']} | override_priority={row['override_priority']:.1f}"
            )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List shortlist hotspot fixtures that would benefit most from manual side confirmation.")
    parser.add_argument("--shortlist-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_helper(args.shortlist_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
