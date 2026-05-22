from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CONTACT_MARKETS = {"fouls_committed", "tackles"}


def build_board(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Specific Contact + Bookings Cascade Board\n\nNo rows matched.\n")
        return df

    team_meta = (
        df.groupby(["fixture_key", "team_name"], as_index=False)
        .agg(
            contact_rows=("market", lambda s: int(pd.Series(s).isin(CONTACT_MARKETS).sum())),
            booking_rows=("market", lambda s: int((pd.Series(s) == "yellow_cards").sum())),
            best_market_hit=("market_hit_rate", "max"),
            best_role_hit=("role_hit_rate", "max"),
            cascade_priority=("team_specific_priority", "sum"),
        )
    )
    team_meta = team_meta[(team_meta["contact_rows"] >= 1) & (team_meta["booking_rows"] >= 1)].copy()
    if team_meta.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        team_meta.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Specific Contact + Bookings Cascade Board\n\nNo team-specific cascade rows matched.\n")
        return team_meta

    team_meta["cascade_strength"] = (
        pd.to_numeric(team_meta["cascade_priority"], errors="coerce").fillna(0.0)
        + 20.0 * pd.to_numeric(team_meta["best_market_hit"], errors="coerce").fillna(0.0)
        + 10.0 * pd.to_numeric(team_meta["best_role_hit"], errors="coerce").fillna(0.0)
    )
    out = df.merge(team_meta[["fixture_key", "team_name", "contact_rows", "booking_rows", "cascade_strength"]], on=["fixture_key", "team_name"], how="inner")
    if "manual_side_override_active" in out.columns:
        out["manual_side_override_active"] = pd.to_numeric(out["manual_side_override_active"], errors="coerce").fillna(0).astype(int)
    out = out.sort_values(["cascade_strength", "fixture_key", "team_name", "team_specific_priority"], ascending=[False, True, True, False]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Team-Specific Contact + Bookings Cascade Board", "", f"- fixtures: {out['fixture_key'].nunique()} | teams: {out[['fixture_key','team_name']].drop_duplicates().shape[0]} | rows: {len(out)}", ""]
    for (fixture_key, team_name), sub in out.groupby(["fixture_key", "team_name"], sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key} | {team_name}")
        lines.append(
            f"- {first['home_team_name']} vs {first['away_team_name']} | contact_rows={int(first['contact_rows'])} | booking_rows={int(first['booking_rows'])} | cascade_strength={first['cascade_strength']:.1f}"
        )
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['player_name']} | {row['market']} | {row['tactical_role']} | family={row['review_family']} | sample={row['sample_bucket']} | market_hit={row['market_hit_rate']:.3f} | role_hit={row['role_hit_rate']:.3f}"
            )
            lines.append(f"  opponent_context={row['opponent_flank_profile']} | {row['opponent_role_context_note']}")
            lines.append(f"  matchup_tag={row['player_vs_player_matchup_tag']} | {row['player_vs_player_matchup_note']}")
            if str(row.get("opponent_striker_profile", "UNSET")) != "UNSET":
                lines.append(
                    f"  striker_profile={row['opponent_striker_profile']} | pressure_tag={row.get('opponent_striker_pressure_tag','UNSET')} | cb_duel_pressure={float(row.get('cb_duel_pressure_score', 0.0)):.3f}"
                )
            if "manual_side_override_active" in row.index and int(row["manual_side_override_active"]) == 1:
                lines.append(
                    f"  manual_override=YES | pitch_side={row.get('manual_pitch_side', 'UNSET')} | overload_target={row.get('manual_overload_target_side', 'UNSET')}"
                )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a team-specific contact + bookings cascade board from the weekend sheet.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_board(args.input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
