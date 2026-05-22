from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _risk_focus_for_role(role: str) -> str:
    role_text = str(role or "")
    if role_text == "Holding midfielder":
        return "missing DM"
    if role_text == "Wide defender / wing-back":
        return "missing full-back"
    if role_text == "Centre-back enforcer":
        return "missing CB duel anchor"
    return "no core structural flag"


def build_shortlist(input_csv: str, output_csv: str, output_md: str, max_fixtures: int = 8) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Specific Contact + Bookings Weekend Shortlist\n\nNo rows matched.\n")
        return df

    fixture_rank = (
        df.groupby(["fixture_key", "home_team_name", "away_team_name", "team_name"], as_index=False)
        .agg(
            cascade_strength=("cascade_strength", "max"),
            rows=("player_name", "count"),
            best_market_hit=("market_hit_rate", "max"),
            manual_override_rows=("manual_side_override_active", "sum") if "manual_side_override_active" in df.columns else ("player_name", lambda s: 0),
        )
        .sort_values(["cascade_strength", "best_market_hit", "rows"], ascending=[False, False, False])
        .head(max_fixtures)
    )
    out = df.merge(fixture_rank[["fixture_key", "team_name", "cascade_strength"]], on=["fixture_key", "team_name", "cascade_strength"], how="inner")
    out["prematch_risk_focus"] = out["tactical_role"].astype(str).map(_risk_focus_for_role)
    out["prematch_risk_note"] = out.apply(
        lambda row: (
            f"If the expected {row['tactical_role'].lower()} changes late, rerun the cascade board before trusting this shortlist row."
            if row["prematch_risk_focus"] != "no core structural flag"
            else "No core DM/full-back/CB structural flag on this row."
        ),
        axis=1,
    )
    out = out.sort_values(["cascade_strength", "fixture_key", "team_name", "team_specific_priority"], ascending=[False, True, True, False]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Team-Specific Contact + Bookings Weekend Shortlist", "", f"- fixtures: {out['fixture_key'].nunique()} | team-sides: {out[['fixture_key','team_name']].drop_duplicates().shape[0]} | rows: {len(out)}", ""]
    for (fixture_key, team_name), sub in out.groupby(["fixture_key", "team_name"], sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key} | {team_name}")
        lines.append(f"- {first['home_team_name']} vs {first['away_team_name']} | cascade_strength={first['cascade_strength']:.1f}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['player_name']} | {row['market']} | {row['tactical_role']} | family={row['review_family']} | sample={row['sample_bucket']} | market_hit={row['market_hit_rate']:.3f} | role_hit={row['role_hit_rate']:.3f}"
            )
            lines.append(f"  opponent_context={row['opponent_flank_profile']} | {row['opponent_role_context_note']}")
            lines.append(f"  matchup_tag={row['player_vs_player_matchup_tag']} | {row['player_vs_player_matchup_note']}")
            lines.append(f"  prematch_risk_focus={row['prematch_risk_focus']} | {row['prematch_risk_note']}")
            if str(row.get("opponent_striker_profile", "UNSET")) != "UNSET":
                lines.append(
                    f"  striker_profile={row['opponent_striker_profile']} | pressure_tag={row.get('opponent_striker_pressure_tag','UNSET')} | cb_duel_pressure={float(row.get('cb_duel_pressure_score', 0.0)):.3f}"
                )
            if "manual_side_override_active" in row.index and int(row['manual_side_override_active']) == 1:
                lines.append(f"  manual_override=YES | pitch_side={row.get('manual_pitch_side','UNSET')} | overload_target={row.get('manual_overload_target_side','UNSET')}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim the team-specific contact+bookings cascade board into a top-fixture weekend shortlist.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-fixtures", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_shortlist(args.input_csv, args.output_csv, args.output_md, args.max_fixtures)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
