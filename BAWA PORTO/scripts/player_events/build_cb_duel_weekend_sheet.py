from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUBTYPE_RANK_BONUS = {
    "DIRECT_TARGET_STRIKER": 8.0,
    "AERIAL_BOX_NINE": 6.5,
    "MOBILE_PRESSING_9": 5.0,
    "CHANNEL_RUNNER_STRIKER": 4.0,
}


def build_sheet(input_csv: str, output_csv: str, output_md: str, max_fixtures: int = 6, max_rows_per_fixture: int = 2) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# CB Duel Weekend Sheet\n\nNo rows matched.\n")
        return df

    cb = df[
        df["tactical_role"].astype(str).eq("Centre-back enforcer")
        & pd.to_numeric(df.get("cb_front_foot_duel_flag", 0), errors="coerce").fillna(0).ge(1)
        & df.get("player_vs_player_matchup_tag", pd.Series("", index=df.index)).astype(str).str.contains("STRIKER_VS_FRONT_FOOT_CB", na=False)
        & df.get("opponent_striker_profile", pd.Series("UNSET", index=df.index)).astype(str).ne("UNSET")
    ].copy()
    if cb.empty:
        cb.to_csv(output_csv, index=False)
        Path(output_md).write_text("# CB Duel Weekend Sheet\n\nNo centre-back duel rows survived the weekend filter.\n")
        return cb

    cb["cb_subtype_rank_bonus"] = (
        cb.get("opponent_striker_profile", pd.Series("UNSET", index=cb.index))
        .astype(str)
        .map(SUBTYPE_RANK_BONUS)
        .fillna(0.0)
    )
    cb["cb_weekend_priority"] = (
        pd.to_numeric(cb.get("score", 0.0), errors="coerce").fillna(0.0)
        + 20.0 * pd.to_numeric(cb.get("market_hit_rate", 0.0), errors="coerce").fillna(0.0)
        + 12.0 * pd.to_numeric(cb.get("role_hit_rate", 0.0), errors="coerce").fillna(0.0)
        + 18.0 * pd.to_numeric(cb.get("cb_duel_pressure_score", 0.0), errors="coerce").fillna(0.0)
        + cb["cb_subtype_rank_bonus"]
    )
    cb["cb_subtype_rank_label"] = (
        cb.get("opponent_striker_profile", pd.Series("UNSET", index=cb.index)).astype(str)
        + " | bonus="
        + cb["cb_subtype_rank_bonus"].map(lambda x: f"{x:.1f}")
    )
    cb = (
        cb.sort_values(
            ["cb_weekend_priority", "cb_subtype_rank_bonus", "cb_duel_pressure_score", "fixture_key", "team_name", "score"],
            ascending=[False, False, False, True, True, False],
        )
        .groupby("fixture_key", group_keys=False)
        .head(max_rows_per_fixture)
        .reset_index(drop=True)
    )
    keep_fixtures = (
        cb.groupby("fixture_key", as_index=False)["cb_weekend_priority"]
        .max()
        .sort_values("cb_weekend_priority", ascending=False)
        .head(max_fixtures)["fixture_key"]
    )
    cb = cb[cb["fixture_key"].isin(keep_fixtures)].copy()
    cb.to_csv(output_csv, index=False)

    lines = ["# CB Duel Weekend Sheet", "", f"- fixtures: {cb['fixture_key'].nunique()} | rows: {len(cb)}", ""]
    lines.append("## Subtype Ranking")
    for profile, sub in cb.groupby("opponent_striker_profile", sort=False):
        lines.append(
            f"- {profile}: rows={len(sub)} | fixtures={sub['fixture_key'].nunique()} | avg_priority={pd.to_numeric(sub['cb_weekend_priority'], errors='coerce').mean():.2f} | subtype_bonus={pd.to_numeric(sub['cb_subtype_rank_bonus'], errors='coerce').mean():.1f}"
        )
    lines.append("")
    for fixture_key, sub in cb.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(f"- {first['home_team_name']} vs {first['away_team_name']} | top_priority={first['cb_weekend_priority']:.2f}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['player_name']} ({row['team_name']}) | {row['market']} | profile={row.get('opponent_striker_profile','UNSET')} | pressure_tag={row.get('opponent_striker_pressure_tag','UNSET')} | cb_duel_pressure={float(row.get('cb_duel_pressure_score', 0.0)):.3f} | subtype_rank={row.get('cb_subtype_rank_label','UNSET')}"
            )
            lines.append(f"  subtype_note={row.get('opponent_striker_subtype_note','UNSET')}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return cb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny weekend sheet from live centre-back duel rows only.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-fixtures", type=int, default=6)
    parser.add_argument("--max-rows-per-fixture", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_sheet(args.input_csv, args.output_csv, args.output_md, args.max_fixtures, args.max_rows_per_fixture)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
