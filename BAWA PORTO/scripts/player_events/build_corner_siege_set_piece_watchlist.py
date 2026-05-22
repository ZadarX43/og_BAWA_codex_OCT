from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _norm_cap(series: pd.Series, cap: float) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    return (out.clip(lower=0.0, upper=cap) / cap).fillna(0.0)


def _string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _profile_label(df: pd.DataFrame) -> pd.Series:
    role = _string(df.get("tactical_role", pd.Series("", index=df.index))).str.lower()
    out = pd.Series("SECONDARY_BOX", index=df.index, dtype="string")
    out = out.mask(role.eq("central striker"), "TARGET_FORWARD")
    out = out.mask(role.eq("wide forward") | role.eq("wide midfielder / winger"), "DELIVERY_WIDE")
    out = out.mask(role.eq("central midfielder"), "SET_PIECE_CREATOR")
    return out


def build_watchlist(input_csv: str, output_csv: str, output_md: str, top_n_per_fixture: int = 4) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = df[df["fixture_attacking_style_label"].astype("string").str.upper().eq("CORNER_SIEGE")].copy()
    if df.empty:
        out = pd.DataFrame()
    else:
        same_side_corners = df.apply(
            lambda r: r.get("home_team_corners_for_l5", 0.0) if str(r.get("player_team_side", "")).upper() == "HOME" else r.get("away_team_corners_for_l5", 0.0),
            axis=1,
        )
        same_side_corners_against = df.apply(
            lambda r: r.get("home_team_corners_against_l5", 0.0) if str(r.get("player_team_side", "")).upper() == "HOME" else r.get("away_team_corners_against_l5", 0.0),
            axis=1,
        )
        role = _string(df.get("tactical_role", pd.Series("", index=df.index))).str.lower()
        role_bonus = (
            role.eq("central striker").astype(float) * 0.18
            + (role.eq("wide forward") | role.eq("wide midfielder / winger")).astype(float) * 0.10
            + role.eq("central midfielder").astype(float) * 0.08
            + role.eq("wide defender / wing-back").astype(float) * 0.04
        )
        df["set_piece_watch_index"] = (
            0.22 * _norm_cap(df.get("fixture_corner_pressure_score", 0.0), 1.0)
            + 0.18 * _norm_cap(same_side_corners, 8.0)
            + 0.10 * _norm_cap(same_side_corners_against, 8.0)
            + 0.12 * _norm_cap(df.get("h2h_total_corners_l5", 0.0), 18.0)
            + 0.12 * _norm_cap(df.get("shots_per90", 0.0), 4.0)
            + 0.10 * _norm_cap(df.get("shots_on_target_per90", 0.0), 2.0)
            + 0.06 * _norm_cap(df.get("goals_per90", 0.0), 1.0)
            + 0.05 * _norm_cap(df.get("assists_per90", 0.0), 0.7)
            + 0.05 * _norm_cap(df.get("key_passes_per90", 0.0), 3.0)
            + role_bonus
        ) * 100.0
        df["set_piece_attack_quality_score"] = (
            0.35 * _norm_cap(df.get("fixture_corner_pressure_score", 0.0), 1.0)
            + 0.25 * _norm_cap(same_side_corners, 8.0)
            + 0.15 * _norm_cap(df.get("h2h_total_corners_l5", 0.0), 18.0)
            + 0.15 * _norm_cap(df.get("og_goal_environment_score", 0.0), 1.0)
            + 0.10 * _norm_cap(df.get("og_battle_on_score", 0.0), 1.0)
        )
        df["set_piece_profile_label"] = _profile_label(df)
        out = (
            df.sort_values(["fixture_key", "set_piece_watch_index"], ascending=[True, False])
            .groupby("fixture_key", as_index=False, group_keys=False)
            .head(top_n_per_fixture)
            .reset_index(drop=True)
        )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    lines = [
        "# CORNER_SIEGE Set-Piece Watchlist",
        "",
        "Note: proxy-only watchlist. We do not have direct headers / aerial shot event data in the normalized layer yet.",
        "",
    ]
    if out.empty:
        lines.append("- no CORNER_SIEGE rows found")
    else:
        for fixture_key, group in out.groupby("fixture_key", sort=False):
            first = group.iloc[0]
            lines.append(f"## {fixture_key}")
            lines.append(f"- {first['home_team_name']} vs {first['away_team_name']} | corner_pressure={first['fixture_corner_pressure_score']:.3f} | set_piece_quality={first['set_piece_attack_quality_score']:.3f}")
            for row in group.itertuples(index=False):
                lines.append(
                    f"- {row.player_name} ({row.team_name}) | {row.set_piece_profile_label} | index={row.set_piece_watch_index:.1f} | shots/90={row.shots_per90} | SOT/90={row.shots_on_target_per90} | goals/90={row.goals_per90} | assists/90={row.assists_per90} | key passes/90={row.key_passes_per90}"
                )
            lines.append("")
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CORNER_SIEGE set-piece watchlist.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--top-n-per-fixture", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_watchlist(args.input, args.output_csv, args.output_md, args.top_n_per_fixture)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
    print(f"fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")


if __name__ == "__main__":
    main()
