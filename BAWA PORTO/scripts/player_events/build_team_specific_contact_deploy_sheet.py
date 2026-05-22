from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CONTACT_MARKETS = {"fouls_committed", "tackles"}


def _risk_focus_for_role(role: str) -> str:
    role_text = str(role or "")
    if role_text == "Holding midfielder":
        return "missing DM"
    if role_text == "Wide defender / wing-back":
        return "missing full-back"
    if role_text == "Centre-back enforcer":
        return "missing CB duel anchor"
    return "no core structural flag"


def build_sheet(input_csv: str, output_csv: str, output_md: str, max_fixtures: int = 10, max_rows_per_fixture: int = 2) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    df = df[df["market"].isin(CONTACT_MARKETS)].copy()
    if df.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Specific Contact Deploy Sheet\n\nNo contact rows matched.\n")
        return df

    df = df[(pd.to_numeric(df["market_hit_rate"], errors="coerce").fillna(0.0) >= 0.5) | (pd.to_numeric(df["role_hit_rate"], errors="coerce").fillna(0.0) >= 0.5)].copy()
    df["contact_deploy_priority"] = (
        pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
        + 18.0 * pd.to_numeric(df["market_hit_rate"], errors="coerce").fillna(0.0)
        + 10.0 * pd.to_numeric(df["role_hit_rate"], errors="coerce").fillna(0.0)
        + 4.0 * pd.to_numeric(df["formation_pressure_score"], errors="coerce").fillna(0.0)
    )
    df = (
        df.sort_values(["fixture_key", "contact_deploy_priority", "score"], ascending=[True, False, False])
        .groupby("fixture_key", group_keys=False)
        .head(max_rows_per_fixture)
        .reset_index(drop=True)
    )
    fixture_rank = (
        df.groupby(["fixture_key", "home_team_name", "away_team_name"], as_index=False)
        .agg(
            fixture_priority=("contact_deploy_priority", "sum"),
            rows=("player_name", "count"),
            best_hit=("market_hit_rate", "max"),
        )
        .sort_values(["fixture_priority", "rows", "best_hit"], ascending=[False, False, False])
        .head(max_fixtures)
    )
    out = df[df["fixture_key"].isin(fixture_rank["fixture_key"])].copy()
    out = out.drop(columns=["fixture_priority"], errors="ignore")
    out = out.merge(fixture_rank[["fixture_key", "fixture_priority"]], on="fixture_key", how="left")
    out["prematch_risk_focus"] = out["tactical_role"].astype(str).map(_risk_focus_for_role)
    out["prematch_risk_note"] = out.apply(
        lambda row: (
            f"If the expected {row['tactical_role'].lower()} changes late, rerun the contact deploy check before trusting this row."
            if row["prematch_risk_focus"] != "no core structural flag"
            else "No core DM/full-back/CB structural flag on this row."
        ),
        axis=1,
    )
    out = out.sort_values(["fixture_priority", "fixture_key", "contact_deploy_priority"], ascending=[False, True, False]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Team-Specific Contact Deploy Sheet", "", f"- fixtures: {out['fixture_key'].nunique()} | rows: {len(out)}", ""]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(f"- {first['home_team_name']} vs {first['away_team_name']} | fixture priority={first['fixture_priority']:.1f}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['player_name']} ({row['team_name']}) | {row['market']} | {row['tactical_role']} | family={row['review_family']} | market_hit={row['market_hit_rate']:.3f} | role_hit={row['role_hit_rate']:.3f} | score={row['score']:.1f}"
            )
            lines.append(f"  opponent_context={row['opponent_flank_profile']} | {row['opponent_role_context_note']}")
            lines.append(f"  prematch_risk_focus={row['prematch_risk_focus']} | {row['prematch_risk_note']}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a contact-only deploy sheet from the team-specific weekend sheet.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-fixtures", type=int, default=10)
    parser.add_argument("--max-rows-per-fixture", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_sheet(args.input_csv, args.output_csv, args.output_md, args.max_fixtures, args.max_rows_per_fixture)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
