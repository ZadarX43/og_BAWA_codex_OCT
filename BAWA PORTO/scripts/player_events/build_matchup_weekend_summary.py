from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _risk_focus_text(sub: pd.DataFrame) -> str:
    lanes = set(sub["matchup_lane"].astype(str))
    focuses: list[str] = []
    if "DM_SCREEN" in lanes:
        focuses.append("missing DM")
    if "WINGER_ISOLATION" in lanes:
        focuses.append("missing full-back")
    if "CB_DUEL_REVIEW" in lanes:
        focuses.append("missing CB duel anchor")
    return ", ".join(focuses) if focuses else "general lineup recheck"


def build_summary(input_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        Path(output_md).write_text("# Matchup Weekend Summary\n\nNo rows matched.\n")
        return df

    dm = df[df["matchup_lane"].astype(str).eq("DM_SCREEN")].copy()
    winger = df[df["matchup_lane"].astype(str).eq("WINGER_ISOLATION")].copy()
    cb = df[df["matchup_lane"].astype(str).eq("CB_DUEL_REVIEW")].copy()

    lines = [
        "# Matchup Weekend Summary",
        "",
        f"- fixtures: {df['fixture_key'].nunique()} | rows: {len(df)}",
        "",
    ]
    if not dm.empty:
        best = dm.iloc[0]
        lines.append("## Best DM Screen Fixture")
        lines.append(f"- `{best['fixture_key']}` | {best['home_team_name']} vs {best['away_team_name']}")
        lines.append(f"- key name: {best['player_name']} ({best['team_name']}) | {best['market']} | {best['tactical_role']}")
        lines.append(f"- prematch risk: {best.get('prematch_risk_feedback_note', 'UNSET')}")
        lines.append("")
    if not winger.empty:
        best = winger.iloc[0]
        lines.append("## Best Winger Isolation Fixture")
        lines.append(f"- `{best['fixture_key']}` | {best['home_team_name']} vs {best['away_team_name']}")
        lines.append(f"- key name: {best['player_name']} ({best['team_name']}) | {best['market']} | {best['tactical_role']}")
        lines.append(f"- prematch risk: {best.get('prematch_risk_feedback_note', 'UNSET')}")
        lines.append("")
    lines.append("## Centre-Back Duel Status")
    if cb.empty:
        lines.append("- No live striker/front-foot-CB rows survived into the weekend top sheet yet.")
    else:
        best = cb.iloc[0]
        lines.append(
            f"- `{best['fixture_key']}` | {best['player_name']} ({best['team_name']}) | profile={best.get('opponent_striker_profile', 'UNSET')} | pressure_tag={best.get('opponent_striker_pressure_tag', 'UNSET')} | cb_duel_pressure={float(best.get('cb_duel_pressure_score', 0.0)):.3f}"
        )
        lines.append(f"- subtype note: {best.get('opponent_striker_subtype_note', 'UNSET')}")
        lines.append(f"- prematch risk: {best.get('prematch_risk_feedback_note', 'UNSET')}")
    lines.append("")
    lines.append("## Fixture Risk Focus")
    for fixture_key, sub in df.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"- `{fixture_key}` | {first['home_team_name']} vs {first['away_team_name']} | focus={_risk_focus_text(sub)}")
    lines.append("")
    Path(output_md).write_text("\n".join(lines))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny summary markdown from the combined matchup weekend top sheet.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_summary(args.input_csv, args.output_md)
    print(f"WROTE: {args.output_md}")
    print(f"rows: {len(out)}")
