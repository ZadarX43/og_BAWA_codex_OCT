from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _section(df: pd.DataFrame, matchup_tag: str, title: str, trust_note: str) -> list[str]:
    lines = [f"## {title}"]
    sub = df[df["player_vs_player_matchup_tag"].eq(matchup_tag)].copy()
    if sub.empty:
        lines.append("No rows matched.")
        lines.append("")
        return lines
    top = sub.sort_values(["registry_tier", "avg_market_hit", "rows"], ascending=[True, False, False]).head(8)
    lines.append(trust_note)
    for _, row in top.iterrows():
        lines.append(
            f"- {row['review_family']} | {row['tactical_role']} | {row['opponent_flank_profile']}: tier={row['registry_tier']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_market_hit={row['avg_market_hit']:.3f} | markets={row['markets']} | teams={row['teams']}"
        )
    lines.append("")
    return lines


def build_rulebook(registry_csv: str, output_md: str) -> None:
    df = pd.read_csv(registry_csv, low_memory=False)
    lines = ["# Matchup Deploy Rulebook", ""]
    if df.empty:
        lines.append("No hotspot rows matched.")
        Path(output_md).write_text("\n".join(lines) + "\n")
        return
    lines.append("## Operating Rules")
    lines.append("- Use matchup tags only when they have reached at least repeating-sample status in the hotspot registry.")
    lines.append("- Prefer contact deployment when the matchup tag and the team-role recurrence both point in the same direction.")
    lines.append("- Treat card lanes as support unless the same matchup family has repeated card evidence, not just contact spillover.")
    lines.append("")
    lines.extend(_section(
        df,
        "ADVANCED_8S_VS_HOLDING_MID",
        "DM Screen",
        "Deploy when a holding midfielder keeps appearing inside `4231v442` under double-pivot central pressure; strongest for fouls and tackles, with cards as a secondary cascade lane.",
    ))
    lines.extend(_section(
        df,
        "WINGER_VS_FULLBACK_ISOLATION",
        "Winger Isolation",
        "Deploy when a wide defender / wing-back keeps surfacing inside `4231v433`; strongest for tackles, with bookings support when the same flank repeatedly gets isolated.",
    ))
    lines.extend(_section(
        df,
        "BOX_MIDFIELD_VS_DM_COLLISION",
        "Box Midfield Grind",
        "Deploy cautiously: this is still a watch-tier collision lane and needs more repeated central-duel sample before it becomes a trusted preset.",
    ))
    lines.append("## Guardrails")
    lines.append("- If a matchup tag is only `WATCH`, keep it in research and do not treat it like a live deploy family.")
    lines.append("- If the card hit rate is weak but the contact hit rate is strong, keep bookings in the cascade/watch lane rather than the primary lane.")
    lines.append("- Manual side overrides should be applied before trusting flank-based tags on elite fixtures.")
    lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny matchup deploy rulebook from the hotspot registry.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_rulebook(args.registry_csv, args.output_md)
    print(f"WROTE: {args.output_md}")
