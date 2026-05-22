from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_rulebook(cb_csv: str, registry_csv: str, output_md: str) -> None:
    cb = pd.read_csv(cb_csv, low_memory=False)
    registry = pd.read_csv(registry_csv, low_memory=False) if Path(registry_csv).exists() else pd.DataFrame()

    lines = ["# CB Duel Deploy Rulebook", ""]
    if cb.empty:
        lines.append("No centre-back duel rows matched.")
        Path(output_md).write_text("\n".join(lines) + "\n")
        return

    lines.append(f"- live fixtures: {cb['fixture_key'].nunique()} | live rows: {len(cb)}")
    lines.append("")
    lines.append("## Core Rule")
    lines.append("- Trust the CB duel lane only when `Centre-back enforcer` rows survive with a non-`UNSET` striker profile and `cb_front_foot_duel_flag = 1`.")
    lines.append("- Prefer `fouls_committed` and `tackles`; treat bookings as crossover support, not the lead market.")
    lines.append("")

    lines.append("## Pressure Tags")
    for pressure_tag, sub in cb.groupby("opponent_striker_pressure_tag", sort=False):
        first = sub.iloc[0]
        lines.append(
            f"- `{pressure_tag}`: fixtures={sub['fixture_key'].nunique()} | avg_cb_duel_pressure={pd.to_numeric(sub['cb_duel_pressure_score'], errors='coerce').mean():.3f} | lead_profile={first['opponent_striker_profile']}"
        )
        lines.append(f"  subtype_note={first.get('opponent_striker_subtype_note', 'UNSET')}")
    lines.append("")

    lines.append("## Best Live Fixtures")
    ranked = cb.sort_values(["cb_weekend_priority", "cb_duel_pressure_score", "score"], ascending=[False, False, False]) if "cb_weekend_priority" in cb.columns else cb
    for fixture_key, sub in ranked.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"- `{fixture_key}` | {first['home_team_name']} vs {first['away_team_name']} | profile={first['opponent_striker_profile']} | pressure_tag={first['opponent_striker_pressure_tag']}")
        lines.append(f"  subtype_note={first.get('opponent_striker_subtype_note', 'UNSET')}")
    lines.append("")

    if not registry.empty:
        reg = registry[registry["player_vs_player_matchup_tag"].astype(str).eq("STRIKER_VS_FRONT_FOOT_CB")].copy()
        lines.append("## Registry Backing")
        if reg.empty:
            lines.append("- No registry rows yet.")
        else:
            for _, row in reg.sort_values(["avg_market_hit", "rows"], ascending=[False, False]).iterrows():
                lines.append(
                    f"- {row['review_family']} | {row['tactical_role']} | {row['opponent_flank_profile']}: tier={row['registry_tier']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_market_hit={row['avg_market_hit']:.3f}"
                )
        lines.append("")

    lines.append("## Deploy Notes")
    lines.append("- `DIRECT_PIN_PRESSURE`: best for body-duel fouls and first-contact tackles.")
    lines.append("- `AERIAL_PIN_PRESSURE`: best when the CB is repeatedly contesting high entries and second-ball chaos.")
    lines.append("- `MOBILE_PRESSURE_FRONT`: more recovery-oriented; prefer fouls first, tackles second.")
    lines.append("- If the CB lane survives only on one-off sample without registry support, keep it review-only.")
    lines.append("")

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny deploy rulebook for the centre-back duel lane.")
    parser.add_argument("--cb-csv", required=True)
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_rulebook(args.cb_csv, args.registry_csv, args.output_md)
    print(f"WROTE: {args.output_md}")
