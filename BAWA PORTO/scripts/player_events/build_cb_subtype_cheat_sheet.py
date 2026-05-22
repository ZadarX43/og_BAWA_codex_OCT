from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUBTYPE_ORDER = [
    "DIRECT_TARGET_STRIKER",
    "AERIAL_BOX_NINE",
    "MOBILE_PRESSING_9",
    "CHANNEL_RUNNER_STRIKER",
]


def build_cheat_sheet(cb_csv: str, output_md: str) -> None:
    cb = pd.read_csv(cb_csv, low_memory=False)
    lines = ["# CB Subtype Cheat Sheet", ""]

    if cb.empty:
        lines.append("No centre-back duel rows matched.")
        Path(output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(output_md).write_text("\n".join(lines) + "\n")
        return

    lines.append("- Beta review only: use as a mechanism cheat sheet, not as an auto-deploy rule.")
    lines.append("- Preferred market order stays `fouls_committed` -> `tackles` -> bookings support.")
    lines.append("")

    grouped = {
        profile: sub.copy()
        for profile, sub in cb.groupby(cb.get("opponent_striker_profile", pd.Series("UNSET", index=cb.index)).astype(str), sort=False)
    }

    for profile in SUBTYPE_ORDER:
        sub = grouped.get(profile)
        if sub is None or sub.empty:
            lines.append(f"## {profile}")
            lines.append("- No live weekend rows yet.")
            lines.append("")
            continue

        first = sub.iloc[0]
        markets = ", ".join(sorted(sub["market"].astype(str).unique()))
        best = sub.sort_values(["cb_weekend_priority", "cb_duel_pressure_score", "score"], ascending=[False, False, False]).iloc[0]
        lines.append(f"## {profile}")
        lines.append(f"- subtype_note: {first.get('opponent_striker_subtype_note', 'UNSET')}")
        lines.append(
            f"- live_rows={len(sub)} | fixtures={sub['fixture_key'].nunique()} | avg_priority={pd.to_numeric(sub['cb_weekend_priority'], errors='coerce').mean():.2f} | avg_cb_duel_pressure={pd.to_numeric(sub['cb_duel_pressure_score'], errors='coerce').mean():.3f}"
        )
        lines.append(f"- preferred_markets: {markets}")
        lines.append(
            f"- best_fixture: `{best['fixture_key']}` | {best['player_name']} ({best['team_name']}) | pressure_tag={best.get('opponent_striker_pressure_tag', 'UNSET')}"
        )
        lines.append("")

    leftovers = [p for p in grouped if p not in SUBTYPE_ORDER and p != "UNSET"]
    for profile in sorted(leftovers):
        sub = grouped[profile]
        first = sub.iloc[0]
        lines.append(f"## {profile}")
        lines.append(f"- subtype_note: {first.get('opponent_striker_subtype_note', 'UNSET')}")
        lines.append(
            f"- live_rows={len(sub)} | fixtures={sub['fixture_key'].nunique()} | avg_priority={pd.to_numeric(sub['cb_weekend_priority'], errors='coerce').mean():.2f}"
        )
        lines.append("")

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny centre-back striker-subtype cheat sheet from the live CB weekend board.")
    parser.add_argument("--cb-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_cheat_sheet(args.cb_csv, args.output_md)
    print(f"WROTE: {args.output_md}")
