from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def render_summary(input_csv: str, output_md: str, title: str = "Player Events Fouls League Comparison") -> Path:
    df = pd.read_csv(input_csv)
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "| League | Threshold | Avg High-Foul Players / Fixture | Top 1 Hit | Top 3 Hit | Top 5 Hit | Top 6 Hit | Top 10 Hit | Top 5 Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.league} | {row.foul_threshold} | {row.avg_actual_high_foul_per_fixture:.2f} | "
            f"{row.top1_fixture_hit_rate:.2f} | {row.top3_fixture_hit_rate:.2f} | {row.top5_fixture_hit_rate:.2f} | "
            f"{row.top6_fixture_hit_rate:.2f} | {row.top10_fixture_hit_rate:.2f} | {row.top5_precision:.3f} |"
        )

    if not df.empty:
        best = df.sort_values(["top5_fixture_hit_rate", "top5_precision"], ascending=[False, False]).iloc[0]
        lines.extend(
            [
                "",
                "## Readout",
                "",
                f"- best current broad fouls lane: `{best['league']}`",
                "- this is still beta shortlist quality, not bookmaker-line pricing",
                "- use this board internally as a shortlist layer, especially where fixture context is already strong",
                "",
                "## Interpretation",
                "",
                "- higher `top5_fixture_hit_rate` means the lane is good at putting at least one or more strong foul candidates into the top shortlist",
                "- `top5_precision` is the cleaner signal for how much clutter remains in that shortlist",
                "- the sweet spot is where both stay healthy without forcing too many names per fixture",
            ]
        )

    out_path = Path(output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a markdown summary from the fouls league comparison csv.")
    parser.add_argument("--input", required=True, help="Input fouls league comparison CSV")
    parser.add_argument("--output-md", required=True, help="Output markdown path")
    parser.add_argument("--title", default="Player Events Fouls League Comparison", help="Markdown title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = render_summary(args.input, args.output_md, args.title)
    print(f"WROTE: {out}")


if __name__ == "__main__":
    main()
