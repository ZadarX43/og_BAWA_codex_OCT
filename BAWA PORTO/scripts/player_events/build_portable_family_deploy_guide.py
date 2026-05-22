from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MARKETS = ["tackles", "fouls_committed", "shots", "shots_on_target"]


def _load(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, low_memory=False)


def build_guide(cross_batch_market_csv: str, portability_csv: str, output_md: str) -> None:
    market_df = _load(cross_batch_market_csv)
    portability_df = _load(portability_csv)

    combined = market_df[market_df["scope"].eq("combined_overall")].copy()
    lines = ["# Portable Family Deploy Guide", ""]
    lines.append(
        "This guide turns the family backtests into a simple market-by-market trust map. `PORTABLE` means the family held up when we widened outside the Europe-core batches. `CONDITIONAL_PORTABLE` means usable but with more caution. `NEEDS_MORE_SAMPLE` means the numbers are attractive but the widened sample is still too thin."
    )
    lines.append("")

    for market in MARKETS:
        lines.append(f"## {market}")
        market_rows = combined[combined["market"].eq(market)].sort_values(["hit_rate", "rows"], ascending=[False, False])
        if market_rows.empty:
            lines.append("No rows matched.")
            lines.append("")
            continue

        best_family = market_rows.iloc[0]["source_family"]
        best_hit = float(market_rows.iloc[0]["hit_rate"])
        lines.append(f"- primary trusted family: `{best_family}` (`hit_rate={best_hit:.3f}`)")

        trust_lines = []
        for row in market_rows.itertuples(index=False):
            family = row.source_family
            portability_row = portability_df[
                portability_df["market"].eq(market) & portability_df["source_family"].eq(family)
            ]
            label = portability_row["portability_label"].iloc[0] if not portability_row.empty else "EUROPE_ONLY_SO_FAR"
            batch4_hit = float(portability_row["batch4_hit_rate"].iloc[0]) if not portability_row.empty else 0.0
            europe_hit = float(portability_row["europe_hit_rate"].iloc[0]) if not portability_row.empty else float(row.hit_rate)
            trust_lines.append(
                f"- `{family}`: combined_hit={float(row.hit_rate):.3f} | europe_core={europe_hit:.3f} | batch4={batch4_hit:.3f} | portability={label}"
            )

        lines.append("- trust ladder:")
        lines.extend(trust_lines)
        lines.append("- deployment note:")
        if market == "tackles":
            lines.append("  - treat `3421v4231`, `4231v442`, and `4231v433` as the main tackle families, with the first one still needing a little more widened contact sample.")
        elif market == "fouls_committed":
            lines.append("  - lean on `4231v442` first for fouls; `4231v433` is usable but a touch softer outside Europe-core.")
        elif market == "shots":
            lines.append("  - `4231v433` is the cleanest default shots family, but `3421v4231` is now a serious attacking-volume ally after batch 4.")
        elif market == "shots_on_target":
            lines.append("  - `4231v433` remains the default SOT family, with `3421v4231` now looking like a strong complementary angle.")
        lines.append("")

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown deploy guide for portable specialist families by market.")
    parser.add_argument("--cross-batch-market-csv", required=True)
    parser.add_argument("--portability-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_guide(args.cross_batch_market_csv, args.portability_csv, args.output_md)
    print(f"WROTE: {args.output_md}")
