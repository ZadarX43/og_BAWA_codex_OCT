from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_style_csv(path: str, market: str, league_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["market"] = market
    df["league_label"] = league_label
    return df


def build_contact_style_summary(
    yc_inputs: list[tuple[str, str]],
    fouls_inputs: list[tuple[str, str]],
    output_csv: str,
    output_md: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path, league_label in yc_inputs:
        frames.append(_load_style_csv(path, "yellow_card", league_label))
    for path, league_label in fouls_inputs:
        frames.append(_load_style_csv(path, "fouls_committed", league_label))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Contact Style Audit Summary\n\nNo rows found.\n")
        return out

    out = out.sort_values(
        ["market", "top5_fixture_hit_rate", "top5_precision", "fixtures"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Contact Style Audit Summary", ""]
    for market, market_df in out.groupby("market", sort=False):
        lines.extend([f"## {market}", ""])
        agg = (
            market_df.groupby("fixture_style_label", as_index=False)
            .agg(
                leagues=("league_label", "nunique"),
                fixtures=("fixtures", "sum"),
                top3_fixture_hit_rate=("top3_fixture_hit_rate", "mean"),
                top5_fixture_hit_rate=("top5_fixture_hit_rate", "mean"),
                top5_precision=("top5_precision", "mean"),
            )
            .sort_values(["top5_fixture_hit_rate", "top5_precision", "fixtures"], ascending=[False, False, False])
        )
        lines.append("### Cross-League Style Ranking")
        for row in agg.itertuples(index=False):
            lines.append(
                f"- {row.fixture_style_label}: leagues={row.leagues} fixtures={row.fixtures} top3_hit={row.top3_fixture_hit_rate:.3f} top5_hit={row.top5_fixture_hit_rate:.3f} top5_precision={row.top5_precision:.3f}"
            )
        lines.extend(["", "### League-Level Detail"])
        for row in market_df.itertuples(index=False):
            lines.append(
                f"- {row.league_label} | {row.fixture_style_label}: fixtures={row.fixtures} top3_hit={row.top3_fixture_hit_rate:.3f} top5_hit={row.top5_fixture_hit_rate:.3f} top5_precision={row.top5_precision:.3f}"
            )
        lines.append("")

    Path(output_md).write_text("\n".join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a combined contact-style audit summary for YC and fouls.")
    parser.add_argument("--yc", action="append", default=[], help="Format: league_label=path/to/yc_style_breakdown.csv")
    parser.add_argument("--fouls", action="append", default=[], help="Format: league_label=path/to/fouls_style_breakdown.csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def _parse_items(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in items:
        league_label, path = item.split("=", 1)
        out.append((path, league_label))
    return out


def main() -> None:
    args = parse_args()
    df = build_contact_style_summary(
        yc_inputs=_parse_items(args.yc),
        fouls_inputs=_parse_items(args.fouls),
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(df)}")


if __name__ == "__main__":
    main()
