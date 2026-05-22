from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from audit_lineup_quality_response import _actuals_for_market, _score_market, _edge_bucket


def audit_quality_style_combo(
    input_csv: str,
    fixtures_csv: str,
    player_stats_csv: str,
    events_csv: str,
    outdir: str,
    sample_size: int = 100,
) -> pd.DataFrame:
    inputs = pd.read_csv(input_csv)
    rows: list[dict] = []
    market_to_style_col = {
        "yellow_cards": "fixture_style_label",
        "fouls_committed": "fixture_style_label",
        "shots": "fixture_attacking_style_label",
        "shots_on_target": "fixture_attacking_style_label",
    }
    for market, style_col in market_to_style_col.items():
        scored, score_col = _score_market(inputs, market)
        actual = _actuals_for_market(market, fixtures_csv, player_stats_csv, events_csv)
        merged = scored.merge(actual, on=["fixture_key", "player_name"], how="left")
        merged["hit_flag"] = merged["hit_flag"].fillna(0).astype(int)
        merged["quality_edge_bucket"] = _edge_bucket(merged.get("starting_xi_quality_edge", 0.0))

        fixture_order = (
            merged[["fixture_key", "match_date"]]
            .drop_duplicates()
            .assign(match_date_ts=lambda x: pd.to_datetime(x["match_date"], errors="coerce"))
            .sort_values(["match_date_ts", "fixture_key"], ascending=[False, False])
        )
        keep_keys = fixture_order.head(sample_size)["fixture_key"].tolist()
        merged = merged[merged["fixture_key"].isin(keep_keys)].copy()
        top = (
            merged.sort_values(["fixture_key", score_col], ascending=[True, False])
            .groupby("fixture_key", as_index=False, group_keys=False)
            .head(5)
            .copy()
        )
        top["market"] = market
        top["style_bucket"] = top[style_col].astype("string").fillna("UNSET")
        top["combo_label"] = top["quality_edge_bucket"].astype("string") + " + " + top["style_bucket"].astype("string")

        grouped = (
            top.groupby(["market", "quality_edge_bucket", "style_bucket", "combo_label"], as_index=False)
            .agg(
                rows=("player_name", "count"),
                fixtures=("fixture_key", "nunique"),
                hit_rate=("hit_flag", "mean"),
                avg_market_score=(score_col, "mean"),
                avg_player_quality_score=("player_quality_score_l5", "mean"),
                avg_quality_edge=("starting_xi_quality_edge", "mean"),
            )
            .sort_values(["market", "hit_rate", "rows"], ascending=[True, False, False])
        )
        rows.append(grouped)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    stem = Path(input_csv).stem
    csv_path = outdir_path / f"{stem}__quality_style_combo_audit_last{sample_size}.csv"
    md_path = outdir_path / f"{stem}__quality_style_combo_audit_last{sample_size}.md"
    out.to_csv(csv_path, index=False)

    lines = [f"# Quality x Style Audit: {stem}", ""]
    for market in ["yellow_cards", "fouls_committed", "shots", "shots_on_target"]:
        lines.append(f"## {market}")
        market_df = out[out["market"].eq(market)].head(10)
        for row in market_df.itertuples(index=False):
            lines.append(
                f"- {row.combo_label}: hit_rate={row.hit_rate:.4f} rows={row.rows} fixtures={row.fixtures} "
                f"avg_score={row.avg_market_score:.2f} avg_quality={row.avg_player_quality_score:.2f} avg_edge={row.avg_quality_edge:.2f}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit quality x style market combinations.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--player-stats-csv", required=True)
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--outdir", default="reports/player_events/quality_audits")
    parser.add_argument("--sample-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = audit_quality_style_combo(
        input_csv=args.input,
        fixtures_csv=args.fixtures_csv,
        player_stats_csv=args.player_stats_csv,
        events_csv=args.events_csv,
        outdir=args.outdir,
        sample_size=args.sample_size,
    )
    print("WROTE:", args.outdir)
    print(df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
