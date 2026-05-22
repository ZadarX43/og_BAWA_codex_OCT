from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from render_fouls_committed_report import _build_fci
from render_player_shots_on_target_report import _build_soti
from render_player_shots_report import _build_psi
from render_yellow_card_report import _build_bpi


def _edge_bucket(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return pd.cut(
        s,
        bins=[-999, -8, -3, 3, 8, 999],
        labels=[
            "Heavy Underdog",
            "Underdog",
            "Balanced",
            "Favorite",
            "Heavy Favorite",
        ],
    ).astype("string")


def _actuals_for_market(market: str, fixtures_csv: str, player_stats_csv: str, events_csv: str | None) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv, usecols=["fixture_id", "fixture_key"])
    stats = pd.read_csv(player_stats_csv)
    stats = stats.merge(fixtures, on="fixture_id", how="left")
    if market == "yellow_cards":
        if not events_csv:
            raise ValueError("events_csv is required for yellow-card audit")
        events = pd.read_csv(events_csv)
        players = (
            pd.read_csv(player_stats_csv, usecols=["fixture_id", "player_id", "player_name"])
            .drop_duplicates(subset=["fixture_id", "player_id"])
        )
        booked = events[
            (events["event_type"].astype("string").str.lower() == "card")
            & (events["event_detail"].astype("string").str.contains("yellow", case=False, na=False))
        ].copy()
        booked = booked.merge(fixtures, on="fixture_id", how="left").merge(players, on=["fixture_id", "player_id"], how="left")
        booked = booked[["fixture_key", "player_name"]].dropna().drop_duplicates()
        booked["hit_flag"] = 1
        return booked
    if market == "fouls_committed":
        out = stats[["fixture_key", "player_name", "fouls_committed"]].copy()
        out["metric_actual"] = pd.to_numeric(out["fouls_committed"], errors="coerce").fillna(0.0)
        out["hit_flag"] = (out["metric_actual"] >= 2).astype(int)
        return out[["fixture_key", "player_name", "metric_actual", "hit_flag"]]
    if market == "shots":
        out = stats[["fixture_key", "player_name", "shots_total"]].copy()
        out["metric_actual"] = pd.to_numeric(out["shots_total"], errors="coerce").fillna(0.0)
        out["hit_flag"] = (out["metric_actual"] >= 2).astype(int)
        return out[["fixture_key", "player_name", "metric_actual", "hit_flag"]]
    if market == "shots_on_target":
        out = stats[["fixture_key", "player_name", "shots_on_target"]].copy()
        out["metric_actual"] = pd.to_numeric(out["shots_on_target"], errors="coerce").fillna(0.0)
        out["hit_flag"] = (out["metric_actual"] >= 1).astype(int)
        return out[["fixture_key", "player_name", "metric_actual", "hit_flag"]]
    raise ValueError(f"Unsupported market: {market}")


def _score_market(inputs: pd.DataFrame, market: str) -> tuple[pd.DataFrame, str]:
    if market == "yellow_cards":
        scored = _build_bpi(inputs).rename(columns={"booking_probability_index": "market_score"})
        return scored, "market_score"
    if market == "fouls_committed":
        scored = _build_fci(inputs).rename(columns={"foul_commitment_index": "market_score"})
        return scored, "market_score"
    if market == "shots":
        scored = _build_psi(inputs).rename(columns={"player_shot_index": "market_score"})
        return scored, "market_score"
    if market == "shots_on_target":
        scored = _build_soti(inputs).rename(columns={"player_sot_index": "market_score"})
        return scored, "market_score"
    raise ValueError(f"Unsupported market: {market}")


def audit_lineup_quality_response(
    input_csv: str,
    fixtures_csv: str,
    player_stats_csv: str,
    outdir: str,
    sample_size: int = 100,
    events_csv: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = pd.read_csv(input_csv)
    out_rows: list[dict] = []
    bucket_rows: list[dict] = []
    markets = ["yellow_cards", "fouls_committed", "shots", "shots_on_target"]

    for market in markets:
        scored, score_col = _score_market(inputs, market)
        actual = _actuals_for_market(market, fixtures_csv, player_stats_csv, events_csv)
        merged = scored.merge(actual, on=["fixture_key", "player_name"], how="left")
        merged["hit_flag"] = merged["hit_flag"].fillna(0).astype(int)
        if "metric_actual" in merged.columns:
            merged["metric_actual"] = pd.to_numeric(merged["metric_actual"], errors="coerce").fillna(0.0)
        else:
            merged["metric_actual"] = 0.0
        merged["quality_edge_bucket"] = _edge_bucket(merged.get("starting_xi_quality_edge", 0.0))
        merged["player_form_tier"] = merged["player_form_tier"].astype("string").fillna("UNSET")

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
        out_rows.append(
            {
                "market": market,
                "fixtures_audited": int(top["fixture_key"].nunique()),
                "top5_pick_hit_rate": round(float(top["hit_flag"].mean()), 4) if len(top) else 0.0,
                "top5_avg_player_quality_score": round(float(pd.to_numeric(top["player_quality_score_l5"], errors="coerce").mean()), 4) if len(top) else 0.0,
                "top5_avg_quality_edge": round(float(pd.to_numeric(top["starting_xi_quality_edge"], errors="coerce").mean()), 4) if len(top) else 0.0,
            }
        )
        for (edge_bucket, form_tier), grp in top.groupby(["quality_edge_bucket", "player_form_tier"], dropna=False):
            bucket_rows.append(
                {
                    "market": market,
                    "quality_edge_bucket": str(edge_bucket) if pd.notna(edge_bucket) else "UNSET",
                    "player_form_tier": str(form_tier) if pd.notna(form_tier) else "UNSET",
                    "rows": int(len(grp)),
                    "fixtures": int(grp["fixture_key"].nunique()),
                    "pick_hit_rate": round(float(grp["hit_flag"].mean()), 4) if len(grp) else 0.0,
                    "avg_market_score": round(float(pd.to_numeric(grp[score_col], errors="coerce").mean()), 4) if len(grp) else 0.0,
                    "avg_player_quality_score": round(float(pd.to_numeric(grp["player_quality_score_l5"], errors="coerce").mean()), 4) if len(grp) else 0.0,
                    "avg_quality_edge": round(float(pd.to_numeric(grp["starting_xi_quality_edge"], errors="coerce").mean()), 4) if len(grp) else 0.0,
                }
            )

    summary_df = pd.DataFrame(out_rows)
    buckets_df = pd.DataFrame(bucket_rows).sort_values(
        ["market", "pick_hit_rate", "rows"], ascending=[True, False, False]
    )

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(input_csv).stem
    summary_path = out / f"{stem}__lineup_quality_market_summary_last{sample_size}.csv"
    buckets_path = out / f"{stem}__lineup_quality_bucket_audit_last{sample_size}.csv"
    md_path = out / f"{stem}__lineup_quality_audit_last{sample_size}.md"
    summary_df.to_csv(summary_path, index=False)
    buckets_df.to_csv(buckets_path, index=False)

    lines = [
        f"# Lineup Quality Audit: {stem}",
        "",
        "## Market Summary",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {row.market}: fixtures={row.fixtures_audited} top5_pick_hit_rate={row.top5_pick_hit_rate} "
            f"avg_player_quality={row.top5_avg_player_quality_score} avg_quality_edge={row.top5_avg_quality_edge}"
        )
    lines.extend(["", "## Best Buckets"])
    for market in summary_df["market"].tolist():
        lines.append(f"### {market}")
        best = buckets_df[buckets_df["market"].eq(market)].head(5)
        for row in best.itertuples(index=False):
            lines.append(
                f"- {row.quality_edge_bucket} | {row.player_form_tier}: hit_rate={row.pick_hit_rate} "
                f"rows={row.rows} avg_quality={row.avg_player_quality_score} avg_edge={row.avg_quality_edge}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines))
    return summary_df, buckets_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit how player/team quality signals affect player-event markets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--player-stats-csv", required=True)
    parser.add_argument("--events-csv", default="")
    parser.add_argument("--outdir", default="reports/player_events/quality_audits")
    parser.add_argument("--sample-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, buckets = audit_lineup_quality_response(
        input_csv=args.input,
        fixtures_csv=args.fixtures_csv,
        player_stats_csv=args.player_stats_csv,
        outdir=args.outdir,
        sample_size=args.sample_size,
        events_csv=args.events_csv or None,
    )
    print("WROTE:", args.outdir)
    print(summary.to_string(index=False))
    print(buckets.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
