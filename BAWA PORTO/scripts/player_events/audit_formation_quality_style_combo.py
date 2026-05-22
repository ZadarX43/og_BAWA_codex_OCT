from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from audit_lineup_quality_response import _edge_bucket
from render_fouls_committed_report import _build_fci
from render_player_shots_on_target_report import _build_soti
from render_player_shots_report import _build_psi
from render_player_tackles_report import _build_tki
from render_yellow_card_report import _build_bpi


def _actuals_for_market(market: str, fixtures_csv: str, player_stats_csv: str, events_csv: str | None) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv, usecols=["fixture_id", "fixture_key"])
    stats = pd.read_csv(player_stats_csv).merge(fixtures, on="fixture_id", how="left")
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
    metric_map = {
        "fouls_committed": ("fouls_committed", 2),
        "tackles": ("tackles", 2),
        "shots": ("shots_total", 2),
        "shots_on_target": ("shots_on_target", 1),
    }
    if market not in metric_map:
        raise ValueError(f"Unsupported market: {market}")
    col, threshold = metric_map[market]
    out = stats[["fixture_key", "player_name", col]].copy()
    out["metric_actual"] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["hit_flag"] = (out["metric_actual"] >= threshold).astype(int)
    return out[["fixture_key", "player_name", "metric_actual", "hit_flag"]]


def _score_market(inputs: pd.DataFrame, market: str) -> tuple[pd.DataFrame, str]:
    if market == "yellow_cards":
        return _build_bpi(inputs).rename(columns={"booking_probability_index": "market_score"}), "market_score"
    if market == "fouls_committed":
        return _build_fci(inputs).rename(columns={"foul_commitment_index": "market_score"}), "market_score"
    if market == "tackles":
        return _build_tki(inputs).rename(columns={"player_tackle_index": "market_score"}), "market_score"
    if market == "shots":
        return _build_psi(inputs).rename(columns={"player_shot_index": "market_score"}), "market_score"
    if market == "shots_on_target":
        return _build_soti(inputs).rename(columns={"player_sot_index": "market_score"}), "market_score"
    raise ValueError(f"Unsupported market: {market}")


def audit_formation_quality_style_combo(
    input_csv: str,
    fixtures_csv: str,
    player_stats_csv: str,
    events_csv: str,
    outdir: str,
    sample_size: int = 100,
    min_rows: int = 5,
) -> pd.DataFrame:
    inputs = pd.read_csv(input_csv)
    rows: list[pd.DataFrame] = []
    market_to_style_col = {
        "yellow_cards": "fixture_style_label",
        "fouls_committed": "fixture_style_label",
        "tackles": "fixture_style_label",
        "shots": "fixture_attacking_style_label",
        "shots_on_target": "fixture_attacking_style_label",
    }
    for market, style_col in market_to_style_col.items():
        scored, score_col = _score_market(inputs, market)
        actual = _actuals_for_market(market, fixtures_csv, player_stats_csv, events_csv or None)
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
        top["formation_bucket"] = top["formation_matchup_label"].astype("string").fillna("UNSET")
        top["combo_label"] = (
            top["quality_edge_bucket"].astype("string")
            + " + "
            + top["style_bucket"].astype("string")
            + " + "
            + top["formation_bucket"].astype("string")
        )
        grouped = (
            top.groupby(
                ["market", "quality_edge_bucket", "style_bucket", "formation_bucket", "combo_label"],
                as_index=False,
            )
            .agg(
                rows=("player_name", "count"),
                fixtures=("fixture_key", "nunique"),
                hit_rate=("hit_flag", "mean"),
                avg_market_score=(score_col, "mean"),
                avg_player_quality_score=("player_quality_score_l5", "mean"),
                avg_quality_edge=("starting_xi_quality_edge", "mean"),
                avg_formation_pressure=("formation_pressure_score", "mean"),
            )
            .sort_values(["market", "hit_rate", "rows"], ascending=[True, False, False])
        )
        rows.append(grouped[grouped["rows"].ge(min_rows)].copy())

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    stem = Path(input_csv).stem
    csv_path = outdir_path / f"{stem}__formation_quality_style_audit_last{sample_size}.csv"
    md_path = outdir_path / f"{stem}__formation_quality_style_audit_last{sample_size}.md"
    out.to_csv(csv_path, index=False)

    lines = [f"# Formation x Quality x Style Audit: {stem}", ""]
    for market in ["yellow_cards", "fouls_committed", "tackles", "shots", "shots_on_target"]:
        lines.append(f"## {market}")
        market_df = out[out["market"].eq(market)].head(12)
        for row in market_df.itertuples(index=False):
            lines.append(
                f"- {row.combo_label}: hit_rate={row.hit_rate:.4f} rows={row.rows} fixtures={row.fixtures} "
                f"avg_score={row.avg_market_score:.2f} avg_quality={row.avg_player_quality_score:.2f} "
                f"avg_edge={row.avg_quality_edge:.2f} avg_form_pressure={row.avg_formation_pressure:.2f}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit formation x quality x style market combinations.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--player-stats-csv", required=True)
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--outdir", default="reports/player_events/quality_audits")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--min-rows", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = audit_formation_quality_style_combo(
        input_csv=args.input,
        fixtures_csv=args.fixtures_csv,
        player_stats_csv=args.player_stats_csv,
        events_csv=args.events_csv,
        outdir=args.outdir,
        sample_size=args.sample_size,
        min_rows=args.min_rows,
    )
    print("WROTE:", args.outdir)
    print(df.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
