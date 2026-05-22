from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from render_player_assists_report import _build_api


def audit_player_assists_predictions(
    input_csv: str,
    fixtures_csv: str,
    player_stats_csv: str,
    outdir: str,
    sample_size: int = 100,
    assist_threshold: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = pd.read_csv(input_csv)
    scored = _build_api(inputs)
    fixtures = pd.read_csv(fixtures_csv, usecols=["fixture_id", "fixture_key", "match_date"])
    actual = pd.read_csv(player_stats_csv, usecols=["fixture_id", "player_name", "assists"])
    actual = actual.merge(fixtures, on="fixture_id", how="left")
    actual["assists"] = pd.to_numeric(actual["assists"], errors="coerce").fillna(0.0)
    actual["assist_hit_flag"] = (actual["assists"] >= assist_threshold).astype(int)
    scored = scored.merge(actual[["fixture_key", "player_name", "assists", "assist_hit_flag"]], on=["fixture_key", "player_name"], how="left")
    scored["assist_hit_flag"] = scored["assist_hit_flag"].fillna(0).astype(int)

    fixture_order = (
        scored[["fixture_key", "match_date"]]
        .drop_duplicates()
        .assign(match_date_ts=lambda x: pd.to_datetime(x["match_date"], errors="coerce"))
        .sort_values(["match_date_ts", "fixture_key"], ascending=[False, False])
    )
    keep_keys = fixture_order.head(sample_size)["fixture_key"].tolist()
    scored = scored[scored["fixture_key"].isin(keep_keys)].copy()

    rows = []
    for fixture_key, group in scored.groupby("fixture_key", sort=False):
        ranked = group.sort_values("player_assist_index", ascending=False).reset_index(drop=True)
        actual_high = ranked[ranked["assist_hit_flag"].eq(1)].copy()
        row = {
            "fixture_key": fixture_key,
            "league": ranked["league"].iloc[0],
            "fixture_attacking_style_label": ranked["fixture_attacking_style_label"].iloc[0] if "fixture_attacking_style_label" in ranked.columns else "",
            "actual_high_assist_count": int(actual_high["assist_hit_flag"].sum()),
        }
        for n in [1, 3, 5]:
            topn = ranked.head(n)
            row[f"top{n}_fixture_hit_rate"] = int(topn["assist_hit_flag"].sum() > 0)
            row[f"top{n}_precision"] = round(float(topn["assist_hit_flag"].mean()), 4) if len(topn) else 0.0
        rows.append(row)

    fixture_df = pd.DataFrame(rows)
    style_df = (
        fixture_df.groupby("fixture_attacking_style_label", as_index=False)
        .agg(
            fixtures=("fixture_key", "count"),
            avg_actual_high_assist_count=("actual_high_assist_count", "mean"),
            top3_fixture_hit_rate=("top3_fixture_hit_rate", "mean"),
            top5_fixture_hit_rate=("top5_fixture_hit_rate", "mean"),
            top5_precision=("top5_precision", "mean"),
        )
        .sort_values(["top5_fixture_hit_rate", "top5_precision", "fixtures"], ascending=[False, False, False])
    )
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "fixtures_audited": len(fixture_df),
        "assist_threshold": assist_threshold,
        "top3_fixture_hit_rate": round(float(fixture_df["top3_fixture_hit_rate"].mean()), 4) if len(fixture_df) else 0.0,
        "top5_fixture_hit_rate": round(float(fixture_df["top5_fixture_hit_rate"].mean()), 4) if len(fixture_df) else 0.0,
        "top5_precision": round(float(fixture_df["top5_precision"].mean()), 4) if len(fixture_df) else 0.0,
    }])
    summary.to_csv(out / (Path(input_csv).stem + f"__assists_audit_summary_last{sample_size}.csv"), index=False)
    style_df.to_csv(out / (Path(input_csv).stem + f"__assists_audit_attack_style_last{sample_size}.csv"), index=False)
    fixture_df.to_csv(out / (Path(input_csv).stem + f"__assists_audit_fixture_level_last{sample_size}.csv"), index=False)
    return summary, style_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit beta player assists predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--player-stats-csv", required=True)
    parser.add_argument("--outdir", default="reports/player_events/assists_audits")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--assist-threshold", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, style_df = audit_player_assists_predictions(
        input_csv=args.input,
        fixtures_csv=args.fixtures_csv,
        player_stats_csv=args.player_stats_csv,
        outdir=args.outdir,
        sample_size=args.sample_size,
        assist_threshold=args.assist_threshold,
    )
    print("WROTE:", args.outdir)
    print(summary.to_string(index=False))
    print(style_df.to_string(index=False))


if __name__ == "__main__":
    main()
