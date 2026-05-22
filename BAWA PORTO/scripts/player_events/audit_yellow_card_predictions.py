from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from render_yellow_card_report import _build_bpi


def _load_actual_bookings(events_csv: str, fixtures_csv: str, player_stats_csv: str) -> pd.DataFrame:
    events = pd.read_csv(events_csv)
    fixtures = pd.read_csv(fixtures_csv, usecols=["fixture_id", "fixture_key", "match_date", "league"])
    players = (
        pd.read_csv(player_stats_csv, usecols=["fixture_id", "player_id", "player_name"])
        .drop_duplicates(subset=["fixture_id", "player_id"])
    )
    booked = events[
        (events["event_type"].astype("string").str.lower() == "card")
        & (events["event_detail"].astype("string").str.contains("yellow", case=False, na=False))
    ].copy()
    booked = booked.merge(fixtures, on="fixture_id", how="left").merge(players, on=["fixture_id", "player_id"], how="left")
    booked = booked[["fixture_id", "fixture_key", "player_id", "player_name"]].dropna(subset=["fixture_key", "player_name"])
    booked["booked_flag"] = 1
    return booked.drop_duplicates(subset=["fixture_key", "player_name"])


def _joined_names(df: pd.DataFrame, col: str = "player_name", limit: int | None = None) -> str:
    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals.ne("")]
    if limit is not None:
        vals = vals.head(limit)
    return " | ".join(vals.tolist())


def _miss_breakdown(ranked: pd.DataFrame, top_n: int) -> tuple[str, str]:
    top = ranked.head(top_n).copy()
    fp = top[top["booked_flag"].eq(0)].copy()
    fn = ranked[(ranked["booked_flag"].eq(1)) & (~ranked["player_name"].isin(top["player_name"]))].copy()

    def _role_counts(x: pd.DataFrame) -> str:
        if x.empty:
            return ""
        counts = x["tactical_role"].fillna("Unknown").astype(str).value_counts()
        return " | ".join(f"{k}:{v}" for k, v in counts.items())

    return _role_counts(fp), _role_counts(fn)


def audit_yellow_card_predictions(
    input_csv: str,
    fixtures_csv: str,
    events_csv: str,
    player_stats_csv: str,
    outdir: str,
    sample_size: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = pd.read_csv(input_csv)
    scored = _build_bpi(inputs)
    actual = _load_actual_bookings(events_csv, fixtures_csv, player_stats_csv)

    scored = scored.merge(actual[["fixture_key", "player_name", "booked_flag"]], on=["fixture_key", "player_name"], how="left")
    scored["booked_flag"] = scored["booked_flag"].fillna(0).astype(int)

    fixture_order = (
        scored[["fixture_key", "match_date"]]
        .drop_duplicates()
        .assign(match_date_ts=lambda x: pd.to_datetime(x["match_date"], errors="coerce"))
        .sort_values(["match_date_ts", "fixture_key"], ascending=[False, False])
    )
    keep_keys = fixture_order.head(sample_size)["fixture_key"].tolist()
    scored = scored[scored["fixture_key"].isin(keep_keys)].copy()

    fixture_rows = []
    for fixture_key, group in scored.groupby("fixture_key", sort=False):
        ranked = group.sort_values("booking_probability_index", ascending=False).reset_index(drop=True)
        total_booked = int(ranked["booked_flag"].sum())
        actual_booked = ranked[ranked["booked_flag"].eq(1)].copy()
        top3_fp_roles, top3_fn_roles = _miss_breakdown(ranked, 3)
        top5_fp_roles, top5_fn_roles = _miss_breakdown(ranked, 5)
        row = {
            "fixture_key": fixture_key,
            "match_date": ranked["match_date"].iloc[0],
            "league": ranked["league"].iloc[0],
            "home_team_name": ranked["home_team_name"].iloc[0],
            "away_team_name": ranked["away_team_name"].iloc[0],
            "fixture_style_label": ranked["fixture_style_label"].iloc[0] if "fixture_style_label" in ranked.columns else "",
            "actual_booked_count": total_booked,
            "top3_predicted_names": _joined_names(ranked.head(3)),
            "top5_predicted_names": _joined_names(ranked.head(5)),
            "actual_booked_names": _joined_names(actual_booked),
            "top3_false_positive_roles": top3_fp_roles,
            "top3_false_negative_roles": top3_fn_roles,
            "top5_false_positive_roles": top5_fp_roles,
            "top5_false_negative_roles": top5_fn_roles,
        }
        for n in [1, 3, 5, 6, 10]:
            topn = ranked.head(n)
            row[f"top{n}_hits"] = int(topn["booked_flag"].sum())
            row[f"top{n}_hit_fixture_flag"] = int(topn["booked_flag"].sum() > 0)
            row[f"top{n}_precision"] = round(float(topn["booked_flag"].mean()), 4) if len(topn) else 0.0
        fixture_rows.append(row)

    fixture_df = pd.DataFrame(fixture_rows).sort_values(["match_date", "fixture_key"], ascending=[False, False])
    role_rows = []
    for role, role_df in scored.groupby("tactical_role", dropna=False):
        role_label = str(role) if pd.notna(role) and str(role).strip() else "Unknown"
        role_rows.append(
            {
                "tactical_role": role_label,
                "rows": int(len(role_df)),
                "actual_booked_rate": round(float(role_df["booked_flag"].mean()), 4),
                "avg_bpi": round(float(role_df["booking_probability_index"].mean()), 4),
            }
        )
    role_df = pd.DataFrame(role_rows).sort_values(["actual_booked_rate", "avg_bpi"], ascending=[False, False])
    style_rows = []
    for style, style_df in fixture_df.groupby("fixture_style_label", dropna=False):
        style_label = str(style) if pd.notna(style) and str(style).strip() else "UNSET"
        row = {
            "fixture_style_label": style_label,
            "fixtures": int(len(style_df)),
            "avg_actual_booked_per_fixture": round(float(style_df["actual_booked_count"].mean()), 4) if len(style_df) else 0.0,
        }
        for n in [1, 3, 5, 6, 10]:
            row[f"top{n}_fixture_hit_rate"] = round(float(style_df[f"top{n}_hit_fixture_flag"].mean()), 4) if len(style_df) else 0.0
            row[f"top{n}_precision"] = round(float(style_df[f"top{n}_precision"].mean()), 4) if len(style_df) else 0.0
        style_rows.append(row)
    style_df = pd.DataFrame(style_rows).sort_values(
        ["top5_fixture_hit_rate", "top5_precision", "fixtures"], ascending=[False, False, False]
    )

    summary = {
        "fixtures_audited": len(fixture_df),
        "avg_actual_booked_per_fixture": round(float(fixture_df["actual_booked_count"].mean()), 4) if len(fixture_df) else 0.0,
    }
    for n in [1, 3, 5, 6, 10]:
        summary[f"top{n}_fixture_hit_rate"] = round(float(fixture_df[f"top{n}_hit_fixture_flag"].mean()), 4) if len(fixture_df) else 0.0
        summary[f"top{n}_mean_hits"] = round(float(fixture_df[f"top{n}_hits"].mean()), 4) if len(fixture_df) else 0.0
        summary[f"top{n}_precision"] = round(float(fixture_df[f"top{n}_precision"].mean()), 4) if len(fixture_df) else 0.0

    summary_df = pd.DataFrame([summary])
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    fixture_out = out_path / (Path(input_csv).stem + f"__audit_fixture_level_last{sample_size}.csv")
    summary_out = out_path / (Path(input_csv).stem + f"__audit_summary_last{sample_size}.csv")
    role_out = out_path / (Path(input_csv).stem + f"__audit_role_breakdown_last{sample_size}.csv")
    style_out = out_path / (Path(input_csv).stem + f"__audit_style_breakdown_last{sample_size}.csv")
    report_out = out_path / (Path(input_csv).stem + f"__audit_report_last{sample_size}.md")
    fixture_df.to_csv(fixture_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    role_df.to_csv(role_out, index=False)
    style_df.to_csv(style_out, index=False)

    lines = [
        f"# Yellow Card Audit Report: {Path(input_csv).stem}",
        "",
        "## Summary",
    ]
    for col, val in summary_df.iloc[0].items():
        lines.append(f"- {col}: {val}")
    lines.extend(["", "## Role Breakdown"])
    for row in role_df.head(12).itertuples(index=False):
        lines.append(
            f"- {row.tactical_role}: booked_rate={row.actual_booked_rate} avg_bpi={row.avg_bpi} rows={row.rows}"
        )
    lines.extend(["", "## Contact Style Breakdown"])
    for row in style_df.head(10).itertuples(index=False):
        lines.append(
            f"- {row.fixture_style_label}: fixtures={row.fixtures} top3_hit={row.top3_fixture_hit_rate} top5_hit={row.top5_fixture_hit_rate} top5_precision={row.top5_precision}"
        )
    lines.extend(["", "## Fixture Samples"])
    sample_cols = [
        "fixture_key",
        "actual_booked_count",
        "top3_predicted_names",
        "actual_booked_names",
        "top3_false_positive_roles",
        "top3_false_negative_roles",
        "top5_predicted_names",
        "top5_false_positive_roles",
        "top5_false_negative_roles",
    ]
    for row in fixture_df.head(10)[sample_cols].itertuples(index=False):
        lines.extend(
            [
                f"### {row.fixture_key}",
                f"- actual booked count: {row.actual_booked_count}",
                f"- top 3 predicted: {row.top3_predicted_names}",
                f"- actual booked: {row.actual_booked_names}",
                f"- top 3 false-positive roles: {row.top3_false_positive_roles or 'none'}",
                f"- top 3 false-negative roles: {row.top3_false_negative_roles or 'none'}",
                f"- top 5 predicted: {row.top5_predicted_names}",
                f"- top 5 false-positive roles: {row.top5_false_positive_roles or 'none'}",
                f"- top 5 false-negative roles: {row.top5_false_negative_roles or 'none'}",
                "",
            ]
        )
    report_out.write_text("\n".join(lines))

    return summary_df, fixture_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit yellow-card beta predictions against actual booking events.")
    parser.add_argument("--input", required=True, help="player_events_fixture_input csv")
    parser.add_argument("--fixtures-csv", required=True, help="fixtures_master csv")
    parser.add_argument("--events-csv", required=True, help="match_events csv")
    parser.add_argument("--player-stats-csv", required=True, help="match_player_stats csv")
    parser.add_argument("--outdir", default="reports/player_events/audits", help="Audit output directory")
    parser.add_argument("--sample-size", type=int, default=100, help="Use the most recent N fixtures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df, fixture_df = audit_yellow_card_predictions(
        input_csv=args.input,
        fixtures_csv=args.fixtures_csv,
        events_csv=args.events_csv,
        player_stats_csv=args.player_stats_csv,
        outdir=args.outdir,
        sample_size=args.sample_size,
    )
    print("WROTE:", args.outdir)
    print(summary_df.to_string(index=False))
    print("top fixture samples:")
    preview_cols = [
        "fixture_key",
        "actual_booked_count",
        "top3_predicted_names",
        "actual_booked_names",
        "top3_false_positive_roles",
        "top3_false_negative_roles",
    ]
    print(fixture_df.head(5)[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
