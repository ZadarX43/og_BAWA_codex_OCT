from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from render_fouls_committed_report import _build_fci


def _load_actual_fouls(player_stats_csv: str, fixtures_csv: str, foul_threshold: int) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv, usecols=["fixture_id", "fixture_key", "match_date", "league"])
    stats = pd.read_csv(
        player_stats_csv,
        usecols=["fixture_id", "player_id", "player_name", "team_id", "fouls_committed", "tackles", "position"],
    )
    merged = stats.merge(fixtures, on="fixture_id", how="left")
    merged["fouls_committed"] = pd.to_numeric(merged["fouls_committed"], errors="coerce").fillna(0.0)
    merged["tackles"] = pd.to_numeric(merged["tackles"], errors="coerce").fillna(0.0)
    merged["high_foul_flag"] = (merged["fouls_committed"] >= foul_threshold).astype(int)
    return merged


def _joined_names(df: pd.DataFrame, col: str, limit: int | None = None) -> str:
    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals.ne("")]
    if limit is not None:
        vals = vals.head(limit)
    return " | ".join(vals.tolist())


def _role_counts(x: pd.DataFrame) -> str:
    if x.empty:
        return ""
    counts = x["tactical_role"].fillna("Unknown").astype(str).value_counts()
    return " | ".join(f"{k}:{v}" for k, v in counts.items())


def audit_fouls_predictions(
    input_csv: str,
    fixtures_csv: str,
    player_stats_csv: str,
    outdir: str,
    sample_size: int = 100,
    foul_threshold: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = pd.read_csv(input_csv)
    scored = _build_fci(inputs)
    actual = _load_actual_fouls(player_stats_csv, fixtures_csv, foul_threshold)

    scored = scored.merge(
        actual[["fixture_key", "player_name", "fouls_committed", "tackles", "high_foul_flag"]],
        on=["fixture_key", "player_name"],
        how="left",
    )
    scored["high_foul_flag"] = scored["high_foul_flag"].fillna(0).astype(int)
    scored["fouls_committed_actual"] = scored["fouls_committed"].fillna(0.0)
    scored["tackles_actual"] = scored["tackles"].fillna(0.0)

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
        ranked = group.sort_values("foul_commitment_index", ascending=False).reset_index(drop=True)
        actual_high = group[group["high_foul_flag"].eq(1)].sort_values(["fouls_committed_actual", "tackles_actual"], ascending=[False, False])
        row = {
            "fixture_key": fixture_key,
            "match_date": ranked["match_date"].iloc[0],
            "league": ranked["league"].iloc[0],
            "home_team_name": ranked["home_team_name"].iloc[0],
            "away_team_name": ranked["away_team_name"].iloc[0],
            "fixture_style_label": ranked["fixture_style_label"].iloc[0] if "fixture_style_label" in ranked.columns else "",
            "actual_high_foul_count": int(actual_high["high_foul_flag"].sum()),
            "top3_predicted_names": _joined_names(ranked.head(3), "player_name"),
            "top5_predicted_names": _joined_names(ranked.head(5), "player_name"),
            "actual_high_foul_names": _joined_names(actual_high, "player_name"),
            "actual_top_foul_names": _joined_names(actual_high, "player_name", limit=5),
        }
        for n in [1, 3, 5, 6, 10]:
            topn = ranked.head(n).copy()
            row[f"top{n}_hits"] = int(topn["high_foul_flag"].sum())
            row[f"top{n}_hit_fixture_flag"] = int(topn["high_foul_flag"].sum() > 0)
            row[f"top{n}_precision"] = round(float(topn["high_foul_flag"].mean()), 4) if len(topn) else 0.0
        fp = ranked.head(5)
        fp = fp[fp["high_foul_flag"].eq(0)]
        fn = actual_high[~actual_high["player_name"].isin(ranked.head(5)["player_name"])]
        row["top5_false_positive_roles"] = _role_counts(fp)
        row["top5_false_negative_roles"] = _role_counts(fn)
        fixture_rows.append(row)

    fixture_df = pd.DataFrame(fixture_rows).sort_values(["match_date", "fixture_key"], ascending=[False, False])
    role_rows = []
    for role, role_df in scored.groupby("tactical_role", dropna=False):
        role_label = str(role) if pd.notna(role) and str(role).strip() else "Unknown"
        role_rows.append(
            {
                "tactical_role": role_label,
                "rows": int(len(role_df)),
                "actual_high_foul_rate": round(float(role_df["high_foul_flag"].mean()), 4),
                "avg_fci": round(float(role_df["foul_commitment_index"].mean()), 4),
            }
        )
    role_df = pd.DataFrame(role_rows).sort_values(["actual_high_foul_rate", "avg_fci"], ascending=[False, False])
    style_rows = []
    for style, style_df in fixture_df.groupby("fixture_style_label", dropna=False):
        style_label = str(style) if pd.notna(style) and str(style).strip() else "UNSET"
        row = {
            "fixture_style_label": style_label,
            "fixtures": int(len(style_df)),
            "avg_actual_high_foul_per_fixture": round(float(style_df["actual_high_foul_count"].mean()), 4) if len(style_df) else 0.0,
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
        "avg_actual_high_foul_per_fixture": round(float(fixture_df["actual_high_foul_count"].mean()), 4) if len(fixture_df) else 0.0,
        "foul_threshold": foul_threshold,
    }
    for n in [1, 3, 5, 6, 10]:
        summary[f"top{n}_fixture_hit_rate"] = round(float(fixture_df[f"top{n}_hit_fixture_flag"].mean()), 4) if len(fixture_df) else 0.0
        summary[f"top{n}_mean_hits"] = round(float(fixture_df[f"top{n}_hits"].mean()), 4) if len(fixture_df) else 0.0
        summary[f"top{n}_precision"] = round(float(fixture_df[f"top{n}_precision"].mean()), 4) if len(fixture_df) else 0.0

    summary_df = pd.DataFrame([summary])
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(input_csv).stem
    fixture_out = out_path / f"{stem}__fouls_audit_fixture_level_last{sample_size}.csv"
    summary_out = out_path / f"{stem}__fouls_audit_summary_last{sample_size}.csv"
    role_out = out_path / f"{stem}__fouls_audit_role_breakdown_last{sample_size}.csv"
    style_out = out_path / f"{stem}__fouls_audit_style_breakdown_last{sample_size}.csv"
    report_out = out_path / f"{stem}__fouls_audit_report_last{sample_size}.md"
    fixture_df.to_csv(fixture_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    role_df.to_csv(role_out, index=False)
    style_df.to_csv(style_out, index=False)

    lines = [
        f"# Fouls Committed Audit Report: {stem}",
        "",
        "## Summary",
    ]
    for col, val in summary_df.iloc[0].items():
        lines.append(f"- {col}: {val}")
    lines.extend(["", "## Role Breakdown"])
    for row in role_df.head(12).itertuples(index=False):
        lines.append(f"- {row.tactical_role}: high_foul_rate={row.actual_high_foul_rate} avg_fci={row.avg_fci} rows={row.rows}")
    lines.extend(["", "## Contact Style Breakdown"])
    for row in style_df.head(10).itertuples(index=False):
        lines.append(
            f"- {row.fixture_style_label}: fixtures={row.fixtures} top3_hit={row.top3_fixture_hit_rate} top5_hit={row.top5_fixture_hit_rate} top5_precision={row.top5_precision}"
        )
    lines.extend(["", "## Fixture Samples"])
    sample_cols = [
        "fixture_key",
        "actual_high_foul_count",
        "top5_predicted_names",
        "actual_high_foul_names",
        "top5_false_positive_roles",
        "top5_false_negative_roles",
    ]
    for row in fixture_df.head(10)[sample_cols].itertuples(index=False):
        lines.extend(
            [
                f"### {row.fixture_key}",
                f"- actual high-foul count: {row.actual_high_foul_count}",
                f"- top 5 predicted: {row.top5_predicted_names}",
                f"- actual high-foul names: {row.actual_high_foul_names}",
                f"- top 5 false-positive roles: {row.top5_false_positive_roles or 'none'}",
                f"- top 5 false-negative roles: {row.top5_false_negative_roles or 'none'}",
                "",
            ]
        )
    report_out.write_text("\n".join(lines))
    return summary_df, fixture_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fouls-committed beta predictions against actual player foul counts.")
    parser.add_argument("--input", required=True, help="player_events_fixture_input csv")
    parser.add_argument("--fixtures-csv", required=True, help="fixtures_master csv")
    parser.add_argument("--player-stats-csv", required=True, help="match_player_stats csv")
    parser.add_argument("--outdir", default="reports/player_events/fouls_audits", help="Audit output directory")
    parser.add_argument("--sample-size", type=int, default=100, help="Use the most recent N fixtures")
    parser.add_argument("--foul-threshold", type=int, default=2, help="Count actual high-foul hits at or above this threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df, fixture_df = audit_fouls_predictions(
        input_csv=args.input,
        fixtures_csv=args.fixtures_csv,
        player_stats_csv=args.player_stats_csv,
        outdir=args.outdir,
        sample_size=args.sample_size,
        foul_threshold=args.foul_threshold,
    )
    print("WROTE:", args.outdir)
    print(summary_df.to_string(index=False))
    preview_cols = [
        "fixture_key",
        "actual_high_foul_count",
        "top5_predicted_names",
        "actual_high_foul_names",
        "top5_false_positive_roles",
        "top5_false_negative_roles",
    ]
    print(fixture_df.head(5)[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
