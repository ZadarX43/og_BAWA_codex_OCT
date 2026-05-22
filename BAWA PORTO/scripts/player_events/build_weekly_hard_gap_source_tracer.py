from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")


def token_match(series: pd.Series, token: str) -> pd.Series:
    return series.astype(str).str.contains(token, case=False, na=False, regex=False)


def detect_fixture(path: Path, home: str, away: str) -> tuple[int, str]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return 0, ""
    cols = df.columns.tolist()
    pairs = [
        ("home_team_name", "away_team_name"),
        ("home", "away"),
    ]
    for home_col, away_col in pairs:
        if home_col in cols and away_col in cols:
            mask = token_match(df[home_col], home) & token_match(df[away_col], away)
            sub = df[mask]
            if not sub.empty:
                date_val = str(sub.iloc[0].get("match_date", sub.iloc[0].get("date_from", "")))[:10]
                return len(sub), date_val
    if "fixture_key" in cols:
        mask = token_match(df["fixture_key"], home.lower().replace(" ", "_")) & token_match(
            df["fixture_key"], away.lower().replace(" ", "_")
        )
        sub = df[mask]
        if not sub.empty:
            date_val = str(sub.iloc[0].get("match_date", sub.iloc[0].get("date_from", "")))[:10]
            return len(sub), date_val
    return 0, ""


def build(route_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    route = pd.read_csv(route_csv, low_memory=False)
    weekly = route[
        route["fixture_key"].isin(
            [
                "2025-02-08_las_palmas_villarreal",
                "2025-02-22_rayo_vallecano_villarreal",
                "2025-05-04_real_madrid_celta_vigo",
            ]
        )
    ].copy()
    rows: list[dict[str, object]] = []
    weekly_root = REPO_ROOT / "predictions_output/walk_forward_team_intelligence_full_validation_3y_weekly_2026_04_22"
    regen_root = REPO_ROOT / "reports/player_events/quality_audits/weekly_ranked_board_regen__2026_05_03/_WINDOW_SLIPS"

    for _, row in weekly.iterrows():
        fixture_key = str(row["fixture_key"])
        label = fixture_key.split("_", 1)[1]
        parts = label.split("_")
        # Known three cases; split robustly from route metadata isn't available, so map from fixture key.
        team_map = {
            "2025-02-08_las_palmas_villarreal": ("Las Palmas", "Villarreal"),
            "2025-02-22_rayo_vallecano_villarreal": ("Rayo Vallecano", "Villarreal"),
            "2025-05-04_real_madrid_celta_vigo": ("Real Madrid", "Celta Vigo"),
        }
        home, away = team_map[fixture_key]
        window_id = str(row["owning_window_id"])
        window_dir = weekly_root / window_id
        file_specs = [
            ("deploy_elite", next(iter(sorted((window_dir / "02_deploy").glob("*__DEPLOY_TIER_ELITE__*.csv"))), None)),
            ("deploy_standard", next(iter(sorted((window_dir / "02_deploy").glob("*__DEPLOY_TIER_STANDARD__*.csv"))), None)),
            ("scored_combined", next(iter(sorted((window_dir / "03_scored").glob("DEPLOY_COMBINED_SCORED_*.csv"))), None)),
            ("regen_ranked_board", next(iter(sorted((regen_root / window_id).glob("ranked_board_*.csv"))), None)),
        ]
        for stage, path in file_specs:
            if path is None:
                rows.append(
                    {
                        "fixture_key": fixture_key,
                        "window_id": window_id,
                        "home_team_name": home,
                        "away_team_name": away,
                        "stage": stage,
                        "path": "",
                        "fixture_found": 0,
                        "rows_found": 0,
                        "matched_date": "",
                        "trace_note": "File missing.",
                    }
                )
                continue
            count, matched_date = detect_fixture(path, home, away)
            rows.append(
                {
                    "fixture_key": fixture_key,
                    "window_id": window_id,
                    "home_team_name": home,
                    "away_team_name": away,
                    "stage": stage,
                    "path": str(path),
                    "fixture_found": int(count > 0),
                    "rows_found": int(count),
                    "matched_date": matched_date,
                    "trace_note": "Fixture found in stage file." if count > 0 else "Fixture absent from stage file.",
                }
            )

    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Weekly Hard-Gap Source Tracer",
        "",
        "- Traces the three remaining weekly-covered La Liga hard gaps through deploy, scored, and regenerated ranked-board stages.",
        "",
    ]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        lines.append(f"## {fixture_key}")
        for _, r in sub.iterrows():
            lines.append(
                f"- `{r['stage']}` | found={int(r['fixture_found'])} | rows={int(r['rows_found'])} | date=`{r['matched_date'] or 'n/a'}`"
            )
            lines.append(f"  path: `{r['path'] or 'missing'}`")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace weekly hard-gap fixtures through the weekly source/deploy/scored/ranked chain.")
    parser.add_argument(
        "--route-csv",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/goal_market_regeneration_route_board.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/weekly_hard_gap_source_tracer.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/weekly_hard_gap_source_tracer.md"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build(args.route_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
