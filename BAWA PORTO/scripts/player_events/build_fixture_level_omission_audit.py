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
    for home_col, away_col in [("home_team_name", "away_team_name"), ("home", "away")]:
        if home_col in cols and away_col in cols:
            sub = df[token_match(df[home_col], home) & token_match(df[away_col], away)]
            if not sub.empty:
                return len(sub), str(sub.iloc[0].get("match_date", ""))[:10]
    if "fixture_key" in cols:
        sub = df[
            token_match(df["fixture_key"], home.lower().replace(" ", "_"))
            & token_match(df["fixture_key"], away.lower().replace(" ", "_"))
        ]
        if not sub.empty:
            return len(sub), str(sub.iloc[0].get("match_date", ""))[:10]
    return 0, ""


def build(output_csv: str, output_md: str) -> pd.DataFrame:
    fixtures = [
        ("2024-06-20_san_jose_earthquakes_portland_timbers", "2024-06", "San Jose Earthquakes", "Portland Timbers"),
        ("2024-07-18_vancouver_whitecaps_sporting_kansas_city", "2024-07", "Vancouver Whitecaps", "Sporting Kansas City"),
    ]
    regen_root = REPO_ROOT / "reports/player_events/quality_audits/frozen_month_regen_archive__2026_05_03"
    harvest_root = REPO_ROOT / "reports/player_events/quality_audits/frozen_month_ranked_harvest__2026_05_03"
    rows: list[dict[str, object]] = []
    for fixture_key, month_tag, home, away in fixtures:
        month_dir = regen_root / month_tag
        specs = [
            ("raw_predictions", month_dir / f"raw_predictions_{month_tag}.csv"),
            ("backtest", month_dir / f"backtest_{month_tag}.csv"),
            ("frozen_gated", month_dir / f"frozen_gated_{month_tag}.csv"),
            ("harvested_ranked", next(iter(sorted((harvest_root / month_tag).rglob("ranked_board_*.csv"))), None)),
        ]
        for stage, path in specs:
            if path is None or not path.exists():
                rows.append(
                    {
                        "fixture_key": fixture_key,
                        "month_tag": month_tag,
                        "home_team_name": home,
                        "away_team_name": away,
                        "stage": stage,
                        "path": "" if path is None else str(path),
                        "fixture_found": 0,
                        "rows_found": 0,
                        "matched_date": "",
                        "omission_note": "Stage file missing." if path is None or not path.exists() else "",
                    }
                )
                continue
            count, matched_date = detect_fixture(path, home, away)
            rows.append(
                {
                    "fixture_key": fixture_key,
                    "month_tag": month_tag,
                    "home_team_name": home,
                    "away_team_name": away,
                    "stage": stage,
                    "path": str(path),
                    "fixture_found": int(count > 0),
                    "rows_found": int(count),
                    "matched_date": matched_date,
                    "omission_note": "Fixture present at this stage." if count > 0 else "Fixture absent at this stage.",
                }
            )

    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Fixture-Level Omission Audit",
        "",
        "- Tracks the two remaining rebuilt-month omissions through raw, backtest, gated, and harvested stages.",
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
    parser = argparse.ArgumentParser(description="Audit fixture-level omissions inside rebuilt frozen-month archives.")
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/fixture_level_omission_audit.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/fixture_level_omission_audit.md"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build(args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
