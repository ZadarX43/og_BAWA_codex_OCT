from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
DEFAULT_LEAGUES = (
    "England Premier League,Spain La Liga,Italy Serie A,Germany Bundesliga,France Ligue 1,"
    "Portugal Liga,Netherlands Eredivisie,Belgium Pro,Scotland Premiership,England Championship,"
    "England EFL League 1,England FA Cup,USA MLS,Brazil Serie A,Champions League,Europa League,"
    "Europa Conference,Japan J1,Norway Eliteserien"
)
SOURCE_FROZEN_ROOTS = {
    "accuracy": REPO_ROOT / "walkforward_frozen_accuracy",
    "valueev_balanced": REPO_ROOT / "walkforward_frozen_valueev_balanced",
    "valueev_aggressive": REPO_ROOT / "walkforward_frozen_valueev_aggressive",
}
TARGET_MARKETS = {"ftr", "btts", "ou25"}


@dataclass
class MonthPlan:
    month_tag: str
    action: str
    source_root: Path | None
    source_month_dir: Path | None


def month_from_fixture_key(fixture_key: str) -> str:
    date_token = str(fixture_key).split("_", 1)[0]
    return datetime.strptime(date_token, "%Y-%m-%d").strftime("%Y-%m")


def load_month_plan(nonweekly_csv: Path) -> list[MonthPlan]:
    df = pd.read_csv(nonweekly_csv, low_memory=False)
    plans: dict[str, MonthPlan] = {}
    for _, row in df.iterrows():
        month_tag = str(row["month_tag"])
        status = str(row["regeneration_status"])
        if month_tag in plans:
            continue
        if status == "HAS_MONTH_ARCHIVE__HARVEST_OR_TARGETED_MONTH_REFRESH":
            chosen_root = None
            chosen_month_dir = None
            archives = [part.strip() for part in str(row.get("existing_frozen_month_archives", "")).split("|") if part.strip()]
            if archives:
                chosen_month_dir = Path(archives[0])
                for profile_name, root in SOURCE_FROZEN_ROOTS.items():
                    if str(root) in str(chosen_month_dir):
                        chosen_root = root
                        break
            plans[month_tag] = MonthPlan(month_tag, "harvest_existing_month_archive", chosen_root, chosen_month_dir)
        else:
            plans[month_tag] = MonthPlan(month_tag, "rebuild_missing_month_archive", None, None)
    return sorted(plans.values(), key=lambda p: p.month_tag)


def normalize_ranked_board_columns(df: pd.DataFrame) -> pd.DataFrame:
    home_col = "home_team_name" if "home_team_name" in df.columns else "home"
    away_col = "away_team_name" if "away_team_name" in df.columns else "away"
    date_col = "match_date"
    selection_col = "selection" if "selection" in df.columns else "bookie_pick"
    keep = [c for c in [home_col, away_col, date_col, "market", selection_col, "fixture_key"] if c in df.columns]
    out = df[keep].copy()
    out = out.rename(columns={home_col: "home", away_col: "away", selection_col: "selection"})
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["market"] = out["market"].astype(str).str.lower().str.strip()
    out["selection"] = out["selection"].astype(str)
    return out


def choose_source_csv(month_dir: Path) -> Path | None:
    candidates = [
        month_dir / f"backtest_{month_dir.name}.csv",
        month_dir / f"frozen_gated_{month_dir.name}.csv",
        month_dir / f"backtest_unscored_{month_dir.name}.csv",
        month_dir / f"raw_predictions_{month_dir.name}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # fallback to any allmarkets backtest-like file
    for pattern in ("*__BACKTEST__FTR_ACCURACY.csv", "*__BACKTEST.csv", "*__BACKTEST_SCORED.csv"):
        hits = sorted(month_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def harvest_month(month_tag: str, source_month_dir: Path, harvest_root: Path, profile_label: str) -> tuple[list[Path], Path | None]:
    source_csv = choose_source_csv(source_month_dir)
    if source_csv is None:
        return [], None
    df = pd.read_csv(source_csv, low_memory=False)
    if "market" not in df.columns:
        return [], source_csv
    out_base = harvest_root / month_tag / profile_label
    out_base.mkdir(parents=True, exist_ok=True)
    ranked_files: list[Path] = []
    ranked = normalize_ranked_board_columns(df)
    for market in sorted(TARGET_MARKETS):
        part = ranked[ranked["market"].eq(market)].copy()
        if part.empty:
            continue
        part["source_csv"] = str(source_csv)
        out_path = out_base / f"ranked_board_{market}_{month_tag}_{profile_label}.csv"
        part.to_csv(out_path, index=False)
        ranked_files.append(out_path)
    return ranked_files, source_csv


def run_month_rebuild(month_tag: str, archive_root: Path, leagues: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        "python3",
        str(REPO_ROOT / "run_frozen_walkforward.py"),
        "--start-month",
        month_tag,
        "--end-month",
        month_tag,
        "--leagues",
        leagues,
        "--markets",
        "ftr,ou25,btts,tg15,tg25",
        "--strict",
        "--ftr-profile",
        "accuracy",
        "--archive-root",
        str(archive_root),
    ]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)


def write_summary(rows: list[dict[str, object]], output_csv: Path, output_md: Path) -> None:
    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    lines = [
        "# Targeted Frozen Month Regeneration Summary",
        "",
        "- Existing frozen month archives were harvested into ranked-board-shaped CSVs.",
        "- Missing months were attempted through `run_frozen_walkforward.py` using the accuracy profile.",
        "",
    ]
    if out.empty:
        lines.append("- No rows.")
    else:
        summary = out.groupby(["action", "result"], dropna=False).agg(rows=("month_tag", "size")).reset_index()
        lines.append("## Summary")
        for _, row in summary.iterrows():
            lines.append(f"- {row['action']} | {row['result']} | rows={int(row['rows'])}")
        lines.append("")
        lines.append("## Months")
        for _, row in out.iterrows():
            lines.append(f"- `{row['month_tag']}` | action=`{row['action']}` | result=`{row['result']}`")
            if str(row.get("source_month_dir", "")).strip():
                lines.append(f"  source: `{row['source_month_dir']}`")
            if str(row.get("harvest_root", "")).strip():
                lines.append(f"  harvest: `{row['harvest_root']}`")
            if str(row.get("note", "")).strip():
                lines.append(f"  note: {row['note']}")
    output_md.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a targeted frozen-month harvest/rebuild workflow.")
    parser.add_argument(
        "--nonweekly-board-csv",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/nonweekly_goal_market_regeneration_board.csv"),
    )
    parser.add_argument(
        "--harvest-root",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/frozen_month_ranked_harvest__2026_05_03"),
    )
    parser.add_argument(
        "--regen-archive-root",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/frozen_month_regen_archive__2026_05_03"),
    )
    parser.add_argument(
        "--summary-csv",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/targeted_frozen_month_regeneration_summary.csv"),
    )
    parser.add_argument(
        "--summary-md",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/targeted_frozen_month_regeneration_summary.md"),
    )
    parser.add_argument("--leagues", default=DEFAULT_LEAGUES)
    parser.add_argument("--skip-rebuild-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = load_month_plan(Path(args.nonweekly_board_csv))
    harvest_root = Path(args.harvest_root)
    regen_archive_root = Path(args.regen_archive_root)
    summary_rows: list[dict[str, object]] = []

    if harvest_root.exists():
        shutil.rmtree(harvest_root)
    harvest_root.mkdir(parents=True, exist_ok=True)
    regen_archive_root.mkdir(parents=True, exist_ok=True)

    for plan in plans:
        if plan.action == "harvest_existing_month_archive":
            if plan.source_month_dir is None or not plan.source_month_dir.exists():
                summary_rows.append(
                    {
                        "month_tag": plan.month_tag,
                        "action": plan.action,
                        "result": "MISSING_SOURCE_MONTH_DIR",
                        "source_month_dir": str(plan.source_month_dir or ""),
                        "harvest_root": "",
                        "note": "Expected frozen month archive was not available at runtime.",
                    }
                )
                continue
            profile_label = plan.source_root.name.replace("walkforward_frozen_", "") if plan.source_root else "unknown_profile"
            ranked_files, source_csv = harvest_month(plan.month_tag, plan.source_month_dir, harvest_root, profile_label)
            summary_rows.append(
                {
                    "month_tag": plan.month_tag,
                    "action": plan.action,
                    "result": "HARVESTED" if ranked_files else "NO_MARKET_ROWS_HARVESTED",
                    "source_month_dir": str(plan.source_month_dir),
                    "harvest_root": str(harvest_root / plan.month_tag / profile_label),
                    "note": str(source_csv) if source_csv else "No source csv found in month dir.",
                }
            )
            continue

        if args.skip_rebuild_missing:
            summary_rows.append(
                {
                    "month_tag": plan.month_tag,
                    "action": plan.action,
                    "result": "SKIPPED_REBUILD",
                    "source_month_dir": "",
                    "harvest_root": "",
                    "note": "Missing-month rebuild was skipped by flag.",
                }
            )
            continue

        month_dir = regen_archive_root / plan.month_tag
        if month_dir.exists():
            ranked_files, source_csv = harvest_month(plan.month_tag, month_dir, harvest_root, "regen_accuracy")
            summary_rows.append(
                {
                    "month_tag": plan.month_tag,
                    "action": plan.action,
                    "result": "EXISTING_REGEN_ARCHIVE_HARVESTED" if ranked_files else "EXISTING_REGEN_ARCHIVE_NO_MARKET_ROWS",
                    "source_month_dir": str(month_dir),
                    "harvest_root": str(harvest_root / plan.month_tag / "regen_accuracy"),
                    "note": str(source_csv) if source_csv else "No source csv found in existing regen archive.",
                }
            )
            continue

        proc = run_month_rebuild(plan.month_tag, regen_archive_root, args.leagues)
        month_dir = regen_archive_root / plan.month_tag
        if proc.returncode != 0:
            summary_rows.append(
                {
                    "month_tag": plan.month_tag,
                    "action": plan.action,
                    "result": "REBUILD_FAILED",
                    "source_month_dir": str(month_dir),
                    "harvest_root": "",
                    "note": (proc.stderr or proc.stdout or "").strip()[:2000],
                }
            )
            continue

        ranked_files, source_csv = harvest_month(plan.month_tag, month_dir, harvest_root, "regen_accuracy")
        summary_rows.append(
            {
                "month_tag": plan.month_tag,
                "action": plan.action,
                "result": "REBUILT_AND_HARVESTED" if ranked_files else "REBUILT_BUT_NO_MARKET_ROWS_HARVESTED",
                "source_month_dir": str(month_dir),
                "harvest_root": str(harvest_root / plan.month_tag / "regen_accuracy"),
                "note": str(source_csv) if source_csv else "No source csv found after rebuild.",
            }
        )

    write_summary(summary_rows, Path(args.summary_csv), Path(args.summary_md))
    print(f"WROTE: {args.summary_csv}")
    print(f"WROTE: {args.summary_md}")


if __name__ == "__main__":
    main()
