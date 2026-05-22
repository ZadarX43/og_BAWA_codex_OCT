from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = REPO_ROOT / "data_sources" / "api_football" / "normalized"
TARGET_TAGS = {"USA_MLS", "Europa_League", "Europa_Conference"}
SEASONS = (2022, 2023, 2024)


def _load_batches() -> dict[str, list[str]]:
    module_path = Path(__file__).resolve().parent / "run_greenlist_specialist_family_batch.py"
    spec = importlib.util.spec_from_file_location("run_greenlist_specialist_family_batch", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load batch definitions from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "BATCHES")


def _greenlist_tags() -> list[str]:
    tags: list[str] = []
    for batch in _load_batches().values():
        tags.extend(batch)
    return sorted({tag for tag in tags if tag in TARGET_TAGS})


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0)
    except OSError:
        return 0


def build(output_csv: Path, output_md: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for tag in _greenlist_tags():
        for season in SEASONS:
            stats_path = NORMALIZED_DIR / f"match_player_stats__{tag}__{season}.csv"
            fixtures_path = NORMALIZED_DIR / f"fixtures_master__{tag}__{season}.csv"
            stats_exists = stats_path.exists()
            fixtures_exists = fixtures_path.exists()
            stats_rows = _count_rows(stats_path)
            fixture_rows = _count_rows(fixtures_path)

            if not stats_exists and not fixtures_exists:
                diagnosis = "ARCHIVE_MISSING"
                note = "No normalized stats or fixtures file exists locally for this league-season."
            elif stats_exists and not fixtures_exists:
                diagnosis = "FIXTURE_FILE_MISSING"
                note = "Stats file exists but the matching fixtures file is missing locally."
            elif not stats_exists and fixtures_exists:
                diagnosis = "STATS_FILE_MISSING"
                note = "Fixtures file exists but the matching player-stats file is missing locally."
            elif stats_rows == 0 and fixture_rows == 0:
                diagnosis = "FILES_EMPTY"
                note = "Both files exist but contain no data rows."
            elif stats_rows == 0:
                diagnosis = "STATS_EMPTY"
                note = "Stats file exists but contains no data rows."
            elif fixture_rows == 0:
                diagnosis = "FIXTURES_EMPTY"
                note = "Fixtures file exists but contains no data rows."
            else:
                diagnosis = "LOCAL_FILES_PRESENT"
                note = "Both normalized files exist locally; any remaining issue would be downstream, not archive absence."

            rows.append(
                {
                    "league_tag": tag,
                    "season": season,
                    "stats_exists": int(stats_exists),
                    "fixtures_exists": int(fixtures_exists),
                    "stats_rows": stats_rows,
                    "fixture_rows": fixture_rows,
                    "diagnosis": diagnosis,
                    "note": note,
                }
            )

    out = pd.DataFrame(rows).sort_values(["league_tag", "season"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Player Events Historical Gap Diagnosis",
        "",
        "- Checks whether the current greenlist historical player-events gaps are missing local normalized files or something subtler like a later join/date issue.",
        "- Scope here is the known weak-coverage leagues: `USA_MLS`, `Europa_League`, `Europa_Conference`.",
        "",
        "## Diagnosis",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"- {row['league_tag']} | season={int(row['season'])} | diagnosis={row['diagnosis']} | "
            f"stats_exists={int(row['stats_exists'])} | fixtures_exists={int(row['fixtures_exists'])} | "
            f"stats_rows={int(row['stats_rows'])} | fixture_rows={int(row['fixture_rows'])}"
        )

    missing = out[out["diagnosis"] != "LOCAL_FILES_PRESENT"]
    lines.extend(
        [
            "",
            "## Call",
        ]
    )
    if missing.empty:
        lines.append("- All target league-seasons have local normalized files, so any remaining coverage issue is likely downstream in joins, filters, or date logic.")
    else:
        lines.append("- The current weak-coverage cells are local archive gaps first, not a hidden match-date join bug.")
        lines.append("- If we want true full-coverage Track A, we likely need additional API Football backfill for those missing league-season files.")
    output_md.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether weak player-events historical coverage is caused by missing archive files or downstream join issues.")
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_historical_gap_diagnosis.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_historical_gap_diagnosis.md"),
    )
    args = parser.parse_args()
    out = build(Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
