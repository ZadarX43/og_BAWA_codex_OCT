from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYER_EVENTS_DIR = REPO_ROOT / "scripts" / "player_events"
FEATURES_DIR = REPO_ROOT / "data_sources" / "api_football" / "features" / "player_events"
COMBINED_BOARDS_DIR = REPO_ROOT / "reports" / "player_events" / "combined_boards"
QUALITY_AUDITS_DIR = REPO_ROOT / "reports" / "player_events" / "quality_audits"
NORMALIZED_DIR = REPO_ROOT / "data_sources" / "api_football" / "normalized"

LEAGUES = ["Italy_Serie_A", "Spain_La_Liga", "Europa_League"]
FAMILY_CONFIGS = {
    "4231v442": {
        "title": "4-2-3-1 vs 4-4-2 Weekend Board",
        "forms": "4-2-3-1 vs 4-4-2,4-4-2 vs 4-2-3-1",
        "super": "formation_4231_vs_442__super_elite_board.csv",
        "attacking": "formation_4231_vs_442__attacking_board.csv",
        "contact": "formation_4231_vs_442__contact_board.csv",
        "merged": "formation_4231_vs_442__final_weekend_merge_board.csv",
        "merged_md": "formation_4231_vs_442__final_weekend_merge_board.md",
    },
    "4231v433": {
        "title": "4-2-3-1 vs 4-3-3 Weekend Board",
        "forms": "4-2-3-1 vs 4-3-3,4-3-3 vs 4-2-3-1",
        "super": "formation_4231_vs_433__super_elite_board.csv",
        "attacking": "formation_4231_vs_433__attacking_board.csv",
        "contact": "formation_4231_vs_433__contact_board.csv",
        "merged": "formation_4231_vs_433__final_weekend_merge_board.csv",
        "merged_md": "formation_4231_vs_433__final_weekend_merge_board.md",
    },
    "3421v4231": {
        "title": "3-4-2-1 vs 4-2-3-1 Weekend Board",
        "forms": "3-4-2-1 vs 4-2-3-1,4-2-3-1 vs 3-4-2-1",
        "super": "formation_3421_vs_4231__super_elite_board.csv",
        "attacking": "formation_3421_vs_4231__attacking_board.csv",
        "contact": "formation_3421_vs_4231__contact_board.csv",
        "merged": "formation_3421_vs_4231__final_weekend_merge_board.csv",
        "merged_md": "formation_3421_vs_4231__final_weekend_merge_board.md",
    },
}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _merge_fixture_inputs() -> None:
    paths = [FEATURES_DIR / f"player_events_fixture_input__{league}__2024.csv" for league in LEAGUES]
    frames = [pd.read_csv(path, low_memory=False) for path in paths if path.exists()]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(FEATURES_DIR / "player_events_fixture_input__EUROPA_LALIGA_SERIEA__2024.csv", index=False)


def run_refresh() -> Path:
    for league in LEAGUES:
        _run([
            sys.executable,
            str(PLAYER_EVENTS_DIR / "build_player_events_fixture_input.py"),
            "--league-tag",
            league,
            "--season",
            "2024",
        ])
    _merge_fixture_inputs()

    combined_input = FEATURES_DIR / "player_events_fixture_input__EUROPA_LALIGA_SERIEA__2024.csv"
    for family, cfg in FAMILY_CONFIGS.items():
        _run([
            sys.executable,
            str(PLAYER_EVENTS_DIR / "build_4231_vs_442_weekend_board.py"),
            "--input",
            str(combined_input),
            "--output-csv",
            str(COMBINED_BOARDS_DIR / cfg["contact"]),
            "--output-md",
            str(COMBINED_BOARDS_DIR / cfg["contact"].replace(".csv", ".md")),
            "--target-formations",
            cfg["forms"],
            "--title",
            cfg["title"],
            "--include-contact",
        ])
        _run([
            sys.executable,
            str(PLAYER_EVENTS_DIR / "build_4231_vs_442_final_weekend_merge_board.py"),
            "--super-csv",
            str(COMBINED_BOARDS_DIR / cfg["super"]),
            "--attacking-csv",
            str(COMBINED_BOARDS_DIR / cfg["attacking"]),
            "--contact-csv",
            str(COMBINED_BOARDS_DIR / cfg["contact"]),
            "--output-csv",
            str(COMBINED_BOARDS_DIR / cfg["merged"]),
            "--output-md",
            str(COMBINED_BOARDS_DIR / cfg["merged_md"]),
            "--title",
            cfg["title"].replace("Weekend Board", "Final Weekend Merge Board"),
        ])

    merged_inputs = ",".join(str(COMBINED_BOARDS_DIR / FAMILY_CONFIGS[f]["merged"]) for f in FAMILY_CONFIGS)
    family_tags = ",".join(FAMILY_CONFIGS.keys())
    _run([
        sys.executable,
        str(PLAYER_EVENTS_DIR / "build_master_specialist_board.py"),
        "--inputs",
        merged_inputs,
        "--family-tags",
        family_tags,
        "--output-csv",
        str(COMBINED_BOARDS_DIR / "master_specialist_board.csv"),
        "--output-md",
        str(COMBINED_BOARDS_DIR / "master_specialist_board.md"),
    ])
    _run([
        sys.executable,
        str(PLAYER_EVENTS_DIR / "build_specialist_final_shortlist_export.py"),
        "--inputs",
        str(COMBINED_BOARDS_DIR / "master_specialist_board.csv"),
        "--output-csv",
        str(COMBINED_BOARDS_DIR / "specialist_final_shortlist_export.csv"),
        "--output-md",
        str(COMBINED_BOARDS_DIR / "specialist_final_shortlist_export.md"),
    ])
    _run([
        sys.executable,
        str(PLAYER_EVENTS_DIR / "build_master_weekend_specialist_sheet.py"),
        "--input-csv",
        str(COMBINED_BOARDS_DIR / "master_specialist_board.csv"),
        "--output-csv",
        str(COMBINED_BOARDS_DIR / "master_weekend_specialist_sheet.csv"),
        "--output-md",
        str(COMBINED_BOARDS_DIR / "master_weekend_specialist_sheet.md"),
        "--max-fixtures",
        "12",
    ])
    _run([
        sys.executable,
        str(PLAYER_EVENTS_DIR / "build_confirmed_lineups_pre_kickoff_helper.py"),
        "--master-weekend-csv",
        str(COMBINED_BOARDS_DIR / "master_weekend_specialist_sheet.csv"),
        "--fixture-input-csv",
        str(combined_input),
        "--output-csv",
        str(COMBINED_BOARDS_DIR / "confirmed_lineups_pre_kickoff_helper.csv"),
        "--output-md",
        str(COMBINED_BOARDS_DIR / "confirmed_lineups_pre_kickoff_helper.md"),
    ])

    summary_path = COMBINED_BOARDS_DIR / "confirmed_lineups_override_refresh_summary.md"
    weekend = pd.read_csv(COMBINED_BOARDS_DIR / "master_weekend_specialist_sheet.csv", low_memory=False)
    helper = pd.read_csv(COMBINED_BOARDS_DIR / "confirmed_lineups_pre_kickoff_helper.csv", low_memory=False)
    lines = [
        "# Confirmed-Lineups Override Refresh Summary",
        "",
        "- rebuilt fixture inputs for: Italy_Serie_A, Spain_La_Liga, Europa_League",
        "- reran specialist contact boards and merged master weekend outputs",
        f"- master weekend rows: {len(weekend)} | fixtures: {weekend['fixture_key'].nunique() if not weekend.empty else 0}",
        f"- helper rows still needing side confirmation: {len(helper)} | fixtures: {helper['fixture_key'].nunique() if not helper.empty else 0}",
        "",
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply confirmed-lineups manual side tags and refresh elite specialist outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    summary = run_refresh()
    print(f"WROTE: {summary}")
