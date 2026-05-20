#!/usr/bin/env python3
from pathlib import Path
import csv
import shutil
import re
import sys
import subprocess
from collections import Counter, defaultdict

# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
DROP_FOLDER = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP")
ARCHIVE_FOLDER = DROP_FOLDER / "_processed"
LAST_RUN_SUMMARY_PATH = DROP_FOLDER / "_last_run_summary.txt"
LAST_RUN_DETAIL_CSV_PATH = DROP_FOLDER / "_last_run_detail.csv"

# First run with DRY_RUN = True to test safely.
DRY_RUN = False

# If True, remove old CSVs in the destination folder before copying new one in.
REPLACE_EXISTING = True

# If True, move processed files into _processed.
# If False, leave originals in the drop folder.
ARCHIVE_PROCESSED_FILES = True


# If True, clean duplicate suffixes like " (1)" from destination filename.
CLEAN_DUPLICATE_SUFFIX = True

# If True, only replace the existing FootyStats raw export for the same
# league + file type, instead of deleting every CSV in the destination folder.
REPLACE_ONLY_MATCHING_RAW_EXPORT = True

# If True, keep a backup copy of any replaced raw export in a _replaced folder
# inside the destination folder before writing the new file.
BACKUP_REPLACED_FILE = True


# Only accept the latest FootyStats season exports.
# Two-season leagues must be 2025-to-2026.
LATEST_MULTI_SEASON_START_YEAR = 2025
LATEST_MULTI_SEASON_END_YEAR = 2026

# Single-calendar-year leagues must be 2026-to-2026.
LATEST_SINGLE_YEAR = 2026

# UX / launcher polish
OPEN_DROP_FOLDER_ON_START = True
ENABLE_COLOR_OUTPUT = True
PLAY_SOUND_ON_COMPLETE = True
SUCCESS_SOUND_PATH = "/System/Library/Sounds/Glass.aiff"
ERROR_SOUND_PATH = "/System/Library/Sounds/Basso.aiff"

# ============================================================
# LEAGUE MAPPING
# key = substring expected in incoming filename
# value = target league folder name in your repo
# ============================================================

LEAGUE_MAP = {
    "england-premier-league": "England Premier League",
    "england-championship": "England Championship",
    "england-efl-league-one": "England EFL League 1",
    "england-fa-cup": "England FA Cup",
    "japan-j1-league": "Japan J1",
    "norway-eliteserien": "Norway Eliteserien",
    "netherlands-eredivisie": "Netherlands Eredivisie",
    "belgium-pro-league": "Belgium Pro",
    "scotland-premiership": "Scotland Premiership",
    "brazil-serie-a": "Brazil Serie A",
    "usa-mls": "USA MLS",
    "portugal-liga-nos": "Portugal Liga",
    "spain-la-liga": "Spain La Liga",
    "italy-serie-a": "Italy Serie A",
    "france-ligue-1": "France Ligue 1",
    "germany-bundesliga": "Germany Bundesliga",
    "europe-uefa-europa-conference-league": "Europa Conference",
    "europe-uefa-europa-league": "Europa League",
    "europe-uefa-champions-league": "Champions League",
    "australia-a-league": "Australia A-League",
    "austria-bundesliga": "Austria Bundesliga",
    "denmark-superliga": "Denmark Superliga",
    "switzerland-super-league": "Swiss Super League",
    "sweden-allsvenskan": "Sweden Allsvenskan",
    "germany-2-bundesliga": "Germany Bundesliga 2",
    "czech-republic-first-league": "Czech First League",
    "south-korea-k-league-1": "South Korea K League",
    "saudi-arabia-pro-league": "Saudi Pro League",
    "saudi-arabia-professional-league": "Saudi Pro League",
    "turkey-super-lig": "Turkey Super Lig",
}

VALID_TYPES = {"matches": "Matches", "teams": "Teams", "players": "Players"}

APPROVED_SINGLE_YEAR_SLUGS = {
    "japan-j1-league",
    "norway-eliteserien",
    "brazil-serie-a",
    "usa-mls",
    "south-korea-k-league-1",
}

RAW_EXPORT_RE = re.compile(
    r"^(?P<slug>[a-z0-9-]+)-(?P<file_type>matches|teams|players)-(?P<start_year>\d{4})-to-(?P<end_year>\d{4})-stats(?: \((?P<dup>\d+)\))?\.csv$",
    flags=re.IGNORECASE,
)

# ============================================================

# ANSI terminal colors
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"

# HELPERS
# ============================================================

def normalise_name(name: str) -> str:
    return name.strip().lower()


def supports_color() -> bool:
    return ENABLE_COLOR_OUTPUT and sys.stdout.isatty()


def colour(text: str, code: str) -> str:
    if not supports_color():
        return text
    return f"{code}{text}{RESET}"


def print_header(title: str) -> None:
    line = "=" * 72
    print("\n" + colour(line, BLUE))
    print(colour(title, BOLD + CYAN))
    print(colour(line, BLUE))


def play_completion_sound(success: bool) -> None:
    if not PLAY_SOUND_ON_COMPLETE:
        return

    sound_path = SUCCESS_SOUND_PATH if success else ERROR_SOUND_PATH
    try:
        subprocess.run(["afplay", sound_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        try:
            subprocess.run(["osascript", "-e", "beep"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def open_drop_folder() -> None:
    if not OPEN_DROP_FOLDER_ON_START:
        return
    try:
        subprocess.run(["open", str(DROP_FOLDER)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def clean_destination_filename(filename: str) -> str:
    """
    Remove duplicate macOS/browser suffixes like:
    'file (1).csv' -> 'file.csv'
    """
    if not CLEAN_DUPLICATE_SUFFIX:
        return filename
    return re.sub(r"\s\(\d+\)(?=\.csv$)", "", filename, flags=re.IGNORECASE)


def parse_raw_export_filename(filename: str) -> dict | None:
    """
    Parse only approved FootyStats raw export filenames.
    Reject anything outside the approved 19 league families.
    """
    cleaned = clean_destination_filename(filename)
    match = RAW_EXPORT_RE.match(cleaned)
    if not match:
        return None

    slug = match.group("slug").lower()
    file_type = match.group("file_type").lower()
    start_year = int(match.group("start_year"))
    end_year = int(match.group("end_year"))

    if slug not in LEAGUE_MAP:
        return None

    return {
        "slug": slug,
        "file_type": file_type,
        "start_year": start_year,
        "end_year": end_year,
        "league_folder": LEAGUE_MAP[slug],
    }


def is_latest_allowed_raw_export(parsed: dict) -> bool:
    """
    Only allow the latest season files:
    - multi-season leagues: 2025-to-2026
    - single-year leagues: 2026-to-2026
    """
    slug = parsed["slug"]
    start_year = parsed["start_year"]
    end_year = parsed["end_year"]

    if slug in APPROVED_SINGLE_YEAR_SLUGS:
        return start_year == LATEST_SINGLE_YEAR and end_year == LATEST_SINGLE_YEAR

    return (
        start_year == LATEST_MULTI_SEASON_START_YEAR
        and end_year == LATEST_MULTI_SEASON_END_YEAR
    )


# Helper to sort existing raw exports by newest season first
def existing_raw_export_sort_key(parsed: dict) -> tuple[int, int]:
    """
    Sort existing raw exports so the newest season wins.
    For example:
    - 2025-to-2026 > 2024-to-2025 > 2018-to-2019
    - 2026-to-2026 > 2025-to-2025 > 2018-to-2018
    """
    return (parsed["end_year"], parsed["start_year"])


# Helper to normalise FootyStats raw export filenames for matching
def canonical_raw_export_name(filename: str) -> str | None:
    """
    Normalise an approved FootyStats raw export filename so we can match only the
    raw export for the same league + type, regardless of season years or duplicate suffixes.

    Return None for anything that is not an approved raw export.
    """
    parsed = parse_raw_export_filename(filename)
    if parsed is None:
        return None
    return f"{parsed['slug']}-{parsed['file_type']}-stats.csv"


def find_existing_matching_raw_export(folder: Path, incoming_filename: str) -> Path | None:
    """
    Return the newest existing approved raw FootyStats export in the folder that matches
    the same league + file type family as the incoming file. Ignore duplicate download
    suffixes and do not match unrelated CSVs such as merged outputs.
    """
    incoming_parsed = parse_raw_export_filename(incoming_filename)
    if incoming_parsed is None:
        return None

    candidates: list[tuple[tuple[int, int], Path]] = []

    for csv_file in sorted(folder.glob("*.csv")):
        existing_parsed = parse_raw_export_filename(csv_file.name)
        if existing_parsed is None:
            continue

        if existing_parsed["slug"] != incoming_parsed["slug"]:
            continue

        if existing_parsed["file_type"] != incoming_parsed["file_type"]:
            continue

        candidates.append((existing_raw_export_sort_key(existing_parsed), csv_file))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def replace_existing_raw_export(folder: Path, incoming_filename: str) -> list[Path]:
    """
    Replace only the matching FootyStats raw export for the same league + type.
    Leave all other CSVs in the folder untouched.
    """
    removed = []
    existing = find_existing_matching_raw_export(folder, incoming_filename)
    if existing is None:
        return removed

    incoming_clean_name = clean_destination_filename(incoming_filename)
    existing_clean_name = clean_destination_filename(existing.name)

    # If the destination already contains the same logical file name, we still want
    # to overwrite it with the newly downloaded copy, but we should not report that we
    # are replacing some unrelated older season.
    removed.append(existing)

    if not DRY_RUN:
        if BACKUP_REPLACED_FILE:
            backup_dir = folder / "_replaced"
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, backup_dir / existing.name)
        existing.unlink()

    return removed


def target_folder_for(file_type: str, league_folder: str) -> Path:
    category_folder = VALID_TYPES[file_type]
    return REPO_ROOT / category_folder / league_folder


# Legacy full-folder replacement helper. Prefer replace_existing_raw_export().
def remove_existing_csvs(folder: Path) -> list[Path]:
    removed = []
    for csv_file in folder.glob("*.csv"):
        removed.append(csv_file)
        if not DRY_RUN:
            csv_file.unlink()
    return removed


def ensure_folder(path: Path) -> None:
    if not DRY_RUN:
        path.mkdir(parents=True, exist_ok=True)


def copy_or_move_file(src: Path, dst: Path) -> None:
    if DRY_RUN:
        return
    shutil.copy2(src, dst)


def archive_source_file(src: Path, archive_dest: Path) -> None:
    if DRY_RUN:
        return
    archive_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(archive_dest))


def write_run_summary(
    files: list[Path],
    moved: list[dict],
    skipped: list[tuple[str, str]],
    errors: list[tuple[str, str]],
) -> None:
    """Persist a compact run summary so the launcher/app can surface it later."""
    league_counter = Counter(item["league"] for item in moved)
    type_counter = Counter(item["type"] for item in moved)
    skip_reason_counter = Counter(reason for _, reason in skipped)
    by_league: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"processed": [], "skipped": []})

    for item in moved:
        by_league[item["league"]]["processed"].append(item["source"])

    for filename, reason in skipped:
        parsed = parse_raw_export_filename(filename)
        league = parsed["league_folder"] if parsed else "UNMAPPED / SKIPPED"
        by_league[league]["skipped"].append(f"{filename} [{reason}]")

    summary_lines = [
        "FOOTYSTATS DROP INGEST - LAST RUN SUMMARY",
        "=" * 72,
        f"Total CSV files seen: {len(files)}",
        f"Processed: {len(moved)}",
        f"Skipped: {len(skipped)}",
        f"Errors: {len(errors)}",
        f"Dry run: {DRY_RUN}",
        "",
        "Processed by league:",
    ]

    if league_counter:
        for league, count in sorted(league_counter.items()):
            summary_lines.append(f"- {league}: {count}")
    else:
        summary_lines.append("- none")

    summary_lines.extend(["", "Processed by file type:"])
    if type_counter:
        for file_type, count in sorted(type_counter.items()):
            summary_lines.append(f"- {file_type}: {count}")
    else:
        summary_lines.append("- none")

    summary_lines.extend(["", "Skip reasons:"])
    if skip_reason_counter:
        for reason, count in sorted(skip_reason_counter.items(), key=lambda item: (-item[1], item[0])):
            summary_lines.append(f"- {reason}: {count}")
    else:
        summary_lines.append("- none")

    if errors:
        summary_lines.extend(["", "Errors:"])
        for filename, reason in errors:
            summary_lines.append(f"- {filename}: {reason}")

    summary_lines.extend(["", "League detail:"])
    for league in sorted(by_league):
        processed_items = by_league[league]["processed"]
        skipped_items = by_league[league]["skipped"]
        summary_lines.append(f"- {league}")
        summary_lines.append(f"  processed: {len(processed_items)}")
        summary_lines.append(f"  skipped: {len(skipped_items)}")
        if processed_items:
            summary_lines.append("  processed files:")
            for name in processed_items:
                summary_lines.append(f"    - {name}")
        if skipped_items:
            summary_lines.append("  skipped files:")
            for detail in skipped_items:
                summary_lines.append(f"    - {detail}")

    if not DRY_RUN:
        try:
            LAST_RUN_SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

            with LAST_RUN_DETAIL_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["status", "league", "file_type", "filename", "detail"])
                for item in moved:
                    writer.writerow(["processed", item["league"], item["type"], item["source"], item["destination"]])
                for filename, reason in skipped:
                    parsed = parse_raw_export_filename(filename)
                    league = parsed["league_folder"] if parsed else ""
                    file_type = parsed["file_type"] if parsed else ""
                    writer.writerow(["skipped", league, file_type, filename, reason])
                for filename, reason in errors:
                    parsed = parse_raw_export_filename(filename)
                    league = parsed["league_folder"] if parsed else ""
                    file_type = parsed["file_type"] if parsed else ""
                    writer.writerow(["error", league, file_type, filename, reason])
        except Exception as exc:
            print(colour(f"WARNING: could not write last-run summary files: {exc}", YELLOW))


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    if not DROP_FOLDER.exists():
        print(f"ERROR: Drop folder does not exist: {DROP_FOLDER}")
        return 1

    if not REPO_ROOT.exists():
        print(f"ERROR: Repo root does not exist: {REPO_ROOT}")
        return 1

    open_drop_folder()

    files = [p for p in DROP_FOLDER.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]

    if not files:
        print(colour(f"No CSV files found in: {DROP_FOLDER}", YELLOW))
        play_completion_sound(success=True)
        return 0

    moved = []
    skipped = []
    errors = []

    for src in sorted(files):
        try:
            parsed = parse_raw_export_filename(src.name)

            if parsed is None:
                skipped.append((src.name, "not an approved FootyStats raw export for the 19 configured leagues"))
                continue

            if not is_latest_allowed_raw_export(parsed):
                skipped.append((src.name, "not the latest allowed season/year export"))
                continue

            detected_type = parsed["file_type"]
            detected_league = parsed["league_folder"]

            destination_folder = target_folder_for(detected_type, detected_league)

            if not destination_folder.exists():
                skipped.append((src.name, f"target folder does not exist: {destination_folder}"))
                continue

            ensure_folder(destination_folder)

            removed_files = []
            if REPLACE_EXISTING:
                if REPLACE_ONLY_MATCHING_RAW_EXPORT:
                    removed_files = replace_existing_raw_export(destination_folder, src.name)
                else:
                    removed_files = remove_existing_csvs(destination_folder)

            clean_name = clean_destination_filename(src.name)
            destination_file = destination_folder / clean_name

            copy_or_move_file(src, destination_file)

            if ARCHIVE_PROCESSED_FILES:
                archive_dest = ARCHIVE_FOLDER / src.name
                archive_source_file(src, archive_dest)

            moved.append({
                "source": src.name,
                "type": detected_type,
                "league": detected_league,
                "destination": str(destination_file),
                "removed": [str(p.name) for p in removed_files],
            })

        except Exception as e:
            errors.append((src.name, str(e)))

    print_header("FOOTYSTATS DROP INGEST SUMMARY")
    print(f"DRY_RUN = {DRY_RUN}")
    print(f"REPLACE_EXISTING = {REPLACE_EXISTING}")
    print(f"REPLACE_ONLY_MATCHING_RAW_EXPORT = {REPLACE_ONLY_MATCHING_RAW_EXPORT}")
    print(f"BACKUP_REPLACED_FILE = {BACKUP_REPLACED_FILE}")
    print(f"ARCHIVE_PROCESSED_FILES = {ARCHIVE_PROCESSED_FILES}")
    print(f"LATEST_MULTI_SEASON = {LATEST_MULTI_SEASON_START_YEAR}-to-{LATEST_MULTI_SEASON_END_YEAR}")
    print(f"LATEST_SINGLE_YEAR = {LATEST_SINGLE_YEAR}-to-{LATEST_SINGLE_YEAR}")
    print()

    if moved:
        moved_title = "WOULD PROCESS:" if DRY_RUN else "PROCESSED:"
        print(colour(moved_title, GREEN))
        for item in moved:
            print(colour(f"- {item['source']}", GREEN))
            print(f"  -> {item['destination']}")
            print(f"  type: {item['type']} | league: {item['league']}")
            if item["removed"]:
                print(f"  replaced existing: {', '.join(item['removed'])}")
            else:
                print(f"  replaced existing: none")
        print()

    if skipped:
        print(colour("SKIPPED:", YELLOW))
        for filename, reason in skipped:
            print(colour(f"- {filename} | reason: {reason}", YELLOW))
        print()

    if errors:
        print(colour("ERRORS:", RED))
        for filename, reason in errors:
            print(colour(f"- {filename} | error: {reason}", RED))
        print()

    processed_label = "Would process" if DRY_RUN else "Processed"
    print(colour("RUN FOOTER", BOLD + CYAN))
    print(f"Total files seen: {len(files)}")
    print(colour(f"{processed_label}: {len(moved)}", GREEN))
    print(colour(f"Skipped: {len(skipped)}", YELLOW if skipped else GREEN))
    print(colour(f"Errors: {len(errors)}", RED if errors else GREEN))
    print(colour("=" * 72, BLUE))

    write_run_summary(files, moved, skipped, errors)

    success = len(errors) == 0
    play_completion_sound(success=success)
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
