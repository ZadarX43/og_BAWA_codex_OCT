#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
DEFAULT_QA = ROOT / "ModelStore" / "logs" / "rebuild_all_merged__qa.csv"
DEFAULT_DEDUPE = ROOT / "ModelStore" / "logs" / "full_merged_baseline__dedupe_report.csv"
DEFAULT_BASELINE_REPORT = ROOT / "ModelStore" / "logs" / "full_merged_baseline__merge_report.csv"


REQUIRED_FAMILIES = {
    "core": ["fixture_key__nonnull", "home_team_name__nonnull", "away_team_name__nonnull", "status__nonnull"],
    "ftr": ["odds_ft_home_team_win__nonnull", "odds_ft_draw__nonnull", "odds_ft_away_team_win__nonnull"],
    "btts": ["odds_btts_yes__nonnull", "odds_btts_no__nonnull"],
    "ou25": ["odds_ft_over25__nonnull", "odds_ft_under25__nonnull"],
}

TARGETED_SYNTH_LEAGUES = {"England Premier League", "Scotland Premiership"}
EXPECTED_MISSING_MERGED_WARNINGS = {"Sweden Allsvenskan"}
H2H_MIN_LEAGUES = 20
POWER_MIN_LEAGUES = 25
ROW_COUNT_TOLERANCE = 0.95


def _to_int(value: str | None) -> int:
    try:
        return int(float((value or "0").strip() or "0"))
    except Exception:
        return 0


def _to_float(value: str | None) -> float:
    try:
        return float((value or "0").strip() or "0")
    except Exception:
        return 0.0


def _nonnull_ratio(csv_path: Path, column: str) -> tuple[bool, float]:
    if not csv_path.exists():
        return False, 0.0
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c == column, low_memory=False)
    except Exception:
        return False, 0.0
    if column not in df.columns or len(df) == 0:
        return False, 0.0
    ratio = float(pd.to_numeric(df[column], errors="coerce").notna().mean())
    return True, ratio


def _baseline_rows(baseline_report_path: Path) -> dict[str, int]:
    if not baseline_report_path.exists():
        return {}
    with baseline_report_path.open(newline="", encoding="utf-8", errors="ignore") as f:
        return {row.get("league", ""): _to_int(row.get("rows")) for row in csv.DictReader(f)}


def evaluate(
    qa_path: Path,
    dedupe_path: Path,
    baseline_report_path: Path,
) -> tuple[bool, list[str], list[str], dict[str, int]]:
    problems: list[str] = []
    warnings: list[str] = []
    stats = {"leagues": 0, "duplicates": 0, "h2h_ok": 0, "power_ok": 0}

    if not qa_path.exists():
        return False, [f"Missing QA report: {qa_path}"], warnings, stats

    with qa_path.open(newline="", encoding="utf-8", errors="ignore") as f:
        qa_rows = list(csv.DictReader(f))

    if not qa_rows:
        return False, ["QA report is empty"], warnings, stats

    stats["leagues"] = len(qa_rows)
    baseline_rows = _baseline_rows(baseline_report_path)
    synth_seen = {}

    for row in qa_rows:
        league = row.get("league", "UNKNOWN")
        exists = str(row.get("exists", "False")) == "True"
        if not exists:
            if league in EXPECTED_MISSING_MERGED_WARNINGS:
                warnings.append(f"{league}: merged missing (expected)")
            else:
                warnings.append(f"{league}: merged missing")
            continue

        merged_path = Path(row.get("merged_path", ""))
        current_rows = _to_int(row.get("rows"))
        previous_rows = baseline_rows.get(league)
        if previous_rows:
            min_rows = max(int(previous_rows * ROW_COUNT_TOLERANCE), 1)
            if current_rows < min_rows:
                problems.append(f"{league}: row count drop {current_rows} < {min_rows}")

        for family, cols in REQUIRED_FAMILIES.items():
            if any(_to_int(row.get(col)) <= 0 for col in cols):
                problems.append(f"{league}: {family} coverage fail")
                break

        has_h2h, h2h_ratio = _nonnull_ratio(merged_path, "h2h_btts_rate")
        if has_h2h and h2h_ratio > 0.0:
            stats["h2h_ok"] += 1

        has_power, power_ratio = _nonnull_ratio(merged_path, "home_power_rating")
        if has_power and power_ratio >= 0.8:
            stats["power_ok"] += 1

        if league in TARGETED_SYNTH_LEAGUES:
            has_synth, synth_ratio = _nonnull_ratio(merged_path, "p_over25_novig")
            synth_seen[league] = synth_ratio if has_synth else 0.0
            if not has_synth or synth_ratio <= 0.0:
                problems.append(f"{league}: targeted synth missing")

    if dedupe_path.exists():
        with dedupe_path.open(newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                dup_total = sum(
                    _to_int(row.get(key))
                    for key in ("fixture_key_dup_rows", "exact_dup_rows", "home_away_date_dup_rows")
                )
                if dup_total > 0:
                    stats["duplicates"] += 1
                    problems.append(f"{row.get('league', 'UNKNOWN')}: duplicate rows {dup_total}")

    if stats["h2h_ok"] < H2H_MIN_LEAGUES:
        problems.append(f"h2h coverage only {stats['h2h_ok']} leagues < {H2H_MIN_LEAGUES}")
    if stats["power_ok"] < POWER_MIN_LEAGUES:
        problems.append(f"power coverage only {stats['power_ok']} leagues < {POWER_MIN_LEAGUES}")

    for league in TARGETED_SYNTH_LEAGUES:
        if league not in synth_seen:
            problems.append(f"{league}: targeted synth league not evaluated")

    return len(problems) == 0, problems, warnings, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate prediction/deploy runs on merged QA health.")
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA)
    parser.add_argument("--dedupe-path", type=Path, default=DEFAULT_DEDUPE)
    parser.add_argument("--baseline-report-path", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="Always exit 0 and report PASS/FAIL in stdout for GUI/orchestration callers.",
    )
    args = parser.parse_args()

    ok, problems, warnings, stats = evaluate(args.qa_path, args.dedupe_path, args.baseline_report_path)
    if ok:
        if warnings:
            preview = "; ".join(warnings[:4])
            if len(warnings) > 4:
                preview += f"; +{len(warnings) - 4} more"
            print(
                f"WARN|QA gate passed with warnings | leagues={stats['leagues']} | "
                f"duplicate_leagues={stats['duplicates']} | h2h_ok={stats['h2h_ok']} | "
                f"power_ok={stats['power_ok']} | {preview}"
            )
        else:
            print(
                f"PASS|QA gate clean for predictions | leagues={stats['leagues']} | "
                f"duplicate_leagues={stats['duplicates']} | h2h_ok={stats['h2h_ok']} | power_ok={stats['power_ok']}"
            )
        return 0

    preview = "; ".join(problems[:6])
    if len(problems) > 6:
        preview += f"; +{len(problems) - 6} more"
    print(f"FAIL|{preview}")
    return 0 if args.no_fail_exit else 2


if __name__ == "__main__":
    raise SystemExit(main())
