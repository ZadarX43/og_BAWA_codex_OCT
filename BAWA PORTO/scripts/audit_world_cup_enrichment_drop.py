#!/usr/bin/env python3
"""Audit added World Cup qualification FootyStats files and Fjelstul archive files."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


DEFAULT_DROP = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP")
DEFAULT_OUTDIR = Path("reports/latest/world_cup_enrichment_drop_audit_2026_05_19")

QUAL_RE = re.compile(
    r"^international-wc-qualification-(?P<region>.+?)-(?P<kind>league|matches|teams|players)-"
    r"(?P<start>\d{4})-to-(?P<end>\d{4})-stats(?: \((?P<dup>\d+)\))?\.csv$"
)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_shape(path: Path) -> tuple[int, int, list[str], str]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return 0, 0, [], str(exc)
    return len(df), len(df.columns), list(df.columns), ""


def audit_qualification(drop: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted(drop.glob("international-wc-qualification-*")):
        if not path.is_file():
            continue
        match = QUAL_RE.match(path.name)
        if not match:
            continue
        info = match.groupdict()
        rows_count, cols_count, columns, err = read_shape(path)
        colset = set(columns)
        rows.append(
            {
                "file": path.name,
                "path": str(path),
                "sha256": file_hash(path),
                "region": info["region"],
                "kind": info["kind"],
                "start_year": int(info["start"]),
                "end_year": int(info["end"]),
                "duplicate_suffix": info.get("dup") or "",
                "rows": rows_count,
                "cols": cols_count,
                "read_error": err,
                "has_odds": int(any(c.startswith("odds_") for c in colset)),
                "has_xg": int(any("xg" in c.lower() for c in colset)),
                "has_player_rating": int("average_rating_overall" in colset),
                "has_market_value": int("market_value" in colset),
                "has_team_ppg": int("points_per_game" in colset),
                "has_match_prematch_ppg": int("Pre-Match PPG (Home)" in colset),
                "columns_sample": "|".join(columns[:40]),
            }
        )
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        return inventory, pd.DataFrame()
    duplicates = inventory.groupby("sha256").filter(lambda g: len(g) > 1).sort_values(["sha256", "file"])
    return inventory, duplicates


def audit_archive(drop: Path) -> pd.DataFrame:
    archive = drop / "archive"
    rows = []
    for path in sorted(archive.glob("*.csv")):
        rows_count, cols_count, columns, err = read_shape(path)
        rows.append(
            {
                "file": path.name,
                "path": str(path),
                "sha256": file_hash(path),
                "rows": rows_count,
                "cols": cols_count,
                "read_error": err,
                "columns_sample": "|".join(columns[:40]),
                "likely_use": archive_use(path.stem),
            }
        )
    return pd.DataFrame(rows)


def archive_use(stem: str) -> str:
    if stem in {"matches", "team_appearances", "qualified_teams", "group_standings", "tournament_standings"}:
        return "HISTORICAL_TEAM_TOURNAMENT_PRIORS"
    if stem in {"squads", "players", "player_appearances"}:
        return "HISTORICAL_PLAYER_SQUAD_PRIORS"
    if stem in {"referees", "referee_appearances", "referee_appointments"}:
        return "HISTORICAL_REFEREE_PRIORS"
    if stem in {"bookings", "goals", "penalty_kicks", "substitutions"}:
        return "HISTORICAL_EVENT_STYLE_PRIORS"
    if stem in {"stadiums", "host_countries"}:
        return "VENUE_HOST_PRIORS"
    return "REFERENCE_METADATA"


def write_summary(outdir: Path, qual: pd.DataFrame, dupes: pd.DataFrame, archive: pd.DataFrame) -> None:
    qual_summary = (
        qual.groupby(["region", "kind", "start_year", "end_year"], dropna=False)
        .agg(
            files=("file", "count"),
            rows=("rows", "max"),
            cols=("cols", "max"),
            has_odds=("has_odds", "max"),
            has_xg=("has_xg", "max"),
            has_player_rating=("has_player_rating", "max"),
            has_market_value=("has_market_value", "max"),
        )
        .reset_index()
        if not qual.empty
        else pd.DataFrame()
    )
    qual_summary.to_csv(outdir / "qualification_file_summary.csv", index=False)
    archive_use_summary = archive.groupby("likely_use").agg(files=("file", "count"), rows=("rows", "sum")).reset_index()
    archive_use_summary.to_csv(outdir / "fjelstul_archive_use_summary.csv", index=False)

    def table(df: pd.DataFrame, cols: list[str]) -> list[str]:
        if df.empty:
            return ["None"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for row in df[cols].itertuples(index=False):
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return lines

    lines = [
        "# World Cup Enrichment Drop Audit",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'qualification_file_inventory.csv'}`",
        f"- `{outdir / 'qualification_file_summary.csv'}`",
        f"- `{outdir / 'qualification_duplicate_files.csv'}`",
        f"- `{outdir / 'fjelstul_archive_inventory.csv'}`",
        f"- `{outdir / 'fjelstul_archive_use_summary.csv'}`",
        "",
        "## Qualification Files",
        "",
        f"- Files: {len(qual)}",
        f"- Duplicate-content files: {len(dupes)}",
        "",
        *table(qual_summary.head(80), ["region", "kind", "start_year", "end_year", "files", "rows", "has_odds", "has_xg", "has_player_rating", "has_market_value"]),
        "",
        "## Fjelstul Archive",
        "",
        f"- Files: {len(archive)}",
        "",
        *table(archive_use_summary, ["likely_use", "files", "rows"]),
        "",
        "## Notes",
        "",
        "- FootyStats qualification `teams` and `players` files are the best immediate 2026 feature source.",
        "- FootyStats qualification `matches` files are useful for historical/backtest calibration and odds/xG context.",
        "- Fjelstul archive is CC-BY-SA 4.0; any productized derivative needs attribution and license review.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop-root", type=Path, default=DEFAULT_DROP)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    qual, dupes = audit_qualification(args.drop_root)
    archive = audit_archive(args.drop_root)
    qual.to_csv(args.outdir / "qualification_file_inventory.csv", index=False)
    dupes.to_csv(args.outdir / "qualification_duplicate_files.csv", index=False)
    archive.to_csv(args.outdir / "fjelstul_archive_inventory.csv", index=False)
    write_summary(args.outdir, qual, dupes, archive)
    print(f"[ok] qualification_files={len(qual)} archive_files={len(archive)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
