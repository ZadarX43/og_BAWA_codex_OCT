#!/usr/bin/env python3
"""Build an API-Football backfill manifest from scored walk-forward rows.

The manifest is intentionally separate from the fetch step so the user can
review the league/season/API-id scope before making any network calls.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCORED_ROOT = Path("predictions_output/hybrid_shadow_walkforward_2026_05_01_parity_rebuild")
DEFAULT_NORMALIZED_DIR = Path("data_sources/api_football/normalized")
DEFAULT_OUT = Path("reports/latest/api_football_2025_2026_backfill_manifest.csv")

FALLBACK_LEAGUE_IDS = {
    "Belgium_Pro": 144,
    "Brazil_Serie_A": 71,
    "Champions_League": 2,
    "England_Championship": 40,
    "England_EFL_League_1": 41,
    "England_FA_Cup": 45,
    "England_Premier_League": 39,
    "Europa_Conference": 848,
    "Europa_League": 3,
    "France_Ligue_1": 61,
    "Germany_Bundesliga": 78,
    "Italy_Serie_A": 135,
    "Japan_J1": 98,
    "Netherlands_Eredivisie": 88,
    "Norway_Eliteserien": 103,
    "Portugal_Liga": 94,
    "Scotland_Premiership": 179,
    "Spain_La_Liga": 140,
    "Switzerland_Super_League": 207,
    "Turkey_Super_Lig": 203,
    "USA_MLS": 253,
}


def league_tag(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def scored_files(root: Path) -> list[Path]:
    return sorted(root.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))


def load_required_cells(scored_root: Path, seasons: set[int]) -> pd.DataFrame:
    wanted = {"league", "match_date", "fixture_key"}
    frames: list[pd.DataFrame] = []
    for path in scored_files(scored_root):
        frame = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["league", "league_tag", "season", "required_scored_rows", "required_unique_fixtures"])
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["match_date_dt"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["season"] = df["match_date_dt"].dt.year.astype("Int64")
    df = df[df["season"].isin(list(seasons))].copy()
    df["league_tag"] = df["league"].map(league_tag)
    out = (
        df.groupby(["league", "league_tag", "season"], dropna=False)
        .agg(required_scored_rows=("fixture_key", "size"), required_unique_fixtures=("fixture_key", "nunique"))
        .reset_index()
    )
    out["season"] = out["season"].astype(int)
    return out


def load_league_id_map(normalized_dir: Path) -> dict[str, int]:
    mapping: dict[str, int] = dict(FALLBACK_LEAGUE_IDS)
    for path in sorted(normalized_dir.glob("fixtures_master__*.csv")):
        try:
            frame = pd.read_csv(path, usecols=lambda c: c in {"league_id"}, nrows=1)
        except Exception:
            continue
        if frame.empty or "league_id" not in frame.columns or pd.isna(frame["league_id"].iloc[0]):
            continue
        stem = path.name.replace("fixtures_master__", "").replace(".csv", "")
        parts = stem.rsplit("__", 1)
        tag = parts[0]
        try:
            mapping[tag] = int(frame["league_id"].iloc[0])
        except Exception:
            continue
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", default=str(DEFAULT_SCORED_ROOT))
    parser.add_argument("--normalized-dir", default=str(DEFAULT_NORMALIZED_DIR))
    parser.add_argument("--seasons", default="2025,2026")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    seasons = {int(x.strip()) for x in args.seasons.split(",") if x.strip()}
    cells = load_required_cells(Path(args.scored_root), seasons)
    id_map = load_league_id_map(Path(args.normalized_dir))
    cells["league_id"] = cells["league_tag"].map(id_map)
    cells["manifest_status"] = cells["league_id"].notna().map(lambda ok: "READY" if ok else "MISSING_LEAGUE_ID")
    cells = cells[
        [
            "manifest_status",
            "league",
            "league_tag",
            "league_id",
            "season",
            "required_scored_rows",
            "required_unique_fixtures",
        ]
    ].sort_values(["manifest_status", "season", "league_tag"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out, index=False)
    print(f"Rows: {len(cells)}")
    print(f"Ready rows: {int(cells['manifest_status'].eq('READY').sum())}")
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
