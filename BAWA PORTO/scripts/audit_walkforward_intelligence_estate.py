#!/usr/bin/env python3
"""Audit whether a walk-forward root has enough intelligence estate coverage.

Research-only. This script does not generate predictions, train models, mutate
deploy gates, or edit any walk-forward outputs. It inventories scored
walk-forward rows and local API-Football feature families so an overnight
intelligence-overlay run has an auditable coverage contract before interpretation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCORED_ROOT = Path("predictions_output/walk_forward_greenlist_btts_no_recent_regime_2026_05_06")
DEFAULT_FEATURES_DIR = Path("data_sources/api_football/features")
DEFAULT_NORMALIZED_DIR = Path("data_sources/api_football/normalized")
DEFAULT_OUTDIR = Path("reports/latest/walkforward_intelligence_estate_audit")

FEATURE_FAMILIES = [
    "team_rolling_features",
    "player_rolling_features",
    "lineup_features",
    "injury_features",
    "matchup_interaction_features",
    "event_features",
    "h2h_regime_features",
    "odds_features",
    "referee_profile_features",
]

NORMALIZED_FAMILIES = [
    "fixtures_master",
    "match_team_stats",
    "match_player_stats",
    "lineups",
    "injuries",
    "sidelined",
    "odds_prematch_long",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def markdown_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "_No rows._"
    text = df.head(max_rows).copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def row_count(path: Path) -> int:
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
    except OSError:
        return 0


def league_tag(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def parse_family_file(path: Path, prefix: str) -> tuple[str, int] | None:
    match = re.match(rf"api_{re.escape(prefix)}__(.+)__(\d{{4}})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def parse_normalized_file(path: Path, prefix: str) -> tuple[str, int] | None:
    match = re.match(rf"{re.escape(prefix)}__(.+)__(\d{{4}})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def scored_files(root: Path, max_files: int = 0) -> list[Path]:
    files = sorted(root.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))
    if max_files > 0:
        return files[:max_files]
    return files


def inventory_scored(root: Path, max_files: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    league_rows: list[pd.DataFrame] = []
    for path in scored_files(root, max_files=max_files):
        df = safe_read_csv(
            path,
            usecols=lambda c: c
            in {
                "window_id",
                "league",
                "match_date",
                "fixture_key",
                "market",
                "deploy_tier",
                "tier",
                "home_team_name",
                "away_team_name",
            },
            low_memory=False,
        )
        if df.empty:
            rows.append({"path": str(path), "rows": 0, "read_status": "EMPTY_OR_READ_ERROR"})
            continue
        if "window_id" not in df.columns:
            df["window_id"] = path.parts[-3] if len(path.parts) >= 3 else ""
        window_id = str(df["window_id"].dropna().iloc[0])
        df["__year"] = pd.to_datetime(df.get("match_date"), errors="coerce").dt.year
        df["__league_tag"] = df["league"].map(league_tag)
        rows.append(
            {
                "window_id": window_id,
                "path": str(path),
                "rows": int(len(df)),
                "fixtures": int(df["fixture_key"].nunique()) if "fixture_key" in df.columns else 0,
                "leagues": int(df["league"].nunique()) if "league" in df.columns else 0,
                "markets": ",".join(sorted(df["market"].dropna().astype(str).unique())) if "market" in df.columns else "",
                "read_status": "OK",
            }
        )
        league_rows.append(df[["window_id", "league", "__league_tag", "__year", "fixture_key", "market"]].copy())
    inventory = pd.DataFrame(rows)
    scored = pd.concat(league_rows, ignore_index=True, sort=False) if league_rows else pd.DataFrame()
    return inventory, scored


def inventory_feature_files(features_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in FEATURE_FAMILIES:
        for path in sorted(features_dir.glob(f"api_{family}__*.csv")):
            parsed = parse_family_file(path, family)
            if parsed is None:
                continue
            tag, season = parsed
            header = safe_read_csv(path, nrows=0)
            rows.append(
                {
                    "source_layer": "features",
                    "family": family,
                    "league_tag": tag,
                    "season": season,
                    "rows": row_count(path),
                    "columns": int(len(header.columns)) if not header.empty or len(header.columns) else 0,
                    "path": str(path),
                }
            )
    return pd.DataFrame(rows)


def inventory_normalized_files(normalized_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in NORMALIZED_FAMILIES:
        for path in sorted(normalized_dir.glob(f"{family}__*.csv")):
            parsed = parse_normalized_file(path, family)
            if parsed is None:
                continue
            tag, season = parsed
            header = safe_read_csv(path, nrows=0)
            rows.append(
                {
                    "source_layer": "normalized",
                    "family": family,
                    "league_tag": tag,
                    "season": season,
                    "rows": row_count(path),
                    "columns": int(len(header.columns)) if not header.empty or len(header.columns) else 0,
                    "path": str(path),
                }
            )
    return pd.DataFrame(rows)


def build_coverage(scored: pd.DataFrame, feature_inventory: pd.DataFrame, normalized_inventory: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    cells = (
        scored.dropna(subset=["league", "__league_tag", "__year"])
        .assign(season=lambda d: pd.to_numeric(d["__year"], errors="coerce").astype("Int64"))
        .groupby(["league", "__league_tag", "season"], dropna=False)
        .agg(
            scored_rows=("fixture_key", "size"),
            scored_fixtures=("fixture_key", "nunique"),
            markets=("market", lambda s: ",".join(sorted(set(str(v) for v in s.dropna())))),
            windows=("window_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"__league_tag": "league_tag"})
    )

    feature_pivot = pd.DataFrame()
    if not feature_inventory.empty:
        feature_pivot = (
            feature_inventory.pivot_table(
                index=["league_tag", "season"],
                columns="family",
                values="rows",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        for family in FEATURE_FAMILIES:
            if family not in feature_pivot.columns:
                feature_pivot[family] = 0
            feature_pivot[f"{family}_rows"] = pd.to_numeric(feature_pivot[family], errors="coerce").fillna(0).astype(int)
            feature_pivot = feature_pivot.drop(columns=[family])

    normalized_pivot = pd.DataFrame()
    if not normalized_inventory.empty:
        normalized_pivot = (
            normalized_inventory.pivot_table(
                index=["league_tag", "season"],
                columns="family",
                values="rows",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        for family in NORMALIZED_FAMILIES:
            if family not in normalized_pivot.columns:
                normalized_pivot[family] = 0
            normalized_pivot[f"normalized_{family}_rows"] = pd.to_numeric(normalized_pivot[family], errors="coerce").fillna(0).astype(int)
            normalized_pivot = normalized_pivot.drop(columns=[family])

    out = cells.copy()
    if not feature_pivot.empty:
        out = out.merge(feature_pivot, on=["league_tag", "season"], how="left")
    if not normalized_pivot.empty:
        out = out.merge(normalized_pivot, on=["league_tag", "season"], how="left")
    row_cols = [col for col in out.columns if col.endswith("_rows")]
    out[row_cols] = out[row_cols].fillna(0)

    out["snapshot_proxy_ok"] = (
        out.get("team_rolling_features_rows", 0).gt(0)
        & out.get("player_rolling_features_rows", 0).gt(0)
        & out.get("normalized_match_player_stats_rows", 0).gt(0)
    )
    out["injury_layer_ok"] = out.get("injury_features_rows", 0).gt(0) | out.get("normalized_injuries_rows", 0).gt(0)
    out["lineup_layer_ok"] = out.get("lineup_features_rows", 0).gt(0) | out.get("normalized_lineups_rows", 0).gt(0)
    out["fixture_context_ok"] = (
        out.get("matchup_interaction_features_rows", 0).gt(0)
        | out.get("event_features_rows", 0).gt(0)
        | out.get("h2h_regime_features_rows", 0).gt(0)
    )
    out["odds_layer_ok"] = out.get("odds_features_rows", 0).gt(0) | out.get("normalized_odds_prematch_long_rows", 0).gt(0)
    out["intelligence_readiness"] = out.apply(classify_readiness, axis=1)
    return out.sort_values(["intelligence_readiness", "league", "season"]).reset_index(drop=True)


def classify_readiness(row: pd.Series) -> str:
    snapshot = bool(row.get("snapshot_proxy_ok", False))
    injury = bool(row.get("injury_layer_ok", False))
    lineup = bool(row.get("lineup_layer_ok", False))
    fixture = bool(row.get("fixture_context_ok", False))
    if snapshot and injury and lineup and fixture:
        return "FULL_SHADOW_READY"
    if snapshot and fixture and (injury or lineup):
        return "PARTIAL_SHADOW_READY"
    if snapshot:
        return "SNAPSHOT_PROXY_ONLY"
    return "COVERAGE_GAP"


def summarize_counts(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    return (
        coverage.groupby("intelligence_readiness", dropna=False)
        .agg(
            league_season_cells=("league", "size"),
            scored_rows=("scored_rows", "sum"),
            scored_fixtures=("scored_fixtures", "sum"),
        )
        .reset_index()
        .sort_values(["scored_fixtures", "league_season_cells"], ascending=False)
    )


def write_report(
    outdir: Path,
    *,
    scored_root: Path,
    feature_inventory: pd.DataFrame,
    normalized_inventory: pd.DataFrame,
    scored_inventory: pd.DataFrame,
    coverage: pd.DataFrame,
    max_files: int,
) -> None:
    counts = summarize_counts(coverage)
    family_totals = (
        pd.concat([feature_inventory, normalized_inventory], ignore_index=True, sort=False)
        .groupby(["source_layer", "family"], dropna=False)
        .agg(files=("path", "count"), rows=("rows", "sum"))
        .reset_index()
        .sort_values(["source_layer", "rows"], ascending=[True, False])
        if not feature_inventory.empty or not normalized_inventory.empty
        else pd.DataFrame()
    )
    gaps = coverage[coverage["intelligence_readiness"].eq("COVERAGE_GAP")].copy() if not coverage.empty else pd.DataFrame()
    missing_injury = coverage[~coverage.get("injury_layer_ok", pd.Series(False, index=coverage.index))].copy() if not coverage.empty else pd.DataFrame()
    missing_lineup = coverage[~coverage.get("lineup_layer_ok", pd.Series(False, index=coverage.index))].copy() if not coverage.empty else pd.DataFrame()

    summary = {
        "generated_at": utc_now(),
        "scored_root": str(scored_root),
        "max_files": max_files,
        "scored_files": int(len(scored_inventory)),
        "coverage_cells": int(len(coverage)),
        "readiness_counts": counts.to_dict(orient="records"),
        "outputs": {
            "coverage": str(outdir / "walkforward_intelligence_estate_coverage.csv"),
            "feature_inventory": str(outdir / "api_feature_inventory.csv"),
            "normalized_inventory": str(outdir / "api_normalized_inventory.csv"),
            "scored_inventory": str(outdir / "walkforward_scored_file_inventory.csv"),
            "summary_md": str(outdir / "SUMMARY.md"),
        },
    }
    lines = [
        "# Walk-Forward Intelligence Estate Audit",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "Research-only. No predictions, gates, ModelStore artifacts, or deploy outputs were changed.",
        "",
        "## Methodology Read",
        "- Walk-forward rows remain the pick/result spine.",
        "- API-Football/player/team/lineup/injury layers are audited as sidecar evidence only.",
        "- `snapshot_proxy_ok` requires lagged team/player estate coverage and match-player history.",
        "- Lineup features are treated as shadow evidence unless their pre-kickoff timestamp contract is separately proven.",
        "- Injury/news shock flags are useful only where injury rows exist and can later be made timestamp-safe.",
        "",
        "## Readiness Counts",
        markdown_table(counts),
        "",
        "## Feature Family Totals",
        markdown_table(family_totals),
        "",
        "## Coverage Gaps",
        markdown_table(gaps[["league", "season", "scored_fixtures", "markets", "intelligence_readiness"]].head(40) if not gaps.empty else gaps),
        "",
        "## Missing Injury Layer",
        markdown_table(missing_injury[["league", "season", "scored_fixtures", "markets"]].head(40) if not missing_injury.empty else missing_injury),
        "",
        "## Missing Lineup Layer",
        markdown_table(missing_lineup[["league", "season", "scored_fixtures", "markets"]].head(40) if not missing_lineup.empty else missing_lineup),
        "",
        "## Overnight Gate",
        "Proceed to overlay scoring only for `FULL_SHADOW_READY`, `PARTIAL_SHADOW_READY`, or explicitly labelled `SNAPSHOT_PROXY_ONLY` cells. "
        "Do not interpret `COVERAGE_GAP` cells as evidence for or against player/team intelligence.",
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    feature_inventory.to_csv(outdir / "api_feature_inventory.csv", index=False)
    normalized_inventory.to_csv(outdir / "api_normalized_inventory.csv", index=False)
    scored_inventory.to_csv(outdir / "walkforward_scored_file_inventory.csv", index=False)
    coverage.to_csv(outdir / "walkforward_intelligence_estate_coverage.csv", index=False)
    counts.to_csv(outdir / "walkforward_intelligence_estate_readiness_counts.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", default=str(DEFAULT_SCORED_ROOT))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--normalized-dir", default=str(DEFAULT_NORMALIZED_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--max-files", type=int, default=0, help="Smoke mode: inspect only the first N scored files.")
    args = parser.parse_args()

    scored_root = Path(args.scored_root)
    outdir = Path(args.outdir)
    scored_inventory, scored = inventory_scored(scored_root, max_files=args.max_files)
    feature_inventory = inventory_feature_files(Path(args.features_dir))
    normalized_inventory = inventory_normalized_files(Path(args.normalized_dir))
    coverage = build_coverage(scored, feature_inventory, normalized_inventory)
    write_report(
        outdir,
        scored_root=scored_root,
        feature_inventory=feature_inventory,
        normalized_inventory=normalized_inventory,
        scored_inventory=scored_inventory,
        coverage=coverage,
        max_files=args.max_files,
    )
    print(f"Scored files: {len(scored_inventory)}")
    print(f"Coverage cells: {len(coverage)}")
    print(f"Outputs: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
