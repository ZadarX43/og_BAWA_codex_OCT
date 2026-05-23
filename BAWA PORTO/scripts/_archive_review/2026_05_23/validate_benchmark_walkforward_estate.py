#!/usr/bin/env python3
"""Validate that an intelligence overlay run is aligned to benchmark estate.

Research-only. This script does not generate predictions, train models, or
change deploy routing. It exists to stop a shadow intelligence backtest from
quietly drifting onto the wrong walk-forward root or a partial API estate.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCORED_ROOT = Path("predictions_output/hybrid_shadow_walkforward_2026_05_01_parity_rebuild")
DEFAULT_FEATURES_DIR = Path("data_sources/api_football/features")
DEFAULT_TG_REFERENCE_DIR = Path("reports/2026-05-06/team_goal_shadow_market_backtest")
DEFAULT_OUTDIR = Path("reports/latest/benchmark_walkforward_estate_validation")

FEATURE_FAMILIES = [
    "injury_features",
    "team_rolling_features",
    "player_rolling_features",
    "matchup_interaction_features",
    "lineup_features",
    "enriched_fixture_features",
    "event_features",
]

REFERENCE_TARGETS = [
    {
        "benchmark": "OU25 calibrated",
        "target_hit_rate": 0.9535,
        "target_rows": 3828,
        "source": "docs/API_FOOTBALL_PLAN.md + reports/2026-05-01 HYBRID_SCOREBOARD_V2",
        "status": "REFERENCE_FLOOR",
    },
    {
        "benchmark": "BTTS calibrated",
        "target_hit_rate": 0.9355,
        "target_rows": 3382,
        "source": "docs/API_FOOTBALL_PLAN.md protected benchmark floor",
        "status": "REFERENCE_FLOOR",
    },
    {
        "benchmark": "Premium value-edge ROI",
        "target_hit_rate": 0.8331,
        "target_rows": 15203,
        "target_roi": 0.5390,
        "source": "reports/2026-05-01 HYBRID_SCOREBOARD_V2 value-edge baseline",
        "status": "REFERENCE_FLOOR",
    },
    {
        "benchmark": "Home Team Over 1.5 premium",
        "target_hit_rate": 0.9324,
        "target_rows": 1643,
        "source": "reports/2026-05-06 team_goal_shadow_market_backtest",
        "status": "VERIFY_FROM_TG_REFERENCE",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def league_tag(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def norm_market(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"over25", "over_25", "over_2_5"}:
        return "ou25"
    return text


def scored_files(root: Path) -> list[Path]:
    return sorted(root.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))


def compute_hit(df: pd.DataFrame) -> pd.Series:
    hit = pd.Series(np.nan, index=df.index, dtype="float64")
    market = df.get("market_norm", pd.Series("", index=df.index)).astype("string")
    ftr = market.eq("ftr")
    ou25 = market.eq("ou25")
    btts = market.eq("btts")
    if "ftr_hit" in df.columns:
        hit.loc[ftr] = num(df.loc[ftr, "ftr_hit"])
    if "ou25_hit" in df.columns:
        hit.loc[ou25] = num(df.loc[ou25, "ou25_hit"])
    if "btts_yes_hit" in df.columns:
        pick_text = (
            df.get("bookie_pick", pd.Series("", index=df.index)).astype("string")
            + " "
            + df.get("selection", pd.Series("", index=df.index)).astype("string")
        ).str.lower()
        no_pick = pick_text.str.contains("no|under|false", regex=True, na=False)
        if "btts_no_hit" in df.columns:
            hit.loc[btts & no_pick] = num(df.loc[btts & no_pick, "btts_no_hit"])
        hit.loc[btts & ~no_pick] = num(df.loc[btts & ~no_pick, "btts_yes_hit"])
    return hit


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    hit = compute_hit(df)
    settled = hit.isin([0, 1])
    wins = int(hit.eq(1).sum())
    odds = num(df.get("bookie_od", np.nan))
    profit = np.where(hit.eq(1), odds - 1.0, np.where(hit.eq(0), -1.0, np.nan))
    return {
        "rows": int(len(df)),
        "settled": int(settled.sum()),
        "wins": wins,
        "losses": int(hit.eq(0).sum()),
        "hit_rate": wins / int(settled.sum()) if int(settled.sum()) else np.nan,
        "profit_units": float(np.nansum(profit)) if len(df) else 0.0,
        "roi": float(np.nansum(profit)) / int(settled.sum()) if int(settled.sum()) else np.nan,
    }


def load_scored(root: Path) -> pd.DataFrame:
    wanted = {
        "league",
        "match_date",
        "home_team_name",
        "away_team_name",
        "fixture_key",
        "market",
        "bookie_pick",
        "selection",
        "bookie_od",
        "meta_overlay_label",
        "value_edge_tier",
        "deploy_tier",
        "tier",
        "ftr_hit",
        "ou25_hit",
        "btts_yes_hit",
        "btts_no_hit",
    }
    frames: list[pd.DataFrame] = []
    for path in scored_files(root):
        frame = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
        frame["window_id"] = path.parts[-3] if len(path.parts) >= 3 else ""
        frame["__source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["market_norm"] = out.get("market", "").map(norm_market)
    out["match_date_dt"] = pd.to_datetime(out.get("match_date"), errors="coerce")
    out["season"] = out["match_date_dt"].dt.year.astype("Int64")
    out["league_tag"] = out.get("league", "").map(league_tag)
    return out


def build_root_signature(df: pd.DataFrame, *, file_count: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"metric": "scored_files", "value": file_count}])
    rows = [
        {"metric": "scored_files", "value": file_count},
        {"metric": "scored_rows", "value": len(df)},
        {"metric": "unique_fixtures", "value": df["fixture_key"].nunique(dropna=True) if "fixture_key" in df.columns else 0},
        {"metric": "competitions", "value": df["league"].nunique(dropna=True) if "league" in df.columns else 0},
        {"metric": "windows", "value": df["window_id"].nunique(dropna=True) if "window_id" in df.columns else 0},
        {"metric": "date_min", "value": str(df["match_date_dt"].min().date()) if df["match_date_dt"].notna().any() else ""},
        {"metric": "date_max", "value": str(df["match_date_dt"].max().date()) if df["match_date_dt"].notna().any() else ""},
    ]
    for market, group in df.groupby("market_norm", dropna=False):
        rows.append({"metric": f"rows_market_{market}", "value": len(group)})
    if "value_edge_tier" in df.columns:
        premium = df[df["value_edge_tier"].astype("string").str.upper().eq("PREMIUM")]
        metrics = summarize(premium)
        rows.extend(
            [
                {"metric": "combined_value_premium_rows", "value": metrics["rows"]},
                {"metric": "combined_value_premium_settled", "value": metrics["settled"]},
                {"metric": "combined_value_premium_hit_rate", "value": metrics["hit_rate"]},
                {"metric": "combined_value_premium_roi", "value": metrics["roi"]},
            ]
        )
    if "meta_overlay_label" in df.columns:
        for label in ["OU25_META_ELITE", "BTTS_META_ELITE", "FTR_META_RANK"]:
            part = df[df["meta_overlay_label"].astype("string").eq(label)]
            metrics = summarize(part)
            rows.extend(
                [
                    {"metric": f"combined_{label}_rows", "value": metrics["rows"]},
                    {"metric": f"combined_{label}_settled", "value": metrics["settled"]},
                    {"metric": f"combined_{label}_hit_rate", "value": metrics["hit_rate"]},
                    {"metric": f"combined_{label}_roi", "value": metrics["roi"]},
                ]
            )
    return pd.DataFrame(rows)


def parse_feature_path(path: Path, family: str) -> tuple[str, int] | None:
    match = re.match(rf"api_{re.escape(family)}__(.+)__(\d{{4}})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def feature_inventory(features_dir: Path, required_cells: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, cell in required_cells.iterrows():
        league = str(cell["league"])
        tag = str(cell["league_tag"])
        season = int(cell["season"])
        for family in FEATURE_FAMILIES:
            path = features_dir / f"api_{family}__{tag}__{season}.csv"
            rows = 0
            unique_fixtures = 0
            exists = path.exists()
            if exists:
                try:
                    frame = pd.read_csv(path, usecols=lambda c: c in {"fixture_key"}, low_memory=False)
                    rows = int(len(frame))
                    unique_fixtures = int(frame["fixture_key"].nunique(dropna=True)) if "fixture_key" in frame.columns else 0
                except Exception:
                    rows = 0
                    unique_fixtures = 0
            records.append(
                {
                    "league": league,
                    "league_tag": tag,
                    "season": season,
                    "feature_family": family,
                    "feature_file_exists": exists,
                    "feature_rows": rows,
                    "feature_unique_fixtures": unique_fixtures,
                    "required_scored_rows": int(cell["required_scored_rows"]),
                    "required_unique_fixtures": int(cell["required_unique_fixtures"]),
                    "coverage_bucket": "FEATURE_READY" if exists and rows > 0 else "MISSING_FEATURE_FILE",
                }
            )
    return pd.DataFrame(records)


def required_cells(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["league", "league_tag", "season", "required_scored_rows", "required_unique_fixtures"])
    cells = (
        df.dropna(subset=["season"])
        .groupby(["league", "league_tag", "season"], dropna=False)
        .agg(required_scored_rows=("fixture_key", "size"), required_unique_fixtures=("fixture_key", "nunique"))
        .reset_index()
    )
    cells["season"] = cells["season"].astype(int)
    return cells


def join_coverage(overlay_dir: Path) -> pd.DataFrame:
    path = overlay_dir / "WALKFORWARD_INTELLIGENCE_OVERLAY_DECISIONS.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    join_cols = [col for col in df.columns if col.endswith("_join_hit")]
    rows: list[dict[str, Any]] = []
    for col in join_cols:
        family = col.replace("_join_hit", "")
        hit = df[col].astype("string").str.lower().isin({"true", "1", "1.0", "yes"})
        rows.append(
            {
                "scope": "ALL",
                "feature_family": family,
                "rows": len(df),
                "join_hits": int(hit.sum()),
                "join_hit_rate": float(hit.mean()) if len(df) else np.nan,
            }
        )
        if "season" in df.columns:
            for season, group in df.groupby("season", dropna=False):
                ghit = group[col].astype("string").str.lower().isin({"true", "1", "1.0", "yes"})
                rows.append(
                    {
                        "scope": f"season_{season}",
                        "feature_family": family,
                        "rows": len(group),
                        "join_hits": int(ghit.sum()),
                        "join_hit_rate": float(ghit.mean()) if len(group) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def target_status(tg_reference_dir: Path) -> pd.DataFrame:
    rows = [dict(item) for item in REFERENCE_TARGETS]
    scorecard = tg_reference_dir / "team_goal_shadow_scorecard_by_product_policy.csv"
    if scorecard.exists():
        df = pd.read_csv(scorecard)
        mask = (
            df.get("shadow_product", "").astype("string").eq("HOME_TEAM_OVER_1_5_SHADOW")
            & df.get("shadow_policy", "").astype("string").eq("TG15_PREMIUM")
        )
        if mask.any():
            row = df[mask].iloc[0]
            for item in rows:
                if item["benchmark"] == "Home Team Over 1.5 premium":
                    item["observed_rows"] = int(row.get("graded", row.get("rows", 0)))
                    item["observed_hit_rate"] = float(row.get("hit_rate", np.nan))
                    item["status"] = "MATCH" if item["observed_rows"] == item["target_rows"] and round(item["observed_hit_rate"], 4) == item["target_hit_rate"] else "DRIFT"
    return pd.DataFrame(rows)


def write_summary(
    outdir: Path,
    *,
    scored_root: Path,
    features_dir: Path,
    overlay_dir: Path,
    tg_reference_dir: Path,
    signature: pd.DataFrame,
    targets: pd.DataFrame,
    inventory: pd.DataFrame,
    joins: pd.DataFrame,
    expected_files: int,
    min_competitions: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    signature.to_csv(outdir / "BENCHMARK_WALKFORWARD_ESTATE_SIGNATURE.csv", index=False)
    targets.to_csv(outdir / "BENCHMARK_TARGETS_STATUS.csv", index=False)
    inventory.to_csv(outdir / "BENCHMARK_API_FEATURE_INVENTORY.csv", index=False)
    joins.to_csv(outdir / "BENCHMARK_INTELLIGENCE_JOIN_COVERAGE.csv", index=False)

    sig = dict(zip(signature["metric"], signature["value"], strict=False))
    critical = []
    if int(sig.get("scored_files", 0) or 0) != expected_files:
        critical.append(f"scored_files expected {expected_files}, observed {sig.get('scored_files')}")
    if int(sig.get("competitions", 0) or 0) < min_competitions:
        critical.append(f"competitions expected >= {min_competitions}, observed {sig.get('competitions')}")

    recent = inventory[inventory["season"].isin([2025, 2026])] if not inventory.empty else pd.DataFrame()
    recent_missing = recent[recent["coverage_bucket"].eq("MISSING_FEATURE_FILE")] if not recent.empty else pd.DataFrame()
    recent_summary = (
        recent.groupby(["season", "feature_family", "coverage_bucket"], dropna=False)
        .size()
        .reset_index(name="league_cells")
        if not recent.empty
        else pd.DataFrame()
    )

    lines = [
        "# Benchmark Walk-Forward Estate Validation",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "Research-only. No deploy gates, ModelStore artifacts, or production routing changed.",
        "",
        "## Inputs",
        f"- scored root: `{scored_root}`",
        f"- features dir: `{features_dir}`",
        f"- overlay dir: `{overlay_dir}`",
        f"- TG reference dir: `{tg_reference_dir}`",
        "",
        "## Root Signature",
        markdown_table(signature),
        "",
        "## Benchmark Targets",
        markdown_table(targets),
        "",
        "## 2025/2026 Feature Coverage",
        markdown_table(recent_summary),
        "",
        "## Intelligence Join Coverage",
        markdown_table(joins),
        "",
        "## Critical Read",
    ]
    if critical:
        lines.extend(f"- `FAIL`: {item}" for item in critical)
    else:
        lines.append("- `PASS`: root-level benchmark identity checks passed.")
    if not recent_missing.empty:
        lines.append(f"- `COVERAGE_GAP`: {len(recent_missing)} required 2025/2026 feature cells are missing files.")
    if joins.empty:
        lines.append("- `JOIN_UNKNOWN`: overlay decision file not found yet, so sidecar join rates were not measured.")
    else:
        diagnostic_suffixes = ("_fixture_key", "_fallback", "_identity_map")
        aggregate_joins = joins[
            joins["scope"].eq("ALL")
            & ~joins["feature_family"].astype("string").str.endswith(diagnostic_suffixes)
        ]
        low_join = aggregate_joins[aggregate_joins["join_hit_rate"].lt(0.90)]
        if not low_join.empty:
            lines.append("- `JOIN_GAP`: at least one sidecar family joined below 90% over the overlay rows.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- The benchmark target rows are reference floors from the May docs/reports; this script verifies estate identity and TG reference parity, then measures current overlay join coverage.",
            "- 2025/2026 API feature gaps must be closed before claiming player/injury/team intelligence lift on the full benchmark estate.",
            "- Lineups remain shadow/proxy unless their provider timestamp is proven pre-kickoff or the feature is explicitly lagged from the previous fixture.",
        ]
    )
    meta = {
        "generated_at": utc_now(),
        "scored_root": str(scored_root),
        "features_dir": str(features_dir),
        "overlay_dir": str(overlay_dir),
        "tg_reference_dir": str(tg_reference_dir),
        "critical_failures": critical,
        "recent_missing_feature_cells": int(len(recent_missing)),
    }
    (outdir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", default=str(DEFAULT_SCORED_ROOT))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--overlay-dir", default="")
    parser.add_argument("--tg-reference-dir", default=str(DEFAULT_TG_REFERENCE_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--expected-scored-files", type=int, default=139)
    parser.add_argument("--min-competitions", type=int, default=24)
    parser.add_argument("--fail-on-critical", action="store_true")
    args = parser.parse_args()

    scored_root = Path(args.scored_root)
    files = scored_files(scored_root)
    df = load_scored(scored_root)
    signature = build_root_signature(df, file_count=len(files))
    cells = required_cells(df)
    inventory = feature_inventory(Path(args.features_dir), cells)
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else Path(args.outdir).parent / "02_market_overlay_backtest"
    joins = join_coverage(overlay_dir)
    targets = target_status(Path(args.tg_reference_dir))
    outdir = Path(args.outdir)
    write_summary(
        outdir,
        scored_root=scored_root,
        features_dir=Path(args.features_dir),
        overlay_dir=overlay_dir,
        tg_reference_dir=Path(args.tg_reference_dir),
        signature=signature,
        targets=targets,
        inventory=inventory,
        joins=joins,
        expected_files=args.expected_scored_files,
        min_competitions=args.min_competitions,
    )

    sig = dict(zip(signature["metric"], signature["value"], strict=False))
    critical = []
    if int(sig.get("scored_files", 0) or 0) != args.expected_scored_files:
        critical.append("scored_files")
    if int(sig.get("competitions", 0) or 0) < args.min_competitions:
        critical.append("competitions")
    print(f"Scored files: {sig.get('scored_files')}")
    print(f"Competitions: {sig.get('competitions')}")
    print(f"Outputs: {outdir}")
    if critical and args.fail_on_critical:
        raise SystemExit(f"Critical benchmark validation failed: {', '.join(critical)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
