#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("predictions_output/ou25_frozen_compare/rulebook_ftr_validation_3yr_19lg_v1")
OUT_CSV = Path("ou25_branch_comparison.csv")
OUT_MD = Path("ou25_branch_comparison.md")


def _safe_num(v: Any) -> float:
    try:
        if v is None or v == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _extract_summary_metrics(obj: dict[str, Any]) -> dict[str, Any]:
    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
    return {
        "rows": obj.get("rows", summary.get("rows")),
        "hit": obj.get("hit", summary.get("hit")),
        "roi": obj.get("roi", summary.get("roi")),
        "avg_odds": obj.get("avg_od", obj.get("avg_odds", summary.get("avg_od", summary.get("avg_odds")))),
        "markets": obj.get("markets", summary.get("markets")),
        "leagues": obj.get("leagues", summary.get("leagues")),
    }


def _detect_pick_mode(branch_name: str) -> str:
    name = branch_name.lower()
    if "over_only" in name or "over-only" in name:
        return "over_only"
    if "under_only" in name or "under-only" in name:
        return "under_only"
    return "combined"


def _detect_sweep_type(branch_name: str) -> str:
    name = branch_name.lower()
    if "baseline" in name:
        return "baseline"
    if "topq" in name:
        return "top_q"
    if "band1" in name:
        return "band1"
    if "band2" in name:
        return "band2"
    if "mode_" in name:
        return "pick_mode"
    return "other"


def _read_summary_json(summary_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"failed to read {summary_path}: {exc}")
        return None


rows: list[dict[str, Any]] = []

if not ROOT.exists():
    raise SystemExit(f"OU25 sweep root not found: {ROOT}")

for backtest_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
    for branch_dir in sorted([p for p in backtest_dir.iterdir() if p.is_dir()]):
        summary_files = sorted(branch_dir.glob("*__SUMMARY.json"))
        if not summary_files:
            print(f"missing summary json in {branch_dir}")
            continue

        if len(summary_files) > 1:
            print(f"multiple summary json files in {branch_dir}, using first: {summary_files[0].name}")

        summary_path = summary_files[0]
        obj = _read_summary_json(summary_path)
        if obj is None:
            continue

        metrics = _extract_summary_metrics(obj)
        artifact_info = obj.get("artifacts") if isinstance(obj.get("artifacts"), dict) else {}
        config_info = obj.get("config") if isinstance(obj.get("config"), dict) else {}

        rows.append(
            {
                "dataset": backtest_dir.name,
                "branch": branch_dir.name,
                "sweep_type": _detect_sweep_type(branch_dir.name),
                "pick_mode": _detect_pick_mode(branch_dir.name),
                "rows": _safe_num(metrics.get("rows")),
                "hit": _safe_num(metrics.get("hit")),
                "roi": _safe_num(metrics.get("roi")),
                "avg_odds": _safe_num(metrics.get("avg_odds")),
                "markets": _safe_num(metrics.get("markets")),
                "leagues": _safe_num(metrics.get("leagues")),
                "top_q": _safe_num(config_info.get("top_q")),
                "ou25_band1_low": _safe_num((config_info.get("ou25_band1") or [None, None])[0]),
                "ou25_band1_high": _safe_num((config_info.get("ou25_band1") or [None, None])[1]),
                "ou25_band2_low": _safe_num((config_info.get("ou25_band2") or [None, None])[0]),
                "ou25_band2_high": _safe_num((config_info.get("ou25_band2") or [None, None])[1]),
                "summary_json": str(summary_path),
                "filtered_csv": artifact_info.get("filtered_csv"),
                "market_summary_csv": artifact_info.get("market_summary_csv"),
                "league_summary_csv": artifact_info.get("league_summary_csv"),
            }
        )

if not rows:
    raise SystemExit(f"No OU25 summary JSON files found under {ROOT}")


df = pd.DataFrame(rows)
for col in [
    "rows",
    "hit",
    "roi",
    "avg_odds",
    "markets",
    "leagues",
    "top_q",
    "ou25_band1_low",
    "ou25_band1_high",
    "ou25_band2_low",
    "ou25_band2_high",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

sort_cols = ["roi", "hit", "rows", "avg_odds"]
df = df.sort_values(sort_cols, ascending=[False, False, False, False]).reset_index(drop=True)

df.to_csv(OUT_CSV, index=False)

md_cols = [
    "branch",
    "sweep_type",
    "pick_mode",
    "rows",
    "hit",
    "roi",
    "avg_odds",
    "top_q",
    "ou25_band1_low",
    "ou25_band1_high",
    "ou25_band2_low",
    "ou25_band2_high",
]
md_df = df.loc[:, [c for c in md_cols if c in df.columns]].copy()

lines: list[str] = []
lines.append("# OU25 Frozen Branch Comparison")
lines.append("")
lines.append(f"- Root: `{ROOT}`")
lines.append(f"- Branches found: `{len(df)}`")
lines.append("")
try:
    lines.append(md_df.to_markdown(index=False))
except Exception:
    lines.append(md_df.to_string(index=False))
lines.append("")
OUT_MD.write_text("\n".join(lines), encoding="utf-8")

print(df.to_string(index=False))
print(f"\nWROTE: {OUT_CSV}")
print(f"WROTE: {OUT_MD}")