#!/usr/bin/env python3
"""Repeat QA for OU25_RESTORE_NOW_SHADOW on real deploy-shaped boards.

Research-only runner. It combines ELITE/STANDARD/OBSERVE tier CSVs for a live
board, applies the Phase 8H C4 sidecar policy, and reports only the
OU25_RESTORE_NOW_SHADOW candidates.

Safety contract:
  - no production files are changed
  - source deploy_tier/tier must remain unchanged
  - output is shadow-only instrumentation
  - BTTS/FTR/API-Football promotion is out of scope
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deploy_rulebook_research_phase8h_c4_shadow as c4_shadow  # noqa: E402


DEFAULT_OUTDIR = Path("reports/2026-05-06/ou25_c4_live_shadow_repeat_qa")
WATCH_LEAGUES = ["USA MLS", "Spain La Liga", "Netherlands Eredivisie", "Japan J1"]
TARGET_STAGE = "OU25_RESTORE_NOW_SHADOW"


@dataclass(frozen=True)
class BoardSet:
    board_dir: Path
    base_name: str
    elite: Path
    standard: Path
    observe: Path

    @property
    def max_mtime(self) -> float:
        return max(self.elite.stat().st_mtime, self.standard.stat().st_mtime, self.observe.stat().st_mtime)

    @property
    def fixture_range(self) -> str:
        match = re.search(r"(\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2})", self.base_name)
        return match.group(1) if match else self.base_name


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
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
    return "\n".join(lines)


def is_real_board_dir(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    joined = str(path).lower()
    if any(token in joined for token in ["walk_forward", "_tmp", "smoke", "research", "shadow_parity_audit"]):
        return False
    return bool(re.search(r"predictions_output/\d{4}-\d{2}-\d{2}$", str(path)))


def discover_boards(root: Path) -> list[BoardSet]:
    grouped: dict[tuple[Path, str], dict[str, Path]] = {}
    for path in root.rglob("*__DEPLOY_TIER_*__PRESET_V1__FTR_accuracy.csv"):
        if not is_real_board_dir(path.parent):
            continue
        tier_match = re.search(r"__DEPLOY_TIER_(ELITE|STANDARD|OBSERVE)__", path.name)
        if not tier_match:
            continue
        base = path.name.split("__DEPLOY_TIER_", 1)[0]
        key = (path.parent, base)
        grouped.setdefault(key, {})[tier_match.group(1).lower()] = path

    boards = []
    for (board_dir, base_name), tiers in grouped.items():
        if {"elite", "standard", "observe"}.issubset(tiers):
            boards.append(
                BoardSet(
                    board_dir=board_dir,
                    base_name=base_name,
                    elite=tiers["elite"],
                    standard=tiers["standard"],
                    observe=tiers["observe"],
                )
            )
    return sorted(boards, key=lambda board: board.max_mtime, reverse=True)


def boards_from_dir(board_dir: Path) -> list[BoardSet]:
    grouped: dict[str, dict[str, Path]] = {}
    for path in board_dir.glob("*__DEPLOY_TIER_*__PRESET_V1__FTR_accuracy.csv"):
        tier_match = re.search(r"__DEPLOY_TIER_(ELITE|STANDARD|OBSERVE)__", path.name)
        if not tier_match:
            continue
        base = path.name.split("__DEPLOY_TIER_", 1)[0]
        grouped.setdefault(base, {})[tier_match.group(1).lower()] = path

    boards = []
    for base_name, tiers in grouped.items():
        if {"elite", "standard", "observe"}.issubset(tiers):
            boards.append(
                BoardSet(
                    board_dir=board_dir,
                    base_name=base_name,
                    elite=tiers["elite"],
                    standard=tiers["standard"],
                    observe=tiers["observe"],
                )
            )
    return sorted(boards, key=lambda board: board.max_mtime, reverse=True)


def load_tier_file(path: Path, tier: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "deploy_tier" not in df.columns:
        df["deploy_tier"] = tier
    if "tier" not in df.columns:
        df["tier"] = df["deploy_tier"]
    df["source_tier_file"] = path.name
    df["source_tier_expected"] = tier
    return df


def combine_board(board: BoardSet) -> pd.DataFrame:
    frames = [
        load_tier_file(board.elite, "ELITE"),
        load_tier_file(board.standard, "STANDARD"),
        load_tier_file(board.observe, "OBSERVE"),
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def count_by(df: pd.DataFrame, column: str, name: str = "rows") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[column, name])
    return (
        df[column]
        .astype("string")
        .fillna("UNKNOWN")
        .str.strip()
        .replace("", "UNKNOWN")
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name=name)
    )


def source_tier_counts(df: pd.DataFrame) -> pd.DataFrame:
    tier = df.get("deploy_tier", pd.Series("", index=df.index)).astype("string").fillna("").str.upper().str.strip()
    tier = tier.mask(tier.eq(""), "UNKNOWN")
    return tier.value_counts(dropna=False).rename_axis("source_tier").reset_index(name="rows")


def watch_table(selected_ou25: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["league", "rows", "source_elite", "source_standard", "source_observe", "watch_league"]
    if selected_ou25.empty:
        return pd.DataFrame(columns=base_cols)

    tier = selected_ou25.get("deploy_tier", pd.Series("", index=selected_ou25.index)).astype("string").fillna("").str.upper().str.strip()
    tmp = selected_ou25.assign(source_tier=tier.mask(tier.eq(""), "UNKNOWN"))
    pivot = (
        tmp.pivot_table(index="league", columns="source_tier", values="_phase8h_c4_dedupe_key", aggfunc="count", fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["ELITE", "STANDARD", "OBSERVE"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["rows"] = pivot[["ELITE", "STANDARD", "OBSERVE"]].sum(axis=1)
    pivot["watch_league"] = pivot["league"].isin(WATCH_LEAGUES)
    pivot = pivot.rename(columns={"ELITE": "source_elite", "STANDARD": "source_standard", "OBSERVE": "source_observe"})
    return pivot[base_cols].sort_values(["watch_league", "rows", "league"], ascending=[False, False, True])


def compact_stage_counts(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=["phase8h_c4_shadow_stage", "rows"])
    return (
        selected["phase8h_c4_shadow_stage"]
        .astype("string")
        .fillna("")
        .replace("", "UNKNOWN")
        .value_counts(dropna=False)
        .rename_axis("phase8h_c4_shadow_stage")
        .reset_index(name="rows")
    )


def run_board(board: BoardSet, policy: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    board_slug = re.sub(r"[^A-Za-z0-9_]+", "_", board.fixture_range.replace("-", "_"))
    board_outdir = outdir / board_slug
    board_outdir.mkdir(parents=True, exist_ok=True)

    source = combine_board(board)
    annotated = c4_shadow.apply_shadow_policy(source, policy)
    selected = c4_shadow.selected_rows(annotated)
    selected_ou25 = selected[selected["phase8h_c4_shadow_stage"].eq(TARGET_STAGE)].copy()

    deploy_changed = int(
        (
            source.get("deploy_tier", pd.Series("", index=source.index)).fillna("").astype(str)
            != annotated.get("deploy_tier", pd.Series("", index=annotated.index)).fillna("").astype(str)
        ).sum()
    )
    tier_changed = int(
        (
            source.get("tier", pd.Series("", index=source.index)).fillna("").astype(str)
            != annotated.get("tier", pd.Series("", index=annotated.index)).fillna("").astype(str)
        ).sum()
    )

    source_path = board_outdir / f"{board_slug}__DEPLOY_TIERS_COMBINED_FOR_OU25_C4_SHADOW_QA.csv"
    annotated_path = board_outdir / f"{board_slug}__OU25_C4_SHADOW_ANNOTATED.csv"
    selected_path = board_outdir / f"{board_slug}__OU25_C4_SHADOW_SELECTED_ALL_STAGES.csv"
    ou25_path = board_outdir / f"{board_slug}__OU25_RESTORE_NOW_SHADOW_SELECTED.csv"
    watch_path = board_outdir / f"{board_slug}__OU25_RESTORE_NOW_WATCH_TABLE.csv"
    tier_counts_path = board_outdir / f"{board_slug}__SOURCE_TIER_COUNTS.csv"
    selected_tier_path = board_outdir / f"{board_slug}__OU25_SELECTED_SOURCE_TIER_COUNTS.csv"
    stage_counts_path = board_outdir / f"{board_slug}__SHADOW_STAGE_COUNTS.csv"
    summary_path = board_outdir / f"{board_slug}__OU25_C4_LIVE_SHADOW_REPEAT_QA_SUMMARY.md"

    source.to_csv(source_path, index=False)
    annotated.to_csv(annotated_path, index=False)
    selected.to_csv(selected_path, index=False)
    selected_ou25.to_csv(ou25_path, index=False)

    source_counts = source_tier_counts(source)
    selected_tier_counts = source_tier_counts(selected_ou25)
    watch = watch_table(selected_ou25)
    stage_counts = compact_stage_counts(selected)

    source_counts.to_csv(tier_counts_path, index=False)
    selected_tier_counts.to_csv(selected_tier_path, index=False)
    watch.to_csv(watch_path, index=False)
    stage_counts.to_csv(stage_counts_path, index=False)

    status = "PASS" if deploy_changed == 0 and tier_changed == 0 else "FAIL_TIER_MUTATION"
    summary_df = pd.DataFrame(
        [
            {
                "status": status,
                "board_dir": str(board.board_dir),
                "fixture_range": board.fixture_range,
                "source_rows": int(len(source)),
                "ou25_restore_now_rows": int(len(selected_ou25)),
                "deploy_tier_changed": deploy_changed,
                "tier_changed": tier_changed,
            }
        ]
    )

    lines = [
        "# OU25 C4 Live Shadow Repeat QA",
        "",
        "Research-only QA. No source deploy files or production policy files were changed.",
        "",
        f"- Board directory: `{board.board_dir}`",
        f"- Fixture range: `{board.fixture_range}`",
        f"- Status: `{status}`",
        "",
        "## Safety Check",
        markdown_table(summary_df),
        "",
        "## Source Tier Counts",
        markdown_table(source_counts),
        "",
        "## OU25 Restore-Now Source Tier Counts",
        markdown_table(selected_tier_counts),
        "",
        "## Shadow Stage Counts",
        markdown_table(stage_counts),
        "",
        "## Watch Table",
        markdown_table(watch),
        "",
        "## Promotion Guard",
        "",
        "- This output is instrumentation only.",
        "- No BTTS promotion before OU25 proves stable on repeated real boards.",
        "- No API-Football league promotion without league-specific backtest proof.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": status,
        "board_dir": str(board.board_dir),
        "fixture_range": board.fixture_range,
        "source_rows": int(len(source)),
        "ou25_restore_now_rows": int(len(selected_ou25)),
        "deploy_tier_changed": deploy_changed,
        "tier_changed": tier_changed,
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="predictions_output", help="Prediction output root for auto-discovery.")
    parser.add_argument("--board-dir", default="", help="Specific live board directory containing tier CSVs.")
    parser.add_argument("--policy", default=str(c4_shadow.DEFAULT_POLICY), help="C4 ring policy CSV.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory.")
    parser.add_argument("--all", action="store_true", help="Run all discovered real boards instead of latest only.")
    parser.add_argument("--limit", type=int, default=1, help="Max boards to run when not using --all.")
    args = parser.parse_args()

    policy = pd.read_csv(args.policy)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.board_dir:
        boards = boards_from_dir(Path(args.board_dir))
    else:
        boards = discover_boards(Path(args.root))

    if not boards:
        raise SystemExit("No complete real deploy tier board found.")

    if not args.all:
        boards = boards[: max(1, int(args.limit))]

    records = [run_board(board, policy, outdir) for board in boards]
    index = pd.DataFrame(records)
    index_path = outdir / "ou25_c4_live_shadow_repeat_qa_index.csv"
    index.to_csv(index_path, index=False)

    latest_summary = Path(records[0]["summary_path"])
    print(f"[ok] boards={len(records)}")
    print(f"[ok] wrote {index_path}")
    print(f"[ok] latest summary {latest_summary}")

    if any(record["status"] != "PASS" for record in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
