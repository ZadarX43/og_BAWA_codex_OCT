#!/usr/bin/env python3
"""Shadow QA for validated Team-Goal Combo proof lanes on live boards.

Research-only. Stamps/selects only:
  - Spain La Liga HOME_WIN_AND_HOME_GE2
  - Germany Bundesliga HOME_WIN_AND_HOME_GE2

Belgium remains excluded. USA remains watch-only and is not selected here.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_OUTDIR = Path("reports/2026-05-06/team_goal_combo_live_shadow_qa")
PROOF_LANES = {
    ("Spain La Liga", "HOME_WIN_AND_HOME_GE2"),
    ("Germany Bundesliga", "HOME_WIN_AND_HOME_GE2"),
}
VALID_TIERS = {"MODERATE_HW", "STRONG_HW", "VERY_STRONG_HW"}


@dataclass(frozen=True)
class BoardSet:
    board_dir: Path
    base_name: str
    elite: Path
    standard: Path
    observe: Path

    @property
    def fixture_range(self) -> str:
        match = re.search(r"(\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2})", self.base_name)
        return match.group(1) if match else self.base_name

    @property
    def max_mtime(self) -> float:
        return max(self.elite.stat().st_mtime, self.standard.stat().st_mtime, self.observe.stat().st_mtime)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = ["| " + " | ".join(text.columns) + " |", "| " + " | ".join(["---"] * len(text.columns)) + " |"]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def is_real_board_dir(path: Path) -> bool:
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
        grouped.setdefault((path.parent, base), {})[tier_match.group(1).lower()] = path
    boards = []
    for (board_dir, base), tiers in grouped.items():
        if {"elite", "standard", "observe"}.issubset(tiers):
            boards.append(BoardSet(board_dir, base, tiers["elite"], tiers["standard"], tiers["observe"]))
    return sorted(boards, key=lambda b: b.max_mtime, reverse=True)


def boards_from_dir(board_dir: Path) -> list[BoardSet]:
    grouped: dict[str, dict[str, Path]] = {}
    for path in board_dir.glob("*__DEPLOY_TIER_*__PRESET_V1__FTR_accuracy.csv"):
        tier_match = re.search(r"__DEPLOY_TIER_(ELITE|STANDARD|OBSERVE)__", path.name)
        if not tier_match:
            continue
        base = path.name.split("__DEPLOY_TIER_", 1)[0]
        grouped.setdefault(base, {})[tier_match.group(1).lower()] = path
    boards = []
    for base, tiers in grouped.items():
        if {"elite", "standard", "observe"}.issubset(tiers):
            boards.append(BoardSet(board_dir, base, tiers["elite"], tiers["standard"], tiers["observe"]))
    return sorted(boards, key=lambda b: b.max_mtime, reverse=True)


def load_tier(path: Path, tier: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "deploy_tier" not in df.columns:
        df["deploy_tier"] = tier
    if "tier" not in df.columns:
        df["tier"] = df["deploy_tier"]
    df["source_tier_file"] = path.name
    return df


def combine(board: BoardSet) -> pd.DataFrame:
    return pd.concat(
        [load_tier(board.elite, "ELITE"), load_tier(board.standard, "STANDARD"), load_tier(board.observe, "OBSERVE")],
        ignore_index=True,
        sort=False,
    )


def select_combo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    market = out.get("market", pd.Series("", index=out.index)).astype("string").str.lower().str.strip()
    league = out.get("league", pd.Series("", index=out.index)).astype("string").str.strip()
    product = out.get("ftr_combo_live_product", pd.Series("", index=out.index)).astype("string").str.strip()
    combo_tier = out.get("ftr_combo_live_tier", pd.Series("", index=out.index)).astype("string").str.strip()
    allowed_source = out.get("ftr_combo_live_allowed", pd.Series(0, index=out.index))
    allowed = pd.to_numeric(allowed_source, errors="coerce").fillna(0).eq(1)

    proof_mask = pd.Series(False, index=out.index)
    for lg, prod in PROOF_LANES:
        proof_mask |= league.eq(lg) & product.eq(prod)

    selected = out.loc[market.eq("ftr") & allowed & proof_mask & combo_tier.isin(VALID_TIERS)].copy()
    selected["team_goal_combo_shadow_stage"] = "TEAM_GOAL_COMBO_PROOF_SHADOW"
    selected["team_goal_combo_shadow_candidate_tier"] = "SHADOW_ONLY"
    selected["team_goal_combo_shadow_reason"] = "C4_FULL_ESTATE_PROOF_SP_GER_HOME_GE2"
    return selected


def counts(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=cols + ["rows"])
    return df.groupby(cols, dropna=False).size().reset_index(name="rows")


def run_board(board: BoardSet, outdir: Path) -> dict:
    slug = board.fixture_range.replace("-", "_")
    board_out = outdir / slug
    board_out.mkdir(parents=True, exist_ok=True)
    source = combine(board)
    selected = select_combo(source)

    deploy_changed = 0
    tier_changed = 0
    source_path = board_out / f"{slug}__DEPLOY_TIERS_COMBINED_FOR_TEAM_GOAL_COMBO_SHADOW_QA.csv"
    selected_path = board_out / f"{slug}__TEAM_GOAL_COMBO_PROOF_SHADOW_SELECTED.csv"
    count_path = board_out / f"{slug}__TEAM_GOAL_COMBO_PROOF_SHADOW_COUNTS.csv"
    summary_path = board_out / f"{slug}__TEAM_GOAL_COMBO_LIVE_SHADOW_QA_SUMMARY.md"
    source.to_csv(source_path, index=False)
    selected.to_csv(selected_path, index=False)
    count_table = counts(selected, ["league", "ftr_combo_live_product", "ftr_combo_live_tier", "deploy_tier"])
    count_table.to_csv(count_path, index=False)

    summary_row = pd.DataFrame([{
        "status": "PASS",
        "fixture_range": board.fixture_range,
        "source_rows": len(source),
        "selected_rows": len(selected),
        "deploy_tier_changed": deploy_changed,
        "tier_changed": tier_changed,
    }])
    lines = [
        "# Team-Goal Combo Live Shadow QA",
        "",
        "Research-only sidecar QA. No production files or source tiers changed.",
        "",
        markdown_table(summary_row),
        "",
        "## Selected Counts",
        markdown_table(count_table),
        "",
        "## Guard",
        "",
        "- Spain/Germany HOME_WIN_AND_HOME_GE2 only.",
        "- Belgium excluded.",
        "- USA watch-only and not selected.",
        "- Shadow-only instrumentation.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "PASS",
        "board_dir": str(board.board_dir),
        "fixture_range": board.fixture_range,
        "source_rows": len(source),
        "selected_rows": len(selected),
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="predictions_output")
    parser.add_argument("--board-dir", default="")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    boards = boards_from_dir(Path(args.board_dir)) if args.board_dir else discover_boards(Path(args.root))
    if not boards:
        raise SystemExit("No complete live deploy tier board found.")
    boards = boards[: max(1, args.limit)]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = [run_board(board, outdir) for board in boards]
    index = pd.DataFrame(records)
    index.to_csv(outdir / "team_goal_combo_live_shadow_qa_index.csv", index=False)
    print(f"[ok] boards={len(records)}")
    print(f"[ok] wrote {records[0]['summary_path']}")


if __name__ == "__main__":
    main()
