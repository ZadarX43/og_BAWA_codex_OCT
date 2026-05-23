#!/usr/bin/env python3
"""Live-board shadow QA for strict FTR + BTTS combo candidates.

Research-only. Builds same-fixture FTR + BTTS synthetic combo rows from live
deploy tier files, applies only window-stable discovery thresholds, and writes
shadow outputs without mutating source deploy tiers.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_POLICY = Path(
    "reports/2026-05-06/ftr_btts_combo_window_stability/"
    "ftr_btts_combo_window_threshold_candidate_stability.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/ftr_btts_combo_live_shadow_qa")

SPECIAL_COMPS = {
    "England FA Cup",
    "Champions League",
    "Europa League",
    "Europa Conference",
}


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


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


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
    for base, tiers in grouped.items():
        if {"elite", "standard", "observe"}.issubset(tiers):
            boards.append(BoardSet(board_dir, base, tiers["elite"], tiers["standard"], tiers["observe"]))
    return sorted(boards, key=lambda board: board.max_mtime, reverse=True)


def load_tier(path: Path, tier: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "deploy_tier" not in df.columns:
        df["deploy_tier"] = tier
    if "tier" not in df.columns:
        df["tier"] = df["deploy_tier"]
    df["source_tier_file"] = path.name
    df["source_tier_expected"] = tier
    return df


def combine(board: BoardSet) -> pd.DataFrame:
    return pd.concat(
        [
            load_tier(board.elite, "ELITE"),
            load_tier(board.standard, "STANDARD"),
            load_tier(board.observe, "OBSERVE"),
        ],
        ignore_index=True,
        sort=False,
    )


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_market_norm"] = out.get("market", pd.Series("", index=out.index)).astype("string").str.lower().str.strip()
    out["_selection_norm"] = (
        out.get("selection", out.get("bookie_pick", pd.Series("", index=out.index)))
        .astype("string")
        .fillna("")
        .str.upper()
        .str.strip()
    )
    return out


def build_combo_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize(df)
    ftr = df[df["_market_norm"].eq("ftr") & df["_selection_norm"].isin(["HOME", "AWAY"])].copy()
    btts = df[df["_market_norm"].eq("btts") & df["_selection_norm"].isin(["YES", "NO"])].copy()
    if ftr.empty or btts.empty:
        return pd.DataFrame()

    keep_ftr = [
        "league",
        "match_date",
        "home_team_name",
        "away_team_name",
        "fixture_key",
        "_selection_norm",
        "deploy_tier",
        "tier",
        "bookie_od",
        "model_p_for_bookie",
        "p_meta_ftr",
        "ftr_margin",
        "pick_side_mass_top3",
        "pick_side_margin_top3",
        "value_edge",
        "value_edge_tier",
        "context_reason_codes",
        "source_tier_file",
    ]
    keep_btts = [
        "league",
        "fixture_key",
        "_selection_norm",
        "deploy_tier",
        "tier",
        "bookie_od",
        "model_p_for_bookie",
        "p_meta_btts",
        "cs_mass_btts_yes",
        "cs_mass_btts_no",
        "p00_est",
        "p_home_fts",
        "p_away_fts",
        "value_edge",
        "value_edge_tier",
        "source_tier_file",
    ]

    ftr = ftr[[c for c in keep_ftr if c in ftr.columns]].rename(
        columns={
            "_selection_norm": "ftr_side",
            "deploy_tier": "ftr_deploy_tier",
            "tier": "ftr_tier",
            "bookie_od": "ftr_od",
            "model_p_for_bookie": "ftr_model_p",
            "value_edge": "ftr_value_edge",
            "value_edge_tier": "ftr_value_edge_tier",
            "source_tier_file": "ftr_source_tier_file",
        }
    )
    btts = btts[[c for c in keep_btts if c in btts.columns]].rename(
        columns={
            "_selection_norm": "btts_side",
            "deploy_tier": "btts_deploy_tier",
            "tier": "btts_tier",
            "bookie_od": "btts_od",
            "model_p_for_bookie": "btts_model_p",
            "value_edge": "btts_value_edge",
            "value_edge_tier": "btts_value_edge_tier",
            "source_tier_file": "btts_source_tier_file",
        }
    )

    merged = ftr.merge(btts, on=["league", "fixture_key"], how="inner")
    if merged.empty:
        return merged

    merged["combo_product"] = merged["ftr_side"] + "_AND_BTTS_" + merged["btts_side"]
    merged["synthetic_combo_od"] = num(merged.get("ftr_od", np.nan)) * num(merged.get("btts_od", np.nan))
    merged["synthetic_combo_model_p"] = num(merged.get("ftr_model_p", np.nan)) * num(merged.get("btts_model_p", np.nan))
    merged["synthetic_combo_implied"] = 1.0 / merged["synthetic_combo_od"].replace(0, np.nan)
    merged["synthetic_combo_value_edge"] = merged["synthetic_combo_model_p"] - merged["synthetic_combo_implied"]
    merged["combo_research_family"] = np.where(
        merged["combo_product"].astype("string").str.endswith("_NO"),
        "FTR_PLUS_BTTS_NO",
        "SPECIAL_COMP_FTR_PLUS_BTTS_YES",
    )
    return merged


def load_policy(
    path: Path,
    *,
    min_windows: int,
    min_graded: int,
    min_hit_rate: float,
    min_p25_hit_rate: float,
    max_negative_roi_windows: int,
    limit: int,
) -> pd.DataFrame:
    policy = pd.read_csv(path)
    required = {"league", "combo_product", "candidate_feature", "candidate_threshold"}
    missing = sorted(required - set(policy.columns))
    if missing:
        raise SystemExit(f"Policy missing required columns: {missing}")

    filtered = policy[
        num(policy.get("active_windows", 0)).ge(min_windows)
        & num(policy.get("graded", 0)).ge(min_graded)
        & num(policy.get("hit_rate", 0)).ge(min_hit_rate)
        & num(policy.get("p25_window_hit_rate", 0)).ge(min_p25_hit_rate)
        & num(policy.get("windows_negative_roi", 999)).le(max_negative_roi_windows)
    ].copy()

    product = filtered["combo_product"].astype("string")
    league = filtered["league"].astype("string")
    is_no = product.isin(["HOME_AND_BTTS_NO", "AWAY_AND_BTTS_NO"])
    is_special_yes = product.isin(["HOME_AND_BTTS_YES", "AWAY_AND_BTTS_YES"]) & league.isin(SPECIAL_COMPS)
    filtered = filtered[is_no | is_special_yes].copy()

    filtered = filtered.sort_values(["hit_rate", "graded", "roi"], ascending=[False, False, False])
    return filtered.head(limit).reset_index(drop=True)


def apply_policy(combo: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    if combo.empty or policy.empty:
        return pd.DataFrame()

    selected = []
    for _, rule in policy.iterrows():
        feature = str(rule["candidate_feature"])
        if feature not in combo.columns:
            continue
        mask = (
            combo["league"].astype("string").eq(str(rule["league"]))
            & combo["combo_product"].astype("string").eq(str(rule["combo_product"]))
            & num(combo[feature]).ge(float(rule["candidate_threshold"]))
        )
        part = combo.loc[mask].copy()
        if part.empty:
            continue
        part["ftr_btts_combo_shadow_stage"] = "FTR_BTTS_COMBO_STRICT_SHADOW"
        part["ftr_btts_combo_candidate_id"] = str(rule.get("candidate_id", ""))
        part["ftr_btts_combo_candidate_feature"] = feature
        part["ftr_btts_combo_candidate_threshold"] = float(rule["candidate_threshold"])
        part["ftr_btts_combo_backtest_hit_rate"] = float(rule.get("hit_rate", np.nan))
        part["ftr_btts_combo_backtest_graded"] = int(rule.get("graded", 0))
        part["ftr_btts_combo_backtest_active_windows"] = int(rule.get("active_windows", 0))
        selected.append(part)

    return pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()


def dedupe_selected(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    sort_cols = [
        "ftr_btts_combo_backtest_hit_rate",
        "ftr_btts_combo_backtest_graded",
        "synthetic_combo_value_edge",
    ]
    out = selected.sort_values(sort_cols, ascending=[False, False, False]).copy()
    dedupe_key = ["fixture_key", "combo_product"]
    out["ftr_btts_combo_shadow_rule_count"] = out.groupby(dedupe_key)["ftr_btts_combo_candidate_id"].transform("count")
    return out.drop_duplicates(dedupe_key, keep="first").reset_index(drop=True)


def counts(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=cols + ["rows"])
    return df.groupby(cols, dropna=False).size().reset_index(name="rows")


def run_board(board: BoardSet, policy: pd.DataFrame, outdir: Path) -> dict:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", board.fixture_range.replace("-", "_"))
    board_out = outdir / slug
    board_out.mkdir(parents=True, exist_ok=True)

    source = combine(board)
    combo = build_combo_rows(source)
    selected_raw = apply_policy(combo, policy)
    selected = dedupe_selected(selected_raw)
    shadow_cols = [
        "ftr_btts_combo_shadow_stage",
        "ftr_btts_combo_candidate_id",
        "ftr_btts_combo_candidate_feature",
        "ftr_btts_combo_candidate_threshold",
        "ftr_btts_combo_backtest_hit_rate",
        "ftr_btts_combo_backtest_graded",
        "ftr_btts_combo_backtest_active_windows",
        "ftr_btts_combo_shadow_rule_count",
    ]
    if selected_raw.empty:
        selected_raw = combo.head(0).copy()
        for col in shadow_cols:
            selected_raw[col] = pd.Series(dtype="object")
    if selected.empty:
        selected = selected_raw.head(0).copy()
        for col in shadow_cols:
            if col not in selected.columns:
                selected[col] = pd.Series(dtype="object")

    deploy_changed = 0
    tier_changed = 0

    source_path = board_out / f"{slug}__DEPLOY_TIERS_COMBINED_FOR_FTR_BTTS_COMBO_SHADOW_QA.csv"
    combo_path = board_out / f"{slug}__FTR_BTTS_COMBO_LIVE_COMBO_ROWS.csv"
    selected_raw_path = board_out / f"{slug}__FTR_BTTS_COMBO_STRICT_SHADOW_SELECTED_RAW.csv"
    selected_path = board_out / f"{slug}__FTR_BTTS_COMBO_STRICT_SHADOW_SELECTED_DEDUPED.csv"
    policy_path = board_out / f"{slug}__FTR_BTTS_COMBO_STRICT_SHADOW_POLICY_USED.csv"
    count_path = board_out / f"{slug}__FTR_BTTS_COMBO_STRICT_SHADOW_COUNTS.csv"
    summary_path = board_out / f"{slug}__FTR_BTTS_COMBO_LIVE_SHADOW_QA_SUMMARY.md"

    source.to_csv(source_path, index=False)
    combo.to_csv(combo_path, index=False)
    selected_raw.to_csv(selected_raw_path, index=False)
    selected.to_csv(selected_path, index=False)
    policy.to_csv(policy_path, index=False)

    count_table = counts(
        selected,
        ["league", "combo_product", "combo_research_family", "ftr_deploy_tier", "btts_deploy_tier"],
    )
    count_table.to_csv(count_path, index=False)

    summary_row = pd.DataFrame(
        [
            {
                "status": "PASS",
                "fixture_range": board.fixture_range,
                "source_rows": len(source),
                "combo_rows": len(combo),
                "selected_raw_rows": len(selected_raw),
                "selected_deduped_rows": len(selected),
                "policy_rules_used": len(policy),
                "deploy_tier_changed": deploy_changed,
                "tier_changed": tier_changed,
            }
        ]
    )

    lines = [
        "# FTR + BTTS Combo Live Shadow QA",
        "",
        "Research-only sidecar QA. No production files or source tiers changed.",
        "",
        markdown_table(summary_row),
        "",
        "## Selected Counts",
        markdown_table(count_table),
        "",
        "## Policy Guard",
        "",
        "- Strict window-stable threshold candidates only.",
        "- `HOME/AWAY + BTTS NO` can appear across leagues when policy-qualified.",
        "- `HOME/AWAY + BTTS YES` remains special-competition-only.",
        "- Shadow-only instrumentation; no deploy tier or source tier mutation.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "PASS",
        "board_dir": str(board.board_dir),
        "fixture_range": board.fixture_range,
        "source_rows": len(source),
        "combo_rows": len(combo),
        "selected_raw_rows": len(selected_raw),
        "selected_deduped_rows": len(selected),
        "policy_rules_used": len(policy),
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="predictions_output")
    parser.add_argument("--board-dir", default="")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--policy-limit", type=int, default=30)
    parser.add_argument("--min-windows", type=int, default=20)
    parser.add_argument("--min-graded", type=int, default=40)
    parser.add_argument("--min-hit-rate", type=float, default=0.95)
    parser.add_argument("--min-p25-hit-rate", type=float, default=1.0)
    parser.add_argument("--max-negative-roi-windows", type=int, default=1)
    args = parser.parse_args()

    boards = boards_from_dir(Path(args.board_dir)) if args.board_dir else discover_boards(Path(args.root))
    if not boards:
        raise SystemExit("No complete live deploy tier board found.")
    boards = boards[: max(1, args.limit)]

    policy = load_policy(
        Path(args.policy),
        min_windows=args.min_windows,
        min_graded=args.min_graded,
        min_hit_rate=args.min_hit_rate,
        min_p25_hit_rate=args.min_p25_hit_rate,
        max_negative_roi_windows=args.max_negative_roi_windows,
        limit=args.policy_limit,
    )
    if policy.empty:
        raise SystemExit("No policy rows survived strict filters.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = [run_board(board, policy, outdir) for board in boards]
    index = pd.DataFrame(records)
    index.to_csv(outdir / "ftr_btts_combo_live_shadow_qa_index.csv", index=False)
    print(f"[ok] boards={len(records)}")
    print(f"[ok] policy_rules={len(policy)}")
    print(f"[ok] wrote {records[0]['summary_path']}")


if __name__ == "__main__":
    main()
