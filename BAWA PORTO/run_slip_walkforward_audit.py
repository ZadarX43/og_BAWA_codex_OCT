#!/usr/bin/env python3
"""run_slip_walkforward_audit.py

Walk-forward slip audit for the current slip policy stack.

Goals:
  - scan walk-forward windows
  - rerun or load slip outputs per window
  - score slip products against historical outcomes
  - write master summaries for slip sizes and weak-link behavior

This script is intentionally deterministic and heuristic-first. It is designed
to tell us:
  - whether top P1/P2 legs really survive better in larger accas
  - whether avoid/safe flags are useful
  - which slip sizes remain viable
  - which losing legs keep poisoning larger slips
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SLIP_SIZES = [5, 6, 7, 8, 10, 12, 14]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run walk-forward slip audit over scored deploy windows")
    p.add_argument("--walkforward-root", required=True, help="Walk-forward root containing w*/02_deploy and w*/03_scored")
    p.add_argument("--audit-outdir", required=True, help="Output directory for audit CSVs and optional per-window slips")
    p.add_argument("--rerun-slip-formatter", action="store_true", help="Rebuild per-window slip outputs using current slip_formatter.py")
    p.add_argument("--include-observe", action="store_true", help="Pass through to slip_formatter.py")
    p.add_argument("--block-regime-flags", action="store_true", help="Pass through to slip_formatter.py")
    p.add_argument("--max-slip-per-league", type=int, default=2)
    p.add_argument("--max-slip-per-market", type=int, default=3)
    p.add_argument("--max-slip-per-context-family", type=int, default=2)
    p.add_argument("--large-acca-threshold", type=int, default=5)
    p.add_argument("--monster-mode", choices=["purity", "volume", "both"], default="both")
    return p.parse_args()


def derive_hit(df: pd.DataFrame) -> pd.Series:
    market = df.get("market", pd.Series("", index=df.index)).astype("string").fillna("").str.lower().str.strip()
    ftr_hit = pd.to_numeric(df.get("ftr_hit", np.nan), errors="coerce")
    ou25_hit = pd.to_numeric(df.get("ou25_hit", np.nan), errors="coerce")
    btts_yes_hit = pd.to_numeric(df.get("btts_yes_hit", np.nan), errors="coerce")

    hit = pd.Series(np.nan, index=df.index, dtype="float64")
    hit = hit.mask(market.eq("ftr"), ftr_hit)
    hit = hit.mask(market.eq("ou25"), ou25_hit)
    hit = hit.mask(market.eq("btts"), btts_yes_hit)
    return hit


def normalize_selection(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.upper().str.strip()


def locate_window_files(window_dir: Path) -> dict[str, Path | None]:
    deploy_dir = window_dir / "02_deploy"
    scored_dir = window_dir / "03_scored"

    elite = next(deploy_dir.glob("*__DEPLOY_TIER_ELITE__*.csv"), None) if deploy_dir.exists() else None
    standard = next(deploy_dir.glob("*__DEPLOY_TIER_STANDARD__*.csv"), None) if deploy_dir.exists() else None
    scored = next(scored_dir.glob("DEPLOY_COMBINED_SCORED_*.csv"), None) if scored_dir.exists() else None
    if scored is None and scored_dir.exists():
        scored = next(scored_dir.glob("DEPLOY_CANDIDATES_RAW_SCORED_*.csv"), None)
    return {"elite": elite, "standard": standard, "scored": scored}


def run_slip_formatter_for_window(window_dir: Path, outdir: Path, args: argparse.Namespace) -> None:
    files = locate_window_files(window_dir)
    elite = files["elite"]
    standard = files["standard"]
    if elite is None or standard is None:
        raise RuntimeError(f"Missing deploy tier files for {window_dir.name}")

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "slip_formatter.py"),
        "--inputs",
        str(elite),
        str(standard),
        "--outdir",
        str(outdir),
        "--max-slip-per-league",
        str(args.max_slip_per_league),
        "--max-slip-per-market",
        str(args.max_slip_per_market),
        "--max-slip-per-context-family",
        str(args.max_slip_per_context_family),
        "--large-acca-threshold",
        str(args.large_acca_threshold),
    ]
    if args.include_observe:
        cmd.append("--include-observe")
    if args.block_regime_flags:
        cmd.append("--block-regime-flags")

    subprocess.run(cmd, check=True)


def latest_ranked_board_csv(slip_outdir: Path) -> Path | None:
    files = sorted(slip_outdir.glob("ranked_board_*.csv"))
    files = [
        f for f in files
        if "_ftr_" not in f.name and "_btts_" not in f.name and "_ou25_" not in f.name and "family_summary" not in f.name
    ]
    return files[-1] if files else None


def load_ranked_board(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    for col in ["fixture_key", "market"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").str.strip()
    if "market" in df.columns:
        df["market"] = df["market"].str.lower()
    if "selection" in df.columns:
        df["selection"] = normalize_selection(df["selection"])
    return df


def load_scored_board(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    df["market"] = df.get("market", pd.Series("", index=df.index)).astype("string").fillna("").str.lower().str.strip()
    df["selection"] = normalize_selection(df.get("bookie_pick", df.get("selection", pd.Series("", index=df.index))))
    df["fixture_key"] = df.get("fixture_key", pd.Series("", index=df.index)).astype("string").fillna("").str.strip()
    df["hit"] = derive_hit(df)
    keep = [c for c in [
        "fixture_key", "market", "selection", "hit",
        "ftr_hit", "ou25_hit", "btts_yes_hit",
    ] if c in df.columns]
    return df[keep].copy()


def merge_board_with_hits(board: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return board
    merged = board.merge(
        scored[["fixture_key", "market", "selection", "hit"]].drop_duplicates(
            subset=["fixture_key", "market", "selection"]
        ),
        on=["fixture_key", "market", "selection"],
        how="left",
    )
    merged["graded_flag"] = merged["hit"].notna().astype(int)
    return merged


def summarize_leg_outcomes(board: pd.DataFrame, window_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    graded = board[board["graded_flag"] == 1].copy()
    if graded.empty:
        return pd.DataFrame(), pd.DataFrame()

    graded["window_id"] = window_id
    bucket = (
        graded.groupby(["window_id", "slip_leg_bucket"])
        .agg(rows=("fixture_key", "size"), hit_rate=("hit", "mean"))
        .reset_index()
    )

    flags = (
        graded.groupby(["window_id", "safe_for_large_acca_flag", "avoid_in_acca_flag", "slip_role_hint"])
        .agg(rows=("fixture_key", "size"), hit_rate=("hit", "mean"))
        .reset_index()
    )
    return bucket, flags


def build_prefix_slip(board: pd.DataFrame, size: int) -> pd.DataFrame:
    if len(board) < size:
        return pd.DataFrame()
    return board.head(size).copy()


def _combo_respects_correlation_caps_with_reason(
    combo: list[dict], combo_size: int, args: argparse.Namespace, deep_tail_relaxed: bool = False
) -> tuple[bool, str]:
    league_counts: dict[str, int] = {}
    market_counts: dict[str, int] = {}
    context_counts: dict[str, int] = {}
    eff_league_cap = args.max_slip_per_league
    eff_market_cap = args.max_slip_per_market
    eff_context_cap = args.max_slip_per_context_family
    if combo_size >= 10:
        if eff_league_cap is not None:
            eff_league_cap = max(eff_league_cap, 4)
        if eff_market_cap is not None:
            eff_market_cap = max(eff_market_cap, 5)
        if eff_context_cap is not None:
            eff_context_cap = max(eff_context_cap, 5)
        if deep_tail_relaxed:
            if eff_league_cap is not None:
                eff_league_cap = max(eff_league_cap, 5)
            if eff_market_cap is not None:
                eff_market_cap = max(eff_market_cap, 6)
            if eff_context_cap is not None:
                eff_context_cap = max(eff_context_cap, 6)

    for row in combo:
        lg = str(row.get("league", "") or "")
        mk = str(row.get("market", "") or "").strip().lower()
        ctx = str(row.get("team_context_filter_family", "") or "").strip().upper() or "GENERAL"
        league_counts[lg] = league_counts.get(lg, 0) + 1
        market_counts[mk] = market_counts.get(mk, 0) + 1
        context_counts[ctx] = context_counts.get(ctx, 0) + 1

        if eff_league_cap is not None and league_counts[lg] > eff_league_cap:
            return False, "league_cap_blocked"
        if eff_market_cap is not None and market_counts[mk] > eff_market_cap:
            return False, "market_cap_blocked"
        if eff_context_cap is not None and context_counts[ctx] > eff_context_cap:
            return False, "context_cap_blocked"

    if combo_size >= 10:
        if any(int(pd.to_numeric(r.get("monster_candidate_eligible", 0), errors="coerce") or 0) != 1 for r in combo):
            return False, "monster_candidate_ineligible"
    elif combo_size >= args.large_acca_threshold:
        if any(int(pd.to_numeric(r.get("safe_for_large_acca_flag", 0), errors="coerce") or 0) != 1 for r in combo):
            return False, "large_acca_ineligible"
    return True, "ok"


def monster_modes(args: argparse.Namespace) -> list[str]:
    return ["purity", "volume"] if args.monster_mode == "both" else [args.monster_mode]


def build_constructed_slip(
    board: pd.DataFrame, size: int, args: argparse.Namespace, monster_mode: str
) -> tuple[pd.DataFrame, str, int, dict[str, int]]:
    slot_meta = {
        "core_slots_required": 0,
        "extension_slots_required": 0,
        "deep_tail_slots_required": 0,
        "core_rows_used": 0,
        "extension_rows_used": 0,
        "deep_tail_strict_rows_used": 0,
        "deep_tail_soft_rows_used": 0,
        "deep_tail_fallback_rows_used": 0,
        "deep_tail_relaxed_cap_rows_used": 0,
        "deep_tail_relaxed_cap_mode": int(monster_mode == "volume"),
    }
    if len(board) < size:
        return pd.DataFrame(), "not_enough_rows", 0, slot_meta
    safe_col = "monster_candidate_eligible" if size >= 10 else "safe_for_large_acca_flag"
    candidate_pool = board[pd.to_numeric(board.get(safe_col, 0), errors="coerce").fillna(0).eq(1)].copy() if safe_col in board.columns else board.copy()
    if size >= 10 and "monster_candidate_score" in candidate_pool.columns:
        candidate_pool = candidate_pool.sort_values(["monster_candidate_score", "rank"], ascending=[False, True])
    candidate_pool_size = int(len(candidate_pool))
    if candidate_pool_size < size:
        return pd.DataFrame(), f"not_enough_{safe_col}", candidate_pool_size, slot_meta
    chosen: list[dict] = []
    blocker_counts: dict[str, int] = {}
    source_records = candidate_pool.to_dict("records") if size >= 10 else board.to_dict("records")

    def _consume_stage(
        stage_records: list[dict],
        needed_total: int,
        stage_name: str,
        strict_fixture_keys: set[str] | None = None,
        soft_fixture_keys: set[str] | None = None,
        deep_tail_relaxed: bool = False,
    ) -> bool:
        nonlocal chosen, blocker_counts
        if needed_total <= len(chosen):
            return True
        for row in stage_records:
            if any(str(existing.get("fixture_key", "") or "") == str(row.get("fixture_key", "") or "") for existing in chosen):
                continue
            candidate = chosen + [row]
            ok, blocker_reason = _combo_respects_correlation_caps_with_reason(candidate, size, args, deep_tail_relaxed=deep_tail_relaxed)
            if not ok:
                blocker_counts[blocker_reason] = blocker_counts.get(blocker_reason, 0) + 1
                continue
            chosen = candidate
            if stage_name == "core":
                slot_meta["core_rows_used"] += 1
            elif stage_name == "extension":
                slot_meta["extension_rows_used"] += 1
            elif stage_name == "deep_tail":
                fx = str(row.get("fixture_key", "") or "")
                if strict_fixture_keys is not None and fx in strict_fixture_keys:
                    slot_meta["deep_tail_strict_rows_used"] += 1
                elif soft_fixture_keys is not None and fx in soft_fixture_keys:
                    slot_meta["deep_tail_soft_rows_used"] += 1
                else:
                    slot_meta["deep_tail_fallback_rows_used"] += 1
                if deep_tail_relaxed:
                    slot_meta["deep_tail_relaxed_cap_rows_used"] += 1
            if len(chosen) >= needed_total:
                break
        return len(chosen) >= needed_total

    if size >= 10:
        core_target = min(size, 8)
        extension_target = min(max(size - core_target, 0), 2)
        deep_tail_target = max(size - core_target - extension_target, 0)
        slot_meta["core_slots_required"] = core_target
        slot_meta["extension_slots_required"] = extension_target
        slot_meta["deep_tail_slots_required"] = deep_tail_target

        extension_records = [
            r for r in source_records
            if int(pd.to_numeric(r.get("monster_extension_pool_a_eligible", 0), errors="coerce") or 0) == 1
        ]
        deep_tail_records = [
            r for r in source_records
            if int(pd.to_numeric(r.get("monster_deep_tail_eligible", 0), errors="coerce") or 0) == 1
        ]
        deep_tail_soft_records = [
            r for r in source_records
            if int(pd.to_numeric(r.get("monster_deep_tail_soft_eligible", 0), errors="coerce") or 0) == 1
        ]
        deep_tail_fixture_keys = {str(r.get("fixture_key", "") or "") for r in deep_tail_records}
        deep_tail_soft_fixture_keys = {
            str(r.get("fixture_key", "") or "")
            for r in deep_tail_soft_records
            if str(r.get("fixture_key", "") or "") not in deep_tail_fixture_keys
        }
        deep_tail_soft_records = [
            r for r in deep_tail_soft_records
            if str(r.get("fixture_key", "") or "") in deep_tail_soft_fixture_keys
        ]
        deep_tail_with_fallback_records = deep_tail_records + deep_tail_soft_records + [
            r for r in extension_records
            if str(r.get("fixture_key", "") or "") not in (deep_tail_fixture_keys | deep_tail_soft_fixture_keys)
        ]

        if not _consume_stage(source_records, core_target, "core"):
            failure_reason = max(sorted(blocker_counts.items()), key=lambda item: item[1])[0] if blocker_counts else "correlation_caps_blocked"
            return pd.DataFrame(), failure_reason, candidate_pool_size, slot_meta
        if not _consume_stage(extension_records, core_target + extension_target, "extension"):
            failure_reason = max(sorted(blocker_counts.items()), key=lambda item: item[1])[0] if blocker_counts else "extension_pool_blocked"
            return pd.DataFrame(), failure_reason, candidate_pool_size, slot_meta
        if not _consume_stage(
            deep_tail_with_fallback_records,
            core_target + extension_target + deep_tail_target,
            "deep_tail",
            deep_tail_fixture_keys,
            deep_tail_soft_fixture_keys,
            monster_mode == "volume",
        ):
            failure_reason = max(sorted(blocker_counts.items()), key=lambda item: item[1])[0] if blocker_counts else "deep_tail_pool_blocked"
            return pd.DataFrame(), failure_reason, candidate_pool_size, slot_meta
    else:
        for row in source_records:
            candidate = chosen + [row]
            ok, blocker_reason = _combo_respects_correlation_caps_with_reason(candidate, size, args)
            if not ok:
                blocker_counts[blocker_reason] = blocker_counts.get(blocker_reason, 0) + 1
                continue
            chosen = candidate
            if len(chosen) >= size:
                break
    if len(chosen) < size:
        if blocker_counts:
            failure_reason = max(
                sorted(blocker_counts.items()),
                key=lambda item: item[1],
            )[0]
        else:
            failure_reason = "correlation_caps_blocked"
        return pd.DataFrame(), failure_reason, candidate_pool_size, slot_meta
    return pd.DataFrame(chosen), "ok", candidate_pool_size, slot_meta


def build_audit_slip(
    board: pd.DataFrame, size: int, args: argparse.Namespace, monster_mode: str
) -> tuple[pd.DataFrame, str, str, int, dict[str, int]]:
    if size >= 10:
        slip, reason, pool_size, slot_meta = build_constructed_slip(board, size, args, monster_mode)
        return slip, "constructed", reason, pool_size, slot_meta
    return build_prefix_slip(board, size), "prefix", "ok", int(len(board)), {
        "core_slots_required": 0,
        "extension_slots_required": 0,
        "deep_tail_slots_required": 0,
        "core_rows_used": 0,
        "extension_rows_used": 0,
        "deep_tail_strict_rows_used": 0,
        "deep_tail_soft_rows_used": 0,
        "deep_tail_fallback_rows_used": 0,
        "deep_tail_relaxed_cap_rows_used": 0,
    }


def score_prefix_slips(board: pd.DataFrame, window_id: str, args: argparse.Namespace, monster_mode: str) -> pd.DataFrame:
    out_rows: list[dict] = []
    for size in SLIP_SIZES:
        slip, build_mode, failure_reason, candidate_pool_size, slot_meta = build_audit_slip(board, size, args, monster_mode)
        if slip.empty:
            out_rows.append({
                "window_id": window_id,
                "monster_mode": monster_mode,
                "slip_size": size,
                "build_mode": build_mode,
                "buildable_flag": 0,
                "failure_reason": failure_reason,
                "candidate_pool_size": candidate_pool_size,
                "monster_safe_rows": int(pd.to_numeric(board.get("safe_for_monster_acca_flag", 0), errors="coerce").fillna(0).eq(1).sum()),
                "monster_candidate_rows": int(pd.to_numeric(board.get("monster_candidate_eligible", 0), errors="coerce").fillna(0).eq(1).sum()),
                "available_rows": int(len(board)),
                "legs": 0,
                "graded_all": 0,
                "survived_all": 0,
                "legs_landed": 0,
                "legs_failed": 0,
                "weakest_failed_rank": np.nan,
                "weakest_failed_safe_flag": np.nan,
                "weakest_failed_avoid_flag": np.nan,
                "weakest_failed_monster_caution_flag": np.nan,
                "weakest_failed_role_hint": "",
                "any_avoid_flag_in_slip": 0,
                "any_monster_caution_flag_in_slip": 0,
                "all_large_acca_safe": 0,
                "p1_p2_only": 0,
                **slot_meta,
            })
            continue
        graded = slip["hit"].notna().all()
        landed = int(pd.to_numeric(slip["hit"], errors="coerce").fillna(0).sum()) if graded else int(pd.to_numeric(slip["hit"], errors="coerce").fillna(0).sum())
        failed = slip[pd.to_numeric(slip["hit"], errors="coerce").fillna(0).eq(0)].copy()
        weakest_failed = failed.sort_values("rank", ascending=False).head(1)
        effective_safe_col = "safe_for_monster_acca_flag" if size >= 10 and "safe_for_monster_acca_flag" in slip.columns else "safe_for_large_acca_flag"
        out_rows.append({
            "window_id": window_id,
            "monster_mode": monster_mode,
            "slip_size": size,
            "build_mode": build_mode,
            "buildable_flag": 1,
            "failure_reason": failure_reason,
            "candidate_pool_size": candidate_pool_size,
            "monster_safe_rows": int(pd.to_numeric(board.get("safe_for_monster_acca_flag", 0), errors="coerce").fillna(0).eq(1).sum()),
            "monster_candidate_rows": int(pd.to_numeric(board.get("monster_candidate_eligible", 0), errors="coerce").fillna(0).eq(1).sum()),
            "available_rows": int(len(board)),
            "legs": int(len(slip)),
            "graded_all": int(slip["graded_flag"].eq(1).all()),
            "survived_all": int(pd.to_numeric(slip["hit"], errors="coerce").fillna(0).eq(1).all()),
            "legs_landed": landed,
            "legs_failed": int(len(slip) - landed),
            "weakest_failed_rank": int(weakest_failed["rank"].iloc[0]) if not weakest_failed.empty else np.nan,
            "weakest_failed_safe_flag": int(weakest_failed[effective_safe_col].iloc[0]) if not weakest_failed.empty and effective_safe_col in weakest_failed.columns else np.nan,
            "weakest_failed_avoid_flag": int(weakest_failed["avoid_in_acca_flag"].iloc[0]) if not weakest_failed.empty and "avoid_in_acca_flag" in weakest_failed.columns else np.nan,
            "weakest_failed_monster_caution_flag": int(weakest_failed["monster_caution_flag"].iloc[0]) if not weakest_failed.empty and "monster_caution_flag" in weakest_failed.columns else np.nan,
            "weakest_failed_role_hint": str(weakest_failed["slip_role_hint"].iloc[0]) if not weakest_failed.empty and "slip_role_hint" in weakest_failed.columns else "",
            "any_avoid_flag_in_slip": int(pd.to_numeric(slip.get("avoid_in_acca_flag", 0), errors="coerce").fillna(0).ge(1).any()),
            "any_monster_caution_flag_in_slip": int(pd.to_numeric(slip.get("monster_caution_flag", 0), errors="coerce").fillna(0).ge(1).any()),
            "all_large_acca_safe": int(pd.to_numeric(slip.get(effective_safe_col, 0), errors="coerce").fillna(0).eq(1).all()),
            "p1_p2_only": int(slip.get("slip_leg_bucket", pd.Series("", index=slip.index)).astype("string").isin(["P1", "P2"]).all()),
            **slot_meta,
        })
    return pd.DataFrame(out_rows)


def main() -> None:
    args = parse_args()
    root = Path(args.walkforward_root)
    outdir = Path(args.audit_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slips_root = outdir / "_WINDOW_SLIPS"
    slips_root.mkdir(parents=True, exist_ok=True)

    window_rows = []
    leg_bucket_rows = []
    leg_flag_rows = []

    windows = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("w")])
    for window_dir in windows:
        files = locate_window_files(window_dir)
        if files["elite"] is None or files["standard"] is None or files["scored"] is None:
            continue

        slip_outdir = slips_root / window_dir.name
        slip_outdir.mkdir(parents=True, exist_ok=True)

        if args.rerun_slip_formatter or latest_ranked_board_csv(slip_outdir) is None:
            run_slip_formatter_for_window(window_dir, slip_outdir, args)

        ranked_path = latest_ranked_board_csv(slip_outdir)
        if ranked_path is None:
            continue

        board = load_ranked_board(ranked_path)
        scored = load_scored_board(files["scored"])
        board = merge_board_with_hits(board, scored)
        if board.empty:
            continue

        board["window_id"] = window_dir.name
        board.to_csv(outdir / f"{window_dir.name}__RANKED_BOARD_SCORED.csv", index=False)

        bucket_df, flags_df = summarize_leg_outcomes(board, window_dir.name)
        if not bucket_df.empty:
            leg_bucket_rows.append(bucket_df)
        if not flags_df.empty:
            leg_flag_rows.append(flags_df)

        for mode in monster_modes(args):
            slip_df = score_prefix_slips(board, window_dir.name, args, mode)
            if not slip_df.empty:
                window_rows.append(slip_df)

    if not window_rows:
        raise SystemExit("No scored window slip outputs were produced.")

    window_summary = pd.concat(window_rows, ignore_index=True)
    bucket_summary = pd.concat(leg_bucket_rows, ignore_index=True) if leg_bucket_rows else pd.DataFrame()
    flag_summary = pd.concat(leg_flag_rows, ignore_index=True) if leg_flag_rows else pd.DataFrame()

    survival = (
        window_summary.groupby(["monster_mode", "slip_size", "build_mode"])
        .agg(
            windows=("window_id", "nunique"),
            buildable_rate=("buildable_flag", "mean"),
            mean_available_rows=("available_rows", "mean"),
            mean_candidate_pool_size=("candidate_pool_size", "mean"),
            mean_monster_safe_rows=("monster_safe_rows", "mean"),
            mean_monster_candidate_rows=("monster_candidate_rows", "mean"),
            complete_slip_rate=("survived_all", "mean"),
            mean_legs_landed=("legs_landed", "mean"),
            mean_legs_failed=("legs_failed", "mean"),
            weakest_failed_avoid_rate=("weakest_failed_avoid_flag", "mean"),
            weakest_failed_monster_caution_rate=("weakest_failed_monster_caution_flag", "mean"),
            all_large_acca_safe_rate=("all_large_acca_safe", "mean"),
            p1_p2_only_rate=("p1_p2_only", "mean"),
            mean_core_slots_required=("core_slots_required", "mean"),
            mean_extension_slots_required=("extension_slots_required", "mean"),
            mean_deep_tail_slots_required=("deep_tail_slots_required", "mean"),
            mean_core_rows_used=("core_rows_used", "mean"),
            mean_extension_rows_used=("extension_rows_used", "mean"),
            mean_deep_tail_strict_rows_used=("deep_tail_strict_rows_used", "mean"),
            mean_deep_tail_soft_rows_used=("deep_tail_soft_rows_used", "mean"),
            mean_deep_tail_fallback_rows_used=("deep_tail_fallback_rows_used", "mean"),
            mean_deep_tail_relaxed_cap_rows_used=("deep_tail_relaxed_cap_rows_used", "mean"),
        )
        .reset_index()
    )
    built_only = window_summary[window_summary["buildable_flag"].eq(1)].copy()
    if not built_only.empty:
        built_survival = (
            built_only.groupby(["monster_mode", "slip_size", "build_mode"])
            .agg(
                complete_slip_rate_when_built=("survived_all", "mean"),
                mean_legs_landed_when_built=("legs_landed", "mean"),
                mean_legs_failed_when_built=("legs_failed", "mean"),
            )
            .reset_index()
        )
        survival = survival.merge(built_survival, on=["monster_mode", "slip_size", "build_mode"], how="left")
    else:
        survival["complete_slip_rate_when_built"] = np.nan
        survival["mean_legs_landed_when_built"] = np.nan
        survival["mean_legs_failed_when_built"] = np.nan

    weak_link_flags = (
        window_summary.groupby(["monster_mode", "slip_size", "build_mode"])
        .agg(
            rows=("window_id", "size"),
            buildable_rate=("buildable_flag", "mean"),
            mean_monster_candidate_rows=("monster_candidate_rows", "mean"),
            weakest_failed_avoid_rate=("weakest_failed_avoid_flag", "mean"),
            weakest_failed_monster_caution_rate=("weakest_failed_monster_caution_flag", "mean"),
            weakest_failed_safe_rate=("weakest_failed_safe_flag", "mean"),
            any_avoid_flag_in_slip_rate=("any_avoid_flag_in_slip", "mean"),
            any_monster_caution_flag_in_slip_rate=("any_monster_caution_flag_in_slip", "mean"),
            mean_core_rows_used=("core_rows_used", "mean"),
            mean_extension_rows_used=("extension_rows_used", "mean"),
            mean_deep_tail_strict_rows_used=("deep_tail_strict_rows_used", "mean"),
            mean_deep_tail_soft_rows_used=("deep_tail_soft_rows_used", "mean"),
            mean_deep_tail_fallback_rows_used=("deep_tail_fallback_rows_used", "mean"),
            mean_deep_tail_relaxed_cap_rows_used=("deep_tail_relaxed_cap_rows_used", "mean"),
        )
        .reset_index()
    )

    out_files = {
        "window_summary": outdir / "SLIP_WALKFORWARD__WINDOW_SUMMARY.csv",
        "survival": outdir / "SLIP_WALKFORWARD__SURVIVAL_BY_SIZE.csv",
        "weak_flags": outdir / "SLIP_WALKFORWARD__WEAK_LINK_FLAGS.csv",
        "bucket": outdir / "SLIP_WALKFORWARD__LEG_BUCKETS_BY_WINDOW.csv",
        "flag": outdir / "SLIP_WALKFORWARD__LEG_FLAGS_BY_WINDOW.csv",
    }
    window_summary.to_csv(out_files["window_summary"], index=False)
    survival.to_csv(out_files["survival"], index=False)
    weak_link_flags.to_csv(out_files["weak_flags"], index=False)
    if not bucket_summary.empty:
        bucket_summary.to_csv(out_files["bucket"], index=False)
    if not flag_summary.empty:
        flag_summary.to_csv(out_files["flag"], index=False)

    print("WROTE:")
    for path in out_files.values():
        if path.exists():
            print(path)

    print("\nSurvival by size:")
    print(survival.to_string(index=False))
    print("\nWeak-link flags:")
    print(weak_link_flags.to_string(index=False))


if __name__ == "__main__":
    main()
