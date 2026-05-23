#!/usr/bin/env python3
"""build_same_fixture_composite_audit.py

Audit same-fixture composite pairs from ranked-board walk-forward outputs.

This is a separate experiment lane from the diff-fixture monster builder. It answers:
- which same-fixture pair families are available
- how often both legs land
- how they behave by timing/season
- how they compare with the current monster-tail baseline
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

WINDOW_PAT = re.compile(r"w\d+_(\d{4})_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build same-fixture composite pair audit")
    p.add_argument("--audit-dir", required=True, help="Audit directory containing ranked-board-scored CSVs")
    p.add_argument("--outdir", required=True, help="Directory for composite audit outputs")
    p.add_argument("--top-pairs-per-window", type=int, default=12, help="Cap exported best pairs per family per window")
    return p.parse_args()


def parse_window_date(window_id: str) -> pd.Timestamp:
    m = WINDOW_PAT.match(str(window_id))
    if not m:
        return pd.NaT
    return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))


def season_phase(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "UNKNOWN"
    month = int(ts.month)
    if month in {8, 9}:
        return "EARLY_SEASON"
    if month in {10, 11, 12}:
        return "AUTUMN_DENSE"
    if month in {1, 2}:
        return "WINTER_RESET"
    if month in {3, 4, 5}:
        return "RUN_IN"
    return "SUMMER_TRANSITION"


def champions_league_phase_flag(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return 0
    return int(ts.month in {9, 10, 11, 12, 2, 3, 4})


def international_break_zone_flag(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return 0
    return int(ts.month in {3, 9, 10, 11})


def _norm_market(v: object) -> str:
    return str(v or "").strip().lower()


def _norm_sel(v: object) -> str:
    return str(v or "").strip().upper()


def classify_pair(a: pd.Series, b: pd.Series) -> tuple[str, str] | None:
    ma, mb = _norm_market(a.get("market")), _norm_market(b.get("market"))
    sa, sb = _norm_sel(a.get("selection")), _norm_sel(b.get("selection"))
    markets = {ma, mb}

    if markets == {"ftr", "ou25"}:
        ou = a if ma == "ou25" else b
        ftr = a if ma == "ftr" else b
        ou_sel = _norm_sel(ou.get("selection"))
        if ou_sel == "OVER25":
            return "FTR_PLUS_OVER25", f"{_norm_sel(ftr.get('selection'))}+OVER25"
        if ou_sel == "UNDER25":
            return "FTR_PLUS_UNDER25", f"{_norm_sel(ftr.get('selection'))}+UNDER25"
    if markets == {"ftr", "btts"}:
        btts = a if ma == "btts" else b
        ftr = a if ma == "ftr" else b
        return "FTR_PLUS_BTTS", f"{_norm_sel(ftr.get('selection'))}+BTTS_{_norm_sel(btts.get('selection'))}"
    if "ftr" in markets and any(m in {"home_goals", "away_goals", "team_goals", "tg15", "team_goals_15"} for m in markets):
        tg = a if ma != "ftr" else b
        ftr = a if ma == "ftr" else b
        tg_sel = _norm_sel(tg.get("selection"))
        if "1.5" in tg_sel or "OVER15" in tg_sel or "OVER_1_5" in tg_sel:
            return "FTR_PLUS_TEAM_GOALS15", f"{_norm_sel(ftr.get('selection'))}+{tg_sel}"
    return None


def pair_score(a: pd.Series, b: pd.Series) -> float:
    return float(pd.to_numeric(a.get("monster_candidate_score", a.get("slip_leg_score", 0.0)), errors="coerce") or 0.0) + float(
        pd.to_numeric(b.get("monster_candidate_score", b.get("slip_leg_score", 0.0)), errors="coerce") or 0.0
    )


def pair_detail_row(window_id: str, a: pd.Series, b: pd.Series, pair_family: str, pair_label: str) -> dict:
    ts = parse_window_date(window_id)
    ahit = pd.to_numeric(a.get("hit"), errors="coerce")
    bhit = pd.to_numeric(b.get("hit"), errors="coerce")
    both_graded = int(pd.notna(ahit) and pd.notna(bhit))
    both_hit = int(both_graded and ahit == 1 and bhit == 1)
    landed_legs = int((0 if pd.isna(ahit) else ahit) + (0 if pd.isna(bhit) else bhit))
    caution_count = int(pd.to_numeric(a.get("monster_caution_flag", 0), errors="coerce") == 1) + int(pd.to_numeric(b.get("monster_caution_flag", 0), errors="coerce") == 1)
    avoid_count = int(pd.to_numeric(a.get("avoid_in_monster_acca_flag", 0), errors="coerce") == 1) + int(pd.to_numeric(b.get("avoid_in_monster_acca_flag", 0), errors="coerce") == 1)
    return {
        "window_id": window_id,
        "window_date_from": ts.date().isoformat() if not pd.isna(ts) else "",
        "window_month": int(ts.month) if not pd.isna(ts) else pd.NA,
        "window_year": int(ts.year) if not pd.isna(ts) else pd.NA,
        "season_phase": season_phase(ts),
        "champions_league_phase_flag": champions_league_phase_flag(ts),
        "international_break_zone_flag": international_break_zone_flag(ts),
        "fixture_key": a.get("fixture_key", ""),
        "league": a.get("league", ""),
        "home": a.get("home", ""),
        "away": a.get("away", ""),
        "pair_family": pair_family,
        "pair_label": pair_label,
        "leg1_market": a.get("market", ""),
        "leg1_selection": a.get("selection", ""),
        "leg1_rank": pd.to_numeric(a.get("rank"), errors="coerce"),
        "leg1_score": pd.to_numeric(a.get("slip_leg_score"), errors="coerce"),
        "leg1_monster_score": pd.to_numeric(a.get("monster_candidate_score"), errors="coerce"),
        "leg1_caution_flag": pd.to_numeric(a.get("monster_caution_flag", 0), errors="coerce"),
        "leg1_caution_reason": a.get("monster_caution_reason", ""),
        "leg2_market": b.get("market", ""),
        "leg2_selection": b.get("selection", ""),
        "leg2_rank": pd.to_numeric(b.get("rank"), errors="coerce"),
        "leg2_score": pd.to_numeric(b.get("slip_leg_score"), errors="coerce"),
        "leg2_monster_score": pd.to_numeric(b.get("monster_candidate_score"), errors="coerce"),
        "leg2_caution_flag": pd.to_numeric(b.get("monster_caution_flag", 0), errors="coerce"),
        "leg2_caution_reason": b.get("monster_caution_reason", ""),
        "pair_score": pair_score(a, b),
        "pair_odds_product": float(pd.to_numeric(a.get("odds"), errors="coerce") or 1.0) * float(pd.to_numeric(b.get("odds"), errors="coerce") or 1.0),
        "pair_monster_caution_count": caution_count,
        "pair_monster_avoid_count": avoid_count,
        "both_graded": both_graded,
        "both_hit": both_hit,
        "landed_legs": landed_legs,
    }


def load_boards(audit_dir: Path) -> list[pd.DataFrame]:
    out = []
    for path in sorted(audit_dir.glob("w*__RANKED_BOARD_SCORED.csv")):
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            continue
        out.append(df)
    return out


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    boards = load_boards(audit_dir)
    if not boards:
        raise SystemExit(f"No ranked board files found under {audit_dir}")

    detail_rows: list[dict] = []
    best_rows: list[pd.DataFrame] = []

    for board in boards:
        window_id = str(board.get("window_id", pd.Series([""])).iloc[0])
        if not window_id:
            continue
        for fixture_key, grp in board.groupby("fixture_key", dropna=False):
            if pd.isna(fixture_key) or len(grp) < 2:
                continue
            grp = grp.sort_values(["rank", "monster_candidate_score"], ascending=[True, False])
            pair_rows = []
            recs = list(grp.to_dict("records"))
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    a = pd.Series(recs[i])
                    b = pd.Series(recs[j])
                    if _norm_market(a.get("market")) == _norm_market(b.get("market")):
                        continue
                    pair_meta = classify_pair(a, b)
                    if pair_meta is None:
                        continue
                    pair_family, pair_label = pair_meta
                    pair_rows.append(pair_detail_row(window_id, a, b, pair_family, pair_label))
            if not pair_rows:
                continue
            pair_df = pd.DataFrame(pair_rows).sort_values(["pair_family", "pair_score"], ascending=[True, False])
            detail_rows.extend(pair_df.to_dict("records"))
            best = pair_df.groupby("pair_family", dropna=False).head(args.top_pairs_per_window)
            best_rows.append(best)

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise SystemExit("No supported same-fixture composite pairs found in ranked boards.")
    best_per_window = pd.concat(best_rows, ignore_index=True) if best_rows else detail.copy()

    detail_path = outdir / "SAME_FIXTURE_COMPOSITES__DETAIL.csv"
    best_path = outdir / "SAME_FIXTURE_COMPOSITES__BEST_PER_WINDOW.csv"
    summary_path = outdir / "SAME_FIXTURE_COMPOSITES__SUMMARY.csv"
    timing_path = outdir / "SAME_FIXTURE_COMPOSITES__TIMING.csv"
    compare_path = outdir / "SAME_FIXTURE_COMPOSITES__VS_MONSTER_BASELINE.csv"

    detail.to_csv(detail_path, index=False)
    best_per_window.to_csv(best_path, index=False)

    summary = (
        detail.groupby(["pair_family", "pair_label"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            windows=("window_id", "nunique"),
            both_graded_rate=("both_graded", "mean"),
            pair_hit_rate=("both_hit", "mean"),
            mean_landed_legs=("landed_legs", "mean"),
            mean_pair_score=("pair_score", "mean"),
            mean_pair_odds_product=("pair_odds_product", "mean"),
            mean_caution_count=("pair_monster_caution_count", "mean"),
            any_caution_rate=("pair_monster_caution_count", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).mean()),
        )
        .reset_index()
        .sort_values(["pair_hit_rate", "rows"], ascending=[False, False])
    )
    summary.to_csv(summary_path, index=False)

    timing = (
        detail.groupby(["pair_family", "season_phase", "window_month", "champions_league_phase_flag", "international_break_zone_flag"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            pair_hit_rate=("both_hit", "mean"),
            mean_landed_legs=("landed_legs", "mean"),
            mean_pair_score=("pair_score", "mean"),
            mean_caution_count=("pair_monster_caution_count", "mean"),
        )
        .reset_index()
        .sort_values(["rows", "pair_hit_rate"], ascending=[False, False])
    )
    timing.to_csv(timing_path, index=False)

    baseline = pd.DataFrame()
    survival_path = audit_dir / "SLIP_WALKFORWARD__SURVIVAL_BY_SIZE.csv"
    if survival_path.exists():
        survival = pd.read_csv(survival_path)
        baseline = survival[survival["build_mode"].astype(str).eq("constructed")].copy()
        if not baseline.empty:
            baseline["monster_mean_leg_hit_rate_when_built"] = baseline["mean_legs_landed_when_built"] / baseline["slip_size"]
            baseline = baseline[[
                "slip_size",
                "complete_slip_rate_when_built",
                "monster_mean_leg_hit_rate_when_built",
            ]]
    if baseline.empty:
        compare = summary.copy()
        compare["baseline_slip_size"] = pd.NA
        compare["monster_complete_slip_rate_when_built"] = pd.NA
        compare["monster_mean_leg_hit_rate_when_built"] = pd.NA
    else:
        compare_rows = []
        for _, srow in summary.iterrows():
            for _, brow in baseline.iterrows():
                compare_rows.append({
                    **srow.to_dict(),
                    "baseline_slip_size": int(brow["slip_size"]),
                    "monster_complete_slip_rate_when_built": brow["complete_slip_rate_when_built"],
                    "monster_mean_leg_hit_rate_when_built": brow["monster_mean_leg_hit_rate_when_built"],
                    "pair_hit_minus_monster_complete": srow["pair_hit_rate"] - brow["complete_slip_rate_when_built"],
                    "pair_hit_minus_monster_leg_hit": srow["pair_hit_rate"] - brow["monster_mean_leg_hit_rate_when_built"],
                })
        compare = pd.DataFrame(compare_rows)
    compare.to_csv(compare_path, index=False)

    print("WROTE:")
    print(detail_path)
    print(best_path)
    print(summary_path)
    print(timing_path)
    print(compare_path)
    print("\nTOP PAIR FAMILIES\n")
    print(summary.head(20).to_string(index=False))
    print("\nTIMING CLUSTERS\n")
    print(timing.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
