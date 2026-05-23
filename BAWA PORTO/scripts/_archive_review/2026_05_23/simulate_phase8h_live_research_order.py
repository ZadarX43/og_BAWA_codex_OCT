#!/usr/bin/env python3
"""Simulate the Phase 8H live research restore order.

Research-only shadow simulator. It applies executable C4 ring stages for:

1. OU25_RESTORE_NOW_SHADOW
2. OU25_RESTORE_WITH_CONFIRM_SHADOW
3. BTTS_RESTORE_NOW_SHADOW
4. BTTS_RESTORE_WITH_CONFIRM_SHADOW

Later stages are emitted as research backlog entries:

5. BTTS_BELGIUM_RECOVERY_RESEARCH
6. FTR_C5_SIDE_SHAPE_RESEARCH
7. TEAM_GOAL_COMBO_RESEARCH

No production rulebook or live deploy files are modified.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROW_LEVEL = Path(
    "reports/2026-05-06/phase8h_full_estate_c4_sweeps/phase8h_replay_row_level_scored.csv"
)
DEFAULT_POLICY = Path(
    "reports/2026-05-06/phase8h_c4_recovery_rings/phase8h_c4_recommended_ring_policy_by_league.csv"
)
DEFAULT_FTR_PAIR_GRID = Path(
    "reports/2026-05-06/phase8h_full_estate_c4_sweeps/phase8h_replay_pair_gate_grid_best.csv"
)
DEFAULT_COMBO_SUMMARY = Path(
    "predictions_output/walk_forward_phase8h_value_layer_full_relock_2026_04_21_r3/_MASTER/FTR_COMBO_MASTER_AUDITS/FTR_COMBO__AUDIT_BUCKET_SUMMARY.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/phase8h_live_research_order_shadow")

STAGE_ORDER = [
    "OU25_RESTORE_NOW_SHADOW",
    "OU25_RESTORE_WITH_CONFIRM_SHADOW",
    "BTTS_RESTORE_NOW_SHADOW",
    "BTTS_RESTORE_WITH_CONFIRM_SHADOW",
]

PHASE8E_THRESHOLDS = {
    ("btts", "England FA Cup"): ("p_meta_btts", 0.80),
    ("btts", "Europa Conference"): ("p_meta_btts", 0.88),
    ("btts", "Champions League"): ("p_meta_btts", 0.80),
    ("btts", "Brazil Serie A"): ("p_meta_btts", 0.85),
    ("btts", "Italy Serie A"): ("p_meta_btts", 0.85),
    ("btts", "Spain La Liga"): ("p_meta_btts", 0.80),
    ("btts", "France Ligue 1"): ("p_meta_btts", 0.88),
    ("btts", "Scotland Premiership"): ("p_meta_btts", 0.80),
    ("btts", "Belgium Pro"): ("p_meta_btts", 0.88),
    ("btts", "Norway Eliteserien"): ("p_meta_btts", 0.80),
    ("btts", "Netherlands Eredivisie"): ("p_meta_btts", 0.88),
    ("btts", "Japan J1"): ("p_meta_btts", 0.80),
    ("btts", "USA MLS"): ("p_meta_btts", 0.85),
    ("ou25", "England FA Cup"): ("p_meta_ou25", 0.80),
    ("ou25", "Europa Conference"): ("p_meta_ou25", 0.80),
    ("ou25", "Germany Bundesliga"): ("p_meta_ou25", 0.88),
    ("ou25", "Europa League"): ("p_meta_ou25", 0.85),
    ("ou25", "Brazil Serie A"): ("p_meta_ou25", 0.80),
    ("ou25", "Champions League"): ("p_meta_ou25", 0.80),
    ("ou25", "Portugal Liga"): ("p_meta_ou25", 0.90),
    ("ou25", "Netherlands Eredivisie"): ("p_meta_ou25", 0.88),
    ("ou25", "Scotland Premiership"): ("p_meta_ou25", 0.90),
    ("ou25", "Spain La Liga"): ("p_meta_ou25", 0.80),
    ("ou25", "Japan J1"): ("p_meta_ou25", 0.80),
    ("ou25", "Norway Eliteserien"): ("p_meta_ou25", 0.85),
    ("ou25", "USA MLS"): ("p_meta_ou25", 0.80),
}


def num(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_policy_source(source: str, market: str, league: str) -> tuple[str, str, float] | None:
    if source == "phase8e_threshold":
        threshold = PHASE8E_THRESHOLDS.get((market, league))
        if not threshold:
            return None
        feature, value = threshold
        return feature, ">=", value
    match = re.match(r"^([A-Za-z0-9_]+)\s*(>=|<=)\s*([-+]?[0-9]*\.?[0-9]+)$", str(source).strip())
    if not match:
        return None
    return match.group(1), match.group(2), float(match.group(3))


def apply_condition(df: pd.DataFrame, feature: str, op: str, threshold: float) -> pd.Series:
    if feature not in df.columns:
        return pd.Series(False, index=df.index)
    values = num(df[feature])
    if op == "<=":
        return values.le(threshold).fillna(False)
    return values.ge(threshold).fillna(False)


def stage_name(market: str, recovery_ring: str) -> str:
    if market == "ou25" and recovery_ring == "RESTORE_NOW":
        return "OU25_RESTORE_NOW_SHADOW"
    if market == "ou25" and recovery_ring == "RESTORE_WITH_CONFIRM":
        return "OU25_RESTORE_WITH_CONFIRM_SHADOW"
    if market == "btts" and recovery_ring == "RESTORE_NOW":
        return "BTTS_RESTORE_NOW_SHADOW"
    if market == "btts" and recovery_ring == "RESTORE_WITH_CONFIRM":
        return "BTTS_RESTORE_WITH_CONFIRM_SHADOW"
    return ""


def apply_restore_stages(rows: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for _, rule in policy.iterrows():
        market = str(rule["market"])
        league = str(rule["league"])
        ring = str(rule["recovery_ring"])
        stage = stage_name(market, ring)
        if not stage:
            continue
        parsed = parse_policy_source(str(rule["policy_source"]), market, league)
        if parsed is None:
            continue
        feature, op, threshold = parsed
        selection = "YES" if market == "btts" else "OVER25"
        pick = rows.get("selection", rows.get("bookie_pick", "")).astype("string").str.upper()
        mask = (
            rows["market_norm"].astype("string").eq(market)
            & rows["league"].astype("string").eq(league)
            & pick.eq(selection)
            & apply_condition(rows, feature, op, threshold)
        )
        part = rows.loc[mask].copy()
        if part.empty:
            continue
        part["research_stage"] = stage
        part["recovery_ring"] = ring
        part["policy_source"] = rule["policy_source"]
        part["policy_feature"] = feature
        part["policy_op"] = op
        part["policy_threshold"] = threshold
        selected.append(part)
    if not selected:
        return pd.DataFrame()

    out = pd.concat(selected, ignore_index=True)
    out["dedupe_key"] = (
        out["league"].astype("string")
        + "||"
        + out["fixture_key"].astype("string")
        + "||"
        + out["market_norm"].astype("string")
        + "||"
        + out.get("selection", out.get("bookie_pick", "")).astype("string")
    )
    out["stage_rank"] = out["research_stage"].map({stage: idx for idx, stage in enumerate(STAGE_ORDER, start=1)})
    return out.sort_values(["dedupe_key", "stage_rank"]).drop_duplicates("dedupe_key", keep="first")


def scorecard(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        hit = num(group["correct"])
        graded = int(hit.notna().sum())
        wins = float((hit == 1).sum())
        odds = num(group.get("bookie_od", pd.Series(dtype=float)))
        profit = np.where(hit == 1, odds - 1.0, np.where(hit == 0, -1.0, np.nan))
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": int((hit == 0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "profit": float(np.nansum(profit)),
                "roi": float(np.nansum(profit) / graded) if graded else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def cumulative_scorecard(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rank, stage in enumerate(STAGE_ORDER, start=1):
        sub = selected[selected["stage_rank"].le(rank)]
        if sub.empty:
            continue
        row = scorecard(sub, [])[0:0] if False else {}
        hit = num(sub["correct"])
        graded = int(hit.notna().sum())
        wins = float((hit == 1).sum())
        odds = num(sub.get("bookie_od", pd.Series(dtype=float)))
        profit = np.where(hit == 1, odds - 1.0, np.where(hit == 0, -1.0, np.nan))
        row.update(
            {
                "through_stage": stage,
                "stage_rank": rank,
                "rows": int(len(sub)),
                "graded": graded,
                "wins": wins,
                "losses": int((hit == 0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "profit": float(np.nansum(profit)),
                "roi": float(np.nansum(profit) / graded) if graded else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def belgium_backlog(rows: pd.DataFrame) -> dict[str, Any]:
    g = rows[rows["market_norm"].astype("string").eq("btts") & rows["league"].astype("string").eq("Belgium Pro")]
    yes = g[g.get("selection", g.get("bookie_pick", "")).astype("string").str.upper().eq("YES")]
    no = g[g.get("selection", g.get("bookie_pick", "")).astype("string").str.upper().eq("NO")]
    hit_yes = num(yes["correct"]) if not yes.empty else pd.Series(dtype=float)
    hit_no = num(no["correct"]) if not no.empty else pd.Series(dtype=float)
    return {
        "research_stage": "BTTS_BELGIUM_RECOVERY_RESEARCH",
        "status": "BLOCKED_RESEARCH",
        "evidence": "Current Belgium BTTS estate is mostly BTTS NO in OBSERVE; old BTTS YES p_meta map has no qualifying rows.",
        "rows": int(len(g)),
        "yes_rows": int(len(yes)),
        "yes_graded": int(hit_yes.notna().sum()),
        "yes_hit_rate": float((hit_yes == 1).sum() / hit_yes.notna().sum()) if hit_yes.notna().sum() else np.nan,
        "no_rows": int(len(no)),
        "no_graded": int(hit_no.notna().sum()),
        "no_hit_rate": float((hit_no == 1).sum() / hit_no.notna().sum()) if hit_no.notna().sum() else np.nan,
    }


def ftr_backlog(pair_grid_path: Path) -> dict[str, Any]:
    if not pair_grid_path.exists():
        return {
            "research_stage": "FTR_C5_SIDE_SHAPE_RESEARCH",
            "status": "MISSING_INPUT",
            "evidence": "Pair gate grid not found.",
        }
    grid = pd.read_csv(pair_grid_path)
    ftr = grid[(grid["market"].eq("ftr")) & (grid["graded"].ge(20))].copy()
    if ftr.empty:
        return {
            "research_stage": "FTR_C5_SIDE_SHAPE_RESEARCH",
            "status": "NEEDS_SWEEP",
            "evidence": "No FTR pair gates with minimum sample.",
        }
    perfect = ftr[ftr["hit_rate"].ge(1.0)]
    return {
        "research_stage": "FTR_C5_SIDE_SHAPE_RESEARCH",
        "status": "HAS_CANDIDATES_NEEDS_C5",
        "evidence": "Full-estate pair sweeps show many FTR side-shape candidates; needs dedicated C5 ring classifier and window stability.",
        "candidate_rows": int(len(ftr)),
        "perfect_candidate_rows": int(len(perfect)),
        "best_graded": int(ftr.sort_values(["hit_rate", "graded"], ascending=[False, False]).iloc[0]["graded"]),
        "best_hit_rate": float(ftr["hit_rate"].max()),
    }


def combo_backlog(combo_path: Path) -> dict[str, Any]:
    if not combo_path.exists():
        return {
            "research_stage": "TEAM_GOAL_COMBO_RESEARCH",
            "status": "MISSING_INPUT",
            "evidence": "Combo summary not found.",
        }
    combo = pd.read_csv(combo_path)
    strong = combo[combo["audit_bucket"].astype("string").str.contains("VERY_STRONG", na=False)].copy()
    return {
        "research_stage": "TEAM_GOAL_COMBO_RESEARCH",
        "status": "HAS_SIGNAL_KEEP_SEPARATE",
        "evidence": "Very-strong home/away GE2 buckets clear strong hit rates, but broad combo markets are weak and must stay separate from FTR.",
        "very_strong_buckets": int(len(strong)),
        "very_strong_graded": int(strong["graded_rows"].sum()) if not strong.empty else 0,
        "very_strong_wins": float(strong["wins"].sum()) if not strong.empty else 0.0,
        "very_strong_hit_rate": float(strong["wins"].sum() / strong["graded_rows"].sum())
        if not strong.empty and strong["graded_rows"].sum()
        else np.nan,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        else:
            show[col] = show[col].astype("string").fillna("")
    headers = list(show.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def write_summary(outdir: Path, stage: pd.DataFrame, cumulative: pd.DataFrame, league: pd.DataFrame, backlog: pd.DataFrame) -> None:
    lines = [
        "# Phase 8H Live Research Order Shadow",
        "",
        "Research-only stage simulation. No production policy files changed.",
        "",
        "## Incremental Stage Scorecard",
        markdown_table(stage),
        "",
        "## Cumulative Scorecard",
        markdown_table(cumulative),
        "",
        "## League Scorecard",
        markdown_table(league),
        "",
        "## Research Backlog",
        markdown_table(backlog),
        "",
        "## Deployment Read",
        "",
        "- Stages 1-4 are executable as shadow selections from the C4 ring policy.",
        "- Stages 5-7 are not deployment stages yet; they need dedicated research proof.",
        "- Keep value edge additive only after these gates.",
        "- Do not edit `deploy_rulebook.py` until this shadow order passes QA and live-output simulation.",
        "",
    ]
    (outdir / "phase8h_live_research_order_shadow_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-level", type=Path, default=DEFAULT_ROW_LEVEL)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--ftr-pair-grid", type=Path, default=DEFAULT_FTR_PAIR_GRID)
    parser.add_argument("--combo-summary", type=Path, default=DEFAULT_COMBO_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.row_level, low_memory=False)
    policy = pd.read_csv(args.policy)
    selected = apply_restore_stages(rows, policy)

    selected.to_csv(args.outdir / "phase8h_live_shadow_selected_rows.csv", index=False)
    stage_card = scorecard(selected, ["research_stage", "market_norm", "recovery_ring"])
    stage_card["stage_rank"] = stage_card["research_stage"].map({stage: idx for idx, stage in enumerate(STAGE_ORDER, start=1)})
    stage_card = stage_card.sort_values("stage_rank")
    stage_card.to_csv(args.outdir / "phase8h_live_shadow_stage_scorecard.csv", index=False)

    cumulative = cumulative_scorecard(selected)
    cumulative.to_csv(args.outdir / "phase8h_live_shadow_cumulative_scorecard.csv", index=False)

    league = scorecard(selected, ["research_stage", "market_norm", "league", "recovery_ring"])
    league["stage_rank"] = league["research_stage"].map({stage: idx for idx, stage in enumerate(STAGE_ORDER, start=1)})
    league = league.sort_values(["stage_rank", "market_norm", "league"])
    league.to_csv(args.outdir / "phase8h_live_shadow_league_scorecard.csv", index=False)

    backlog = pd.DataFrame(
        [
            belgium_backlog(rows),
            ftr_backlog(args.ftr_pair_grid),
            combo_backlog(args.combo_summary),
        ]
    )
    backlog.to_csv(args.outdir / "phase8h_live_shadow_research_backlog.csv", index=False)

    write_summary(args.outdir, stage_card, cumulative, league, backlog)
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

