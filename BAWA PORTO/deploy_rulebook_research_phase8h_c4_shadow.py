#!/usr/bin/env python3
"""Research-only Phase 8H C4 shadow restore annotator.

This script is deliberately not a production deploy rulebook. It applies the
Phase 8H C4 ring policy as sidecar shadow labels to scored/deploy-shaped rows.

Safety properties:
  - never changes deploy_tier/tier
  - never promotes OBSERVE rows
  - never uses value_edge as a gate
  - only emits shadow columns and audit files
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_POLICY = Path(
    "reports/2026-05-06/phase8h_c4_recovery_rings/phase8h_c4_recommended_ring_policy_by_league.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/phase8h_research_deploy_shadow_simulator")

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


def _num(values: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _string_col(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].astype("string").fillna(default)
    return pd.Series(default, index=df.index, dtype="string")


def _market_norm(df: pd.DataFrame) -> pd.Series:
    if "market_norm" in df.columns:
        raw = _string_col(df, "market_norm")
    else:
        raw = _string_col(df, "market")
    return raw.str.lower().str.strip()


def _selection_norm(df: pd.DataFrame) -> pd.Series:
    if "selection" in df.columns:
        raw = _string_col(df, "selection")
    else:
        raw = _string_col(df, "bookie_pick")
    return raw.str.upper().str.strip()


def _stage_name(market: str, recovery_ring: str) -> str:
    if market == "ou25" and recovery_ring == "RESTORE_NOW":
        return "OU25_RESTORE_NOW_SHADOW"
    if market == "ou25" and recovery_ring == "RESTORE_WITH_CONFIRM":
        return "OU25_RESTORE_WITH_CONFIRM_SHADOW"
    if market == "btts" and recovery_ring == "RESTORE_NOW":
        return "BTTS_RESTORE_NOW_SHADOW"
    if market == "btts" and recovery_ring == "RESTORE_WITH_CONFIRM":
        return "BTTS_RESTORE_WITH_CONFIRM_SHADOW"
    return ""


def _parse_policy_source(source: str, market: str, league: str) -> tuple[str, str, float] | None:
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


def _condition(df: pd.DataFrame, feature: str, op: str, threshold: float) -> pd.Series:
    if feature not in df.columns:
        return pd.Series(False, index=df.index)
    values = _num(df[feature])
    if op == "<=":
        return values.le(threshold).fillna(False)
    return values.ge(threshold).fillna(False)


def _append_token(series: pd.Series, mask: pd.Series, token: str) -> pd.Series:
    out = series.astype("string").fillna("")
    m = mask.reindex(out.index, fill_value=False).astype(bool)
    if not bool(m.any()):
        return out

    def add_one(value: Any) -> str:
        existing = [x.strip() for x in str(value or "").split("|") if x.strip()]
        if token not in existing:
            existing.append(token)
        return "|".join(existing)

    out.loc[m] = out.loc[m].map(add_one)
    return out


def _dedupe_key(df: pd.DataFrame) -> pd.Series:
    league = _string_col(df, "league").str.strip()
    fixture = _string_col(df, "fixture_key").str.strip()
    market = _market_norm(df)
    pick = _selection_norm(df)
    if fixture.eq("").all():
        fixture = (
            _string_col(df, "match_date").str.strip()
            + "||"
            + _string_col(df, "home_team_name").str.strip()
            + "||"
            + _string_col(df, "away_team_name").str.strip()
        )
    return league + "||" + fixture + "||" + market + "||" + pick


def apply_shadow_policy(df: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_phase8h_c4_market_norm"] = _market_norm(out)
    out["_phase8h_c4_selection_norm"] = _selection_norm(out)
    out["_phase8h_c4_dedupe_key"] = _dedupe_key(out)

    out["phase8h_c4_shadow_pass"] = 0
    out["phase8h_c4_shadow_stage"] = ""
    out["phase8h_c4_shadow_stage_rank"] = np.nan
    out["phase8h_c4_shadow_recovery_ring"] = ""
    out["phase8h_c4_shadow_policy_source"] = ""
    out["phase8h_c4_shadow_policy_feature"] = ""
    out["phase8h_c4_shadow_policy_op"] = ""
    out["phase8h_c4_shadow_policy_threshold"] = np.nan
    out["phase8h_c4_shadow_candidate_tier"] = ""
    out["phase8h_c4_shadow_reason_codes"] = ""

    stage_rank = {stage: idx for idx, stage in enumerate(STAGE_ORDER, start=1)}
    league = _string_col(out, "league").str.strip()

    for _, rule in policy.iterrows():
        market = str(rule.get("market", "")).strip().lower()
        lg = str(rule.get("league", "")).strip()
        ring = str(rule.get("recovery_ring", "")).strip().upper()
        source = str(rule.get("policy_source", "")).strip()
        stage = _stage_name(market, ring)
        if not stage:
            continue
        parsed = _parse_policy_source(source, market, lg)
        if parsed is None:
            continue
        feature, op, threshold = parsed
        target_pick = "YES" if market == "btts" else "OVER25"
        mask = (
            out["_phase8h_c4_market_norm"].eq(market)
            & league.eq(lg)
            & out["_phase8h_c4_selection_norm"].eq(target_pick)
            & _condition(out, feature, op, threshold)
        )
        if not bool(mask.any()):
            continue

        token = f"PHASE8H_C4_{stage}"
        out.loc[mask, "phase8h_c4_shadow_pass"] = 1
        out.loc[mask, "phase8h_c4_shadow_stage"] = stage
        out.loc[mask, "phase8h_c4_shadow_stage_rank"] = stage_rank[stage]
        out.loc[mask, "phase8h_c4_shadow_recovery_ring"] = ring
        out.loc[mask, "phase8h_c4_shadow_policy_source"] = source
        out.loc[mask, "phase8h_c4_shadow_policy_feature"] = feature
        out.loc[mask, "phase8h_c4_shadow_policy_op"] = op
        out.loc[mask, "phase8h_c4_shadow_policy_threshold"] = threshold
        out.loc[mask, "phase8h_c4_shadow_candidate_tier"] = "SHADOW_ONLY"
        out["phase8h_c4_shadow_reason_codes"] = _append_token(
            out["phase8h_c4_shadow_reason_codes"],
            mask,
            token,
        )

    return out


def selected_rows(annotated: pd.DataFrame) -> pd.DataFrame:
    sel = annotated[annotated["phase8h_c4_shadow_pass"].eq(1)].copy()
    if sel.empty:
        return sel
    return (
        sel.sort_values(["_phase8h_c4_dedupe_key", "phase8h_c4_shadow_stage_rank"])
        .drop_duplicates("_phase8h_c4_dedupe_key", keep="first")
        .copy()
    )


def scorecard(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, group in grouped:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        hit = _num(group["correct"]) if "correct" in group.columns else pd.Series(np.nan, index=group.index)
        graded = int(hit.notna().sum())
        wins = float((hit == 1).sum())
        odds = _num(group["bookie_od"]) if "bookie_od" in group.columns else pd.Series(np.nan, index=group.index)
        profit = np.where(hit == 1, odds - 1.0, np.where(hit == 0, -1.0, np.nan))
        row = dict(zip(group_cols, keys, strict=False)) if group_cols else {}
        row.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": int((hit == 0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "profit": float(np.nansum(profit)) if graded else np.nan,
                "roi": float(np.nansum(profit) / graded) if graded else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def cumulative_scorecard(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rank, stage in enumerate(STAGE_ORDER, start=1):
        sub = sel[_num(sel["phase8h_c4_shadow_stage_rank"]).le(rank)]
        if sub.empty:
            continue
        card = scorecard(sub, [])
        row = card.iloc[0].to_dict()
        row["through_stage"] = stage
        row["stage_rank"] = rank
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    cols = ["through_stage", "stage_rank", "rows", "graded", "wins", "losses", "hit_rate", "profit", "roi"]
    return pd.DataFrame(rows)[cols]


def tier_crosstab(sel: pd.DataFrame) -> pd.DataFrame:
    if sel.empty:
        return pd.DataFrame()
    tier = _string_col(sel, "deploy_tier", "UNKNOWN").str.upper().str.strip()
    tier = tier.mask(tier.eq(""), "UNKNOWN")
    tmp = sel.assign(source_deploy_tier=tier)
    return (
        tmp.groupby(["phase8h_c4_shadow_stage", "source_deploy_tier"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["phase8h_c4_shadow_stage", "source_deploy_tier"])
    )


def compare_selected(sel: pd.DataFrame, compare_path: Path) -> pd.DataFrame:
    other = pd.read_csv(compare_path)
    other_key = (
        other["_phase8h_c4_dedupe_key"]
        if "_phase8h_c4_dedupe_key" in other.columns
        else _dedupe_key(other)
    )
    current = set(sel["_phase8h_c4_dedupe_key"].astype("string"))
    expected = set(other_key.astype("string"))
    return pd.DataFrame(
        [
            {
                "current_selected": len(current),
                "compare_selected": len(expected),
                "intersection": len(current & expected),
                "current_only": len(current - expected),
                "compare_only": len(expected - current),
                "match": current == expected,
            }
        ]
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")

    headers = [str(c) for c in text.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text.iterrows():
        values = [str(row[c]) for c in text.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    *,
    path: Path,
    src: Path,
    selected_path: Path,
    annotated_path: Path | None,
    stage_card: pd.DataFrame,
    cumulative_card: pd.DataFrame,
    tier_table: pd.DataFrame,
    compare_card: pd.DataFrame | None,
) -> None:
    lines = [
        "# Phase 8H C4 Research Deploy Shadow",
        "",
        "Research-only sidecar output. No production policy files or deploy tiers were changed.",
        "",
        f"- Source: `{src}`",
        f"- Selected rows: `{selected_path}`",
    ]
    if annotated_path is not None:
        lines.append(f"- Annotated rows: `{annotated_path}`")
    lines.extend(
        [
            "",
            "## Incremental Stage Scorecard",
            markdown_table(stage_card) if not stage_card.empty else "_No selected rows._",
            "",
            "## Cumulative Scorecard",
            markdown_table(cumulative_card) if not cumulative_card.empty else "_No selected rows._",
            "",
            "## Source Tier Cross-Tab",
            markdown_table(tier_table) if not tier_table.empty else "_No deploy_tier column or no selected rows._",
        ]
    )
    if compare_card is not None:
        lines.extend(["", "## Selection Comparison", markdown_table(compare_card)])
    lines.extend(
        [
            "",
            "## Safety Read",
            "",
            "- `deploy_tier` and `tier` are preserved exactly as source columns.",
            "- Shadow candidates are marked with `phase8h_c4_shadow_candidate_tier=SHADOW_ONLY`.",
            "- Value edge is not used as a restore condition.",
            "- Belgium BTTS, FTR C5, and team-goal combos are outside this executable shadow gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Scored/deploy-shaped CSV to shadow annotate.")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="C4 recommended ring policy CSV.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Directory for shadow outputs.")
    parser.add_argument("--prefix", default="", help="Optional output filename prefix.")
    parser.add_argument("--compare-selected", default="", help="Optional previous selected CSV for key comparison.")
    parser.add_argument("--no-annotated", action="store_true", help="Skip full annotated-row CSV.")
    args = parser.parse_args()

    src = Path(args.src)
    policy_path = Path(args.policy)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    policy = pd.read_csv(policy_path)
    annotated = apply_shadow_policy(df, policy)
    sel = selected_rows(annotated)

    prefix = args.prefix.strip() or src.stem
    selected_path = outdir / f"{prefix}__PHASE8H_C4_SHADOW_SELECTED.csv"
    annotated_path = None if args.no_annotated else outdir / f"{prefix}__PHASE8H_C4_SHADOW_ANNOTATED.csv"
    stage_card_path = outdir / f"{prefix}__PHASE8H_C4_SHADOW_STAGE_SCORECARD.csv"
    cumulative_card_path = outdir / f"{prefix}__PHASE8H_C4_SHADOW_CUMULATIVE_SCORECARD.csv"
    league_card_path = outdir / f"{prefix}__PHASE8H_C4_SHADOW_LEAGUE_SCORECARD.csv"
    tier_table_path = outdir / f"{prefix}__PHASE8H_C4_SHADOW_SOURCE_TIER_CROSSTAB.csv"
    summary_path = outdir / f"{prefix}__PHASE8H_C4_SHADOW_SUMMARY.md"

    sel.to_csv(selected_path, index=False)
    if annotated_path is not None:
        annotated.to_csv(annotated_path, index=False)

    stage_card = scorecard(
        sel,
        ["phase8h_c4_shadow_stage", "_phase8h_c4_market_norm", "phase8h_c4_shadow_recovery_ring"],
    ).sort_values("phase8h_c4_shadow_stage")
    if not stage_card.empty:
        stage_card["stage_rank"] = stage_card["phase8h_c4_shadow_stage"].map(
            {stage: idx for idx, stage in enumerate(STAGE_ORDER, start=1)}
        )
        stage_card = stage_card.sort_values("stage_rank")
    cumulative_card = cumulative_scorecard(sel)
    league_card = scorecard(
        sel,
        ["phase8h_c4_shadow_stage", "_phase8h_c4_market_norm", "league", "phase8h_c4_shadow_recovery_ring"],
    )
    if not league_card.empty:
        league_card["stage_rank"] = league_card["phase8h_c4_shadow_stage"].map(
            {stage: idx for idx, stage in enumerate(STAGE_ORDER, start=1)}
        )
        league_card = league_card.sort_values(["stage_rank", "league"])
    tiers = tier_crosstab(sel)

    stage_card.to_csv(stage_card_path, index=False)
    cumulative_card.to_csv(cumulative_card_path, index=False)
    league_card.to_csv(league_card_path, index=False)
    tiers.to_csv(tier_table_path, index=False)

    compare_card = None
    if args.compare_selected:
        compare_card = compare_selected(sel, Path(args.compare_selected))
        compare_card.to_csv(outdir / f"{prefix}__PHASE8H_C4_SHADOW_COMPARE_SELECTED.csv", index=False)

    write_summary(
        path=summary_path,
        src=src,
        selected_path=selected_path,
        annotated_path=annotated_path,
        stage_card=stage_card,
        cumulative_card=cumulative_card,
        tier_table=tiers,
        compare_card=compare_card,
    )

    print(f"[ok] wrote {selected_path}")
    print(f"[ok] wrote {summary_path}")


if __name__ == "__main__":
    main()
