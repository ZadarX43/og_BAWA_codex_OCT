#!/usr/bin/env python3
"""Reusable helpers for the Correct Score deploy and backtest product layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ALIGN_SUPPORTIVE = {"STRONG_ALIGN", "SOFT_ALIGN", "ALIGNED", "SUPPORTED"}
ALIGN_NEGATIVE = {"CONFLICT", "FRAGILE"}
PREMIUM_LEAGUE_ALLOWLIST = {
    "SPAIN LA LIGA",
    "SCOTLAND PREMIERSHIP",
    "ENGLAND PREMIER LEAGUE",
    "USA MLS",
    "JAPAN J1",
}
COMMON_PREMIUM_SCORELINES = {"1-1", "0-1", "1-0", "0-0", "2-0", "2-1"}
ALIGN_ORDER = {
    "STRONG_ALIGN": 4,
    "SOFT_ALIGN": 3,
    "ALIGNED": 3,
    "SUPPORTED": 3,
    "NEUTRAL": 2,
    "": 1,
    "NONE": 1,
    "FRAGILE": 0,
    "CONFLICT": -1,
}
TIER_ORDER = {"ELITE": 3, "STANDARD": 2, "OBSERVE": 1, "": 0}
MARKET_PREF = {"ftr": 3, "btts": 2, "ou25": 1}
CS_CLASS_ORDER = ["CS_BANKER", "CS_ELITE", "CS_RESCUE", "CS_WATCH"]
PREMIUM_RULE_ORDER = ["PRO_CS_TRIPLE_ALIGN", "PRO_CS_BANKER_PLUS", "PRO_CS_BANKER", "PRO_CS_ELITE", "PREMIUM_WATCH", "PUBLIC"]


@dataclass
class CSConfig:
    elite_top1_min: float = 0.11
    banker_top1_min: float = 0.15
    rescue_top1_min: float = 0.08
    elite_top3_total_min: float = 0.28
    rescue_top3_total_min: float = 0.24
    max_entropy_for_elite: float = 2.85
    max_entropy_for_banker: float = 2.55


def read_csvs(paths: list[Path], usecols: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if usecols is None:
            df = pd.read_csv(path, low_memory=False)
        else:
            header = pd.read_csv(path, nrows=0)
            keep = [c for c in usecols if c in header.columns]
            df = pd.read_csv(path, low_memory=False, usecols=keep)
        df["_source_path"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _safe_float(value: object) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if pd.notna(out) else np.nan


def normalize_market(value: object) -> str:
    return _safe_str(value).lower()


def normalize_align(value: object) -> str:
    return _safe_str(value).upper()


def normalize_league(value: object) -> str:
    return _safe_str(value).upper()


def scoreline_side(score: object) -> str:
    txt = _safe_str(score)
    if "-" not in txt:
        return ""
    try:
        home, away = txt.split("-", 1)
        hg, ag = int(home), int(away)
    except Exception:
        return ""
    if hg > ag:
        return "HOME"
    if hg < ag:
        return "AWAY"
    return "DRAW"


def scoreline_btts(score: object) -> str:
    txt = _safe_str(score)
    if "-" not in txt:
        return ""
    try:
        home, away = txt.split("-", 1)
        hg, ag = int(home), int(away)
    except Exception:
        return ""
    return "YES" if hg > 0 and ag > 0 else "NO"


def scoreline_ou25(score: object) -> str:
    txt = _safe_str(score)
    if "-" not in txt:
        return ""
    try:
        home, away = txt.split("-", 1)
        hg, ag = int(home), int(away)
    except Exception:
        return ""
    return "OVER" if (hg + ag) >= 3 else "UNDER"


def format_actual_score(home_goals: object, away_goals: object) -> str:
    hg = pd.to_numeric(pd.Series([home_goals]), errors="coerce").iloc[0]
    ag = pd.to_numeric(pd.Series([away_goals]), errors="coerce").iloc[0]
    if pd.isna(hg) or pd.isna(ag):
        return ""
    return f"{int(hg)}-{int(ag)}"


def choose_base_row(group: pd.DataFrame) -> pd.Series:
    work = group.copy()
    work["_tier_rank"] = work.get("deploy_tier", "").map(lambda x: TIER_ORDER.get(_safe_str(x).upper(), 0))
    work["_market_rank"] = work.get("market", "").map(lambda x: MARKET_PREF.get(normalize_market(x), 0))
    work["_cs1_p"] = pd.to_numeric(work.get("cs1_p", np.nan), errors="coerce").fillna(-1.0)
    work = work.sort_values(["_tier_rank", "_market_rank", "_cs1_p"], ascending=[False, False, False])
    return work.iloc[0]


def _pick_token(row: pd.Series) -> str:
    for col in ["bookie_pick", "selection"]:
        token = normalize_align(row.get(col, ""))
        if token:
            return token
    return ""


def _heuristic_alignment(row: pd.Series, market: str) -> tuple[str, int]:
    cs1 = _safe_str(row.get("cs1") or row.get("cs_top1_scoreline"))
    pick = _pick_token(row)
    if not cs1 or not pick:
        return "NONE", 0

    if market == "ftr":
        side = scoreline_side(cs1)
        pick_side = {"HOME": "HOME", "AWAY": "AWAY", "DRAW": "DRAW", "1": "HOME", "2": "AWAY", "X": "DRAW"}.get(pick, pick)
        if side == pick_side:
            return "SOFT_ALIGN", 1
        return "CONFLICT", 0

    if market == "btts":
        btts = scoreline_btts(cs1)
        pick_btts = {"YES": "YES", "NO": "NO"}.get(pick, pick)
        if btts == pick_btts:
            return "SOFT_ALIGN", 1
        return "CONFLICT", 0

    if market == "ou25":
        ou = scoreline_ou25(cs1)
        pick_ou = pick
        if pick_ou in {"OVER25", "OVER_2.5", "OVER 2.5"}:
            pick_ou = "OVER"
        if pick_ou in {"UNDER25", "UNDER_2.5", "UNDER 2.5"}:
            pick_ou = "UNDER"
        if ou == pick_ou:
            return "SOFT_ALIGN", 1
        return "CONFLICT", 0

    return "NONE", 0


def _market_alignment(group: pd.DataFrame, market: str, col: str) -> tuple[str, int]:
    sub = group[group.get("market", "").map(normalize_market).eq(market)].copy()
    if sub.empty:
        return "NONE", 0

    align_scores = []
    for _, row in sub.iterrows():
        align = normalize_align(row.get(col, ""))
        score = ALIGN_ORDER.get(align, 1)
        if market == "ftr":
            if int(pd.to_numeric(pd.Series([row.get("cs_aligns_selection_flag", 0)]), errors="coerce").fillna(0).iloc[0]) == 1:
                score = max(score, 4)
        align_scores.append((score, align))

    align_scores.sort(reverse=True)
    best_score, best_align = align_scores[0]
    if not best_align or best_align == "NONE":
        heuristic_scores = [_heuristic_alignment(row, market) for _, row in sub.iterrows()]
        heuristic_scores.sort(key=lambda x: ALIGN_ORDER.get(x[0], 1), reverse=True)
        if heuristic_scores:
            best_align, support_flag = heuristic_scores[0]
            return best_align or "NONE", int(support_flag)
    support_flag = int(best_align in ALIGN_SUPPORTIVE or best_score >= 3)
    return best_align or "NEUTRAL", support_flag


def classify_fixture(row: pd.Series, cfg: CSConfig) -> str:
    top1 = _safe_float(row.get("cs1_p"))
    top3_total = _safe_float(row.get("cs_top3_total_prob"))
    entropy = _safe_float(row.get("cs_entropy"))
    support_count = int(row.get("support_count", 0) or 0)
    fragility = int(row.get("cs_fragility_flag", 0) or 0)
    diffuse = int(row.get("cs_diffuse_flag", 0) or 0)
    banker = int(row.get("cs_banker_flag", 0) or 0)
    conflict_count = int(row.get("conflict_count", 0) or 0)

    if (
        conflict_count == 0
        and fragility == 0
        and diffuse == 0
        and support_count >= 1
        and (
            banker == 1
            or (pd.notna(top1) and top1 >= cfg.banker_top1_min)
            or (
                _safe_str(row.get("cs_concentration_bucket")).upper() in {"HIGH", "VERY_HIGH"}
                and pd.notna(top1)
                and top1 >= cfg.elite_top1_min
                and pd.notna(entropy)
                and entropy <= cfg.max_entropy_for_banker
            )
        )
    ):
        return "CS_BANKER"

    if (
        conflict_count == 0
        and fragility == 0
        and diffuse == 0
        and support_count >= 1
        and pd.notna(top1)
        and top1 >= cfg.elite_top1_min
        and pd.notna(top3_total)
        and top3_total >= cfg.elite_top3_total_min
        and (pd.isna(entropy) or entropy <= cfg.max_entropy_for_elite)
    ):
        return "CS_ELITE"

    if (
        conflict_count == 0
        and fragility == 0
        and support_count >= 2
        and pd.notna(top1)
        and top1 >= cfg.rescue_top1_min
        and pd.notna(top3_total)
        and top3_total >= cfg.rescue_top3_total_min
    ):
        return "CS_RESCUE"

    return "CS_WATCH"


def build_fixture_level_cs(df: pd.DataFrame, cfg: CSConfig | None = None) -> pd.DataFrame:
    cfg = cfg or CSConfig()
    if df is None or df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    fixture_col = "fixture_key" if "fixture_key" in df.columns else None
    if fixture_col is None:
        raise ValueError("Correct score deploy input is missing fixture_key")

    for fixture_key, group in df.groupby(fixture_col, dropna=False):
        base = choose_base_row(group)
        ftr_alignment, ftr_support = _market_alignment(group, "ftr", "cs_ftr_alignment")
        btts_alignment, btts_support = _market_alignment(group, "btts", "cs_btts_alignment")
        ou25_alignment, ou25_support = _market_alignment(group, "ou25", "cs_ou25_alignment")

        support_count = int(ftr_support + btts_support + ou25_support)
        conflict_count = int(
            (ftr_alignment in ALIGN_NEGATIVE)
            + (btts_alignment in ALIGN_NEGATIVE)
            + (ou25_alignment in ALIGN_NEGATIVE)
        )

        top1 = _safe_float(base.get("cs1_p"))
        top2 = _safe_float(base.get("cs2_p"))
        top3 = _safe_float(base.get("cs3_p"))
        top3_total = np.nansum([top1, top2, top3])
        actual_score = format_actual_score(base.get("home_team_goal_count"), base.get("away_team_goal_count"))
        top1_score = _safe_str(base.get("cs1") or base.get("cs_top1_scoreline"))
        top2_score = _safe_str(base.get("cs2") or base.get("cs_top2_scoreline"))
        top3_score = _safe_str(base.get("cs3") or base.get("cs_top3_scoreline"))

        row = {
            "fixture_key": fixture_key,
            "match_date": base.get("match_date"),
            "league": base.get("league"),
            "home_team_name": base.get("home_team_name"),
            "away_team_name": base.get("away_team_name"),
            "deploy_tier": _safe_str(base.get("deploy_tier")).upper(),
            "base_market": normalize_market(base.get("market")),
            "cs1": top1_score,
            "cs1_p": top1,
            "cs2": top2_score,
            "cs2_p": top2,
            "cs3": top3_score,
            "cs3_p": top3,
            "cs_top3_total_prob": top3_total,
            "cs_entropy": _safe_float(base.get("cs_entropy")),
            "cs_concentration_bucket": _safe_str(base.get("cs_concentration_bucket")).upper(),
            "cs_banker_flag": int(base.get("cs_banker_flag", 0) or 0),
            "cs_banker_bonus": _safe_float(base.get("cs_banker_bonus")),
            "cs_supported_flag": int(base.get("cs_supported_flag", 0) or 0),
            "cs_fragility_flag": int(base.get("cs_fragility_flag", 0) or 0),
            "cs_diffuse_flag": int(base.get("cs_diffuse_flag", 0) or 0),
            "cs_one_sided_flag": int(base.get("cs_one_sided_flag", 0) or 0),
            "cs_draw_family_flag": int(base.get("cs_draw_family_flag", 0) or 0),
            "cs_alignment_count": int(base.get("cs_alignment_count", 0) or 0),
            "ftr_alignment": ftr_alignment,
            "btts_alignment": btts_alignment,
            "ou25_alignment": ou25_alignment,
            "ftr_support_flag": ftr_support,
            "btts_support_flag": btts_support,
            "ou25_support_flag": ou25_support,
            "support_count": support_count,
            "conflict_count": conflict_count,
            "pick_side_mass_top3": _safe_float(base.get("pick_side_mass_top3")),
            "pick_side_margin_top3": _safe_float(base.get("pick_side_margin_top3")),
            "actual_score": actual_score,
            "actual_ftr": _safe_str(base.get("actual_ftr")).upper(),
            "actual_over25": _safe_float(base.get("actual_over25")),
            "actual_btts_yes": _safe_float(base.get("actual_btts_yes")),
            "cs1_ftr_side": scoreline_side(top1_score),
            "cs1_btts_pick": scoreline_btts(top1_score),
            "cs1_ou25_pick": scoreline_ou25(top1_score),
        }
        row["cs_class"] = classify_fixture(pd.Series(row), cfg)
        row["exact_hit_flag"] = int(actual_score != "" and top1_score == actual_score)
        row["top3_hit_flag"] = int(actual_score != "" and actual_score in {top1_score, top2_score, top3_score})
        row["cs1_ftr_hit_flag"] = int(row["actual_ftr"] != "" and row["cs1_ftr_side"] == row["actual_ftr"])
        row["cs1_btts_hit_flag"] = int(pd.notna(row["actual_btts_yes"]) and row["cs1_btts_pick"] == ("YES" if float(row["actual_btts_yes"]) == 1.0 else "NO"))
        row["cs1_ou25_hit_flag"] = int(pd.notna(row["actual_over25"]) and row["cs1_ou25_pick"] == ("OVER" if float(row["actual_over25"]) == 1.0 else "UNDER"))
        row["cross_market_note"] = build_correlation_note(pd.Series(row))
        row["public_signal_strength"] = build_signal_strength(pd.Series(row))
        add_premium_profile(row)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["cs_class"] = pd.Categorical(out["cs_class"], CS_CLASS_ORDER, ordered=True)
    return out.sort_values(["cs_class", "cs1_p", "cs_top3_total_prob"], ascending=[True, False, False]).reset_index(drop=True)


def build_signal_strength(row: pd.Series) -> str:
    top1 = _safe_float(row.get("cs1_p"))
    entropy = _safe_float(row.get("cs_entropy"))
    support_count = int(row.get("support_count", 0) or 0)
    if pd.notna(top1) and top1 >= 0.15 and support_count >= 1 and (pd.isna(entropy) or entropy <= 2.55):
        return "BANKER_STRENGTH"
    if pd.notna(top1) and top1 >= 0.11 and support_count >= 1:
        return "ELITE_STRENGTH"
    if support_count >= 2:
        return "RESCUE_STRENGTH"
    return "WATCH_STRENGTH"


def build_correlation_note(row: pd.Series) -> str:
    notes: list[str] = []
    if int(row.get("ftr_support_flag", 0) or 0) == 1:
        notes.append(f"FTR {row.get('ftr_alignment', 'ALIGN')}")
    if int(row.get("btts_support_flag", 0) or 0) == 1:
        notes.append(f"BTTS {row.get('btts_alignment', 'ALIGN')}")
    if int(row.get("ou25_support_flag", 0) or 0) == 1:
        notes.append(f"OU25 {row.get('ou25_alignment', 'ALIGN')}")
    if not notes:
        return "No strong cross-market support."
    return " | ".join(notes)


def add_premium_profile(row: dict) -> None:
    league = normalize_league(row.get("league"))
    cs_class = _safe_str(row.get("cs_class")).upper()
    top1 = _safe_float(row.get("cs1_p"))
    top3_total = _safe_float(row.get("cs_top3_total_prob"))
    entropy = _safe_float(row.get("cs_entropy"))
    concentration = _safe_str(row.get("cs_concentration_bucket")).upper()
    ftr_alignment = normalize_align(row.get("ftr_alignment"))
    btts_alignment = normalize_align(row.get("btts_alignment"))
    ou25_alignment = normalize_align(row.get("ou25_alignment"))
    scoreline = _safe_str(row.get("cs1"))

    premium_league_flag = int(league in PREMIUM_LEAGUE_ALLOWLIST)
    common_scoreline_flag = int(scoreline in COMMON_PREMIUM_SCORELINES)

    pro_cs_banker_flag = int(
        cs_class == "CS_BANKER"
        and premium_league_flag == 1
        and pd.notna(top1)
        and top1 >= 0.18
        and pd.notna(entropy)
        and entropy <= 2.4
    )
    pro_cs_banker_plus_flag = int(
        pro_cs_banker_flag == 1
        and pd.notna(top3_total)
        and top3_total >= 0.50
    )
    pro_cs_triple_align_flag = int(
        cs_class == "CS_BANKER"
        and premium_league_flag == 1
        and ftr_alignment == "STRONG_ALIGN"
        and btts_alignment == "STRONG_ALIGN"
        and ou25_alignment == "STRONG_ALIGN"
    )
    pro_cs_elite_flag = int(
        premium_league_flag == 1
        and pd.notna(top1)
        and top1 >= 0.20
        and pd.notna(top3_total)
        and top3_total >= 0.50
        and pd.notna(entropy)
        and entropy <= 2.4
        and concentration == "ELITE"
        and int(row.get("conflict_count", 0) or 0) <= 1
    )
    premium_watch_flag = int(
        premium_league_flag == 1
        and cs_class == "CS_WATCH"
        and pd.notna(top1)
        and top1 >= 0.20
        and pd.notna(entropy)
        and entropy <= 2.4
        and concentration == "ELITE"
    )

    premium_rule_hits: list[str] = []
    if pro_cs_triple_align_flag:
        premium_rule_hits.append("PRO_CS_TRIPLE_ALIGN")
    if pro_cs_banker_plus_flag:
        premium_rule_hits.append("PRO_CS_BANKER_PLUS")
    if pro_cs_banker_flag:
        premium_rule_hits.append("PRO_CS_BANKER")
    if pro_cs_elite_flag:
        premium_rule_hits.append("PRO_CS_ELITE")
    if premium_watch_flag:
        premium_rule_hits.append("PREMIUM_WATCH")

    primary_rule = premium_rule_hits[0] if premium_rule_hits else "PUBLIC"

    row["premium_league_flag"] = premium_league_flag
    row["common_scoreline_flag"] = common_scoreline_flag
    row["pro_cs_banker_flag"] = pro_cs_banker_flag
    row["pro_cs_banker_plus_flag"] = pro_cs_banker_plus_flag
    row["pro_cs_triple_align_flag"] = pro_cs_triple_align_flag
    row["pro_cs_elite_flag"] = pro_cs_elite_flag
    row["premium_watch_flag"] = premium_watch_flag
    row["premium_rule_hits"] = " | ".join(premium_rule_hits) if premium_rule_hits else ""
    row["premium_primary_rule"] = primary_rule
    row["website_segment"] = "PREMIUM" if premium_rule_hits else "PUBLIC"
    row["premium_priority_rank"] = PREMIUM_RULE_ORDER.index(primary_rule) if primary_rule in PREMIUM_RULE_ORDER else len(PREMIUM_RULE_ORDER)
