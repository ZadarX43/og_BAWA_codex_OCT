#!/usr/bin/env python3
"""
Patch canonical merged odds from the current API-Football overlay bundle.

Purpose
-------
- Fill missing canonical odds in `Matches/__merged__/<LeagueTag>__merged.csv`
  using the normalized current-window overlay bundle under:
    `reports/latest/api_current_context_overlay_window/normalized/`
- Keep the repair auditable and additive:
  - exact / near-exact fixture matching only
  - one-to-one fixture assignment only
  - fill missing values only unless `--overwrite` is supplied

This is designed as a recovery / fallback step for live-window gaps such as
Portugal Liga source odds attrition, without weakening the production audit
that correctly reports missing primary source pairs.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PREFERRED_BOOKMAKERS = [
    "Bet365",
    "Pinnacle",
    "10Bet",
    "William Hill",
]

CANONICAL_ODDS_COLS = [
    "odds_ft_home_team_win",
    "odds_ft_draw",
    "odds_ft_away_team_win",
    "odds_ft_over25",
    "odds_ft_under25",
    "odds_btts_yes",
    "odds_btts_no",
]

GENERIC_TEAM_TOKENS = {
    "fc",
    "cf",
    "cd",
    "gd",
    "ac",
    "sc",
    "club",
    "fk",
    "sv",
}

TEAM_WORD_ALIASES = {
    "porto": "fc porto",
    "estoril": "estoril praia",
    "braga": "sporting braga",
    "nacional": "cd nacional",
    "guimaraes": "vitoria guimaraes",
}


@dataclass
class MatchCandidate:
    merged_idx: int
    overlay_idx: int
    score: float
    home_score: float
    away_score: float
    date_gap_days: int


def _league_tag(league: str) -> str:
    return league.replace(" ", "_")


def _coerce_odds(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.where(vals > 1.0, np.nan)


def _strip_accents(text: object) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", str(text or ""))
        if not unicodedata.combining(ch)
    )


def _normalize_team_name(text: object) -> str:
    raw = _strip_accents(text).lower()
    raw = raw.replace("&", " and ")
    raw = re.sub(r"[^a-z0-9 ]+", " ", raw)
    raw = " ".join(raw.split())
    aliased = TEAM_WORD_ALIASES.get(raw, raw)
    tokens = [tok for tok in aliased.split() if tok and tok not in GENERIC_TEAM_TOKENS]
    return " ".join(tokens)


def _team_similarity(left: object, right: object) -> float:
    a = _normalize_team_name(left)
    b = _normalize_team_name(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    seq = difflib.SequenceMatcher(a=a, b=b).ratio()
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    jac = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
    substring = 1.0 if a in b or b in a else 0.0
    return max(seq, jac, substring)


def _safe_dt(series: pd.Series) -> pd.Series:
    dtv = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        dtv = dtv.dt.tz_convert(None)
    except Exception:
        pass
    return dtv


def _read_overlay_frames(root: Path, league: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tag = _league_tag(league)
    fixture_files = sorted(root.glob(f"fixtures_master__{tag}__*.csv"))
    odds_files = sorted(root.glob(f"odds_prematch_long__{tag}__*.csv"))
    if not fixture_files:
        raise SystemExit(f"Missing overlay fixtures_master for {league}: {root}")
    if not odds_files:
        raise SystemExit(f"Missing overlay odds_prematch_long for {league}: {root}")
    fixtures = pd.read_csv(fixture_files[-1], low_memory=False)
    odds = pd.read_csv(odds_files[-1], low_memory=False)
    return fixtures, odds


def _read_merged(repo_root: Path, league: str) -> Tuple[Path, pd.DataFrame]:
    merged_path = repo_root / "Matches" / "__merged__" / f"{_league_tag(league)}__merged.csv"
    if not merged_path.exists():
        raise SystemExit(f"Missing merged file: {merged_path}")
    merged = pd.read_csv(merged_path, low_memory=False)
    return merged_path, merged


def _select_overlay_fixture_slice(merged: pd.DataFrame, fixtures: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = merged.copy()
    fixtures = fixtures.copy()
    merged["__match_dt"] = _safe_dt(merged.get("match_date", pd.Series(pd.NA, index=merged.index)))
    fixtures["__match_dt"] = _safe_dt(fixtures.get("match_date", pd.Series(pd.NA, index=fixtures.index)))
    merged["__needs_any_odds"] = False
    for col in CANONICAL_ODDS_COLS:
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = _coerce_odds(merged[col])
        merged["__needs_any_odds"] |= merged[col].isna()

    active = merged.loc[merged["__needs_any_odds"]].copy()
    if active.empty:
        return active, fixtures.iloc[0:0].copy()

    min_dt = active["__match_dt"].dropna().min()
    max_dt = active["__match_dt"].dropna().max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        return active, fixtures.copy()
    lo = min_dt - pd.Timedelta(days=1)
    hi = max_dt + pd.Timedelta(days=1)
    fixture_slice = fixtures.loc[fixtures["__match_dt"].between(lo, hi, inclusive="both")].copy()
    return active, fixture_slice


def _build_candidates(merged_slice: pd.DataFrame, fixture_slice: pd.DataFrame) -> List[MatchCandidate]:
    candidates: List[MatchCandidate] = []
    for merged_idx, mrow in merged_slice.iterrows():
        for overlay_idx, orow in fixture_slice.iterrows():
            date_gap = abs((mrow["__match_dt"] - orow["__match_dt"]).days) if pd.notna(mrow["__match_dt"]) and pd.notna(orow["__match_dt"]) else 999
            if date_gap > 1:
                continue
            home_score = _team_similarity(mrow.get("home_team_name"), orow.get("home_team_name"))
            away_score = _team_similarity(mrow.get("away_team_name"), orow.get("away_team_name"))
            if min(home_score, away_score) < 0.72:
                continue
            total = home_score + away_score - (0.05 * date_gap)
            candidates.append(
                MatchCandidate(
                    merged_idx=int(merged_idx),
                    overlay_idx=int(overlay_idx),
                    score=float(total),
                    home_score=float(home_score),
                    away_score=float(away_score),
                    date_gap_days=int(date_gap),
                )
            )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _assign_candidates(candidates: Iterable[MatchCandidate]) -> Dict[int, MatchCandidate]:
    assigned_merged = set()
    assigned_overlay = set()
    final: Dict[int, MatchCandidate] = {}
    for cand in candidates:
        if cand.merged_idx in assigned_merged or cand.overlay_idx in assigned_overlay:
            continue
        assigned_merged.add(cand.merged_idx)
        assigned_overlay.add(cand.overlay_idx)
        final[cand.merged_idx] = cand
    return final


def _pick_selection_rows(df: pd.DataFrame, market_code: str, selections: Dict[str, str]) -> Dict[Tuple[int, str], float]:
    sub = df.loc[df["market_code"].astype(str).eq(market_code)].copy()
    if sub.empty:
        return {}
    sub["odds"] = _coerce_odds(sub["odds"])
    sub = sub.loc[sub["odds"].notna()].copy()
    if sub.empty:
        return {}
    sub["bookmaker_rank"] = sub["bookmaker_name"].astype(str).map(
        {name: idx for idx, name in enumerate(PREFERRED_BOOKMAKERS)}
    ).fillna(len(PREFERRED_BOOKMAKERS))
    sub["latest_rank"] = (1 - pd.to_numeric(sub.get("is_latest_pre_kickoff", 0), errors="coerce").fillna(0)).astype(int)
    sub = sub.sort_values(["fixture_id", "selection_code", "bookmaker_rank", "latest_rank", "odds"], ascending=[True, True, True, True, True])

    out: Dict[Tuple[int, str], float] = {}
    for _, row in sub.iterrows():
        selection = str(row.get("selection_code", "")).strip().upper()
        mapped = selections.get(selection)
        if mapped is None:
            continue
        key = (int(row["fixture_id"]), mapped)
        if key not in out:
            out[key] = float(row["odds"])
    return out


def _pick_ou25_rows(df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    sub = df.loc[df["market_code"].astype(str).eq("OU")].copy()
    if sub.empty:
        return {}
    sub["odds"] = _coerce_odds(sub["odds"])
    sub = sub.loc[sub["odds"].notna()].copy()
    if sub.empty:
        return {}
    line_series = sub.get("line_value", pd.Series(np.nan, index=sub.index))
    sub["selection_code"] = sub.get("selection_code", pd.Series("", index=sub.index)).astype(str).str.upper()
    line_num = pd.to_numeric(line_series, errors="coerce")
    keep = (
        line_num.eq(2.5)
        | sub["selection_code"].isin({"OVER_2.5", "UNDER_2.5"})
    )
    sub = sub.loc[keep].copy()
    if sub.empty:
        return {}
    sub["bookmaker_rank"] = sub["bookmaker_name"].astype(str).map(
        {name: idx for idx, name in enumerate(PREFERRED_BOOKMAKERS)}
    ).fillna(len(PREFERRED_BOOKMAKERS))
    sub["latest_rank"] = (1 - pd.to_numeric(sub.get("is_latest_pre_kickoff", 0), errors="coerce").fillna(0)).astype(int)
    sub = sub.sort_values(["fixture_id", "selection_code", "bookmaker_rank", "latest_rank", "odds"], ascending=[True, True, True, True, True])
    out: Dict[Tuple[int, str], float] = {}
    mapping = {"OVER_2.5": "odds_ft_over25", "UNDER_2.5": "odds_ft_under25"}
    for _, row in sub.iterrows():
        mapped = mapping.get(str(row["selection_code"]).strip().upper())
        if mapped is None:
            continue
        key = (int(row["fixture_id"]), mapped)
        if key not in out:
            out[key] = float(row["odds"])
    return out


def _build_overlay_odds_map(odds: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    ftr = _pick_selection_rows(
        odds,
        "FTR",
        {
            "HOME": "odds_ft_home_team_win",
            "DRAW": "odds_ft_draw",
            "AWAY": "odds_ft_away_team_win",
        },
    )
    btts = _pick_selection_rows(
        odds,
        "BTTS",
        {
            "YES": "odds_btts_yes",
            "NO": "odds_btts_no",
        },
    )
    ou25 = _pick_ou25_rows(odds)

    merged: Dict[int, Dict[str, float]] = {}
    for source in (ftr, btts, ou25):
        for (fixture_id, col), value in source.items():
            merged.setdefault(int(fixture_id), {})[col] = float(value)
    return merged


def patch_overlay_odds(
    repo_root: Path,
    overlay_root: Path,
    league: str,
    overwrite: bool = False,
) -> Dict[str, object]:
    merged_path, merged = _read_merged(repo_root, league)
    fixtures, odds = _read_overlay_frames(overlay_root, league)
    merged_slice, fixture_slice = _select_overlay_fixture_slice(merged, fixtures)
    candidates = _build_candidates(merged_slice, fixture_slice)
    assignments = _assign_candidates(candidates)
    overlay_odds = _build_overlay_odds_map(odds)

    if "api_overlay_odds_backfill_source" not in merged.columns:
        merged["api_overlay_odds_backfill_source"] = ""

    fill_counts = {col: 0 for col in CANONICAL_ODDS_COLS}
    matched_rows = 0
    assignment_rows: List[Dict[str, object]] = []

    for merged_idx, cand in assignments.items():
        overlay_row = fixture_slice.loc[cand.overlay_idx]
        fixture_id = int(overlay_row["fixture_id"])
        odds_map = overlay_odds.get(fixture_id, {})
        if not odds_map:
            continue
        matched_rows += 1
        assignment_rows.append(
            {
                "merged_fixture_key": str(merged.at[merged_idx, "fixture_key"]),
                "overlay_fixture_key": str(overlay_row.get("fixture_key", "")),
                "fixture_id": fixture_id,
                "home_score": cand.home_score,
                "away_score": cand.away_score,
                "date_gap_days": cand.date_gap_days,
            }
        )
        wrote_any = False
        for col, value in odds_map.items():
            cur = _coerce_odds(pd.Series([merged.at[merged_idx, col]])).iloc[0]
            if overwrite or pd.isna(cur):
                merged.at[merged_idx, col] = value
                fill_counts[col] += 1
                wrote_any = True
        if wrote_any:
            merged.at[merged_idx, "api_overlay_odds_backfill_source"] = "api_current_context_overlay_window"

    merged.to_csv(merged_path, index=False)

    summary = {
        "league": league,
        "merged_path": str(merged_path),
        "overlay_root": str(overlay_root),
        "candidate_pairs": len(candidates),
        "assigned_pairs": len(assignments),
        "matched_rows_with_overlay_odds": matched_rows,
        "fill_counts": fill_counts,
        "assignments": assignment_rows,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch merged canonical odds from API-Football overlay bundle")
    ap.add_argument("--root", default=".", help="Repo root")
    ap.add_argument("--overlay-root", default="reports/latest/api_current_context_overlay_window/normalized")
    ap.add_argument("--league", required=True, help="League name, e.g. 'Portugal Liga'")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing canonical odds")
    ap.add_argument("--report-json", default="", help="Optional path for a JSON summary report")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    overlay_root = (root / args.overlay_root).resolve() if not Path(args.overlay_root).is_absolute() else Path(args.overlay_root)
    summary = patch_overlay_odds(root, overlay_root, args.league, overwrite=bool(args.overwrite))

    if args.report_json:
        report_path = Path(args.report_json)
        if not report_path.is_absolute():
            report_path = root / report_path
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"WROTE {report_path}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
