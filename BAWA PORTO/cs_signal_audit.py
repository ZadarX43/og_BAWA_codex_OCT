#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_BASE_OUTDIR = Path("predictions_output/walk_forward")
DEFAULT_MASTER_DIR = DEFAULT_BASE_OUTDIR / "_MASTER"
DEFAULT_OUTDIR = DEFAULT_MASTER_DIR / "CS_SIGNAL_AUDITS"

# ---------------------------------------------------------------------------
# New constants for combo and ranked exports
# ---------------------------------------------------------------------------
DEFAULT_GRADED_MINS = [8, 15]
TOP_EXPORT_LIMIT = 25
RANK_EXPORT_SPECS = [
    ("TOP_HIT_RATE", ["hit_rate", "graded", "wins", "rows"], [False, False, False, False]),
    ("TOP_WINS", ["wins", "hit_rate", "graded", "rows"], [False, False, False, False]),
    ("TOP_GRADED", ["graded", "hit_rate", "wins", "rows"], [False, False, False, False]),
    ("TOP_BALANCED_SCORE", ["balanced_score", "hit_rate", "graded", "wins"], [False, False, False, False]),
]

CS_AUDIT_SPECS = {
    "ftr": {
        "hit_col": "ftr_hit",
        "selection_col": "selection",
        "direction_views": ["ALL"],
        "short_direction_labels": {"ALL": "CS_FTR"},
        "allowed_selections": {"HOME", "DRAW", "AWAY"},
    },
    "ou25": {
        "hit_col": "ou25_hit",
        "selection_col": "selection",
        "direction_views": ["ALL", "OVER25", "UNDER25"],
        "short_direction_labels": {
            "ALL": "CS_OU25",
            "OVER25": "CS_OU25_OVER",
            "UNDER25": "CS_OU25_UNDER",
        },
        "allowed_selections": {"OVER25", "UNDER25"},
    },
    "btts": {
        "hit_col": "btts_yes_hit",
        "selection_col": "selection",
        "direction_views": ["ALL", "YES", "NO"],
        "short_direction_labels": {
            "ALL": "CS_BTTS",
            "YES": "CS_BTTS_YES",
            "NO": "CS_BTTS_NO",
        },
        "allowed_selections": {"YES", "NO"},
    },
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_series_str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    return df.get(col, pd.Series(default, index=df.index)).astype("string").fillna(default)


def safe_series_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")


# ---------------------------------------------------------------------------
# Additional helpers for direction/score
# ---------------------------------------------------------------------------
def first_valid_num_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in cols:
        if col not in df.columns:
            continue
        out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out


def selection_aliases(market_key: str, direction: str) -> set[str]:
    direction = (direction or "").upper().strip()
    if market_key == "ou25":
        mapping = {
            "OVER25": {"OVER25", "OVER", "O25", "OVER_25", "OVER 2.5", "O2.5"},
            "UNDER25": {"UNDER25", "UNDER", "U25", "UNDER_25", "UNDER 2.5", "U2.5"},
        }
        return mapping.get(direction, {direction})
    if market_key == "btts":
        mapping = {
            "YES": {"YES", "Y", "BTTS_YES", "BTTS YES", "BTTS-YES"},
            "NO": {"NO", "N", "BTTS_NO", "BTTS NO", "BTTS-NO"},
        }
        return mapping.get(direction, {direction})
    return {direction}


def coalesce_num(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in cols:
        if col not in df.columns:
            continue
        out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out


def normalize_text_band(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("UNBANDABLE")


def band_series(
    series: pd.Series,
    bins: list[float],
    labels: list[str],
    *,
    absolute: bool = False,
) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if absolute:
        vals = vals.abs()
    out = pd.cut(vals, bins=bins, labels=labels, right=True)
    return out.astype("string").fillna("UNBANDABLE")


# ---------------------------------------------------------------------------
# Score parsing and top3 mass derivation helpers
# ---------------------------------------------------------------------------

def parse_score_token(value: object) -> tuple[float, float]:
    text = str(value or "").strip()
    if not text:
        return (np.nan, np.nan)

    text = text.replace(":", "-").replace("_", "-").replace(" ", "")
    parts = text.split("-")
    if len(parts) != 2:
        return (np.nan, np.nan)

    try:
        home = float(parts[0])
        away = float(parts[1])
        return (home, away)
    except Exception:
        return (np.nan, np.nan)


def derive_top3_score_masses(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    score_cols = ["cs1", "cs2", "cs3"]
    prob_cols = ["cs1_p", "cs2_p", "cs3_p"]

    home_win_mass = pd.Series(0.0, index=out.index, dtype="float64")
    draw_mass = pd.Series(0.0, index=out.index, dtype="float64")
    away_win_mass = pd.Series(0.0, index=out.index, dtype="float64")

    total_le2_mass = pd.Series(0.0, index=out.index, dtype="float64")
    total_ge3_mass = pd.Series(0.0, index=out.index, dtype="float64")
    exact2_mass = pd.Series(0.0, index=out.index, dtype="float64")

    mass_11 = pd.Series(0.0, index=out.index, dtype="float64")
    mass_21_12 = pd.Series(0.0, index=out.index, dtype="float64")
    mass_00_10_01 = pd.Series(0.0, index=out.index, dtype="float64")

    btts_yes_mass = pd.Series(0.0, index=out.index, dtype="float64")
    btts_no_mass = pd.Series(0.0, index=out.index, dtype="float64")
    clean_sheet_home_mass = pd.Series(0.0, index=out.index, dtype="float64")
    clean_sheet_away_mass = pd.Series(0.0, index=out.index, dtype="float64")
    any_00_mass = pd.Series(0.0, index=out.index, dtype="float64")

    entropy_terms = []

    for score_col, prob_col in zip(score_cols, prob_cols):
        if score_col not in out.columns or prob_col not in out.columns:
            continue

        probs = pd.to_numeric(out[prob_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        parsed = out[score_col].apply(parse_score_token)
        home = parsed.apply(lambda x: x[0]).astype("float64")
        away = parsed.apply(lambda x: x[1]).astype("float64")
        valid = home.notna() & away.notna()

        total = home + away
        is_home_win = valid & (home > away)
        is_draw = valid & (home == away)
        is_away_win = valid & (away > home)

        is_total_le2 = valid & (total <= 2)
        is_total_ge3 = valid & (total >= 3)
        is_exact2 = valid & (total == 2)

        is_11 = valid & (home == 1) & (away == 1)
        is_21_12 = valid & (((home == 2) & (away == 1)) | ((home == 1) & (away == 2)))
        is_00_10_01 = valid & (
            ((home == 0) & (away == 0))
            | ((home == 1) & (away == 0))
            | ((home == 0) & (away == 1))
        )

        is_btts_yes = valid & (home >= 1) & (away >= 1)
        is_btts_no = valid & ((home == 0) | (away == 0))
        is_clean_sheet_home = valid & (away == 0)
        is_clean_sheet_away = valid & (home == 0)
        is_00 = valid & (home == 0) & (away == 0)

        home_win_mass = home_win_mass + probs.where(is_home_win, 0.0)
        draw_mass = draw_mass + probs.where(is_draw, 0.0)
        away_win_mass = away_win_mass + probs.where(is_away_win, 0.0)

        total_le2_mass = total_le2_mass + probs.where(is_total_le2, 0.0)
        total_ge3_mass = total_ge3_mass + probs.where(is_total_ge3, 0.0)
        exact2_mass = exact2_mass + probs.where(is_exact2, 0.0)

        mass_11 = mass_11 + probs.where(is_11, 0.0)
        mass_21_12 = mass_21_12 + probs.where(is_21_12, 0.0)
        mass_00_10_01 = mass_00_10_01 + probs.where(is_00_10_01, 0.0)

        btts_yes_mass = btts_yes_mass + probs.where(is_btts_yes, 0.0)
        btts_no_mass = btts_no_mass + probs.where(is_btts_no, 0.0)
        clean_sheet_home_mass = clean_sheet_home_mass + probs.where(is_clean_sheet_home, 0.0)
        clean_sheet_away_mass = clean_sheet_away_mass + probs.where(is_clean_sheet_away, 0.0)
        any_00_mass = any_00_mass + probs.where(is_00, 0.0)

        entropy_terms.append(probs.where(probs > 0.0, np.nan))

    out["cs_mass_home_win_top3"] = home_win_mass
    out["cs_mass_draw_top3"] = draw_mass
    out["cs_mass_away_win_top3"] = away_win_mass

    out["cs_mass_total_le2_top3"] = total_le2_mass
    out["cs_mass_total_ge3_top3"] = total_ge3_mass
    out["cs_mass_exact2_top3"] = exact2_mass

    out["cs_mass_1_1_top3"] = mass_11
    out["cs_mass_2_1_1_2_top3"] = mass_21_12
    out["cs_mass_0_0_1_0_0_1_top3"] = mass_00_10_01

    out["cs_mass_btts_yes_top3"] = btts_yes_mass
    out["cs_mass_btts_no_top3"] = btts_no_mass
    out["cs_mass_clean_sheet_home_top3"] = clean_sheet_home_mass
    out["cs_mass_clean_sheet_away_top3"] = clean_sheet_away_mass
    out["cs_mass_any_0_0_top3"] = any_00_mass

    if entropy_terms:
        entropy = pd.Series(0.0, index=out.index, dtype="float64")
        for probs in entropy_terms:
            entropy = entropy + (-(probs * np.log(probs))).fillna(0.0)
        out["cs_entropy_top3_from_scores"] = entropy
    else:
        out["cs_entropy_top3_from_scores"] = pd.Series(np.nan, index=out.index, dtype="float64")

    top_probs = [pd.to_numeric(out[c], errors="coerce") for c in prob_cols if c in out.columns]
    if top_probs:
        top3_sum = pd.concat(top_probs, axis=1).sum(axis=1, min_count=1)
        out["cs_top3_prob_sum_from_scores"] = pd.to_numeric(top3_sum, errors="coerce")
        out["cs_top1_prob_from_scores"] = pd.to_numeric(out.get("cs1_p", np.nan), errors="coerce")
        out["cs_concentration_ratio_from_scores"] = np.where(
            out["cs_top3_prob_sum_from_scores"].gt(0),
            out["cs_top1_prob_from_scores"] / out["cs_top3_prob_sum_from_scores"],
            np.nan,
        )
    else:
        out["cs_top3_prob_sum_from_scores"] = pd.Series(np.nan, index=out.index, dtype="float64")
        out["cs_top1_prob_from_scores"] = pd.Series(np.nan, index=out.index, dtype="float64")
        out["cs_concentration_ratio_from_scores"] = pd.Series(np.nan, index=out.index, dtype="float64")

    return out


def summarize_slice(df: pd.DataFrame, hit_col: str) -> dict[str, float | int]:
    graded_mask = pd.to_numeric(df.get(hit_col, np.nan), errors="coerce").notna()
    graded = int(graded_mask.sum())
    wins = float(pd.to_numeric(df.loc[graded_mask, hit_col], errors="coerce").fillna(0).sum()) if graded else 0.0
    losses = int(graded - wins) if graded else 0
    return {
        "rows": int(len(df)),
        "graded": graded,
        "wins": wins,
        "losses": losses,
        "hit_rate": float(wins / graded) if graded else np.nan,
        "avg_bookie_od": float(pd.to_numeric(df.get("bookie_od", np.nan), errors="coerce").mean()) if len(df) else np.nan,
        "avg_model_p_for_bookie": float(pd.to_numeric(df.get("model_p_for_bookie", np.nan), errors="coerce").mean()) if len(df) else np.nan,
    }


def grouped_summary(df: pd.DataFrame, group_cols: list[str], hit_col: str) -> pd.DataFrame:
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        cols = group_cols + ["rows", "graded", "wins", "losses", "hit_rate", "avg_bookie_od", "avg_model_p_for_bookie"]
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, object]] = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(summarize_slice(grp, hit_col))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in ["hit_rate", "rows", "graded"] if c in out.columns]
    return out.sort_values(sort_cols, ascending=[False, False, False][: len(sort_cols)]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Additional helpers for combos and direction/score
# ---------------------------------------------------------------------------

def add_balanced_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    graded = pd.to_numeric(out.get("graded", np.nan), errors="coerce").fillna(0.0)
    hit_rate = pd.to_numeric(out.get("hit_rate", np.nan), errors="coerce")
    out["balanced_score"] = hit_rate * np.log1p(graded)
    return out


def direction_hit_series(df: pd.DataFrame, market_key: str, direction: str, base_hit_col: str) -> pd.Series:
    base = safe_series_num(df, base_hit_col)
    if direction == "ALL":
        return base

    home_goals = first_valid_num_series(df, [
        "home_team_goal_count",
        "home_goals",
        "full_time_home_goals",
        "ft_home_goals",
        "hg",
    ])
    away_goals = first_valid_num_series(df, [
        "away_team_goal_count",
        "away_goals",
        "full_time_away_goals",
        "ft_away_goals",
        "ag",
    ])
    total_goals = first_valid_num_series(df, [
        "total_goal_count",
        "total_goals",
        "ft_total_goals",
    ])

    if total_goals.notna().sum() == 0 and home_goals.notna().any() and away_goals.notna().any():
        total_goals = home_goals + away_goals

    if market_key == "ou25":
        actual_over25 = first_valid_num_series(df, [
            "actual_over25",
            "is_over25",
            "over25_actual",
        ])
        if actual_over25.notna().sum() == 0:
            actual_over25 = pd.Series(
                np.where(total_goals.notna(), (total_goals >= 3).astype(float), np.nan),
                index=df.index,
                dtype="float64",
            )
        if actual_over25.notna().sum() == 0:
            actual_over25 = base.copy()

        if direction == "OVER25":
            return actual_over25
        if direction == "UNDER25":
            return pd.Series(
                np.where(actual_over25.notna(), 1.0 - actual_over25, np.nan),
                index=df.index,
                dtype="float64",
            )

    if market_key == "btts":
        actual_btts_yes = first_valid_num_series(df, [
            "actual_btts_yes",
            "is_btts_yes",
            "btts_yes_actual",
        ])
        if actual_btts_yes.notna().sum() == 0 and home_goals.notna().any() and away_goals.notna().any():
            actual_btts_yes = pd.Series(
                np.where(
                    home_goals.notna() & away_goals.notna(),
                    ((home_goals >= 1) & (away_goals >= 1)).astype(float),
                    np.nan,
                ),
                index=df.index,
                dtype="float64",
            )
        if actual_btts_yes.notna().sum() == 0:
            actual_btts_yes = base.copy()

        if direction == "YES":
            return actual_btts_yes
        if direction == "NO":
            return pd.Series(
                np.where(actual_btts_yes.notna(), 1.0 - actual_btts_yes, np.nan),
                index=df.index,
                dtype="float64",
            )

    return base


def build_direction_frame(df: pd.DataFrame, market_key: str, direction: str) -> tuple[pd.DataFrame, str]:
    spec = CS_AUDIT_SPECS[market_key]
    base_hit_col = spec["hit_col"]
    selection_col = spec["selection_col"]

    out = df.copy()
    if direction != "ALL":
        aliases = selection_aliases(market_key, direction)
        selection_vals = safe_series_str(out, selection_col).str.upper().str.strip()
        out = out.loc[selection_vals.isin(aliases)].copy()

    direction_hit_col = f"__direction_hit__{market_key}__{direction}"
    out[direction_hit_col] = pd.to_numeric(
        direction_hit_series(out, market_key, direction, base_hit_col),
        errors="coerce",
    )
    return out, direction_hit_col


def export_ranked_views(df: pd.DataFrame, base_prefix: str, stem: str, outdir: Path) -> None:
    ranked = add_balanced_score(df)

    graded_num = pd.to_numeric(ranked.get("graded", np.nan), errors="coerce").fillna(0.0)
    hit_rate_num = pd.to_numeric(ranked.get("hit_rate", np.nan), errors="coerce")
    wins_num = pd.to_numeric(ranked.get("wins", np.nan), errors="coerce")
    rows_num = pd.to_numeric(ranked.get("rows", np.nan), errors="coerce")

    ranked["graded"] = graded_num
    ranked["hit_rate"] = hit_rate_num
    ranked["wins"] = wins_num
    ranked["rows"] = rows_num
    ranked["balanced_score"] = hit_rate_num * np.log1p(graded_num.fillna(0.0))

    for graded_min in DEFAULT_GRADED_MINS:
        sub = ranked.loc[graded_num >= graded_min].copy()

        for rank_name, cols, ascending in RANK_EXPORT_SPECS:
            present_cols = [c for c in cols if c in ranked.columns]
            if not present_cols:
                continue

            out_path = outdir / f"{base_prefix}__{stem}__{rank_name}__GRADED_MIN_{graded_min}.csv"

            if sub.empty:
                ranked.head(0).to_csv(out_path, index=False)
                continue

            asc = ascending[: len(present_cols)]
            top = (
                sub.sort_values(present_cols, ascending=asc, na_position="last")
                .head(TOP_EXPORT_LIMIT)
                .reset_index(drop=True)
            )
            top.to_csv(out_path, index=False)

    fallback = ranked.loc[graded_num > 0].copy()
    for rank_name, cols, ascending in RANK_EXPORT_SPECS:
        present_cols = [c for c in cols if c in ranked.columns]
        if not present_cols:
            continue

        out_path = outdir / f"{base_prefix}__{stem}__{rank_name}__GRADED_MIN_FALLBACK.csv"

        if fallback.empty:
            ranked.head(0).to_csv(out_path, index=False)
            continue

        asc = ascending[: len(present_cols)]
        top = (
            fallback.sort_values(present_cols, ascending=asc, na_position="last")
            .head(TOP_EXPORT_LIMIT)
            .reset_index(drop=True)
        )
        top.to_csv(out_path, index=False)



def combo_summary(df: pd.DataFrame, hit_col: str, combo_name: str, logic: str, rules: str) -> dict[str, object]:
    row = {
        "combo_name": combo_name,
        "logic": logic,
        "rules": rules,
    }
    row.update(summarize_slice(df, hit_col))
    row["avg_feature_value"] = np.nan
    row["avg_bookie_od"] = float(pd.to_numeric(df.get("bookie_od", np.nan), errors="coerce").mean()) if len(df) else np.nan
    row["avg_model_p_for_bookie"] = float(pd.to_numeric(df.get("model_p_for_bookie", np.nan), errors="coerce").mean()) if len(df) else np.nan
    row["balanced_score"] = (
        float(row["hit_rate"]) * float(np.log1p(row["graded"]))
        if pd.notna(row.get("hit_rate")) and pd.notna(row.get("graded")) and float(row["graded"]) > 0
        else np.nan
    )
    return row


# ---------------------------------------------------------------------------
# Load scored files
# ---------------------------------------------------------------------------
def discover_windows(base_outdir: Path, windows_csv: str = "") -> list[Path]:
    windows = sorted([p for p in base_outdir.iterdir() if p.is_dir() and p.name != "_MASTER"])
    if not windows_csv.strip():
        return windows

    allow = {x.strip() for x in windows_csv.split(",") if x.strip()}
    return [w for w in windows if w.name in allow]


def discover_scored_csvs(base_outdir: Path, windows_csv: str = "") -> list[Path]:
    scored_paths: list[Path] = []
    seen: set[Path] = set()

    for window_dir in discover_windows(base_outdir, windows_csv):
        scored_dir = window_dir / "03_scored"
        if not scored_dir.exists():
            continue

        for path in sorted(scored_dir.glob("DEPLOY_COMBINED_SCORED_*.csv")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            scored_paths.append(path)

    return scored_paths


def load_scored_frames(base_outdir: Path, windows_csv: str = "") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    scored_paths = discover_scored_csvs(base_outdir, windows_csv)
    log(f"[cs-signal-audit] discovered scored csvs={len(scored_paths)}")

    for path in scored_paths:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            log(f"[skip] failed reading {path}: {exc}")
            continue

        window_id = path.parent.parent.name
        df["window_id"] = window_id
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["market"] = safe_series_str(out, "market").str.lower().str.strip()
    out["selection"] = safe_series_str(out, "selection").str.upper().str.strip()
    out["league"] = safe_series_str(out, "league").str.strip()
    out["source_tier_file"] = safe_series_str(out, "source_tier_file").str.upper().str.strip()
    return out


def sort_combined_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sort_cols: list[str] = []
    for col in [
        "window_id",
        "league",
        "match_date",
        "home_team_name",
        "away_team_name",
        "market",
        "selection",
    ]:
        if col in out.columns:
            sort_cols.append(col)

    if sort_cols:
        out = out.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return out


def write_market_combined_csv(df: pd.DataFrame, out_path: Path) -> None:
    out = sort_combined_market_frame(df)
    out.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# CS feature engineering
# ---------------------------------------------------------------------------
def attach_cs_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = derive_top3_score_masses(out)

    # FTR side masses / support
    out["cs_mass_home_win"] = coalesce_num(out, [
        "cs_mass_home_win", "cs_mass_home_win_top3", "p_home_pois", "cs_home_win_mass"
    ])
    out["cs_mass_draw"] = coalesce_num(out, [
        "cs_mass_draw", "cs_mass_draw_top3", "p_draw_pois", "cs_draw_mass"
    ])
    out["cs_mass_away_win"] = coalesce_num(out, [
        "cs_mass_away_win", "cs_mass_away_win_top3", "p_away_pois", "cs_away_win_mass"
    ])

    out["pick_side_mass_top3"] = coalesce_num(out, ["pick_side_mass_top3", "cs_pick_side_mass_top3"])
    out["pick_side_margin_top3"] = coalesce_num(out, ["pick_side_margin_top3", "cs_pick_side_margin_top3"])
    out["draw_top3_mass"] = coalesce_num(out, ["draw_top3_mass", "cs_draw_top3_mass", "cs_mass_draw_top3"])
    out["cs_top3_match_ftr"] = coalesce_num(out, ["cs_top3_match_ftr"])

    # Top-N mass / concentration / entropy
    out["cs_top1_prob"] = coalesce_num(out, ["cs_top1_prob", "cs_top1_prob_from_scores", "cs1_p"])
    out["cs_top3_prob_sum"] = coalesce_num(out, ["cs_top3_prob_sum", "cs_top3_prob_sum_from_scores", "cs_top3_mass", "cs_trunc_mass_0_6"])
    out["cs_top6_prob_sum"] = coalesce_num(out, ["cs_top6_prob_sum", "cs_top3_prob_sum_from_scores"])
    out["cs_entropy_top3"] = coalesce_num(out, ["cs_entropy_top3", "cs_top3_entropy", "cs_entropy_top3_from_scores"])
    out["cs_entropy_top6"] = coalesce_num(out, ["cs_entropy_top6", "cs_entropy_top3_from_scores"])

    out["cs_concentration_ratio"] = coalesce_num(out, ["cs_concentration_ratio", "cs_concentration_ratio_from_scores"])

    # OU25 support masses
    out["cs_mass_total_le2"] = coalesce_num(out, ["cs_mass_total_le2", "cs_mass_total_le2_top3"])
    out["cs_mass_total_ge3"] = coalesce_num(out, ["cs_mass_total_ge3", "cs_mass_total_ge3_top3"])
    out["cs_mass_exact2"] = coalesce_num(out, ["cs_mass_exact2", "cs_mass_exact2_top3"])
    out["cs_mass_1_1"] = coalesce_num(out, ["cs_mass_1_1", "cs_mass_1_1_top3", "cs11_p"])
    out["cs_mass_2_1_1_2"] = coalesce_num(out, ["cs_mass_2_1_1_2", "cs_mass_2_1_1_2_top3"])
    out["cs_mass_0_0_1_0_0_1"] = coalesce_num(out, [
        "cs_mass_0_0_1_0_0_1", "cs_mass_0_0_1_0_0_1_top3", "cs00_top3_mass"
    ])

    # BTTS support masses
    out["cs_mass_btts_yes"] = coalesce_num(out, ["cs_mass_btts_yes", "cs_mass_btts_yes_top3", "btts_top3_mass"])
    out["cs_mass_btts_no"] = coalesce_num(out, ["cs_mass_btts_no", "cs_mass_btts_no_top3"])
    out["cs_mass_clean_sheet_home"] = coalesce_num(out, ["cs_mass_clean_sheet_home", "cs_mass_clean_sheet_home_top3"])
    out["cs_mass_clean_sheet_away"] = coalesce_num(out, ["cs_mass_clean_sheet_away", "cs_mass_clean_sheet_away_top3"])
    out["cs_mass_any_0_0"] = coalesce_num(out, ["cs_mass_any_0_0", "cs_mass_any_0_0_top3", "p00_est", "cs00_top3_mass"])

    out["cs_mass_btts_yes_ratio"] = np.where(
        pd.to_numeric(out.get("cs_mass_btts_no", np.nan), errors="coerce") > 0,
        pd.to_numeric(out.get("cs_mass_btts_yes", np.nan), errors="coerce")
        / pd.to_numeric(out.get("cs_mass_btts_no", np.nan), errors="coerce"),
        np.nan,
    )

    # Bands
    out["pick_side_margin_band"] = band_series(
        out["pick_side_margin_top3"],
        bins=[-np.inf, 0.02, 0.05, 0.10, 0.15, 0.25, np.inf],
        labels=["<=0.02", "0.02-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.25", ">0.25"],
        absolute=True,
    )
    out["draw_top3_mass_band"] = band_series(
        out["draw_top3_mass"],
        bins=[-np.inf, 0.05, 0.10, 0.15, 0.20, 0.30, np.inf],
        labels=["<=0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", ">0.30"],
    )
    out["total_mass_band"] = band_series(
        out["cs_mass_total_ge3"],
        bins=[-np.inf, 0.30, 0.40, 0.50, 0.60, 0.70, np.inf],
        labels=["<=0.30", "0.30-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", ">0.70"],
    )
    out["mutual_score_mass_band"] = band_series(
        out["cs_mass_btts_yes"],
        bins=[-np.inf, 0.20, 0.30, 0.40, 0.50, 0.60, np.inf],
        labels=["<=0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.60", ">0.60"],
    )
    out["clean_sheet_mass_band"] = band_series(
        out[["cs_mass_clean_sheet_home", "cs_mass_clean_sheet_away"]].max(axis=1),
        bins=[-np.inf, 0.20, 0.30, 0.40, 0.50, 0.60, np.inf],
        labels=["<=0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.60", ">0.60"],
    )

    out["is_11_trap"] = np.where(
        pd.to_numeric(out.get("cs_mass_1_1", np.nan), errors="coerce") >= 0.10,
        "YES",
        "NO",
    )

    return out


#
# ---------------------------------------------------------------------------
# Combo builder functions for each market
# ---------------------------------------------------------------------------

def build_ftr_combo_rows(sub: pd.DataFrame, hit_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    side_mass = pd.to_numeric(sub.get("pick_side_mass_top3", np.nan), errors="coerce")
    margin = pd.to_numeric(sub.get("pick_side_margin_top3", np.nan), errors="coerce")
    draw_mass = pd.to_numeric(sub.get("draw_top3_mass", np.nan), errors="coerce")

    combos = [
        (
            "FTR_CS_RESCUE_MASS_MARGIN_ONLY",
            "all_rules",
            "pick_side_mass_top3 >= 0.4 | pick_side_margin_top3 >= 0.15",
            (side_mass >= 0.40) & (margin >= 0.15),
        ),
        (
            "FTR_CS_RESCUE_BALANCED",
            "all_rules",
            "pick_side_mass_top3 >= 0.4 | pick_side_margin_top3 >= 0.15 | draw_top3_mass <= 0.10",
            (side_mass >= 0.40) & (margin >= 0.15) & (draw_mass <= 0.10),
        ),
        (
            "FTR_CS_RESCUE_STRICT",
            "all_rules",
            "pick_side_mass_top3 >= 0.4 | pick_side_margin_top3 >= 0.15 | draw_top3_mass <= 0.05",
            (side_mass >= 0.40) & (margin >= 0.15) & (draw_mass <= 0.05),
        ),
    ]

    for combo_name, logic, rules, mask in combos:
        g = sub.loc[mask.fillna(False)].copy()
        rows.append(combo_summary(g, hit_col, combo_name, logic, rules))
    return rows


def build_ou25_combo_rows(sub: pd.DataFrame, hit_col: str, direction: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ge3 = pd.to_numeric(sub.get("cs_mass_total_ge3", np.nan), errors="coerce")
    le2 = pd.to_numeric(sub.get("cs_mass_total_le2", np.nan), errors="coerce")
    trap11 = pd.to_numeric(sub.get("cs_mass_1_1", np.nan), errors="coerce")
    blank_low_total = pd.to_numeric(sub.get("cs_mass_0_0_1_0_0_1", np.nan), errors="coerce")

    if direction == "OVER25":
        combos = [
            (
                "OU25_OVER_MUTUAL_ENVIRONMENT",
                "all_rules",
                "cs_mass_total_ge3 >= 0.20 and cs_mass_total_le2 <= 0.20",
                (ge3 >= 0.20) & (le2 <= 0.20),
            ),
            (
                "OU25_OVER_TOTAL_GE3_STRICT",
                "all_rules",
                "cs_mass_total_ge3 >= 0.30 and cs_mass_total_le2 <= 0.20",
                (ge3 >= 0.30) & (le2 <= 0.20),
            ),
            (
                "OU25_OVER_ANY_PRIMARY_PLUS_CS",
                "any_primary",
                "ANY(cs_mass_total_ge3 >= 0.20) with SUPPORT(cs_mass_total_le2 <= 0.20)",
                (ge3 >= 0.20) & (le2 <= 0.20),
            ),
        ]
    elif direction == "UNDER25":
        low_total_primary = (le2 >= 0.22)
        trap_primary = (trap11 >= 0.04)
        blank_primary = (blank_low_total >= 0.10)
        compressed_support = (ge3 <= 0.28)

        combos = [
            (
                "OU25_UNDER_COMPRESSED_CORE",
                "all_rules",
                "cs_mass_total_le2 >= 0.22 and cs_mass_total_ge3 <= 0.28",
                low_total_primary & compressed_support,
            ),
            (
                "OU25_UNDER_11_TRAP",
                "all_rules",
                "cs_mass_total_le2 >= 0.22 and cs_mass_1_1 >= 0.04",
                low_total_primary & trap_primary,
            ),
            (
                "OU25_UNDER_BLANK_LOW_TOTAL",
                "all_rules",
                "cs_mass_0_0_1_0_0_1 >= 0.10 and cs_mass_total_le2 >= 0.22",
                blank_primary & low_total_primary,
            ),
            (
                "OU25_UNDER_DOUBLE_COMPRESSION",
                "all_rules",
                "cs_mass_total_le2 >= 0.22 and cs_mass_total_ge3 <= 0.28 and cs_mass_0_0_1_0_0_1 >= 0.10",
                low_total_primary & compressed_support & blank_primary,
            ),
            (
                "OU25_UNDER_ANY_PRIMARY_PLUS_COMPRESSION",
                "any_primary",
                "ANY(cs_mass_total_le2 >= 0.22, cs_mass_1_1 >= 0.04, cs_mass_0_0_1_0_0_1 >= 0.10) with SUPPORT(cs_mass_total_ge3 <= 0.28)",
                (low_total_primary | trap_primary | blank_primary) & compressed_support,
            ),
        ]
    else:
        combos = [
            (
                "OU25_ALL_TOTAL_GE3_STRICT",
                "all_rules",
                "cs_mass_total_ge3 >= 0.30 and cs_mass_total_le2 <= 0.20",
                (ge3 >= 0.30) & (le2 <= 0.20),
            ),
            (
                "OU25_ALL_TOTAL_LE2_STRICT",
                "all_rules",
                "cs_mass_total_le2 >= 0.22 and cs_mass_1_1 >= 0.04",
                (le2 >= 0.22) & (trap11 >= 0.04),
            ),
        ]

    for combo_name, logic, rules, mask in combos:
        g = sub.loc[mask.fillna(False)].copy()
        rows.append(combo_summary(g, hit_col, combo_name, logic, rules))
    return rows


def build_btts_combo_rows(sub: pd.DataFrame, hit_col: str, direction: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    btts_yes_mass = pd.to_numeric(sub.get("cs_mass_btts_yes", np.nan), errors="coerce")
    clean_home = pd.to_numeric(sub.get("cs_mass_clean_sheet_home", np.nan), errors="coerce")
    clean_away = pd.to_numeric(sub.get("cs_mass_clean_sheet_away", np.nan), errors="coerce")
    p00 = pd.to_numeric(sub.get("cs_mass_any_0_0", np.nan), errors="coerce")
    btts_no_mass = pd.to_numeric(sub.get("cs_mass_btts_no", np.nan), errors="coerce")
    btts_yes_ratio = pd.to_numeric(sub.get("cs_mass_btts_yes_ratio", np.nan), errors="coerce")

    clean_sheet_shadow = pd.concat([clean_home, clean_away], axis=1).max(axis=1)
    double_clean_sheet_shadow = pd.concat([clean_home, clean_away], axis=1).min(axis=1)

    if direction == "YES":
        combos = [
            (
                "BTTS_YES_CS_VETO_COMPRESSION",
                "all_rules",
                "cs_mass_clean_sheet_home <= 0.20 and cs_mass_any_0_0 <= 0.10",
                (clean_home <= 0.20) & (p00 <= 0.10),
            ),
            (
                "BTTS_YES_MUTUAL_SCORE_CORE",
                "all_rules",
                "cs_mass_btts_yes >= 0.30 and cs_mass_any_0_0 <= 0.10",
                (btts_yes_mass >= 0.30) & (p00 <= 0.10),
            ),
            (
                "BTTS_YES_MUTUAL_SCORE_STRICT",
                "all_rules",
                "cs_mass_btts_yes >= 0.30 and cs_mass_clean_sheet_home <= 0.20 and cs_mass_any_0_0 <= 0.10",
                (btts_yes_mass >= 0.30) & (clean_home <= 0.20) & (p00 <= 0.10),
            ),
        ]
    elif direction == "NO":
        nil_nil_primary = (p00 >= 0.03)
        clean_sheet_primary = (clean_sheet_shadow >= 0.15)
        low_mutual_primary = (btts_yes_mass <= 0.28)
        no_mass_primary = (btts_no_mass >= 0.20)
        asymmetric_support = (btts_yes_ratio <= 1.20)

        combos = [
            (
                "BTTS_NO_NIL_NIL_BLANK_CORE",
                "all_rules",
                "cs_mass_any_0_0 >= 0.03 and max(clean_sheet_home, clean_sheet_away) >= 0.15",
                nil_nil_primary & clean_sheet_primary,
            ),
            (
                "BTTS_NO_DOUBLE_CLEAN_SHEET_SHADOW",
                "all_rules",
                "cs_mass_clean_sheet_home >= 0.12 and cs_mass_clean_sheet_away >= 0.08",
                (clean_home >= 0.12) & (clean_away >= 0.08),
            ),
            (
                "BTTS_NO_LOW_MUTUAL_SCORE",
                "all_rules",
                "cs_mass_btts_yes <= 0.28 and (cs_mass_any_0_0 >= 0.03 or max(clean_sheet_home, clean_sheet_away) >= 0.15)",
                low_mutual_primary & (nil_nil_primary | clean_sheet_primary),
            ),
            (
                "BTTS_NO_ASYMMETRIC_BLANK",
                "all_rules",
                "cs_mass_btts_no >= 0.20 and cs_mass_btts_yes_ratio <= 1.20",
                no_mass_primary & asymmetric_support,
            ),
            (
                "BTTS_NO_ANY_PRIMARY_PLUS_SUPPORT",
                "any_primary",
                "ANY(cs_mass_any_0_0 >= 0.03, max(clean_sheet_home, clean_sheet_away) >= 0.15, cs_mass_btts_yes <= 0.28, cs_mass_btts_no >= 0.20) with SUPPORT(cs_mass_btts_yes_ratio <= 1.20 or min(clean_sheet_home, clean_sheet_away) >= 0.08)",
                (nil_nil_primary | clean_sheet_primary | low_mutual_primary | no_mass_primary)
                & (asymmetric_support | (double_clean_sheet_shadow >= 0.08)),
            ),
        ]
    else:
        combos = [
            (
                "BTTS_ALL_NIL_NIL_COMPRESSION",
                "all_rules",
                "cs_mass_any_0_0 <= 0.10 and cs_mass_clean_sheet_home <= 0.20",
                (p00 <= 0.10) & (clean_home <= 0.20),
            ),
            (
                "BTTS_ALL_MUTUAL_SCORE_CORE",
                "all_rules",
                "cs_mass_btts_yes >= 0.30 and cs_mass_any_0_0 <= 0.10",
                (btts_yes_mass >= 0.30) & (p00 <= 0.10),
            ),
        ]

    for combo_name, logic, rules, mask in combos:
        g = sub.loc[mask.fillna(False)].copy()
        rows.append(combo_summary(g, hit_col, combo_name, logic, rules))
    return rows

def build_ftr_audits(df: pd.DataFrame, outdir: Path) -> None:
    base = df.loc[df["market"].eq("ftr")].copy()
    if base.empty:
        return

    spec = CS_AUDIT_SPECS["ftr"]
    for direction in spec["direction_views"]:
        sub, hit_col = build_direction_frame(base, "ftr", direction)
        if sub.empty:
            continue

        prefix = spec["short_direction_labels"][direction]
        log(f"[cs-signal-audit] graded_{prefix}={int(pd.to_numeric(sub.get(hit_col, np.nan), errors='coerce').notna().sum())}")
        write_market_combined_csv(sub, outdir / f"{prefix}__COMBINED.csv")
        grouped_summary(sub, ["league"], hit_col).to_csv(outdir / f"{prefix}__BY_LEAGUE.csv", index=False)
        grouped_summary(sub, ["source_tier_file"], hit_col).to_csv(outdir / f"{prefix}__BY_TIER.csv", index=False)
        grouped_summary(sub, ["pick_side_margin_band"], hit_col).to_csv(outdir / f"{prefix}__BY_PICK_SIDE_MARGIN_BAND.csv", index=False)
        grouped_summary(sub, ["draw_top3_mass_band"], hit_col).to_csv(outdir / f"{prefix}__BY_DRAW_TOP3_MASS_BAND.csv", index=False)

        sweeps: list[dict[str, object]] = []
        margin = pd.to_numeric(sub.get("pick_side_margin_top3", np.nan), errors="coerce")
        draw_mass = pd.to_numeric(sub.get("draw_top3_mass", np.nan), errors="coerce")
        side_mass = pd.to_numeric(sub.get("pick_side_mass_top3", np.nan), errors="coerce")

        for margin_floor in [0.02, 0.05, 0.10, 0.15, 0.20]:
            g = sub.loc[margin >= margin_floor].copy()
            sweeps.append({
                "filter_family": "pick_side_margin_floor",
                "filter_value": f">={margin_floor:.2f}",
                **summarize_slice(g, hit_col),
            })

        for draw_cap in [0.05, 0.10, 0.15, 0.20, 0.25]:
            g = sub.loc[draw_mass <= draw_cap].copy()
            sweeps.append({
                "filter_family": "draw_top3_mass_cap",
                "filter_value": f"<={draw_cap:.2f}",
                **summarize_slice(g, hit_col),
            })

        for side_floor in [0.40, 0.50, 0.60, 0.70]:
            g = sub.loc[side_mass >= side_floor].copy()
            sweeps.append({
                "filter_family": "pick_side_mass_floor",
                "filter_value": f">={side_floor:.2f}",
                **summarize_slice(g, hit_col),
            })

        if sweeps:
            sweeps_df = (
                pd.DataFrame(sweeps)
                .sort_values(["hit_rate", "graded", "rows"], ascending=[False, False, False], na_position="last")
                .reset_index(drop=True)
            )
            sweeps_df = add_balanced_score(sweeps_df)
            sweeps_df.to_csv(outdir / f"{prefix}__SHORTLIST_SWEEPS.csv", index=False)
            export_ranked_views(sweeps_df, prefix, "TOP_FEATURE_SWEEPS", outdir)

        combo_rows = build_ftr_combo_rows(sub, hit_col)
        combo_df = pd.DataFrame(combo_rows)
        if combo_df.empty:
            combo_df = pd.DataFrame(columns=[
                "combo_name", "logic", "rules", "market", "feature_family", "feature",
                "rows", "graded", "wins", "losses", "hit_rate", "avg_feature_value",
                "avg_bookie_od", "avg_model_p_for_bookie", "balanced_score",
            ])
        else:
            combo_df["market"] = "ftr"
            combo_df["feature_family"] = "cs_derived"
            combo_df["feature"] = combo_df["combo_name"]
            combo_df = combo_df[[
                "combo_name", "logic", "rules", "market", "feature_family", "feature",
                "rows", "graded", "wins", "losses", "hit_rate", "avg_feature_value",
                "avg_bookie_od", "avg_model_p_for_bookie",
            ]]
            combo_df = add_balanced_score(combo_df)
            combo_df = combo_df.sort_values(
                ["balanced_score", "hit_rate", "graded", "wins"],
                ascending=[False, False, False, False],
                na_position="last",
            ).reset_index(drop=True)

        combo_df.to_csv(outdir / f"{prefix}__COMBO_SWEEPS.csv", index=False)
        export_ranked_views(combo_df, prefix, "TOP_COMBOS", outdir)
        log(f"[cs-signal-audit] combo_rows_{prefix}={len(combo_df)}")


def build_ou25_audits(df: pd.DataFrame, outdir: Path) -> None:
    base = df.loc[df["market"].eq("ou25")].copy()
    if base.empty:
        return

    spec = CS_AUDIT_SPECS["ou25"]
    for direction in spec["direction_views"]:
        sub, hit_col = build_direction_frame(base, "ou25", direction)
        if sub.empty:
            continue

        prefix = spec["short_direction_labels"][direction]
        log(f"[cs-signal-audit] graded_{prefix}={int(pd.to_numeric(sub.get(hit_col, np.nan), errors='coerce').notna().sum())}")
        write_market_combined_csv(sub, outdir / f"{prefix}__COMBINED.csv")
        grouped_summary(sub, ["league"], hit_col).to_csv(outdir / f"{prefix}__BY_LEAGUE.csv", index=False)
        grouped_summary(sub, ["total_mass_band"], hit_col).to_csv(outdir / f"{prefix}__BY_TOTAL_MASS_BAND.csv", index=False)
        grouped_summary(sub, ["is_11_trap"], hit_col).to_csv(outdir / f"{prefix}__1_1_TRAP_AUDIT.csv", index=False)
        grouped_summary(sub, ["source_tier_file"], hit_col).to_csv(outdir / f"{prefix}__BY_TIER.csv", index=False)

        sweeps: list[dict[str, object]] = []
        ge3 = pd.to_numeric(sub.get("cs_mass_total_ge3", np.nan), errors="coerce")
        le2 = pd.to_numeric(sub.get("cs_mass_total_le2", np.nan), errors="coerce")
        trap11 = pd.to_numeric(sub.get("cs_mass_1_1", np.nan), errors="coerce")

        for floor in [0.40, 0.50, 0.60, 0.70]:
            g = sub.loc[ge3 >= floor].copy()
            sweeps.append({
                "filter_family": "cs_mass_total_ge3_floor",
                "filter_value": f">={floor:.2f}",
                **summarize_slice(g, hit_col),
            })

        for floor in [0.35, 0.40, 0.50, 0.60, 0.70]:
            g = sub.loc[le2 >= floor].copy()
            sweeps.append({
                "filter_family": "cs_mass_total_le2_floor",
                "filter_value": f">={floor:.2f}",
                **summarize_slice(g, hit_col),
            })

        for cap in [0.05, 0.08, 0.10, 0.12, 0.15]:
            g = sub.loc[trap11 <= cap].copy()
            sweeps.append({
                "filter_family": "cs_mass_1_1_cap",
                "filter_value": f"<={cap:.2f}",
                **summarize_slice(g, hit_col),
            })

        if sweeps:
            sweeps_df = (
                pd.DataFrame(sweeps)
                .sort_values(["hit_rate", "graded", "rows"], ascending=[False, False, False], na_position="last")
                .reset_index(drop=True)
            )
            sweeps_df = add_balanced_score(sweeps_df)
            sweeps_df.to_csv(outdir / f"{prefix}__SHORTLIST_SWEEPS.csv", index=False)
            export_ranked_views(sweeps_df, prefix, "TOP_FEATURE_SWEEPS", outdir)

        combo_rows = build_ou25_combo_rows(sub, hit_col, direction)
        combo_df = pd.DataFrame(combo_rows)
        if combo_df.empty:
            combo_df = pd.DataFrame(columns=[
                "combo_name", "logic", "rules", "market", "feature_family", "feature",
                "rows", "graded", "wins", "losses", "hit_rate", "avg_feature_value",
                "avg_bookie_od", "avg_model_p_for_bookie", "balanced_score",
            ])
        else:
            combo_df["market"] = "ou25"
            combo_df["feature_family"] = "cs_derived"
            combo_df["feature"] = combo_df["combo_name"]
            combo_df = combo_df[[
                "combo_name", "logic", "rules", "market", "feature_family", "feature",
                "rows", "graded", "wins", "losses", "hit_rate", "avg_feature_value",
                "avg_bookie_od", "avg_model_p_for_bookie",
            ]]
            combo_df = add_balanced_score(combo_df)
            combo_df = combo_df.sort_values(
                ["balanced_score", "hit_rate", "graded", "wins"],
                ascending=[False, False, False, False],
                na_position="last",
            ).reset_index(drop=True)

        combo_df.to_csv(outdir / f"{prefix}__COMBO_SWEEPS.csv", index=False)
        export_ranked_views(combo_df, prefix, "TOP_COMBOS", outdir)
        log(f"[cs-signal-audit] combo_rows_{prefix}={len(combo_df)}")


def build_btts_audits(df: pd.DataFrame, outdir: Path) -> None:
    base = df.loc[df["market"].eq("btts")].copy()
    if base.empty:
        return

    spec = CS_AUDIT_SPECS["btts"]
    for direction in spec["direction_views"]:
        sub, hit_col = build_direction_frame(base, "btts", direction)
        if sub.empty:
            continue

        prefix = spec["short_direction_labels"][direction]
        log(f"[cs-signal-audit] graded_{prefix}={int(pd.to_numeric(sub.get(hit_col, np.nan), errors='coerce').notna().sum())}")
        write_market_combined_csv(sub, outdir / f"{prefix}__COMBINED.csv")
        grouped_summary(sub, ["league"], hit_col).to_csv(outdir / f"{prefix}__BY_LEAGUE.csv", index=False)
        grouped_summary(sub, ["mutual_score_mass_band"], hit_col).to_csv(outdir / f"{prefix}__BY_MUTUAL_SCORE_MASS_BAND.csv", index=False)
        grouped_summary(sub, ["clean_sheet_mass_band"], hit_col).to_csv(outdir / f"{prefix}__BY_CLEAN_SHEET_MASS_BAND.csv", index=False)
        grouped_summary(sub, ["source_tier_file"], hit_col).to_csv(outdir / f"{prefix}__BY_TIER.csv", index=False)

        sweeps: list[dict[str, object]] = []
        btts_yes_mass = pd.to_numeric(sub.get("cs_mass_btts_yes", np.nan), errors="coerce")
        clean_sheet = pd.concat([
            pd.to_numeric(sub.get("cs_mass_clean_sheet_home", np.nan), errors="coerce"),
            pd.to_numeric(sub.get("cs_mass_clean_sheet_away", np.nan), errors="coerce"),
        ], axis=1).max(axis=1)
        ratio = pd.to_numeric(sub.get("cs_mass_btts_yes_ratio", np.nan), errors="coerce")
        p00 = pd.to_numeric(sub.get("cs_mass_any_0_0", np.nan), errors="coerce")

        for floor in [0.30, 0.40, 0.50, 0.60]:
            g = sub.loc[btts_yes_mass >= floor].copy()
            sweeps.append({
                "filter_family": "cs_mass_btts_yes_floor",
                "filter_value": f">={floor:.2f}",
                **summarize_slice(g, hit_col),
            })

        for cap in [0.20, 0.30, 0.40, 0.50]:
            g = sub.loc[clean_sheet <= cap].copy()
            sweeps.append({
                "filter_family": "clean_sheet_mass_cap",
                "filter_value": f"<={cap:.2f}",
                **summarize_slice(g, hit_col),
            })

        for floor in [1.0, 1.2, 1.5, 2.0]:
            g = sub.loc[ratio >= floor].copy()
            sweeps.append({
                "filter_family": "btts_yes_ratio_floor",
                "filter_value": f">={floor:.2f}",
                **summarize_slice(g, hit_col),
            })

        for cap in [0.03, 0.05, 0.08, 0.10]:
            g = sub.loc[p00 <= cap].copy()
            sweeps.append({
                "filter_family": "cs_mass_any_0_0_cap",
                "filter_value": f"<={cap:.2f}",
                **summarize_slice(g, hit_col),
            })

        if sweeps:
            sweeps_df = (
                pd.DataFrame(sweeps)
                .sort_values(["hit_rate", "graded", "rows"], ascending=[False, False, False], na_position="last")
                .reset_index(drop=True)
            )
            sweeps_df = add_balanced_score(sweeps_df)
            sweeps_df.to_csv(outdir / f"{prefix}__SHORTLIST_SWEEPS.csv", index=False)
            export_ranked_views(sweeps_df, prefix, "TOP_FEATURE_SWEEPS", outdir)    

        combo_rows = build_btts_combo_rows(sub, hit_col, direction)
        combo_df = pd.DataFrame(combo_rows)
        if combo_df.empty:
            combo_df = pd.DataFrame(columns=[
                "combo_name", "logic", "rules", "market", "feature_family", "feature",
                "rows", "graded", "wins", "losses", "hit_rate", "avg_feature_value",
                "avg_bookie_od", "avg_model_p_for_bookie", "balanced_score",
            ])
        else:
            combo_df["market"] = "btts"
            combo_df["feature_family"] = "cs_derived"
            combo_df["feature"] = combo_df["combo_name"]
            combo_df = combo_df[[
                "combo_name", "logic", "rules", "market", "feature_family", "feature",
                "rows", "graded", "wins", "losses", "hit_rate", "avg_feature_value",
                "avg_bookie_od", "avg_model_p_for_bookie",
            ]]
            combo_df = add_balanced_score(combo_df)
            combo_df = combo_df.sort_values(
                ["balanced_score", "hit_rate", "graded", "wins"],
                ascending=[False, False, False, False],
                na_position="last",
            ).reset_index(drop=True)

        combo_df.to_csv(outdir / f"{prefix}__COMBO_SWEEPS.csv", index=False)
        export_ranked_views(combo_df, prefix, "TOP_COMBOS", outdir)
        log(f"[cs-signal-audit] combo_rows_{prefix}={len(combo_df)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit correct-score derived signal families from already-scored weekend portfolio outputs.")
    ap.add_argument("--base-outdir", default=str(DEFAULT_BASE_OUTDIR), help="Walk-forward root containing per-window folders.")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory for CS audit CSVs.")
    ap.add_argument("--windows", default="", help="Optional comma-separated window_ids subset.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_outdir = Path(args.base_outdir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_scored_frames(base_outdir, args.windows)
    if df.empty:
        log("[cs-signal-audit] no scored files found.")
        return

    log(f"[cs-signal-audit] loaded rows={len(df)} windows={df['window_id'].nunique()}")
    if "window_id" in df.columns:
        window_ids = sorted({str(x).strip() for x in df["window_id"].dropna().unique() if str(x).strip()})
        log(f"[cs-signal-audit] windows={','.join(window_ids)}")
    df = attach_cs_features(df)
    for market_key, spec in CS_AUDIT_SPECS.items():
        market_df = df.loc[safe_series_str(df, "market").str.lower().eq(market_key)].copy()
        for direction in spec["direction_views"]:
            direction_df, direction_hit_col = build_direction_frame(market_df, market_key, direction)
            graded_preview = int(pd.to_numeric(direction_df.get(direction_hit_col, np.nan), errors="coerce").notna().sum())
            log(f"[cs-signal-audit] preview_graded_{market_key}_{direction}={graded_preview}")
    write_market_combined_csv(df, outdir / "CS_SIGNAL_AUDIT__COMBINED.csv")

    build_ftr_audits(df, outdir)
    build_ou25_audits(df, outdir)
    build_btts_audits(df, outdir)
    for market_name in ["ftr", "ou25", "btts"]:
        market_rows = int(safe_series_str(df, "market").str.lower().eq(market_name).sum())
        log(f"[cs-signal-audit] market_rows_{market_name}={market_rows}")

    for col in [
        "draw_top3_mass",
        "cs_mass_total_ge3",
        "cs_mass_total_le2",
        "cs_mass_1_1",
        "cs_mass_btts_yes",
        "cs_mass_btts_no",
        "cs_mass_clean_sheet_home",
        "cs_mass_clean_sheet_away",
        "cs_mass_btts_yes_ratio",
        "cs_mass_any_0_0",
    ]:
        if col in df.columns:
            nonnull = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
            log(f"[cs-signal-audit] coverage {col}={nonnull}/{len(df)}")
    log(f"[cs-signal-audit] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()