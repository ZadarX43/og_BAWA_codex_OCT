#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("predictions_output/ou25_frozen_compare/rulebook_ftr_validation_3yr_19lg_v1")
OUT_AUDIT = Path("ou25_forensic_audit.csv")
OUT_SPOT = Path("ou25_fixture_spotcheck_samples.csv")

KEY_COLS = [
    "league",
    "match_date",
    "home_team_name",
    "away_team_name",
    "market",
    "bookie_pick",
]

FULL_JOIN_KEYS = [
    "league",
    "match_date",
    "home_team_name",
    "away_team_name",
    "market",
    "bookie_pick",
]

NO_MATCH_DATE_JOIN_KEYS = [
    "league",
    "home_team_name",
    "away_team_name",
    "market",
    "bookie_pick",
]

MINIMAL_JOIN_KEYS = [
    "home_team_name",
    "away_team_name",
    "market",
    "bookie_pick",
]

POSTMATCH_COLS = [
    "home_team_goal_count",
    "away_team_goal_count",
    "correct",
]

FILTERING_CANDIDATES = [
    "bookie_od",
    "bookie_pick",
]

RANKING_CANDIDATES = [
    "score",
    "model_p_for_bookie",
]

SELECTION_CANDIDATES = [
    "bookie_od",
    "bookie_pick",
    "score",
    "model_p_for_bookie",
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df.copy()
    return df.loc[:, ~df.columns.duplicated()].copy()


def _fixture_key_series(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in KEY_COLS if c in df.columns]
    if not cols:
        return pd.Series(dtype="string")
    return df[cols].astype(str).agg(" || ".join, axis=1)


def _read_summary_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_rows(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sample(min(n, len(df)), random_state=random_state)


def _normalize_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_to_norm = {
        "league": "league_norm",
        "match_date": "match_date_norm",
        "home_team_name": "home_team_name_norm",
        "away_team_name": "away_team_name_norm",
        "market": "market_norm",
        "bookie_pick": "bookie_pick_norm",
    }

    for raw_col, norm_col in raw_to_norm.items():
        if raw_col not in out.columns:
            continue
        out[norm_col] = out[raw_col].astype(str).str.strip()
        if raw_col != "match_date":
            out[norm_col] = out[norm_col].str.lower()

    if "bookie_pick_norm" in out.columns:
        out["bookie_pick_norm"] = (
            out["bookie_pick_norm"]
            .str.replace("over25", "over", regex=False)
            .str.replace("under25", "under", regex=False)
        )

    if "match_date_norm" in out.columns:
        out["match_date_norm"] = out["match_date_norm"].str.replace(" 00:00:00", "", regex=False)
        out["match_date_norm"] = out["match_date_norm"].str.replace("T00:00:00", "", regex=False)
        out["match_date_norm"] = out["match_date_norm"].str.replace(r"\s+", " ", regex=True)

    return out


NORM_KEY_MAP = {
    "league": "league_norm",
    "match_date": "match_date_norm",
    "home_team_name": "home_team_name_norm",
    "away_team_name": "away_team_name_norm",
    "market": "market_norm",
    "bookie_pick": "bookie_pick_norm",
}

FULL_JOIN_KEYS_NORM = [NORM_KEY_MAP[c] for c in FULL_JOIN_KEYS]
NO_MATCH_DATE_JOIN_KEYS_NORM = [NORM_KEY_MAP[c] for c in NO_MATCH_DATE_JOIN_KEYS]
MINIMAL_JOIN_KEYS_NORM = [NORM_KEY_MAP[c] for c in MINIMAL_JOIN_KEYS]


def _join_match_count(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> int:
    usable = [c for c in keys if c in left.columns and c in right.columns]
    if not usable:
        return 0
    merged = left[usable].merge(
        right[usable].drop_duplicates(),
        on=usable,
        how="left",
        indicator=True,
    )
    return int((merged["_merge"] == "both").sum())


def _best_available_keys(left: pd.DataFrame, right: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [c for c in preferred if c in left.columns and c in right.columns]


def _build_join_probe(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left_cols = [
        c
        for c in [
            "league",
            "match_date",
            "home_team_name",
            "away_team_name",
            "market",
            "bookie_pick",
            "league_norm",
            "match_date_norm",
            "home_team_name_norm",
            "away_team_name_norm",
            "market_norm",
            "bookie_pick_norm",
        ]
        if c in left.columns
    ]
    right_cols = [c for c in left_cols if c in right.columns]
    usable = [
        c
        for c in [
            "league_norm",
            "match_date_norm",
            "home_team_name_norm",
            "away_team_name_norm",
            "market_norm",
            "bookie_pick_norm",
        ]
        if c in left.columns and c in right.columns
    ]
    if not usable:
        return pd.DataFrame()

    probe = left[left_cols].merge(
        right[right_cols].drop_duplicates(),
        on=usable,
        how="left",
        indicator=True,
        suffixes=("_filtered", "_backtest"),
    )
    return probe


audit_rows: list[dict[str, Any]] = []
spot_rows: list[dict[str, Any]] = []

if not ROOT.exists():
    raise SystemExit(f"OU25 sweep root not found: {ROOT}")

for dataset_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
    backtest_csv = Path("predictions_output/backtests/19l_3y_IMP40") / f"{dataset_dir.name}.csv"
    if not backtest_csv.exists():
        # fallback to exact expected canonical file for current run layout
        backtest_csv = Path("predictions_output/backtests/19l_3y_IMP40/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST.csv")

    if not backtest_csv.exists():
        print(f"missing canonical backtest for dataset: {dataset_dir.name}")
        continue

    bt = _normalize_key_cols(_dedupe_columns(_safe_read_csv(backtest_csv)))
    bt = bt[bt["market"].astype(str).str.lower() == "ou25"].copy()

    for branch_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        summary_files = sorted(branch_dir.glob("*__SUMMARY.json"))
        filtered_files = sorted([p for p in branch_dir.glob("*.csv") if "__MARKET_SUMMARY" not in p.name and "__LEAGUE_SUMMARY" not in p.name and "__TIER_" not in p.name])

        if not summary_files:
            print(f"missing summary json in {branch_dir}")
            continue
        if not filtered_files:
            print(f"missing filtered csv in {branch_dir}")
            continue

        summary_path = summary_files[0]
        filtered_path = filtered_files[0]

        fg = _normalize_key_cols(_dedupe_columns(_safe_read_csv(filtered_path)))
        fg = fg[fg["market"].astype(str).str.lower() == "ou25"].copy()

        bt_keys = list(_fixture_key_series(bt))
        fg_keys = list(_fixture_key_series(fg))
        bt_key_set = set(bt_keys)

        filtered_missing_from_backtest = sum(1 for k in fg_keys if k not in bt_key_set)
        duplicate_filtered_fixture_rows = int(pd.Series(fg_keys).duplicated().sum()) if fg_keys else 0

        merged_rows = None
        merge_miss_rows = None
        duplicate_join_rows = None
        status = "ok"
        join_stage_used = None

        full_match_count = _join_match_count(fg, bt, FULL_JOIN_KEYS_NORM)
        no_match_date_match_count = _join_match_count(fg, bt, NO_MATCH_DATE_JOIN_KEYS_NORM)
        minimal_match_count = _join_match_count(fg, bt, MINIMAL_JOIN_KEYS_NORM)
        join_probe = _build_join_probe(fg, bt)

        match_date_join_miss_suspected = False
        team_name_join_miss_suspected = False
        bookie_pick_join_miss_suspected = False

        if full_match_count > 0:
            join_stage_used = "full"
            merge_keys = _best_available_keys(fg, bt, FULL_JOIN_KEYS_NORM)
        elif no_match_date_match_count > 0:
            join_stage_used = "no_match_date"
            merge_keys = _best_available_keys(fg, bt, NO_MATCH_DATE_JOIN_KEYS_NORM)
            match_date_join_miss_suspected = True
        elif minimal_match_count > 0:
            join_stage_used = "minimal"
            merge_keys = _best_available_keys(fg, bt, MINIMAL_JOIN_KEYS_NORM)
            match_date_join_miss_suspected = True
            team_name_join_miss_suspected = True
        else:
            join_stage_used = "none"
            merge_keys = []
            match_date_join_miss_suspected = True
            team_name_join_miss_suspected = True
            bookie_pick_join_miss_suspected = True

        if join_stage_used == "none" and not join_probe.empty:
            if "bookie_pick_backtest" in join_probe.columns:
                bookie_pick_join_miss_suspected = bool(join_probe["bookie_pick_backtest"].notna().any())
            if "home_team_name_backtest" in join_probe.columns or "away_team_name_backtest" in join_probe.columns:
                team_name_join_miss_suspected = not bool(join_probe["_merge"].eq("both").any())

        if not merge_keys:
            status = "merge_miss"
            merged_rows = int(len(fg))
            merge_miss_rows = int(len(fg))
            duplicate_join_rows = 0
        else:
            bt_merge_cols = list(dict.fromkeys(merge_keys + [c for c in SELECTION_CANDIDATES if c in bt.columns]))
            fg_merge_cols = list(dict.fromkeys(merge_keys + [c for c in SELECTION_CANDIDATES if c in fg.columns]))

            bt_merge = bt.loc[:, bt_merge_cols].drop_duplicates().copy()
            fg_merge = fg.loc[:, fg_merge_cols].copy()

            merged = fg_merge.merge(
                bt_merge,
                on=merge_keys,
                how="left",
                indicator=True,
                suffixes=("_filtered", "_backtest"),
            )
            merged_rows = int(len(merged))
            duplicate_join_rows = int(max(len(merged) - len(fg_merge), 0))
            merge_miss_rows = int((merged["_merge"] != "both").sum())

            if merge_miss_rows > 0:
                status = "merge_miss"

        filtering_present = [c for c in FILTERING_CANDIDATES if c in fg.columns]
        ranking_present = [c for c in RANKING_CANDIDATES if c in fg.columns]
        selection_present = [c for c in SELECTION_CANDIDATES if c in fg.columns]
        postmatch_present = [c for c in POSTMATCH_COLS if c in fg.columns]
        postmatch_selection_cols = [c for c in selection_present if c in POSTMATCH_COLS]
        postmatch_present_but_not_used = [c for c in postmatch_present if c not in selection_present]

        audit_rows.append(
            {
                "dataset": dataset_dir.name,
                "branch": branch_dir.name,
                "status": status,
                "join_stage_used": join_stage_used,
                "full_match_count": int(full_match_count),
                "no_match_date_match_count": int(no_match_date_match_count),
                "minimal_match_count": int(minimal_match_count),
                "match_date_join_miss_suspected": bool(match_date_join_miss_suspected),
                "team_name_join_miss_suspected": bool(team_name_join_miss_suspected),
                "bookie_pick_join_miss_suspected": bool(bookie_pick_join_miss_suspected),
                "join_probe_rows": int(len(join_probe)) if not join_probe.empty else 0,
                "join_probe_any_match": bool(join_probe["_merge"].eq("both").any()) if not join_probe.empty else False,
                "full_join_keys_used": ",".join(FULL_JOIN_KEYS_NORM),
                "no_match_date_join_keys_used": ",".join(NO_MATCH_DATE_JOIN_KEYS_NORM),
                "minimal_join_keys_used": ",".join(MINIMAL_JOIN_KEYS_NORM),
                "backtest_rows_ou25": int(len(bt)),
                "filtered_rows": int(len(fg)),
                "merged_rows": merged_rows,
                "merge_miss_rows": merge_miss_rows,
                "duplicate_join_rows": duplicate_join_rows,
                "duplicate_filtered_fixture_rows": duplicate_filtered_fixture_rows,
                "filtered_rows_missing_from_backtest": int(filtered_missing_from_backtest),
                "filtering_columns_checked": ",".join(FILTERING_CANDIDATES),
                "ranking_columns_checked": ",".join(RANKING_CANDIDATES),
                "filtering_columns_present_in_filtered": ",".join(filtering_present),
                "ranking_columns_present_in_filtered": ",".join(ranking_present),
                "selection_columns_present_in_filtered": ",".join(selection_present),
                "selection_leak_suspected": bool(len(postmatch_selection_cols) > 0),
                "postmatch_selection_columns_present": ",".join(postmatch_selection_cols),
                "postmatch_cols_present_but_not_used": ",".join(postmatch_present_but_not_used),
                "postmatch_cols_present_in_filtered": ",".join(postmatch_present),
            }
        )

        winners = fg[pd.to_numeric(fg.get("correct"), errors="coerce") == 1].copy() if "correct" in fg.columns else fg.iloc[0:0].copy()
        losers = fg[pd.to_numeric(fg.get("correct"), errors="coerce") == 0].copy() if "correct" in fg.columns else fg.iloc[0:0].copy()

        for sample_type, sample_df in [("winner", _sample_rows(winners, 10)), ("loser", _sample_rows(losers, 10))]:
            for _, r in sample_df.iterrows():
                spot_rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "branch": branch_dir.name,
                        "sample_type": sample_type,
                        "league": r.get("league"),
                        "match_date": r.get("match_date"),
                        "home_team_name": r.get("home_team_name"),
                        "away_team_name": r.get("away_team_name"),
                        "market": r.get("market"),
                        "bookie_pick": r.get("bookie_pick"),
                        "league_norm": r.get("league_norm"),
                        "match_date_norm": r.get("match_date_norm"),
                        "home_team_name_norm": r.get("home_team_name_norm"),
                        "away_team_name_norm": r.get("away_team_name_norm"),
                        "market_norm": r.get("market_norm"),
                        "bookie_pick_norm": r.get("bookie_pick_norm"),
                        "bookie_od": r.get("bookie_od"),
                        "score": r.get("score"),
                        "model_p_for_bookie": r.get("model_p_for_bookie"),
                        "correct": r.get("correct"),
                    }
                )

audit_df = pd.DataFrame(audit_rows).sort_values(["status", "filtered_rows", "branch"], ascending=[True, False, True]).reset_index(drop=True)
spot_df = pd.DataFrame(spot_rows).sort_values(["branch", "sample_type", "league"]).reset_index(drop=True)

audit_df.to_csv(OUT_AUDIT, index=False)
spot_df.to_csv(OUT_SPOT, index=False)

print(audit_df.to_string(index=False))
print(f"\nWROTE: {OUT_AUDIT}")
print(f"WROTE: {OUT_SPOT}")