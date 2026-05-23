#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

MONTHS = ["2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03"]

BRANCHES = {
    "accuracy": {
        "root": Path("walkforward_frozen_accuracy"),
        "tag": "FTR_ACCURACY",
    },
    "valueev_balanced": {
        "root": Path("walkforward_frozen_valueev_balanced"),
        "tag": "FTR_VALUEEV_BALANCED",
    },
    "valueev_aggressive": {
        "root": Path("walkforward_frozen_valueev_aggressive"),
        "tag": "FTR_VALUEEV_AGGRESSIVE",
    },
}

POSTMATCH_COLS = [
    "home_team_goal_count",
    "away_team_goal_count",
    "correct",
]

KEY_COLS = [
    "league",
    "match_date",
    "home_team_name",
    "away_team_name",
    "market",
    "bookie_pick",
]

VALUE_CHECK_COLS = [
    "bookie_od",
    "model_p_for_bookie",
    "p_home_pois",
    "p_draw_pois",
    "p_away_pois",
    "ftr_valueev_edge",
]

FILTERING_COLS_BY_BRANCH = {
    "accuracy": [
        "bookie_od",
        "bookie_implied_used",
        "ftr_margin",
        "bookie_pick",
    ],
    "valueev_balanced": [
        "bookie_od",
        "ftr_valueev_edge",
        "bookie_pick",
    ],
    "valueev_aggressive": [
        "bookie_od",
        "ftr_valueev_edge",
        "bookie_pick",
    ],
}

RANKING_COLS_BY_BRANCH = {
    "accuracy": ["score", "bookie_implied_used"],
    "valueev_balanced": ["ftr_valueev_edge"],
    "valueev_aggressive": ["ftr_valueev_edge"],
}

POSTMATCH_FILTER_RISK_COLS = [
    "correct",
    "home_team_goal_count",
    "away_team_goal_count",
]


def key_cols_present(df: pd.DataFrame) -> list[str]:
    return [c for c in KEY_COLS if c in df.columns]


def fixture_key_series(df: pd.DataFrame) -> pd.Series:
    cols = key_cols_present(df)
    if not cols:
        return pd.Series(dtype="string")
    return df[cols].astype(str).agg(" || ".join, axis=1)



def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)



def _filtered_path(branch_root: Path, month: str) -> Path:
    return branch_root / month / f"frozen_gated_{month}.csv"



def _backtest_path(branch_root: Path, month: str) -> Path:
    return branch_root / month / f"backtest_{month}.csv"



def _sample_rows(df: pd.DataFrame, n: int, *, random_state: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sample(min(n, len(df)), random_state=random_state)



def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df.copy()
    return df.loc[:, ~df.columns.duplicated()].copy()



def _numeric_or_nan(row: pd.Series, col: str):
    if col not in row.index:
        return pd.NA
    return pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]


def _present_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _selection_leak_suspected(selection_cols_present: list[str]) -> bool:
    return any(c in POSTMATCH_FILTER_RISK_COLS for c in selection_cols_present)



audit_rows: list[dict] = []
spot_rows: list[dict] = []

for branch, meta in BRANCHES.items():
    root = meta["root"]

    for month in MONTHS:
        backtest = _backtest_path(root, month)
        filtered = _filtered_path(root, month)

        if not backtest.exists():
            print(f"missing backtest: {backtest}")
            continue
        if not filtered.exists():
            print(f"missing filtered csv for {branch} {month}: {filtered}")
            continue

        bt = _dedupe_columns(_safe_read_csv(backtest))
        fg = _dedupe_columns(_safe_read_csv(filtered))

        bt_keys = list(fixture_key_series(bt))
        fg_keys = list(fixture_key_series(fg))

        bt_key_set = set(bt_keys)
        filtered_missing_from_backtest = sum(1 for k in fg_keys if k not in bt_key_set)
        duplicate_filtered_fixture_rows = int(pd.Series(fg_keys).duplicated().sum()) if fg_keys else 0

        poisson_cols_present = all(c in bt.columns for c in ["p_home_pois", "p_draw_pois", "p_away_pois"])
        postmatch_cols_present = [c for c in POSTMATCH_COLS if c in fg.columns]

        filtering_columns_checked = FILTERING_COLS_BY_BRANCH.get(branch, [])
        ranking_columns_checked = RANKING_COLS_BY_BRANCH.get(branch, [])

        filtering_columns_present_in_filtered = _present_cols(fg, filtering_columns_checked)
        ranking_columns_present_in_filtered = _present_cols(fg, ranking_columns_checked)

        selection_columns_checked = list(dict.fromkeys(filtering_columns_checked + ranking_columns_checked))
        selection_columns_present_in_filtered = _present_cols(fg, selection_columns_checked)
        postmatch_selection_columns_present = [c for c in selection_columns_present_in_filtered if c in POSTMATCH_FILTER_RISK_COLS]
        selection_leak_suspected = _selection_leak_suspected(selection_columns_present_in_filtered)
        postmatch_cols_present_but_not_used = [
            c for c in postmatch_cols_present if c not in selection_columns_present_in_filtered
        ]

        merge_keys = key_cols_present(fg)
        selection_mismatch_cols: list[str] = []
        duplicate_join_rows = None
        merged_rows = None
        merge_miss_rows = None
        status = "ok"

        if not merge_keys:
            status = "missing_merge_keys"
        else:
            bt_merge_cols = list(dict.fromkeys(merge_keys + [c for c in VALUE_CHECK_COLS if c in bt.columns]))
            fg_merge_cols = list(dict.fromkeys(merge_keys + [c for c in VALUE_CHECK_COLS if c in fg.columns]))

            bt_merge = bt.loc[:, bt_merge_cols].copy()
            fg_merge = fg.loc[:, fg_merge_cols].copy()

            bt_merge = bt_merge.drop_duplicates().copy()
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

            for col in VALUE_CHECK_COLS:
                fcol = f"{col}_filtered"
                bcol = f"{col}_backtest"
                if fcol in merged.columns and bcol in merged.columns:
                    left = pd.to_numeric(merged[fcol], errors="coerce")
                    right = pd.to_numeric(merged[bcol], errors="coerce")
                    mismatch = (left.notna() | right.notna()) & ~left.fillna(-999999).round(10).eq(right.fillna(-999999).round(10))
                    if bool(mismatch.any()):
                        selection_mismatch_cols.append(col)

        audit_rows.append(
            {
                "month": month,
                "branch": branch,
                "status": status,
                "backtest_rows": int(len(bt)),
                "filtered_rows": int(len(fg)),
                "merged_rows": merged_rows,
                "merge_miss_rows": merge_miss_rows,
                "duplicate_join_rows": duplicate_join_rows,
                "duplicate_filtered_fixture_rows": duplicate_filtered_fixture_rows,
                "filtered_rows_missing_from_backtest": int(filtered_missing_from_backtest),
                "selection_mismatch_cols": ",".join(selection_mismatch_cols),
                "filtering_columns_checked": ",".join(filtering_columns_checked),
                "ranking_columns_checked": ",".join(ranking_columns_checked),
                "filtering_columns_present_in_filtered": ",".join(filtering_columns_present_in_filtered),
                "ranking_columns_present_in_filtered": ",".join(ranking_columns_present_in_filtered),
                "selection_columns_present_in_filtered": ",".join(selection_columns_present_in_filtered),
                "selection_leak_suspected": bool(selection_leak_suspected),
                "postmatch_selection_columns_present": ",".join(postmatch_selection_columns_present),
                "postmatch_cols_present_but_not_used": ",".join(postmatch_cols_present_but_not_used),
                "poisson_cols_present": bool(poisson_cols_present),
                "postmatch_cols_present_in_filtered": ",".join(postmatch_cols_present),
            }
        )

        winners = fg[fg["correct"] == 1].copy() if "correct" in fg.columns else fg.iloc[0:0].copy()
        losers = fg[fg["correct"] == 0].copy() if "correct" in fg.columns else fg.iloc[0:0].copy()

        winners_sample = _sample_rows(winners, 10, random_state=42)
        losers_sample = _sample_rows(losers, 10, random_state=42)

        for label, samp in [("winner", winners_sample), ("loser", losers_sample)]:
            for _, r in samp.iterrows():
                spot_rows.append(
                    {
                        "month": month,
                        "branch": branch,
                        "sample_type": label,
                        "league": r.get("league"),
                        "match_date": r.get("match_date"),
                        "home_team_name": r.get("home_team_name"),
                        "away_team_name": r.get("away_team_name"),
                        "market": r.get("market"),
                        "bookie_pick": r.get("bookie_pick"),
                        "bookie_od": _numeric_or_nan(r, "bookie_od"),
                        "model_p_for_bookie": _numeric_or_nan(r, "model_p_for_bookie"),
                        "p_home_pois": _numeric_or_nan(r, "p_home_pois"),
                        "p_draw_pois": _numeric_or_nan(r, "p_draw_pois"),
                        "p_away_pois": _numeric_or_nan(r, "p_away_pois"),
                        "ftr_valueev_edge": _numeric_or_nan(r, "ftr_valueev_edge"),
                        "correct": _numeric_or_nan(r, "correct"),
                    }
                )


audit_df = pd.DataFrame(audit_rows).sort_values(["month", "branch"]).reset_index(drop=True)
spot_df = pd.DataFrame(spot_rows).sort_values(["month", "branch", "sample_type"]).reset_index(drop=True)

audit_out = Path("walkforward_forensic_audit.csv")
spot_out = Path("walkforward_fixture_spotcheck_samples.csv")

audit_df.to_csv(audit_out, index=False)
spot_df.to_csv(spot_out, index=False)

print(audit_df)
print(f"\nWROTE: {audit_out}")
print(f"WROTE: {spot_out}")