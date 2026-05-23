#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

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


def _safe_read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _fixture_key(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in KEY_COLS if c in df.columns]
    if not cols:
        return pd.Series(dtype="string")
    return df[cols].astype(str).agg(" || ".join, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--months", required=True)
    ap.add_argument("--out-audit", required=True)
    ap.add_argument("--out-spot", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    months = [x.strip() for x in args.months.split(",") if x.strip()]

    audit_rows = []
    spot_rows = []

    for branch_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for month in months:
            bt = branch_dir / month / f"backtest_{month}.csv"
            fg = branch_dir / month / f"frozen_gated_{month}.csv"
            if not bt.exists() or not fg.exists():
                continue

            bt_df = _safe_read(bt)
            fg_df = _safe_read(fg)

            bt_keys = set(_fixture_key(bt_df))
            fg_keys = list(_fixture_key(fg_df))
            missing = sum(1 for k in fg_keys if k not in bt_keys)
            dupes = int(pd.Series(fg_keys).duplicated().sum()) if fg_keys else 0

            audit_rows.append(
                {
                    "month": month,
                    "branch": branch_dir.name,
                    "backtest_rows": int(len(bt_df)),
                    "filtered_rows": int(len(fg_df)),
                    "filtered_rows_missing_from_backtest": int(missing),
                    "duplicate_filtered_fixture_rows": dupes,
                    "postmatch_cols_present": ",".join([c for c in POSTMATCH_COLS if c in fg_df.columns]),
                }
            )

            if "correct" in fg_df.columns:
                winners = fg_df[fg_df["correct"] == 1].head(10).copy()
                losers = fg_df[fg_df["correct"] == 0].head(10).copy()
                for label, samp in [("winner", winners), ("loser", losers)]:
                    for _, r in samp.iterrows():
                        spot_rows.append(
                            {
                                "month": month,
                                "branch": branch_dir.name,
                                "sample_type": label,
                                "league": r.get("league"),
                                "match_date": r.get("match_date"),
                                "home_team_name": r.get("home_team_name"),
                                "away_team_name": r.get("away_team_name"),
                                "market": r.get("market"),
                                "bookie_pick": r.get("bookie_pick"),
                                "bookie_od": r.get("bookie_od"),
                                "model_p_for_bookie": r.get("model_p_for_bookie"),
                                "score": r.get("score"),
                                "correct": r.get("correct"),
                            }
                        )

    audit_df = pd.DataFrame(audit_rows).sort_values(["month", "branch"]).reset_index(drop=True)
    spot_df = pd.DataFrame(spot_rows).sort_values(["month", "branch", "sample_type"]).reset_index(drop=True)

    audit_df.to_csv(args.out_audit, index=False)
    spot_df.to_csv(args.out_spot, index=False)
    print(audit_df)
    print(f"WROTE: {args.out_audit}")
    print(f"WROTE: {args.out_spot}")


if __name__ == "__main__":
    main()

