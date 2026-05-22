from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import brier_score_loss

from common import REPORTS_DIR


SOURCE_DATE_COLUMNS = [
    "opp_allowed_source_max_date",
    "minutes_source_max_date",
    "player_history_source_max_date",
]


def _fit_proxy_brier(df: pd.DataFrame, shuffled: bool = False) -> float:
    work = df.copy()
    work = work.dropna(subset=["match_date", "actual_tackles", "expected_minutes_proof"]).copy()
    if work.empty:
        return float("nan")
    work["match_date"] = pd.to_datetime(work["match_date"], errors="coerce")
    work = work.sort_values("match_date").reset_index(drop=True)
    if shuffled:
        work["match_date"] = np.random.default_rng(42).permutation(work["match_date"].to_numpy())
        work = work.sort_values("match_date").reset_index(drop=True)

    split_idx = int(len(work) * 0.8)
    train = work.iloc[:split_idx].copy()
    test = work.iloc[split_idx:].copy()
    if train.empty or test.empty:
        return float("nan")

    feat_cols = [
        "tackles_per90",
        "interceptions_per90",
        "duels_total_per90",
        "formation_pressure_score",
        "fixture_midfield_grind_score",
        "opp_tackles_allowed_pos_l10",
        "expected_minutes_proof",
    ]
    for col in feat_cols:
        train[col] = pd.to_numeric(train.get(col), errors="coerce").fillna(0.0)
        test[col] = pd.to_numeric(test.get(col), errors="coerce").fillna(0.0)
    exposure_train = (pd.to_numeric(train["expected_minutes_proof"], errors="coerce").fillna(0.0) / 90.0).clip(lower=0.1)
    exposure_test = (pd.to_numeric(test["expected_minutes_proof"], errors="coerce").fillna(0.0) / 90.0).clip(lower=0.1)
    y_train = pd.to_numeric(train["actual_tackles"], errors="coerce").fillna(0.0)
    y_test_hit = pd.to_numeric(test["actual_tackles"], errors="coerce").fillna(0.0).ge(2).astype(int)
    rate_train = y_train / exposure_train

    model = PoissonRegressor(alpha=0.01, max_iter=300)
    model.fit(train[feat_cols], rate_train, sample_weight=exposure_train)
    mu = model.predict(test[feat_cols]) * exposure_test
    p_ge2 = 1.0 - np.exp(-mu) * (1.0 + mu)
    return float(brier_score_loss(y_test_hit, np.clip(p_ge2, 1e-6, 1 - 1e-6)))


def run_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | str]]:
    work = df.copy()
    work["match_date"] = pd.to_datetime(work["match_date"], errors="coerce")

    rows = []
    for col in SOURCE_DATE_COLUMNS:
        if col not in work.columns:
            rows.append({"check_name": col, "status": "MISSING", "fail_rows": 0, "notes": "column not present"})
            continue
        src = pd.to_datetime(work[col], errors="coerce")
        fail = int(((src.notna()) & (src >= work["match_date"])).sum())
        rows.append({"check_name": col, "status": "PASS" if fail == 0 else "FAIL", "fail_rows": fail, "notes": "source date must be strictly before match date"})

    realized_same_day = int((pd.to_datetime(work.get("match_date"), errors="coerce") == pd.to_datetime(work.get("player_history_source_max_date"), errors="coerce")).sum()) if "player_history_source_max_date" in work.columns else 0
    rows.append({"check_name": "same_day_history_contamination", "status": "PASS" if realized_same_day == 0 else "FAIL", "fail_rows": realized_same_day, "notes": "rolling player history should not include same-day source row"})

    normal_brier = _fit_proxy_brier(work, shuffled=False)
    shuffled_brier = _fit_proxy_brier(work, shuffled=True)
    proxy_status = "PASS" if pd.notna(normal_brier) and pd.notna(shuffled_brier) and shuffled_brier > normal_brier else "WARN"
    rows.append({"check_name": "shuffle_date_proxy", "status": proxy_status, "fail_rows": 0, "notes": f"normal_brier={normal_brier:.4f}; shuffled_brier={shuffled_brier:.4f}"})

    report = pd.DataFrame(rows)
    summary = {
        "overall_status": "PASS" if report["status"].isin(["FAIL"]).sum() == 0 else "FAIL",
        "normal_brier": round(normal_brier, 6) if pd.notna(normal_brier) else "nan",
        "shuffled_brier": round(shuffled_brier, 6) if pd.notna(shuffled_brier) else "nan",
    }
    return report, summary


def build(input_csv: Path, output_csv: Path, output_md: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    report, summary = run_audit(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    lines = [
        "# Tackles Proof Leak Audit",
        "",
        f"- overall_status: `{summary['overall_status']}`",
        f"- proxy normal brier: `{summary['normal_brier']}`",
        f"- proxy shuffled-date brier: `{summary['shuffled_brier']}`",
        "",
        "## Checks",
    ]
    for _, row in report.iterrows():
        lines.append(f"- {row['check_name']} | status={row['status']} | fail_rows={int(row['fail_rows'])} | notes={row['notes']}")
    output_md.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a proof-focused leak audit on the tackles model dataset.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default=str(REPORTS_DIR / "leak_audit_checks.csv"))
    parser.add_argument("--output-md", default=str(REPORTS_DIR / "leak_audit_report.md"))
    args = parser.parse_args()
    out = build(Path(args.input_csv), Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
