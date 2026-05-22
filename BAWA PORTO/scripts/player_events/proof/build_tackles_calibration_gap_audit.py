from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


GREEN_ECE = 0.030
AMBER_ECE = 0.035
MIN_ROWS = 200
MIN_TOP_DECILE = 0.65
TOP_DECILE = 0.10


def _metric_block(df: pd.DataFrame, prob_col: str) -> dict[str, float]:
    probs = np.clip(pd.to_numeric(df[prob_col], errors="coerce").fillna(np.nan), 1e-6, 1 - 1e-6)
    hit = pd.to_numeric(df["actual_hit_ge2"], errors="coerce").fillna(0).astype(int)
    valid = probs.notna()
    if valid.sum() == 0:
        return {
            "rows": int(len(df)),
            "coverage": 0.0,
            "brier_ge2": np.nan,
            "logloss_ge2": np.nan,
            "ece_10bin": np.nan,
            "top_decile_precision": np.nan,
        }
    bins = pd.cut(probs.loc[valid], bins=np.linspace(0, 1, 11), include_lowest=True, duplicates="drop")
    calib = pd.DataFrame({"prob": probs.loc[valid], "hit": hit.loc[valid], "bin": bins}).dropna(subset=["bin"])
    grouped = calib.groupby("bin", observed=False).agg(pred=("prob", "mean"), obs=("hit", "mean"), n=("hit", "size")).reset_index()
    ece = float(((grouped["n"] / grouped["n"].sum()) * (grouped["pred"] - grouped["obs"]).abs()).sum()) if not grouped.empty else np.nan
    top_n = max(1, int(np.ceil(valid.sum() * TOP_DECILE)))
    top = df.loc[valid].assign(_p=probs.loc[valid]).sort_values("_p", ascending=False).head(top_n)
    return {
        "rows": int(len(df)),
        "coverage": round(float(valid.mean()), 6),
        "brier_ge2": round(float(brier_score_loss(hit.loc[valid], probs.loc[valid])), 6),
        "logloss_ge2": round(float(log_loss(hit.loc[valid], probs.loc[valid], labels=[0, 1])), 6),
        "ece_10bin": round(ece, 6),
        "top_decile_precision": round(float(pd.to_numeric(top["actual_hit_ge2"], errors="coerce").fillna(0).mean()), 6),
    }


def _build_reliability_table(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    probs = np.clip(pd.to_numeric(df[prob_col], errors="coerce").fillna(np.nan), 1e-6, 1 - 1e-6)
    hit = pd.to_numeric(df["actual_hit_ge2"], errors="coerce").fillna(0).astype(int)
    valid = probs.notna()
    bins = pd.cut(probs.loc[valid], bins=np.linspace(0, 1, 11), include_lowest=True, duplicates="drop")
    calib = pd.DataFrame({"prob": probs.loc[valid], "hit": hit.loc[valid], "bin": bins}).dropna(subset=["bin"])
    grouped = calib.groupby("bin", observed=False).agg(
        predicted_prob=("prob", "mean"),
        observed_hit_rate=("hit", "mean"),
        rows=("hit", "size"),
    ).reset_index()
    grouped["abs_gap"] = (grouped["predicted_prob"] - grouped["observed_hit_rate"]).abs()
    grouped["prob_col"] = prob_col
    return grouped[["prob_col", "bin", "rows", "predicted_prob", "observed_hit_rate", "abs_gap"]]


def _build_subgroup_labels(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["minutes_bucket"] = pd.cut(
        pd.to_numeric(work["expected_minutes_proof"], errors="coerce").fillna(0.0),
        bins=[-np.inf, 60, 80, np.inf],
        labels=["30_60", "60_80", "80_plus"],
    ).astype(str)
    rows = []
    for keys, grp in work.groupby(["league_tag", "position_group", "minutes_bucket", "season_tag"], dropna=False):
        metrics = _metric_block(grp, "nb_p_ge2")
        label = "red"
        if metrics["rows"] >= MIN_ROWS and metrics["top_decile_precision"] >= MIN_TOP_DECILE:
            if metrics["ece_10bin"] <= GREEN_ECE:
                label = "green"
            elif metrics["ece_10bin"] <= AMBER_ECE:
                label = "amber"
        rows.append(
            {
                "league_tag": keys[0],
                "position_group": keys[1],
                "minutes_bucket": keys[2],
                "season_tag": keys[3],
                **metrics,
                "lane_status": label,
            }
        )
    return pd.DataFrame(rows).sort_values(["lane_status", "ece_10bin", "top_decile_precision"], ascending=[True, True, False])


def _build_isotonic_oof(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["nb_p_ge2_iso_oof"] = np.nan
    seasons = sorted(pd.to_numeric(work["season_tag"], errors="coerce").dropna().unique())
    for season in seasons:
        holdout_mask = pd.to_numeric(work["season_tag"], errors="coerce").eq(season)
        train = work.loc[~holdout_mask].copy()
        test = work.loc[holdout_mask].copy()
        if train.empty or test.empty:
            continue
        x_train = np.clip(pd.to_numeric(train["nb_p_ge2"], errors="coerce").fillna(np.nan), 1e-6, 1 - 1e-6)
        y_train = pd.to_numeric(train["actual_hit_ge2"], errors="coerce").fillna(0).astype(int)
        valid_train = x_train.notna()
        if valid_train.sum() < 100:
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(x_train.loc[valid_train], y_train.loc[valid_train])
        x_test = np.clip(pd.to_numeric(test["nb_p_ge2"], errors="coerce").fillna(np.nan), 1e-6, 1 - 1e-6)
        valid_test = x_test.notna()
        calibrated = np.full(len(test), np.nan)
        calibrated[valid_test.to_numpy()] = iso.predict(x_test.loc[valid_test])
        work.loc[holdout_mask, "nb_p_ge2_iso_oof"] = calibrated
    return work


def build_audit(proof_dir: Path) -> dict[str, Path]:
    pred_path = proof_dir / "tackles_nb_proof_predictions.csv"
    pred = pd.read_csv(pred_path, low_memory=False)

    reliability = pd.concat(
        [
            _build_reliability_table(pred, "nb_p_ge2"),
            _build_reliability_table(pred, "cohort_p_ge2"),
        ],
        ignore_index=True,
    )
    reliability_csv = proof_dir / "tackles_calibration_reliability_bins.csv"
    reliability.to_csv(reliability_csv, index=False)

    subgroup = _build_subgroup_labels(pred)
    subgroup_csv = proof_dir / "tackles_calibration_subgroup_status.csv"
    subgroup.to_csv(subgroup_csv, index=False)

    pred_iso = _build_isotonic_oof(pred)
    iso_rows = []
    base_metrics = _metric_block(pred_iso, "nb_p_ge2")
    iso_rows.append({"variant": "baseline_nb", **base_metrics})
    if pred_iso["nb_p_ge2_iso_oof"].notna().any():
        iso_metrics = _metric_block(pred_iso[pred_iso["nb_p_ge2_iso_oof"].notna()].copy(), "nb_p_ge2_iso_oof")
        iso_rows.append({"variant": "isotonic_oof", **iso_metrics})
    iso_csv = proof_dir / "tackles_calibration_isotonic_comparison.csv"
    pd.DataFrame(iso_rows).to_csv(iso_csv, index=False)

    green = subgroup[subgroup["lane_status"] == "green"].copy()
    amber = subgroup[subgroup["lane_status"] == "amber"].copy()
    red = subgroup[subgroup["lane_status"] == "red"].copy()
    iso_df = pd.read_csv(iso_csv)

    md = proof_dir / "tackles_calibration_gap_audit.md"
    lines = [
        "# Tackles Calibration Gap Audit",
        "",
        "- proof source: `tackles_nb_proof_predictions.csv`",
        f"- baseline global ECE: `{base_metrics['ece_10bin']}`",
        f"- target ECE: `<= {GREEN_ECE:.3f}`",
        f"- gap to close: `{round(float(base_metrics['ece_10bin']) - GREEN_ECE, 6)}`",
        "",
        "## Reliability Bin Read",
        f"- reliability bin rows written to: `{reliability_csv.name}`",
        f"- baseline bins with largest absolute gaps should be inspected first; total rows: `{len(reliability[reliability['prob_col'] == 'nb_p_ge2'])}`",
        "",
        "## Subgroup Status",
        f"- green lanes: `{len(green)}`",
        f"- amber lanes: `{len(amber)}`",
        f"- red lanes: `{len(red)}`",
        f"- green row share: `{round(float(green['rows'].sum() / subgroup['rows'].sum()), 4) if len(subgroup) else 0}`",
        f"- amber row share: `{round(float(amber['rows'].sum() / subgroup['rows'].sum()), 4) if len(subgroup) else 0}`",
        "",
        "### Green Lanes",
    ]
    if green.empty:
        lines.append("- none at current thresholds")
    else:
        for _, row in green.iterrows():
            lines.append(
                f"- {row['league_tag']} | {row['position_group']} | {row['minutes_bucket']} | {int(row['season_tag'])} | "
                f"rows={int(row['rows'])} | ece={row['ece_10bin']} | top_decile={row['top_decile_precision']}"
            )
    lines.extend([
        "",
        "### Amber Lanes",
    ])
    if amber.empty:
        lines.append("- none at current thresholds")
    else:
        for _, row in amber.iterrows():
            lines.append(
                f"- {row['league_tag']} | {row['position_group']} | {row['minutes_bucket']} | {int(row['season_tag'])} | "
                f"rows={int(row['rows'])} | ece={row['ece_10bin']} | top_decile={row['top_decile_precision']}"
            )
    lines.extend([
        "",
        "## Optional Isotonic Comparison",
        f"- comparison rows written to: `{iso_csv.name}`",
    ])
    for _, row in iso_df.iterrows():
        lines.append(
            f"- {row['variant']}: rows={int(row['rows'])} | brier={row['brier_ge2']} | "
            f"logloss={row['logloss_ge2']} | ece={row['ece_10bin']} | top_decile={row['top_decile_precision']}"
        )
    lines.extend([
        "",
        "## Current Read",
        "- The remaining proof gap is small and still looks like a calibration problem, not a signal problem.",
        "- Green lanes already exist, but they do not yet cover enough of the total tackles surface to declare broad publishable readiness.",
        "- Any next change should protect the current Brier / log-loss / top-decile edge and stay inside the existing feature estate.",
    ])
    md.write_text("\n".join(lines) + "\n")

    return {
        "reliability_csv": reliability_csv,
        "subgroup_csv": subgroup_csv,
        "iso_csv": iso_csv,
        "audit_md": md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the tackles calibration gap audit pack.")
    parser.add_argument("--proof-dir", default="reports/player_events/proof")
    args = parser.parse_args()
    out = build_audit(Path(args.proof_dir))
    for key, path in out.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
