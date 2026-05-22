from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_ROOT = REPO_ROOT / "reports" / "player_events" / "backtests"


def _latest_backtest_dir() -> Path:
    dirs = sorted(BACKTEST_ROOT.glob("player_events_3y_backtest__*/"), reverse=True)
    if not dirs:
        raise FileNotFoundError("No player-events 3Y backtest directories found.")
    return dirs[0]


def _priority_value(text: str) -> float:
    return {
        "APPLY_SOFT_PATCH": 3.0,
        "TRIAL_PATCH": 2.0,
        "WATCH_ONLY": 1.0,
    }.get(str(text or "").upper(), 0.0)


def build(backtest_dir: Path, output_csv: Path, output_md: Path) -> pd.DataFrame:
    patch_csv = backtest_dir / "player_events_3y_patch_proposal.csv"
    shadow_csv = backtest_dir / "player_events_3y_shadow_priority.csv"
    tuning_csv = backtest_dir / "player_events_3y_threshold_tuning.csv"

    patch = pd.read_csv(patch_csv)
    shadow = pd.read_csv(shadow_csv) if shadow_csv.exists() else pd.DataFrame()
    tuning = pd.read_csv(tuning_csv)

    keep_signals = {"LOWER_SCORE_GATE", "RAISE_SCORE_GATE"}
    patch = patch[patch["tuning_signal"].isin(keep_signals)].copy()
    tuning = tuning[tuning["tuning_signal"].isin(keep_signals)].copy()

    if patch.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        patch.to_csv(output_csv, index=False)
        output_md.write_text("# Player Events 3Y Tuning Action Board\n\nNo directional tuning cohorts found.\n")
        return patch

    shadow = shadow.rename(
        columns={
            "runner_rows": "shadow_runner_rows",
            "priority_score": "shadow_priority_score",
            "priority_bucket": "shadow_priority_bucket",
        }
    )
    merge_cols = [
        "market",
        "review_family",
        "prematch_risk_focus",
        "patch_confidence",
        "proposed_gate_action",
        "proposed_score_cut_shift",
    ]
    shadow_cols = merge_cols + [
        "shadow_runner_rows",
        "current_selected",
        "shadow_selected",
        "newly_admitted",
        "newly_removed",
        "current_hits",
        "shadow_hits",
        "newly_admitted_hits",
        "newly_removed_hits",
        "net_hit_gain",
        "admit_hit_rate",
        "sample_stability",
        "shadow_priority_score",
        "shadow_priority_bucket",
    ]
    if not shadow.empty:
        patch = patch.merge(shadow[shadow_cols], on=merge_cols, how="left")

    patch["direction_strength"] = patch["avg_score_delta"].abs()
    patch["patch_confidence_score"] = patch["patch_confidence"].map(_priority_value).fillna(0.0)
    patch["sample_support"] = patch["rows"].fillna(0) + patch["fixtures"].fillna(0)
    patch["hit_confidence"] = patch["avg_expected_hit"].fillna(0.0)
    patch["shadow_priority_score"] = pd.to_numeric(patch.get("shadow_priority_score"), errors="coerce").fillna(0.0)
    patch["net_hit_gain"] = pd.to_numeric(patch.get("net_hit_gain"), errors="coerce").fillna(0.0)
    patch["sample_stability"] = pd.to_numeric(patch.get("sample_stability"), errors="coerce").fillna(0.0)

    patch["action_rank_score"] = (
        patch["patch_confidence_score"] * 4.0
        + patch["sample_support"] * 0.6
        + patch["hit_confidence"] * 3.0
        + patch["direction_strength"] * 0.2
        + patch["shadow_priority_score"] * 0.8
        + patch["net_hit_gain"] * 1.5
        + patch["sample_stability"] * 2.0
    ).round(3)

    patch = patch.sort_values(
        [
            "action_rank_score",
            "patch_confidence_score",
            "rows",
            "fixtures",
            "direction_strength",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    patch["recommended_trial_order"] = patch.index + 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    patch.to_csv(output_csv, index=False)

    lines = [
        "# Player Events 3Y Tuning Action Board",
        "",
        "- Ranks the directional tuning cohorts from the latest dedicated player-events 3Y backtest.",
        "- Focuses on `LOWER_SCORE_GATE`, `RAISE_SCORE_GATE`, and the patch-confidence ladder so we know what to trial first.",
        "",
        "## Trial First",
    ]
    for _, row in patch.head(8).iterrows():
        lines.append(
            f"- rank={int(row['recommended_trial_order'])} | "
            f"{row['market']} | {row['review_family']} | risk={row['prematch_risk_focus']} | "
            f"signal={row['tuning_signal']} | patch={row['patch_confidence']} | "
            f"action={row['proposed_gate_action']} {row['proposed_score_cut_shift']:+.1f} | "
            f"rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | "
            f"expected_hit={float(row['avg_expected_hit']):.3f} | "
            f"score_delta={float(row['avg_score_delta']):.2f} | "
            f"rank_score={float(row['action_rank_score']):.3f}"
        )

    lines.extend(
        [
            "",
            "## Summary",
            f"- APPLY_SOFT_PATCH cohorts: `{int((patch['patch_confidence'] == 'APPLY_SOFT_PATCH').sum())}`",
            f"- TRIAL_PATCH cohorts: `{int((patch['patch_confidence'] == 'TRIAL_PATCH').sum())}`",
            f"- WATCH_ONLY directional cohorts: `{int((patch['patch_confidence'] == 'WATCH_ONLY').sum())}`",
            "",
            "## Interpretation",
            "- Higher ranks combine directional tuning pressure, cohort size, expected-hit quality, and any shadow-patch evidence.",
            "- `APPLY_SOFT_PATCH` rows should usually be trialed before pure `TRIAL_PATCH` rows unless the shadow evidence is clearly weak.",
            "- `WATCH_ONLY` can still appear on the board, but they should not beat stronger directional cohorts without more sample.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")
    return patch


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ranked tuning action board from the latest player-events 3Y backtest pack.")
    parser.add_argument("--backtest-dir", default="", help="Optional explicit player-events 3Y backtest directory.")
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_3y_tuning_action_board.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_3y_tuning_action_board.md"),
    )
    args = parser.parse_args()

    backtest_dir = Path(args.backtest_dir) if args.backtest_dir else _latest_backtest_dir()
    out = build(backtest_dir, Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
