#!/usr/bin/env python3
"""Build a weekend beta prep pack for player-intelligence live testing.

Research-only. Reads the unified live shadow dashboard and rolling outcome
ledger, then emits a compact readiness report for manual review. It does not
create priced odds, deploy picks, slips, or production routing changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DASHBOARD = (
    ROOT
    / "reports"
    / "2026-05-08"
    / "live_shadow_research_dashboard_with_fixture_markets"
    / "2026_05_02_to_2026_05_04"
    / "2026_05_02_to_2026_05_04__LIVE_SHADOW_RESEARCH_DASHBOARD.csv"
)
DEFAULT_LEDGER = (
    ROOT
    / "reports"
    / "player_events"
    / "live_shadow_outcomes"
    / "PLAYER_EVENT_LIVE_OUTCOME_LEDGER_V2_WITH_KEY_PASS_ASSIST.csv"
)
DEFAULT_COPY_PACK = ROOT / "reports" / "2026-05-07" / "player_event_dashboard_copy_pack_weekend_beta" / "PLAYER_EVENT_DASHBOARD_COPY_PACK.csv"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_event_weekend_beta_prep_pack"
DEFAULT_TACTICAL_REGISTRY = ROOT / "reports" / "2026-05-08" / "tactical_feature_registry" / "TACTICAL_FEATURE_REGISTRY.csv"

BETA_FAMILIES = {
    "PLAYER_EVENT_INTERACTION",
    "PLAYER_EVENT_TACKLES",
    "KEEPER_SAVES_INTELLIGENCE",
    "CORNERS_INTELLIGENCE",
    "KEY_PASS_ASSIST_INTELLIGENCE",
    "FIXTURE_MARKET_INTELLIGENCE",
}


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def registry_for_stage(registry: pd.DataFrame, stage: str) -> dict[str, str]:
    if registry.empty or "target_shadow_stages" not in registry.columns:
        return {
            "tactical_feature_ids": "",
            "tactical_feature_families": "",
            "tactical_leakage_risk_max": "",
        }
    stage_text = str(stage)
    mask = registry["target_shadow_stages"].astype(str).str.contains(stage_text, regex=False, na=False)
    subset = registry[mask].copy()
    if subset.empty:
        return {
            "tactical_feature_ids": "",
            "tactical_feature_families": "",
            "tactical_leakage_risk_max": "",
        }
    risks = subset["leakage_risk"].astype(str)
    return {
        "tactical_feature_ids": "|".join(dict.fromkeys(subset["feature_id"].astype(str))),
        "tactical_feature_families": "|".join(dict.fromkeys(subset["family"].astype(str))),
        "tactical_leakage_risk_max": "MEDIUM" if risks.str.contains("MEDIUM", na=False).any() else "LOW",
    }


def hit_rate_status(graded: int, hit_rate: float, rows: int) -> str:
    if graded <= 0:
        return "AWAIT_RESULTS"
    if graded < 10:
        return "TINY_SAMPLE"
    if hit_rate >= 0.65 and graded >= 20:
        return "LIVE_TEST_CORE_WATCH"
    if hit_rate >= 0.55 and graded >= 20:
        return "LIVE_TEST_CONFIRM"
    if hit_rate >= 0.45:
        return "ACCUMULATE_ONLY"
    if rows >= 20:
        return "TIGHTEN_OR_HOLD"
    return "WATCH_ONLY"


def summarize_dashboard(dashboard: pd.DataFrame) -> pd.DataFrame:
    if dashboard.empty:
        return pd.DataFrame()
    pe = dashboard[dashboard["shadow_family"].astype(str).isin(BETA_FAMILIES)].copy()
    return (
        pe.groupby(["shadow_family", "shadow_stage", "watch_priority"], dropna=False)
        .size()
        .reset_index(name="shadow_rows")
        .sort_values(["shadow_family", "shadow_stage", "watch_priority"])
    )


def summarize_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=["shadow_stage", "watch_priority", "ledger_rows", "graded", "hits", "hit_rate", "pending"])
    ledger = ledger.copy()
    ledger["actual_hit"] = num(ledger.get("actual_hit", pd.Series(np.nan, index=ledger.index)))
    rows = []
    for key, group in ledger.groupby(["shadow_stage", "watch_priority"], dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        graded_mask = group["outcome_status"].astype(str).eq("GRADED")
        graded = group[graded_mask]
        hits = float(num(graded.get("actual_hit", pd.Series(dtype=float))).sum()) if not graded.empty else 0.0
        rows.append(
            {
                "shadow_stage": key[0],
                "watch_priority": key[1],
                "ledger_rows": int(len(group)),
                "graded": int(len(graded)),
                "hits": int(hits),
                "hit_rate": float(hits / len(graded)) if len(graded) else np.nan,
                "pending": int((~graded_mask).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_readiness(dashboard_summary: pd.DataFrame, ledger_summary: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    if dashboard_summary.empty:
        return pd.DataFrame()
    out = dashboard_summary.merge(ledger_summary, on=["shadow_stage", "watch_priority"], how="left")
    for col in ["ledger_rows", "graded", "hits", "pending"]:
        out[col] = num(out.get(col, pd.Series(0, index=out.index))).fillna(0).astype(int)
    out["hit_rate"] = num(out.get("hit_rate", pd.Series(np.nan, index=out.index)))
    out["readiness_status"] = [
        hit_rate_status(int(graded), float(hit_rate) if pd.notna(hit_rate) else np.nan, int(rows))
        for graded, hit_rate, rows in zip(out["graded"], out["hit_rate"], out["shadow_rows"])
    ]
    registry_tags = [registry_for_stage(registry, stage) for stage in out["shadow_stage"]]
    out["tactical_feature_ids"] = [tags["tactical_feature_ids"] for tags in registry_tags]
    out["tactical_feature_families"] = [tags["tactical_feature_families"] for tags in registry_tags]
    out["tactical_leakage_risk_max"] = [tags["tactical_leakage_risk_max"] for tags in registry_tags]
    priority_rank = {
        "LIVE_TEST_CORE_WATCH": 5,
        "LIVE_TEST_CONFIRM": 4,
        "ACCUMULATE_ONLY": 3,
        "TINY_SAMPLE": 2,
        "AWAIT_RESULTS": 1,
        "WATCH_ONLY": 1,
        "TIGHTEN_OR_HOLD": 0,
    }
    out["_rank"] = out["readiness_status"].map(priority_rank).fillna(0)
    return out.sort_values(["_rank", "graded", "shadow_rows"], ascending=[False, False, False]).drop(columns=["_rank"])


def markdown_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            work[col] = work[col].astype("string").fillna("")
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in work.columns) + " |")
    return "\n".join(lines)


def write_report(
    outdir: Path,
    readiness: pd.DataFrame,
    dashboard_path: Path,
    ledger_path: Path,
    copy_pack_path: Path,
    registry_path: Path,
) -> None:
    status_counts = (
        readiness.groupby("readiness_status", dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        if not readiness.empty
        else pd.DataFrame()
    )
    core = readiness[readiness["readiness_status"].isin(["LIVE_TEST_CORE_WATCH", "LIVE_TEST_CONFIRM"])].copy()
    hold = readiness[readiness["readiness_status"].isin(["TIGHTEN_OR_HOLD", "AWAIT_RESULTS"])].copy()
    tactical = (
        readiness[readiness["tactical_feature_families"].astype(str).ne("")]
        .groupby(["tactical_feature_families", "readiness_status"], dropna=False)
        .agg(rows=("shadow_stage", "size"), shadow_rows=("shadow_rows", "sum"), graded=("graded", "sum"), hits=("hits", "sum"))
        .reset_index()
        .sort_values(["graded", "shadow_rows"], ascending=[False, False])
    )
    lines = [
        "# Player Event Weekend Beta Prep Pack",
        "",
        "Manual-review readiness pack for the first weekend player-intelligence live test.",
        "",
        "## Safety",
        "- Beta player intelligence only.",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Do not treat rows as final bets without manual review, lineup context, and repeated live outcome evidence.",
        "",
        "## Inputs",
        f"- dashboard: `{dashboard_path}`",
        f"- outcome ledger: `{ledger_path}`",
        f"- copy pack: `{copy_pack_path}`",
        f"- tactical registry: `{registry_path}`",
        "",
        "## Readiness Counts",
        markdown_table(status_counts),
        "",
        "## Weekend Core / Confirm Candidates",
        markdown_table(core, max_rows=40),
        "",
        "## Hold / Await Results",
        markdown_table(hold, max_rows=40),
        "",
        "## Tactical Registry Hooks",
        markdown_table(tactical, max_rows=60),
        "",
        "## Operating Posture For This Weekend",
        "- Use the board as an intelligence layer beside the restored goal-model spine.",
        "- Prioritise keeper saves 1.5+, key passes 0.5+, fouled 0.5+, and tackles 1.5+ where live evidence is already grading.",
        "- Keep assists, key passes 1.5+, corners, SOT, and higher tackle/save thresholds as watch-only unless manual context is exceptional.",
        "- Confirm expected starters and minutes before surfacing any player row prominently.",
        "- After fixtures settle, rerun the outcome tracker and accumulator before judging performance.",
        "",
        "## Next Commands After Refresh",
        "```bash",
        ".venv/bin/python scripts/player_events/build_player_event_shadow_outcome_tracker_v2.py \\",
        "  --shadow-board reports/2026-05-07/live_shadow_research_dashboard_with_key_pass_assist/2026_05_02_to_2026_05_04/2026_05_02_to_2026_05_04__LIVE_SHADOW_RESEARCH_DASHBOARD.csv \\",
        "  --outdir reports/2026-05-07/player_event_shadow_outcome_tracker_v2_with_key_pass_assist",
        "```",
        "",
        "```bash",
        ".venv/bin/python scripts/player_events/build_player_event_live_outcome_accumulator.py \\",
        "  --tracker-rows reports/2026-05-07/player_event_shadow_outcome_tracker_v2_with_key_pass_assist/PLAYER_EVENT_SHADOW_OUTCOME_TRACKER_ROWS.csv \\",
        "  --ledger reports/player_events/live_shadow_outcomes/PLAYER_EVENT_LIVE_OUTCOME_LEDGER_V2_WITH_KEY_PASS_ASSIST.csv \\",
        "  --outdir reports/2026-05-07/player_event_live_outcome_accumulator_v2_with_key_pass_assist",
        "```",
    ]
    (outdir / "PLAYER_EVENT_WEEKEND_BETA_PREP_PACK.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--copy-pack", type=Path, default=DEFAULT_COPY_PACK)
    parser.add_argument("--tactical-registry", type=Path, default=DEFAULT_TACTICAL_REGISTRY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    if not args.dashboard.exists():
        raise SystemExit(f"Missing dashboard: {args.dashboard}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    dashboard = read_csv_if_exists(args.dashboard)
    ledger = read_csv_if_exists(args.ledger)
    registry = read_csv_if_exists(args.tactical_registry)
    readiness = build_readiness(summarize_dashboard(dashboard), summarize_ledger(ledger), registry)
    readiness.to_csv(args.outdir / "PLAYER_EVENT_WEEKEND_BETA_READINESS.csv", index=False)
    write_report(args.outdir, readiness, args.dashboard, args.ledger, args.copy_pack, args.tactical_registry)

    print(f"WROTE {args.outdir}")
    print(f"readiness_rows={len(readiness)}")
    if not readiness.empty:
        print(readiness[["shadow_stage", "watch_priority", "shadow_rows", "graded", "hit_rate", "readiness_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
