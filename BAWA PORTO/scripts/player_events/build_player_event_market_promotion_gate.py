#!/usr/bin/env python3
"""Build a player-event market promotion/watch gate.

Research-only. Combines historical proof with accumulated live-shadow outcomes
and classifies markets for dashboard prominence. It does not create priced
odds, deploy picks, slips, or production routing changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "reports" / "player_events" / "live_shadow_outcomes" / "PLAYER_EVENT_LIVE_OUTCOME_LEDGER_COMBINED.csv"
DEFAULT_TOP_SLICES = ROOT / "reports" / "2026-05-06" / "player_event_threshold_stability_audit" / "player_event_top_slice_stability.csv"
DEFAULT_LIVE_CANDIDATES = ROOT / "reports" / "2026-05-06" / "player_event_threshold_stability_audit" / "player_event_live_shadow_candidate_cells.csv"
DEFAULT_FOULED_CANDIDATES = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "player_fouled_interaction_policy_audit_strict_opponent"
    / "player_fouled_interaction_policy_candidates.csv"
)
DEFAULT_CARDS_CELLS = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "player_cards_hazard_audit_foundation"
    / "player_cards_hazard_threshold_cells.csv"
)
DEFAULT_CORNERS_CELLS = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "corners_intelligence_audit_foundation"
    / "corners_intelligence_threshold_cells.csv"
)
DEFAULT_KEEPER_SAVES_CELLS = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "keeper_saves_intelligence_audit_foundation"
    / "keeper_saves_intelligence_threshold_cells.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_event_market_promotion_gate"


STAGE_MARKET_MAP = {
    "PLAYER_FOULED_0_5_INTERACTION_WATCH": "Player Fouled 0.5+",
    "PLAYER_FOULED_1_5_INTERACTION_WATCH": "Player Fouled 1.5+",
    "PLAYER_SHOTS_1_5_INTERACTION_WATCH": "Player Shots 1.5+",
    "PLAYER_SHOTS_2_5_INTERACTION_WATCH": "Player Shots 2.5+",
    "PLAYER_SOT_0_5_INTERACTION_WATCH": "Player SOT 0.5+",
    "PLAYER_TACKLES_1_5_LIVE_SHADOW": "Player Tackles 1.5+",
    "PLAYER_TACKLES_2_5_LIVE_SHADOW": "Player Tackles 2.5+",
}

HISTORICAL_MARKET_MAP = {
    "shots": "Player Shots 0.5+",
    "shots_ge2": "Player Shots 1.5+",
    "shots_ge3": "Player Shots 2.5+",
    "shots_on_target": "Player SOT 0.5+",
    "sot_ge2_attackers": "Player SOT 1.5+",
    "sot_ge3_attackers": "Player SOT 2.5+",
    "fouls_committed": "Player Fouls Committed 1.5+",
    "tackles": "Player Tackles 1.5+",
}

REQUIRED_MARKETS = [
    "Player Shots 0.5+",
    "Player Shots 1.5+",
    "Player Shots 2.5+",
    "Player SOT 0.5+",
    "Player SOT 1.5+",
    "Player Fouled 0.5+",
    "Player Fouled 1.5+",
    "Player Fouls Committed 0.5+",
    "Player Fouls Committed 1.5+",
    "Player Tackles 1.5+",
    "Player Tackles 2.5+",
    "Player Cards 0.5+",
    "Corners Intelligence",
    "Keeper Saves Intelligence",
]


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def status_rank(status: str) -> int:
    return {"CORE_WATCH": 0, "WATCH": 1, "RESEARCH_ONLY": 2, "DO_NOT_USE": 3}.get(status, 9)


def historical_status(rows: int, hit_rate: float, stable_share: float, prelabel: str = "") -> str:
    if prelabel in {"CORE_WATCH", "WATCH", "RESEARCH_ONLY", "DO_NOT_USE"}:
        return prelabel
    if rows <= 0 or pd.isna(hit_rate):
        return "DO_NOT_USE"
    stable = 0.0 if pd.isna(stable_share) else stable_share
    if rows >= 500 and hit_rate >= 0.78 and stable >= 0.90:
        return "CORE_WATCH"
    if rows >= 500 and hit_rate >= 0.64 and stable >= 0.80:
        return "WATCH"
    if rows >= 100:
        return "RESEARCH_ONLY"
    return "DO_NOT_USE"


def live_status(rows: int, graded: int, hit_rate: float) -> tuple[str, str]:
    if rows <= 0:
        return "NO_LIVE_ROWS", "No current accumulated live-shadow rows yet."
    if graded < 25:
        if graded >= 8 and pd.notna(hit_rate) and hit_rate < 0.25:
            return "EARLY_NEGATIVE", "Tiny sample, but early graded rows are materially weak."
        return "PENDING_SAMPLE", "Live sample is too small; keep accumulating."
    if pd.isna(hit_rate):
        return "PENDING_SAMPLE", "Live rows exist but are not graded yet."
    if hit_rate >= 0.65:
        return "LIVE_GREEN", "Live graded sample is supportive."
    if hit_rate >= 0.50:
        return "LIVE_AMBER", "Live graded sample is usable but not promotion-grade."
    if hit_rate >= 0.35:
        return "LIVE_WEAK", "Live graded sample is below target."
    return "LIVE_RED", "Live graded sample is materially weak."


def final_status(hist_status: str, live_code: str) -> str:
    status = hist_status
    if live_code in {"LIVE_RED", "EARLY_NEGATIVE"}:
        return "DO_NOT_USE"
    if live_code == "LIVE_WEAK" and status_rank(status) < status_rank("RESEARCH_ONLY"):
        return "RESEARCH_ONLY"
    if live_code == "LIVE_AMBER" and status == "CORE_WATCH":
        return "WATCH"
    return status


def build_historical(
    top_slices: pd.DataFrame,
    live_candidates: pd.DataFrame,
    fouled: pd.DataFrame,
    cards_cells: pd.DataFrame,
    corners_cells: pd.DataFrame,
    keeper_saves_cells: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    hist = pd.concat([top_slices, live_candidates], ignore_index=True, sort=False)
    if not hist.empty:
        hist["market_display_name"] = hist["market"].astype(str).map(HISTORICAL_MARKET_MAP).fillna(hist.get("display_market", ""))
        for market, group in hist.groupby("market_display_name", dropna=False):
            group = group.copy()
            group["_score"] = num(group.get("hit_rate", np.nan)) + num(group.get("stable_month_share", np.nan)).fillna(0) * 0.05
            best = group.sort_values(["recommended_beta_label", "_score", "rows"], ascending=[True, False, False]).iloc[0]
            rows.append(
                {
                    "market_display": market,
                    "historical_rows": int(best.get("rows", 0) or 0),
                    "historical_hit_rate": float(best.get("hit_rate", np.nan)),
                    "historical_stable_month_share": float(best.get("stable_month_share", np.nan)),
                    "historical_source": "threshold_stability",
                    "historical_detail": str(best.get("prediction_variant", "")),
                    "historical_prelabel": "",
                }
            )
    if not fouled.empty:
        for market, group in fouled.groupby("display_market", dropna=False):
            best = group.sort_values(["interaction_label", "interaction_hit", "interaction_rows"], ascending=[True, False, False]).iloc[0]
            rows.append(
                {
                    "market_display": market,
                    "historical_rows": int(best.get("interaction_rows", 0) or 0),
                    "historical_hit_rate": float(best.get("interaction_hit", np.nan)),
                    "historical_stable_month_share": float(best.get("stable_month_share_vs_baseline", np.nan)),
                    "historical_source": "fouled_interaction_policy",
                    "historical_detail": str(best.get("interaction_label", "")),
                    "historical_prelabel": "",
                }
            )
    if not cards_cells.empty:
        cards = cards_cells.copy()
        cards = cards[cards.get("recommended_beta_label", pd.Series("", index=cards.index)).astype(str).ne("DO_NOT_USE")]
        if not cards.empty:
            cards["_rank"] = cards["recommended_beta_label"].astype(str).map(
                {"CORE_WATCH": 0, "WATCH": 1, "RESEARCH_ONLY": 2, "DO_NOT_USE": 3}
            ).fillna(9)
            best = cards.sort_values(
                ["_rank", "lift_vs_baseline", "hit_rate", "graded_rows"],
                ascending=[True, False, False, False],
            ).iloc[0]
            rows.append(
                {
                    "market_display": "Player Cards 0.5+",
                    "historical_rows": int(best.get("graded_rows", 0) or 0),
                    "historical_hit_rate": float(best.get("hit_rate", np.nan)),
                    "historical_stable_month_share": float(best.get("stable_month_share_vs_baseline", np.nan)),
                    "historical_source": "player_cards_hazard_audit",
                    "historical_detail": str(best.get("cell_label", "")),
                    "historical_prelabel": str(best.get("recommended_beta_label", "")),
                }
            )
    if not corners_cells.empty:
        corners = corners_cells.copy()
        corners = corners[corners.get("recommended_beta_label", pd.Series("", index=corners.index)).astype(str).ne("DO_NOT_USE")]
        if not corners.empty:
            corners["_rank"] = corners["recommended_beta_label"].astype(str).map(
                {"CORE_WATCH": 0, "WATCH": 1, "RESEARCH_ONLY": 2, "DO_NOT_USE": 3}
            ).fillna(9)
            best = corners.sort_values(
                ["_rank", "lift_vs_baseline", "hit_rate", "graded_rows"],
                ascending=[True, False, False, False],
            ).iloc[0]
            rows.append(
                {
                    "market_display": "Corners Intelligence",
                    "historical_rows": int(best.get("graded_rows", 0) or 0),
                    "historical_hit_rate": float(best.get("hit_rate", np.nan)),
                    "historical_stable_month_share": float(best.get("stable_month_share_vs_baseline", np.nan)),
                    "historical_source": "corners_intelligence_audit",
                    "historical_detail": f"{best.get('market_display', '')}:{best.get('cell_label', '')}",
                    "historical_prelabel": str(best.get("recommended_beta_label", "")),
                }
            )
    if not keeper_saves_cells.empty:
        keeper = keeper_saves_cells.copy()
        keeper = keeper[keeper.get("recommended_beta_label", pd.Series("", index=keeper.index)).astype(str).ne("DO_NOT_USE")]
        if not keeper.empty:
            keeper["_rank"] = keeper["recommended_beta_label"].astype(str).map(
                {"CORE_WATCH": 0, "WATCH": 1, "RESEARCH_ONLY": 2, "DO_NOT_USE": 3}
            ).fillna(9)
            best = keeper.sort_values(
                ["_rank", "lift_vs_baseline", "hit_rate", "graded_rows"],
                ascending=[True, False, False, False],
            ).iloc[0]
            rows.append(
                {
                    "market_display": "Keeper Saves Intelligence",
                    "historical_rows": int(best.get("graded_rows", 0) or 0),
                    "historical_hit_rate": float(best.get("hit_rate", np.nan)),
                    "historical_stable_month_share": float(best.get("stable_month_share_vs_baseline", np.nan)),
                    "historical_source": "keeper_saves_intelligence_audit",
                    "historical_detail": f"{best.get('market_display', '')}:{best.get('cell_label', '')}",
                    "historical_prelabel": str(best.get("recommended_beta_label", "")),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["market_display"])
    out = pd.DataFrame(rows)
    out["_rank"] = [
        status_rank(
            historical_status(
                row["historical_rows"],
                row["historical_hit_rate"],
                row["historical_stable_month_share"],
                str(row.get("historical_prelabel", "")),
            )
        )
        for _, row in out.iterrows()
    ]
    out = out.sort_values(["market_display", "_rank", "historical_hit_rate", "historical_rows"], ascending=[True, True, False, False])
    return out.drop_duplicates("market_display", keep="first").drop(columns="_rank")


def build_live(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=["market_display"])
    ledger = ledger.copy()
    ledger["market_display"] = ledger["shadow_stage"].astype(str).map(STAGE_MARKET_MAP).fillna(ledger.get("expression", ""))
    rows = []
    for market, group in ledger.groupby("market_display", dropna=False):
        graded = group[group["outcome_status"].astype(str).eq("GRADED")]
        hits = float(num(graded.get("actual_hit", pd.Series(dtype=float))).sum()) if not graded.empty else 0.0
        rows.append(
            {
                "market_display": market,
                "live_rows": int(len(group)),
                "live_graded": int(len(graded)),
                "live_hits": int(hits),
                "live_hit_rate": float(hits / len(graded)) if len(graded) else np.nan,
                "live_pending": int(len(group) - len(graded)),
            }
        )
    return pd.DataFrame(rows)


def recommendation_reason(row: pd.Series) -> str:
    parts = []
    hist_status = row.get("historical_status", "DO_NOT_USE")
    live_code = row.get("live_status_code", "NO_LIVE_ROWS")
    parts.append(f"historical={hist_status}")
    parts.append(f"live={live_code}")
    if pd.notna(row.get("historical_hit_rate", np.nan)):
        parts.append(f"hist_hr={row['historical_hit_rate']:.3f}")
    if pd.notna(row.get("live_hit_rate", np.nan)):
        parts.append(f"live_hr={row['live_hit_rate']:.3f}")
    if int(row.get("live_graded", 0) or 0) < 25:
        parts.append("live_sample_small")
    return " | ".join(parts)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--top-slices", type=Path, default=DEFAULT_TOP_SLICES)
    parser.add_argument("--live-candidates", type=Path, default=DEFAULT_LIVE_CANDIDATES)
    parser.add_argument("--fouled-candidates", type=Path, default=DEFAULT_FOULED_CANDIDATES)
    parser.add_argument("--cards-cells", type=Path, default=DEFAULT_CARDS_CELLS)
    parser.add_argument("--corners-cells", type=Path, default=DEFAULT_CORNERS_CELLS)
    parser.add_argument("--keeper-saves-cells", type=Path, default=DEFAULT_KEEPER_SAVES_CELLS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    historical = build_historical(
        read(args.top_slices),
        read(args.live_candidates),
        read(args.fouled_candidates),
        read(args.cards_cells),
        read(args.corners_cells),
        read(args.keeper_saves_cells),
    )
    live = build_live(read(args.ledger))
    base = pd.DataFrame({"market_display": REQUIRED_MARKETS})
    gate = base.merge(historical, on="market_display", how="left").merge(live, on="market_display", how="left")
    for col in ["historical_rows", "live_rows", "live_graded", "live_hits", "live_pending"]:
        gate[col] = num(gate.get(col, pd.Series(0, index=gate.index))).fillna(0).astype(int)
    for col in ["historical_hit_rate", "historical_stable_month_share", "live_hit_rate"]:
        gate[col] = num(gate.get(col, pd.Series(np.nan, index=gate.index)))
    gate["historical_status"] = [
        historical_status(
            row["historical_rows"],
            row["historical_hit_rate"],
            row["historical_stable_month_share"],
            str(row.get("historical_prelabel", "")),
        )
        for _, row in gate.iterrows()
    ]
    live_pairs = [live_status(int(row["live_rows"]), int(row["live_graded"]), row["live_hit_rate"]) for _, row in gate.iterrows()]
    gate["live_status_code"] = [pair[0] for pair in live_pairs]
    gate["live_status_note"] = [pair[1] for pair in live_pairs]
    gate["recommended_status"] = [final_status(row["historical_status"], row["live_status_code"]) for _, row in gate.iterrows()]
    gate["dashboard_prominence"] = np.select(
        [
            gate["recommended_status"].eq("CORE_WATCH"),
            gate["recommended_status"].eq("WATCH"),
        ],
        ["PRIMARY_PLAYER_INTEL", "SECONDARY_PLAYER_INTEL"],
        default="RESEARCH_DRAWER_ONLY",
    )
    gate["promotion_allowed"] = False
    gate["recommendation_reason"] = gate.apply(recommendation_reason, axis=1)
    gate = gate.sort_values(
        ["recommended_status", "historical_hit_rate", "live_graded"],
        key=lambda s: s.map({"CORE_WATCH": 0, "WATCH": 1, "RESEARCH_ONLY": 2, "DO_NOT_USE": 3}).fillna(s)
        if s.name == "recommended_status"
        else s,
        ascending=[True, False, False],
    )
    gate.to_csv(args.outdir / "PLAYER_EVENT_MARKET_PROMOTION_GATE.csv", index=False)
    counts = gate.groupby(["recommended_status", "dashboard_prominence"], dropna=False).size().reset_index(name="markets")
    counts.to_csv(args.outdir / "PLAYER_EVENT_MARKET_PROMOTION_GATE_COUNTS.csv", index=False)

    lines = [
        "# Player Event Market Promotion Gate",
        "",
        "Research-only market classification for player-event dashboard prominence.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- `promotion_allowed` is hard-coded false; statuses are dashboard/intelligence posture only.",
        "",
        "## Counts",
        markdown_table(counts),
        "",
        "## Market Gate",
        markdown_table(
            gate[
                [
                    "market_display",
                    "recommended_status",
                    "dashboard_prominence",
                    "historical_rows",
                    "historical_hit_rate",
                    "historical_stable_month_share",
                    "live_rows",
                    "live_graded",
                    "live_hit_rate",
                    "live_status_code",
                    "recommendation_reason",
                ]
            ],
            max_rows=80,
        ),
    ]
    (args.outdir / "PLAYER_EVENT_MARKET_PROMOTION_GATE.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE {args.outdir}")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
