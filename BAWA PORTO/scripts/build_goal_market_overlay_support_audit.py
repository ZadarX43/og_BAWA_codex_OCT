#!/usr/bin/env python3
"""Build a clear support audit for deploy goal-market picks.

This is a reporting layer only. It answers whether the existing fixture/team
intelligence signal supports, reviews, or contradicts each published FTR, BTTS,
and OU25 deploy pick. Player-event and fixture-market shadows are included as
context, but they do not override the market-specific signal verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


MARKET_SIGNAL_COLUMNS = {
    "FTR": ("ftr_signal_pick", "ftr_signal_state", "ftr_signal_score"),
    "OU25": ("ou25_signal_pick", "ou25_signal_state", "ou25_signal_score"),
    "BTTS": ("btts_signal_pick", "btts_signal_state", "btts_signal_score"),
}

GOAL_PLAYER_MARKETS = {"PLAYER_SHOTS", "PLAYER_SOT"}
CONTACT_PLAYER_MARKETS = {"PLAYER_TACKLES", "PLAYER_FOULS", "PLAYER_FOULED", "PLAYER_CARDS"}
STRONG_PLAYER_LABELS = {"SHADOW_CORE", "STRONG_WATCH"}


def read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def market_pick(row: pd.Series) -> str:
    market = clean_text(row.get("market")).upper()
    pick = clean_text(row.get("model_pick")).upper()
    if market == "OU25":
        if "OVER" in pick:
            return "OVER25"
        if "UNDER" in pick:
            return "UNDER25"
    if market == "BTTS":
        if "YES" in pick:
            return "YES"
        if "NO" in pick:
            return "NO"
    return pick


def verdict(row: pd.Series) -> tuple[str, str]:
    alignment = clean_text(row.get("site_signal_alignment")).lower()
    state = clean_text(row.get("market_signal_state")).upper()
    pick = clean_text(row.get("market_signal_pick")).upper()
    model_pick = clean_text(row.get("normalized_model_pick")).upper()
    if alignment == "supports_model":
        return "SUPPORTS_PICK", f"{pick} signal matches deploy pick"
    if alignment == "conflicts_model":
        return "CONTRADICTS_PICK", f"{pick} {state} signal conflicts with deploy pick {model_pick}"
    if alignment == "review":
        return "REVIEW_NEUTRAL", f"{state or 'WATCH'} signal is not strong enough to confirm or reject"
    if not pick:
        return "NO_SIGNAL", "No market-specific overlay signal available"
    if pick == model_pick and state in {"BOOST", "WATCH"}:
        return "SUPPORTS_PICK", f"{pick} {state} signal matches deploy pick"
    if pick != model_pick and state == "AVOID":
        return "CONTRADICTS_PICK", f"{pick} AVOID signal conflicts with deploy pick {model_pick}"
    return "REVIEW_NEUTRAL", f"{pick or 'WATCH'} {state or 'WATCH'} signal needs manual review"


def player_context(player: pd.DataFrame) -> pd.DataFrame:
    if player.empty or "fixture_key" not in player.columns:
        return pd.DataFrame(columns=["fixture_key"])
    work = player.copy()
    work["market_family"] = work.get("market_family", "").astype(str)
    work["confidence_label"] = work.get("confidence_label", "").astype(str)
    work["predicted_hit_rate_pct"] = pd.to_numeric(work.get("predicted_hit_rate_pct"), errors="coerce")
    strong = work["confidence_label"].isin(STRONG_PLAYER_LABELS)
    goal = work["market_family"].isin(GOAL_PLAYER_MARKETS)
    contact = work["market_family"].isin(CONTACT_PLAYER_MARKETS)
    grouped = work.groupby("fixture_key", dropna=False)
    rows = []
    for fixture_key, group in grouped:
        strong_group = group[group["confidence_label"].isin(STRONG_PLAYER_LABELS)]
        goal_group = group[group["market_family"].isin(GOAL_PLAYER_MARKETS) & group["confidence_label"].isin(STRONG_PLAYER_LABELS)]
        contact_group = group[group["market_family"].isin(CONTACT_PLAYER_MARKETS) & group["confidence_label"].isin(STRONG_PLAYER_LABELS)]
        rows.append(
            {
                "fixture_key": fixture_key,
                "player_event_rows": int(len(group)),
                "player_event_strong_rows": int(len(strong_group)),
                "player_goal_pressure_rows": int(len(goal_group)),
                "player_contact_pressure_rows": int(len(contact_group)),
                "player_goal_pressure_avg_hit_rate": round(float(goal_group["predicted_hit_rate_pct"].mean()), 2)
                if not goal_group.empty
                else "",
                "player_context_note": player_note(len(goal_group), len(contact_group)),
            }
        )
    return pd.DataFrame(rows)


def player_note(goal_rows: int, contact_rows: int) -> str:
    parts = []
    if goal_rows >= 8:
        parts.append("strong player goal-event pressure")
    elif goal_rows > 0:
        parts.append("some player goal-event pressure")
    else:
        parts.append("no strong player goal-event pressure")
    if contact_rows >= 8:
        parts.append("strong contact/card pressure")
    elif contact_rows > 0:
        parts.append("some contact/card pressure")
    return "; ".join(parts)


def fixture_context(fixture_market: pd.DataFrame) -> pd.DataFrame:
    if fixture_market.empty or "fixture_key" not in fixture_market.columns:
        return pd.DataFrame(columns=["fixture_key"])
    work = fixture_market.copy()
    work["watch_priority"] = work.get("watch_priority", "").astype(str)
    work["shadow_stage"] = work.get("shadow_stage", "").astype(str)
    confirm = work["watch_priority"].eq("PRIORITY_CONFIRM")
    rows = []
    for fixture_key, group in work.groupby("fixture_key", dropna=False):
        confirm_group = group[group["watch_priority"].eq("PRIORITY_CONFIRM")]
        rows.append(
            {
                "fixture_key": fixture_key,
                "fixture_shadow_rows": int(len(group)),
                "fixture_priority_confirm_rows": int(len(confirm_group)),
                "fixture_priority_confirm_stages": "|".join(dict.fromkeys(confirm_group["shadow_stage"].astype(str))),
            }
        )
    return pd.DataFrame(rows)


def load_h2h_context(h2h_root: Path | None) -> pd.DataFrame:
    if h2h_root is None or not h2h_root.exists():
        return pd.DataFrame(columns=["fixture_key"])
    roots = [h2h_root]
    nested = h2h_root / "fixture_h2h_support"
    if nested.exists():
        roots.insert(0, nested)
    rows = []
    seen = set()
    for root in roots:
        for path in sorted(root.glob("*.json")):
            if path.name == "index.json":
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            fixture_key = clean_text(payload.get("fixture_key") or path.stem)
            if not fixture_key or fixture_key in seen:
                continue
            seen.add(fixture_key)
            rows.append(
                {
                    "fixture_key": fixture_key,
                    "h2h_sample_size": int(float(payload.get("sample_size") or 0)),
                    "h2h_goal_environment": payload.get("goal_environment", ""),
                    "h2h_btts_regime": payload.get("btts_regime", ""),
                    "h2h_over25_rate": payload.get("over25_rate", ""),
                    "h2h_draw_rate": payload.get("draw_rate", ""),
                    "h2h_booking_heat": payload.get("booking_heat", ""),
                    "h2h_fallback_mode": payload.get("fallback_mode", ""),
                    "h2h_coverage_status": payload.get("coverage_status", ""),
                    "h2h_summary": payload.get("summary", ""),
                }
            )
    return pd.DataFrame(rows)


def h2h_number(row: pd.Series, col: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def h2h_verdict(row: pd.Series) -> tuple[str, str]:
    sample = int(float(row.get("h2h_sample_size") or 0))
    if sample <= 0:
        return "H2H_UNAVAILABLE", "No publish-safe H2H sample for this fixture"
    market = clean_text(row.get("market")).upper()
    pick = clean_text(row.get("normalized_model_pick")).upper()
    if market == "OU25":
        rate = h2h_number(row, "h2h_over25_rate")
        if rate is None:
            return "H2H_REVIEW", "H2H Over 2.5 rate missing"
        if pick == "OVER25" and rate >= 58:
            return "H2H_SUPPORTS_PICK", f"H2H Over 2.5 rate {rate:.0f}% supports OVER25"
        if pick == "OVER25" and rate <= 42:
            return "H2H_CONTRADICTS_PICK", f"H2H Over 2.5 rate {rate:.0f}% argues against OVER25"
        if pick == "UNDER25" and rate <= 42:
            return "H2H_SUPPORTS_PICK", f"H2H Over 2.5 rate {rate:.0f}% supports UNDER25"
        if pick == "UNDER25" and rate >= 58:
            return "H2H_CONTRADICTS_PICK", f"H2H Over 2.5 rate {rate:.0f}% argues against UNDER25"
        return "H2H_REVIEW", f"H2H Over 2.5 rate {rate:.0f}% is mixed"
    if market == "BTTS":
        rate = h2h_number(row, "h2h_btts_regime")
        if rate is None:
            return "H2H_REVIEW", "H2H BTTS regime missing"
        if pick == "YES" and rate >= 58:
            return "H2H_SUPPORTS_PICK", f"H2H BTTS rate {rate:.0f}% supports YES"
        if pick == "YES" and rate <= 42:
            return "H2H_CONTRADICTS_PICK", f"H2H BTTS rate {rate:.0f}% argues against YES"
        if pick == "NO" and rate <= 42:
            return "H2H_SUPPORTS_PICK", f"H2H BTTS rate {rate:.0f}% supports NO"
        if pick == "NO" and rate >= 58:
            return "H2H_CONTRADICTS_PICK", f"H2H BTTS rate {rate:.0f}% argues against NO"
        return "H2H_REVIEW", f"H2H BTTS rate {rate:.0f}% is mixed"
    if market == "FTR":
        draw = h2h_number(row, "h2h_draw_rate")
        if draw is None:
            return "H2H_REVIEW", "H2H draw regime missing"
        if pick == "DRAW" and draw >= 34:
            return "H2H_SUPPORTS_PICK", f"H2H draw rate {draw:.0f}% supports DRAW"
        if pick != "DRAW" and draw >= 40:
            return "H2H_CONTRADICTS_PICK", f"H2H draw rate {draw:.0f}% is a caution against a decisive FTR pick"
        return "H2H_REVIEW", f"H2H draw rate {draw:.0f}% does not strongly confirm a non-draw FTR pick"
    return "H2H_REVIEW", "H2H support not defined for this market"


def build_audit(compare: pd.DataFrame, player: pd.DataFrame, fixture_market: pd.DataFrame, h2h: pd.DataFrame) -> pd.DataFrame:
    if compare.empty:
        return pd.DataFrame()
    rows = compare[compare.get("market", "").astype(str).str.upper().isin(MARKET_SIGNAL_COLUMNS)].copy()
    rows["market"] = rows["market"].astype(str).str.upper()
    rows["normalized_model_pick"] = rows.apply(market_pick, axis=1)
    for market, cols in MARKET_SIGNAL_COLUMNS.items():
        pick_col, state_col, score_col = cols
        mask = rows["market"].eq(market)
        rows.loc[mask, "market_signal_pick"] = rows.loc[mask, pick_col].astype(str)
        rows.loc[mask, "market_signal_state"] = rows.loc[mask, state_col].astype(str)
        rows.loc[mask, "market_signal_score"] = rows.loc[mask, score_col]
    verdicts = rows.apply(verdict, axis=1, result_type="expand")
    rows["overlay_support_verdict"] = verdicts[0]
    rows["overlay_support_reason"] = verdicts[1]
    player_ctx = player_context(player)
    fixture_ctx = fixture_context(fixture_market)
    if not player_ctx.empty:
        rows = rows.merge(player_ctx, on="fixture_key", how="left")
    if not fixture_ctx.empty:
        rows = rows.merge(fixture_ctx, on="fixture_key", how="left")
    if not h2h.empty:
        rows = rows.merge(h2h, on="fixture_key", how="left")
    h2h_verdicts = rows.apply(h2h_verdict, axis=1, result_type="expand")
    rows["h2h_support_verdict"] = h2h_verdicts[0]
    rows["h2h_support_reason"] = h2h_verdicts[1]
    keep = [
        "match_date",
        "league",
        "fixture_key",
        "home_team",
        "away_team",
        "deploy_tier",
        "market",
        "normalized_model_pick",
        "model_prob",
        "bookie_implied_novig",
        "value_edge",
        "overlay_support_verdict",
        "overlay_support_reason",
        "market_signal_pick",
        "market_signal_state",
        "market_signal_score",
        "ftr_signal_pick",
        "ftr_signal_state",
        "ou25_signal_pick",
        "ou25_signal_state",
        "btts_signal_pick",
        "btts_signal_state",
        "team_intel_overlay_reason",
        "h2h_support_verdict",
        "h2h_support_reason",
        "h2h_sample_size",
        "h2h_goal_environment",
        "h2h_btts_regime",
        "h2h_over25_rate",
        "h2h_draw_rate",
        "h2h_booking_heat",
        "h2h_fallback_mode",
        "player_event_rows",
        "player_event_strong_rows",
        "player_goal_pressure_rows",
        "player_goal_pressure_avg_hit_rate",
        "player_contact_pressure_rows",
        "player_context_note",
        "fixture_shadow_rows",
        "fixture_priority_confirm_rows",
        "fixture_priority_confirm_stages",
    ]
    for col in keep:
        if col not in rows.columns:
            rows[col] = ""
    return rows[keep].sort_values(["overlay_support_verdict", "deploy_tier", "match_date", "league", "fixture_key"])


def write_markdown(outdir: Path, audit: pd.DataFrame, compare_path: Path, player_path: Path | None, fixture_path: Path | None, h2h_root: Path | None) -> None:
    counts = audit["overlay_support_verdict"].value_counts(dropna=False).to_dict() if not audit.empty else {}
    market_counts = (
        audit.groupby(["market", "overlay_support_verdict"], dropna=False).size().reset_index(name="rows")
        if not audit.empty
        else pd.DataFrame(columns=["market", "overlay_support_verdict", "rows"])
    )
    lines = [
        "# Goal Market Overlay Support Audit",
        "",
        "Reporting layer only. This does not alter deploy routing, tiers, vetoes, or slips.",
        "",
        "## Inputs",
        f"- compare rows: `{compare_path}`",
        f"- player-event dashboard: `{player_path or ''}`",
        f"- fixture-market board: `{fixture_path or ''}`",
        f"- H2H support root: `{h2h_root or ''}`",
        "",
        "## Verdict Counts",
    ]
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Market Counts", ""])
    if not market_counts.empty:
        lines.extend(markdown_table(market_counts))
    conflicts = audit[audit["overlay_support_verdict"].eq("CONTRADICTS_PICK")] if not audit.empty else pd.DataFrame()
    h2h_counts = (
        audit.groupby(["market", "h2h_support_verdict"], dropna=False).size().reset_index(name="rows")
        if not audit.empty
        else pd.DataFrame(columns=["market", "h2h_support_verdict", "rows"])
    )
    lines.extend(["", "## H2H Counts", ""])
    if not h2h_counts.empty:
        lines.extend(markdown_table(h2h_counts))
    lines.extend(["", "## Contradictions", ""])
    if conflicts.empty:
        lines.append("No contradictions flagged.")
    else:
        cols = ["match_date", "league", "home_team", "away_team", "deploy_tier", "market", "normalized_model_pick", "overlay_support_reason", "h2h_support_verdict", "h2h_support_reason"]
        lines.extend(markdown_table(conflicts[cols]))
    (outdir / "GOAL_MARKET_OVERLAY_SUPPORT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [clean_text(row.get(col)).replace("|", "/") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-rows", type=Path, required=True)
    parser.add_argument("--player-event-dashboard", type=Path, default=None)
    parser.add_argument("--fixture-market-board", type=Path, default=None)
    parser.add_argument("--h2h-root", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    compare = read_csv(args.compare_rows)
    player = read_csv(args.player_event_dashboard)
    fixture_market = read_csv(args.fixture_market_board)
    h2h = load_h2h_context(args.h2h_root)
    audit = build_audit(compare, player, fixture_market, h2h)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "GOAL_MARKET_OVERLAY_SUPPORT_AUDIT.csv"
    audit.to_csv(out_csv, index=False)
    write_markdown(args.outdir, audit, args.compare_rows, args.player_event_dashboard, args.fixture_market_board, args.h2h_root)
    print(f"WROTE {args.outdir}")
    print(f"rows={len(audit)}")
    if not audit.empty:
        print(audit.groupby(["market", "overlay_support_verdict"], dropna=False).size().reset_index(name="rows").to_string(index=False))


if __name__ == "__main__":
    main()
