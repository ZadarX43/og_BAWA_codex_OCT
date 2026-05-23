#!/usr/bin/env python3
"""Build player-event interaction watch rows for the live shadow dashboard.

Research-only. This promotes the recent-form x opponent-allowance proof cells
into forward-facing watch labels only. It does not create priced odds, deploy
picks, or mutations to source boards.

When exact recent-form/opponent-allowance columns are present, rows are stamped
as EXACT_INTERACTION. When they are not present, --allow-proof-label-only can
emit clearly marked proof-label watch rows from the existing hit-rate band
board so the app/dashboard can surface the lane while live feature joins are
being completed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYER_EVENT_BOARD = (
    ROOT / "reports" / "2026-05-06" / "player_event_hitrate_band_board" / "PLAYER_EVENT_HITRATE_BAND_DASHBOARD.csv"
)
DEFAULT_POLICY = (
    ROOT
    / "reports"
    / "2026-05-06"
    / "player_event_recent_form_opponent_allowance_interaction_audit"
    / "player_event_recent_form_opponent_allowance_top_candidates.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_event_interaction_live_shadow_board"


MARKET_MAP = {
    "shots_ge2": {
        "market_family": "PLAYER_SHOTS",
        "threshold": 1.5,
        "shadow_stage": "PLAYER_SHOTS_1_5_INTERACTION_WATCH",
        "expression": "Player Shots 1.5+",
    },
    "shots_ge3": {
        "market_family": "PLAYER_SHOTS",
        "threshold": 2.5,
        "shadow_stage": "PLAYER_SHOTS_2_5_INTERACTION_WATCH",
        "expression": "Player Shots 2.5+",
    },
    "shots_on_target": {
        "market_family": "PLAYER_SOT",
        "threshold": 0.5,
        "shadow_stage": "PLAYER_SOT_0_5_INTERACTION_WATCH",
        "expression": "Player SOT 0.5+",
    },
    "fouled_ge1": {
        "market_family": "PLAYER_FOULED",
        "threshold": 0.5,
        "shadow_stage": "PLAYER_FOULED_0_5_INTERACTION_WATCH",
        "expression": "Player Fouled 0.5+",
        "default_policy": {
            "market": "fouled_ge1",
            "recent_feature": "attacker_recent_fouls_won_per90_l8",
            "recent_threshold": 1.377551,
            "opponent_feature": "opp_attack_allowed_role_fouls_drawn_per_player_l10",
            "opponent_threshold": 1.272727,
            "context_feature": "fixture_foul_density_score",
            "context_threshold": 0.7496,
            "interaction_label": "FOULED_INTERACTION_CORE",
            "interaction_hit": 0.786820,
            "interaction_rows": 3748,
            "lift_vs_recent_only": 0.050414,
            "lift_vs_opponent_only": 0.073990,
        },
    },
    "fouled_ge2": {
        "market_family": "PLAYER_FOULED",
        "threshold": 1.5,
        "shadow_stage": "PLAYER_FOULED_1_5_INTERACTION_WATCH",
        "expression": "Player Fouled 1.5+",
        "default_policy": {
            "market": "fouled_ge2",
            "recent_feature": "attacker_recent_fouls_won_per90_l8",
            "recent_threshold": 1.377551,
            "opponent_feature": "opp_attack_allowed_role_fouls_drawn_per_match_l10",
            "opponent_threshold": 2.0,
            "context_feature": "fixture_wide_duel_score",
            "context_threshold": 0.9137,
            "interaction_label": "FOULED_INTERACTION_CORE",
            "interaction_hit": 0.476114,
            "interaction_rows": 3098,
            "lift_vs_recent_only": 0.060042,
            "lift_vs_opponent_only": 0.097323,
        },
    },
}

CONFIDENCE_PRIORITY = {
    "SHADOW_CORE": 100,
    "STRONG_WATCH": 80,
    "ALT_WATCH": 70,
    "WATCH": 60,
    "INFO_ONLY": 20,
}

SHADOW_COLUMNS = [
    "shadow_family",
    "shadow_stage",
    "fixture_key",
    "match_date",
    "league",
    "home_team_name",
    "away_team_name",
    "expression",
    "source_market",
    "source_selection",
    "source_deploy_tier",
    "source_tier",
    "combo_product",
    "combo_tier",
    "bookie_od",
    "model_prob",
    "value_edge",
    "value_edge_tier",
    "backtest_hit_rate",
    "backtest_graded",
    "league_stability_bucket",
    "team_stability_bucket",
    "watch_priority",
    "watch_flag",
    "guardrail",
    "reason",
    "player_name",
    "team_name",
    "player_team_side",
    "position_group",
    "tactical_role",
    "predicted_hit_rate",
    "predicted_hit_rate_pct",
    "confidence_label",
    "support_score",
    "context_reason_codes",
    "lineup_watch_flags",
    "expected_minutes",
    "formation_matchup_label",
    "formation_pressure_score",
    "fixture_style_label",
    "fixture_attacking_style_label",
    "fixture_foul_density_score",
    "fixture_wide_duel_score",
    "fixture_territorial_stress_score",
    "fixture_attack_pressure_score",
    "referee_name",
    "ref_cards_per_match",
    "interaction_match_mode",
    "interaction_recent_feature",
    "interaction_opponent_feature",
    "interaction_context_feature",
    "interaction_context_threshold",
    "interaction_lift_vs_recent_only",
    "interaction_lift_vs_opponent_only",
    "interaction_label",
    "fouled_context_cell",
    "fouled_context_cell_label",
]


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def scalar(row: pd.Series, col: str, default: Any = "") -> Any:
    if col not in row.index:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    return value


def market_policy(
    policy: pd.DataFrame,
    market: str,
    available_columns: set[str],
    default_policy: dict[str, Any] | None = None,
) -> pd.Series | None:
    if policy.empty or "market" not in policy.columns:
        return pd.Series(default_policy) if default_policy else None
    rows = policy[policy["market"].astype("string").eq(market)].copy()
    if rows.empty:
        return pd.Series(default_policy) if default_policy else None
    rows = rows[
        rows["recent_feature"].astype("string").isin(available_columns)
        & rows["opponent_feature"].astype("string").isin(available_columns)
    ].copy()
    if rows.empty:
        return pd.Series(default_policy) if default_policy else None
    label_rank = {"INTERACTION_CORE": 0, "INTERACTION_READY": 1, "CONFIRMER_ONLY": 2, "WATCH": 3}
    rows["_rank"] = rows["interaction_label"].map(label_rank).fillna(9)
    rows = rows.sort_values(
        ["_rank", "interaction_hit", "lift_vs_recent_only", "interaction_rows"],
        ascending=[True, False, False, False],
    )
    return rows.iloc[0]


def exact_mask(board: pd.DataFrame, policy_row: pd.Series) -> pd.Series | None:
    recent_feature = str(policy_row.get("recent_feature", ""))
    opponent_feature = str(policy_row.get("opponent_feature", ""))
    if recent_feature not in board.columns or opponent_feature not in board.columns:
        return None
    mask = (
        num(board[recent_feature]).ge(float(policy_row.get("recent_threshold", np.nan)))
        & num(board[opponent_feature]).ge(float(policy_row.get("opponent_threshold", np.nan)))
    )
    context_feature = str(policy_row.get("context_feature", ""))
    if context_feature and context_feature != "nan":
        if context_feature not in board.columns:
            return None
        mask = mask & num(board[context_feature]).ge(float(policy_row.get("context_threshold", np.nan)))
    return mask


def proof_label_mask(board: pd.DataFrame) -> pd.Series:
    confidence = board.get("confidence_label", pd.Series("", index=board.index)).astype("string")
    return confidence.isin(["SHADOW_CORE", "STRONG_WATCH", "ALT_WATCH", "WATCH"])


def watch_priority(row: pd.Series, policy_row: pd.Series, exact: bool) -> str:
    label = str(policy_row.get("interaction_label", ""))
    market = str(policy_row.get("market", ""))
    confidence = str(row.get("confidence_label", ""))
    if exact and label == "FOULED_INTERACTION_CORE" and market == "fouled_ge1" and confidence == "SHADOW_CORE":
        return "PRIORITY_CORE"
    if exact and label == "FOULED_INTERACTION_CORE" and market == "fouled_ge2":
        return "PRIORITY_CONFIRM"
    if exact and label == "INTERACTION_CORE" and confidence in {"SHADOW_CORE", "STRONG_WATCH"}:
        return "PRIORITY_CORE"
    if label == "INTERACTION_CORE" and confidence in {"SHADOW_CORE", "STRONG_WATCH"}:
        return "PRIORITY_CONFIRM"
    if label in {"INTERACTION_READY", "CONFIRMER_ONLY"}:
        return "PRIORITY_CONFIRM"
    return "WATCH_ONLY_NOT_PROMOTION"


def context_bucket(value: Any, low: float, high: float, labels: tuple[str, str, str]) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if np.isnan(numeric):
        numeric = 0.0
    if numeric >= high:
        return labels[2]
    if numeric >= low:
        return labels[1]
    return labels[0]


def fouled_context_cell(row: pd.Series, market: str) -> tuple[str, str]:
    if market not in {"fouled_ge1", "fouled_ge2"}:
        return "", ""
    role = str(row.get("tactical_role", "") or "UNKNOWN")
    formation_bucket = context_bucket(
        row.get("formation_pressure_score", 0),
        0.16,
        0.35,
        ("FORMATION_LOW", "FORMATION_MED", "FORMATION_HIGH"),
    )
    foul_bucket = context_bucket(
        row.get("fixture_foul_density_score", 0),
        0.70,
        0.82,
        ("FOUL_DENSITY_LOW", "FOUL_DENSITY_MED", "FOUL_DENSITY_HIGH"),
    )
    wide_bucket = context_bucket(
        row.get("fixture_wide_duel_score", 0),
        0.88,
        0.94,
        ("WIDE_DUEL_LOW", "WIDE_DUEL_MED", "WIDE_DUEL_HIGH"),
    )
    cell = f"{role}|{formation_bucket}|{foul_bucket}|{wide_bucket}"
    ready_ge1 = {
        "Wide forward|FORMATION_LOW|FOUL_DENSITY_HIGH|WIDE_DUEL_HIGH",
        "Wide midfielder / winger|FORMATION_LOW|FOUL_DENSITY_HIGH|WIDE_DUEL_HIGH",
        "Wide midfielder / winger|FORMATION_LOW|FOUL_DENSITY_HIGH|WIDE_DUEL_MED",
    }
    watch_ge1 = {
        "Wide forward|FORMATION_HIGH|FOUL_DENSITY_HIGH|WIDE_DUEL_HIGH",
        "Wide midfielder / winger|FORMATION_LOW|FOUL_DENSITY_HIGH|WIDE_DUEL_LOW",
        "Wide midfielder / winger|FORMATION_LOW|FOUL_DENSITY_MED|WIDE_DUEL_MED",
        "Wide midfielder / winger|FORMATION_LOW|FOUL_DENSITY_MED|WIDE_DUEL_HIGH",
    }
    core_ge2 = {
        "Wide midfielder / winger|FORMATION_LOW|FOUL_DENSITY_HIGH|WIDE_DUEL_HIGH",
        "Wide forward|FORMATION_LOW|FOUL_DENSITY_HIGH|WIDE_DUEL_HIGH",
    }
    ready_ge2 = {
        "Wide midfielder / winger|FORMATION_LOW|FOUL_DENSITY_HIGH|WIDE_DUEL_MED",
    }
    watch_ge2 = {
        "Wide forward|FORMATION_HIGH|FOUL_DENSITY_HIGH|WIDE_DUEL_HIGH",
        "Wide forward|FORMATION_LOW|FOUL_DENSITY_HIGH|WIDE_DUEL_MED",
    }
    if market == "fouled_ge1" and cell in ready_ge1:
        return cell, "FOULED_CONTEXT_READY"
    if market == "fouled_ge1" and cell in watch_ge1:
        return cell, "FOULED_CONTEXT_WATCH"
    if market == "fouled_ge2" and cell in core_ge2:
        return cell, "FOULED_CONTEXT_CORE"
    if market == "fouled_ge2" and cell in ready_ge2:
        return cell, "FOULED_CONTEXT_READY"
    if market == "fouled_ge2" and cell in watch_ge2:
        return cell, "FOULED_CONTEXT_WATCH"
    return cell, ""


def build_shadow_board(
    board: pd.DataFrame,
    policy: pd.DataFrame,
    *,
    allow_proof_label_only: bool,
    per_market_limit: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for market, config in MARKET_MAP.items():
        rows = board[
            board.get("market_family", pd.Series("", index=board.index)).astype("string").eq(config["market_family"])
            & num(board.get("threshold", np.nan)).round(3).eq(float(config["threshold"]))
        ].copy()
        if rows.empty:
            continue
        policy_row = market_policy(policy, market, set(rows.columns), config.get("default_policy"))
        if policy_row is None:
            continue

        mask = exact_mask(rows, policy_row)
        match_mode = "EXACT_INTERACTION"
        if mask is None:
            if not allow_proof_label_only:
                continue
            mask = proof_label_mask(rows)
            match_mode = "PROOF_LABEL_ONLY_MISSING_LIVE_FEATURES"
        selected = rows[mask].copy()
        if selected.empty and allow_proof_label_only:
            selected = rows[proof_label_mask(rows)].copy()
            match_mode = "PROOF_LABEL_ONLY_MISSING_LIVE_FEATURES"
        if selected.empty:
            continue

        selected["_confidence_rank"] = selected.get("confidence_label", "").map(CONFIDENCE_PRIORITY).fillna(0)
        selected = selected.sort_values(
            ["_confidence_rank", "predicted_hit_rate", "support_score", "match_date"],
            ascending=[False, False, False, True],
        ).head(per_market_limit)

        for _, row in selected.iterrows():
            exact = match_mode == "EXACT_INTERACTION"
            priority = watch_priority(row, policy_row, exact)
            context_cell, context_label = fouled_context_cell(row, market)
            records.append(
                {
                    "shadow_family": "PLAYER_EVENT_INTERACTION",
                    "shadow_stage": config["shadow_stage"],
                    "fixture_key": scalar(row, "fixture_key"),
                    "match_date": scalar(row, "match_date"),
                    "league": scalar(row, "league", scalar(row, "competition")),
                    "home_team_name": scalar(row, "home_team_name"),
                    "away_team_name": scalar(row, "away_team_name"),
                    "expression": config["expression"],
                    "source_market": scalar(row, "market_family"),
                    "source_selection": scalar(row, "player_name"),
                    "source_deploy_tier": "PLAYER_EVENT_BETA",
                    "source_tier": scalar(row, "confidence_label"),
                    "combo_product": "",
                    "combo_tier": scalar(row, "confidence_label"),
                    "bookie_od": np.nan,
                    "model_prob": scalar(row, "predicted_hit_rate", np.nan),
                    "value_edge": np.nan,
                    "value_edge_tier": "",
                    "backtest_hit_rate": policy_row.get("interaction_hit", np.nan),
                    "backtest_graded": policy_row.get("interaction_rows", np.nan),
                    "league_stability_bucket": "",
                    "team_stability_bucket": "",
                    "watch_priority": priority,
                    "watch_flag": True,
                    "guardrail": (
                        "PLAYER_EVENT_BETA_ONLY|NO_PRICED_ODDS|NO_DEPLOY_PROMOTION|"
                        f"{match_mode}"
                    ),
                    "reason": (
                        f"{policy_row.get('interaction_label', '')}|"
                        f"lift_vs_recent={policy_row.get('lift_vs_recent_only', np.nan)}|"
                        f"lift_vs_opp={policy_row.get('lift_vs_opponent_only', np.nan)}|"
                        f"recent={policy_row.get('recent_feature', '')}|"
                        f"opp={policy_row.get('opponent_feature', '')}|"
                        f"context={policy_row.get('context_feature', '')}"
                    ),
                    "player_name": scalar(row, "player_name"),
                    "team_name": scalar(row, "team_name"),
                    "player_team_side": scalar(row, "player_team_side"),
                    "position_group": scalar(row, "position_group"),
                    "tactical_role": scalar(row, "tactical_role"),
                    "predicted_hit_rate": scalar(row, "predicted_hit_rate", np.nan),
                    "predicted_hit_rate_pct": scalar(row, "predicted_hit_rate_pct", np.nan),
                    "confidence_label": scalar(row, "confidence_label"),
                    "support_score": scalar(row, "support_score", np.nan),
                    "context_reason_codes": scalar(row, "context_reason_codes"),
                    "lineup_watch_flags": scalar(row, "lineup_watch_flags"),
                    "expected_minutes": scalar(row, "expected_minutes", np.nan),
                    "formation_matchup_label": scalar(row, "formation_matchup_label"),
                    "formation_pressure_score": scalar(row, "formation_pressure_score", np.nan),
                    "fixture_style_label": scalar(row, "fixture_style_label"),
                    "fixture_attacking_style_label": scalar(row, "fixture_attacking_style_label"),
                    "fixture_foul_density_score": scalar(row, "fixture_foul_density_score", np.nan),
                    "fixture_wide_duel_score": scalar(row, "fixture_wide_duel_score", np.nan),
                    "fixture_territorial_stress_score": scalar(row, "fixture_territorial_stress_score", np.nan),
                    "fixture_attack_pressure_score": scalar(row, "fixture_attack_pressure_score", np.nan),
                    "referee_name": scalar(row, "referee_name"),
                    "ref_cards_per_match": scalar(row, "ref_cards_per_match", np.nan),
                    "interaction_match_mode": match_mode,
                    "interaction_recent_feature": policy_row.get("recent_feature", ""),
                    "interaction_opponent_feature": policy_row.get("opponent_feature", ""),
                    "interaction_context_feature": policy_row.get("context_feature", ""),
                    "interaction_context_threshold": policy_row.get("context_threshold", np.nan),
                    "interaction_lift_vs_recent_only": policy_row.get("lift_vs_recent_only", np.nan),
                    "interaction_lift_vs_opponent_only": policy_row.get("lift_vs_opponent_only", np.nan),
                    "interaction_label": policy_row.get("interaction_label", ""),
                    "fouled_context_cell": context_cell,
                    "fouled_context_cell_label": context_label,
                }
            )
    if not records:
        return pd.DataFrame(columns=SHADOW_COLUMNS)
    return pd.DataFrame(records).reindex(columns=SHADOW_COLUMNS)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows)
    cols = list(work.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = round(value, 6)
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, shadow: pd.DataFrame, board_path: Path, policy_path: Path) -> None:
    summary = (
        shadow.groupby(["shadow_stage", "interaction_match_mode", "watch_priority"], dropna=False)
        .size()
        .reset_index(name="rows")
        if not shadow.empty
        else pd.DataFrame(columns=["shadow_stage", "interaction_match_mode", "watch_priority", "rows"])
    )
    top_cols = [
        "match_date",
        "league",
        "home_team_name",
        "away_team_name",
        "player_name",
        "team_name",
        "expression",
        "predicted_hit_rate_pct",
        "watch_priority",
        "interaction_match_mode",
        "interaction_label",
    ]
    lines = [
        "# Player Event Interaction Live Shadow Board",
        "",
        "Research-only watch labels for recent attacker form x opponent role allowance cells.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing or source board mutation.",
        "- `PROOF_LABEL_ONLY_MISSING_LIVE_FEATURES` means exact live recent/opponent interaction columns were not available yet.",
        "",
        "## Sources",
        f"- player-event board: `{board_path}`",
        f"- interaction policy: `{policy_path}`",
        "",
        "## Counts",
        markdown_table(summary),
        "",
        "## Top Watch Rows",
        markdown_table(shadow[[c for c in top_cols if c in shadow.columns]], max_rows=40) if not shadow.empty else "_No rows._",
    ]
    (outdir / "PLAYER_EVENT_INTERACTION_LIVE_SHADOW_BOARD.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-event-board", type=Path, default=DEFAULT_PLAYER_EVENT_BOARD)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--per-market-limit", type=int, default=200)
    parser.add_argument("--allow-proof-label-only", action="store_true")
    args = parser.parse_args()

    if not args.player_event_board.exists():
        raise SystemExit(f"Missing player-event board: {args.player_event_board}")
    if not args.policy.exists():
        raise SystemExit(f"Missing interaction policy: {args.policy}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    board = pd.read_csv(args.player_event_board, low_memory=False)
    policy = pd.read_csv(args.policy, low_memory=False)
    shadow = build_shadow_board(
        board,
        policy,
        allow_proof_label_only=args.allow_proof_label_only,
        per_market_limit=args.per_market_limit,
    )
    out_path = args.outdir / "PLAYER_EVENT_INTERACTION_LIVE_SHADOW_BOARD.csv"
    shadow.to_csv(out_path, index=False)
    write_report(args.outdir, shadow, args.player_event_board, args.policy)

    print(f"WROTE {args.outdir}")
    print(f"rows={len(shadow)}")
    if not shadow.empty:
        print(
            shadow.groupby(["shadow_stage", "interaction_match_mode", "watch_priority"], dropna=False)
            .size()
            .reset_index(name="rows")
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
