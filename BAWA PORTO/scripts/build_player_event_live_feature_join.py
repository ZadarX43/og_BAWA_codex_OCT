#!/usr/bin/env python3
"""Join live/player-event boards to interaction proof features.

Research-only bridge for the player-event dashboard. It enriches a player-event
hit-rate band board with:

- rolling attacker recent form (`attacker_recent_*`)
- rolling opponent role allowance (`opp_attack_allowed_*`)

The output is still beta/intelligence-only and must not be treated as priced
player-prop odds or live deployment routing.
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
DEFAULT_RECENT_FORM = (
    ROOT / "reports" / "2026-05-06" / "player_attacker_recent_form_features" / "player_attacker_recent_form_features.csv"
)
DEFAULT_OPPONENT_ALLOWANCE = (
    ROOT
    / "reports"
    / "2026-05-06"
    / "player_event_opponent_attack_allowance_features"
    / "player_event_opponent_attack_allowance_features.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_event_live_feature_join"

JOIN_KEYS = ["fixture_key", "team_name", "player_name"]
RECENT_PREFIX = "attacker_recent_"
OPP_PREFIX = "opp_attack_allowed_"


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_text(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def add_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in JOIN_KEYS:
        if col not in out.columns:
            out[col] = ""
    out["_join_fixture_key"] = out["fixture_key"].fillna("").astype(str)
    out["_join_team_name"] = norm_text(out["team_name"])
    out["_join_player_name"] = norm_text(out["player_name"])
    return out


def select_feature_cols(path: Path, prefixes: tuple[str, ...]) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    requested = JOIN_KEYS + [
        "fixture_id",
        "team_id",
        "player_id",
        "league_tag",
        "season_tag",
        "match_date",
        "player_team_side",
        "attack_role_group",
        "tactical_role",
    ]
    feature_cols = [col for col in cols if col.startswith(prefixes)]
    return [col for col in requested + feature_cols if col in cols]


def load_feature_table(path: Path, prefixes: tuple[str, ...], source_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    usecols = select_feature_cols(path, prefixes)
    features = pd.read_csv(path, usecols=usecols, low_memory=False)
    features = add_join_keys(features)
    feature_cols = [col for col in features.columns if col.startswith(prefixes)]
    keep = ["_join_fixture_key", "_join_team_name", "_join_player_name"] + feature_cols
    optional_meta = [
        "fixture_id",
        "team_id",
        "player_id",
        "league_tag",
        "season_tag",
        "attack_role_group",
        "player_team_side",
    ]
    keep += [col for col in optional_meta if col in features.columns and col not in keep]
    deduped = features[keep].drop_duplicates(["_join_fixture_key", "_join_team_name", "_join_player_name"], keep="last")
    deduped[f"{source_name}_feature_count"] = deduped[feature_cols].notna().sum(axis=1) if feature_cols else 0
    return deduped


def coverage_report(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["league", "market_family", "threshold_name"]
    if enriched.empty:
        return pd.DataFrame()
    for key, group in enriched.groupby([col for col in group_cols if col in enriched.columns], dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip([col for col in group_cols if col in enriched.columns], key))
        recent_ready = group["attacker_recent_feature_count"].gt(0) if "attacker_recent_feature_count" in group.columns else pd.Series(False, index=group.index)
        opp_ready = group["opponent_attack_allowance_feature_count"].gt(0) if "opponent_attack_allowance_feature_count" in group.columns else pd.Series(False, index=group.index)
        exact_ready = recent_ready & opp_ready
        rows.append(
            {
                **rec,
                "rows": int(len(group)),
                "fixtures": int(group["fixture_key"].nunique()) if "fixture_key" in group.columns else 0,
                "players": int(group["player_name"].nunique()) if "player_name" in group.columns else 0,
                "recent_rows": int(recent_ready.sum()),
                "opponent_allowance_rows": int(opp_ready.sum()),
                "exact_interaction_ready_rows": int(exact_ready.sum()),
                "exact_interaction_ready_rate": float(exact_ready.mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["exact_interaction_ready_rows", "rows"], ascending=[False, False])


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


def write_report(outdir: Path, enriched: pd.DataFrame, coverage: pd.DataFrame, board_path: Path) -> None:
    exact_ready = (
        enriched["player_event_exact_interaction_feature_ready"].astype(bool)
        if "player_event_exact_interaction_feature_ready" in enriched.columns
        else pd.Series(False, index=enriched.index)
    )
    lines = [
        "# Player Event Live Feature Join",
        "",
        "Research-only enrichment bridge for exact player-event interaction shadow labels.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Joined features are context/watch intelligence only.",
        "",
        "## Source",
        f"- player-event board: `{board_path}`",
        "",
        "## Output",
        f"- rows: `{len(enriched)}`",
        f"- exact interaction ready rows: `{int(exact_ready.sum())}`",
        f"- exact interaction ready rate: `{float(exact_ready.mean()) if len(enriched) else 0.0:.2%}`",
        "",
        "## Coverage",
        markdown_table(coverage),
    ]
    (outdir / "PLAYER_EVENT_LIVE_FEATURE_JOIN.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-event-board", type=Path, default=DEFAULT_PLAYER_EVENT_BOARD)
    parser.add_argument("--recent-form", type=Path, default=DEFAULT_RECENT_FORM)
    parser.add_argument("--opponent-allowance", type=Path, default=DEFAULT_OPPONENT_ALLOWANCE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    if not args.player_event_board.exists():
        raise SystemExit(f"Missing player-event board: {args.player_event_board}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    board = pd.read_csv(args.player_event_board, low_memory=False)
    enriched = add_join_keys(board)

    recent = load_feature_table(args.recent_form, (RECENT_PREFIX,), "attacker_recent")
    if not recent.empty:
        enriched = enriched.merge(recent, on=["_join_fixture_key", "_join_team_name", "_join_player_name"], how="left")
    elif "attacker_recent_feature_count" not in enriched.columns:
        enriched["attacker_recent_feature_count"] = 0

    opponent = load_feature_table(args.opponent_allowance, (OPP_PREFIX,), "opponent_attack_allowance")
    if not opponent.empty:
        enriched = enriched.merge(
            opponent,
            on=["_join_fixture_key", "_join_team_name", "_join_player_name"],
            how="left",
            suffixes=("", "_opp_feature"),
        )
    elif "opponent_attack_allowance_feature_count" not in enriched.columns:
        enriched["opponent_attack_allowance_feature_count"] = 0

    for col in ["attacker_recent_feature_count", "opponent_attack_allowance_feature_count"]:
        if col not in enriched.columns:
            enriched[col] = 0
        enriched[col] = num(enriched[col]).fillna(0)
    enriched["player_event_exact_interaction_feature_ready"] = (
        enriched["attacker_recent_feature_count"].gt(0)
        & enriched["opponent_attack_allowance_feature_count"].gt(0)
    )
    enriched["player_event_feature_join_mode"] = np.where(
        enriched["player_event_exact_interaction_feature_ready"],
        "EXACT_INTERACTION_FEATURES_READY",
        "MISSING_RECENT_OR_OPPONENT_ALLOWANCE",
    )

    drop_cols = ["_join_fixture_key", "_join_team_name", "_join_player_name"]
    output = enriched.drop(columns=[col for col in drop_cols if col in enriched.columns])
    output_path = args.outdir / "PLAYER_EVENT_HITRATE_BAND_DASHBOARD__WITH_INTERACTION_FEATURES.csv"
    output.to_csv(output_path, index=False)
    coverage = coverage_report(output)
    coverage.to_csv(args.outdir / "PLAYER_EVENT_LIVE_FEATURE_JOIN_COVERAGE.csv", index=False)
    write_report(args.outdir, output, coverage, args.player_event_board)

    print(f"WROTE {args.outdir}")
    print(f"rows={len(output)} exact_ready={int(output['player_event_exact_interaction_feature_ready'].sum())}")


if __name__ == "__main__":
    main()
