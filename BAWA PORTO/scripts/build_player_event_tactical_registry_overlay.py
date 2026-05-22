#!/usr/bin/env python3
"""Overlay tactical registry tags onto player-event fixture input rows.

Research-only companion output. It enriches current player-event fixture inputs
with tactical feature IDs/families from the registry, without changing the
source fixture input file or production deploy logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_INPUTS = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "player_event_current_board_fixture_inputs_clean"
    / "CURRENT_BOARD_PLAYER_EVENT_FIXTURE_INPUTS_ALL.csv"
)
DEFAULT_REGISTRY = ROOT / "reports" / "2026-05-08" / "tactical_feature_registry" / "TACTICAL_FEATURE_REGISTRY.csv"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-08" / "player_event_tactical_registry_overlay"


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def registry_lookup(registry: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return registry.set_index("feature_id").to_dict(orient="index") if not registry.empty else {}


def feature_ids_for_row(row: pd.Series) -> list[str]:
    ids: list[str] = []
    role = clean_text(row.get("tactical_role")).lower()
    position = clean_text(row.get("position_group")).lower()
    lineup_flags = clean_text(row.get("lineup_watch_flags"))
    wide_score = float(row.get("fixture_wide_duel_score", 0) or 0)
    attack_pressure = float(row.get("fixture_attack_pressure_score", 0) or 0)
    territory = float(row.get("fixture_territorial_stress_score", 0) or 0)
    corner_pressure = float(row.get("fixture_corner_pressure_score", 0) or 0)
    ref_cards = float(row.get("ref_cards_per_match", 0) or 0)
    key_passes = float(row.get("key_passes_per90", 0) or 0)
    shots = float(row.get("shots_per90", 0) or 0)
    tackles = float(row.get("tackles_per90", 0) or 0)
    fouls_won = float(row.get("fouls_won_per90", 0) or 0)

    if attack_pressure >= 0.70 or territory >= 0.65:
        ids.append("GOAL_RATE_FIELD_TILT")
    if attack_pressure >= 0.75 and territory >= 0.60:
        ids.append("REST_DEFENCE_TRANSITION_EXPOSURE")
    if wide_score >= 0.70 or "wide" in role or "wing" in role:
        ids.append("WIDE_ISOLATION_DRIBBLER")
    if ("wide forward" in role or "inside" in role or "forward" in position) and shots >= 1.4:
        ids.append("WIDE_ISOLATION_INSIDE_FORWARD_SHOT")
    if "striker" in role or "forward" in position:
        ids.append("ROLE_ALLOWANCE_STRIKER_ARCHETYPE")
    if "holding" in role or "destroyer" in role or ("midfielder" in position and tackles >= 1.2):
        ids.append("ROLE_ALLOWANCE_MIDFIELD_DESTROYER")
    if "full" in role or "wing-back" in role or ("defender" in position and wide_score >= 0.55):
        ids.append("ROLE_ALLOWANCE_FULLBACK_OVERLAP")
    if corner_pressure >= 0.65:
        ids.append("SET_PIECE_CORNER_PRESSURE")
    if attack_pressure >= 0.75:
        ids.append("KEEPER_WORKLOAD_SOT_PRESSURE")
    if ref_cards >= 4.0 or float(row.get("fixture_foul_density_score", 0) or 0) >= 0.70:
        ids.append("CARD_FOUL_ECOSYSTEM")
    if lineup_flags and lineup_flags != "NO_LINEUP_WATCH_FLAG":
        ids.append("LINEUP_FRAGILITY_PRESS_BREAKER")
    if ("midfield" in role or "midfielder" in position) and lineup_flags:
        ids.append("LINEUP_FRAGILITY_REST_DEFENCE_MID")
    if key_passes >= 1.2:
        ids.append("ROLE_ALLOWANCE_FULLBACK_OVERLAP" if "defender" in position else "GOAL_RATE_CENTRAL_ACCESS")
    if fouls_won >= 1.2:
        ids.append("WIDE_ISOLATION_DRIBBLER")
    return list(dict.fromkeys(ids))


def enrich(fixtures: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    lookup = registry_lookup(registry)
    out = fixtures.copy()
    feature_lists = [feature_ids_for_row(row) for _, row in out.iterrows()]
    out["tactical_feature_ids"] = ["|".join(ids) for ids in feature_lists]
    out["tactical_feature_families"] = [
        "|".join(dict.fromkeys(str(lookup.get(fid, {}).get("family", "")) for fid in ids if lookup.get(fid)))
        for ids in feature_lists
    ]
    out["tactical_target_markets"] = [
        "|".join(dict.fromkeys(str(lookup.get(fid, {}).get("target_markets", "")) for fid in ids if lookup.get(fid)))
        for ids in feature_lists
    ]
    out["tactical_leakage_risk_max"] = [
        "MEDIUM" if any("MEDIUM" in str(lookup.get(fid, {}).get("leakage_risk", "")) for fid in ids)
        else ("LOW" if ids else "")
        for ids in feature_lists
    ]
    out["tactical_registry_note"] = np.where(
        out["tactical_feature_ids"].astype(str).ne(""),
        "TACTICAL_REGISTRY_TAGGED_RESEARCH_ONLY",
        "NO_TACTICAL_REGISTRY_TAG",
    )
    return out


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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


def write_report(outdir: Path, enriched: pd.DataFrame, fixture_inputs: Path, registry_path: Path) -> None:
    tagged = enriched[enriched["tactical_feature_ids"].astype(str).ne("")]
    family_counts = (
        tagged["tactical_feature_families"]
        .astype(str)
        .str.split("|")
        .explode()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .rename_axis("family")
        .reset_index(name="rows")
    )
    sample_cols = [
        "match_date",
        "league",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "tactical_role",
        "tactical_feature_ids",
        "tactical_feature_families",
    ]
    lines = [
        "# Player Event Tactical Registry Overlay",
        "",
        "Research-only overlay of tactical feature registry tags onto player-event fixture input rows.",
        "",
        "## Safety",
        "- Does not modify source fixture inputs.",
        "- Does not change production prediction, deploy routing, tiers, or slips.",
        "",
        "## Inputs",
        f"- fixture inputs: `{fixture_inputs}`",
        f"- registry: `{registry_path}`",
        "",
        "## Overall",
        f"- rows: `{len(enriched)}`",
        f"- tagged rows: `{len(tagged)}`",
        "",
        "## Family Counts",
        markdown_table(family_counts),
        "",
        "## Sample Tagged Rows",
        markdown_table(tagged[[c for c in sample_cols if c in tagged.columns]], max_rows=40),
    ]
    (outdir / "PLAYER_EVENT_TACTICAL_REGISTRY_OVERLAY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-inputs", type=Path, default=DEFAULT_FIXTURE_INPUTS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    if not args.fixture_inputs.exists():
        raise SystemExit(f"Missing fixture inputs: {args.fixture_inputs}")
    if not args.registry.exists():
        raise SystemExit(f"Missing tactical registry: {args.registry}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    fixtures = pd.read_csv(args.fixture_inputs, low_memory=False)
    registry = pd.read_csv(args.registry, low_memory=False)
    enriched = enrich(fixtures, registry)
    enriched.to_csv(args.outdir / "PLAYER_EVENT_TACTICAL_REGISTRY_OVERLAY.csv", index=False)
    write_report(args.outdir, enriched, args.fixture_inputs, args.registry)
    print(f"WROTE {args.outdir}")
    print(f"rows={len(enriched)} tagged={int(enriched['tactical_feature_ids'].astype(str).ne('').sum())}")


if __name__ == "__main__":
    main()
