from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = REPO_ROOT / "data_sources" / "api_football" / "features" / "player_events"


PORTABLE_PRESET_MAP = {
    "tackles": {
        "4231v442": "CORE_PORTABLE",
        "4231v433": "CORE_PORTABLE",
        "3421v4231": "HIGH_HITRATE_NEEDS_MORE_SAMPLE",
    },
    "fouls_committed": {
        "4231v442": "CORE_PORTABLE",
        "4231v433": "CONDITIONAL_PORTABLE",
        "3421v4231": "PROBE_ONLY",
    },
    "shots": {
        "4231v433": "CORE_PORTABLE",
        "3421v4231": "CORE_PORTABLE",
        "4231v442": "SUPPORT_PORTABLE",
    },
    "shots_on_target": {
        "4231v433": "CORE_PORTABLE",
        "3421v4231": "CORE_PORTABLE",
        "4231v442": "CONDITIONAL_PORTABLE",
    },
}

ELITE_MARKET_RULES = {
    "tackles": {
        "allowed_tiers": {"CORE_PORTABLE", "HIGH_HITRATE_NEEDS_MORE_SAMPLE"},
        "allowed_buckets": {"P1_SUPER_ELITE", "P2_CONTACT_STACK"},
        "min_score": 97.0,
        "min_quality": 0.74,
        "max_per_fixture": 1,
    },
    "fouls_committed": {
        "allowed_tiers": {"CORE_PORTABLE"},
        "allowed_buckets": {"P1_SUPER_ELITE", "P2_CONTACT_STACK"},
        "min_score": 100.0,
        "min_quality": 0.74,
        "max_per_fixture": 1,
    },
    "shots": {
        "allowed_tiers": {"CORE_PORTABLE"},
        "allowed_buckets": {"P1_SUPER_ELITE", "P2_ATTACK_STACK"},
        "min_score": 108.0,
        "min_quality": 0.75,
        "max_per_fixture": 1,
    },
    "shots_on_target": {
        "allowed_tiers": {"CORE_PORTABLE"},
        "allowed_buckets": {"P1_SUPER_ELITE", "P2_ATTACK_STACK"},
        "min_score": 114.0,
        "min_quality": 0.75,
        "max_per_fixture": 1,
    },
}


def _batch_tag_from_board_name(name: str) -> str:
    stem = Path(name).stem
    return stem.split("__master_specialist_board")[0].replace("__", "").upper()


def _load_role_lookup(batch_tags: set[str]) -> pd.DataFrame:
    frames = []
    for batch_tag in sorted(batch_tags):
        path = FEATURES_DIR / f"player_events_fixture_input__{batch_tag}__2024.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            continue
        keep = [
            "fixture_key",
            "team_name",
            "player_name",
            "player_team_side",
            "position_group",
            "tactical_role",
            "blocks_per90",
            "duels_total_per90",
            "duels_won_per90",
            "aerial_duel_loss_rate",
            "cb_duel_pressure_score",
            "cb_front_foot_duel_flag",
            "opponent_striker_profile",
            "opponent_striker_pressure_tag",
            "opponent_striker_context_note",
            "opponent_striker_subtype_note",
        ]
        keep = [c for c in keep if c in df.columns]
        if {"fixture_key", "team_name", "player_name"}.issubset(keep):
            frames.append(df[keep].drop_duplicates(subset=["fixture_key", "team_name", "player_name"]))
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["fixture_key", "team_name", "player_name"]) if frames else pd.DataFrame()


def build_presets(inputs: list[str], output_csv: str, output_md: str) -> pd.DataFrame:
    frames = []
    for path in inputs:
        csv_path = Path(path)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty:
            continue
        df["source_batch_board"] = csv_path.name
        df["source_batch_tag"] = _batch_tag_from_board_name(csv_path.name)
        frames.append(df)

    board = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if board.empty:
        out = pd.DataFrame()
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Portable Market Specialist Presets\n\nNo rows matched.\n")
        return out

    role_lookup = _load_role_lookup(set(board["source_batch_tag"].astype(str).unique()))
    if not role_lookup.empty:
        for col in ["player_team_side", "position_group", "tactical_role"]:
            if col not in board.columns:
                board[col] = pd.NA
        board = board.merge(
            role_lookup,
            on=["fixture_key", "team_name", "player_name"],
            how="left",
            suffixes=("", "_lookup"),
        )
        for col in ["player_team_side", "position_group", "tactical_role"]:
            lookup_col = f"{col}_lookup"
            if lookup_col in board.columns:
                board[col] = board[col].fillna(board[lookup_col])
                board = board.drop(columns=[lookup_col])

    board["preset_tier"] = board.apply(
        lambda row: PORTABLE_PRESET_MAP.get(str(row.get("market", "")), {}).get(str(row.get("source_family_tag", "")), ""),
        axis=1,
    )
    board = board[board["preset_tier"].astype(str).ne("")].copy()
    if board.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Portable Market Specialist Presets\n\nNo rows matched.\n")
        return board

    board["preset_rank"] = board["preset_tier"].map(
        {
            "CORE_PORTABLE": 1,
            "SUPPORT_PORTABLE": 2,
            "CONDITIONAL_PORTABLE": 3,
            "HIGH_HITRATE_NEEDS_MORE_SAMPLE": 4,
            "PROBE_ONLY": 5,
        }
    ).fillna(9)

    elite_frames = []
    for market, rules in ELITE_MARKET_RULES.items():
        sub = board[board["market"].astype(str).eq(market)].copy()
        if sub.empty:
            continue
        base = sub[
            sub["preset_tier"].astype(str).isin(rules["allowed_tiers"])
            & sub["priority_bucket"].astype(str).isin(rules["allowed_buckets"])
            & pd.to_numeric(sub["market_score"], errors="coerce").fillna(0.0).ge(rules["min_score"])
            & pd.to_numeric(sub["fixture_quality_score"], errors="coerce").fillna(0.0).ge(rules["min_quality"])
        ].copy()
        sub = base
        if market in {"fouls_committed", "tackles"}:
            sub = sub[
                pd.to_numeric(sub["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-3.0)
            ].copy()
            cb_extra = board[
                board["market"].astype(str).eq(market)
                & board["priority_bucket"].astype(str).isin({"P1_SUPER_ELITE", "P2_CONTACT_STACK"})
                & board["tactical_role"].astype(str).eq("Centre-back enforcer")
                & pd.to_numeric(board.get("cb_front_foot_duel_flag", 0), errors="coerce").fillna(0.0).ge(1.0)
                & pd.to_numeric(board.get("cb_duel_pressure_score", 0.0), errors="coerce").fillna(0.0).ge(0.58)
                & board.get("opponent_striker_profile", pd.Series("UNSET", index=board.index)).astype(str).ne("UNSET")
                & pd.to_numeric(board["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-3.0)
                & pd.to_numeric(board["fixture_quality_score"], errors="coerce").fillna(0.0).ge(0.70)
                & pd.to_numeric(board["market_score"], errors="coerce").fillna(0.0).ge(rules["min_score"] - 8.0)
            ].copy()
            if not cb_extra.empty:
                cb_extra["cb_duel_slice_flag"] = 1
                sub["cb_duel_slice_flag"] = sub.get("cb_duel_slice_flag", 0)
                sub = pd.concat([sub, cb_extra], ignore_index=True).drop_duplicates(
                    subset=["fixture_key", "team_name", "player_name", "market"],
                    keep="first",
                )
        if market in {"shots", "shots_on_target"}:
            sub = sub[
                pd.to_numeric(sub["starting_xi_quality_edge"], errors="coerce").fillna(0.0).ge(3.0)
            ].copy()
        if sub.empty:
            continue
        if "cb_duel_slice_flag" not in sub.columns:
            sub["cb_duel_slice_flag"] = 0
        sub = (
            sub.sort_values(
                ["cb_duel_slice_flag", "preset_rank", "priority_rank", "market_score", "fixture_quality_score", "player_quality_score_l5"],
                ascending=[False, True, True, False, False, False],
            )
            .groupby(["market", "fixture_key"], group_keys=False)
            .head(rules["max_per_fixture"])
        )
        elite_frames.append(sub)

    out = pd.concat(elite_frames, ignore_index=True) if elite_frames else pd.DataFrame(columns=board.columns)
    if not out.empty:
        out = (
            out.sort_values(
                ["market", "preset_rank", "priority_rank", "market_score", "fixture_quality_score"],
                ascending=[True, True, True, False, False],
            )
            .groupby("fixture_key", group_keys=False)
            .head(3)
            .reset_index(drop=True)
        )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Portable Market Specialist Presets", "", "- elite subset rules applied:", "- only trusted portable tiers by market", "- only `P1_SUPER_ELITE` / `P2_*_STACK` buckets", "- stronger market-score and fixture-quality floors", "- max 1 row per fixture per market", "- max 3 rows total per fixture", ""]
    for market, sub in out.groupby("market", sort=False):
        lines.append(f"## {market}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['preset_tier']}: {row['fixture_key']} | {row['player_name']} ({row['team_name']}) | family={row['source_family_tag']} | score={row['market_score']:.1f} | quality={row['fixture_quality_score']:.3f} | bucket={row['priority_bucket']}"
            )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build market-specific specialist presets from portable families.")
    parser.add_argument("--inputs", required=True, help="Comma-separated master specialist board csv paths")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inputs = [x.strip() for x in args.inputs.split(",") if x.strip()]
    out = build_presets(inputs, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
