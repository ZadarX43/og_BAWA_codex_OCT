#!/usr/bin/env python3
"""Build an API-Football context overlay board for FootyStats-generated picks.

Research/app-facing only. This joins API-Football absence, lineup, referee,
rest, and tactical style context onto deploy/live-board rows as annotation.
It does not change deploy tiers, picks, rulebooks, or source files.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("predictions_output")
DEFAULT_OUTDIR = Path("reports/2026-05-06/api_context_overlay_board")
PLAYER_EVENTS_DIR = Path("data_sources/api_football/features/player_events")
API_FEATURES_DIR = Path("data_sources/api_football/features")


@dataclass(frozen=True)
class BoardSet:
    board_dir: Path
    base_name: str
    elite: Path
    standard: Path
    observe: Path

    @property
    def fixture_range(self) -> str:
        match = re.search(r"(\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2})", self.base_name)
        return match.group(1) if match else self.base_name

    @property
    def max_mtime(self) -> float:
        return max(self.elite.stat().st_mtime, self.standard.stat().st_mtime, self.observe.stat().st_mtime)


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def normalize_name(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    aliases = {
        "fc barcelona": "barcelona",
        "barcelona": "barcelona",
        "real madrid cf": "real madrid",
        "real madrid": "real madrid",
        "bayern munchen": "bayern munchen",
        "bayern munich": "bayern munchen",
        "psg": "psg",
        "paris saint germain": "psg",
    }
    return aliases.get(text, text)


def join_key(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df.get("match_date"), errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    home = df.get("home_team_name", pd.Series("", index=df.index)).map(normalize_name)
    away = df.get("away_team_name", pd.Series("", index=df.index)).map(normalize_name)
    return date + "|" + home + "|" + away


def is_real_board_dir(path: Path) -> bool:
    joined = str(path).lower()
    if any(token in joined for token in ["walk_forward", "_tmp", "smoke", "research", "shadow_parity_audit"]):
        return False
    return bool(re.search(r"predictions_output/\d{4}-\d{2}-\d{2}$", str(path)))


def discover_boards(root: Path) -> list[BoardSet]:
    grouped: dict[tuple[Path, str], dict[str, Path]] = {}
    for path in root.rglob("*__DEPLOY_TIER_*__PRESET_V1__FTR_accuracy.csv"):
        if not is_real_board_dir(path.parent):
            continue
        tier_match = re.search(r"__DEPLOY_TIER_(ELITE|STANDARD|OBSERVE)__", path.name)
        if not tier_match:
            continue
        base = path.name.split("__DEPLOY_TIER_", 1)[0]
        grouped.setdefault((path.parent, base), {})[tier_match.group(1).lower()] = path
    boards = []
    for (board_dir, base), tiers in grouped.items():
        if {"elite", "standard", "observe"}.issubset(tiers):
            boards.append(BoardSet(board_dir, base, tiers["elite"], tiers["standard"], tiers["observe"]))
    return sorted(boards, key=lambda board: board.max_mtime, reverse=True)


def boards_from_dir(board_dir: Path) -> list[BoardSet]:
    grouped: dict[str, dict[str, Path]] = {}
    for path in board_dir.glob("*__DEPLOY_TIER_*__PRESET_V1__FTR_accuracy.csv"):
        tier_match = re.search(r"__DEPLOY_TIER_(ELITE|STANDARD|OBSERVE)__", path.name)
        if not tier_match:
            continue
        base = path.name.split("__DEPLOY_TIER_", 1)[0]
        grouped.setdefault(base, {})[tier_match.group(1).lower()] = path
    boards = []
    for base, tiers in grouped.items():
        if {"elite", "standard", "observe"}.issubset(tiers):
            boards.append(BoardSet(board_dir, base, tiers["elite"], tiers["standard"], tiers["observe"]))
    return sorted(boards, key=lambda board: board.max_mtime, reverse=True)


def load_tier(path: Path, tier: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "deploy_tier" not in df.columns:
        df["deploy_tier"] = tier
    if "tier" not in df.columns:
        df["tier"] = df["deploy_tier"]
    df["source_tier_file"] = path.name
    return df


def combine_board(board: BoardSet) -> pd.DataFrame:
    return pd.concat(
        [
            load_tier(board.elite, "ELITE"),
            load_tier(board.standard, "STANDARD"),
            load_tier(board.observe, "OBSERVE"),
        ],
        ignore_index=True,
        sort=False,
    )


def first_non_null(group: pd.DataFrame, col: str) -> Any:
    if col not in group.columns:
        return np.nan
    values = group[col].dropna()
    return values.iloc[0] if len(values) else np.nan


def max_num(group: pd.DataFrame, col: str) -> float:
    if col not in group.columns:
        return np.nan
    values = num(group[col]).dropna()
    return float(values.max()) if len(values) else np.nan


def max_flag(group: pd.DataFrame, col: str) -> int:
    if col not in group.columns:
        return 0
    values = num(group[col]).dropna()
    return int(values.max()) if len(values) else 0


def fixture_base_rows(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tier_rank = {"ELITE": 3, "STANDARD": 2, "OBSERVE": 1}
    for fixture_key, group in source.groupby("fixture_key", dropna=False):
        tiers = sorted(set(group.get("deploy_tier", pd.Series(dtype=str)).dropna().astype(str)))
        markets = sorted(set(group.get("market", pd.Series(dtype=str)).dropna().astype(str)))
        best_tier = max(tiers, key=lambda t: tier_rank.get(t, 0)) if tiers else ""
        row = {
            "fixture_key": fixture_key,
            "match_date": first_non_null(group, "match_date"),
            "league": first_non_null(group, "league"),
            "home_team_name": first_non_null(group, "home_team_name"),
            "away_team_name": first_non_null(group, "away_team_name"),
            "source_markets_present": "|".join(markets),
            "source_deploy_tiers_present": "|".join(tiers),
            "best_source_deploy_tier": best_tier,
            "footystats_pick_rows": int(len(group)),
            "elite_rows": int(group.get("deploy_tier", pd.Series(dtype=str)).astype(str).eq("ELITE").sum()),
            "standard_rows": int(group.get("deploy_tier", pd.Series(dtype=str)).astype(str).eq("STANDARD").sum()),
            "observe_rows": int(group.get("deploy_tier", pd.Series(dtype=str)).astype(str).eq("OBSERVE").sum()),
            "max_model_prob": max_num(group, "model_p_for_bookie"),
            "max_value_edge": max_num(group, "value_edge"),
            "max_p_meta_btts": max_num(group, "p_meta_btts"),
            "max_p_meta_ou25": max_num(group, "p_meta_ou25"),
            "max_p_meta_ftr": max_num(group, "p_meta_ftr"),
            "max_cs_mass_over25": max_num(group, "cs_mass_over25"),
            "max_cs_mass_btts_yes": max_num(group, "cs_mass_btts_yes"),
        }
        rows.append(row)
    base = pd.DataFrame(rows)
    if not base.empty:
        base["_join_key"] = join_key(base)
    return base


def load_feature_files(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(".").glob(pattern)):
        try:
            frame = pd.read_csv(path, low_memory=False)
            frame["__source_file"] = str(path)
            frames.append(frame)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def aggregate_player_fixture_context() -> pd.DataFrame:
    pe = load_feature_files(str(PLAYER_EVENTS_DIR / "player_events_fixture_input__*.csv"))
    if pe.empty:
        return pd.DataFrame()
    pe["_join_key"] = join_key(pe)
    rows = []
    for key, group in pe.groupby("_join_key", dropna=False):
        rows.append(
            {
                "_join_key": key,
                "api_fixture_key": first_non_null(group, "fixture_key"),
                "api_league": first_non_null(group, "league"),
                "api_competition": first_non_null(group, "competition"),
                "api_context_rows": int(len(group)),
                "api_expected_starters": int(num(group.get("expected_start_flag", pd.Series(dtype=float))).eq(1).sum()),
                "api_players_listed": int(group.get("player_name", pd.Series(dtype=str)).nunique()),
                "api_referee_name": first_non_null(group, "referee_name"),
                "ref_cards_per_match": max_num(group, "ref_cards_per_match"),
                "ref_foul_to_card_ratio": max_num(group, "ref_foul_to_card_ratio"),
                "ref_dissent_strictness": max_num(group, "ref_dissent_strictness"),
                "ref_timewasting_strictness": max_num(group, "ref_timewasting_strictness"),
                "market_yellow_cards_available": max_flag(group, "market_yellow_cards_available"),
                "market_fouls_available": max_flag(group, "market_fouls_available"),
                "market_team_cards_available": max_flag(group, "market_team_cards_available"),
                "home_formation": first_non_null(group[group.get("player_team_side", "").astype(str).eq("HOME")] if "player_team_side" in group else group, "team_formation"),
                "away_formation": first_non_null(group[group.get("player_team_side", "").astype(str).eq("AWAY")] if "player_team_side" in group else group, "team_formation"),
                "formation_matchup_label": first_non_null(group, "formation_matchup_label"),
                "formation_mismatch_flag": max_flag(group, "formation_mismatch_flag"),
                "formation_pressure_score": max_num(group, "formation_pressure_score"),
                "fixture_style_label": first_non_null(group, "fixture_style_label"),
                "fixture_attacking_style_label": first_non_null(group, "fixture_attacking_style_label"),
                "fixture_foul_density_score": max_num(group, "fixture_foul_density_score"),
                "fixture_tackle_density_score": max_num(group, "fixture_tackle_density_score"),
                "fixture_midfield_grind_score": max_num(group, "fixture_midfield_grind_score"),
                "fixture_wide_duel_score": max_num(group, "fixture_wide_duel_score"),
                "fixture_attack_pressure_score": max_num(group, "fixture_attack_pressure_score"),
                "fixture_corner_pressure_score": max_num(group, "fixture_corner_pressure_score"),
                "fixture_territorial_stress_score": max_num(group, "fixture_territorial_stress_score"),
                "og_goal_environment_score": max_num(group, "og_goal_environment_score"),
                "og_battle_on_score": max_num(group, "og_battle_on_score"),
                "starting_xi_quality_edge_abs": float(num(group.get("starting_xi_quality_edge", pd.Series(dtype=float))).abs().max()),
                "min_days_rest": float(num(group.get("days_rest", pd.Series(dtype=float))).replace(0, np.nan).min()),
                "max_minutes_last_3": max_num(group, "minutes_last_3_matches"),
                "avg_minutes_last_3": float(num(group.get("minutes_last_3_matches", pd.Series(dtype=float))).mean()),
                "recent_injury_return_count": int(num(group.get("recent_injury_return_flag", pd.Series(dtype=float))).eq(1).sum()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_injury_context() -> pd.DataFrame:
    injuries = load_feature_files(str(API_FEATURES_DIR / "api_injury_features__*.csv"))
    if injuries.empty:
        return pd.DataFrame(columns=["_join_key"])
    injuries["_join_key"] = join_key(injuries)
    keep = [
        "_join_key",
        "home_injured_players_count",
        "away_injured_players_count",
        "home_suspended_players_count",
        "away_suspended_players_count",
        "home_missing_defenders_count",
        "away_missing_defenders_count",
        "home_missing_midfielders_count",
        "away_missing_midfielders_count",
        "home_missing_attackers_count",
        "away_missing_attackers_count",
        "home_absence_severity_score",
        "away_absence_severity_score",
        "absence_severity_delta",
    ]
    keep = [col for col in keep if col in injuries.columns]
    return injuries[keep].drop_duplicates("_join_key")


def classify_overlay(row: pd.Series) -> tuple[str, str]:
    if not bool(row.get("api_context_match", False)):
        return "NO_API_CONTEXT_MATCH", "NO_LOCAL_API_CONTEXT_FOR_FIXTURE"
    reasons = []
    if row.get("home_injured_players_count", 0) >= 4 or row.get("away_injured_players_count", 0) >= 4:
        reasons.append("HIGH_INJURY_COUNT")
    if row.get("home_suspended_players_count", 0) > 0 or row.get("away_suspended_players_count", 0) > 0:
        reasons.append("SUSPENSION_PRESENT")
    if max(row.get("home_absence_severity_score", 0), row.get("away_absence_severity_score", 0)) >= 0.60:
        reasons.append("ABSENCE_SEVERITY")
    if row.get("formation_mismatch_flag", 0) == 1 or row.get("starting_xi_quality_edge_abs", 0) >= 12:
        reasons.append("LINEUP_SHAPE_EDGE")
    if row.get("min_days_rest", np.nan) and pd.notna(row.get("min_days_rest")) and row.get("min_days_rest") <= 3:
        reasons.append("REST_SHORT")
    if row.get("max_minutes_last_3", 0) >= 260:
        reasons.append("HEAVY_MINUTES")
    if row.get("ref_cards_per_match", 0) >= 5 or row.get("ref_dissent_strictness", 0) >= 0.65:
        reasons.append("STRICT_REF")
    if (
        row.get("fixture_foul_density_score", 0) >= 0.62
        or row.get("fixture_tackle_density_score", 0) >= 0.62
        or row.get("fixture_midfield_grind_score", 0) >= 0.64
        or str(row.get("fixture_style_label", "")).upper() in {"AGGRESSIVE_BOTH", "MIDFIELD_GRIND", "WIDE_DUEL_GAME"}
    ):
        reasons.append("CONTACT_MARKET_HOTSPOT")
    if (
        row.get("fixture_attack_pressure_score", 0) >= 0.66
        or row.get("fixture_corner_pressure_score", 0) >= 0.66
        or row.get("fixture_territorial_stress_score", 0) >= 0.66
        or str(row.get("fixture_attacking_style_label", "")).upper() in {"ATTACK_WAVE", "CORNER_SIEGE", "TERRITORY_TILT"}
    ):
        reasons.append("ATTACK_EVENT_HOTSPOT")
    if row.get("og_goal_environment_score", 0) >= 0.60:
        reasons.append("GOAL_ENV_SUPPORT")

    downgrade_tokens = {"HIGH_INJURY_COUNT", "SUSPENSION_PRESENT", "ABSENCE_SEVERITY", "REST_SHORT", "HEAVY_MINUTES"}
    hotspot_tokens = {"CONTACT_MARKET_HOTSPOT", "ATTACK_EVENT_HOTSPOT", "STRICT_REF"}
    if any(token in reasons for token in downgrade_tokens):
        action = "WATCH_DOWNGRADE_REVIEW"
    elif any(token in reasons for token in hotspot_tokens):
        action = "PLAYER_EVENT_RESEARCH_HOTSPOT"
    elif reasons:
        action = "ANNOTATE_ONLY"
    else:
        action = "API_CONTEXT_CLEAN"
        reasons.append("NO_MAJOR_CONTEXT_FLAG")
    return action, "|".join(reasons)


def build_overlay(board: BoardSet) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = combine_board(board)
    base = fixture_base_rows(source)
    player_context = aggregate_player_fixture_context()
    injury_context = aggregate_injury_context()
    context = player_context.merge(injury_context, on="_join_key", how="left") if not player_context.empty else injury_context
    if context.empty:
        overlay = base.copy()
        overlay["api_context_match"] = False
    else:
        overlay = base.merge(context, on="_join_key", how="left")
        overlay["api_context_match"] = overlay["api_fixture_key"].notna() if "api_fixture_key" in overlay.columns else False
    for col in [
        "home_injured_players_count",
        "away_injured_players_count",
        "home_suspended_players_count",
        "away_suspended_players_count",
        "home_absence_severity_score",
        "away_absence_severity_score",
        "formation_mismatch_flag",
        "ref_cards_per_match",
        "fixture_foul_density_score",
        "fixture_tackle_density_score",
        "fixture_attack_pressure_score",
        "fixture_corner_pressure_score",
        "og_goal_environment_score",
    ]:
        if col not in overlay.columns:
            overlay[col] = np.nan
    actions = overlay.apply(classify_overlay, axis=1, result_type="expand")
    overlay["api_overlay_action"] = actions[0]
    overlay["api_context_reason_codes"] = actions[1]
    overlay["guardrail"] = "ANNOTATION_ONLY_NO_DEPLOY_TIER_MUTATION"
    overlay = overlay.drop(columns=["_join_key"], errors="ignore")
    summary = (
        overlay.groupby(["api_overlay_action"], dropna=False)
        .agg(fixtures=("fixture_key", "nunique"), pick_rows=("footystats_pick_rows", "sum"))
        .reset_index()
        .sort_values(["fixtures"], ascending=False)
    )
    return overlay, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--board-dir", default="")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    boards = boards_from_dir(Path(args.board_dir)) if args.board_dir else discover_boards(Path(args.root))
    if not boards:
        raise SystemExit("No complete live deploy tier board found.")
    boards = boards[: max(1, args.limit)]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for board in boards:
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", board.fixture_range.replace("-", "_"))
        overlay, summary = build_overlay(board)
        overlay_path = outdir / f"{slug}__API_CONTEXT_OVERLAY_BOARD.csv"
        summary_path = outdir / f"{slug}__API_CONTEXT_OVERLAY_SUMMARY.csv"
        md_path = outdir / f"{slug}__API_CONTEXT_OVERLAY_BOARD.md"
        overlay.to_csv(overlay_path, index=False)
        summary.to_csv(summary_path, index=False)
        matched = int(overlay["api_context_match"].sum()) if "api_context_match" in overlay.columns else 0
        lines = [
            "# API Context Overlay Board",
            "",
            "Annotation-only API-Football context joined to FootyStats-generated picks.",
            "",
            "## Safety",
            "- No deploy tiers changed.",
            "- No picks promoted or removed.",
            "- API-Football is context, not core pick generation.",
            "",
            "## Summary",
            markdown_table(summary),
            "",
            "## Watch Rows",
            markdown_table(
                overlay[
                    [
                        "api_overlay_action",
                        "league",
                        "match_date",
                        "home_team_name",
                        "away_team_name",
                        "best_source_deploy_tier",
                        "source_markets_present",
                        "api_context_match",
                        "api_referee_name",
                        "ref_cards_per_match",
                        "home_injured_players_count",
                        "away_injured_players_count",
                        "fixture_style_label",
                        "fixture_attacking_style_label",
                        "api_context_reason_codes",
                    ]
                ].head(50)
            ),
        ]
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        index_rows.append(
            {
                "fixture_range": board.fixture_range,
                "fixtures": int(overlay["fixture_key"].nunique()),
                "api_context_matches": matched,
                "overlay_path": str(overlay_path),
                "summary_path": str(md_path),
            }
        )
    index = pd.DataFrame(index_rows)
    index.to_csv(outdir / "api_context_overlay_index.csv", index=False)
    print(f"WROTE {outdir}")
    print(index.to_string(index=False))


if __name__ == "__main__":
    main()
