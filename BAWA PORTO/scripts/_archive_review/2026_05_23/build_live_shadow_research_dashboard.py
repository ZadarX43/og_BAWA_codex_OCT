#!/usr/bin/env python3
"""Build a unified live shadow research dashboard.

Research-only. Combines:
  - OU25_RESTORE_NOW_SHADOW
  - Team-goal combo proof shadow
  - Strict FTR + BTTS combo shadow
  - Optional player-event interaction watch labels
  - Optional team shots/SOT/corner/keeper-save pressure context

This reads live deploy tier files and writes dashboard artifacts only. It does
not mutate deploy_tier, tier, deploy_rulebook.py, or source prediction files.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import deploy_rulebook_research_phase8h_c4_shadow as c4_shadow  # noqa: E402
import build_ftr_c5_side_shape_ring_classifier as ftr_c5_shadow  # noqa: E402
import build_team_goal_shadow_market_backtest as team_goal_market_shadow  # noqa: E402
import run_ftr_btts_combo_live_shadow_qa as ftr_btts_shadow  # noqa: E402
import run_team_goal_combo_live_shadow_qa as team_goal_shadow  # noqa: E402


DEFAULT_OUTDIR = Path("reports/2026-05-06/live_shadow_research_dashboard")
DEFAULT_TEAM_GOAL_MARKET_SCORECARD = Path(
    "reports/2026-05-06/team_goal_shadow_market_backtest/team_goal_shadow_scorecard_by_product_policy.csv"
)
DEFAULT_TEAM_GOAL_STABILITY_DIR = Path("reports/2026-05-06/team_goal_threshold_stability")
DEFAULT_FTR_C5_POLICY = Path(
    "reports/2026-05-06/ftr_c5_side_shape_ring_classifier/ftr_c5_recommended_league_policy.csv"
)
DEFAULT_PLAYER_EVENT_INTERACTION_BOARD = Path(
    "reports/2026-05-06/player_event_interaction_live_shadow_board/PLAYER_EVENT_INTERACTION_LIVE_SHADOW_BOARD.csv"
)
DEFAULT_TEAM_SHOTS_PROFILE = Path(
    "reports/2026-05-06/team_shots_profile/TEAM_SHOTS_PROFILE_FIXTURE_VIEW.csv"
)
DEFAULT_TEAM_SHOTS_PROFILE_ROWS = Path(
    "reports/2026-05-06/team_shots_profile/TEAM_SHOTS_PROFILE_ROWS.csv"
)
TARGET_OU25_STAGE = "OU25_RESTORE_NOW_SHADOW"
WATCH_OU25_LEAGUES = {"USA MLS", "Spain La Liga", "Netherlands Eredivisie", "Japan J1"}


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


def num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return num(df[col])


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def load_team_goal_stability(stability_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    league_path = stability_dir / "team_goal_threshold_stability_by_league.csv"
    team_path = stability_dir / "team_goal_threshold_stability_by_team.csv"
    league = pd.read_csv(league_path) if league_path.exists() else pd.DataFrame()
    team = pd.read_csv(team_path) if team_path.exists() else pd.DataFrame()
    return league, team


def load_team_shots_profile(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    profile = pd.read_csv(path, low_memory=False)
    if "fixture_key" not in profile.columns:
        return pd.DataFrame()
    profile = profile.copy()
    profile["fixture_key"] = profile["fixture_key"].astype(str)
    keep = [
        "fixture_key",
        "home_team_expected_shots",
        "away_team_expected_shots",
        "match_expected_shots",
        "home_team_expected_sot",
        "away_team_expected_sot",
        "match_expected_sot",
        "home_team_expected_corners",
        "away_team_expected_corners",
        "match_expected_corners",
        "home_keeper_save_pressure_score",
        "away_keeper_save_pressure_score",
        "home_team_shots_profile_labels",
        "away_team_shots_profile_labels",
        "home_team_shots_profile_mode",
        "away_team_shots_profile_mode",
    ]
    keep = [col for col in keep if col in profile.columns]
    return profile[keep].drop_duplicates("fixture_key", keep="last")


def load_team_shots_profile_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path, low_memory=False)
    required = {"team_name", "fixture_key", "league_tag", "match_date"}
    if not required.issubset(rows.columns):
        return pd.DataFrame()
    rows = rows.copy()
    rows["match_date"] = pd.to_datetime(rows["match_date"], errors="coerce")
    rows["team_name_norm"] = rows["team_name"].map(norm_text)
    for col in [
        "team_shots_for_l5",
        "team_shots_for_l10",
        "team_sot_for_l5",
        "team_sot_for_l10",
        "team_box_shots_for_l5",
        "team_box_shots_for_l10",
        "team_corners_for_l5",
        "team_corners_for_l10",
        "team_high_shot_volume_rate_l5",
        "team_high_sot_volume_rate_l5",
        "team_corner_5plus_rate_l5",
        "team_shots_allowed_l5",
        "team_shots_allowed_l10",
        "team_sot_allowed_l5",
        "team_sot_allowed_l10",
        "team_box_shots_allowed_l5",
        "team_corners_allowed_l5",
        "team_corners_allowed_l10",
    ]:
        if col not in rows.columns:
            rows[col] = np.nan
        rows[col] = num(rows[col])
    return rows.sort_values(["match_date", "fixture_key"]).reset_index(drop=True)


def filter_team_shots_profile(profile: pd.DataFrame, fixture_keys: set[str]) -> pd.DataFrame:
    if profile.empty or "fixture_key" not in profile.columns:
        return pd.DataFrame()
    out = profile[profile["fixture_key"].astype(str).isin(fixture_keys)].copy()
    if out.empty:
        return out
    home_labels = out.get("home_team_shots_profile_labels", pd.Series("", index=out.index)).fillna("").astype(str)
    away_labels = out.get("away_team_shots_profile_labels", pd.Series("", index=out.index)).fillna("").astype(str)
    out["team_shots_profile_context_ready"] = True
    out["team_shots_profile_any_label"] = home_labels.ne("") | away_labels.ne("")
    out["team_shots_profile_context_mode"] = "EXACT_FIXTURE_PROFILE"
    return out


def expected_attack(team: pd.Series | None, opponent: pd.Series | None, metric: str) -> float:
    if team is None or opponent is None:
        return np.nan
    if metric == "shots":
        return float(
            0.45 * team.get("team_shots_for_l5", np.nan)
            + 0.25 * team.get("team_shots_for_l10", np.nan)
            + 0.20 * opponent.get("team_shots_allowed_l5", np.nan)
            + 0.10 * opponent.get("team_shots_allowed_l10", np.nan)
        )
    if metric == "sot":
        return float(
            0.45 * team.get("team_sot_for_l5", np.nan)
            + 0.25 * team.get("team_sot_for_l10", np.nan)
            + 0.20 * opponent.get("team_sot_allowed_l5", np.nan)
            + 0.10 * opponent.get("team_sot_allowed_l10", np.nan)
        )
    if metric == "box_shots":
        return float(
            0.50 * team.get("team_box_shots_for_l5", np.nan)
            + 0.25 * team.get("team_box_shots_for_l10", np.nan)
            + 0.25 * opponent.get("team_box_shots_allowed_l5", np.nan)
        )
    if metric == "corners":
        return float(
            0.50 * team.get("team_corners_for_l5", np.nan)
            + 0.25 * team.get("team_corners_for_l10", np.nan)
            + 0.25 * opponent.get("team_corners_allowed_l5", np.nan)
        )
    return np.nan


def profile_thresholds(profile_rows: pd.DataFrame, league_tag: str) -> dict[str, float]:
    league_rows = profile_rows[profile_rows["league_tag"].astype(str).eq(str(league_tag))]
    if league_rows.empty:
        league_rows = profile_rows
    return {
        "expected_shots_p75": float(num(league_rows.get("team_expected_shots", pd.Series(dtype=float))).quantile(0.75)),
        "expected_sot_p75": float(num(league_rows.get("team_expected_sot", pd.Series(dtype=float))).quantile(0.75)),
        "expected_corners_p75": float(num(league_rows.get("team_expected_corners", pd.Series(dtype=float))).quantile(0.75)),
        "keeper_sot_p80": float(num(league_rows.get("keeper_save_pressure_score", pd.Series(dtype=float))).quantile(0.80)),
        "shots_for_l5_p70": float(num(league_rows.get("team_shots_for_l5", pd.Series(dtype=float))).quantile(0.70)),
        "sot_for_l5_p70": float(num(league_rows.get("team_sot_for_l5", pd.Series(dtype=float))).quantile(0.70)),
    }


def projected_team_labels(team: pd.Series | None, expected: dict[str, float], thresholds: dict[str, float]) -> str:
    if team is None:
        return ""
    labels: list[str] = []
    if (
        pd.notna(expected["shots"])
        and expected["shots"] >= thresholds["expected_shots_p75"]
        and float(team.get("team_shots_for_l5", 0) or 0) >= thresholds["shots_for_l5_p70"]
    ):
        labels.append("TEAM_SHOTS_CORE")
    if (
        pd.notna(expected["sot"])
        and expected["sot"] >= thresholds["expected_sot_p75"]
        and float(team.get("team_sot_for_l5", 0) or 0) >= thresholds["sot_for_l5_p70"]
    ):
        labels.append("TEAM_SOT_CORE")
    if (
        pd.notna(expected["corners"])
        and expected["corners"] >= thresholds["expected_corners_p75"]
        and float(team.get("team_corner_5plus_rate_l5", 0) or 0) >= 0.40
    ):
        labels.append("CORNER_PRESSURE_WATCH")
    if (
        pd.notna(expected["sot"])
        and expected["sot"] >= thresholds["keeper_sot_p80"]
        and float(team.get("team_high_sot_volume_rate_l5", 0) or 0) >= 0.40
    ):
        labels.append("KEEPER_SAVE_WATCH")
    if (
        pd.notna(expected["shots"])
        and pd.notna(expected["sot"])
        and expected["shots"] >= thresholds["expected_shots_p75"]
        and expected["sot"] >= thresholds["expected_sot_p75"]
        and float(team.get("team_high_shot_volume_rate_l5", 0) or 0) >= 0.40
    ):
        labels.append("ATTACK_VOLUME_WATCH")
    return "|".join(labels)


def project_team_shots_profile(fixtures: pd.DataFrame, profile_rows: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty or profile_rows.empty:
        return pd.DataFrame()
    latest = (
        profile_rows.dropna(subset=["team_name_norm"])
        .sort_values(["match_date", "fixture_key"])
        .drop_duplicates("team_name_norm", keep="last")
        .set_index("team_name_norm")
    )

    def lookup_team(raw_name: Any) -> pd.Series | None:
        key = norm_text(raw_name)
        if not key:
            return None
        if key in latest.index:
            return latest.loc[key]
        compact_key = re.sub(r"\b(fc|cf|sc|afc|cd|ud|rcd|1)\b", " ", key)
        compact_key = re.sub(r"\s+", " ", compact_key).strip()
        candidates = []
        for candidate in latest.index:
            compact_candidate = re.sub(r"\b(fc|cf|sc|afc|cd|ud|rcd|1)\b", " ", str(candidate))
            compact_candidate = re.sub(r"\s+", " ", compact_candidate).strip()
            if len(compact_key) >= 5 and (compact_key in compact_candidate or compact_candidate in compact_key):
                candidates.append((abs(len(compact_candidate) - len(compact_key)), candidate))
        if not candidates:
            return None
        _, best = sorted(candidates, key=lambda item: item[0])[0]
        return latest.loc[best]

    records = []
    for _, fixture in fixtures.iterrows():
        home_name = scalar(fixture, "home_team_name")
        away_name = scalar(fixture, "away_team_name")
        home = lookup_team(home_name)
        away = lookup_team(away_name)
        if home is None and away is None:
            continue
        if home is not None:
            league_tag = str(home.get("league_tag", ""))
        elif away is not None:
            league_tag = str(away.get("league_tag", ""))
        else:
            league_tag = ""
        thresholds = profile_thresholds(profile_rows, league_tag)
        home_expected = {
            "shots": expected_attack(home, away, "shots"),
            "sot": expected_attack(home, away, "sot"),
            "box_shots": expected_attack(home, away, "box_shots"),
            "corners": expected_attack(home, away, "corners"),
        }
        away_expected = {
            "shots": expected_attack(away, home, "shots"),
            "sot": expected_attack(away, home, "sot"),
            "box_shots": expected_attack(away, home, "box_shots"),
            "corners": expected_attack(away, home, "corners"),
        }
        home_labels = projected_team_labels(home, home_expected, thresholds)
        away_labels = projected_team_labels(away, away_expected, thresholds)
        records.append(
            {
                "fixture_key": scalar(fixture, "fixture_key"),
                "home_team_expected_shots": home_expected["shots"],
                "away_team_expected_shots": away_expected["shots"],
                "match_expected_shots": home_expected["shots"] + away_expected["shots"],
                "home_team_expected_sot": home_expected["sot"],
                "away_team_expected_sot": away_expected["sot"],
                "match_expected_sot": home_expected["sot"] + away_expected["sot"],
                "home_team_expected_corners": home_expected["corners"],
                "away_team_expected_corners": away_expected["corners"],
                "match_expected_corners": home_expected["corners"] + away_expected["corners"],
                "home_keeper_save_pressure_score": home_expected["sot"],
                "away_keeper_save_pressure_score": away_expected["sot"],
                "home_team_shots_profile_labels": home_labels,
                "away_team_shots_profile_labels": away_labels,
                "home_team_shots_profile_mode": "LATEST_TEAM_PROFILE_PROJECTION" if home is not None else "MISSING_TEAM_PROFILE",
                "away_team_shots_profile_mode": "LATEST_TEAM_PROFILE_PROJECTION" if away is not None else "MISSING_TEAM_PROFILE",
                "team_shots_profile_context_ready": home is not None or away is not None,
                "team_shots_profile_any_label": bool(home_labels or away_labels),
                "team_shots_profile_context_mode": "LATEST_TEAM_PROFILE_PROJECTION",
            }
        )
    return pd.DataFrame(records).drop_duplicates("fixture_key", keep="last") if records else pd.DataFrame()


def team_shots_context_for_board(
    fixtures: pd.DataFrame,
    exact_profile: pd.DataFrame,
    profile_rows: pd.DataFrame,
    fixture_keys: set[str],
) -> pd.DataFrame:
    exact = filter_team_shots_profile(exact_profile, fixture_keys)
    exact_keys = set(exact.get("fixture_key", pd.Series(dtype=str)).astype(str))
    missing_fixtures = fixtures[~fixtures["fixture_key"].astype(str).isin(exact_keys)].copy()
    projected = project_team_shots_profile(missing_fixtures, profile_rows)
    if exact.empty:
        return projected
    if projected.empty:
        return exact
    return pd.concat([exact, projected], ignore_index=True, sort=False).drop_duplicates("fixture_key", keep="first")


def stability_lookup(df: pd.DataFrame, keys: list[str]) -> dict[tuple, str]:
    if df.empty or "stability_bucket" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        out[tuple(str(row.get(key, "")) for key in keys)] = str(row.get("stability_bucket", ""))
    return out


def team_goal_watch_priority(league_bucket: str, team_bucket: str, product: str) -> str:
    buckets = {league_bucket, team_bucket}
    if "TEAM_CORE" in buckets or "SHADOW_CORE" in buckets:
        return "PRIORITY_CORE"
    if "TEAM_CONFIRM" in buckets or "SHADOW_CONFIRM" in buckets:
        return "PRIORITY_CONFIRM"
    if "MICRO_ONLY" in buckets:
        return "MICRO_ONLY"
    if product in {"HOME_TEAM_OVER_2_5_SHADOW", "AWAY_TEAM_OVER_2_5_SHADOW", "MATCH_OVER_3_5_SHADOW"}:
        return "WATCH_ONLY_NOT_PROMOTION"
    return "LOW_PRIORITY_SHADOW"


def watch_priority_rank(priority: str) -> int:
    order = {
        "PRIORITY_CORE": 100,
        "PRIORITY_CONFIRM": 80,
        "MICRO_ONLY": 60,
        "WATCH_ONLY_NOT_PROMOTION": 40,
        "LOW_PRIORITY_SHADOW": 20,
    }
    return order.get(priority, 0)


def ftr_c5_watch_priority(bucket: str) -> str:
    if bucket == "RESTORE_NOW_SHADOW":
        return "PRIORITY_CORE"
    if bucket == "RESTORE_WITH_CONFIRM_SHADOW":
        return "PRIORITY_CONFIRM"
    if bucket == "MICRO_ONLY":
        return "MICRO_ONLY"
    return "LOW_PRIORITY_SHADOW"


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
    df = pd.read_csv(path)
    if "deploy_tier" not in df.columns:
        df["deploy_tier"] = tier
    if "tier" not in df.columns:
        df["tier"] = df["deploy_tier"]
    df["source_tier_file"] = path.name
    df["source_tier_expected"] = tier
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


def scalar(row: pd.Series, col: str, default: Any = "") -> Any:
    if col not in row.index:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    return value


def coalesce_first(group: pd.DataFrame, col: str) -> Any:
    if col not in group.columns:
        return np.nan
    values = group[col].dropna()
    return values.iloc[0] if len(values) else np.nan


def coalesce_numeric_max(group: pd.DataFrame, col: str) -> float:
    if col not in group.columns:
        return np.nan
    values = num(group[col]).dropna()
    return float(values.max()) if len(values) else np.nan


def fixture_level_from_board(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fixture_key, group in source.groupby("fixture_key", dropna=False):
        row = {"fixture_key": fixture_key}
        for col in [
            "league",
            "match_date",
            "home_team_name",
            "away_team_name",
            "home_team_ge2_candidate_flag",
            "away_team_ge2_candidate_flag",
            "home_team_high_scoring_flag",
            "away_team_high_scoring_flag",
            "deploy_tier",
            "tier",
        ]:
            row[col] = coalesce_first(group, col)
        for col in [
            "p_home_ge2",
            "p_away_ge2",
            "p_home_ge3",
            "p_away_ge3",
            "pois_home_ge2",
            "pois_away_ge2",
            "pois_home_ge3",
            "pois_away_ge3",
            "home_ge2_confidence",
            "away_ge2_confidence",
            "cs_mass_over25",
            "mass_4plus_goals",
            "p_meta_ou25",
            "p_meta_btts",
            "p_meta_ftr",
        ]:
            row[col] = coalesce_numeric_max(group, col)
        row["source_tiers_present"] = "|".join(sorted(set(group.get("deploy_tier", pd.Series(dtype=str)).dropna().astype(str))))
        row["source_markets_present"] = "|".join(sorted(set(group.get("market", pd.Series(dtype=str)).dropna().astype(str))))
        row["window_id"] = "LIVE_BOARD"
        rows.append(row)
    fixtures = pd.DataFrame(rows)
    for side in ["home", "away"]:
        ge2 = f"p_{side}_ge2"
        ge3 = f"p_{side}_ge3"
        fixtures[ge2] = num(fixtures.get(ge2, np.nan)).where(
            num(fixtures.get(ge2, np.nan)).notna(),
            num(fixtures.get(f"pois_{side}_ge2", np.nan)),
        )
        fixtures[ge2] = num(fixtures[ge2]).where(
            num(fixtures[ge2]).notna(),
            num(fixtures.get(f"{side}_ge2_confidence", np.nan)),
        )
        fixtures[ge3] = num(fixtures.get(ge3, np.nan)).where(
            num(fixtures.get(ge3, np.nan)).notna(),
            num(fixtures.get(f"pois_{side}_ge3", np.nan)),
        )
    return fixtures


def normalize_ou25(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        league = str(scalar(row, "league"))
        records.append(
            {
                "shadow_family": "OU25_C4",
                "shadow_stage": scalar(row, "phase8h_c4_shadow_stage"),
                "fixture_key": scalar(row, "fixture_key"),
                "match_date": scalar(row, "match_date"),
                "league": league,
                "home_team_name": scalar(row, "home_team_name"),
                "away_team_name": scalar(row, "away_team_name"),
                "expression": "OU25_OVER",
                "source_market": scalar(row, "market"),
                "source_selection": scalar(row, "selection", scalar(row, "bookie_pick")),
                "source_deploy_tier": scalar(row, "deploy_tier"),
                "source_tier": scalar(row, "tier"),
                "combo_product": "",
                "combo_tier": "",
                "bookie_od": scalar(row, "bookie_od", np.nan),
                "model_prob": scalar(row, "model_p_for_bookie", np.nan),
                "value_edge": scalar(row, "value_edge", np.nan),
                "value_edge_tier": scalar(row, "value_edge_tier"),
                "backtest_hit_rate": np.nan,
                "backtest_graded": np.nan,
                "watch_flag": league in WATCH_OU25_LEAGUES,
                "guardrail": "SHADOW_ONLY_NO_TIER_MUTATION",
                "reason": "C4_OU25_RESTORE_NOW_RING",
            }
        )
    return pd.DataFrame(records)


def normalize_team_goal(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        records.append(
            {
                "shadow_family": "TEAM_GOAL_COMBO",
                "shadow_stage": scalar(row, "team_goal_combo_shadow_stage"),
                "fixture_key": scalar(row, "fixture_key"),
                "match_date": scalar(row, "match_date"),
                "league": scalar(row, "league"),
                "home_team_name": scalar(row, "home_team_name"),
                "away_team_name": scalar(row, "away_team_name"),
                "expression": scalar(row, "ftr_combo_live_product"),
                "source_market": scalar(row, "market"),
                "source_selection": scalar(row, "selection", scalar(row, "bookie_pick")),
                "source_deploy_tier": scalar(row, "deploy_tier"),
                "source_tier": scalar(row, "tier"),
                "combo_product": scalar(row, "ftr_combo_live_product"),
                "combo_tier": scalar(row, "ftr_combo_live_tier"),
                "bookie_od": scalar(row, "bookie_od", np.nan),
                "model_prob": scalar(row, "p_meta_ftr", scalar(row, "model_p_for_bookie", np.nan)),
                "value_edge": scalar(row, "value_edge", np.nan),
                "value_edge_tier": scalar(row, "value_edge_tier"),
                "backtest_hit_rate": np.nan,
                "backtest_graded": np.nan,
                "watch_flag": False,
                "guardrail": "SHADOW_ONLY_NO_TIER_MUTATION",
                "reason": scalar(row, "team_goal_combo_shadow_reason", "SPAIN_GERMANY_HOME_GE2_PROOF"),
            }
        )
    return pd.DataFrame(records)


def normalize_team_goal_markets(
    rows: pd.DataFrame,
    scorecard: pd.DataFrame,
    league_stability: pd.DataFrame,
    team_stability: pd.DataFrame,
) -> pd.DataFrame:
    score_lookup = {}
    if not scorecard.empty:
        for _, row in scorecard.iterrows():
            score_lookup[(row.get("shadow_product"), row.get("shadow_policy"))] = {
                "hit_rate": row.get("hit_rate", np.nan),
                "graded": row.get("graded", np.nan),
            }
    league_lookup = stability_lookup(league_stability, ["shadow_product", "shadow_policy", "league"])
    team_lookup = stability_lookup(team_stability, ["shadow_product", "shadow_policy", "league", "shadow_team_name"])

    records = []
    for _, row in rows.iterrows():
        product = scalar(row, "shadow_product")
        policy = scalar(row, "shadow_policy")
        league = scalar(row, "league")
        team_name = scalar(row, "shadow_team_name", "")
        score = score_lookup.get((product, policy), {})
        league_bucket = league_lookup.get((str(product), str(policy), str(league)), "")
        team_bucket = team_lookup.get((str(product), str(policy), str(league), str(team_name)), "")
        priority = team_goal_watch_priority(league_bucket, team_bucket, str(product))
        watch = watch_priority_rank(priority) >= 40
        records.append(
            {
                "shadow_family": "TEAM_GOAL_MARKET",
                "shadow_stage": product,
                "fixture_key": scalar(row, "fixture_key"),
                "match_date": scalar(row, "match_date"),
                "league": league,
                "home_team_name": scalar(row, "home_team_name"),
                "away_team_name": scalar(row, "away_team_name"),
                "expression": product,
                "source_market": "model_only_team_goal",
                "source_selection": team_name or "MATCH",
                "source_deploy_tier": scalar(row, "source_tiers_present"),
                "source_tier": scalar(row, "source_tiers_present"),
                "combo_product": "",
                "combo_tier": policy,
                "bookie_od": np.nan,
                "model_prob": scalar(row, "model_prob", np.nan),
                "value_edge": np.nan,
                "value_edge_tier": "",
                "backtest_hit_rate": score.get("hit_rate", np.nan),
                "backtest_graded": score.get("graded", np.nan),
                "league_stability_bucket": league_bucket,
                "team_stability_bucket": team_bucket,
                "watch_priority": priority,
                "watch_flag": watch,
                "guardrail": "MODEL_ONLY_NO_DIRECT_TG_ODDS|SHADOW_ONLY_NO_TIER_MUTATION",
                "reason": scalar(row, "reason"),
            }
        )
    return pd.DataFrame(records)


def apply_ftr_c5_policy(source: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty or "market" not in source.columns:
        return pd.DataFrame()
    ftr_rows = source[source["market"].astype("string").str.lower().eq("ftr")].copy()
    if ftr_rows.empty:
        return ftr_rows
    if "selection" not in ftr_rows.columns:
        ftr_rows["selection"] = ftr_rows.get("bookie_pick", "")
    ftr_rows["selection_norm"] = ftr_rows["selection"].astype("string").str.upper().str.strip()

    config_by_ring = {config["ring"]: config for config in ftr_c5_shadow.RING_CONFIGS}
    outputs = []
    for _, rule in policy.iterrows():
        bucket = str(rule.get("c5_bucket", ""))
        ring = str(rule.get("c5_ring", ""))
        league = str(rule.get("league", ""))
        config = config_by_ring.get(ring)
        if not config or bucket == "OBSERVE":
            continue
        rows = ftr_rows[ftr_rows["league"].astype("string").eq(league)].copy()
        if rows.empty:
            continue
        mask = pd.Series(True, index=rows.index)
        mask &= num_series(rows, "p_meta_ftr").ge(config["p_meta_ftr"])
        mask &= num_series(rows, "pick_side_margin_top3").ge(config["pick_side_margin_top3"])
        mask &= num_series(rows, "pick_side_mass_top3").ge(config["pick_side_mass_top3"])
        mask &= num_series(rows, "ftr_margin").ge(config["ftr_margin"])
        mask &= ftr_c5_shadow.side_power_pass(rows["selection_norm"], num_series(rows, "power_diff"), config["side_power_abs"])
        if "cat_xgb_grid_ftr_agreement_count" in rows.columns:
            mask &= num(rows["cat_xgb_grid_ftr_agreement_count"]).ge(config["agreement_count"])
        for gap_col in ["grid_vs_cat_ftr_gap", "grid_vs_xgb_ftr_gap"]:
            if gap_col in rows.columns:
                mask &= num(rows[gap_col]).le(config["grid_gap_max"])
        if "value_edge" in config and "value_edge" in rows.columns:
            mask &= num(rows["value_edge"]).ge(config["value_edge"])
        selected = rows[mask].copy()
        if selected.empty:
            continue
        selected["ftr_c5_shadow_stage"] = "FTR_C5_SIDE_SHAPE"
        selected["ftr_c5_ring"] = ring
        selected["ftr_c5_bucket"] = bucket
        selected["ftr_c5_backtest_hit_rate"] = rule.get("hit_rate", np.nan)
        selected["ftr_c5_backtest_graded"] = rule.get("graded", np.nan)
        selected["ftr_c5_active_windows"] = rule.get("active_windows", np.nan)
        selected["ftr_c5_p25_window_hit_rate"] = rule.get("p25_window_hit_rate", np.nan)
        selected["ftr_c5_guardrail"] = "SHADOW_ONLY_NO_TIER_MUTATION"
        outputs.append(selected)
    if not outputs:
        return pd.DataFrame()
    out = pd.concat(outputs, ignore_index=True, sort=False)
    priority = {"RESTORE_NOW_SHADOW": 100, "RESTORE_WITH_CONFIRM_SHADOW": 80, "MICRO_ONLY": 60}
    out["_bucket_rank"] = out["ftr_c5_bucket"].map(priority).fillna(0)
    out = out.sort_values(["fixture_key", "selection_norm", "_bucket_rank"], ascending=[True, True, False])
    return out.drop_duplicates(["fixture_key", "selection_norm"], keep="first").drop(columns=["_bucket_rank"])


def normalize_ftr_c5(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        bucket = scalar(row, "ftr_c5_bucket")
        priority = ftr_c5_watch_priority(str(bucket))
        records.append(
            {
                "shadow_family": "FTR_C5_SIDE_SHAPE",
                "shadow_stage": scalar(row, "ftr_c5_bucket"),
                "fixture_key": scalar(row, "fixture_key"),
                "match_date": scalar(row, "match_date"),
                "league": scalar(row, "league"),
                "home_team_name": scalar(row, "home_team_name"),
                "away_team_name": scalar(row, "away_team_name"),
                "expression": f"FTR_{scalar(row, 'selection_norm', scalar(row, 'selection', scalar(row, 'bookie_pick')))}",
                "source_market": scalar(row, "market"),
                "source_selection": scalar(row, "selection", scalar(row, "bookie_pick")),
                "source_deploy_tier": scalar(row, "deploy_tier"),
                "source_tier": scalar(row, "tier"),
                "combo_product": "",
                "combo_tier": scalar(row, "ftr_c5_ring"),
                "bookie_od": scalar(row, "bookie_od", np.nan),
                "model_prob": scalar(row, "model_p_for_bookie", np.nan),
                "value_edge": scalar(row, "value_edge", np.nan),
                "value_edge_tier": scalar(row, "value_edge_tier"),
                "backtest_hit_rate": scalar(row, "ftr_c5_backtest_hit_rate", np.nan),
                "backtest_graded": scalar(row, "ftr_c5_backtest_graded", np.nan),
                "league_stability_bucket": bucket,
                "team_stability_bucket": "",
                "watch_priority": priority,
                "watch_flag": watch_priority_rank(priority) >= 40,
                "guardrail": "SHADOW_ONLY_NO_TIER_MUTATION",
                "reason": (
                    f"{scalar(row, 'ftr_c5_ring')}|"
                    f"windows={scalar(row, 'ftr_c5_active_windows', '')}|"
                    f"p25={scalar(row, 'ftr_c5_p25_window_hit_rate', '')}"
                ),
            }
        )
    return pd.DataFrame(records)


def dedupe_team_goal_market_live(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    priority = {
        "TG15_PREMIUM": 100,
        "TG25_MONSTER": 95,
        "MO35_GOALMASS_DOMINANCE": 90,
        "TG15_CORE_WATCH": 85,
        "TG15_ALLOWLIST": 80,
        "TG15_MODEL": 70,
        "MO35_MONSTER_SIDE": 60,
    }
    out = rows.copy()
    out["_policy_priority"] = out["shadow_policy"].map(priority).fillna(0)
    out = out.sort_values(["fixture_key", "shadow_product", "shadow_side", "_policy_priority"], ascending=[True, True, True, False])
    out = out.drop_duplicates(["fixture_key", "shadow_product", "shadow_side"], keep="first")
    return out.drop(columns=["_policy_priority"]).reset_index(drop=True)


def normalize_ftr_btts(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        records.append(
            {
                "shadow_family": "FTR_BTTS_COMBO",
                "shadow_stage": scalar(row, "ftr_btts_combo_shadow_stage"),
                "fixture_key": scalar(row, "fixture_key"),
                "match_date": scalar(row, "match_date"),
                "league": scalar(row, "league"),
                "home_team_name": scalar(row, "home_team_name"),
                "away_team_name": scalar(row, "away_team_name"),
                "expression": scalar(row, "combo_product"),
                "source_market": "ftr+btts",
                "source_selection": f"{scalar(row, 'ftr_side')} + BTTS {scalar(row, 'btts_side')}",
                "source_deploy_tier": f"FTR:{scalar(row, 'ftr_deploy_tier')}|BTTS:{scalar(row, 'btts_deploy_tier')}",
                "source_tier": f"FTR:{scalar(row, 'ftr_tier')}|BTTS:{scalar(row, 'btts_tier')}",
                "combo_product": scalar(row, "combo_product"),
                "combo_tier": "",
                "bookie_od": scalar(row, "synthetic_combo_od", np.nan),
                "model_prob": scalar(row, "synthetic_combo_model_p", np.nan),
                "value_edge": scalar(row, "synthetic_combo_value_edge", np.nan),
                "value_edge_tier": "",
                "backtest_hit_rate": scalar(row, "ftr_btts_combo_backtest_hit_rate", np.nan),
                "backtest_graded": scalar(row, "ftr_btts_combo_backtest_graded", np.nan),
                "watch_flag": False,
                "guardrail": "SHADOW_ONLY_NO_TIER_MUTATION",
                "reason": scalar(row, "ftr_btts_combo_candidate_id"),
            }
        )
    return pd.DataFrame(records)


def normalize_player_event_interactions(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    records = []
    for _, row in rows.iterrows():
        records.append(
            {
                "shadow_family": scalar(row, "shadow_family", "PLAYER_EVENT_INTERACTION"),
                "shadow_stage": scalar(row, "shadow_stage"),
                "fixture_key": scalar(row, "fixture_key"),
                "match_date": scalar(row, "match_date"),
                "league": scalar(row, "league"),
                "home_team_name": scalar(row, "home_team_name"),
                "away_team_name": scalar(row, "away_team_name"),
                "expression": scalar(row, "expression"),
                "source_market": scalar(row, "source_market"),
                "source_selection": scalar(row, "source_selection", scalar(row, "player_name")),
                "source_deploy_tier": scalar(row, "source_deploy_tier", "PLAYER_EVENT_BETA"),
                "source_tier": scalar(row, "source_tier", scalar(row, "confidence_label")),
                "combo_product": scalar(row, "combo_product"),
                "combo_tier": scalar(row, "combo_tier", scalar(row, "confidence_label")),
                "bookie_od": np.nan,
                "model_prob": scalar(row, "model_prob", scalar(row, "predicted_hit_rate", np.nan)),
                "value_edge": np.nan,
                "value_edge_tier": "",
                "backtest_hit_rate": scalar(row, "backtest_hit_rate", np.nan),
                "backtest_graded": scalar(row, "backtest_graded", np.nan),
                "league_stability_bucket": scalar(row, "league_stability_bucket"),
                "team_stability_bucket": scalar(row, "team_stability_bucket"),
                "watch_priority": scalar(row, "watch_priority", "WATCH_ONLY_NOT_PROMOTION"),
                "watch_flag": bool(scalar(row, "watch_flag", True)),
                "guardrail": scalar(row, "guardrail", "PLAYER_EVENT_BETA_ONLY|NO_PRICED_ODDS|NO_DEPLOY_PROMOTION"),
                "reason": scalar(row, "reason"),
                "player_name": scalar(row, "player_name", scalar(row, "source_selection")),
                "team_name": scalar(row, "team_name"),
                "player_team_side": scalar(row, "player_team_side"),
                "position_group": scalar(row, "position_group"),
                "tactical_role": scalar(row, "tactical_role"),
                "predicted_hit_rate_pct": scalar(row, "predicted_hit_rate_pct", np.nan),
                "confidence_label": scalar(row, "confidence_label"),
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
                "interaction_match_mode": scalar(row, "interaction_match_mode"),
                "interaction_label": scalar(row, "interaction_label"),
                "fouled_context_cell": scalar(row, "fouled_context_cell"),
                "fouled_context_cell_label": scalar(row, "fouled_context_cell_label"),
                "team_corner_pressure_score": scalar(row, "team_corner_pressure_score", np.nan),
                "team_expected_corners": scalar(row, "team_expected_corners", np.nan),
                "team_expected_corners_pct": scalar(row, "team_expected_corners_pct", np.nan),
                "keeper_save_pressure_score": scalar(row, "keeper_save_pressure_score", np.nan),
                "keeper_expected_sot_faced": scalar(row, "keeper_expected_sot_faced", np.nan),
                "keeper_expected_sot_faced_pct": scalar(row, "keeper_expected_sot_faced_pct", np.nan),
                "attacking_team_name": scalar(row, "attacking_team_name"),
                "booking_pressure_side": scalar(row, "booking_pressure_side"),
                "booking_pressure_mode": scalar(row, "booking_pressure_mode"),
                "booking_context_label": scalar(row, "booking_context_label"),
                "booking_hazard_score": scalar(row, "booking_hazard_score", np.nan),
                "booking_role_cell": scalar(row, "booking_role_cell"),
                "booking_side_pressure_score": scalar(row, "booking_side_pressure_score", np.nan),
                "booking_player_contact_score": scalar(row, "booking_player_contact_score", np.nan),
                "booking_fixture_contact_score": scalar(row, "booking_fixture_contact_score", np.nan),
                "booking_recent_fouls_committed_per90": scalar(row, "booking_recent_fouls_committed_per90", np.nan),
                "booking_goal_state_signal": scalar(row, "booking_goal_state_signal", np.nan),
            }
        )
    return pd.DataFrame(records)


def count_by(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=cols + ["rows"])
    return df.groupby(cols, dropna=False).size().reset_index(name="rows")


def run_board(
    board: BoardSet,
    *,
    ou25_policy: pd.DataFrame,
    ftr_c5_policy: pd.DataFrame,
    ftr_btts_policy: pd.DataFrame,
    team_goal_market_scorecard: pd.DataFrame,
    team_goal_league_stability: pd.DataFrame,
    team_goal_team_stability: pd.DataFrame,
    player_event_interactions: pd.DataFrame,
    team_shots_profile: pd.DataFrame,
    team_shots_profile_rows: pd.DataFrame,
    outdir: Path,
) -> dict[str, Any]:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", board.fixture_range.replace("-", "_"))
    board_out = outdir / slug
    board_out.mkdir(parents=True, exist_ok=True)

    source = combine_board(board)
    source_fixture_keys = set(source.get("fixture_key", pd.Series(dtype=str)).dropna().astype(str))
    fixtures = fixture_level_from_board(source)
    board_team_shots_profile = team_shots_context_for_board(
        fixtures,
        team_shots_profile,
        team_shots_profile_rows,
        source_fixture_keys,
    )
    if not player_event_interactions.empty and "fixture_key" in player_event_interactions.columns:
        board_player_event_interactions = player_event_interactions[
            player_event_interactions["fixture_key"].astype(str).isin(source_fixture_keys)
        ].copy()
    else:
        board_player_event_interactions = pd.DataFrame()

    annotated = c4_shadow.apply_shadow_policy(source, ou25_policy)
    selected_all_c4 = c4_shadow.selected_rows(annotated)
    selected_ou25 = selected_all_c4[selected_all_c4["phase8h_c4_shadow_stage"].eq(TARGET_OU25_STAGE)].copy()
    selected_ftr_c5 = apply_ftr_c5_policy(source, ftr_c5_policy)

    fixtures = team_goal_market_shadow.attach_allowlists(
        fixtures,
        team_goal_market_shadow.load_allowlist(team_goal_market_shadow.DEFAULT_HOME_ALLOWLIST, "home"),
        team_goal_market_shadow.load_allowlist(team_goal_market_shadow.DEFAULT_AWAY_ALLOWLIST, "away"),
    )
    selected_team_goal_market = team_goal_market_shadow.build_candidates(fixtures)
    if not selected_team_goal_market.empty:
        selected_team_goal_market = selected_team_goal_market.merge(
            fixtures[["fixture_key", "source_tiers_present", "source_markets_present"]],
            on="fixture_key",
            how="left",
        )
        selected_team_goal_market = dedupe_team_goal_market_live(selected_team_goal_market)

    selected_team_goal = team_goal_shadow.select_combo(source)

    combo_rows = ftr_btts_shadow.build_combo_rows(source)
    selected_ftr_btts_raw = ftr_btts_shadow.apply_policy(combo_rows, ftr_btts_policy)
    selected_ftr_btts = ftr_btts_shadow.dedupe_selected(selected_ftr_btts_raw)
    if selected_ftr_btts.empty:
        selected_ftr_btts = selected_ftr_btts_raw.head(0).copy()

    deploy_changed = int(
        (
            source.get("deploy_tier", pd.Series("", index=source.index)).fillna("").astype(str)
            != annotated.get("deploy_tier", pd.Series("", index=annotated.index)).fillna("").astype(str)
        ).sum()
    )
    tier_changed = int(
        (
            source.get("tier", pd.Series("", index=source.index)).fillna("").astype(str)
            != annotated.get("tier", pd.Series("", index=annotated.index)).fillna("").astype(str)
        ).sum()
    )

    dashboard = pd.concat(
        [
            normalize_ou25(selected_ou25),
            normalize_ftr_c5(selected_ftr_c5),
            normalize_team_goal_markets(
                selected_team_goal_market,
                team_goal_market_scorecard,
                team_goal_league_stability,
                team_goal_team_stability,
            ),
            normalize_team_goal(selected_team_goal),
            normalize_ftr_btts(selected_ftr_btts),
            normalize_player_event_interactions(board_player_event_interactions),
        ],
        ignore_index=True,
        sort=False,
    )
    if not dashboard.empty and not board_team_shots_profile.empty:
        dashboard["fixture_key"] = dashboard["fixture_key"].astype(str)
        dashboard = dashboard.merge(board_team_shots_profile, on="fixture_key", how="left")
        dashboard["team_shots_profile_context_ready"] = (
            dashboard["team_shots_profile_context_ready"].astype("boolean").fillna(False).astype(bool)
        )
        dashboard["team_shots_profile_any_label"] = (
            dashboard["team_shots_profile_any_label"].astype("boolean").fillna(False).astype(bool)
        )
    if dashboard.empty:
        dashboard = pd.DataFrame(
            columns=[
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
                "team_shots_profile_context_ready",
                "team_shots_profile_any_label",
            ]
        )

    dashboard_path = board_out / f"{slug}__LIVE_SHADOW_RESEARCH_DASHBOARD.csv"
    counts_path = board_out / f"{slug}__LIVE_SHADOW_RESEARCH_COUNTS.csv"
    watch_path = board_out / f"{slug}__LIVE_SHADOW_RESEARCH_WATCH_TABLE.csv"
    team_shots_context_path = board_out / f"{slug}__TEAM_SHOTS_PROFILE_CONTEXT.csv"
    summary_path = board_out / f"{slug}__LIVE_SHADOW_RESEARCH_DASHBOARD_SUMMARY.md"

    dashboard.to_csv(dashboard_path, index=False)
    board_team_shots_profile.to_csv(team_shots_context_path, index=False)

    counts = count_by(dashboard, ["shadow_family", "shadow_stage"])
    counts.to_csv(counts_path, index=False)

    watch_cols = [
        "shadow_family",
        "league",
        "home_team_name",
        "away_team_name",
        "expression",
        "player_name",
        "team_name",
        "source_deploy_tier",
        "bookie_od",
        "model_prob",
        "value_edge",
        "league_stability_bucket",
        "team_stability_bucket",
        "watch_priority",
        "watch_flag",
        "home_team_shots_profile_labels",
        "away_team_shots_profile_labels",
        "match_expected_shots",
        "match_expected_sot",
        "match_expected_corners",
        "team_corner_pressure_score",
        "team_expected_corners",
        "team_expected_corners_pct",
        "keeper_save_pressure_score",
        "keeper_expected_sot_faced",
        "keeper_expected_sot_faced_pct",
        "attacking_team_name",
        "confidence_label",
        "lineup_watch_flags",
        "lineup_confirmation_status",
        "expected_minutes",
        "formation_matchup_label",
        "formation_pressure_score",
        "fixture_style_label",
        "fixture_foul_density_score",
        "fixture_wide_duel_score",
        "ref_cards_per_match",
        "interaction_match_mode",
        "attack_watch_gate_label",
        "booking_pressure_side",
        "booking_pressure_mode",
        "booking_context_label",
        "booking_hazard_score",
        "booking_role_cell",
        "player_sub_swap_review_mode",
        "fouled_context_cell_label",
    ]
    for col in watch_cols:
        if col not in dashboard.columns:
            dashboard[col] = np.nan
    watch = dashboard[watch_cols].copy() if not dashboard.empty else pd.DataFrame(columns=watch_cols)
    if "watch_priority" in watch.columns:
        watch["_watch_rank"] = watch["watch_priority"].map(lambda value: watch_priority_rank(str(value)))
        watch = watch.sort_values(["watch_flag", "_watch_rank", "shadow_family", "league"], ascending=[False, False, True, True])
        watch = watch.drop(columns=["_watch_rank"])
    else:
        watch = watch.sort_values(["watch_flag", "shadow_family", "league"], ascending=[False, True, True])
    watch.to_csv(watch_path, index=False)

    status = "PASS" if deploy_changed == 0 and tier_changed == 0 else "FAIL_TIER_MUTATION"
    summary = pd.DataFrame(
        [
            {
                "status": status,
                "fixture_range": board.fixture_range,
                "source_rows": len(source),
                "dashboard_rows": len(dashboard),
                "ou25_rows": len(selected_ou25),
                "team_goal_combo_rows": len(selected_team_goal),
                "team_goal_market_rows": len(selected_team_goal_market),
                "ftr_c5_rows": len(selected_ftr_c5),
                "ftr_btts_combo_rows": len(selected_ftr_btts),
                "player_event_interaction_rows": len(board_player_event_interactions),
                "player_event_interaction_source_rows": len(player_event_interactions),
                "team_shots_profile_context_rows": len(board_team_shots_profile),
                "team_shots_profile_labeled_context_rows": int(
                    board_team_shots_profile.get(
                        "team_shots_profile_any_label",
                        pd.Series(False, index=board_team_shots_profile.index),
                    ).fillna(False).sum()
                ),
                "team_shots_profile_source_rows": len(team_shots_profile),
                "team_shots_profile_projection_source_rows": len(team_shots_profile_rows),
                "ftr_btts_combo_source_rows": len(combo_rows),
                "ftr_btts_policy_rules": len(ftr_btts_policy),
                "deploy_tier_changed": deploy_changed,
                "tier_changed": tier_changed,
            }
        ]
    )

    lines = [
        "# Live Shadow Research Dashboard",
        "",
        "Unified research-only dashboard for forward-facing shadow systems.",
        "",
        "## Safety",
        markdown_table(summary),
        "",
        "## Shadow Counts",
        markdown_table(counts),
        "",
        "## Watch Table",
        markdown_table(watch.head(40)),
        "",
        "## Guardrails",
        "",
        "- No source deploy tier files changed.",
        "- No production rulebook changes.",
        "- Value remains additive only.",
        "- Shadow rows are instrumentation, not deployment promotion.",
        "- `OU25_RESTORE_NOW_SHADOW` remains the first repeat-QA priority.",
        "- Team-goal market shadows are model-only because direct team-goal odds are not in the scored board.",
        "- `TEAM_CORE` / `SHADOW_CORE` cells are watch-table priority only, not deployment permission.",
        "- TG25 and match-over-3.5 shadows remain watch-only-not-promotion.",
        "- Team-goal combo remains Spain/Germany proof shadow only.",
        "- FTR C5 side-shape rows are shadow-only and league/ring classified; they do not promote FTR.",
        "- FTR+BTTS combo remains strict sparse shadow instrumentation.",
        "- Player-event interaction rows are beta watch labels only: no priced prop odds and no deploy promotion.",
        "- Confirmed-starter attack-watch rows are separate from strict interaction proof and remain shots/SOT beta instrumentation only.",
        "- Side-aware booking rows are beta card-pressure instrumentation only and must be outcome-accumulated before promotion.",
        "- Player Sub Swap products require separate role-chain grading and are not mixed into named-player outcome evidence.",
        "- Team-shots profile columns are annotation-only API context for shots/SOT/corners/keeper-save pressure.",
        "- World Cup remains OBSERVE-only until fixtures, odds, squads, injuries, venues, and rest/travel context are normalized.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": status,
        "board_dir": str(board.board_dir),
        "fixture_range": board.fixture_range,
        "source_rows": len(source),
        "dashboard_rows": len(dashboard),
        "ou25_rows": len(selected_ou25),
        "ftr_c5_rows": len(selected_ftr_c5),
        "team_goal_market_rows": len(selected_team_goal_market),
        "team_goal_combo_rows": len(selected_team_goal),
        "ftr_btts_combo_rows": len(selected_ftr_btts),
        "player_event_interaction_rows": len(board_player_event_interactions),
        "team_shots_profile_context_rows": len(board_team_shots_profile),
        "team_shots_profile_labeled_context_rows": int(
            board_team_shots_profile.get(
                "team_shots_profile_any_label",
                pd.Series(False, index=board_team_shots_profile.index),
            ).fillna(False).sum()
        ),
        "summary_path": str(summary_path),
        "dashboard_path": str(dashboard_path),
        "team_shots_context_path": str(team_shots_context_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="predictions_output")
    parser.add_argument("--board-dir", default="")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--ou25-policy", default=str(c4_shadow.DEFAULT_POLICY))
    parser.add_argument("--ftr-c5-policy", default=str(DEFAULT_FTR_C5_POLICY))
    parser.add_argument("--ftr-btts-policy", default=str(ftr_btts_shadow.DEFAULT_POLICY))
    parser.add_argument("--team-goal-market-scorecard", default=str(DEFAULT_TEAM_GOAL_MARKET_SCORECARD))
    parser.add_argument("--team-goal-stability-dir", default=str(DEFAULT_TEAM_GOAL_STABILITY_DIR))
    parser.add_argument("--player-event-interaction-board", default=str(DEFAULT_PLAYER_EVENT_INTERACTION_BOARD))
    parser.add_argument(
        "--player-event-extra-board",
        action="append",
        default=[],
        help="Optional additional player-event shadow board(s) to concatenate into the unified dashboard.",
    )
    parser.add_argument("--team-shots-profile", default=str(DEFAULT_TEAM_SHOTS_PROFILE))
    parser.add_argument("--team-shots-profile-rows", default=str(DEFAULT_TEAM_SHOTS_PROFILE_ROWS))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ftr-btts-policy-limit", type=int, default=30)
    parser.add_argument("--ftr-btts-min-windows", type=int, default=20)
    parser.add_argument("--ftr-btts-min-graded", type=int, default=40)
    parser.add_argument("--ftr-btts-min-hit-rate", type=float, default=0.95)
    parser.add_argument("--ftr-btts-min-p25-hit-rate", type=float, default=1.0)
    parser.add_argument("--ftr-btts-max-negative-roi-windows", type=int, default=1)
    args = parser.parse_args()

    boards = boards_from_dir(Path(args.board_dir)) if args.board_dir else discover_boards(Path(args.root))
    if not boards:
        raise SystemExit("No complete live deploy tier board found.")
    if not args.all:
        boards = boards[: max(1, args.limit)]

    ou25_policy = pd.read_csv(args.ou25_policy)
    ftr_c5_policy = pd.read_csv(args.ftr_c5_policy) if Path(args.ftr_c5_policy).exists() else pd.DataFrame()
    ftr_btts_policy = ftr_btts_shadow.load_policy(
        Path(args.ftr_btts_policy),
        min_windows=args.ftr_btts_min_windows,
        min_graded=args.ftr_btts_min_graded,
        min_hit_rate=args.ftr_btts_min_hit_rate,
        min_p25_hit_rate=args.ftr_btts_min_p25_hit_rate,
        max_negative_roi_windows=args.ftr_btts_max_negative_roi_windows,
        limit=args.ftr_btts_policy_limit,
    )
    team_goal_market_scorecard = (
        pd.read_csv(args.team_goal_market_scorecard)
        if Path(args.team_goal_market_scorecard).exists()
        else pd.DataFrame()
    )
    team_goal_league_stability, team_goal_team_stability = load_team_goal_stability(Path(args.team_goal_stability_dir))
    player_event_interactions = (
        pd.read_csv(args.player_event_interaction_board)
        if Path(args.player_event_interaction_board).exists()
        else pd.DataFrame()
    )
    extra_player_event_boards = [
        pd.read_csv(path)
        for raw_path in args.player_event_extra_board
        for path in [Path(raw_path)]
        if path.exists()
    ]
    if extra_player_event_boards:
        player_event_interactions = pd.concat(
            [player_event_interactions, *extra_player_event_boards],
            ignore_index=True,
            sort=False,
        )
    team_shots_profile = load_team_shots_profile(Path(args.team_shots_profile))
    team_shots_profile_rows = load_team_shots_profile_rows(Path(args.team_shots_profile_rows))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = [
        run_board(
            board,
            ou25_policy=ou25_policy,
            ftr_c5_policy=ftr_c5_policy,
            ftr_btts_policy=ftr_btts_policy,
            team_goal_market_scorecard=team_goal_market_scorecard,
            team_goal_league_stability=team_goal_league_stability,
            team_goal_team_stability=team_goal_team_stability,
            player_event_interactions=player_event_interactions,
            team_shots_profile=team_shots_profile,
            team_shots_profile_rows=team_shots_profile_rows,
            outdir=outdir,
        )
        for board in boards
    ]
    index = pd.DataFrame(records)
    index_path = outdir / "live_shadow_research_dashboard_index.csv"
    index.to_csv(index_path, index=False)

    print(f"[ok] boards={len(records)}")
    print(f"[ok] wrote {index_path}")
    print(f"[ok] latest summary {records[0]['summary_path']}")

    if any(record["status"] != "PASS" for record in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
