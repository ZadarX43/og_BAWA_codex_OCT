#!/usr/bin/env python3
"""Backtest model-only team-goal shadow markets over walk-forward scored files.

Research-only. These markets do not require direct team-goal odds:
  - HOME_TEAM_OVER_1_5_SHADOW
  - AWAY_TEAM_OVER_1_5_SHADOW
  - HOME_TEAM_OVER_2_5_SHADOW for monster-side pockets
  - AWAY_TEAM_OVER_2_5_SHADOW for monster-side pockets
  - MATCH_OVER_3_5_SHADOW when total-goal mass and dominant-team mass agree

Outputs hit-rate and window/league/team stability only. No deploy behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SCORED_ROOT = Path("predictions_output/walk_forward_phase8h_value_layer_full_relock_2026_04_21_r3")
DEFAULT_OUTDIR = Path("reports/2026-05-06/team_goal_shadow_market_backtest")
DEFAULT_HOME_ALLOWLIST = Path("reports/2026-04-21/PHASE9B_HOME_GE2_ALLOWLIST_DRAFT.csv")
DEFAULT_AWAY_ALLOWLIST = Path("reports/2026-04-21/PHASE9B_AWAY_GE2_ALLOWLIST_DRAFT.csv")
DEFAULT_MERGED_ROOT = Path("Matches/__merged__")


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


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


def scored_files(root: Path) -> list[Path]:
    return sorted(root.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))


def safe_max(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return max(finite) if finite else np.nan


def league_from_merged_path(path: Path) -> str:
    return path.name.replace("__merged.csv", "").replace("_", " ")


def load_truth(merged_root: Path) -> pd.DataFrame:
    parts = []
    columns = {
        "fixture_key",
        "match_date",
        "home_team_name",
        "away_team_name",
        "home_team_goal_count",
        "away_team_goal_count",
        "status",
    }
    for path in sorted(merged_root.glob("*__merged.csv")):
        try:
            df = pd.read_csv(path, usecols=lambda c: c in columns)
        except Exception:
            continue
        if df.empty or "fixture_key" not in df.columns:
            continue
        df["league"] = league_from_merged_path(path)
        parts.append(df)
    if not parts:
        return pd.DataFrame(
            columns=[
                "league",
                "fixture_key",
                "truth_home_goals",
                "truth_away_goals",
                "actual_match_over35",
            ]
        )
    truth = pd.concat(parts, ignore_index=True, sort=False)
    if "status" in truth.columns:
        status = norm_text(truth["status"]).str.lower()
        truth = truth[status.eq("") | status.eq("complete")].copy()
    truth["truth_home_goals"] = num(truth.get("home_team_goal_count", np.nan))
    truth["truth_away_goals"] = num(truth.get("away_team_goal_count", np.nan))
    truth["actual_match_over35"] = (truth["truth_home_goals"] + truth["truth_away_goals"]).ge(4).astype(float)
    truth.loc[truth["truth_home_goals"].isna() | truth["truth_away_goals"].isna(), "actual_match_over35"] = np.nan
    keep = ["league", "fixture_key", "truth_home_goals", "truth_away_goals", "actual_match_over35"]
    return truth[keep].dropna(subset=["fixture_key"]).drop_duplicates(["league", "fixture_key"], keep="last")


def load_allowlist(path: Path, side: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["league", "team", f"{side}_allowlist_tier", f"{side}_allowlist_hit"])
    df = pd.read_csv(path)
    keep = df[
        df.get("recommended_status", "").astype("string").eq("RESEARCH_READY_DOMESTIC")
        & df.get("allowlist_tier", "").astype("string").isin(["CORE", "STRONG"])
        & num(df.get("sample_ok_flag", 0)).eq(1)
        & num(df.get("ge2_edge_flag", 0)).eq(1)
        & num(df.get("scoring_edge_flag", 0)).eq(1)
    ].copy()
    return keep[["league", "team", "allowlist_tier"]].rename(
        columns={"team": f"{side}_allowlist_team", "allowlist_tier": f"{side}_allowlist_tier"}
    )


def coalesce_first(group: pd.DataFrame, col: str):
    if col not in group.columns:
        return np.nan
    series = group[col].dropna()
    return series.iloc[0] if len(series) else np.nan


def fixture_level(path: Path) -> pd.DataFrame:
    usecols = lambda c: c in {
        "league",
        "match_date",
        "home_team_name",
        "away_team_name",
        "fixture_key",
        "market",
        "selection",
        "window_id",
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
        "home_team_ge2_candidate_flag",
        "away_team_ge2_candidate_flag",
        "home_team_high_scoring_flag",
        "away_team_high_scoring_flag",
        "home_team_goal_count",
        "away_team_goal_count",
        "actual_tg15_home",
        "actual_tg15_away",
        "actual_tg25_home",
        "actual_tg25_away",
        "actual_over25",
        "cs_mass_over25",
        "mass_4plus_goals",
        "p_meta_ou25",
        "p_meta_btts",
        "p_meta_ftr",
        "value_edge",
        "value_edge_tier",
        "bookie_od",
        "model_p_for_bookie",
        "ftr_combo_live_product",
        "ftr_combo_live_tier",
        "team_context_label",
    }
    df = pd.read_csv(path, usecols=usecols)
    if df.empty:
        return df
    if "window_id" not in df.columns:
        df["window_id"] = path.parts[-3] if len(path.parts) >= 3 else ""

    group_cols = ["fixture_key"]
    rows = []
    for fixture_key, group in df.groupby(group_cols, dropna=False):
        if isinstance(fixture_key, tuple):
            fixture_key = fixture_key[0]
        row = {"fixture_key": fixture_key}
        for col in [
            "league",
            "match_date",
            "home_team_name",
            "away_team_name",
            "window_id",
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
            "home_team_ge2_candidate_flag",
            "away_team_ge2_candidate_flag",
            "home_team_high_scoring_flag",
            "away_team_high_scoring_flag",
            "home_team_goal_count",
            "away_team_goal_count",
            "actual_tg15_home",
            "actual_tg15_away",
            "actual_tg25_home",
            "actual_tg25_away",
            "actual_over25",
            "cs_mass_over25",
            "mass_4plus_goals",
            "p_meta_ou25",
            "p_meta_btts",
            "p_meta_ftr",
            "ftr_combo_live_product",
            "ftr_combo_live_tier",
            "team_context_label",
        ]:
            row[col] = coalesce_first(group, col)
        rows.append(row)
    out = pd.DataFrame(rows)

    for side in ["home", "away"]:
        ge2 = f"p_{side}_ge2"
        ge3 = f"p_{side}_ge3"
        out[ge2] = num(out[ge2]).where(num(out[ge2]).notna(), num(out.get(f"pois_{side}_ge2", np.nan)))
        out[ge2] = num(out[ge2]).where(num(out[ge2]).notna(), num(out.get(f"{side}_ge2_confidence", np.nan)))
        out[ge3] = num(out[ge3]).where(num(out[ge3]).notna(), num(out.get(f"pois_{side}_ge3", np.nan)))

    hg = num(out.get("home_team_goal_count", np.nan))
    ag = num(out.get("away_team_goal_count", np.nan))
    out["actual_home_tg15"] = num(out.get("actual_tg15_home", np.nan)).where(num(out.get("actual_tg15_home", np.nan)).notna(), hg.ge(2).astype(float))
    out["actual_away_tg15"] = num(out.get("actual_tg15_away", np.nan)).where(num(out.get("actual_tg15_away", np.nan)).notna(), ag.ge(2).astype(float))
    out["actual_home_tg25"] = num(out.get("actual_tg25_home", np.nan)).where(num(out.get("actual_tg25_home", np.nan)).notna(), hg.ge(3).astype(float))
    out["actual_away_tg25"] = num(out.get("actual_tg25_away", np.nan)).where(num(out.get("actual_tg25_away", np.nan)).notna(), ag.ge(3).astype(float))
    for col in ["actual_home_tg15", "actual_away_tg15", "actual_home_tg25", "actual_away_tg25"]:
        out.loc[out[col].isna() & hg.isna() & ag.isna(), col] = np.nan
    out["actual_match_over35"] = (hg + ag).ge(4).astype(float)
    out.loc[hg.isna() | ag.isna(), "actual_match_over35"] = np.nan
    return out


def attach_allowlists(fixtures: pd.DataFrame, home_allow: pd.DataFrame, away_allow: pd.DataFrame) -> pd.DataFrame:
    out = fixtures.copy()
    out = out.merge(
        home_allow,
        left_on=["league", "home_team_name"],
        right_on=["league", "home_allowlist_team"],
        how="left",
    )
    out = out.merge(
        away_allow,
        left_on=["league", "away_team_name"],
        right_on=["league", "away_allowlist_team"],
        how="left",
    )
    out["home_allowlist_hit"] = out["home_allowlist_tier"].notna()
    out["away_allowlist_hit"] = out["away_allowlist_tier"].notna()
    return out


def attach_truth(fixtures: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if truth.empty:
        return fixtures
    out = fixtures.merge(truth, on=["league", "fixture_key"], how="left", suffixes=("", "_truth"))
    out["home_team_goal_count"] = num(out.get("home_team_goal_count", np.nan)).where(
        num(out.get("home_team_goal_count", np.nan)).notna(),
        num(out.get("truth_home_goals", np.nan)),
    )
    out["away_team_goal_count"] = num(out.get("away_team_goal_count", np.nan)).where(
        num(out.get("away_team_goal_count", np.nan)).notna(),
        num(out.get("truth_away_goals", np.nan)),
    )
    out["actual_match_over35"] = num(out.get("actual_match_over35", np.nan)).where(
        num(out.get("actual_match_over35", np.nan)).notna(),
        num(out.get("actual_match_over35_truth", np.nan)),
    )
    return out


def add_candidate(rows: list[dict], fixture: pd.Series, product: str, policy: str, side: str, prob: float, hit_col: str, reason: str) -> None:
    rows.append(
        {
            "league": fixture["league"],
            "match_date": fixture["match_date"],
            "home_team_name": fixture["home_team_name"],
            "away_team_name": fixture["away_team_name"],
            "fixture_key": fixture["fixture_key"],
            "window_id": fixture["window_id"],
            "shadow_product": product,
            "shadow_policy": policy,
            "shadow_side": side,
            "shadow_team_name": fixture["home_team_name"] if side == "HOME" else fixture["away_team_name"] if side == "AWAY" else "",
            "model_prob": prob,
            "cs_mass_over25": fixture.get("cs_mass_over25", np.nan),
            "mass_4plus_goals": fixture.get("mass_4plus_goals", np.nan),
            "p_meta_ou25": fixture.get("p_meta_ou25", np.nan),
            "p_home_ge2": fixture.get("p_home_ge2", np.nan),
            "p_away_ge2": fixture.get("p_away_ge2", np.nan),
            "p_home_ge3": fixture.get("p_home_ge3", np.nan),
            "p_away_ge3": fixture.get("p_away_ge3", np.nan),
            "home_allowlist_tier": fixture.get("home_allowlist_tier", ""),
            "away_allowlist_tier": fixture.get("away_allowlist_tier", ""),
            "correct": fixture.get(hit_col, np.nan),
            "reason": reason,
        }
    )


def build_candidates(fixtures: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, f in fixtures.iterrows():
        h_ge2 = float(f.get("p_home_ge2", np.nan)) if pd.notna(f.get("p_home_ge2", np.nan)) else np.nan
        a_ge2 = float(f.get("p_away_ge2", np.nan)) if pd.notna(f.get("p_away_ge2", np.nan)) else np.nan
        h_ge3 = float(f.get("p_home_ge3", np.nan)) if pd.notna(f.get("p_home_ge3", np.nan)) else np.nan
        a_ge3 = float(f.get("p_away_ge3", np.nan)) if pd.notna(f.get("p_away_ge3", np.nan)) else np.nan
        over25 = float(f.get("cs_mass_over25", np.nan)) if pd.notna(f.get("cs_mass_over25", np.nan)) else np.nan
        mass4 = float(f.get("mass_4plus_goals", np.nan)) if pd.notna(f.get("mass_4plus_goals", np.nan)) else np.nan
        p_meta_ou25 = float(f.get("p_meta_ou25", np.nan)) if pd.notna(f.get("p_meta_ou25", np.nan)) else np.nan
        h_allow = bool(f.get("home_allowlist_hit", False))
        a_allow = bool(f.get("away_allowlist_hit", False))
        h_core = str(f.get("home_allowlist_tier", "")).upper() == "CORE"
        a_core = str(f.get("away_allowlist_tier", "")).upper() == "CORE"

        if np.isfinite(h_ge2):
            if h_ge2 >= 0.56 and (not np.isfinite(over25) or over25 >= 0.52):
                add_candidate(rows, f, "HOME_TEAM_OVER_1_5_SHADOW", "TG15_MODEL", "HOME", h_ge2, "actual_home_tg15", "p_home_ge2>=0.56 + goal-mass")
            if h_allow and h_ge2 >= 0.50 and (not np.isfinite(over25) or over25 >= 0.48):
                add_candidate(rows, f, "HOME_TEAM_OVER_1_5_SHADOW", "TG15_ALLOWLIST", "HOME", h_ge2, "actual_home_tg15", "allowlist + p_home_ge2>=0.50")
            if h_core and h_ge2 >= 0.58 and np.isfinite(over25) and over25 >= 0.42:
                add_candidate(rows, f, "HOME_TEAM_OVER_1_5_SHADOW", "TG15_CORE_WATCH", "HOME", h_ge2, "actual_home_tg15", "CORE allowlist + p_home_ge2>=0.58 + soft total mass")
            if h_ge2 >= 0.63 and (not np.isfinite(over25) or over25 >= 0.60) and (not np.isfinite(p_meta_ou25) or p_meta_ou25 >= 0.80):
                add_candidate(rows, f, "HOME_TEAM_OVER_1_5_SHADOW", "TG15_PREMIUM", "HOME", h_ge2, "actual_home_tg15", "p_home_ge2>=0.63 + OU25/meta mass")

        if np.isfinite(a_ge2):
            if a_ge2 >= 0.56 and (not np.isfinite(over25) or over25 >= 0.52):
                add_candidate(rows, f, "AWAY_TEAM_OVER_1_5_SHADOW", "TG15_MODEL", "AWAY", a_ge2, "actual_away_tg15", "p_away_ge2>=0.56 + goal-mass")
            if a_allow and a_ge2 >= 0.50 and (not np.isfinite(over25) or over25 >= 0.48):
                add_candidate(rows, f, "AWAY_TEAM_OVER_1_5_SHADOW", "TG15_ALLOWLIST", "AWAY", a_ge2, "actual_away_tg15", "allowlist + p_away_ge2>=0.50")
            if a_core and a_ge2 >= 0.62 and np.isfinite(over25) and over25 >= 0.42:
                add_candidate(rows, f, "AWAY_TEAM_OVER_1_5_SHADOW", "TG15_CORE_WATCH", "AWAY", a_ge2, "actual_away_tg15", "CORE allowlist + p_away_ge2>=0.62 + soft total mass")
            if a_ge2 >= 0.63 and (not np.isfinite(over25) or over25 >= 0.60) and (not np.isfinite(p_meta_ou25) or p_meta_ou25 >= 0.80):
                add_candidate(rows, f, "AWAY_TEAM_OVER_1_5_SHADOW", "TG15_PREMIUM", "AWAY", a_ge2, "actual_away_tg15", "p_away_ge2>=0.63 + OU25/meta mass")

        if np.isfinite(h_ge3) and h_core and h_ge3 >= 0.30 and h_ge2 >= 0.60 and (not np.isfinite(mass4) or mass4 >= 0.35):
            add_candidate(rows, f, "HOME_TEAM_OVER_2_5_SHADOW", "TG25_MONSTER", "HOME", h_ge3, "actual_home_tg25", "CORE monster + p_home_ge3>=0.30")
        if np.isfinite(a_ge3) and a_core and a_ge3 >= 0.30 and a_ge2 >= 0.60 and (not np.isfinite(mass4) or mass4 >= 0.35):
            add_candidate(rows, f, "AWAY_TEAM_OVER_2_5_SHADOW", "TG25_MONSTER", "AWAY", a_ge3, "actual_away_tg25", "CORE monster + p_away_ge3>=0.30")

        dominant_ge2 = safe_max([h_ge2, a_ge2])
        dominant_ge3 = safe_max([h_ge3, a_ge3])
        if np.isfinite(over25) and np.isfinite(mass4) and over25 >= 0.68 and mass4 >= 0.42 and dominant_ge2 >= 0.60:
            add_candidate(rows, f, "MATCH_OVER_3_5_SHADOW", "MO35_GOALMASS_DOMINANCE", "MATCH", mass4, "actual_match_over35", "mass4>=0.42 + cs_mass_over25>=0.68 + dominant GE2")
        if np.isfinite(mass4) and np.isfinite(dominant_ge3) and mass4 >= 0.35 and dominant_ge3 >= 0.35 and dominant_ge2 >= 0.58:
            add_candidate(rows, f, "MATCH_OVER_3_5_SHADOW", "MO35_MONSTER_SIDE", "MATCH", mass4, "actual_match_over35", "mass4>=0.35 + dominant GE3")

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["candidate_key"] = out["fixture_key"].astype("string") + "|" + out["shadow_product"].astype("string") + "|" + out["shadow_policy"].astype("string")
    return out.drop_duplicates("candidate_key").reset_index(drop=True)


def scorecard(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        hit = num(group["correct"])
        graded = int(hit.notna().sum())
        wins = float(hit.eq(1).sum())
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": int(hit.eq(0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "avg_model_prob": float(num(group["model_prob"]).mean()),
                "active_windows": int(group["window_id"].nunique()) if "window_id" in group.columns else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def stability(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    win = scorecard(df, group_cols + ["window_id"])
    rows = []
    for keys, group in win.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        graded = int(group["graded"].sum())
        wins = float(group["wins"].sum())
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "active_windows": int(group["window_id"].nunique()),
                "rows": int(group["rows"].sum()),
                "graded": graded,
                "wins": wins,
                "losses": int(group["losses"].sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "median_rows_per_window": float(group["rows"].median()),
                "windows_below_70": int(group["hit_rate"].lt(0.70).sum()),
                "windows_below_60": int(group["hit_rate"].lt(0.60).sum()),
                "mean_window_hit_rate": float(group["hit_rate"].mean()),
                "median_window_hit_rate": float(group["hit_rate"].median()),
                "p25_window_hit_rate": float(group["hit_rate"].quantile(0.25)),
                "p10_window_hit_rate": float(group["hit_rate"].quantile(0.10)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", default=str(DEFAULT_SCORED_ROOT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--home-allowlist", default=str(DEFAULT_HOME_ALLOWLIST))
    parser.add_argument("--away-allowlist", default=str(DEFAULT_AWAY_ALLOWLIST))
    parser.add_argument("--merged-root", default=str(DEFAULT_MERGED_ROOT))
    parser.add_argument("--limit-files", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = scored_files(Path(args.scored_root))
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit("No scored files found.")

    home_allow = load_allowlist(Path(args.home_allowlist), "home")
    away_allow = load_allowlist(Path(args.away_allowlist), "away")

    fixture_parts = []
    for path in files:
        fixture_parts.append(fixture_level(path))
    fixtures = pd.concat(fixture_parts, ignore_index=True, sort=False)
    fixtures = attach_allowlists(fixtures, home_allow, away_allow)
    fixtures = attach_truth(fixtures, load_truth(Path(args.merged_root)))
    fixtures.to_csv(outdir / "team_goal_shadow_fixture_level.csv", index=False)

    candidates = build_candidates(fixtures)
    candidates.to_csv(outdir / "team_goal_shadow_market_candidates.csv", index=False)

    product_policy = scorecard(candidates, ["shadow_product", "shadow_policy"])
    product_policy.to_csv(outdir / "team_goal_shadow_scorecard_by_product_policy.csv", index=False)

    league = scorecard(candidates, ["shadow_product", "shadow_policy", "league"])
    league.to_csv(outdir / "team_goal_shadow_scorecard_by_league.csv", index=False)

    team = scorecard(candidates, ["shadow_product", "shadow_policy", "league", "shadow_team_name"])
    team = team[team["graded"].ge(5)].sort_values(["hit_rate", "graded"], ascending=[False, False])
    team.to_csv(outdir / "team_goal_shadow_scorecard_by_team.csv", index=False)

    stable = stability(candidates, ["shadow_product", "shadow_policy"])
    stable.to_csv(outdir / "team_goal_shadow_window_stability.csv", index=False)

    league_stable = stability(candidates, ["shadow_product", "shadow_policy", "league"])
    league_stable.to_csv(outdir / "team_goal_shadow_league_window_stability.csv", index=False)

    top_policy = product_policy.sort_values(["hit_rate", "graded"], ascending=[False, False])
    top_league = league[league["graded"].ge(25)].sort_values(["hit_rate", "graded"], ascending=[False, False]).head(30)
    top_team = team.head(30)

    summary = [
        "# Team Goal Shadow Market Backtest",
        "",
        "Fresh research-only walk-forward backtest for model-only team-goal shadow markets.",
        "",
        f"- Source scored files: `{len(files)}`",
        f"- Fixture rows: `{len(fixtures)}`",
        f"- Shadow candidate rows: `{len(candidates)}`",
        "",
        "## Product / Policy Scorecard",
        markdown_table(top_policy),
        "",
        "## Window Stability",
        markdown_table(stable.sort_values(["hit_rate", "graded"], ascending=[False, False])),
        "",
        "## Best League Cells",
        markdown_table(top_league),
        "",
        "## Best Team Cells",
        markdown_table(top_team),
        "",
        "## Read",
        "",
        "- This is hit-rate validation only; direct team-goal odds are not required.",
        "- TG15 policies are candidates for live dashboard shadow instrumentation.",
        "- TG25 and MO35 should stay monster/goal-mass only until live repeats prove stability.",
    ]
    (outdir / "team_goal_shadow_market_backtest_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print(f"[ok] files={len(files)} fixtures={len(fixtures)} candidates={len(candidates)}")
    print(f"[ok] wrote {outdir}")


if __name__ == "__main__":
    main()
