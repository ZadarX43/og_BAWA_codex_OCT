#!/usr/bin/env python3
"""
Research-only EPL player-event system validation pack.

This script does not recalibrate live thresholds or write deploy artifacts. It
uses local API-Football EPL actuals as a historical lab to test whether player
event feature shapes survive over time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


EVENT_COLS = [
    "shots_total",
    "shots_on_target",
    "fouls_committed",
    "fouls_drawn",
    "tackles",
    "yellow_cards",
    "red_cards",
    "saves",
    "dribbles_attempted",
    "dribbles_successful",
    "duels_total",
    "duels_won",
]

MARKETS = {
    "shots_ge1": "shots_total",
    "sot_ge1": "shots_on_target",
    "fouls_committed_ge1": "fouls_committed",
    "fouls_drawn_ge1": "fouls_drawn",
    "tackles_ge1": "tackles",
    "card_any": "card_any",
    "keeper_saves_ge2": "saves",
}

BOOKMAKER_WORDING = {
    "shots_ge1": "Player 1+ shot",
    "sot_ge1": "Player 1+ shot on target",
    "fouls_committed_ge1": "Player 1+ foul committed",
    "fouls_drawn_ge1": "Player 1+ foul won / drawn / to be fouled",
    "tackles_ge1": "Player 1+ tackle",
    "card_any": "Player to be carded",
    "keeper_saves_ge2": "Goalkeeper 2+ saves",
}

SETTLEMENT_WORDING = {
    "shots_ge1": "API `shots_total >= 1`",
    "sot_ge1": "API `shots_on_target >= 1`",
    "fouls_committed_ge1": "API `fouls_committed >= 1`",
    "fouls_drawn_ge1": "API `fouls_drawn >= 1`",
    "tackles_ge1": "API `tackles >= 1`",
    "card_any": "API `yellow_cards + red_cards >= 1`",
    "keeper_saves_ge2": "API `saves >= 2`, keepers only",
}


def safe_auc(y: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce")
    score = pd.to_numeric(score, errors="coerce")
    mask = y.notna() & score.notna()
    if mask.sum() < 100 or y[mask].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], score[mask]))


def safe_ap(y: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce")
    score = pd.to_numeric(score, errors="coerce")
    mask = y.notna() & score.notna()
    if mask.sum() < 100 or y[mask].nunique() < 2:
        return float("nan")
    return float(average_precision_score(y[mask], score[mask]))


def top_bucket_lift(df: pd.DataFrame, target: str, score_col: str, frac: float = 0.10) -> tuple[float, float, int]:
    work = df[[target, score_col]].dropna().copy()
    if len(work) < 100:
        return float("nan"), float("nan"), 0
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna()
    if work.empty:
        return float("nan"), float("nan"), 0
    n = max(1, int(round(len(work) * frac)))
    top = work.sort_values(score_col, ascending=False).head(n)
    base = float(work[target].mean())
    hit = float(top[target].mean())
    lift = hit / base if base > 0 else float("nan")
    return hit, lift, len(top)


def rolling_prior(df: pd.DataFrame, group_cols: list[str], value_col: str, window: int = 10, min_periods: int = 3) -> pd.Series:
    return (
        df.groupby(group_cols, dropna=False)[value_col]
        .transform(lambda s: s.shift().rolling(window, min_periods=min_periods).mean())
    )


def expanding_count_prior(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    return df.groupby(group_cols, dropna=False).cumcount()


def load_seasons(normalized_dir: Path, league_tag: str, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_frames = []
    fixture_frames = []
    team_frames = []
    for season in seasons:
        player_path = normalized_dir / f"match_player_stats__{league_tag}__{season}.csv"
        fixture_path = normalized_dir / f"fixtures_master__{league_tag}__{season}.csv"
        team_path = normalized_dir / f"match_team_stats__{league_tag}__{season}.csv"
        if not player_path.exists() or not fixture_path.exists() or not team_path.exists():
            continue
        players = pd.read_csv(player_path, low_memory=False)
        fixtures = pd.read_csv(fixture_path, low_memory=False)
        teams = pd.read_csv(team_path, low_memory=False)
        players["season"] = season
        fixtures["season"] = season
        teams["season"] = season
        player_frames.append(players)
        fixture_frames.append(fixtures)
        team_frames.append(teams)
    if not player_frames:
        raise FileNotFoundError("No player-stat files found for requested seasons.")
    return pd.concat(player_frames, ignore_index=True), pd.concat(fixture_frames, ignore_index=True), pd.concat(team_frames, ignore_index=True)


def build_dataset(players: pd.DataFrame, fixtures: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    fixtures = fixtures.copy()
    players = players.copy()
    team_stats = team_stats.copy()

    fixtures["match_date"] = pd.to_datetime(fixtures["match_date"], errors="coerce")
    for col in ["fixture_id", "home_team_id", "away_team_id"]:
        fixtures[col] = pd.to_numeric(fixtures[col], errors="coerce")
    for col in ["fixture_id", "team_id"]:
        players[col] = pd.to_numeric(players[col], errors="coerce")
        team_stats[col] = pd.to_numeric(team_stats[col], errors="coerce")

    keep_fixture_cols = [
        "fixture_id",
        "season",
        "match_date",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "referee_name",
    ]
    df = players.merge(fixtures[keep_fixture_cols], on=["fixture_id", "season"], how="left")
    df["is_home"] = (df["team_id"] == df["home_team_id"]).astype(int)
    df["opponent_team_id"] = np.where(df["is_home"].eq(1), df["away_team_id"], df["home_team_id"])
    df["opponent_team_name"] = np.where(df["is_home"].eq(1), df["away_team_name"], df["home_team_name"])
    df["card_any"] = (
        pd.to_numeric(df.get("yellow_cards", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("red_cards", 0), errors="coerce").fillna(0)
    )

    for col in EVENT_COLS + ["minutes", "started_flag", "rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[pd.to_numeric(df["minutes"], errors="coerce").fillna(0).gt(0)].copy()
    df["position"] = df["position"].fillna("UNK").astype(str).str.upper().str[:1]
    df["is_keeper"] = df["position"].eq("G").astype(int)
    df["is_defender"] = df["position"].eq("D").astype(int)
    df["is_midfielder"] = df["position"].eq("M").astype(int)
    df["is_forward"] = df["position"].eq("F").astype(int)
    df["is_attacker_mid_fwd"] = df["position"].isin(["M", "F"]).astype(int)

    team_stats = team_stats.merge(
        fixtures[["fixture_id", "season", "match_date", "referee_name"]],
        on=["fixture_id", "season"],
        how="left",
    )
    team_stats["match_date"] = pd.to_datetime(team_stats["match_date"], errors="coerce")
    team_stats = team_stats.sort_values(["match_date", "fixture_id", "team_id"]).copy()
    team_numeric_cols = [
        "shots_total",
        "shots_on_goal",
        "possession_pct",
        "fouls_for",
        "yellow_cards",
        "red_cards",
        "corners_for",
        "goals_for",
        "goals_against",
    ]
    for col in team_numeric_cols:
        team_stats[col] = pd.to_numeric(team_stats.get(col, np.nan), errors="coerce")
        team_stats[f"team_{col}_l5"] = rolling_prior(team_stats, ["team_id"], col, window=5, min_periods=3)

    own_prior_cols = ["fixture_id", "team_id"] + [f"team_{col}_l5" for col in team_numeric_cols]
    own = team_stats[own_prior_cols].copy()
    opp = own.rename(
        columns={
            "team_id": "opponent_team_id",
            **{f"team_{col}_l5": f"opp_{col}_l5" for col in team_numeric_cols},
        }
    )
    df = df.merge(own, on=["fixture_id", "team_id"], how="left")
    df = df.merge(opp, on=["fixture_id", "opponent_team_id"], how="left")

    referee_cards = (
        team_stats.groupby(["fixture_id", "season", "match_date", "referee_name"], dropna=False)
        .agg(fixture_cards=("yellow_cards", "sum"), fixture_reds=("red_cards", "sum"))
        .reset_index()
        .sort_values(["match_date", "fixture_id"])
    )
    referee_cards["ref_cards_l10"] = rolling_prior(referee_cards, ["referee_name"], "fixture_cards", window=10, min_periods=3)
    df = df.merge(
        referee_cards[["fixture_id", "referee_name", "ref_cards_l10"]],
        on=["fixture_id", "referee_name"],
        how="left",
    )

    df = df.sort_values(["match_date", "fixture_id", "team_id", "player_id"]).copy()
    df["prior_player_appearances"] = expanding_count_prior(df, ["player_id"])

    for col in ["shots_total", "shots_on_target", "fouls_committed", "fouls_drawn", "tackles", "card_any", "saves", "dribbles_attempted", "duels_total"]:
        per90 = f"{col}_per90"
        df[per90] = np.where(df["minutes"].gt(0), df[col] / df["minutes"] * 90.0, np.nan)
        df[f"{col}_per90_l10"] = rolling_prior(df, ["player_id"], per90, window=10, min_periods=3)
        df[f"{col}_hit_l10"] = rolling_prior(df.assign(_hit=(df[col] >= 1).astype(int)), ["player_id"], "_hit", window=10, min_periods=3)

    df["saves_ge2_hit_l10"] = rolling_prior(df.assign(_hit=(df["saves"] >= 2).astype(int)), ["player_id"], "_hit", window=10, min_periods=3)

    for market, col in MARKETS.items():
        threshold = 2 if market == "keeper_saves_ge2" else 1
        df[market] = (pd.to_numeric(df[col], errors="coerce").fillna(0) >= threshold).astype(int)

    return df


def add_signal_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_shots_ge1"] = (
        out["shots_total_per90_l10"].fillna(0) * 0.65
        + out["shots_total_hit_l10"].fillna(0) * 0.35
        + out["is_forward"] * 0.25
        + out["is_midfielder"] * 0.08
        + out["team_shots_total_l5"].fillna(out["team_shots_total_l5"].median()) * 0.015
    )
    out["score_sot_ge1"] = (
        out["shots_on_target_per90_l10"].fillna(0) * 0.70
        + out["shots_on_target_hit_l10"].fillna(0) * 0.50
        + out["shots_total_per90_l10"].fillna(0) * 0.10
        + out["is_forward"] * 0.20
    )
    out["score_fouls_committed_ge1"] = (
        out["fouls_committed_per90_l10"].fillna(0) * 0.65
        + out["tackles_per90_l10"].fillna(0) * 0.20
        + out["is_midfielder"] * 0.15
        + out["is_defender"] * 0.10
        + out["opp_possession_pct_l5"].fillna(out["opp_possession_pct_l5"].median()) * 0.006
    )
    out["score_fouls_drawn_ge1"] = (
        out["fouls_drawn_per90_l10"].fillna(0) * 0.65
        + out["dribbles_attempted_per90_l10"].fillna(0) * 0.18
        + out["is_forward"] * 0.15
        + out["is_midfielder"] * 0.10
    )
    out["score_tackles_ge1"] = (
        out["tackles_per90_l10"].fillna(0) * 0.75
        + out["is_defender"] * 0.12
        + out["is_midfielder"] * 0.16
        + out["opp_possession_pct_l5"].fillna(out["opp_possession_pct_l5"].median()) * 0.004
    )
    out["score_card_any"] = (
        out["card_any_hit_l10"].fillna(0) * 0.55
        + out["fouls_committed_per90_l10"].fillna(0) * 0.22
        + out["tackles_per90_l10"].fillna(0) * 0.12
        + out["is_defender"] * 0.10
        + out["is_midfielder"] * 0.08
        + out["ref_cards_l10"].fillna(out["ref_cards_l10"].median()) * 0.03
    )
    out["score_keeper_saves_ge2"] = (
        out["saves_per90_l10"].fillna(0) * 0.65
        + out["saves_ge2_hit_l10"].fillna(0) * 0.50
        + out["opp_shots_total_l5"].fillna(out["opp_shots_total_l5"].median()) * 0.035
        + out["opp_shots_on_goal_l5"].fillna(out["opp_shots_on_goal_l5"].median()) * 0.08
    )
    return out


def assign_system_style_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eligible = out["prior_player_appearances"].ge(5)
    for market in MARKETS:
        score_col = f"score_{market}"
        label_col = f"{market}_system_style_label"
        out[label_col] = "NO_HISTORY"
        scoped = eligible & out[score_col].notna()
        if market == "keeper_saves_ge2":
            scoped &= out["is_keeper"].eq(1)
        if scoped.sum() < 100:
            continue
        q90 = out.loc[scoped, score_col].quantile(0.90)
        q80 = out.loc[scoped, score_col].quantile(0.80)
        q65 = out.loc[scoped, score_col].quantile(0.65)
        out.loc[scoped & out[score_col].ge(q90), label_col] = "SHADOW_CORE_PROXY"
        out.loc[scoped & out[score_col].lt(q90) & out[score_col].ge(q80), label_col] = "STRONG_WATCH_PROXY"
        out.loc[scoped & out[score_col].lt(q80) & out[score_col].ge(q65), label_col] = "ALT_WATCH_PROXY"
        out.loc[scoped & out[score_col].lt(q65), label_col] = "OBSERVE_PROXY"
    return out


def market_feature_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market in MARKETS:
        scope = df[df["prior_player_appearances"].ge(5)].copy()
        if market == "keeper_saves_ge2":
            scope = scope[scope["is_keeper"].eq(1)].copy()
        target = market
        score = f"score_{market}"
        top_hit, top_lift, top_rows = top_bucket_lift(scope, target, score, 0.10)
        rows.append(
            {
                "market": market,
                "bookmaker_wording": BOOKMAKER_WORDING[market],
                "api_settlement": SETTLEMENT_WORDING[market],
                "rows": len(scope),
                "fixtures": scope["fixture_id"].nunique(),
                "base_hit_rate": float(scope[target].mean()) if len(scope) else float("nan"),
                "auc": safe_auc(scope[target], scope[score]),
                "average_precision": safe_ap(scope[target], scope[score]),
                "top10_hit_rate": top_hit,
                "top10_lift": top_lift,
                "top10_rows": top_rows,
            }
        )
    return pd.DataFrame(rows)


def label_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market in MARKETS:
        label_col = f"{market}_system_style_label"
        scope = df[df["prior_player_appearances"].ge(5)].copy()
        if market == "keeper_saves_ge2":
            scope = scope[scope["is_keeper"].eq(1)].copy()
        base = float(scope[market].mean()) if len(scope) else float("nan")
        for label, group in scope.groupby(label_col, dropna=False):
            if len(group) < 50:
                continue
            hit = float(group[market].mean())
            rows.append(
                {
                    "market": market,
                    "label": label,
                    "rows": len(group),
                    "fixtures": group["fixture_id"].nunique(),
                    "hit_rate": hit,
                    "base_hit_rate": base,
                    "lift_vs_market": hit / base if base > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["market", "lift_vs_market"], ascending=[True, False])


def role_reliability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    played = df[df["prior_player_appearances"].ge(5)].copy()
    for market in MARKETS:
        for role, group in played.groupby("position", dropna=False):
            if len(group) < 250:
                continue
            if market == "keeper_saves_ge2" and role != "G":
                continue
            rows.append(
                {
                    "market": market,
                    "position": role,
                    "rows": len(group),
                    "hit_rate": float(group[market].mean()),
                    "avg_score": float(group[f"score_{market}"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["market", "hit_rate"], ascending=[True, False])


def season_persistence(df: pd.DataFrame) -> pd.DataFrame:
    played = df[df["minutes"].gt(0)].copy()
    rows = []
    agg = (
        played.groupby(["season", "player_id", "player_name", "position"], dropna=False)
        .agg(
            minutes=("minutes", "sum"),
            appearances=("fixture_id", "nunique"),
            shots_per90=("shots_total", lambda s: np.nan),
        )
        .reset_index()
    )
    sums = (
        played.groupby(["season", "player_id"], dropna=False)[
            ["shots_total", "shots_on_target", "fouls_committed", "fouls_drawn", "tackles", "card_any"]
        ]
        .sum()
        .reset_index()
    )
    mins = played.groupby(["season", "player_id"], dropna=False)["minutes"].sum().reset_index()
    season_df = agg.drop(columns=["shots_per90"]).merge(sums, on=["season", "player_id"], how="left").merge(mins, on=["season", "player_id"], how="left", suffixes=("", "_total"))
    season_df["minutes"] = season_df["minutes_total"].fillna(season_df["minutes"])
    for col in ["shots_total", "shots_on_target", "fouls_committed", "fouls_drawn", "tackles", "card_any"]:
        season_df[f"{col}_per90"] = np.where(season_df["minutes"].gt(0), season_df[col] / season_df["minutes"] * 90.0, np.nan)

    for prev_season in sorted(season_df["season"].dropna().unique()):
        next_season = int(prev_season) + 1
        prev = season_df[(season_df["season"].eq(prev_season)) & (season_df["minutes"].ge(600))]
        nxt = season_df[(season_df["season"].eq(next_season)) & (season_df["minutes"].ge(300))]
        merged = prev.merge(nxt, on="player_id", suffixes=("_prev", "_next"))
        if len(merged) < 30:
            continue
        for col in ["shots_total", "shots_on_target", "fouls_committed", "fouls_drawn", "tackles", "card_any"]:
            a = f"{col}_per90_prev"
            b = f"{col}_per90_next"
            corr = merged[[a, b]].corr().iloc[0, 1]
            prev_top = merged[a].ge(merged[a].quantile(0.75))
            next_top = merged[b].ge(merged[b].quantile(0.75))
            rows.append(
                {
                    "metric": f"{col}_per90",
                    "prev_season": int(prev_season),
                    "next_season": next_season,
                    "returning_players": len(merged),
                    "pearson_corr": float(corr),
                    "prev_top_quartile_next_top_quartile_rate": float(next_top[prev_top].mean()) if prev_top.any() else float("nan"),
                    "all_next_top_quartile_rate": float(next_top.mean()),
                }
            )
    return pd.DataFrame(rows)


def interaction_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["prior_player_appearances"].ge(5)].copy()
    rows = []

    def add(name: str, market: str, mask: pd.Series, available: str = "AVAILABLE_PROXY") -> None:
        scoped = work[mask.fillna(False)].copy()
        base_scope = work.copy()
        if market == "keeper_saves_ge2":
            base_scope = base_scope[base_scope["is_keeper"].eq(1)]
        if len(scoped) < 100 or len(base_scope) < 100:
            return
        base = float(base_scope[market].mean())
        hit = float(scoped[market].mean())
        rows.append(
            {
                "interaction": name,
                "market": market,
                "availability": available,
                "rows": len(scoped),
                "fixtures": scoped["fixture_id"].nunique(),
                "hit_rate": hit,
                "base_hit_rate": base,
                "lift": hit / base if base > 0 else float("nan"),
            }
        )

    q = lambda col, val: work[col].ge(work[col].quantile(val))
    add("high-shot player persistence", "shots_ge1", q("shots_total_per90_l10", 0.75))
    add("high-SOT player persistence", "sot_ge1", q("shots_on_target_per90_l10", 0.75))
    add("attacking mid/forward SOT profile", "sot_ge1", work["is_attacker_mid_fwd"].eq(1) & q("shots_on_target_per90_l10", 0.70))
    add("D/M high foul-committed profile", "fouls_committed_ge1", work["position"].isin(["D", "M"]) & q("fouls_committed_per90_l10", 0.70))
    add("M/F high fouls-drawn profile", "fouls_drawn_ge1", work["position"].isin(["M", "F"]) & q("fouls_drawn_per90_l10", 0.70))
    add("tackling profile predicts tackles", "tackles_ge1", q("tackles_per90_l10", 0.75))
    add("tackling profile predicts fouls committed", "fouls_committed_ge1", q("tackles_per90_l10", 0.75))
    add("tackling profile predicts cards", "card_any", q("tackles_per90_l10", 0.75))
    add("D/M vs high-possession opponent", "fouls_committed_ge1", work["position"].isin(["D", "M"]) & q("opp_possession_pct_l5", 0.70))
    add("foul-heavy player x strict referee", "card_any", q("fouls_committed_per90_l10", 0.70) & q("ref_cards_l10", 0.70))
    add("keeper x opponent shot pressure", "keeper_saves_ge2", work["is_keeper"].eq(1) & q("opp_shots_total_l5", 0.70))
    add("striker x low-block opponent proxy", "shots_ge1", work["is_forward"].eq(1) & work["opp_possession_pct_l5"].le(work["opp_possession_pct_l5"].quantile(0.35)))
    add("winger/fullback side matchup", "fouls_drawn_ge1", pd.Series(False, index=work.index), "MISSING_SIDE_COORDINATES")
    add("DM x transition opponent", "fouls_committed_ge1", work["is_midfielder"].eq(1) & q("opp_shots_total_l5", 0.70), "AVAILABLE_PROXY_NO_DM_SUBROLE")
    return pd.DataFrame(rows).sort_values(["market", "lift"], ascending=[True, False])


def recommendations(feature_scores: pd.DataFrame, label_scores: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in feature_scores.iterrows():
        market = row["market"]
        core = label_scores[(label_scores["market"].eq(market)) & (label_scores["label"].eq("SHADOW_CORE_PROXY"))]
        core_lift = float(core["lift_vs_market"].iloc[0]) if not core.empty else float("nan")
        core_hit = float(core["hit_rate"].iloc[0]) if not core.empty else float("nan")
        auc = float(row["auc"])
        if market == "card_any":
            action = "watchlist_only"
            reason = "Cards lift exists, but absolute hit rate and event volatility remain too noisy for core slips."
        elif market in {"shots_ge1", "sot_ge1"} and auc >= 0.65 and core_lift >= 1.35:
            action = "keep"
            reason = "Player persistence and role profile survive strongly across the EPL API archive."
        elif market in {"fouls_committed_ge1", "fouls_drawn_ge1", "tackles_ge1"} and core_lift >= 1.15:
            action = "keep_with_locked_wording"
            reason = "Contact shape survives, but market wording and lineup role must be locked before slips."
        elif market == "keeper_saves_ge2" and core_lift >= 1.10:
            action = "needs_lineup_confirmation"
            reason = "Keeper-save pressure survives, but only after confirmed keeper and opponent shot-pressure context."
        else:
            action = "demote"
            reason = "Historical lift is not strong enough for a confident slip leg without extra matchup context."
        rows.append(
            {
                "market": market,
                "bookmaker_wording": BOOKMAKER_WORDING[market],
                "api_settlement": SETTLEMENT_WORDING[market],
                "recommendation": action,
                "auc": auc,
                "shadow_core_proxy_hit_rate": core_hit,
                "shadow_core_proxy_lift": core_lift,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    outdir: Path,
    df: pd.DataFrame,
    feature_scores: pd.DataFrame,
    label_scores: pd.DataFrame,
    interactions: pd.DataFrame,
    persistence: pd.DataFrame,
    role: pd.DataFrame,
    recs: pd.DataFrame,
) -> None:
    lines = [
        "# API-Football EPL Player-Event System Validation Pack",
        "",
        "- Scope: research only. No live thresholds, deploy routes, or Sunday board logic were changed.",
        "- Purpose: test whether feature shapes survive historically before any future recalibration is considered.",
        "- Seasons: EPL API-Football `2022 -> 2025` local archive.",
        "- Method: timestamp-safe lagged player/team/referee features only; current-match outcomes are used only for scoring.",
        "",
        "## Market Shape Score",
        "",
    ]
    for _, row in feature_scores.sort_values("auc", ascending=False).iterrows():
        lines.append(
            f"- `{row['market']}` | AUC `{row['auc']:.3f}` | AP `{row['average_precision']:.3f}` | "
            f"base `{row['base_hit_rate']:.1%}` | top10 `{row['top10_hit_rate']:.1%}` | lift `{row['top10_lift']:.2f}x`"
        )

    lines.extend(["", "## Recommendations", ""])
    for _, row in recs.iterrows():
        lines.append(
            f"- `{row['market']}`: `{row['recommendation']}`. {row['reason']} "
            f"Book wording: {row['bookmaker_wording']}. Settlement: {row['api_settlement']}."
        )

    lines.extend(["", "## Interaction Reads", ""])
    for _, row in interactions.sort_values("lift", ascending=False).head(12).iterrows():
        lines.append(
            f"- `{row['interaction']}` -> `{row['market']}` | {row['availability']} | rows `{int(row['rows'])}` | "
            f"hit `{row['hit_rate']:.1%}` vs base `{row['base_hit_rate']:.1%}` | lift `{row['lift']:.2f}x`"
        )

    missing = interactions[interactions["availability"].astype(str).str.startswith("MISSING")]
    if not missing.empty:
        lines.extend(["", "## Missing Shape Inputs", ""])
        for _, row in missing.iterrows():
            lines.append(f"- `{row['interaction']}`: `{row['availability']}`.")

    lines.extend(["", "## Language Lock", ""])
    lines.extend(
        [
            "- `fouls_committed_ge1`: player commits the foul. Commentary pattern: `Foul by PLAYER`. API settlement: `fouls_committed`.",
            "- `fouls_drawn_ge1`: player is fouled / wins the free kick. Commentary pattern: `PLAYER wins a free kick`. API settlement: `fouls_drawn`.",
            "- Do not export shorthand `1+ foul`; every Sunday leg needs bookmaker wording and API settlement wording.",
            "",
            "## Output Files",
            "",
            f"- `{outdir / 'api_epl_player_event_validation_feature_scores.csv'}`",
            f"- `{outdir / 'api_epl_player_event_validation_system_style_labels.csv'}`",
            f"- `{outdir / 'api_epl_player_event_validation_interactions.csv'}`",
            f"- `{outdir / 'api_epl_player_event_validation_player_season_persistence.csv'}`",
            f"- `{outdir / 'api_epl_player_event_validation_role_reliability.csv'}`",
            f"- `{outdir / 'api_epl_player_event_validation_recommendations.csv'}`",
        ]
    )
    (outdir / "API_FOOTBALL_EPL_PLAYER_EVENT_SYSTEM_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(normalized_dir: Path, league_tag: str, seasons: list[int], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    players, fixtures, team_stats = load_seasons(normalized_dir, league_tag, seasons)
    df = build_dataset(players, fixtures, team_stats)
    df = add_signal_scores(df)
    df = assign_system_style_labels(df)

    feature_scores = market_feature_scores(df)
    labels = label_scorecard(df)
    interactions = interaction_scorecard(df)
    persistence = season_persistence(df)
    role = role_reliability(df)
    recs = recommendations(feature_scores, labels, interactions)

    export_cols = [
        "fixture_id",
        "season",
        "match_date",
        "home_team_name",
        "away_team_name",
        "team_name",
        "opponent_team_name",
        "player_id",
        "player_name",
        "position",
        "minutes",
        "started_flag",
        "prior_player_appearances",
    ]
    for market in MARKETS:
        export_cols.extend([market, f"score_{market}", f"{market}_system_style_label"])
    df[[c for c in export_cols if c in df.columns]].to_csv(outdir / "api_epl_player_event_validation_rows.csv", index=False)
    feature_scores.to_csv(outdir / "api_epl_player_event_validation_feature_scores.csv", index=False)
    labels.to_csv(outdir / "api_epl_player_event_validation_system_style_labels.csv", index=False)
    interactions.to_csv(outdir / "api_epl_player_event_validation_interactions.csv", index=False)
    persistence.to_csv(outdir / "api_epl_player_event_validation_player_season_persistence.csv", index=False)
    role.to_csv(outdir / "api_epl_player_event_validation_role_reliability.csv", index=False)
    recs.to_csv(outdir / "api_epl_player_event_validation_recommendations.csv", index=False)
    write_report(outdir, df, feature_scores, labels, interactions, persistence, role, recs)
    print(f"[ok] rows={len(df)} fixtures={df['fixture_id'].nunique()} seasons={','.join(map(str, seasons))}")
    print(f"[ok] wrote {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-only API-Football EPL player-event validation pack.")
    parser.add_argument("--normalized-dir", default="data_sources/api_football/normalized")
    parser.add_argument("--league-tag", default="England_Premier_League")
    parser.add_argument("--seasons", default="2022,2023,2024,2025")
    parser.add_argument("--outdir", default="reports/latest/api_football_epl_player_event_system_validation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        normalized_dir=Path(args.normalized_dir),
        league_tag=args.league_tag,
        seasons=[int(part.strip()) for part in args.seasons.split(",") if part.strip()],
        outdir=Path(args.outdir),
    )
