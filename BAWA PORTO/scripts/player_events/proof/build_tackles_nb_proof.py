from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

from common import REPORTS_DIR, TARGET_LEAGUES, TARGET_SEASONS, load_fixture_inputs, normalized_path
from build_minutes_offset import build as build_minutes_offset
from build_opponent_allowed_features import build as build_opponent_allowed
from leak_audit import build as build_leak_audit, run_audit

WINDOW_DAYS = 1095
EVAL_SEASONS = {2023, 2024}
BURN_IN_DAYS = 90
MIN_EXACT_ROWS = 8
SHRINK_K = 10.0
TOP_DECILE = 0.10
TARGET_LINE = 2
MODEL_ALPHA = 0.05
MODEL_MAX_ITER = 2000
RARE_CATEGORY_MIN = 25
CORR_DROP_THRESHOLD = 0.995

CAT_COLS = [
    "league_tag",
    "position_group",
    "tactical_role",
    "formation_matchup_label",
    "fixture_style_label",
    "fixture_attacking_style_label",
    "player_team_side",
]
NUM_COLS = [
    "player_tackles_per90_l5_shrunk",
    "player_tackles_per90_l10_shrunk",
    "player_tackles_per90_season_shrunk",
    "interceptions_per90",
    "duels_total_per90",
    "formation_pressure_score",
    "fixture_midfield_grind_score",
    "fixture_wide_duel_score",
    "ref_dissent_strictness",
    "ref_timewasting_strictness",
    "og_balance_score",
    "og_battle_on_flag",
    "match_stakes_score",
    "starting_xi_quality_edge",
    "opp_tackles_allowed_def_l10",
    "opp_tackles_allowed_mid_l10",
    "opp_tackles_allowed_pos_l10",
    "opp_possession_share_l10",
    "opp_dribble_attempts_l10",
    "player_form_rating_l5",
    "player_quality_score_l5",
    "expected_minutes_proof",
]


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _norm_position(pos: str) -> str:
    text = str(pos or "").strip().upper()[:1]
    return {"D": "Defender", "M": "Midfielder", "F": "Forward", "G": "Goalkeeper"}.get(text, "Unknown")


def _load_actual_lookup(league: str, season: int) -> pd.DataFrame:
    fixtures = pd.read_csv(normalized_path("fixtures_master", league, season), low_memory=False)
    stats = pd.read_csv(normalized_path("match_player_stats", league, season), low_memory=False)
    fixtures = fixtures[["fixture_id", "fixture_key", "home_team_id", "away_team_id", "home_team_name", "away_team_name", "match_date", "kickoff_ts_utc"]]
    merged = stats.merge(fixtures, on="fixture_id", how="left")
    merged["team_name"] = np.where(merged["team_id"].eq(merged["home_team_id"]), merged["home_team_name"], merged["away_team_name"])
    merged["position_group"] = merged["position"].map(_norm_position)
    cols = [
        "fixture_id",
        "fixture_key",
        "match_date",
        "kickoff_ts_utc",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "position",
        "position_group",
        "minutes",
        "started_flag",
        "subbed_on_flag",
        "tackles",
        "interceptions",
        "duels_total",
    ]
    out = merged[cols].copy()
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["kickoff_ts_utc"] = pd.to_datetime(out["kickoff_ts_utc"], errors="coerce", utc=True)
    out["league_tag"] = league
    out["season_tag"] = season
    return out.sort_values(["kickoff_ts_utc", "fixture_id", "team_id", "player_id"]).reset_index(drop=True)


def _build_dataset() -> pd.DataFrame:
    fixture_inputs = load_fixture_inputs(TARGET_LEAGUES, TARGET_SEASONS, force=False)
    if fixture_inputs.empty:
        return fixture_inputs

    actual_frames = [_load_actual_lookup(league, season) for league in TARGET_LEAGUES for season in TARGET_SEASONS]
    actuals = pd.concat(actual_frames, ignore_index=True)

    opp_csv = REPORTS_DIR / "opponent_allowed_features.csv"
    opp_md = REPORTS_DIR / "opponent_allowed_features.md"
    mins_csv = REPORTS_DIR / "minutes_offset.csv"
    mins_md = REPORTS_DIR / "minutes_offset.md"
    if not opp_csv.exists():
        build_opponent_allowed(opp_csv, opp_md)
    if not mins_csv.exists():
        build_minutes_offset(mins_csv, mins_md)
    opp = pd.read_csv(opp_csv, low_memory=False)
    mins = pd.read_csv(mins_csv, low_memory=False)
    if "match_date" in opp.columns:
        opp["match_date"] = pd.to_datetime(opp["match_date"], errors="coerce")
    if "match_date" in mins.columns:
        mins["match_date"] = pd.to_datetime(mins["match_date"], errors="coerce")

    work = fixture_inputs.copy()
    work["position_group"] = work["position_group"].astype(str)
    work = work[work["position_group"].isin(["Defender", "Midfielder"])].copy()
    work["match_date"] = pd.to_datetime(work["match_date"], errors="coerce")

    merged = work.merge(
        actuals,
        on=["fixture_key", "team_name", "player_name", "league_tag", "season_tag"],
        how="left",
        suffixes=("", "_actual"),
    )
    merged["match_date"] = pd.to_datetime(merged["match_date"], errors="coerce")
    merged["team_id"] = pd.to_numeric(merged["team_id"], errors="coerce")
    merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce")
    merged["actual_tackles"] = pd.to_numeric(merged["tackles"], errors="coerce")
    merged["actual_minutes"] = pd.to_numeric(merged["minutes"], errors="coerce")
    merged["actual_started_flag"] = pd.to_numeric(merged["started_flag"], errors="coerce").fillna(0).astype(int)
    merged["actual_hit_ge2"] = merged["actual_tackles"].fillna(0).ge(TARGET_LINE).astype(int)

    merged = merged.merge(opp, on=["fixture_id", "team_id", "fixture_key", "match_date", "league_tag", "season_tag"], how="left")
    merged = merged.merge(
        mins[["fixture_id", "team_id", "player_id", "expected_start_prob", "expected_minutes_if_start", "expected_minutes_if_sub", "expected_minutes_proof", "minutes_history_apps_l5", "minutes_source_max_date", "league_tag", "season_tag"]],
        on=["fixture_id", "team_id", "player_id", "league_tag", "season_tag"],
        how="left",
    )

    merged["opp_tackles_allowed_pos_l10"] = np.where(
        merged["position_group"].eq("Defender"),
        pd.to_numeric(merged["opp_tackles_allowed_def_l10"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["opp_tackles_allowed_mid_l10"], errors="coerce").fillna(0.0),
    )

    # rolling player tackle history for shrinkage and leak-auditable source dates
    hist_rows = []
    for (league, season), grp in actuals.groupby(["league_tag", "season_tag"], dropna=False):
        grp = grp.sort_values(["kickoff_ts_utc", "fixture_id", "team_id", "player_id"]).reset_index(drop=True)
        history: dict[int, list[dict]] = {}
        for _, row in grp.iterrows():
            player_id = int(row["player_id"])
            prev = list(reversed(history.get(player_id, [])))
            l5 = prev[:5]
            l10 = prev[:10]
            season_prev = [r for r in prev if int(r.get("season_tag", season)) == int(season)]
            mins_l5 = sum(float(r.get("minutes", 0.0) or 0.0) for r in l5)
            mins_l10 = sum(float(r.get("minutes", 0.0) or 0.0) for r in l10)
            mins_season = sum(float(r.get("minutes", 0.0) or 0.0) for r in season_prev)
            tackles_l5 = sum(float(r.get("tackles", 0.0) or 0.0) for r in l5)
            tackles_l10 = sum(float(r.get("tackles", 0.0) or 0.0) for r in l10)
            tackles_season = sum(float(r.get("tackles", 0.0) or 0.0) for r in season_prev)
            hist_rows.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "team_id": int(row["team_id"]),
                    "player_id": player_id,
                    "league_tag": league,
                    "season_tag": season,
                    "player_tackle_apps_l5": len(l5),
                    "player_tackle_apps_l10": len(l10),
                    "player_tackles_per90_l5": _safe_div(tackles_l5 * 90.0, mins_l5),
                    "player_tackles_per90_l10": _safe_div(tackles_l10 * 90.0, mins_l10),
                    "player_tackles_per90_season": _safe_div(tackles_season * 90.0, mins_season),
                    "player_history_source_max_date": max([str(r.get("match_date", "")) for r in l10], default=""),
                }
            )
            history.setdefault(player_id, []).append(row.to_dict())
    hist_df = pd.DataFrame(hist_rows)
    merged = merged.merge(hist_df, on=["fixture_id", "team_id", "player_id", "league_tag", "season_tag"], how="left")

    # shrink toward position-group prior using same-season pooled priors
    priors = (
        merged.groupby("position_group", dropna=False)
        .agg(
            pos_tackle_rate_l5=("player_tackles_per90_l5", lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).mean()),
            pos_tackle_rate_l10=("player_tackles_per90_l10", lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).mean()),
            pos_tackle_rate_season=("player_tackles_per90_season", lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).mean()),
        )
        .reset_index()
    )
    merged = merged.merge(priors, on="position_group", how="left")
    apps_l5 = pd.to_numeric(merged["player_tackle_apps_l5"], errors="coerce").fillna(0.0)
    apps_l10 = pd.to_numeric(merged["player_tackle_apps_l10"], errors="coerce").fillna(0.0)
    merged["player_tackles_per90_l5_shrunk"] = (apps_l5 * pd.to_numeric(merged["player_tackles_per90_l5"], errors="coerce").fillna(0.0) + SHRINK_K * pd.to_numeric(merged["pos_tackle_rate_l5"], errors="coerce").fillna(0.0)) / (apps_l5 + SHRINK_K)
    merged["player_tackles_per90_l10_shrunk"] = (apps_l10 * pd.to_numeric(merged["player_tackles_per90_l10"], errors="coerce").fillna(0.0) + SHRINK_K * pd.to_numeric(merged["pos_tackle_rate_l10"], errors="coerce").fillna(0.0)) / (apps_l10 + SHRINK_K)
    merged["player_tackles_per90_season_shrunk"] = (apps_l10 * pd.to_numeric(merged["player_tackles_per90_season"], errors="coerce").fillna(0.0) + SHRINK_K * pd.to_numeric(merged["pos_tackle_rate_season"], errors="coerce").fillna(0.0)) / (apps_l10 + SHRINK_K)

    merged = merged.dropna(subset=["fixture_id", "team_id", "player_id", "match_date", "actual_tackles", "expected_minutes_proof"]).copy()
    merged = merged[pd.to_numeric(merged["expected_minutes_proof"], errors="coerce").fillna(0.0).ge(30.0)].copy()
    merged = merged.sort_values(["match_date", "fixture_key", "team_name", "player_name"]).reset_index(drop=True)
    return merged


def _cohort_probability(train: pd.DataFrame, row: pd.Series) -> tuple[float | None, str, int]:
    exact = train[(train["formation_matchup_label"] == row["formation_matchup_label"]) & (train["tactical_role"] == row["tactical_role"])]
    if len(exact) >= MIN_EXACT_ROWS:
        return float(exact["actual_hit_ge2"].mean()), "EXACT", len(exact)
    family = train[(train["position_group"] == row["position_group"]) & (train["tactical_role"] == row["tactical_role"])]
    if len(family) >= 20:
        return float(family["actual_hit_ge2"].mean()), "FAMILY_ROLE", len(family)
    role = train[train["tactical_role"] == row["tactical_role"]]
    if len(role) >= 40:
        return float(role["actual_hit_ge2"].mean()), "ROLE_MARKET", len(role)
    market = train
    if len(market) >= 80:
        return float(market["actual_hit_ge2"].mean()), "MARKET_ONLY", len(market)
    return None, "NO_PREDICTION", 0




def _fit_statsmodels_nb(X_train: pd.DataFrame, y_train: pd.Series, exposure_train: pd.Series):
    exog = sm.add_constant(X_train, has_constant="add").astype(float)
    exposure_arr = np.asarray(exposure_train, dtype=float)
    poisson_seed = sm.GLM(
        y_train.astype(float),
        exog,
        family=sm.families.Poisson(),
        exposure=exposure_arr,
    ).fit(maxiter=200, disp=0)
    mu_seed = np.clip(poisson_seed.predict(exog, exposure=exposure_arr), 1e-6, None)
    alpha = _estimate_alpha(y_train.to_numpy(dtype=float), mu_seed)
    family = sm.families.NegativeBinomial(alpha=alpha)
    try:
        nb_glm = sm.GLM(
            y_train.astype(float),
            exog,
            family=family,
            exposure=exposure_arr,
        ).fit(maxiter=200, disp=0)
    except Exception:
        nb_glm = sm.GLM(
            y_train.astype(float),
            exog,
            family=family,
            exposure=exposure_arr,
        ).fit_regularized(
            method="elastic_net",
            alpha=1e-4,
            L1_wt=0.0,
            maxiter=500,
        )
    return nb_glm, max(alpha, 1e-6)


def _predict_statsmodels_nb(model, X: pd.DataFrame, exposure: pd.Series | np.ndarray) -> np.ndarray:
    exog = sm.add_constant(X, has_constant="add").astype(float)
    exog_arr = np.asarray(exog, dtype=float)
    exposure_arr = np.asarray(exposure, dtype=float)
    pred = model.predict(exog_arr, exposure=exposure_arr)
    return np.clip(np.asarray(pred, dtype=float), 1e-6, None)

def _estimate_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu.astype(float), 1e-6, None)
    numer = np.maximum(((y - mu) ** 2 - mu), 0.0).sum()
    denom = np.maximum((mu ** 2).sum(), 1e-6)
    return float(max(numer / denom, 1e-6))


def _prob_ge_k(mu: np.ndarray, alpha: float, k: int) -> np.ndarray:
    r = 1.0 / max(alpha, 1e-6)
    p = r / (r + np.clip(mu, 1e-6, None))
    return 1.0 - nbinom.cdf(k - 1, r, p)


def _bucket_rare_categories(train: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series]:
    train = train.astype(str).fillna("UNSET")
    test = test.astype(str).fillna("UNSET")
    counts = train.value_counts(dropna=False)
    keep = set(counts[counts >= RARE_CATEGORY_MIN].index.astype(str))
    if not keep:
        return (
            pd.Series(["OTHER"] * len(train), index=train.index),
            pd.Series(["OTHER"] * len(test), index=test.index),
        )
    return train.where(train.isin(keep), "OTHER"), test.where(test.isin(keep), "OTHER")


def _nb_failure_diagnostics(X_train: pd.DataFrame, y_train: pd.Series, exposure_train: pd.Series, exc: Exception) -> dict[str, object]:
    arr = X_train.to_numpy(dtype=float, copy=False)
    rank = int(np.linalg.matrix_rank(arr)) if arr.size else 0
    zero_var_cols = [col for col in X_train.columns if float(X_train[col].var()) <= 1e-12]
    corr_pairs = 0
    max_abs_corr = np.nan
    if X_train.shape[1] > 1:
        corr = X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        max_abs_corr = float(upper.max().max()) if not upper.empty else np.nan
        corr_pairs = int((upper > CORR_DROP_THRESHOLD).sum().sum())
    return {
        "status": "nb_fit_failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:400],
        "train_rows": int(len(X_train)),
        "train_cols": int(X_train.shape[1]),
        "matrix_rank": rank,
        "rank_deficit": int(X_train.shape[1] - rank),
        "zero_var_cols": len(zero_var_cols),
        "high_corr_pairs": corr_pairs,
        "max_abs_corr": None if pd.isna(max_abs_corr) else round(float(max_abs_corr), 6),
        "y_mean": round(float(pd.to_numeric(y_train, errors="coerce").fillna(0.0).mean()), 6),
        "y_var": round(float(pd.to_numeric(y_train, errors="coerce").fillna(0.0).var()), 6),
        "exposure_min": round(float(pd.to_numeric(exposure_train, errors="coerce").fillna(0.0).min()), 6),
        "exposure_mean": round(float(pd.to_numeric(exposure_train, errors="coerce").fillna(0.0).mean()), 6),
    }


def _prepare_matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train[NUM_COLS + CAT_COLS].copy()
    X_test = test[NUM_COLS + CAT_COLS].copy()
    for col in NUM_COLS:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0.0)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(0.0)
        lo = float(X_train[col].quantile(0.01))
        hi = float(X_train[col].quantile(0.99))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            X_train[col] = X_train[col].clip(lower=lo, upper=hi)
            X_test[col] = X_test[col].clip(lower=lo, upper=hi)
        mean = float(X_train[col].mean())
        std = float(X_train[col].std())
        if np.isfinite(std) and std > 1e-9:
            X_train[col] = (X_train[col] - mean) / std
            X_test[col] = (X_test[col] - mean) / std
        else:
            X_train[col] = X_train[col] - mean
            X_test[col] = X_test[col] - mean
    for col in CAT_COLS:
        X_train[col], X_test[col] = _bucket_rare_categories(X_train[col], X_test[col])
    X_train = pd.get_dummies(X_train, columns=CAT_COLS, dummy_na=False)
    X_test = pd.get_dummies(X_test, columns=CAT_COLS, dummy_na=False)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    zero_var = [col for col in X_train.columns if float(X_train[col].var()) <= 1e-12]
    if zero_var:
        X_train = X_train.drop(columns=zero_var)
        X_test = X_test.drop(columns=zero_var, errors="ignore")
    if X_train.shape[1] > 1:
        corr = X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if (upper[col] > CORR_DROP_THRESHOLD).any()]
        if drop_cols:
            X_train = X_train.drop(columns=drop_cols)
            X_test = X_test.drop(columns=drop_cols, errors="ignore")
    return X_train, X_test


def _metric_block(df: pd.DataFrame, prefix: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {"slice": prefix, "rows": int(len(df))}
    if df.empty:
        out.update({"coverage": 0.0, "mae_count": np.nan, "brier_ge2": np.nan, "logloss_ge2": np.nan, "ece_10bin": np.nan, "top_decile_precision": np.nan})
        return out
    probs = np.clip(pd.to_numeric(df[f"{prefix}_p_ge2"], errors="coerce").fillna(np.nan), 1e-6, 1 - 1e-6)
    hit = pd.to_numeric(df["actual_hit_ge2"], errors="coerce").fillna(0).astype(int)
    valid = probs.notna()
    out["coverage"] = round(float(valid.mean()), 6)
    if valid.sum() == 0:
        out.update({"mae_count": np.nan, "brier_ge2": np.nan, "logloss_ge2": np.nan, "ece_10bin": np.nan, "top_decile_precision": np.nan})
        return out
    if f"{prefix}_lambda" in df.columns:
        lam = pd.to_numeric(df.loc[valid, f"{prefix}_lambda"], errors="coerce").fillna(0.0)
        out["mae_count"] = round(float(mean_absolute_error(pd.to_numeric(df.loc[valid, "actual_tackles"], errors="coerce").fillna(0.0), lam)), 6)
    else:
        out["mae_count"] = np.nan
    out["brier_ge2"] = round(float(brier_score_loss(hit.loc[valid], probs.loc[valid])), 6)
    out["logloss_ge2"] = round(float(log_loss(hit.loc[valid], probs.loc[valid], labels=[0, 1])), 6)

    bins = pd.cut(probs.loc[valid], bins=np.linspace(0, 1, 11), include_lowest=True, duplicates="drop")
    calib = pd.DataFrame({"prob": probs.loc[valid], "hit": hit.loc[valid], "bin": bins}).dropna(subset=["bin"])
    if calib.empty:
        out["ece_10bin"] = np.nan
    else:
        grouped = calib.groupby("bin", observed=False).agg(pred=("prob", "mean"), obs=("hit", "mean"), n=("hit", "size")).reset_index()
        out["ece_10bin"] = round(float(((grouped["n"] / grouped["n"].sum()) * (grouped["pred"] - grouped["obs"]).abs()).sum()), 6)

    top_n = max(1, int(np.ceil(valid.sum() * TOP_DECILE)))
    top = df.loc[valid].sort_values(f"{prefix}_p_ge2", ascending=False).head(top_n)
    out["top_decile_precision"] = round(float(pd.to_numeric(top["actual_hit_ge2"], errors="coerce").fillna(0).mean()), 6)
    return out


def _write_reliability(df: pd.DataFrame, prob_col: str, output_csv: Path, output_svg: Path, output_png: Path | None = None) -> None:
    probs = pd.to_numeric(df[prob_col], errors="coerce")
    hit = pd.to_numeric(df["actual_hit_ge2"], errors="coerce").fillna(0).astype(int)
    valid = probs.notna()
    bins = pd.cut(probs.loc[valid], bins=np.linspace(0, 1, 11), include_lowest=True, duplicates="drop")
    calib = pd.DataFrame({"prob": probs.loc[valid], "hit": hit.loc[valid], "bin": bins}).dropna(subset=["bin"])
    if calib.empty:
        pd.DataFrame().to_csv(output_csv, index=False)
        output_svg.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='420'></svg>")
        if output_png is not None:
            plt.figure(figsize=(6.4, 4.2))
            plt.savefig(output_png, dpi=150, bbox_inches="tight")
            plt.close()
        return
    grouped = calib.groupby("bin", observed=False).agg(pred=("prob", "mean"), obs=("hit", "mean"), n=("hit", "size")).reset_index()
    grouped["se"] = np.sqrt((grouped["obs"] * (1 - grouped["obs"])) / grouped["n"].clip(lower=1))
    grouped["lo"] = (grouped["obs"] - 1.96 * grouped["se"]).clip(lower=0)
    grouped["hi"] = (grouped["obs"] + 1.96 * grouped["se"]).clip(upper=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)

    W, H, PAD = 640, 420, 50
    def xy(x: float, y: float) -> tuple[float, float]:
        return PAD + x * (W - 2 * PAD), H - PAD - y * (H - 2 * PAD)

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}'>"]
    parts.append("<rect width='100%' height='100%' fill='white'/>")
    x1, y1 = xy(0, 0)
    x2, y2 = xy(1, 1)
    parts.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#cccccc' stroke-width='2'/>")
    parts.append(f"<rect x='{PAD}' y='{PAD}' width='{W-2*PAD}' height='{H-2*PAD}' fill='none' stroke='black' stroke-width='1'/>")
    for _, row in grouped.iterrows():
        px, py = xy(float(row['pred']), float(row['obs']))
        _, ylo = xy(float(row['pred']), float(row['lo']))
        _, yhi = xy(float(row['pred']), float(row['hi']))
        parts.append(f"<line x1='{px}' y1='{ylo}' x2='{px}' y2='{yhi}' stroke='#1f77b4' stroke-width='2'/>")
        parts.append(f"<circle cx='{px}' cy='{py}' r='5' fill='#1f77b4'/>")
    parts.append("<text x='320' y='24' text-anchor='middle' font-size='16'>Tackles Proof Reliability</text>")
    parts.append("<text x='320' y='405' text-anchor='middle' font-size='12'>Predicted P(tackles ≥ 2)</text>")
    parts.append("<text x='18' y='210' transform='rotate(-90 18 210)' text-anchor='middle' font-size='12'>Observed Hit Rate</text>")
    parts.append("</svg>")
    output_svg.write_text("".join(parts))
    if output_png is not None:
        plt.figure(figsize=(6.4, 4.2))
        plt.plot([0, 1], [0, 1], color="#cccccc")
        plt.errorbar(grouped["pred"], grouped["obs"], yerr=[grouped["obs"] - grouped["lo"], grouped["hi"] - grouped["obs"]], fmt='o', color="#1f77b4")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Predicted P(tackles >= 2)")
        plt.ylabel("Observed Hit Rate")
        plt.title("Tackles Proof Reliability")
        plt.tight_layout()
        plt.savefig(output_png, dpi=150)
        plt.close()




def _build_subgroup_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["minutes_bucket"] = pd.cut(pd.to_numeric(work["expected_minutes_proof"], errors="coerce").fillna(0.0), bins=[-np.inf, 60, 80, np.inf], labels=["30_60", "60_80", "80_plus"]).astype(str)
    rows = []
    for keys, grp in work.groupby(["league_tag", "position_group", "minutes_bucket", "season_tag"], dropna=False):
        row = _metric_block(grp, "nb")
        rows.append({
            "league_tag": keys[0],
            "position_group": keys[1],
            "minutes_bucket": keys[2],
            "season_tag": keys[3],
            "rows": row["rows"],
            "coverage": row["coverage"],
            "mae_count": row["mae_count"],
            "brier_ge2": row["brier_ge2"],
            "logloss_ge2": row["logloss_ge2"],
            "ece_10bin": row["ece_10bin"],
            "top_decile_precision": row["top_decile_precision"],
        })
    return pd.DataFrame(rows).sort_values(["league_tag", "position_group", "minutes_bucket", "season_tag"])


def _build_top_decile_composition(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    valid = df[pd.to_numeric(df["nb_p_ge2"], errors="coerce").notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    top_n = max(1, int(np.ceil(len(valid) * TOP_DECILE)))
    top = valid.sort_values("nb_p_ge2", ascending=False).head(top_n).copy()
    top["minutes_bucket"] = pd.cut(pd.to_numeric(top["expected_minutes_proof"], errors="coerce").fillna(0.0), bins=[-np.inf, 60, 80, np.inf], labels=["30_60", "60_80", "80_plus"]).astype(str)
    rows = []
    rows.append({"dimension": "overall", "value": "ALL", "rows": len(top), "distinct_players": top["player_name"].nunique(), "distinct_teams": top["team_name"].nunique(), "hit_rate": round(float(pd.to_numeric(top["actual_hit_ge2"], errors="coerce").fillna(0).mean()), 6)})
    for dim in ["league_tag", "position_group", "minutes_bucket", "season_tag", "team_name"]:
        grp = top.groupby(dim, dropna=False).agg(rows=("fixture_key", "size"), distinct_players=("player_name", pd.Series.nunique), distinct_teams=("team_name", pd.Series.nunique), hit_rate=("actual_hit_ge2", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).mean())).reset_index()
        for _, r in grp.iterrows():
            rows.append({"dimension": dim, "value": r[dim], "rows": int(r["rows"]), "distinct_players": int(r["distinct_players"]), "distinct_teams": int(r["distinct_teams"]), "hit_rate": round(float(r["hit_rate"]), 6)})
    return pd.DataFrame(rows)

def build_proof(outdir: Path, max_test_dates: int | None = None, use_statsmodels: bool = True) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = _build_dataset()
    dataset_csv = outdir / "tackles_nb_proof_dataset.csv"
    dataset.to_csv(dataset_csv, index=False)

    leak_csv = outdir / "leak_audit_checks.csv"
    leak_md = outdir / "leak_audit_report.md"
    build_leak_audit(dataset_csv, leak_csv, leak_md)

    if dataset.empty:
        raise RuntimeError("Proof dataset is empty.")

    eval_rows = dataset[dataset["season_tag"].isin(EVAL_SEASONS)].copy()
    eval_start = eval_rows["match_date"].min() + pd.Timedelta(days=BURN_IN_DAYS)
    eval_rows = eval_rows[eval_rows["match_date"] >= eval_start].copy()
    test_dates = sorted(eval_rows["match_date"].dropna().unique())
    if max_test_dates is not None:
        test_dates = test_dates[:max_test_dates]

    preds: list[pd.DataFrame] = []
    fit_diags: list[dict[str, object]] = []
    for dt in test_dates:
        test = eval_rows[eval_rows["match_date"].eq(dt)].copy()
        train = dataset[(dataset["match_date"] < dt) & (dataset["match_date"] >= dt - pd.Timedelta(days=WINDOW_DAYS))].copy()
        if train.empty or test.empty:
            continue
        X_train, X_test = _prepare_matrix(train, test)
        exposure_train = (pd.to_numeric(train["expected_minutes_proof"], errors="coerce").fillna(0.0) / 90.0).clip(lower=0.1)
        exposure_test = (pd.to_numeric(test["expected_minutes_proof"], errors="coerce").fillna(0.0) / 90.0).clip(lower=0.1)
        y_train = pd.to_numeric(train["actual_tackles"], errors="coerce").fillna(0.0)
        rate_train = y_train / exposure_train

        candidate_rows = []

        if use_statsmodels:
            try:
                nb_result, alpha_nb = _fit_statsmodels_nb(X_train, y_train, exposure_train)
                train_mu_nb = _predict_statsmodels_nb(nb_result, X_train, exposure_train)
                p_nb = _prob_ge_k(train_mu_nb, alpha_nb, TARGET_LINE)
                brier_nb = brier_score_loss(train["actual_hit_ge2"], np.clip(p_nb, 1e-6, 1 - 1e-6))
                candidate_rows.append(("STATSMODELS_NB", brier_nb, alpha_nb, nb_result))
                fit_diags.append({
                    "match_date": str(pd.Timestamp(dt).date()),
                    "status": "nb_fit_ok",
                    "train_rows": int(len(X_train)),
                    "train_cols": int(X_train.shape[1]),
                    "alpha_nb": round(float(alpha_nb), 6),
                    "train_brier": round(float(brier_nb), 6),
                })
            except Exception as exc:
                diag = _nb_failure_diagnostics(X_train, y_train, exposure_train, exc)
                diag["match_date"] = str(pd.Timestamp(dt).date())
                fit_diags.append(diag)

        poisson = PoissonRegressor(alpha=MODEL_ALPHA, max_iter=MODEL_MAX_ITER)
        poisson.fit(X_train, rate_train, sample_weight=exposure_train)
        train_mu_poisson = np.clip(poisson.predict(X_train) * exposure_train, 1e-6, None)
        alpha_poisson = _estimate_alpha(y_train.to_numpy(dtype=float), train_mu_poisson)
        p_poisson = _prob_ge_k(train_mu_poisson, alpha_poisson, TARGET_LINE)
        brier_poisson = brier_score_loss(train["actual_hit_ge2"], np.clip(p_poisson, 1e-6, 1 - 1e-6))
        candidate_rows.append(("POISSON_BASELINE", brier_poisson, alpha_poisson, poisson))

        lgbm = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            learning_rate=0.05,
            n_estimators=150,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=0.10,
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_train, rate_train, sample_weight=exposure_train)
        train_mu_lgbm = np.clip(lgbm.predict(X_train) * exposure_train, 1e-6, None)
        alpha_lgbm = _estimate_alpha(y_train.to_numpy(dtype=float), train_mu_lgbm)
        p_lgbm = _prob_ge_k(train_mu_lgbm, alpha_lgbm, TARGET_LINE)
        brier_lgbm = brier_score_loss(train["actual_hit_ge2"], np.clip(p_lgbm, 1e-6, 1 - 1e-6))
        candidate_rows.append(("LIGHTGBM_TWEEDIE_CHALLENGER", brier_lgbm, alpha_lgbm, lgbm))

        chosen_family = "STATSMODELS_NB" if any(r[0] == "STATSMODELS_NB" for r in candidate_rows) else sorted(candidate_rows, key=lambda x: x[1])[0][0]
        if chosen_family == "STATSMODELS_NB":
            chosen_model = [r[3] for r in candidate_rows if r[0] == "STATSMODELS_NB"][0]
            alpha = [r[2] for r in candidate_rows if r[0] == "STATSMODELS_NB"][0]
            test_mu = _predict_statsmodels_nb(chosen_model, X_test, exposure_test)
        elif chosen_family == "LIGHTGBM_TWEEDIE_CHALLENGER":
            chosen_model = [r[3] for r in candidate_rows if r[0] == "LIGHTGBM_TWEEDIE_CHALLENGER"][0]
            alpha = [r[2] for r in candidate_rows if r[0] == "LIGHTGBM_TWEEDIE_CHALLENGER"][0]
            test_mu = np.clip(chosen_model.predict(X_test) * exposure_test, 1e-6, None)
        else:
            chosen_model = [r[3] for r in candidate_rows if r[0] == chosen_family][0]
            alpha = [r[2] for r in candidate_rows if r[0] == chosen_family][0]
            test_mu = np.clip(chosen_model.predict(X_test) * exposure_test, 1e-6, None)

        challenger_family = "LIGHTGBM_TWEEDIE_CHALLENGER"
        challenger_model = [r[3] for r in candidate_rows if r[0] == challenger_family][0]
        challenger_alpha = [r[2] for r in candidate_rows if r[0] == challenger_family][0]
        challenger_mu = np.clip(challenger_model.predict(X_test) * exposure_test, 1e-6, None)
        challenger_p = _prob_ge_k(challenger_mu, challenger_alpha, TARGET_LINE)

        test_p = _prob_ge_k(test_mu, alpha, TARGET_LINE)

        cohort_probs = []
        cohort_sources = []
        cohort_rows = []
        for _, row in test.iterrows():
            p, src, n = _cohort_probability(train, row)
            cohort_probs.append(p)
            cohort_sources.append(src)
            cohort_rows.append(n)

        block = test.copy()
        block["nb_lambda"] = test_mu
        block["nb_p_ge2"] = test_p
        block["nb_alpha"] = alpha
        block["nb_model_family"] = chosen_family
        block["challenger_lambda"] = challenger_mu
        block["challenger_p_ge2"] = challenger_p
        block["challenger_alpha"] = challenger_alpha
        block["challenger_model_family"] = challenger_family
        block["cohort_p_ge2"] = cohort_probs
        block["cohort_source"] = cohort_sources
        block["cohort_rows"] = cohort_rows
        preds.append(block)

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    pred_csv = outdir / "tackles_nb_proof_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    reliability_csv = outdir / "tackles_nb_proof_reliability.csv"
    reliability_svg = outdir / "tackles_nb_proof_reliability.svg"
    reliability_png = outdir / "tackles_nb_proof_reliability.png"
    if not pred_df.empty:
        _write_reliability(pred_df, "nb_p_ge2", reliability_csv, reliability_svg, reliability_png)

    metric_rows = []
    if not pred_df.empty:
        metric_rows.append(_metric_block(pred_df, "nb"))
        metric_rows.append(_metric_block(pred_df, "cohort"))
        shared = pred_df[pd.to_numeric(pred_df["cohort_p_ge2"], errors="coerce").notna()].copy()
        metric_rows.append(_metric_block(shared, "nb"))
        metric_rows[-1]["slice"] = "nb_shared_rows"
        metric_rows.append(_metric_block(shared, "cohort"))
        metric_rows[-1]["slice"] = "cohort_shared_rows"
        for group_name in ["league_tag", "position_group", "season_tag", "player_team_side"]:
            for value, grp in pred_df.groupby(group_name, dropna=False):
                row = _metric_block(grp, "nb")
                row["slice"] = f"nb_{group_name}_{value}"
                metric_rows.append(row)
                row2 = _metric_block(grp, "cohort")
                row2["slice"] = f"cohort_{group_name}_{value}"
                metric_rows.append(row2)
        mins_bucket = pd.cut(pd.to_numeric(pred_df["expected_minutes_proof"], errors="coerce").fillna(0.0), bins=[-np.inf, 60, 80, np.inf], labels=["30_60", "60_80", "80_plus"])
        pred_df["minutes_bucket"] = mins_bucket.astype(str)
        for value, grp in pred_df.groupby("minutes_bucket", dropna=False):
            row = _metric_block(grp, "nb")
            row["slice"] = f"nb_minutes_{value}"
            metric_rows.append(row)
            row2 = _metric_block(grp, "cohort")
            row2["slice"] = f"cohort_minutes_{value}"
            metric_rows.append(row2)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_csv = outdir / "tackles_nb_proof_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    subgroup_csv = outdir / "tackles_nb_proof_subgroups.csv"
    subgroup_df = _build_subgroup_table(pred_df)
    subgroup_df.to_csv(subgroup_csv, index=False)

    top_decile_csv = outdir / "tackles_nb_top_decile_composition.csv"
    top_decile_df = _build_top_decile_composition(pred_df)
    top_decile_df.to_csv(top_decile_csv, index=False)

    fit_diag_csv = outdir / "tackles_nb_primary_fit_diagnostics.csv"
    fit_diag_md = outdir / "tackles_nb_primary_fit_diagnostics.md"
    fit_diag_df = pd.DataFrame(fit_diags)
    fit_diag_df.to_csv(fit_diag_csv, index=False)
    fit_lines = [
        "# Tackles NB Primary Fit Diagnostics",
        "",
        f"- total walkforward dates reviewed: `{len(test_dates)}`",
        f"- statsmodels enabled: `{use_statsmodels}`",
        f"- successful NB folds: `{int((fit_diag_df.get('status') == 'nb_fit_ok').sum()) if not fit_diag_df.empty and 'status' in fit_diag_df.columns else 0}`",
        f"- failed NB folds: `{int((fit_diag_df.get('status') == 'nb_fit_failed').sum()) if not fit_diag_df.empty and 'status' in fit_diag_df.columns else 0}`",
    ]
    if not fit_diag_df.empty and "status" in fit_diag_df.columns and (fit_diag_df["status"] == "nb_fit_failed").any():
        fail = fit_diag_df[fit_diag_df["status"] == "nb_fit_failed"].copy()
        fit_lines.extend([
            "",
            "## Failure Snapshot",
            f"- median rank deficit: `{round(float(pd.to_numeric(fail['rank_deficit'], errors='coerce').fillna(0).median()), 2)}`",
            f"- median zero-var cols: `{round(float(pd.to_numeric(fail['zero_var_cols'], errors='coerce').fillna(0).median()), 2)}`",
            f"- median high-corr pairs: `{round(float(pd.to_numeric(fail['high_corr_pairs'], errors='coerce').fillna(0).median()), 2)}`",
            f"- most common error type: `{fail['error_type'].mode().iloc[0] if 'error_type' in fail.columns and not fail['error_type'].dropna().empty else 'UNKNOWN'}`",
        ])
    fit_diag_md.write_text("\n".join(fit_lines) + "\n")

    vs = pd.DataFrame()
    decision = {}
    if not metrics_df.empty:
        overall_nb = metrics_df[metrics_df["slice"].eq("nb")].iloc[0].to_dict()
        overall_cohort = metrics_df[metrics_df["slice"].eq("cohort")].iloc[0].to_dict()
        vs = pd.DataFrame([
            {"metric": "coverage", "nb": overall_nb.get("coverage"), "cohort": overall_cohort.get("coverage")},
            {"metric": "mae_count", "nb": overall_nb.get("mae_count"), "cohort": overall_cohort.get("mae_count")},
            {"metric": "brier_ge2", "nb": overall_nb.get("brier_ge2"), "cohort": overall_cohort.get("brier_ge2")},
            {"metric": "logloss_ge2", "nb": overall_nb.get("logloss_ge2"), "cohort": overall_cohort.get("logloss_ge2")},
            {"metric": "ece_10bin", "nb": overall_nb.get("ece_10bin"), "cohort": overall_cohort.get("ece_10bin")},
            {"metric": "top_decile_precision", "nb": overall_nb.get("top_decile_precision"), "cohort": overall_cohort.get("top_decile_precision")},
        ])
        mae_win = pd.notna(overall_cohort.get("mae_count")) and pd.notna(overall_nb.get("mae_count")) and float(overall_nb.get("mae_count")) <= float(overall_cohort.get("mae_count")) * 0.95
        brier_win = pd.notna(overall_cohort.get("brier_ge2")) and pd.notna(overall_nb.get("brier_ge2")) and float(overall_nb.get("brier_ge2")) <= float(overall_cohort.get("brier_ge2")) * 0.97
        logloss_win = pd.notna(overall_cohort.get("logloss_ge2")) and pd.notna(overall_nb.get("logloss_ge2")) and float(overall_nb.get("logloss_ge2")) <= float(overall_cohort.get("logloss_ge2")) * 0.97
        ece_win = pd.notna(overall_nb.get("ece_10bin")) and float(overall_nb.get("ece_10bin")) <= 0.03
        top_decile_win = pd.notna(overall_cohort.get("top_decile_precision")) and pd.notna(overall_nb.get("top_decile_precision")) and float(overall_nb.get("top_decile_precision")) >= float(overall_cohort.get("top_decile_precision")) + 0.04
        coverage_win = pd.notna(overall_nb.get("coverage")) and float(overall_nb.get("coverage")) >= 0.90
        win_flags = {
            "mae_win": mae_win,
            "brier_win": brier_win,
            "logloss_win": logloss_win,
            "ece_win": ece_win,
            "top_decile_win": top_decile_win,
            "coverage_win": coverage_win,
        }
        decision["overall_call"] = "NB_PROOF_WINS" if all(win_flags.values()) else "NB_PROOF_NOT_YET_WINNING"
        decision.update(win_flags)
    vs_csv = outdir / "tackles_nb_proof_vs_cohort.csv"
    vs.to_csv(vs_csv, index=False)

    decision_md = outdir / "PHASE1_PROOF_DECISION.md"
    leak_report, leak_summary = run_audit(dataset)
    lines = [
        "# Phase 1 Proof Decision",
        "",
        "- Market: `tackles`",
        "- Leagues: `England_Premier_League`, `Spain_La_Liga`",
        "- Architecture in this environment: `count-rate proof stack with exposure-weighted sklearn baseline plus empirical NB dispersion`.",
        f"- proof mode: `{'FULL_FAST_STACK' if not use_statsmodels else 'FULL_WITH_STATSMODELS_CANDIDATE'}`",
        f"- walkforward rows scored: `{len(pred_df)}`",
        f"- walkforward dates scored: `{pred_df['match_date'].nunique() if not pred_df.empty else 0}`",
        f"- leak audit status: `{leak_summary['overall_status']}`",
        "",
    ]
    if decision:
        lines.append(f"- overall_call: `{decision['overall_call']}`")
        for key, value in decision.items():
            if key == "overall_call":
                continue
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- No decision could be made because predictions were empty.")
    lines.extend([
        "",
        "## Read",
        "- Treat this as the first architecture proof artifact, not the final player-events engine verdict.",
        "- Tactical features remain valuable as model inputs and subgroup audit layers; the point of this proof is whether a count-rate engine beats the cohort-gate benchmark on tackles.",
    ])
    decision_md.write_text("\n".join(lines) + "\n")

    return {
        "dataset_csv": dataset_csv,
        "predictions_csv": pred_csv,
        "metrics_csv": metrics_csv,
        "reliability_csv": reliability_csv,
        "reliability_svg": reliability_svg,
        "reliability_png": reliability_png,
        "vs_cohort_csv": vs_csv,
        "subgroup_csv": subgroup_csv,
        "top_decile_csv": top_decile_csv,
        "fit_diag_csv": fit_diag_csv,
        "fit_diag_md": fit_diag_md,
        "leak_audit_md": leak_md,
        "decision_md": decision_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the tackles Phase 1 probability-engine proof pack.")
    parser.add_argument("--outdir", default=str(REPORTS_DIR))
    parser.add_argument("--max-test-dates", type=int, default=0, help="Optional cap for quicker sanity runs.")
    parser.add_argument("--disable-statsmodels", action="store_true", help="Skip statsmodels NB candidate and use the faster sklearn proof stack only.")
    args = parser.parse_args()
    out = build_proof(Path(args.outdir), max_test_dates=(args.max_test_dates or None), use_statsmodels=not args.disable_statsmodels)
    for key, path in out.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
