from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = REPO_ROOT / "data_sources" / "api_football" / "normalized"
COMBINED_GREENLIST_STATS = NORMALIZED_DIR / "match_player_stats__GREENLIST_FULL_3Y__2022_2024.csv"
COMBINED_GREENLIST_FIXTURES = NORMALIZED_DIR / "fixtures_master__GREENLIST_FULL_3Y__2022_2024.csv"
EXCLUDED_TAG_PREFIXES = ("GREENLIST_BATCH", "EUROPA_LALIGA_SERIEA", "GREENLIST_FULL_3Y")

CONFIDENCE_MAP = {
    "LOW": 0.35,
    "MEDIUM": 0.55,
    "HIGH": 0.75,
}

WINDOW_DAYS = 1095
MIN_EXACT_ROWS = 3
HIT_THRESHOLD = 0.55
ACTUAL_THRESHOLDS = {
    "fouls_committed": 2.0,
    "tackles": 2.0,
    "shots": 2.0,
    "shots_on_target": 1.0,
    "yellow_cards": 1.0,
}


def _load_score_cut_overrides(path: str | None) -> dict[tuple[str, str, str], float]:
    if not path:
        return {}
    override_path = Path(path)
    if not override_path.exists():
        return {}
    df = pd.read_csv(override_path, low_memory=False)
    if df.empty:
        return {}
    out: dict[tuple[str, str, str], float] = {}
    for _, row in df.iterrows():
        key = (
            str(row.get("market", "")),
            str(row.get("review_family", "")),
            str(row.get("prematch_risk_focus", "")),
        )
        out[key] = float(pd.to_numeric(row.get("applied_score_cut_shift"), errors="coerce") or 0.0)
    return out


def _load_hit_threshold_overrides(path: str | None) -> dict[tuple[str, str, str, str], float]:
    if not path:
        return {}
    override_path = Path(path)
    if not override_path.exists():
        return {}
    df = pd.read_csv(override_path, low_memory=False)
    if df.empty:
        return {}
    out: dict[tuple[str, str, str, str], float] = {}
    for _, row in df.iterrows():
        key = (
            str(row.get("market", "")),
            str(row.get("review_family", "")),
            str(row.get("prematch_risk_focus", "")),
            str(row.get("lookback_source", "")),
        )
        out[key] = float(pd.to_numeric(row.get("applied_hit_threshold"), errors="coerce") or HIT_THRESHOLD)
    return out


def _load_market_history_gate(path: str | None) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    gate_path = Path(path)
    if not gate_path.exists():
        return {}
    df = pd.read_csv(gate_path, low_memory=False)
    if df.empty:
        return {}
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        market = str(row.get("market", ""))
        out[market] = {
            "min_prior_hits": float(pd.to_numeric(row.get("min_prior_hits"), errors="coerce") or 0.0),
            "min_prior_apps": float(pd.to_numeric(row.get("min_prior_apps"), errors="coerce") or 0.0),
            "min_prior_hit_rate": float(pd.to_numeric(row.get("min_prior_hit_rate"), errors="coerce") or 0.0),
            "min_hits_l3": float(pd.to_numeric(row.get("min_hits_l3"), errors="coerce") or 0.0),
            "min_hits_l5": float(pd.to_numeric(row.get("min_hits_l5"), errors="coerce") or 0.0),
            "min_hits_l8": float(pd.to_numeric(row.get("min_hits_l8"), errors="coerce") or 0.0),
            "min_hits_l10": float(pd.to_numeric(row.get("min_hits_l10"), errors="coerce") or 0.0),
            "min_hit_rate_l3": float(pd.to_numeric(row.get("min_hit_rate_l3"), errors="coerce") or 0.0),
            "min_hit_rate_l5": float(pd.to_numeric(row.get("min_hit_rate_l5"), errors="coerce") or 0.0),
            "min_hit_rate_l8": float(pd.to_numeric(row.get("min_hit_rate_l8"), errors="coerce") or 0.0),
            "min_hit_rate_l10": float(pd.to_numeric(row.get("min_hit_rate_l10"), errors="coerce") or 0.0),
        }
    return out


def _market_stat_col(market: str) -> str | None:
    return {
        "shots": "shots_total",
        "shots_on_target": "shots_on_target",
        "tackles": "tackles",
        "fouls_committed": "fouls_committed",
        "yellow_cards": "yellow_cards",
    }.get(str(market))


def _recent_market_history(player_history: pd.DataFrame, stat_col: str, threshold: float) -> dict[str, float]:
    vals = pd.to_numeric(player_history.get(stat_col), errors="coerce")
    vals = vals[vals.notna()]
    out: dict[str, float] = {}
    for n in (3, 5, 8, 10):
        recent = vals.head(n)
        apps = int(recent.notna().sum())
        hits = int((recent >= threshold).sum()) if apps else 0
        hit_rate = float(hits / apps) if apps else 0.0
        out[f"apps_l{n}"] = apps
        out[f"hits_l{n}"] = hits
        out[f"hit_rate_l{n}"] = hit_rate
    return out


def _risk_focus_for_role(role: str) -> str:
    role_text = str(role or "")
    if role_text == "Holding midfielder":
        return "missing DM"
    if role_text == "Wide defender / wing-back":
        return "missing full-back"
    if role_text == "Centre-back enforcer":
        return "missing CB duel anchor"
    return "no core structural flag"


def _normalize_board(df: pd.DataFrame, board_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["board_name"] = board_name
    if "market" not in out.columns:
        out["market"] = "yellow_cards"
    out["review_family"] = out.get("review_family", out.get("source_family", "UNSET"))
    out["fixture_family"] = out.get("formation_matchup_label", "UNSET")
    out["score"] = pd.to_numeric(out.get("score", out.get("market_score", 0.0)), errors="coerce").fillna(0.0)
    out["match_date"] = pd.to_datetime(out.get("match_date"), errors="coerce")
    out["prematch_risk_focus"] = out.get("tactical_role", pd.Series("UNSET", index=out.index)).astype(str).map(_risk_focus_for_role)
    if "market_hit_rate" in out.columns:
        out["observed_success_score"] = pd.to_numeric(out["market_hit_rate"], errors="coerce").fillna(0.0)
        out["observed_label_mode"] = "REALIZED_GROUP"
    else:
        out["observed_success_score"] = (
            out.get("market_confidence", pd.Series("", index=out.index))
            .astype(str)
            .str.upper()
            .map(CONFIDENCE_MAP)
            .fillna(0.0)
        )
        out["observed_label_mode"] = "CONFIDENCE_PROXY"
    out["observed_success_flag"] = out["observed_success_score"].ge(HIT_THRESHOLD).astype(int)
    keep_cols = [
        "fixture_key",
        "match_date",
        "competition",
        "league",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "market",
        "tactical_role",
        "review_family",
        "fixture_family",
        "score",
        "fixture_quality_score",
        "formation_pressure_score",
        "player_quality_score_l5",
        "starting_xi_quality_edge",
        "board_name",
        "prematch_risk_focus",
        "observed_success_score",
        "observed_success_flag",
        "observed_label_mode",
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = 0.0 if col in {"score", "fixture_quality_score", "formation_pressure_score", "player_quality_score_l5", "starting_xi_quality_edge", "observed_success_score"} else "UNSET"
    optional_cols = [c for c in ["source_family", "priority_bucket", "sample_bucket"] if c in out.columns]
    return out[keep_cols + optional_cols].copy()


def _load_settled_player_actuals() -> pd.DataFrame:
    keep_cols = [
        "fixture_key",
        "match_date",
        "player_name",
        "shots_total",
        "shots_on_target",
        "tackles",
        "fouls_committed",
        "yellow_cards",
    ]
    if COMBINED_GREENLIST_STATS.exists() and COMBINED_GREENLIST_FIXTURES.exists():
        stats_df = pd.read_csv(COMBINED_GREENLIST_STATS, low_memory=False)
        fixtures_df = pd.read_csv(COMBINED_GREENLIST_FIXTURES, low_memory=False)
        merged = stats_df.copy()
        if "fixture_key" not in merged.columns or merged["fixture_key"].isna().any():
            fixture_lookup = fixtures_df[["fixture_id", "fixture_key"]].drop_duplicates()
            merged = merged.merge(fixture_lookup, on="fixture_id", how="left", suffixes=("", "_fixture"))
            if "fixture_key_fixture" in merged.columns:
                if "fixture_key" in merged.columns:
                    merged["fixture_key"] = merged["fixture_key"].fillna(merged["fixture_key_fixture"])
                else:
                    merged["fixture_key"] = merged["fixture_key_fixture"]
                merged = merged.drop(columns=["fixture_key_fixture"])
        merged["player_name"] = merged["player_name"].astype(str).str.strip()
        keep = [col for col in keep_cols if col in merged.columns]
        merged = merged[keep].copy()
        return merged.dropna(subset=["fixture_key", "player_name"]).drop_duplicates(subset=["fixture_key", "player_name"])

    stats_frames: list[pd.DataFrame] = []
    fixture_frames: list[pd.DataFrame] = []
    for stats_path in NORMALIZED_DIR.glob("match_player_stats__*__20*.csv"):
        name = stats_path.name.replace("match_player_stats__", "")
        league_tag = name.rsplit("__", 1)[0]
        if league_tag.startswith(EXCLUDED_TAG_PREFIXES):
            continue
        fixture_path = NORMALIZED_DIR / stats_path.name.replace("match_player_stats__", "fixtures_master__")
        if not fixture_path.exists():
            continue
        stats = pd.read_csv(
            stats_path,
            low_memory=False,
            usecols=[
                "fixture_id",
                "player_name",
                "shots_total",
                "shots_on_target",
                "tackles",
                "fouls_committed",
                "yellow_cards",
            ],
        )
        fixtures = pd.read_csv(fixture_path, low_memory=False, usecols=["fixture_id", "fixture_key"])
        stats_frames.append(stats)
        fixture_frames.append(fixtures)
    if not stats_frames or not fixture_frames:
        return pd.DataFrame()
    stats_df = pd.concat(stats_frames, ignore_index=True)
    fixtures_df = pd.concat(fixture_frames, ignore_index=True).drop_duplicates(subset=["fixture_id", "fixture_key"])
    merged = stats_df.merge(fixtures_df, on="fixture_id", how="left")
    merged["player_name"] = merged["player_name"].astype(str).str.strip()
    merged = merged[keep_cols].copy()
    return merged.dropna(subset=["fixture_key", "player_name"]).drop_duplicates(subset=["fixture_key", "player_name"])


def _apply_settled_actuals(df: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["settled_actual_available"] = 0
    out["actual_value"] = pd.NA
    if actuals.empty:
        return out

    joined = out.merge(actuals, on=["fixture_key", "player_name"], how="left")
    for market, stat_col in [
        ("fouls_committed", "fouls_committed"),
        ("tackles", "tackles"),
        ("shots", "shots_total"),
        ("shots_on_target", "shots_on_target"),
        ("yellow_cards", "yellow_cards"),
    ]:
        mask = joined["market"].astype(str).eq(market)
        joined.loc[mask, "actual_value"] = pd.to_numeric(joined.loc[mask, stat_col], errors="coerce")

    joined["settled_actual_available"] = joined["actual_value"].notna().astype(int)
    for market, threshold in ACTUAL_THRESHOLDS.items():
        mask = joined["market"].astype(str).eq(market) & joined["actual_value"].notna()
        hit = (pd.to_numeric(joined.loc[mask, "actual_value"], errors="coerce") >= threshold).astype(int)
        joined.loc[mask, "observed_success_score"] = hit.astype(float)
        joined.loc[mask, "observed_success_flag"] = hit
        joined.loc[mask, "observed_label_mode"] = "SETTLED_PLAYER_STAT"
    return joined


def _select_window(prior: pd.DataFrame, row: pd.Series) -> tuple[pd.DataFrame, str]:
    exact = prior[
        (prior["market"] == row["market"])
        & (prior["review_family"] == row["review_family"])
        & (prior["tactical_role"] == row["tactical_role"])
        & (prior["fixture_family"] == row["fixture_family"])
    ]
    if len(exact) >= MIN_EXACT_ROWS:
        return exact, "EXACT"

    family_role = prior[
        (prior["market"] == row["market"])
        & (prior["review_family"] == row["review_family"])
        & (prior["tactical_role"] == row["tactical_role"])
    ]
    if len(family_role) >= MIN_EXACT_ROWS:
        return family_role, "FAMILY_ROLE"

    role_market = prior[
        (prior["market"] == row["market"])
        & (prior["tactical_role"] == row["tactical_role"])
    ]
    if len(role_market) >= MIN_EXACT_ROWS:
        return role_market, "ROLE_MARKET"

    market_only = prior[prior["market"] == row["market"]]
    if len(market_only) >= MIN_EXACT_ROWS:
        return market_only, "MARKET_ONLY"

    return prior, "GLOBAL"


def build_runner(
    master_csv: str,
    bookings_csv: str,
    team_weekend_csv: str,
    output_csv: str,
    output_md: str,
    overrides_csv: str | None = None,
    hit_threshold_overrides_csv: str | None = None,
    market_history_gate_csv: str | None = None,
) -> pd.DataFrame:
    frames = [
        _normalize_board(pd.read_csv(master_csv, low_memory=False), "MASTER_SPECIALIST"),
        _normalize_board(pd.read_csv(bookings_csv, low_memory=False), "BOOKINGS_SUPER_ELITE"),
        _normalize_board(pd.read_csv(team_weekend_csv, low_memory=False), "TEAM_SPECIFIC_WEEKEND"),
    ]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if combined.empty:
        combined.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Player Market Rolling 3Y Walkforward\n\nNo rows matched.\n")
        return combined

    actuals = _load_settled_player_actuals()
    settle_actuals = actuals.drop(columns=["match_date"], errors="ignore") if not actuals.empty else actuals
    combined = _apply_settled_actuals(combined, settle_actuals)
    combined = combined.dropna(subset=["match_date"]).sort_values(["match_date", "fixture_key", "player_name"]).reset_index(drop=True)
    score_cut_overrides = _load_score_cut_overrides(overrides_csv)
    hit_threshold_overrides = _load_hit_threshold_overrides(hit_threshold_overrides_csv)
    market_history_gate = _load_market_history_gate(market_history_gate_csv)
    if not actuals.empty and "match_date" in actuals.columns:
        actuals["match_date"] = pd.to_datetime(actuals["match_date"], errors="coerce")
        actuals["player_name"] = actuals["player_name"].astype(str).str.strip()

    results: list[dict] = []
    for idx, row in combined.iterrows():
        prior = combined.iloc[:idx].copy()
        window_start = row["match_date"] - pd.Timedelta(days=WINDOW_DAYS)
        prior = prior[prior["match_date"].between(window_start, row["match_date"] - pd.Timedelta(days=1), inclusive="both")]
        cohort, source = _select_window(prior, row)
        lookback_rows = len(cohort)
        lookback_fixtures = cohort["fixture_key"].nunique() if lookback_rows else 0
        expected_hit = float(pd.to_numeric(cohort["observed_success_score"], errors="coerce").mean()) if lookback_rows else 0.0
        expected_score = float(pd.to_numeric(cohort["score"], errors="coerce").mean()) if lookback_rows else 0.0
        score_delta = round(float(row["score"]) - expected_score, 4)
        applied_hit_threshold = hit_threshold_overrides.get(
            (str(row["market"]), str(row["review_family"]), str(row["prematch_risk_focus"]), str(source)),
            HIT_THRESHOLD,
        )
        applied_shift = score_cut_overrides.get(
            (str(row["market"]), str(row["review_family"]), str(row["prematch_risk_focus"])),
            0.0,
        )
        prior_market_apps = 0
        prior_market_hits = 0
        prior_market_hit_rate = 0.0
        recent_market = {
            "apps_l3": 0,
            "hits_l3": 0,
            "hit_rate_l3": 0.0,
            "apps_l5": 0,
            "hits_l5": 0,
            "hit_rate_l5": 0.0,
            "apps_l8": 0,
            "hits_l8": 0,
            "hit_rate_l8": 0.0,
            "apps_l10": 0,
            "hits_l10": 0,
            "hit_rate_l10": 0.0,
        }
        history_gate_blocked = 0
        if not actuals.empty and "match_date" in actuals.columns:
            player_history = actuals[
                (actuals["player_name"].astype(str) == str(row["player_name"]))
                & (actuals["match_date"] < row["match_date"])
                & (actuals["match_date"] >= row["match_date"] - pd.Timedelta(days=WINDOW_DAYS))
            ].sort_values("match_date", ascending=False).copy()
            stat_col = _market_stat_col(str(row["market"]))
            threshold = ACTUAL_THRESHOLDS.get(str(row["market"]))
            if stat_col and threshold is not None and stat_col in player_history.columns:
                vals = pd.to_numeric(player_history[stat_col], errors="coerce")
                prior_market_apps = int(vals.notna().sum())
                prior_market_hits = int((vals >= threshold).sum())
                prior_market_hit_rate = float(prior_market_hits / prior_market_apps) if prior_market_apps else 0.0
                recent_market = _recent_market_history(player_history, stat_col, threshold)
        gate_cfg = market_history_gate.get(str(row["market"]), {})
        if gate_cfg:
            if (
                prior_market_hits < gate_cfg.get("min_prior_hits", 0.0)
                or prior_market_apps < gate_cfg.get("min_prior_apps", 0.0)
                or prior_market_hit_rate < gate_cfg.get("min_prior_hit_rate", 0.0)
                or recent_market["hits_l3"] < gate_cfg.get("min_hits_l3", 0.0)
                or recent_market["hits_l5"] < gate_cfg.get("min_hits_l5", 0.0)
                or recent_market["hits_l8"] < gate_cfg.get("min_hits_l8", 0.0)
                or recent_market["hits_l10"] < gate_cfg.get("min_hits_l10", 0.0)
                or recent_market["hit_rate_l3"] < gate_cfg.get("min_hit_rate_l3", 0.0)
                or recent_market["hit_rate_l5"] < gate_cfg.get("min_hit_rate_l5", 0.0)
                or recent_market["hit_rate_l8"] < gate_cfg.get("min_hit_rate_l8", 0.0)
                or recent_market["hit_rate_l10"] < gate_cfg.get("min_hit_rate_l10", 0.0)
            ):
                history_gate_blocked = 1
        selection_gate = int(
            lookback_rows >= MIN_EXACT_ROWS
            and expected_hit >= applied_hit_threshold
            and score_delta >= applied_shift
            and history_gate_blocked == 0
        )
        near_miss_flag = int(selection_gate == 1 and int(row["observed_success_flag"]) == 0)
        missed_correct_flag = int(selection_gate == 0 and int(row["observed_success_flag"]) == 1)
        results.append(
            {
                **row.to_dict(),
                "walkforward_window_start": window_start.date().isoformat(),
                "lookback_source": source,
                "lookback_rows": lookback_rows,
                "lookback_fixtures": lookback_fixtures,
                "expected_hit_rate_3y": round(expected_hit, 4),
                "expected_score_3y": round(expected_score, 4),
                "score_delta_vs_3y": score_delta,
                "prior_market_apps_3y": prior_market_apps,
                "prior_market_hits_3y": prior_market_hits,
                "prior_market_hit_rate_3y": round(float(prior_market_hit_rate), 4),
                "prior_market_apps_l3": int(recent_market["apps_l3"]),
                "prior_market_hits_l3": int(recent_market["hits_l3"]),
                "prior_market_hit_rate_l3": round(float(recent_market["hit_rate_l3"]), 4),
                "prior_market_apps_l5": int(recent_market["apps_l5"]),
                "prior_market_hits_l5": int(recent_market["hits_l5"]),
                "prior_market_hit_rate_l5": round(float(recent_market["hit_rate_l5"]), 4),
                "prior_market_apps_l8": int(recent_market["apps_l8"]),
                "prior_market_hits_l8": int(recent_market["hits_l8"]),
                "prior_market_hit_rate_l8": round(float(recent_market["hit_rate_l8"]), 4),
                "prior_market_apps_l10": int(recent_market["apps_l10"]),
                "prior_market_hits_l10": int(recent_market["hits_l10"]),
                "prior_market_hit_rate_l10": round(float(recent_market["hit_rate_l10"]), 4),
                "history_gate_blocked_flag": history_gate_blocked,
                "applied_hit_threshold": round(float(applied_hit_threshold), 4),
                "applied_score_cut_shift": round(float(applied_shift), 4),
                "settled_actual_available": int(row.get("settled_actual_available", 0) or 0),
                "actual_value": row.get("actual_value", pd.NA),
                "selection_gate_flag": selection_gate,
                "near_miss_flag": near_miss_flag,
                "missed_correct_flag": missed_correct_flag,
            }
        )

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Player Market Rolling 3Y Walkforward",
        "",
        "- Rolling 1095-day lookback by market/family/role, with hierarchical fallback when exact cohorts are still sparse.",
        "- Evidence mode is upgraded: `SETTLED_PLAYER_STAT` rows use actual player match stats wherever normalized sources support them; the remainder still fall back to `REALIZED_GROUP` or `CONFIDENCE_PROXY`.",
        f"- score-cut override file: `{overrides_csv or 'NONE'}`",
        f"- hit-threshold override file: `{hit_threshold_overrides_csv or 'NONE'}`",
        f"- market-history gate file: `{market_history_gate_csv or 'NONE'}`",
        f"- rows={len(out)} | fixtures={out['fixture_key'].nunique()} | markets={out['market'].astype(str).nunique()}",
        "",
    ]
    summary = (
        out.groupby(["market", "review_family"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", pd.Series.nunique),
            avg_expected_hit=("expected_hit_rate_3y", "mean"),
            observed_hit=("observed_success_flag", "mean"),
            near_misses=("near_miss_flag", "sum"),
            missed_correct=("missed_correct_flag", "sum"),
            settled_rows=("settled_actual_available", "sum"),
        )
        .reset_index()
        .sort_values(["avg_expected_hit", "observed_hit", "rows"], ascending=[False, False, False])
    )
    lines.append("## Market / Family Summary")
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['market']} | {r['review_family']} | rows={int(r['rows'])} | fixtures={int(r['fixtures'])} | settled_rows={int(r['settled_rows'])} | expected_hit_3y={r['avg_expected_hit']:.3f} | observed_hit={r['observed_hit']:.3f} | near_misses={int(r['near_misses'])} | missed_correct={int(r['missed_correct'])}"
        )
    lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a rolling 3-year walkforward runner across the broader player-market stack.")
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--bookings-csv", required=True)
    parser.add_argument("--team-weekend-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--overrides-csv", default="")
    parser.add_argument("--hit-threshold-overrides-csv", default="")
    parser.add_argument("--market-history-gate-csv", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_runner(
        args.master_csv,
        args.bookings_csv,
        args.team_weekend_csv,
        args.output_csv,
        args.output_md,
        args.overrides_csv,
        args.hit_threshold_overrides_csv,
        args.market_history_gate_csv,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
