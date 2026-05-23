#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
MASTER = ROOT / "predictions_output" / "walk_forward" / "_MASTER" / "BTTS_WINNER_EXCLUSION_AUDIT"
WINNER_AUDIT = MASTER / "BTTS_YES_WINNERS__ALL_WITH_EXCLUSIONS.csv"
OUTDIR = MASTER / "FULL_UNIVERSE_SCENARIO_VALIDATION"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# League family controls
# ----------------------------
WEAK_LEAGUE_FAMILY = {
    "Spain La Liga",
    "France Ligue 1",
    "USA MLS",
    "Germany Bundesliga 2",
}

FTS_OVERRIDE_LEAGUES = {
    "Netherlands Eredivisie",
    "Europa Conference",
    "Japan J1",
    "Belgium Pro",
    "England FA Cup",
}

# ----------------------------
# Helpers
# ----------------------------
def norm_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def norm_upper(s: pd.Series) -> pd.Series:
    return norm_str(s).str.upper()


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_col(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def make_key(df: pd.DataFrame) -> pd.Series:
    parts = []
    for c in ["league", "fixture_key", "market", "bookie_pick", "__window"]:
        if c in df.columns:
            parts.append(norm_str(df[c]))
        else:
            parts.append(pd.Series("", index=df.index, dtype="string"))
    return parts[0] + "|" + parts[1] + "|" + parts[2] + "|" + parts[3] + "|" + parts[4]


def classify_signal_bucket(sig: pd.Series) -> pd.Series:
    su = norm_upper(sig)
    out = pd.Series("OTHER", index=su.index, dtype="string")
    out.loc[su.eq("VERY_STRONG_YES")] = "VERY_STRONG_YES"
    out.loc[su.eq("STRONG_YES")] = "STRONG_YES"
    out.loc[su.eq("WEAK_YES")] = "WEAK_YES"
    out.loc[su.eq("NEUTRAL")] = "NEUTRAL"
    return out


def scenario_metrics(df: pd.DataFrame, scenario_name: str) -> dict:
    rows = int(len(df))
    graded = int(df["btts_yes_hit"].notna().sum()) if "btts_yes_hit" in df.columns else 0
    wins = int(to_num(df["btts_yes_hit"]).fillna(0).sum()) if "btts_yes_hit" in df.columns else 0
    losses = int(graded - wins)

    odds = to_num(safe_col(df, "bookie_od"))
    hit_raw = to_num(safe_col(df, "btts_yes_hit"))
    hit = hit_raw.fillna(0)

    profit = ((odds - 1.0) * hit) - (1.0 * (1.0 - hit))
    profit = profit.where(hit_raw.notna(), np.nan)

    level_stake_profit = float(profit.fillna(0).sum()) if rows else 0.0
    roi = float(level_stake_profit / graded) if graded > 0 else np.nan
    hit_rate = float(wins / graded) if graded > 0 else np.nan

    return {
        "scenario": scenario_name,
        "rows": rows,
        "graded": graded,
        "wins": wins,
        "losses": losses,
        "hit_rate": hit_rate,
        "roi_level_stake": roi,
        "level_stake_profit": level_stake_profit,
        "avg_bookie_od": float(odds.mean()) if rows else np.nan,
        "avg_model_p_for_bookie": float(to_num(safe_col(df, "model_p_for_bookie")).mean()) if rows else np.nan,
    }


def grouped_metrics(df: pd.DataFrame, scenario_name: str, group_col: str) -> pd.DataFrame:
    out = []
    for g, sub in df.groupby(group_col, dropna=False):
        row = scenario_metrics(sub.copy(), scenario_name)
        row[group_col] = g
        out.append(row)
    if not out:
        return pd.DataFrame(
            columns=[
                group_col,
                "scenario",
                "rows",
                "graded",
                "wins",
                "losses",
                "hit_rate",
                "roi_level_stake",
                "level_stake_profit",
                "avg_bookie_od",
                "avg_model_p_for_bookie",
            ]
        )
    cols = [
        group_col,
        "scenario",
        "rows",
        "graded",
        "wins",
        "losses",
        "hit_rate",
        "roi_level_stake",
        "level_stake_profit",
        "avg_bookie_od",
        "avg_model_p_for_bookie",
    ]
    return pd.DataFrame(out)[cols]


def first_present(df: pd.DataFrame, candidates: list[str], default=np.nan) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(default, index=df.index)


def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"skip read fail: {path} :: {e}")
        return None


# ----------------------------
# Load winner audit map
# ----------------------------
if not WINNER_AUDIT.exists():
    raise SystemExit(f"Missing winner audit file: {WINNER_AUDIT}")

wmap = pd.read_csv(WINNER_AUDIT, low_memory=False)
if "__window" not in wmap.columns:
    raise SystemExit("Winner audit must contain __window")

wmap["__row_key"] = make_key(wmap)
winner_lookup_cols = [
    "__row_key",
    "first_exclusion",
]
winner_lookup_cols = [c for c in winner_lookup_cols if c in wmap.columns]
wmap = wmap[winner_lookup_cols].drop_duplicates("__row_key", keep="first")

# ----------------------------
# Build full universe from RAW + SCORED enrichment
# ----------------------------
window_dirs = sorted(ROOT.glob("predictions_output/walk_forward/w*"))
if not window_dirs:
    raise SystemExit("No walk_forward window directories found.")

frames: list[pd.DataFrame] = []
for wdir in window_dirs:
    source_dir = wdir / "01_source"
    raw_candidates = sorted(source_dir.glob("BOOKIE_IMP20_ALLMARKETS_*.csv"))
    raw_fp = None
    for cand in raw_candidates:
        name = cand.name
        if "__DEPLOY_" in name:
            continue
        raw_fp = cand
        break

    scored_dir = wdir / "03_scored"
    scored_candidates = sorted(scored_dir.glob("DEPLOY_COMBINED_SCORED_*.csv"))
    scored_fp = scored_candidates[0] if scored_candidates else None

    if raw_fp is None or not raw_fp.exists() or scored_fp is None or not scored_fp.exists():
        continue

    raw_df = load_csv(raw_fp)
    scored_df = load_csv(scored_fp)
    if raw_df is None or scored_df is None:
        continue

    window = wdir.name
    raw_df["__window"] = window
    raw_df["__raw_source_file"] = str(raw_fp)
    raw_df["__raw_source_type"] = "01_source/BOOKIE_IMP20_ALLMARKETS"
    scored_df["__window"] = window
    scored_df["__scored_source_file"] = str(scored_fp)
    scored_df["__scored_source_type"] = "03_scored/DEPLOY_COMBINED_SCORED"

    raw_mk = norm_str(safe_col(raw_df, "market")).str.lower()
    raw_pick = norm_upper(safe_col(raw_df, "bookie_pick"))
    raw_sel = norm_upper(safe_col(raw_df, "selection"))

    raw_sub = raw_df.loc[raw_mk.eq("btts") & (raw_pick.eq("YES") | raw_sel.eq("YES"))].copy()
    if raw_sub.empty:
        continue

    scored_mk = norm_str(safe_col(scored_df, "market")).str.lower()
    scored_pick = norm_upper(safe_col(scored_df, "bookie_pick"))
    scored_sel = norm_upper(safe_col(scored_df, "selection"))
    scored_sub = scored_df.loc[scored_mk.eq("btts") & (scored_pick.eq("YES") | scored_sel.eq("YES"))].copy()

    raw_sub["__row_key"] = make_key(raw_sub)
    scored_sub["__row_key"] = make_key(scored_sub)

    enrich_cols = [
        "__row_key",
        "__scored_source_file",
        "__scored_source_type",
        "actual_btts_yes",
        "btts_yes_hit",
        "cs_max",
        "btts_top3_mass",
        "cs00_top3_mass",
        "top3_any_00",
        "fts_sum",
        "cs_sum",
        "signal_btts_runtime",
        "signal_btts",
        "signal_btts_fixture",
        "signal_btts_side",
        "product",
        "model_lane",
        "p00_est",
        "exp_goals_sum",
        "bookie_od",
        "model_p_for_bookie",
        "home_ge2_confidence",
        "away_ge2_confidence",
        "p_home_fts",
        "p_away_fts",
        "league",
        "fixture_key",
        "market",
        "bookie_pick",
        "selection",
    ]
    enrich_cols = [c for c in enrich_cols if c in scored_sub.columns]
    enrich_df = scored_sub[enrich_cols].drop_duplicates("__row_key", keep="first")

    merged = raw_sub.merge(enrich_df, on="__row_key", how="left", suffixes=("", "__scored"))

    if "__scored_source_file__scored" in merged.columns:
        merged["__scored_source_file"] = merged["__scored_source_file__scored"]
        merged.drop(columns=["__scored_source_file__scored"], inplace=True)

    if "__scored_source_type__scored" in merged.columns:
        merged["__scored_source_type"] = merged["__scored_source_type__scored"]
        merged.drop(columns=["__scored_source_type__scored"], inplace=True)

    if "__scored_source_file" not in merged.columns:
        merged["__scored_source_file"] = str(scored_fp)

    if "__scored_source_type" not in merged.columns:
        merged["__scored_source_type"] = "03_scored/DEPLOY_COMBINED_SCORED"

    for key_col in [
        "league",
        "fixture_key",
        "market",
        "bookie_pick",
        "selection",
        "bookie_od",
        "model_p_for_bookie",
        "home_ge2_confidence",
        "away_ge2_confidence",
        "p_home_fts",
        "p_away_fts",
        "p00_est",
        "exp_goals_sum",
        "actual_btts_yes",
        "btts_yes_hit",
        "cs_max",
        "btts_top3_mass",
        "cs00_top3_mass",
        "top3_any_00",
        "fts_sum",
        "cs_sum",
        "signal_btts_runtime",
        "signal_btts",
        "signal_btts_fixture",
        "signal_btts_side",
        "product",
        "model_lane",
    ]:
        scored_col = f"{key_col}__scored"
        if scored_col in merged.columns:
            merged[key_col] = first_present(merged, [key_col, scored_col])
            merged.drop(columns=[scored_col], inplace=True)

    frames.append(merged)

if not frames:
    raise SystemExit("No BTTS YES candidate rows found from RAW universe joined to SCORED enrichment.")


all_df = pd.concat(frames, ignore_index=True, sort=False)
all_df = all_df.drop_duplicates("__row_key", keep="first").copy()

print("\nSOURCE FILE CHECK")
raw_source_col = "__raw_source_file"
scored_source_col = "__scored_source_file"

if raw_source_col in all_df.columns:
    raw_source_counts = all_df[raw_source_col].value_counts(dropna=False)
    print(f"raw_source_files_used={len(raw_source_counts)}")
else:
    print("raw_source_files_used=0 (column missing)")

if scored_source_col in all_df.columns:
    scored_source_counts = all_df[scored_source_col].value_counts(dropna=False)
    print(f"scored_source_files_used={len(scored_source_counts)}")
else:
    print("scored_source_files_used=0 (column missing)")
    scored_source_candidates = [c for c in all_df.columns if "scored_source_file" in c]
    if scored_source_candidates:
        print(f"available_scored_source_columns={scored_source_candidates}")

# ----------------------------
# Attach winner exclusion data where available
# ----------------------------
all_df = all_df.merge(wmap, on="__row_key", how="left")
# winner audit is diagnostics-only: never let it supply live-pass logic columns
winner_logic_cols = [
    "signal_btts_eval",
    "signal_btts_runtime_eval",
    "label_pass",
    "brazil_block",
    "weak_yes_block",
    "neutral_or_other_block",
    "ge2_pass",
    "model_floor_pass",
    "fts_pass",
    "cs_pass",
    "double_blank_pass",
    "confirmation_pass",
    "final_live_pass",
]
for col in winner_logic_cols:
    if col in all_df.columns:
        all_df.drop(columns=[col], inplace=True)

# ----------------------------
# Core normalised fields
# ----------------------------
all_df["league"] = norm_str(safe_col(all_df, "league"))
all_df["signal_btts_runtime_eval"] = norm_upper(
    first_present(all_df, ["signal_btts_runtime", "signal_btts"])
)
all_df["signal_btts_eval"] = norm_upper(
    first_present(all_df, ["signal_btts"])
)
all_df["signal_bucket_explicit"] = classify_signal_bucket(all_df["signal_btts_runtime_eval"])

all_df["actual_btts_yes"] = to_num(safe_col(all_df, "actual_btts_yes"))
all_df["btts_yes_hit"] = to_num(safe_col(all_df, "btts_yes_hit"))
all_df["bookie_od"] = to_num(safe_col(all_df, "bookie_od"))
all_df["model_p_for_bookie"] = to_num(safe_col(all_df, "model_p_for_bookie"))
all_df["cs_max"] = to_num(safe_col(all_df, "cs_max"))
all_df["btts_top3_mass"] = to_num(safe_col(all_df, "btts_top3_mass"))
all_df["p00_est"] = to_num(safe_col(all_df, "p00_est"))
all_df["exp_goals_sum"] = to_num(safe_col(all_df, "exp_goals_sum"))

# recompute live-pass logic from raw+scored universe only
all_df["label_pass"] = (
    all_df["signal_btts_runtime_eval"].isin(["STRONG_YES", "VERY_STRONG_YES"])
).astype(int)

all_df["brazil_block"] = (
    all_df["league"].eq("Brazil Serie A")
    & all_df["signal_btts_runtime_eval"].eq("VERY_STRONG_YES")
).astype(int)

all_df["weak_yes_block"] = all_df["signal_btts_runtime_eval"].eq("WEAK_YES").astype(int)

all_df["neutral_or_other_block"] = (
    ~all_df["signal_btts_runtime_eval"].isin(["STRONG_YES", "VERY_STRONG_YES", "WEAK_YES"])
).astype(int)

ge2h = to_num(safe_col(all_df, "home_ge2_confidence"))
ge2a = to_num(safe_col(all_df, "away_ge2_confidence"))
ge2_min = pd.concat([ge2h, ge2a], axis=1).min(axis=1)
ge2_floor = pd.Series(0.22, index=all_df.index)
ge2_floor.loc[all_df["signal_btts_runtime_eval"].eq("STRONG_YES")] = 0.30
all_df["ge2_pass"] = (ge2_min.isna() | (ge2_min >= ge2_floor)).astype(int)

imp_used = to_num(first_present(all_df, ["bookie_implied_used", "bookie_implied"]))
floor = np.maximum(0.53, imp_used.fillna(0.0))
all_df["model_floor_pass"] = (all_df["model_p_for_bookie"].fillna(-1e9) >= floor).astype(int)

ph = to_num(safe_col(all_df, "p_home_fts"))
pa = to_num(safe_col(all_df, "p_away_fts"))
fts_max = pd.concat([ph, pa], axis=1).max(axis=1)
fts_cap = pd.Series(0.30, index=all_df.index)
fts_cap.loc[all_df["signal_btts_runtime_eval"].eq("STRONG_YES")] = 0.30
all_df["fts_pass"] = (fts_max.isna() | (fts_max <= fts_cap)).astype(int)

cs_max = to_num(safe_col(all_df, "cs_max"))
keep_cs = (~all_df["signal_btts_runtime_eval"].eq("STRONG_YES")) | cs_max.isna() | (cs_max <= 0.40)
all_df["cs_pass"] = keep_cs.astype(int)

all_df["double_blank_pass"] = 1
all_df["confirmation_pass"] = 1

all_df["final_live_pass"] = (
    all_df["label_pass"].eq(1)
    & all_df["brazil_block"].eq(0)
    & all_df["ge2_pass"].eq(1)
    & all_df["model_floor_pass"].eq(1)
    & all_df["fts_pass"].eq(1)
    & all_df["cs_pass"].eq(1)
    & all_df["double_blank_pass"].eq(1)
    & all_df["confirmation_pass"].eq(1)
).astype(int)

# ----------------------------
# Scenario inclusion rules
# ----------------------------
sig = all_df["signal_btts_runtime_eval"]
lg = all_df["league"]

sc_current = all_df["final_live_pass"].eq(1)
sc_weak_family_only = sig.eq("WEAK_YES") & lg.isin(WEAK_LEAGUE_FAMILY)
sc_live_plus_weak_family = sc_current | sc_weak_family_only

sc_fts_override = (
    lg.isin(FTS_OVERRIDE_LEAGUES)
    & all_df["fts_pass"].eq(0)
    & all_df["label_pass"].eq(1)
    & all_df["brazil_block"].eq(0)
    & all_df["ge2_pass"].eq(1)
    & all_df["model_floor_pass"].eq(1)
    & all_df["cs_pass"].eq(1)
    & all_df["double_blank_pass"].eq(1)
    & all_df["confirmation_pass"].eq(1)
)

sc_live_plus_fts = sc_current | sc_fts_override
sc_live_plus_weak_plus_fts = sc_current | sc_weak_family_only | sc_fts_override
sc_selective_neutral = sc_current.copy()

scenario_masks = {
    "SCENARIO_CURRENT_LIVE_REPAIRED": sc_current,
    "SCENARIO_WEAK_BY_LEAGUE_FAMILY_ONLY": sc_weak_family_only,
    "SCENARIO_LIVE_PLUS_WEAK_BY_LEAGUE": sc_live_plus_weak_family,
    "SCENARIO_LIVE_PLUS_FTS_OVERRIDE": sc_live_plus_fts,
    "SCENARIO_LIVE_PLUS_WEAK_PLUS_FTS_OVERRIDE": sc_live_plus_weak_plus_fts,
    "SCENARIO_LIVE_PLUS_SELECTIVE_NEUTRAL": sc_selective_neutral,
}

# ----------------------------
# Summaries
# ----------------------------
summary_rows = []
by_league_frames = []
by_signal_frames = []
by_first_excl_frames = []
included_dump = []

for scen, mask in scenario_masks.items():
    sub = all_df.loc[mask.fillna(False)].copy()
    sub["scenario"] = scen

    summary_rows.append(scenario_metrics(sub, scen))
    by_league_frames.append(grouped_metrics(sub, scen, "league"))
    by_signal_frames.append(grouped_metrics(sub, scen, "signal_bucket_explicit"))

    fx = sub.copy()
    fx["first_exclusion"] = norm_str(safe_col(fx, "first_exclusion")).replace("", "BASELINE_OR_UNKNOWN")
    by_first_excl_frames.append(grouped_metrics(fx, scen, "first_exclusion"))

    included_dump.append(sub)

summary = pd.DataFrame(summary_rows).sort_values(["level_stake_profit", "rows"], ascending=[False, False]).reset_index(drop=True)
by_league = pd.concat(by_league_frames, ignore_index=True) if by_league_frames else pd.DataFrame()
by_signal = pd.concat(by_signal_frames, ignore_index=True) if by_signal_frames else pd.DataFrame()
by_first_excl = pd.concat(by_first_excl_frames, ignore_index=True) if by_first_excl_frames else pd.DataFrame()
included_all = pd.concat(included_dump, ignore_index=True) if included_dump else pd.DataFrame()

# ----------------------------
# Compare vs baseline
# ----------------------------
base = summary.loc[summary["scenario"].eq("SCENARIO_CURRENT_LIVE_REPAIRED")].copy()
if base.empty:
    raise SystemExit("Baseline scenario missing from summary.")
base = base.iloc[0]

compare = summary.copy()
compare["delta_rows_vs_baseline"] = compare["rows"] - int(base["rows"])
compare["delta_wins_vs_baseline"] = compare["wins"] - int(base["wins"])
compare["delta_losses_vs_baseline"] = compare["losses"] - int(base["losses"])
compare["delta_hit_rate_vs_baseline"] = compare["hit_rate"] - float(base["hit_rate"]) if pd.notna(base["hit_rate"]) else np.nan
compare["delta_roi_vs_baseline"] = compare["roi_level_stake"] - float(base["roi_level_stake"]) if pd.notna(base["roi_level_stake"]) else np.nan
compare["delta_profit_vs_baseline"] = compare["level_stake_profit"] - float(base["level_stake_profit"])

# ----------------------------
# Marginal adds vs baseline
# ----------------------------
base_keys = set(all_df.loc[sc_current.fillna(False), "__row_key"].tolist())

marginal_rows = []
marginal_by_league_frames = []
marginal_by_signal_frames = []
marginal_by_first_excl_frames = []
marginal_detail = []

for scen, mask in scenario_masks.items():
    if scen == "SCENARIO_CURRENT_LIVE_REPAIRED":
        continue
    sub = all_df.loc[mask.fillna(False)].copy()
    sub = sub.loc[~sub["__row_key"].isin(base_keys)].copy()
    sub["scenario"] = scen

    met = scenario_metrics(sub, scen)
    met["marginal_vs_baseline"] = 1
    marginal_rows.append(met)

    marginal_by_league_frames.append(grouped_metrics(sub, scen, "league"))
    marginal_by_signal_frames.append(grouped_metrics(sub, scen, "signal_bucket_explicit"))

    fx = sub.copy()
    fx["first_exclusion"] = norm_str(safe_col(fx, "first_exclusion")).replace("", "BASELINE_OR_UNKNOWN")
    marginal_by_first_excl_frames.append(grouped_metrics(fx, scen, "first_exclusion"))

    marginal_detail.append(sub)

marginal_summary = pd.DataFrame(marginal_rows).sort_values(["level_stake_profit", "rows"], ascending=[False, False]).reset_index(drop=True)
marginal_by_league_frames = [df for df in marginal_by_league_frames if not df.empty]
marginal_by_signal_frames = [df for df in marginal_by_signal_frames if not df.empty]
marginal_by_first_excl_frames = [df for df in marginal_by_first_excl_frames if not df.empty]

marginal_by_league = pd.concat(marginal_by_league_frames, ignore_index=True) if marginal_by_league_frames else pd.DataFrame()
marginal_by_signal = pd.concat(marginal_by_signal_frames, ignore_index=True) if marginal_by_signal_frames else pd.DataFrame()
marginal_by_first_excl = pd.concat(marginal_by_first_excl_frames, ignore_index=True) if marginal_by_first_excl_frames else pd.DataFrame()
marginal_detail_df = pd.concat(marginal_detail, ignore_index=True) if marginal_detail else pd.DataFrame()

# ----------------------------
# Writes
# ----------------------------
summary.to_csv(OUTDIR / "SCENARIO_SUMMARY.csv", index=False)
compare.to_csv(OUTDIR / "SCENARIO_COMPARE_VS_BASELINE.csv", index=False)
marginal_summary.to_csv(OUTDIR / "SCENARIO_MARGINAL_SUMMARY_VS_BASELINE.csv", index=False)
all_df.to_csv(OUTDIR / "FULL_UNIVERSE_BTTS_CANDIDATES__RAW_PLUS_SCORED_ENRICHMENT.csv", index=False)

if not by_league.empty:
    by_league.sort_values(["scenario", "level_stake_profit", "rows"], ascending=[True, False, False]).to_csv(
        OUTDIR / "SCENARIO_BY_LEAGUE.csv", index=False
    )

if not by_signal.empty:
    by_signal.sort_values(["scenario", "level_stake_profit", "rows"], ascending=[True, False, False]).to_csv(
        OUTDIR / "SCENARIO_BY_SIGNAL_BUCKET.csv", index=False
    )

if not by_first_excl.empty:
    by_first_excl.sort_values(["scenario", "level_stake_profit", "rows"], ascending=[True, False, False]).to_csv(
        OUTDIR / "SCENARIO_BY_FIRST_EXCLUSION.csv", index=False
    )

if not included_all.empty:
    included_all.to_csv(OUTDIR / "SCENARIO_INCLUDED_ROWS__ALL.csv", index=False)

if not marginal_detail_df.empty:
    marginal_detail_df.to_csv(OUTDIR / "SCENARIO_MARGINAL_ROWS__VS_BASELINE.csv", index=False)

if not marginal_by_league.empty:
    marginal_by_league.sort_values(["scenario", "level_stake_profit", "rows"], ascending=[True, False, False]).to_csv(
        OUTDIR / "SCENARIO_MARGINAL_BY_LEAGUE__VS_BASELINE.csv", index=False
    )

if not marginal_by_signal.empty:
    marginal_by_signal.sort_values(["scenario", "level_stake_profit", "rows"], ascending=[True, False, False]).to_csv(
        OUTDIR / "SCENARIO_MARGINAL_BY_SIGNAL_BUCKET__VS_BASELINE.csv", index=False
    )

if not marginal_by_first_excl.empty:
    marginal_by_first_excl.sort_values(["scenario", "level_stake_profit", "rows"], ascending=[True, False, False]).to_csv(
        OUTDIR / "SCENARIO_MARGINAL_BY_FIRST_EXCLUSION__VS_BASELINE.csv", index=False
    )

# ----------------------------
# Console output
# ----------------------------
print(f"OUTDIR: {OUTDIR}")
print("\nUNIVERSE")
print(f"rows={len(all_df)}")
print(f"graded={int(all_df['btts_yes_hit'].notna().sum())}")
print(f"wins={int(to_num(all_df['btts_yes_hit']).fillna(0).sum())}")
print(f"losses={int(all_df['btts_yes_hit'].notna().sum() - to_num(all_df['btts_yes_hit']).fillna(0).sum())}")

print("\nSCENARIO SUMMARY")
print(summary.to_string(index=False))

print("\nCOMPARE VS BASELINE")
print(compare.to_string(index=False))

print("\nMARGINAL SUMMARY VS BASELINE")
print(marginal_summary.to_string(index=False))

if not by_league.empty:
    print("\nTOP LEAGUE VIEW")
    print(
        by_league.sort_values(["scenario", "level_stake_profit", "rows"], ascending=[True, False, False])
        .groupby("scenario", as_index=False, group_keys=False)
        .head(15)
        .to_string(index=False)
    )

if not by_signal.empty:
    print("\nTOP SIGNAL VIEW")
    print(
        by_signal.sort_values(["scenario", "level_stake_profit", "rows"], ascending=[True, False, False])
        .to_string(index=False)
    )

if not marginal_by_league.empty:
    print("\nMARGINAL TOP LEAGUE VIEW")
    print(
        marginal_by_league.sort_values(["scenario", "level_stake_profit", "rows"], ascending=[True, False, False])
        .groupby("scenario", as_index=False, group_keys=False)
        .head(15)
        .to_string(index=False)
    )

print("\nDONE")