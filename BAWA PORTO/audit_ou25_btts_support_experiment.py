#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

SRC = Path("predictions_output/2026-03-21/BOOKIE_IMP20_ALLMARKETS_2026-03-21_to_2026-03-23.csv")
OU25_DETAIL = Path("predictions_output/2026-03-21/MANUAL_RESULTS__OU25_OVER_DETAIL.csv")
BTTS_DETAIL = Path("predictions_output/2026-03-21/MANUAL_RESULTS__BTTS_YES_DETAIL.csv")
OUTDIR = SRC.parent

def s_str(df, col, default=""):
    return df.get(col, pd.Series(default, index=df.index)).astype("string").fillna(default)

def s_num(df, col):
    return pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")

def rate_summary(df, hit_col):
    graded = df[hit_col].notna()
    wins = pd.to_numeric(df.loc[graded, hit_col], errors="coerce").fillna(0).sum()
    n = int(graded.sum())
    return {
        "rows": int(len(df)),
        "graded": n,
        "wins": float(wins),
        "hit_rate": float(wins / n) if n > 0 else np.nan,
    }

def add_ou25_support_flags(df):
    out = df.copy()

    out["market_l"] = s_str(out, "market").str.lower().str.strip()
    out["selection_u"] = s_str(out, "selection", s_str(out, "bookie_pick")).str.upper().str.strip()
    out["signal_over25_src"] = s_str(out, "signal_over25").str.upper().str.strip()
    out["signal_btts_src"] = s_str(out, "signal_btts").str.upper().str.strip()

    out["is_ou25_over"] = out["market_l"].eq("ou25") & out["selection_u"].eq("OVER25")

    out["model_p"] = s_num(out, "model_p_for_bookie")
    out["xg_sum"] = s_num(out, "xg_sum_pre_match")
    out["exp_goals_sum_num"] = s_num(out, "exp_goals_sum")

    out["goaliness_home"] = s_num(out, "goaliness_avg_5_home")
    out["goaliness_away"] = s_num(out, "goaliness_avg_5_away")
    out["goaliness_avg"] = (out["goaliness_home"] + out["goaliness_away"]) / 2.0

    out["over25_home"] = s_num(out, "over25_rate_5_home")
    out["over25_away"] = s_num(out, "over25_rate_5_away")
    out["under25_home"] = s_num(out, "under25_rate_5_home")
    out["under25_away"] = s_num(out, "under25_rate_5_away")

    out["scored_home"] = s_num(out, "scored_rate_5_home")
    out["scored_away"] = s_num(out, "scored_rate_5_away")
    out["cs_home"] = s_num(out, "clean_sheet_rate_5_home")
    out["cs_away"] = s_num(out, "clean_sheet_rate_5_away")

    out["cs1"] = s_str(out, "cs1").str.strip()

    out["ou25_model_support"] = out["model_p"] >= 0.50
    out["ou25_xg_support"] = out["xg_sum"] >= 2.55
    out["ou25_exp_support"] = out["exp_goals_sum_num"] >= 2.60
    out["ou25_goaliness_support"] = out["goaliness_avg"] >= 2.50
    out["ou25_overrate_support"] = (out["over25_home"] >= 0.40) & (out["over25_away"] >= 0.40)
    out["ou25_scored_support"] = (out["scored_home"] >= 0.60) & (out["scored_away"] >= 0.60)

    out["ou25_support_bucket_count"] = (
        out["ou25_xg_support"].fillna(False).astype(int)
        + out["ou25_exp_support"].fillna(False).astype(int)
        + ((out["ou25_goaliness_support"] | out["ou25_overrate_support"] | out["ou25_scored_support"]).fillna(False).astype(int))
    )

    out["ou25_under_trap"] = (
        (out["under25_home"] >= 0.60)
        & (out["under25_away"] >= 0.60)
    )

    out["ou25_clean_sheet_trap"] = (
        (out["cs_home"] >= 0.40)
        & (out["cs_away"] >= 0.40)
    )

    out["ou25_cs_under_magnet"] = out["cs1"].isin(["0-0", "1-0", "0-1"])

    out["ou25_support_pass"] = (
        out["is_ou25_over"]
        & out["ou25_model_support"]
        & (out["ou25_support_bucket_count"] >= 2)
    )

    out["ou25_support_pass_strict"] = (
        out["ou25_support_pass"]
        & (~out["ou25_under_trap"].fillna(False))
        & (~out["ou25_clean_sheet_trap"].fillna(False))
    )

    if "signal_over25" not in out.columns:
        out["signal_over25"] = out["signal_over25_src"]
    if "signal_btts" not in out.columns:
        out["signal_btts"] = out["signal_btts_src"]
    return out

def add_btts_support_flags(df):
    out = df.copy()

    out["market_l"] = s_str(out, "market").str.lower().str.strip()
    out["selection_u"] = s_str(out, "selection", s_str(out, "bookie_pick")).str.upper().str.strip()
    out["signal_btts_u"] = s_str(out, "signal_btts_runtime", s_str(out, "signal_btts")).str.upper().str.strip()
    out["signal_btts_runtime_src"] = s_str(out, "signal_btts_runtime").str.upper().str.strip()
    out["signal_btts_src"] = s_str(out, "signal_btts").str.upper().str.strip()

    out["is_btts_yes"] = out["market_l"].eq("btts") & out["selection_u"].eq("YES")

    out["model_p"] = s_num(out, "model_p_for_bookie")
    out["xg_sum"] = s_num(out, "xg_sum_pre_match")
    out["exp_goals_sum_num"] = s_num(out, "exp_goals_sum")

    out["scored_home"] = s_num(out, "scored_rate_5_home")
    out["scored_away"] = s_num(out, "scored_rate_5_away")
    out["conceded_home"] = s_num(out, "conceded_rate_5_home")
    out["conceded_away"] = s_num(out, "conceded_rate_5_away")
    out["cs_home"] = s_num(out, "clean_sheet_rate_5_home")
    out["cs_away"] = s_num(out, "clean_sheet_rate_5_away")
    out["btts_home"] = s_num(out, "btts_rate_5_home")
    out["btts_away"] = s_num(out, "btts_rate_5_away")

    out["btts_model_support"] = out["model_p"] >= 0.50
    out["btts_signal_support"] = out["signal_btts_u"].isin(["WEAK_YES", "STRONG_YES", "VERY_STRONG_YES"])
    out["btts_scored_support"] = (out["scored_home"] >= 0.60) & (out["scored_away"] >= 0.60)
    out["btts_conceded_support"] = (out["conceded_home"] >= 0.60) & (out["conceded_away"] >= 0.60)
    out["btts_clean_support"] = (out["cs_home"] <= 0.40) & (out["cs_away"] <= 0.40)
    out["btts_rate_support"] = (out["btts_home"] >= 0.40) & (out["btts_away"] >= 0.40)
    out["btts_xg_support"] = out["xg_sum"] >= 2.45
    out["btts_exp_support"] = out["exp_goals_sum_num"] >= 2.45

    out["btts_structure_support"] = (
        out["btts_conceded_support"].fillna(False)
        | out["btts_rate_support"].fillna(False)
    )

    out["btts_support_pass"] = (
        out["is_btts_yes"]
        & out["btts_model_support"]
        & out["btts_signal_support"]
        & out["btts_scored_support"]
        & out["btts_clean_support"]
        & out["btts_structure_support"]
        & (out["btts_xg_support"] | out["btts_exp_support"])
    )
    if "signal_btts_runtime" not in out.columns:
        out["signal_btts_runtime"] = out["signal_btts_runtime_src"]
    if "signal_btts" not in out.columns:
        out["signal_btts"] = out["signal_btts_src"]

    return out

def main():
    src = pd.read_csv(SRC)

    ou25 = pd.read_csv(OU25_DETAIL)
    btts = pd.read_csv(BTTS_DETAIL)

    src = add_ou25_support_flags(src)
    src = add_btts_support_flags(src)

    key_cols = ["fixture_key", "market", "selection"]

    ou25_merge_cols = key_cols + [
        "ou25_support_pass",
        "ou25_support_pass_strict",
        "ou25_model_support",
        "ou25_xg_support",
        "ou25_exp_support",
        "ou25_goaliness_support",
        "ou25_overrate_support",
        "ou25_scored_support",
        "ou25_support_bucket_count",
        "ou25_under_trap",
        "ou25_clean_sheet_trap",
        "ou25_cs_under_magnet",
        "xg_sum",
        "exp_goals_sum_num",
        "goaliness_avg",
        "over25_home",
        "over25_away",
        "scored_home",
        "scored_away",
        "under25_home",
        "under25_away",
        "cs_home",
        "cs_away",
        "signal_over25",
        "signal_btts",
        "signal_btts_runtime",
    ]
    ou25_merge_cols = [c for c in ou25_merge_cols if c in src.columns]

    ou25 = ou25.merge(
        src[ou25_merge_cols],
        on=key_cols,
        how="left",
    )

    btts_merge_cols = key_cols + [
        "btts_support_pass",
        "btts_model_support",
        "btts_signal_support",
        "btts_scored_support",
        "btts_conceded_support",
        "btts_clean_support",
        "btts_rate_support",
        "btts_xg_support",
        "btts_exp_support",
        "xg_sum",
        "exp_goals_sum_num",
        "scored_home",
        "scored_away",
        "conceded_home",
        "conceded_away",
        "cs_home",
        "cs_away",
        "btts_home",
        "btts_away",
        "signal_btts",
        "signal_btts_runtime",
    ]
    btts_merge_cols = [c for c in btts_merge_cols if c in src.columns]

    btts = btts.merge(
        src[btts_merge_cols],
        on=key_cols,
        how="left",
    )

    if "signal_over25" not in ou25.columns:
        ou25["signal_over25"] = ""
    else:
        ou25["signal_over25"] = ou25["signal_over25"].astype("string").fillna("")

    if "signal_btts" not in ou25.columns:
        ou25["signal_btts"] = ""
    else:
        ou25["signal_btts"] = ou25["signal_btts"].astype("string").fillna("")

    if "signal_btts_runtime" not in ou25.columns:
        ou25["signal_btts_runtime"] = ""
    else:
        ou25["signal_btts_runtime"] = ou25["signal_btts_runtime"].astype("string").fillna("")

    if "signal_btts" not in btts.columns:
        btts["signal_btts"] = ""
    else:
        btts["signal_btts"] = btts["signal_btts"].astype("string").fillna("")

    if "signal_btts_runtime" not in btts.columns:
        btts["signal_btts_runtime"] = ""
    else:
        btts["signal_btts_runtime"] = btts["signal_btts_runtime"].astype("string").fillna("")

    # OU25 summaries
    ou25_all = rate_summary(ou25, "ou25_hit")
    ou25_pass = rate_summary(ou25[ou25["ou25_support_pass"] == True], "ou25_hit")
    ou25_pass_strict = rate_summary(ou25[ou25["ou25_support_pass_strict"] == True], "ou25_hit")

    ou25_summary = pd.DataFrame([
        {"profile": "ALL_OU25_OVER", **ou25_all},
        {"profile": "SUPPORT_PASS", **ou25_pass},
        {"profile": "SUPPORT_PASS_STRICT", **ou25_pass_strict},
    ])

    ou25_signal_group_col = "signal_over25" if "signal_over25" in ou25.columns else None
    if ou25_signal_group_col is None:
        ou25["signal_over25_group"] = ""
        ou25_signal_group_col = "signal_over25_group"

    ou25_by_signal = (
        ou25.groupby([ou25_signal_group_col, "ou25_support_pass"], dropna=False)
        .agg(rows=("fixture_key", "size"),
             graded=("ou25_hit", lambda s: s.notna().sum()),
             wins=("ou25_hit", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()))
        .reset_index()
    )
    ou25_by_signal["hit_rate"] = ou25_by_signal["wins"] / ou25_by_signal["graded"].replace(0, np.nan)
    ou25_by_signal = ou25_by_signal.rename(columns={ou25_signal_group_col: "signal_over25_group"})

    ou25_wins_supported = ou25[(ou25["ou25_hit"] == 1) & (ou25["ou25_support_pass"] == True)].copy()
    ou25_losses_supported = ou25[(ou25["ou25_hit"] == 0) & (ou25["ou25_support_pass"] == True)].copy()

    # BTTS summaries
    btts_all = rate_summary(btts, "btts_yes_hit")
    btts_pass = rate_summary(btts[btts["btts_support_pass"] == True], "btts_yes_hit")

    btts_summary = pd.DataFrame([
        {"profile": "ALL_BTTS_YES", **btts_all},
        {"profile": "SUPPORT_PASS", **btts_pass},
    ])

    btts_signal_group_col = "signal_btts_runtime" if "signal_btts_runtime" in btts.columns else "signal_btts"
    btts_by_signal = (
        btts.groupby([btts_signal_group_col, "btts_support_pass"], dropna=False)
        .agg(rows=("fixture_key", "size"),
             graded=("btts_yes_hit", lambda s: s.notna().sum()),
             wins=("btts_yes_hit", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()))
        .reset_index()
    )
    btts_by_signal["hit_rate"] = btts_by_signal["wins"] / btts_by_signal["graded"].replace(0, np.nan)
    btts_by_signal = btts_by_signal.rename(columns={btts_signal_group_col: "signal_btts_group"})

    btts_wins_supported = btts[(btts["btts_yes_hit"] == 1) & (btts["btts_support_pass"] == True)].copy()
    btts_losses_supported = btts[(btts["btts_yes_hit"] == 0) & (btts["btts_support_pass"] == True)].copy()

    # write
    ou25_summary.to_csv(OUTDIR / "EXPERIMENT__OU25_SUPPORT_SUMMARY.csv", index=False)
    ou25_by_signal.to_csv(OUTDIR / "EXPERIMENT__OU25_SUPPORT_BY_SIGNAL.csv", index=False)
    ou25_wins_supported.to_csv(OUTDIR / "EXPERIMENT__OU25_SUPPORT_WINS.csv", index=False)
    ou25_losses_supported.to_csv(OUTDIR / "EXPERIMENT__OU25_SUPPORT_LOSSES.csv", index=False)

    btts_summary.to_csv(OUTDIR / "EXPERIMENT__BTTS_YES_SUPPORT_SUMMARY.csv", index=False)
    btts_by_signal.to_csv(OUTDIR / "EXPERIMENT__BTTS_YES_SUPPORT_BY_SIGNAL.csv", index=False)
    btts_wins_supported.to_csv(OUTDIR / "EXPERIMENT__BTTS_YES_SUPPORT_WINS.csv", index=False)
    btts_losses_supported.to_csv(OUTDIR / "EXPERIMENT__BTTS_YES_SUPPORT_LOSSES.csv", index=False)

    print("\nOU25 SUPPORT SUMMARY")
    print(ou25_summary.to_string(index=False))

    print("\nBTTS YES SUPPORT SUMMARY")
    print(btts_summary.to_string(index=False))

    print("\nWROTE:")
    for f in [
        "EXPERIMENT__OU25_SUPPORT_SUMMARY.csv",
        "EXPERIMENT__OU25_SUPPORT_BY_SIGNAL.csv",
        "EXPERIMENT__OU25_SUPPORT_WINS.csv",
        "EXPERIMENT__OU25_SUPPORT_LOSSES.csv",
        "EXPERIMENT__BTTS_YES_SUPPORT_SUMMARY.csv",
        "EXPERIMENT__BTTS_YES_SUPPORT_BY_SIGNAL.csv",
        "EXPERIMENT__BTTS_YES_SUPPORT_WINS.csv",
        "EXPERIMENT__BTTS_YES_SUPPORT_LOSSES.csv",
    ]:
        print(OUTDIR / f)

if __name__ == "__main__":
    main()