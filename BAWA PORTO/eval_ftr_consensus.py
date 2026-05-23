#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from prediction_overlay import _match_key, _coalesce_match_date_series
except Exception:
    _match_key = None
    _coalesce_match_date_series = None

def actual_ftr(hg, ag):
    if pd.isna(hg) or pd.isna(ag):
        return None
    if hg > ag: return "HOME"
    if hg < ag: return "AWAY"
    return "DRAW"

def load_matches(league: str, matches_root="Matches") -> pd.DataFrame:
    mdir = Path(matches_root) / league
    if not mdir.exists():
        return pd.DataFrame()
    # prefer matches.csv if present else all csvs
    p = mdir / "matches.csv"
    files = [p] if p.exists() else sorted(mdir.glob("*.csv"))
    frames=[]
    for f in files:
        try:
            df=pd.read_csv(f)
            df["__src_csv"]=f.name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df=pd.concat(frames, ignore_index=True, sort=False)

    # dates
    if callable(_coalesce_match_date_series):
        try:
            df["match_date"]=_coalesce_match_date_series(df)
        except Exception:
            pass

    # key
    if callable(_match_key):
        try:
            df["fixture_key"]=df.apply(_match_key, axis=1)
        except Exception:
            pass

    # goals
    for c in ("home_team_goal_count","away_team_goal_count"):
        if c in df.columns:
            df[c]=pd.to_numeric(df[c], errors="coerce")

    # realised only
    if "home_team_goal_count" in df.columns and "away_team_goal_count" in df.columns:
        df=df[df[["home_team_goal_count","away_team_goal_count"]].notna().all(axis=1)].copy()

    # dedupe by fixture_key if possible
    if "fixture_key" in df.columns:
        df["fixture_key"]=df["fixture_key"].astype("string").fillna("").str.strip()
        df=df[df["fixture_key"].ne("")].copy()
        df=df.drop_duplicates(subset=["fixture_key"], keep="first").copy()

    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--consensus", required=True, help="...__FTR_CONSENSUS.csv")
    ap.add_argument("--league", required=True)
    ap.add_argument("--matches-root", default="Matches")
    ap.add_argument("--bins", type=int, default=5, help="Number of quantile bins for XOG percentile tables (default: 5)")
    ap.add_argument("--top-n", type=int, default=10, help="Top-N table size for XOG spread ranking (default: 10)")
    ap.add_argument("--show-buckets", action="store_true", help="Show accuracy by XOG pick_score and XOG spread quantile bins")
    args=ap.parse_args()

    cons = pd.read_csv(args.consensus)
    cons["fixture_key"] = cons["fixture_key"].astype("string").fillna("").str.strip()

    # Filter to the requested league if present (prevents join_rate/accuracy distortion when consensus contains multiple leagues)
    if "league" in cons.columns:
        cons["league"] = cons["league"].astype(str).str.strip()
        cons = cons[cons["league"].eq(str(args.league).strip())].copy()

    m=load_matches(args.league, args.matches_root)
    if m.empty:
        raise SystemExit("No match rows loaded")

    # build actual outcome
    m["actual_pick"]=m.apply(lambda r: actual_ftr(r.get("home_team_goal_count"), r.get("away_team_goal_count")), axis=1)

    # join
    j = cons.merge(m[["fixture_key","actual_pick"]], on="fixture_key", how="inner")
    join_rate = float(len(j)) / float(max(len(cons), 1))
    j_ok = j.copy()

    # metrics
    j_ok["consensus_pick"] = j_ok.get("consensus_pick", "").astype(str).str.upper().str.strip()
    j_ok["actual_pick"] = j_ok.get("actual_pick", "").astype(str).str.upper().str.strip()
    j_ok["hit"] = (j_ok["consensus_pick"] == j_ok["actual_pick"]).astype(int)
    acc=float(j_ok["hit"].mean()) if len(j_ok) else 0.0

    print("league:", args.league)
    print("rows:", len(cons), "joined:", len(j_ok), "join_rate:", round(join_rate,3))
    print("accuracy:", round(acc,3))

    def by(col):
        if col not in j_ok.columns:
            return
        grp=j_ok.groupby(col)["hit"].agg(["count","mean"]).reset_index().sort_values("mean", ascending=False)
        grp["mean"]=grp["mean"].round(3)
        print("\nby", col)
        print(grp.to_string(index=False))

    by("consensus_lane")
    by("xog_tier")
    by("consensus_tier")

    def bucket(col: str, q: int = 5) -> None:
        if col not in j_ok.columns:
            return
        s = pd.to_numeric(j_ok[col], errors="coerce")
        tmp = j_ok.loc[s.notna(), [col, "hit"]].copy()
        if tmp.empty:
            return
        try:
            tmp["bin"] = pd.qcut(pd.to_numeric(tmp[col], errors="coerce"), int(q), duplicates="drop")
        except Exception:
            return
        g = tmp.groupby("bin")["hit"].agg(["count", "mean"]).reset_index()
        g["mean"] = g["mean"].round(3)
        print(f"\nBy {col} quantile bins (q={q}):")
        print(g.to_string(index=False))

    if bool(getattr(args, "show_buckets", False)):
        bucket("xog_pick_score", q=int(getattr(args, "bins", 5)))
        bucket("xog_spread", q=int(getattr(args, "bins", 5)))

        # Top-N by spread (if present)
        if "xog_spread" in j_ok.columns:
            topn = int(getattr(args, "top_n", 10))
            s = pd.to_numeric(j_ok["xog_spread"], errors="coerce")
            top = j_ok.loc[s.notna()].copy()
            top["xog_spread"] = s.loc[s.notna()]
            top = top.sort_values("xog_spread", ascending=False).head(max(0, topn))
            if not top.empty:
                print(f"\nTop {topn} by xog_spread:")
                cols = [c for c in ["league","home_team_name","away_team_name","consensus_pick","actual_pick","xog_pick_score","xog_spread","xog_tier","consensus_lane","hit"] if c in top.columns]
                print(top[cols].to_string(index=False))
                try:
                    print("acc_topN_spread:", round(float(top["hit"].mean()), 3), "| n=", len(top))
                except Exception:
                    pass

if __name__=="__main__":
    main()