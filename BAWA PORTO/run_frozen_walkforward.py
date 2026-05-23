
#!/usr/bin/env python3
"""run_frozen_walkforward.py

Frozen monthly walk-forward runner.

Purpose
-------
Run month-by-month out-of-sample evaluation using the EXISTING production
ModelStore only. This script does NOT retrain anything and does NOT use
ModelStore_wf.

Per month it will:
  1) run bookie_allmarkets.py for the month window
  2) locate the produced ALLMARKETS csv
  3) run backtest_deploy_csv.py against that export
  4) optionally run apply_frozen_product_gates.py using the learned frozen product thresholds
  5) archive raw + gated artifacts into walkforward_frozen/YYYY-MM/
  6) build simple month / league / market summaries from the gated file when available

Design rules
------------
- Production models only: ModelStore/
- No retraining
- No sandbox
- No mutation of production artifacts
- Archive outputs into a dedicated walkforward_frozen/ tree

FTR design notes
----------------
- FTR value profile = value-priced FTR selections (legacy long-odds / EV style)
- FTR accuracy profile = moderate favourites, not a mirror image of value
- If an accuracy profile shows ALL rows but zero MARKET::ftr, that is an FTR
  accuracy dead-zone, not a failed batch run.

Default frozen profile
----------------------
This runner preserves frozen threshold defaults learned from backtesting and
replays them through apply_frozen_product_gates.py for frozen OOS evaluation.

Example
-------
python run_frozen_walkforward.py \
  --start-month 2024-11 \
  --end-month 2025-01 \
  --leagues "England Premier League,England Championship,England EFL League 1,England FA Cup,Japan J1,Norway Eliteserien,Netherlands Eredivisie,Belgium Pro,Scotland Premiership,Brazil Serie A,USA MLS,Portugal Liga,Spain La Liga,Italy Serie A,France Ligue 1,Germany Bundesliga,Europa Conference,Europa League,Champions League" \
  --markets ftr,ou25,btts,tg15,tg25 \
  --strict \
  --frozen-profile april2025
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# date helpers
# -----------------------------

def _parse_month(s: str) -> date:
    dt = datetime.strptime(s, "%Y-%m").date()
    return date(dt.year, dt.month, 1)


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _month_end(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    nxt = date(d.year, d.month + 1, 1)
    return nxt - timedelta(days=1)


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, _month_end(date(y, m, 1)).day)
    return date(y, m, day)


def _month_range(start_month: str, end_month: str) -> List[date]:
    start = _parse_month(start_month)
    end = _parse_month(end_month)
    out: List[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = _add_months(cur, 1)
    return out


# -----------------------------
# process helpers
# -----------------------------

def _quote(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _run(cmd: List[str], *, cwd: Optional[Path] = None, dry_run: bool = False) -> None:
    print(f">>> {_quote(cmd)}")
    if dry_run:
        return
    try:
        r = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            text=True,
            capture_output=True,
        )
        if r.stdout:
            print(r.stdout.rstrip())
        if r.stderr:
            print(r.stderr.rstrip())
    except subprocess.CalledProcessError as e:
        if getattr(e, "stdout", None):
            print(str(e.stdout).rstrip())
        if getattr(e, "stderr", None):
            print(str(e.stderr).rstrip())
        raise


# -----------------------------
# file discovery helpers
# -----------------------------

def _find_latest_matching(root: Path, pattern: str) -> Optional[Path]:
    matches = []
    for p in root.rglob(pattern):
        try:
            matches.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            continue
    if not matches:
        return None
    matches.sort(key=lambda t: t[0])
    return matches[-1][1]


def _find_allmarkets_csv(predictions_root: Path, date_from: str, date_to: str) -> Optional[Path]:
    patt = f"BOOKIE_IMP*_ALLMARKETS_{date_from}_to_{date_to}.csv"
    return _find_latest_matching(predictions_root, patt)


def _find_backtest_csv(search_root: Path, allmarkets_path: Path) -> Optional[Path]:
    exact = search_root / f"{allmarkets_path.stem}__BACKTEST.csv"
    if exact.exists():
        return exact
    return _find_latest_matching(search_root, f"{allmarkets_path.stem}__BACKTEST.csv")


def _find_backtest_summary_csv(search_root: Path, allmarkets_path: Path) -> Optional[Path]:
    exact = search_root / f"{allmarkets_path.stem}__BACKTEST_SUMMARY.csv"
    if exact.exists():
        return exact
    return _find_latest_matching(search_root, f"{allmarkets_path.stem}__BACKTEST_SUMMARY.csv")


# -----------------------------
# additional file discovery helpers
# -----------------------------

def _find_backtest_unscored_csv(search_root: Path, allmarkets_path: Path) -> Optional[Path]:
    exact = search_root / f"{allmarkets_path.stem}__BACKTEST_UNSCORED.csv"
    if exact.exists():
        return exact
    return _find_latest_matching(search_root, f"{allmarkets_path.stem}__BACKTEST_UNSCORED.csv")


def _find_frozen_gated_csv(search_root: Path, backtest_path: Path) -> Optional[Path]:
    candidates = [
        search_root / f"{backtest_path.stem}__FROZEN_GATED.csv",
        search_root / f"{backtest_path.stem}__PRODUCT_GATES.csv",
        search_root / f"{backtest_path.stem}__FROZEN_PRODUCTS.csv",
        search_root / f"{backtest_path.stem}__FTR_ACCURACY.csv",
        search_root / f"{backtest_path.stem}__FTR_VALUE.csv",
        search_root / f"{backtest_path.stem}__FTR_VALUEEV_BALANCED.csv",
        search_root / f"{backtest_path.stem}__FTR_VALUEEV_AGGRESSIVE.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    for patt in (
        f"{backtest_path.stem}__FROZEN*.csv",
        f"{backtest_path.stem}__PRODUCT*.csv",
        f"{backtest_path.stem}__*GATES*.csv",
        f"{backtest_path.stem}__FTR_*.csv",
    ):
        hit = _find_latest_matching(search_root, patt)
        if hit is not None and "__MARKET_SUMMARY" not in hit.name and "__LEAGUE_SUMMARY" not in hit.name and "__SUMMARY" not in hit.name:
            return hit
    return None


def _find_frozen_gated_md(search_root: Path, backtest_path: Path) -> Optional[Path]:
    candidates = [
        search_root / f"{backtest_path.stem}__FROZEN_GATED.md",
        search_root / f"{backtest_path.stem}__PRODUCT_GATES.md",
        search_root / f"{backtest_path.stem}__FROZEN_PRODUCTS.md",
        search_root / f"{backtest_path.stem}__FTR_ACCURACY.md",
        search_root / f"{backtest_path.stem}__FTR_VALUE.md",
        search_root / f"{backtest_path.stem}__FTR_VALUEEV_BALANCED.md",
        search_root / f"{backtest_path.stem}__FTR_VALUEEV_AGGRESSIVE.md",
    ]
    for p in candidates:
        if p.exists():
            return p
    for patt in (
        f"{backtest_path.stem}__FROZEN*.md",
        f"{backtest_path.stem}__PRODUCT*.md",
        f"{backtest_path.stem}__*GATES*.md",
        f"{backtest_path.stem}__FTR_*.md",
    ):
        hit = _find_latest_matching(search_root, patt)
        if hit is not None and "__MARKET_SUMMARY" not in hit.name and "__LEAGUE_SUMMARY" not in hit.name and "__SUMMARY" not in hit.name:
            return hit
    return None


def _find_frozen_tier_csv(search_root: Path, backtest_path: Path, tier_name: str) -> Optional[Path]:
    tier = str(tier_name).strip().upper()
    candidates = [
        search_root / f"{backtest_path.stem}__FROZEN_TIER_{tier}.csv",
        search_root / f"{backtest_path.stem}__PRODUCT_TIER_{tier}.csv",
        search_root / f"{backtest_path.stem}__TIER_{tier}.csv",
        search_root / f"{backtest_path.stem}__FTR_ACCURACY__TIER_{tier}.csv",
        search_root / f"{backtest_path.stem}__FTR_VALUE__TIER_{tier}.csv",
        search_root / f"{backtest_path.stem}__FTR_VALUEEV_BALANCED__TIER_{tier}.csv",
        search_root / f"{backtest_path.stem}__FTR_VALUEEV_AGGRESSIVE__TIER_{tier}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    for patt in (
        f"{backtest_path.stem}__*TIER_{tier}.csv",
        f"{backtest_path.stem}__FROZEN*{tier}*.csv",
        f"{backtest_path.stem}__PRODUCT*{tier}*.csv",
        f"{backtest_path.stem}__FTR_*TIER_{tier}.csv",
    ):
        hit = _find_latest_matching(search_root, patt)
        if hit is not None:
            return hit
    return None




# -----------------------------
# summaries
# -----------------------------



# -----------------------------
# frozen profile helpers
# -----------------------------

FROZEN_PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "april2025": {
        "btts_max": 1.62,
        "ou25_band1_low": 1.24,
        "ou25_band1_high": 1.72,
        "ou25_band2_low": 1.82,
        "ou25_band2_high": 1.91,
        "ftr_value_min": 2.14,
        "ftr_accuracy_min_conf": 0.68,
        "ftr_accuracy_max_od": 1.85,
        "ftr_accuracy_min_margin": 0.06,
        "ftr_accuracy_home_away_only": True,
        "ftr_valueev_od_min": 1.80,
        "ftr_valueev_balanced_edge_min": 1.05,
        "ftr_valueev_aggressive_edge_min": 1.08,
        "top_q": 0.70,
        "tg_vig": 0.04,
        "tg15_cap": 0.93,
        "tg25_cap": 0.88,
        "tg_gamma": 1.35,
        "tg15_max_od": 2.50,
        "tg25_max_od": 6.00,
        "notes": {
            "profile_family": "april2025_frozen_products",
            "ftr_design": {
                "value": "Legacy/value-priced FTR selections (default bookie_od >= 2.14)",
                "accuracy": "Moderate favourites; separate product, not a mirror image of value",
                "valueev_balanced": "ValueEV Balanced = bookie_od >= 1.80 and edge >= 1.05",
                "valueev_aggressive": "ValueEV Aggressive = bookie_od >= 1.80 and edge >= 1.08",
                "accuracy_deadzone_note": "If ALL rows appear but MARKET::ftr is zero, that is an accuracy dead-zone, not a failed batch run."
            },
            "frozen_gates_note": "This runner now calls apply_frozen_product_gates.py for proper frozen OOS replay.",
        },
    }
}


def _load_profile(name: str) -> Dict[str, Any]:
    key = str(name or "april2025").strip().lower()
    if key not in FROZEN_PROFILE_DEFAULTS:
        raise SystemExit(f"Unknown frozen profile: {name}")
    return dict(FROZEN_PROFILE_DEFAULTS[key])


# -----------------------------
# markets helpers
# -----------------------------

def _parse_markets(markets_raw: str) -> List[str]:
    vals = [x.strip().lower() for x in str(markets_raw).split(",") if x.strip()]
    if not vals:
        raise SystemExit("No markets requested. Expected comma-separated values such as: ftr,ou25,btts,tg15,tg25")

    allowed = {"ftr", "ou25", "btts", "tg15", "tg25"}
    bad = [m for m in vals if m not in allowed]
    if bad:
        raise SystemExit(f"Unsupported markets requested: {bad}. Allowed: {sorted(allowed)}")

    deduped: List[str] = []
    seen = set()
    for m in vals:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def _market_counts(backtest_csv: Path) -> Dict[str, int]:
    df = pd.read_csv(backtest_csv, low_memory=False)
    if "market" not in df.columns:
        return {}
    s = df["market"].astype(str).str.lower().str.strip()
    vc = s.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in vc.items()}

# -----------------------------
# helper: should run frozen gates
def _should_run_frozen_gates(skip_frozen_gates: bool, requested_markets: List[str], ftr_profile: str) -> bool:
    if skip_frozen_gates:
        return False
    requested = [str(x).strip().lower() for x in requested_markets]
    if "ftr" not in requested:
        return False
    allowed_profiles = {"accuracy", "value", "valueev_balanced", "valueev_aggressive"}
    return str(ftr_profile).strip().lower() in allowed_profiles


# -----------------------------
# summaries
# -----------------------------

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _summarise_backtest(backtest_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(backtest_csv, low_memory=False)

    required = {"league", "market", "correct", "bookie_od"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"BACKTEST missing required columns: {missing}")

    d = df.copy()
    d["league"] = d["league"].astype(str)
    d["market"] = d["market"].astype(str).str.lower().str.strip()
    d["correct"] = _safe_num(d["correct"])
    d["bookie_od"] = _safe_num(d["bookie_od"])

    scored = d[d["correct"].notna()].copy()
    roi_df = scored[scored["bookie_od"].notna()].copy()

    total_rows = int(len(d))
    scored_rows = int(len(scored))
    unscored_rows = int(total_rows - scored_rows)

    def _roi(g: pd.DataFrame) -> float:
        if g.empty:
            return float("nan")
        return float((g["correct"] * g["bookie_od"] - 1.0).mean())

    def _src_breakdown(g: pd.DataFrame) -> Dict[str, int]:
        if "bookie_od_source" not in g.columns:
            return {}
        s = g["bookie_od_source"].astype(str).fillna("unknown").str.strip()
        s = s.replace({"": "unknown"})
        return {str(k): int(v) for k, v in s.value_counts(dropna=False).to_dict().items()}

    market_rows: List[Dict[str, object]] = []
    for market, g in scored.groupby("market", dropna=False):
        gr = roi_df[roi_df["market"] == market]
        src_counts = _src_breakdown(g)
        market_rows.append(
            {
                "market": market,
                "rows": int(len(g)),
                "hit": float(g["correct"].mean()) if len(g) else float("nan"),
                "avg_od": float(gr["bookie_od"].mean()) if len(gr) else float("nan"),
                "roi": _roi(gr),
                "n_bookie": int(src_counts.get("bookmaker", 0)),
                "n_model_fair": int(src_counts.get("model_fair", 0)),
                "n_unknown": int(sum(v for k, v in src_counts.items() if k not in {"bookmaker", "model_fair"})),
            }
        )
    market_summary = pd.DataFrame(market_rows).sort_values(["roi", "rows"], ascending=[False, False])

    league_rows: List[Dict[str, object]] = []
    for league, g in scored.groupby("league", dropna=False):
        gr = roi_df[roi_df["league"] == league]
        src_counts = _src_breakdown(g)
        league_rows.append(
            {
                "league": league,
                "rows": int(len(g)),
                "hit": float(g["correct"].mean()) if len(g) else float("nan"),
                "avg_od": float(gr["bookie_od"].mean()) if len(gr) else float("nan"),
                "roi": _roi(gr),
                "n_bookie": int(src_counts.get("bookmaker", 0)),
                "n_model_fair": int(src_counts.get("model_fair", 0)),
                "n_unknown": int(sum(v for k, v in src_counts.items() if k not in {"bookmaker", "model_fair"})),
            }
        )
    league_summary = pd.DataFrame(league_rows).sort_values(["roi", "rows"], ascending=[False, False])

    top_market = None
    worst_market = None
    if not market_summary.empty:
        top_market = str(market_summary.iloc[0]["market"])
        worst_market = str(market_summary.iloc[-1]["market"])

    top_league = None
    worst_league = None
    if not league_summary.empty:
        top_league = str(league_summary.iloc[0]["league"])
        worst_league = str(league_summary.iloc[-1]["league"])

    month_summary: Dict[str, object] = {
        "rows": int(len(scored)),
        "hit": float(scored["correct"].mean()) if len(scored) else float("nan"),
        "avg_od": float(roi_df["bookie_od"].mean()) if len(roi_df) else float("nan"),
        "roi": _roi(roi_df),
        "rows_total": total_rows,
        "rows_scored": scored_rows,
        "rows_unscored": unscored_rows,
        "markets": int(market_summary["market"].nunique()) if not market_summary.empty else 0,
        "leagues": int(league_summary["league"].nunique()) if not league_summary.empty else 0,
        "top_market": top_market,
        "worst_market": worst_market,
        "top_league": top_league,
        "worst_league": worst_league,
    }

    return market_summary, league_summary, month_summary




# -----------------------------
# main monthly runner
# -----------------------------

def _copy_if_exists(src: Optional[Path], dst: Path) -> None:
    if src is None or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")




# -----------------------------
# bookie/frozen-gates command helpers
# -----------------------------

def _build_bookie_cmd(
    bookie_cmd: str,
    *,
    date_from: str,
    date_to: str,
    leagues_clean: str,
    markets: str,
    matches_root: str,
    modelstore: str,
    strict: bool,
    implied_min: float,
    ftr_implied_min: Optional[float],
    ou25_implied_min: Optional[float],
    btts_implied_min: Optional[float],
    tg15_pmin: Optional[float],
    tg25_pmin: Optional[float],
) -> List[str]:
    cmd = shlex.split(bookie_cmd)
    cmd += [
        "--date-from", str(date_from),
        "--date-to", str(date_to),
        "--leagues", str(leagues_clean),
        "--markets", str(markets),
        "--matches-root", str(matches_root),
        "--modelstore", str(modelstore),
        "--implied-min", str(implied_min),
    ]
    if ftr_implied_min is not None:
        cmd += ["--ftr-implied-min", str(ftr_implied_min)]
    if ou25_implied_min is not None:
        cmd += ["--ou25-implied-min", str(ou25_implied_min)]
    if btts_implied_min is not None:
        cmd += ["--btts-implied-min", str(btts_implied_min)]
    if tg15_pmin is not None:
        cmd += ["--tg15-pmin", str(tg15_pmin)]
    if tg25_pmin is not None:
        cmd += ["--tg25-pmin", str(tg25_pmin)]
    if strict:
        cmd += ["--strict"]
    return cmd


def _build_frozen_gates_cmd(
    frozen_gates_cmd: str,
    *,
    src_csv: Path,
    outdir: Path,
    profile: Dict[str, Any],
    ftr_profile: str,
) -> List[str]:
    cmd = shlex.split(frozen_gates_cmd)
    cmd += [
        "--src", str(src_csv),
        "--outdir", str(outdir),
        "--ftr-profile", str(ftr_profile),
        "--btts-max", str(profile["btts_max"]),
        "--ou25-band1-low", str(profile["ou25_band1_low"]),
        "--ou25-band1-high", str(profile["ou25_band1_high"]),
        "--ou25-band2-low", str(profile["ou25_band2_low"]),
        "--ou25-band2-high", str(profile["ou25_band2_high"]),
        "--top-q", str(profile["top_q"]),
    ]

    ftr_profile_key = str(ftr_profile).strip().lower()
    if ftr_profile_key == "accuracy":
        cmd += ["--ftr-max-od", str(profile["ftr_accuracy_max_od"])]
        if bool(profile.get("ftr_accuracy_home_away_only", False)):
            cmd += ["--ftr-home-away-only"]
    elif ftr_profile_key == "value":
        cmd += ["--ftr-valueev-od-min", str(profile["ftr_value_min"])]
    elif ftr_profile_key == "valueev_balanced":
        cmd += [
            "--ftr-valueev-od-min", str(profile["ftr_valueev_od_min"]),
            "--ftr-valueev-edge-min", str(profile["ftr_valueev_balanced_edge_min"]),
        ]
    elif ftr_profile_key == "valueev_aggressive":
        cmd += [
            "--ftr-valueev-od-min", str(profile["ftr_valueev_od_min"]),
            "--ftr-valueev-edge-min", str(profile["ftr_valueev_aggressive_edge_min"]),
        ]
    else:
        raise SystemExit(f"Unsupported ftr_profile for frozen gates: {ftr_profile}")

    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen monthly walk-forward using existing production ModelStore only")
    ap.add_argument("--start-month", required=True, help="YYYY-MM")
    ap.add_argument("--end-month", required=True, help="YYYY-MM")

    ap.add_argument("--leagues", required=True, help="Comma-separated leagues")
    ap.add_argument("--markets", default="ftr,ou25,btts,tg15,tg25")

    ap.add_argument("--bookie-cmd", default="python3 bookie_allmarkets.py")
    ap.add_argument("--backtest-cmd", default="python backtest_deploy_csv.py")
    ap.add_argument("--frozen-gates-cmd", default="python apply_frozen_product_gates.py")

    ap.add_argument("--modelstore", default="ModelStore", help="Frozen production modelstore")
    ap.add_argument("--matches-root", default="Matches")
    ap.add_argument("--predictions-root", default="predictions_output")
    ap.add_argument("--archive-root", default="walkforward_frozen")
    ap.add_argument("--frozen-profile", default="april2025", help="Frozen learned gating profile name")
    ap.add_argument(
        "--ftr-profile",
        default="accuracy",
        choices=["accuracy", "value", "valueev_balanced", "valueev_aggressive"],
        help="FTR product profile to apply in frozen product gate stage",
    )
    ap.add_argument("--skip-frozen-gates", action="store_true", help="Skip frozen product gate stage and summarise raw backtest only")

    ap.add_argument("--implied-min", type=float, default=0.62)
    ap.add_argument("--ftr-implied-min", type=float, default=None)
    ap.add_argument("--ou25-implied-min", type=float, default=0.50)
    ap.add_argument("--btts-implied-min", type=float, default=0.50)
    ap.add_argument("--tg15-pmin", type=float, default=0.52)
    ap.add_argument("--tg25-pmin", type=float, default=0.35)

    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-bookie", action="store_true")
    ap.add_argument("--skip-backtest", action="store_true")

    args = ap.parse_args()
    requested_markets = _parse_markets(args.markets)
    requested_markets_csv = ",".join(requested_markets)
    run_ftr_frozen_gates = _should_run_frozen_gates(
        skip_frozen_gates=bool(args.skip_frozen_gates),
        requested_markets=requested_markets,
        ftr_profile=str(args.ftr_profile),
    )

    if "ftr" not in requested_markets:
        raise SystemExit(
            "Refusing to run without ftr in --markets. This protects the existing FTR walk-forward workflow. "
            "Use markets such as: ftr,ou25,btts,tg15,tg25"
        )

    frozen_profile = _load_profile(args.frozen_profile)

    months = _month_range(args.start_month, args.end_month)
    archive_root = Path(args.archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)

    predictions_root = Path(args.predictions_root)

    month_rows: List[Dict[str, object]] = []
    market_rows_all: List[pd.DataFrame] = []
    league_rows_all: List[pd.DataFrame] = []

    leagues_clean = ",".join([x.strip() for x in str(args.leagues).split(",") if x.strip()])
    print(f"REQUESTED MARKETS: {requested_markets_csv}")

    for m in months:
        month_tag = m.strftime("%Y-%m")
        date_from = _month_start(m).isoformat()
        date_to = _month_end(m).isoformat()
        outdir = archive_root / month_tag
        outdir.mkdir(parents=True, exist_ok=True)

        run_config_path = outdir / f"run_config_{month_tag}.json"

        print(f"\n=== FROZEN WALKFORWARD {month_tag} | {date_from} → {date_to} ===")

        # 1) bookie export using production ModelStore only
        bookie_cmd = _build_bookie_cmd(
            args.bookie_cmd,
            date_from=date_from,
            date_to=date_to,
            leagues_clean=leagues_clean,
            markets=requested_markets_csv,
            matches_root=str(args.matches_root),
            modelstore=str(args.modelstore),
            strict=bool(args.strict),
            implied_min=float(args.implied_min),
            ftr_implied_min=args.ftr_implied_min,
            ou25_implied_min=args.ou25_implied_min,
            btts_implied_min=args.btts_implied_min,
            tg15_pmin=args.tg15_pmin,
            tg25_pmin=args.tg25_pmin,
        )

        backtest_cmd_preview = shlex.split(args.backtest_cmd) + [
            "--deploy-csv", f"<ALLMARKETS:{date_from}_to_{date_to}>",
            "--matches-root", str(args.matches_root),
            "--outdir", str(outdir),
        ]

        frozen_gates_cmd_preview = None
        if run_ftr_frozen_gates:
            frozen_gates_cmd_preview = _build_frozen_gates_cmd(
                args.frozen_gates_cmd,
                src_csv=Path(f"<BACKTEST:{date_from}_to_{date_to}>"),
                outdir=outdir,
                profile=frozen_profile,
                ftr_profile=str(args.ftr_profile),
            )

        _write_json(
            run_config_path,
            {
                "month": month_tag,
                "date_from": date_from,
                "date_to": date_to,
                "leagues": [x.strip() for x in leagues_clean.split(",") if x.strip()],
                "markets": requested_markets,
                "markets_csv": requested_markets_csv,
                "modelstore": str(args.modelstore),
                "matches_root": str(args.matches_root),
                "predictions_root": str(args.predictions_root),
                "bookie_cmd": bookie_cmd,
                "backtest_cmd": backtest_cmd_preview,
                "frozen_gates_cmd": frozen_gates_cmd_preview,
                "frozen_profile_name": str(args.frozen_profile),
                "frozen_profile": frozen_profile,
                "ftr_profile": str(args.ftr_profile),
                "run_ftr_frozen_gates": bool(run_ftr_frozen_gates),
                "runner_purpose": "Generate canonical monthly scored walk-forward backtests with FTR preserved and OU25 included when requested.",
                "guardrails": {
                    "ftr_required_in_markets": True,
                    "ou25_gating_delegated_to_ou25_runner": True,
                    "frozen_gates_applied_here_only_for_ftr_profiles": True,
                },
                "bookie_flag_validation": {
                    "validated_against_bookie_allmarkets_argparse": True,
                    "validated_flags": [
                        "--date-from",
                        "--date-to",
                        "--leagues",
                        "--markets",
                        "--matches-root",
                        "--modelstore",
                        "--implied-min",
                        "--ftr-implied-min",
                        "--ou25-implied-min",
                        "--btts-implied-min",
                        "--tg15-pmin",
                        "--tg25-pmin",
                        "--strict",
                    ],
                },
                "frozen_gates_flag_validation": {
                    "validated_for_apply_frozen_product_gates": True,
                    "expected_flags": [
                        "--src",
                        "--outdir",
                        "--ftr-profile",
                        "--ftr-max-od",
                        "--ftr-home-away-only",
                        "--ftr-valueev-edge-min",
                        "--ftr-valueev-od-min",
                        "--ou25-band1-low",
                        "--ou25-band1-high",
                        "--ou25-band2-low",
                        "--ou25-band2-high",
                        "--btts-max",
                        "--top-q",
                    ],
                },
            },
        )

        if not args.skip_bookie:
            _run(bookie_cmd, dry_run=args.dry_run)

        if args.dry_run:
            continue

        allmarkets_csv = _find_allmarkets_csv(predictions_root, date_from, date_to)
        if allmarkets_csv is None:
            raise SystemExit(f"Could not locate ALLMARKETS csv for {month_tag} under {predictions_root}")

        # 2) backtest
        if not args.skip_backtest:
            cmd = shlex.split(args.backtest_cmd)
            cmd += [
                "--deploy-csv", str(allmarkets_csv),
                "--matches-root", str(args.matches_root),
                "--outdir", str(outdir),
            ]
            _run(cmd, dry_run=args.dry_run)

        backtest_csv = _find_backtest_csv(outdir, allmarkets_csv)
        if backtest_csv is None:
            backtest_csv = _find_backtest_csv(predictions_root, allmarkets_csv)
        if backtest_csv is None:
            raise SystemExit(f"Could not locate BACKTEST csv for {month_tag}")

        market_counts = _market_counts(backtest_csv)
        print(f"BACKTEST MARKET COUNTS: {market_counts}")

        if "ftr" not in market_counts:
            raise SystemExit(
                f"Generated backtest for {month_tag} does not contain ftr rows. "
                f"This would risk the existing FTR walk-forward workflow. Counts: {market_counts}"
            )

        if "ou25" in requested_markets and "ou25" not in market_counts:
            print(
                f"WARNING: OU25 was requested but no ou25 rows were found in the generated backtest for {month_tag}. "
                f"Counts: {market_counts}"
            )

        backtest_summary_csv = _find_backtest_summary_csv(outdir, allmarkets_csv)
        if backtest_summary_csv is None:
            backtest_summary_csv = _find_backtest_summary_csv(predictions_root, allmarkets_csv)

        backtest_unscored_csv = _find_backtest_unscored_csv(outdir, allmarkets_csv)
        if backtest_unscored_csv is None:
            backtest_unscored_csv = _find_backtest_unscored_csv(predictions_root, allmarkets_csv)

        frozen_gated_csv: Optional[Path] = None
        frozen_gated_md: Optional[Path] = None
        frozen_tier_elite_csv: Optional[Path] = None
        frozen_tier_standard_csv: Optional[Path] = None
        frozen_tier_observe_csv: Optional[Path] = None

        if run_ftr_frozen_gates:
            frozen_gates_cmd = _build_frozen_gates_cmd(
                args.frozen_gates_cmd,
                src_csv=backtest_csv,
                outdir=outdir,
                profile=frozen_profile,
                ftr_profile=str(args.ftr_profile),
            )
            _run(frozen_gates_cmd, dry_run=args.dry_run)
            frozen_gated_csv = _find_frozen_gated_csv(outdir, backtest_csv)
            frozen_gated_md = _find_frozen_gated_md(outdir, backtest_csv)
            frozen_tier_elite_csv = _find_frozen_tier_csv(outdir, backtest_csv, "ELITE")
            frozen_tier_standard_csv = _find_frozen_tier_csv(outdir, backtest_csv, "STANDARD")
            frozen_tier_observe_csv = _find_frozen_tier_csv(outdir, backtest_csv, "OBSERVE")

            try:
                cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
            cfg["backtest_cmd"] = shlex.split(args.backtest_cmd) + [
                "--deploy-csv", str(allmarkets_csv),
                "--matches-root", str(args.matches_root),
                "--outdir", str(outdir),
            ]
            cfg["frozen_gates_cmd"] = frozen_gates_cmd
            cfg["resolved_artifacts"] = {
                "allmarkets_csv": str(allmarkets_csv),
                "backtest_csv": str(backtest_csv),
                "frozen_gated_csv": str(frozen_gated_csv) if frozen_gated_csv else None,
                "frozen_gated_md": str(frozen_gated_md) if frozen_gated_md else None,
                "frozen_tier_elite_csv": str(frozen_tier_elite_csv) if frozen_tier_elite_csv else None,
                "frozen_tier_standard_csv": str(frozen_tier_standard_csv) if frozen_tier_standard_csv else None,
                "frozen_tier_observe_csv": str(frozen_tier_observe_csv) if frozen_tier_observe_csv else None,
            }
            _write_json(run_config_path, cfg)
        else:
            try:
                cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
            cfg["frozen_gates_cmd"] = None
            cfg["resolved_artifacts"] = {
                "allmarkets_csv": str(allmarkets_csv),
                "backtest_csv": str(backtest_csv),
                "frozen_gated_csv": None,
                "frozen_gated_md": None,
                "frozen_tier_elite_csv": None,
                "frozen_tier_standard_csv": None,
                "frozen_tier_observe_csv": None,
            }
            _write_json(run_config_path, cfg)

        summary_source_csv = frozen_gated_csv if frozen_gated_csv is not None else backtest_csv

        # 3) archive raw core files with stable names
        raw_predictions_dst = outdir / f"raw_predictions_{month_tag}.csv"
        backtest_dst = outdir / f"backtest_{month_tag}.csv"
        backtest_unscored_dst = outdir / f"backtest_unscored_{month_tag}.csv"
        backtest_summary_dst = outdir / f"backtest_summary_{month_tag}.csv"
        frozen_gated_dst = outdir / f"frozen_gated_{month_tag}.csv"
        frozen_gated_md_dst = outdir / f"frozen_gated_{month_tag}.md"
        frozen_tier_elite_dst = outdir / f"frozen_tier_elite_{month_tag}.csv"
        frozen_tier_standard_dst = outdir / f"frozen_tier_standard_{month_tag}.csv"
        frozen_tier_observe_dst = outdir / f"frozen_tier_observe_{month_tag}.csv"

        _copy_if_exists(allmarkets_csv, raw_predictions_dst)
        _copy_if_exists(backtest_csv, backtest_dst)
        _copy_if_exists(backtest_summary_csv, backtest_summary_dst)
        _copy_if_exists(backtest_unscored_csv, backtest_unscored_dst)
        _copy_if_exists(frozen_gated_csv, frozen_gated_dst)
        _copy_if_exists(frozen_gated_md, frozen_gated_md_dst)
        _copy_if_exists(frozen_tier_elite_csv, frozen_tier_elite_dst)
        _copy_if_exists(frozen_tier_standard_csv, frozen_tier_standard_dst)
        _copy_if_exists(frozen_tier_observe_csv, frozen_tier_observe_dst)

        # 4) summaries
        market_summary, league_summary, month_summary = _summarise_backtest(summary_source_csv)
        market_summary.insert(0, "month", month_tag)
        league_summary.insert(0, "month", month_tag)
        month_summary["month"] = month_tag
        month_summary["date_from"] = date_from
        month_summary["date_to"] = date_to
        month_summary["modelstore"] = str(args.modelstore)
        month_summary["frozen_profile_name"] = str(args.frozen_profile)
        month_summary["frozen_profile"] = frozen_profile
        month_summary["ftr_profile"] = str(args.ftr_profile)
        month_summary["requested_markets"] = requested_markets
        month_summary["backtest_market_counts"] = market_counts
        month_summary["summary_source_csv"] = str(summary_source_csv)
        month_summary["frozen_gated_csv"] = str(frozen_gated_dst) if frozen_gated_csv is not None else None
        month_summary["frozen_gated_md"] = str(frozen_gated_md_dst) if frozen_gated_md is not None else None
        month_summary["frozen_tier_elite_csv"] = str(frozen_tier_elite_dst) if frozen_tier_elite_csv is not None else None
        month_summary["frozen_tier_standard_csv"] = str(frozen_tier_standard_dst) if frozen_tier_standard_csv is not None else None
        month_summary["frozen_tier_observe_csv"] = str(frozen_tier_observe_dst) if frozen_tier_observe_csv is not None else None
        month_summary["raw_predictions_csv"] = str(raw_predictions_dst)
        month_summary["backtest_csv"] = str(backtest_dst)
        month_summary["backtest_unscored_csv"] = str(backtest_unscored_dst) if backtest_unscored_csv is not None else None
        month_summary["backtest_summary_csv"] = str(backtest_summary_dst) if backtest_summary_csv is not None else None
        month_summary["run_config_json"] = str(run_config_path)

        market_summary_path = outdir / f"market_summary_{month_tag}.csv"
        league_summary_path = outdir / f"league_summary_{month_tag}.csv"
        month_summary_path = outdir / f"summary_{month_tag}.json"

        market_summary.to_csv(market_summary_path, index=False)
        league_summary.to_csv(league_summary_path, index=False)
        _write_json(month_summary_path, month_summary)

        month_rows.append(month_summary)
        market_rows_all.append(market_summary)
        league_rows_all.append(league_summary)

        if run_ftr_frozen_gates:
            print(f"FROZEN GATES MODE: apply_frozen_product_gates.py | ftr_profile={args.ftr_profile}")
        else:
            print("FROZEN GATES MODE: skipped in run_frozen_walkforward.py (OU25 gating delegated to OU25 runner)")
        print(f"WROTE: {raw_predictions_dst}")
        print(f"WROTE: {backtest_dst}")
        if backtest_unscored_csv is not None:
            print(f"WROTE: {backtest_unscored_dst}")
        if backtest_summary_csv is not None:
            print(f"WROTE: {backtest_summary_dst}")
        if frozen_gated_csv is not None:
            print(f"WROTE: {frozen_gated_dst}")
        if frozen_gated_md is not None:
            print(f"WROTE: {frozen_gated_md_dst}")
        if frozen_tier_elite_csv is not None:
            print(f"WROTE: {frozen_tier_elite_dst}")
        if frozen_tier_standard_csv is not None:
            print(f"WROTE: {frozen_tier_standard_dst}")
        if frozen_tier_observe_csv is not None:
            print(f"WROTE: {frozen_tier_observe_dst}")
        print(f"WROTE: {market_summary_path}")
        print(f"WROTE: {league_summary_path}")
        print(f"WROTE: {month_summary_path}")

    if args.dry_run:
        return

    # 5) aggregate all months
    summaries_dir = archive_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    if month_rows:
        month_df = pd.DataFrame(month_rows)
        month_df.to_csv(summaries_dir / "all_months_summary.csv", index=False)
        print(f"WROTE: {summaries_dir / 'all_months_summary.csv'}")

    if market_rows_all:
        market_all = pd.concat(market_rows_all, axis=0, ignore_index=True)
        market_all.to_csv(summaries_dir / "by_market_all_months.csv", index=False)
        print(f"WROTE: {summaries_dir / 'by_market_all_months.csv'}")

    if league_rows_all:
        league_all = pd.concat(league_rows_all, axis=0, ignore_index=True)
        league_all.to_csv(summaries_dir / "by_league_all_months.csv", index=False)
        print(f"WROTE: {summaries_dir / 'by_league_all_months.csv'}")


if __name__ == "__main__":
    main()