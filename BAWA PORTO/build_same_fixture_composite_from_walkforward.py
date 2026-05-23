#!/usr/bin/env python3
"""Audit same-fixture composite pairs from pre-dedup walk-forward candidate tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

WINDOW_PAT = re.compile(r"w\d+_(\d{4})_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build same-fixture composite audit from pre-dedup walk-forward tables")
    p.add_argument("--walkforward-root", required=True, help="Walk-forward root containing w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv")
    p.add_argument("--outdir", required=True, help="Directory for composite outputs")
    p.add_argument("--baseline-audit-dir", help="Optional slip walk-forward audit dir for monster baseline comparison")
    p.add_argument("--top-pairs-per-window", type=int, default=12)
    return p.parse_args()


def parse_window_date(window_id: str) -> pd.Timestamp:
    m = WINDOW_PAT.match(str(window_id))
    if not m:
        return pd.NaT
    return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))


def season_phase(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "UNKNOWN"
    m = int(ts.month)
    if m in {8, 9}:
        return "EARLY_SEASON"
    if m in {10, 11, 12}:
        return "AUTUMN_DENSE"
    if m in {1, 2}:
        return "WINTER_RESET"
    if m in {3, 4, 5}:
        return "RUN_IN"
    return "SUMMER_TRANSITION"


def champions_league_phase_flag(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return 0
    return int(ts.month in {9, 10, 11, 12, 2, 3, 4})


def international_break_zone_flag(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return 0
    return int(ts.month in {3, 9, 10, 11})


def nm(v: object) -> str:
    return str(v or "").strip().lower()


def ns(v: object) -> str:
    return str(v or "").strip().upper()


def classify_pair(a: pd.Series, b: pd.Series) -> tuple[str, str] | None:
    ma, mb = nm(a.get("market")), nm(b.get("market"))
    markets = {ma, mb}
    if markets == {"ftr", "ou25"}:
        ftr = a if ma == "ftr" else b
        ou = a if ma == "ou25" else b
        ou_sel = ns(ou.get("selection"))
        if ou_sel in {"OVER25", "UNDER25"}:
            return f"FTR_PLUS_{ou_sel}", f"{ns(ftr.get('selection'))}+{ou_sel}"
    if markets == {"ftr", "btts"}:
        ftr = a if ma == "ftr" else b
        btts = a if ma == "btts" else b
        return "FTR_PLUS_BTTS", f"{ns(ftr.get('selection'))}+BTTS_{ns(btts.get('selection'))}"
    if "ftr" in markets and any(m in {"home_goals", "away_goals", "team_goals", "tg15", "team_goals_15"} for m in markets):
        ftr = a if ma == "ftr" else b
        tg = a if ma != "ftr" else b
        tg_sel = ns(tg.get("selection"))
        if "1.5" in tg_sel or tg_sel in {"OVER15", "OVER_1_5"}:
            return "FTR_PLUS_TEAM_GOALS15", f"{ns(ftr.get('selection'))}+{tg_sel}"
    return None


def pair_score(a: pd.Series, b: pd.Series) -> float:
    a_score = pd.to_numeric(
        a.get("score", a.get("meta_super_score", a.get("model_p_for_bookie", 0.0))),
        errors="coerce",
    )
    b_score = pd.to_numeric(
        b.get("score", b.get("meta_super_score", b.get("model_p_for_bookie", 0.0))),
        errors="coerce",
    )
    return float(0.0 if pd.isna(a_score) else a_score) + float(0.0 if pd.isna(b_score) else b_score)


def leg_odds_value(row: pd.Series) -> float | None:
    market = nm(row.get("market"))
    selection = ns(row.get("selection"))
    candidates: list[object] = [row.get("bookie_od")]
    if market == "ftr":
        if selection == "HOME":
            candidates.insert(0, row.get("od_home"))
        elif selection == "DRAW":
            candidates.insert(0, row.get("od_draw"))
        elif selection == "AWAY":
            candidates.insert(0, row.get("od_away"))
    elif market == "btts":
        if selection == "YES":
            candidates.insert(0, row.get("od_yes"))
        elif selection == "NO":
            candidates.insert(0, row.get("od_no"))
    elif market == "ou25":
        if selection == "OVER25":
            candidates.insert(0, row.get("odds_ft_over25"))
        elif selection == "UNDER25":
            candidates.insert(0, row.get("odds_ft_under25"))

    for cand in candidates:
        val = pd.to_numeric(cand, errors="coerce")
        if pd.notna(val) and float(val) > 1.0:
            return float(val)
    return None


def leg_hit_value(row: pd.Series) -> float | None:
    market = nm(row.get("market"))
    selection = ns(row.get("selection"))
    if market == "ftr":
        return pd.to_numeric(row.get("ftr_hit"), errors="coerce")
    if market == "ou25":
        return pd.to_numeric(row.get("ou25_hit"), errors="coerce")
    if market == "btts":
        if selection == "YES":
            return pd.to_numeric(row.get("btts_yes_hit"), errors="coerce")
        if selection == "NO":
            return pd.to_numeric(row.get("btts_no_hit"), errors="coerce")
    return None


def pair_row(window_id: str, a: pd.Series, b: pd.Series, pair_family: str, pair_label: str) -> dict:
    ts = parse_window_date(window_id)
    ahit = leg_hit_value(a)
    bhit = leg_hit_value(b)
    aodds = leg_odds_value(a)
    bodds = leg_odds_value(b)
    both_graded = int(pd.notna(ahit) and pd.notna(bhit))
    both_hit = int(both_graded and ahit == 1 and bhit == 1)
    landed_legs = int((0 if pd.isna(ahit) else ahit) + (0 if pd.isna(bhit) else bhit))
    return {
        "window_id": window_id,
        "window_date_from": ts.date().isoformat() if not pd.isna(ts) else "",
        "window_month": int(ts.month) if not pd.isna(ts) else pd.NA,
        "window_year": int(ts.year) if not pd.isna(ts) else pd.NA,
        "season_phase": season_phase(ts),
        "champions_league_phase_flag": champions_league_phase_flag(ts),
        "international_break_zone_flag": international_break_zone_flag(ts),
        "fixture_key": a.get("fixture_key", ""),
        "league": a.get("league", ""),
        "home": a.get("home", ""),
        "away": a.get("away", ""),
        "pair_family": pair_family,
        "pair_label": pair_label,
        "leg1_market": a.get("market", ""),
        "leg1_selection": a.get("selection", ""),
        "leg1_rank": pd.to_numeric(a.get("candidate_rank", pd.NA), errors="coerce"),
        "leg1_score": pd.to_numeric(a.get("score", a.get("meta_super_score", pd.NA)), errors="coerce"),
        "leg1_bucket": a.get("standard_reporting_bucket", ""),
        "leg1_odds": aodds,
        "leg2_market": b.get("market", ""),
        "leg2_selection": b.get("selection", ""),
        "leg2_rank": pd.to_numeric(b.get("candidate_rank", pd.NA), errors="coerce"),
        "leg2_score": pd.to_numeric(b.get("score", b.get("meta_super_score", pd.NA)), errors="coerce"),
        "leg2_bucket": b.get("standard_reporting_bucket", ""),
        "leg2_odds": bodds,
        "pair_score": pair_score(a, b),
        "pair_odds_product": (aodds * bodds) if (aodds is not None and bodds is not None) else pd.NA,
        "pair_caution_count": int(pd.to_numeric(a.get("slip_leg_caution_flag", 0), errors="coerce") == 1) + int(pd.to_numeric(b.get("slip_leg_caution_flag", 0), errors="coerce") == 1),
        "pair_monster_caution_count": int(pd.to_numeric(a.get("monster_caution_flag", 0), errors="coerce") == 1) + int(pd.to_numeric(b.get("monster_caution_flag", 0), errors="coerce") == 1),
        "pair_monster_avoid_count": int(pd.to_numeric(a.get("avoid_in_monster_acca_flag", 0), errors="coerce") == 1) + int(pd.to_numeric(b.get("avoid_in_monster_acca_flag", 0), errors="coerce") == 1),
        "both_graded": both_graded,
        "both_hit": both_hit,
        "landed_legs": landed_legs,
    }


def load_candidate_tables(root: Path) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for window_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith('w')]):
        scored_files = sorted((window_dir / '03_scored').glob('DEPLOY_COMBINED_SCORED_*.csv'))
        if not scored_files:
            continue
        df = pd.read_csv(scored_files[-1], low_memory=False)
        if df.empty or 'fixture_key' not in df.columns:
            continue
        out.append((window_dir.name, df))
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.walkforward_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tables = load_candidate_tables(root)
    if not tables:
        raise SystemExit(f'No DEPLOY_COMBINED_SCORED files found under {root}')

    detail_rows = []
    best_frames = []
    for window_id, df in tables:
        pair_rows = []
        for fixture_key, grp in df.groupby('fixture_key', dropna=False):
            if pd.isna(fixture_key) or len(grp) < 2:
                continue
            score_col = 'score' if 'score' in grp.columns else 'meta_super_score' if 'meta_super_score' in grp.columns else 'model_p_for_bookie'
            grp = grp.sort_values([score_col, 'candidate_rank'], ascending=[False, True])
            recs = [row for _, row in grp.iterrows()]
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    a, b = recs[i], recs[j]
                    if nm(a.get('market')) == nm(b.get('market')):
                        continue
                    meta = classify_pair(a, b)
                    if meta is None:
                        continue
                    pair_family, pair_label = meta
                    pair_rows.append(pair_row(window_id, a, b, pair_family, pair_label))
        if not pair_rows:
            continue
        pair_df = pd.DataFrame(pair_rows).sort_values(['pair_family', 'pair_score'], ascending=[True, False])
        detail_rows.extend(pair_df.to_dict('records'))
        best_frames.append(pair_df.groupby('pair_family', dropna=False).head(args.top_pairs_per_window))

    if not detail_rows:
        raise SystemExit('No supported same-fixture composite pairs found in DEPLOY_COMBINED_SCORED tables.')

    detail = pd.DataFrame(detail_rows)
    best = pd.concat(best_frames, ignore_index=True) if best_frames else detail.copy()

    detail_path = outdir / 'SAME_FIXTURE_COMPOSITES__DETAIL.csv'
    best_path = outdir / 'SAME_FIXTURE_COMPOSITES__BEST_PER_WINDOW.csv'
    summary_path = outdir / 'SAME_FIXTURE_COMPOSITES__SUMMARY.csv'
    timing_path = outdir / 'SAME_FIXTURE_COMPOSITES__TIMING.csv'
    compare_path = outdir / 'SAME_FIXTURE_COMPOSITES__VS_MONSTER_BASELINE.csv'

    detail.to_csv(detail_path, index=False)
    best.to_csv(best_path, index=False)

    summary = (
        detail.groupby(['pair_family', 'pair_label'], dropna=False)
        .agg(
            rows=('fixture_key', 'size'),
            windows=('window_id', 'nunique'),
            both_graded_rate=('both_graded', 'mean'),
            pair_hit_rate=('both_hit', 'mean'),
            mean_landed_legs=('landed_legs', 'mean'),
            mean_pair_score=('pair_score', 'mean'),
            mean_pair_odds_product=('pair_odds_product', 'mean'),
            mean_monster_caution_count=('pair_monster_caution_count', 'mean'),
            any_monster_caution_rate=('pair_monster_caution_count', lambda s: (pd.to_numeric(s, errors='coerce').fillna(0) > 0).mean()),
        )
        .reset_index()
        .sort_values(['pair_hit_rate', 'rows'], ascending=[False, False])
    )
    summary.to_csv(summary_path, index=False)

    timing = (
        detail.groupby(['pair_family', 'season_phase', 'window_month', 'champions_league_phase_flag', 'international_break_zone_flag'], dropna=False)
        .agg(
            rows=('fixture_key', 'size'),
            pair_hit_rate=('both_hit', 'mean'),
            mean_landed_legs=('landed_legs', 'mean'),
            mean_pair_score=('pair_score', 'mean'),
            mean_monster_caution_count=('pair_monster_caution_count', 'mean'),
        )
        .reset_index()
        .sort_values(['rows', 'pair_hit_rate'], ascending=[False, False])
    )
    timing.to_csv(timing_path, index=False)

    baseline = pd.DataFrame()
    if args.baseline_audit_dir:
        survival_path = Path(args.baseline_audit_dir) / 'SLIP_WALKFORWARD__SURVIVAL_BY_SIZE.csv'
        if survival_path.exists():
            survival = pd.read_csv(survival_path)
            baseline = survival[survival['build_mode'].astype(str).eq('constructed')].copy()
            if not baseline.empty:
                baseline['monster_mean_leg_hit_rate_when_built'] = baseline['mean_legs_landed_when_built'] / baseline['slip_size']
                baseline = baseline[['slip_size', 'complete_slip_rate_when_built', 'monster_mean_leg_hit_rate_when_built']]
    if baseline.empty:
        compare = summary.copy()
    else:
        rows = []
        for _, srow in summary.iterrows():
            for _, brow in baseline.iterrows():
                rows.append({
                    **srow.to_dict(),
                    'baseline_slip_size': int(brow['slip_size']),
                    'monster_complete_slip_rate_when_built': brow['complete_slip_rate_when_built'],
                    'monster_mean_leg_hit_rate_when_built': brow['monster_mean_leg_hit_rate_when_built'],
                    'pair_hit_minus_monster_complete': srow['pair_hit_rate'] - brow['complete_slip_rate_when_built'],
                    'pair_hit_minus_monster_leg_hit': srow['pair_hit_rate'] - brow['monster_mean_leg_hit_rate_when_built'],
                })
        compare = pd.DataFrame(rows)
    compare.to_csv(compare_path, index=False)

    print('WROTE:')
    print(detail_path)
    print(best_path)
    print(summary_path)
    print(timing_path)
    print(compare_path)
    print('\nPAIR SUMMARY\n')
    print(summary.head(20).to_string(index=False))
    print('\nTIMING\n')
    print(timing.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
