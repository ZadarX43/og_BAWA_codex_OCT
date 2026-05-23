#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import pandas as pd


KEY_COLS = ["league", "fixture_key", "market", "bookie_pick"]
OUTCOME_JOIN_COLS = ["league", "fixture_key", "market"]
EXACT_JOIN_COLS = ["league", "fixture_key", "market", "bookie_pick"]
DEFAULT_SHADOW = "predictions_output/BTTS_TAG_TEST/BTTS_SHADOW_MICRO_BUCKET_TEST_B.csv"
DEFAULT_LIVE = "predictions_output/BTTS_TAG_TEST/BTTS_LIVE_WHITELIST_BASELINE.csv"
DEFAULT_OUT = "predictions_output/BTTS_TAG_TEST/BTTS_SHADOW_AUDIT_SUMMARY.md"


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")



def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return pd.read_csv(p)


def _series_str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="string")
    return df[col].astype("string").fillna(default).str.strip()




def _canonical_market(df: pd.DataFrame) -> pd.Series:
    market = _series_str(df, "market").str.lower()
    return market


# Helper: market counts
def _market_counts(df: pd.DataFrame) -> dict[str, int]:
    market = _series_str(df, "market").str.lower()
    vc = market.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


# Extract date from fixture_key
def _date_from_fixture_key(df: pd.DataFrame) -> pd.Series:
    fk = _series_str(df, "fixture_key")
    return fk.str.extract(r"^(\d{4}_\d{2}_\d{2})", expand=False).astype("string")



def _canonical_bookie_pick(df: pd.DataFrame) -> pd.Series:
    pick = _series_str(df, "bookie_pick")
    sel = _series_str(df, "selection")
    out = pick.where(pick.ne(""), sel)
    return out.str.upper().str.strip()



def _make_join_frame(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["league", "fixture_key", "market"], "input dataframe")
    out = df.copy()
    out["_join_league"] = _series_str(out, "league")
    out["_join_fixture_key"] = _series_str(out, "fixture_key")
    out["_join_market"] = _canonical_market(out)
    out["_join_bookie_pick"] = _canonical_bookie_pick(out)
    out["_k"] = (
        out["_join_league"]
        + "|"
        + out["_join_fixture_key"]
        + "|"
        + out["_join_market"]
        + "|"
        + out["_join_bookie_pick"]
    ).astype("string")
    return out



def _make_outcome_join_frame(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, OUTCOME_JOIN_COLS, "input dataframe")
    out = df.copy()
    out["_outcome_league"] = _series_str(out, "league")
    out["_outcome_fixture_key"] = _series_str(out, "fixture_key")
    out["_outcome_market"] = _canonical_market(out)
    out["_outcome_k"] = (
        out["_outcome_league"]
        + "|"
        + out["_outcome_fixture_key"]
        + "|"
        + out["_outcome_market"]
    ).astype("string")
    return out


def _make_key(df: pd.DataFrame) -> pd.Series:
    joined = _make_join_frame(df)
    return joined["_k"]


def _make_outcome_key(df: pd.DataFrame) -> pd.Series:
    joined = _make_outcome_join_frame(df)
    return joined["_outcome_k"]



def _series_upper(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="string")
    return df[col].astype("string").fillna(default).str.upper().str.strip()



def _safe_win_series(df: pd.DataFrame) -> pd.Series:
    if "win" not in df.columns:
        return pd.Series(False, index=df.index, dtype="boolean")
    s = df["win"]
    if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
        return s.astype("boolean").fillna(False)
    s = s.astype("string").str.upper().str.strip()
    return s.map({"TRUE": True, "FALSE": False}).astype("boolean").fillna(False)


def _bucket_metrics(df: pd.DataFrame, label: str) -> dict:
    win_s = _safe_win_series(df)
    resolved_s = df["actual_btts"].notna() if "actual_btts" in df.columns else pd.Series(False, index=df.index)

    resolved_win_s = win_s.loc[resolved_s] if len(win_s) else pd.Series(dtype="boolean")

    out: dict[str, object] = {
        "label": label,
        "rows": int(len(df)),
        "resolved": int(resolved_s.sum()),
        "wins": int(resolved_win_s.fillna(False).sum()) if len(resolved_win_s) else 0,
        "losses": int((~resolved_win_s.fillna(False)).sum()) if len(resolved_win_s) else 0,
    }

    if int(out["resolved"]) > 0 and "actual_btts" in df.columns:
        out["hit_rate"] = float(win_s.loc[resolved_s].mean())
    else:
        out["hit_rate"] = None

    if "selection" in df.columns:
        out["selection_mix"] = df["selection"].astype("string").value_counts(dropna=False).to_dict()
    else:
        out["selection_mix"] = {}

    if "signal_btts_runtime" in df.columns:
        out["runtime_mix"] = df["signal_btts_runtime"].astype("string").value_counts(dropna=False).to_dict()
    else:
        out["runtime_mix"] = {}

    for col in ["edge", "model_p_for_bookie"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                out[f"{col}_stats"] = {
                    "count": int(s.count()),
                    "min": float(s.min()),
                    "median": float(s.median()),
                    "mean": float(s.mean()),
                    "max": float(s.max()),
                }
            else:
                out[f"{col}_stats"] = {}
        else:
            out[f"{col}_stats"] = {}

    return out



def _fmt_pct(x: object) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "n/a"



def _fmt_num(x: object) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "n/a"



def _dict_lines(title: str, d: dict) -> list[str]:
    lines = [f"### {title}"]
    if not d:
        lines.append("- none")
        return lines
    for k, v in d.items():
        lines.append(f"- {k}: {v}")
    return lines



def _stats_lines(title: str, d: dict) -> list[str]:
    lines = [f"### {title}"]
    if not d:
        lines.append("- n/a")
        return lines
    lines.append(f"- count: {d.get('count', 'n/a')}")
    lines.append(f"- min: {_fmt_num(d.get('min'))}")
    lines.append(f"- median: {_fmt_num(d.get('median'))}")
    lines.append(f"- mean: {_fmt_num(d.get('mean'))}")
    lines.append(f"- max: {_fmt_num(d.get('max'))}")
    return lines


# Coverage diagnostics lines
def _coverage_lines(title: str, d: dict) -> list[str]:
    lines = [f"### {title}"]
    if not d:
        lines.append("- n/a")
        return lines

    for k in [
        "shadow_rows",
        "live_rows",
        "results_rows",
        "shadow_fixture_overlap",
        "live_fixture_overlap",
        "shadow_league_overlap",
        "live_league_overlap",
        "shadow_date_overlap",
        "live_date_overlap",
    ]:
        if k in d:
            lines.append(f"- {k}: {d[k]}")

    if d.get("shadow_missing_fixture_examples"):
        lines.append("- shadow_missing_fixture_examples:")
        for x in d["shadow_missing_fixture_examples"]:
            lines.append(f"  - {x}")

    if d.get("live_missing_fixture_examples"):
        lines.append("- live_missing_fixture_examples:")
        for x in d["live_missing_fixture_examples"]:
            lines.append(f"  - {x}")

    return lines



def _top_rows(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    sort_col = "edge" if "edge" in df.columns else None
    if sort_col:
        return df.sort_values(sort_col, ascending=False).head(n).copy()
    return df.head(n).copy()



def _rows_to_markdown(df: pd.DataFrame, title: str) -> list[str]:
    lines = [f"### {title}"]
    if df.empty:
        lines.append("No rows.")
        return lines

    cols = [
        c
        for c in [
            "league",
            "home_team_name",
            "away_team_name",
            "selection",
            "model_top_pick",
            "actual_btts",
            "win",
            "signal_btts_runtime",
            "edge",
            "model_p_for_bookie",
            "product",
            "context_reason_codes",
        ]
        if c in df.columns
    ]
    try:
        lines.append(df[cols].to_markdown(index=False))
    except Exception:
        header = "| " + " | ".join(cols) + " |"
        divider = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = []
        for _, row in df[cols].iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if pd.isna(v):
                    vals.append("")
                else:
                    vals.append(str(v).replace("\n", " ").replace("|", "\\|"))
            body.append("| " + " | ".join(vals) + " |")
        lines.extend([header, divider, *body])
    return lines


# New function: _unresolved_rows
def _unresolved_rows(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty or "actual_btts" not in df.columns:
        return df.head(0).copy()
    out = df.loc[df["actual_btts"].isna()].copy()
    if out.empty:
        return out
    sort_cols = [c for c in ["league", "fixture_key", "home_team_name", "away_team_name"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=True)
    return out.head(n).copy()




# Coverage summary function
def _coverage_summary(shadow_eval: pd.DataFrame, live_eval: pd.DataFrame, resolved_full: pd.DataFrame | None) -> dict:
    if resolved_full is None or resolved_full.empty:
        return {}

    resolved_btts = resolved_full.loc[_canonical_market(resolved_full).eq("btts")].copy()
    if resolved_btts.empty:
        return {
            "shadow_rows": int(len(shadow_eval)),
            "live_rows": int(len(live_eval)),
            "results_rows": 0,
            "shadow_fixture_overlap": 0,
            "live_fixture_overlap": 0,
            "shadow_league_overlap": 0,
            "live_league_overlap": 0,
            "shadow_date_overlap": 0,
            "live_date_overlap": 0,
        }

    shadow_fixture = set(_series_str(shadow_eval, "fixture_key").tolist())
    live_fixture = set(_series_str(live_eval, "fixture_key").tolist())
    results_fixture = set(_series_str(resolved_btts, "fixture_key").tolist())

    shadow_league = set(_series_str(shadow_eval, "league").tolist())
    live_league = set(_series_str(live_eval, "league").tolist())
    results_league = set(_series_str(resolved_btts, "league").tolist())

    shadow_dates = set(_date_from_fixture_key(shadow_eval).dropna().astype("string").tolist())
    live_dates = set(_date_from_fixture_key(live_eval).dropna().astype("string").tolist())
    results_dates = set(_date_from_fixture_key(resolved_btts).dropna().astype("string").tolist())

    shadow_missing_fixture_examples = sorted(list(shadow_fixture - results_fixture))[:5]
    live_missing_fixture_examples = sorted(list(live_fixture - results_fixture))[:5]

    return {
        "shadow_rows": int(len(shadow_eval)),
        "live_rows": int(len(live_eval)),
        "results_rows": int(len(resolved_btts)),
        "shadow_fixture_overlap": int(len(shadow_fixture & results_fixture)),
        "live_fixture_overlap": int(len(live_fixture & results_fixture)),
        "shadow_league_overlap": int(len(shadow_league & results_league)),
        "live_league_overlap": int(len(live_league & results_league)),
        "shadow_date_overlap": int(len(shadow_dates & results_dates)),
        "live_date_overlap": int(len(live_dates & results_dates)),
        "shadow_missing_fixture_examples": shadow_missing_fixture_examples,
        "live_missing_fixture_examples": live_missing_fixture_examples,
    }


def build_markdown(
    shadow_eval: pd.DataFrame,
    live_eval: pd.DataFrame,
    shadow_path: Path,
    live_path: Path,
    results_path: Path | None,
    resolved_full: pd.DataFrame | None = None,
) -> str:
    shadow_metrics = _bucket_metrics(shadow_eval, "shadow_micro_bucket")
    live_metrics = _bucket_metrics(live_eval, "live_whitelist_baseline")
    coverage = _coverage_summary(shadow_eval, live_eval, resolved_full)

    lines: list[str] = []
    lines.append("# BTTS Shadow Bucket Audit")
    lines.append("")
    lines.append(f"- Shadow source: `{shadow_path}`")
    lines.append(f"- Live whitelist source: `{live_path}`")
    lines.append(f"- Results source: `{results_path}`" if results_path is not None else "- Results source: none supplied")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- Shadow rows: {shadow_metrics['rows']}")
    lines.append(f"- Live whitelist rows: {live_metrics['rows']}")
    lines.append(f"- Shadow resolved: {shadow_metrics['resolved']}")
    lines.append(f"- Live whitelist resolved: {live_metrics['resolved']}")
    lines.append(f"- Shadow hit rate: {_fmt_pct(shadow_metrics['hit_rate'])}")
    lines.append(f"- Live whitelist hit rate: {_fmt_pct(live_metrics['hit_rate'])}")
    lines.append("")

    if coverage:
        lines.append("## Results coverage diagnostics")
        lines.extend(_coverage_lines("Coverage", coverage))
        lines.append("")

    lines.append("## Shadow bucket")
    lines.extend(_dict_lines("Selection mix", shadow_metrics["selection_mix"]))
    lines.append("")
    lines.extend(_dict_lines("Runtime mix", shadow_metrics["runtime_mix"]))
    lines.append("")
    lines.extend(_stats_lines("Edge stats", shadow_metrics["edge_stats"]))
    lines.append("")
    lines.extend(_stats_lines("Model probability stats", shadow_metrics["model_p_for_bookie_stats"]))
    lines.append("")

    lines.append("## Live whitelist baseline")
    lines.extend(_dict_lines("Selection mix", live_metrics["selection_mix"]))
    lines.append("")
    lines.extend(_dict_lines("Runtime mix", live_metrics["runtime_mix"]))
    lines.append("")
    lines.extend(_stats_lines("Edge stats", live_metrics["edge_stats"]))
    lines.append("")
    lines.extend(_stats_lines("Model probability stats", live_metrics["model_p_for_bookie_stats"]))
    lines.append("")

    lines.extend(_rows_to_markdown(_top_rows(shadow_eval, n=10), "Top shadow rows"))
    lines.append("")
    lines.extend(_rows_to_markdown(_top_rows(live_eval, n=15), "Top live whitelist rows"))
    lines.append("")
    lines.extend(_rows_to_markdown(_unresolved_rows(shadow_eval, n=10), "Unresolved shadow rows"))
    lines.append("")
    lines.extend(_rows_to_markdown(_unresolved_rows(live_eval, n=15), "Unresolved live whitelist rows"))
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"



def main() -> int:
    ap = argparse.ArgumentParser(description="Audit BTTS shadow micro-bucket against current live whitelist.")
    ap.add_argument("--shadow", default=DEFAULT_SHADOW, help="Path to BTTS shadow bucket CSV")
    ap.add_argument("--live", default=DEFAULT_LIVE, help="Path to BTTS live whitelist baseline CSV")
    ap.add_argument("--results", default=None, help="Optional resolved results CSV containing actual_btts")
    ap.add_argument("--results-col", default="actual_btts", help="Resolved BTTS column name in results CSV")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Markdown output path")
    ap.add_argument("--print-top", type=int, default=10, help="How many top shadow rows to print to console")
    args = ap.parse_args()

    shadow_path = Path(args.shadow)
    live_path = Path(args.live)
    out_path = Path(args.out)

    try:
        shadow = _read_csv(shadow_path, "shadow CSV")
        live = _read_csv(live_path, "live whitelist CSV")

        _require_columns(shadow, EXACT_JOIN_COLS, "shadow CSV")
        _require_columns(live, EXACT_JOIN_COLS, "live whitelist CSV")

        shadow = _make_join_frame(shadow)
        live = _make_join_frame(live)

        shadow_eval = shadow.copy()
        live_eval = live.copy()
        results_path: Path | None = None
        resolved_full: pd.DataFrame | None = None

        if args.results:
            results_path = Path(args.results)
            resolved = _read_csv(results_path, "results CSV").copy()
            _require_columns(resolved, OUTCOME_JOIN_COLS, "results CSV")
            resolved = _make_outcome_join_frame(resolved)
            resolved_full = resolved.copy()

            # Build a minimal join table: outcome key -> actual_btts
            if args.results_col in resolved_full.columns:
                resolved_k = resolved_full[["_outcome_k", args.results_col]].copy().rename(columns={args.results_col: "actual_btts"})
                resolved_k["actual_btts"] = resolved_k["actual_btts"].astype("string").fillna("").str.upper().str.strip()
            else:
                need_goal_cols = ["home_team_goal_count", "away_team_goal_count"]
                missing_goal_cols = [c for c in need_goal_cols if c not in resolved_full.columns]
                if missing_goal_cols:
                    raise ValueError(
                        f"results CSV missing result column '{args.results_col}' and also missing goal columns: {missing_goal_cols}"
                    )

                gh = pd.to_numeric(resolved_full["home_team_goal_count"], errors="coerce")
                ga = pd.to_numeric(resolved_full["away_team_goal_count"], errors="coerce")

                actual_btts = pd.Series(pd.NA, index=resolved_full.index, dtype="string")
                both_known = gh.notna() & ga.notna()
                actual_btts.loc[both_known & (gh >= 1) & (ga >= 1)] = "YES"
                actual_btts.loc[both_known & ~((gh >= 1) & (ga >= 1))] = "NO"

                resolved_k = resolved_full[["_outcome_k"]].copy()
                resolved_k["actual_btts"] = actual_btts

            resolved_k = resolved_k.dropna(subset=["_outcome_k"]).copy()
            resolved_k["actual_btts"] = resolved_k["actual_btts"].astype("string").replace({"": pd.NA})
            resolved_k = resolved_k.drop_duplicates(subset=["_outcome_k"], keep="last")

            # Diagnostics should use the full results frame (it still has league/fixture_key/market)
            print("results market counts:", _market_counts(resolved_full))
            resolved_btts = resolved_full.loc[_canonical_market(resolved_full).eq("btts")].copy()
            print("results BTTS rows:", len(resolved_btts))
            shadow_fixture_set = set(_series_str(shadow_eval, "fixture_key").tolist())
            live_fixture_set = set(_series_str(live_eval, "fixture_key").tolist())
            results_fixture_set = set(_series_str(resolved_btts, "fixture_key").tolist())
            shadow_fixture_overlap = shadow_fixture_set & results_fixture_set
            live_fixture_overlap = live_fixture_set & results_fixture_set
            print("shadow fixture overlap with BTTS results:", len(shadow_fixture_overlap))
            print("live fixture overlap with BTTS results:", len(live_fixture_overlap))
            if not shadow_fixture_overlap:
                print("shadow fixtures absent from results BTTS universe (sample):")
                for x in sorted(list(shadow_fixture_set - results_fixture_set))[:5]:
                    print(x)
            if not live_fixture_overlap:
                print("live fixtures absent from results BTTS universe (sample):")
                for x in sorted(list(live_fixture_set - results_fixture_set))[:10]:
                    print(x)
            if not resolved_btts.empty:
                probe_cols = [
                    c for c in [
                        "league", "fixture_key", "market", "bookie_pick", "selection",
                        "_outcome_market", "_outcome_k"
                    ] if c in resolved_btts.columns
                ]
                print("\nResolved BTTS key sample:")
                print(resolved_btts[probe_cols].head(10).to_string(index=False))

            shadow_eval = _make_outcome_join_frame(shadow_eval)
            live_eval = _make_outcome_join_frame(live_eval)

            resolved_join = resolved_k.copy()
            shadow_eval = shadow_eval.merge(resolved_join, on="_outcome_k", how="left")
            live_eval = live_eval.merge(resolved_join, on="_outcome_k", how="left")

            shadow_match_rate = float(shadow_eval["actual_btts"].notna().mean()) if len(shadow_eval) else 0.0
            live_match_rate = float(live_eval["actual_btts"].notna().mean()) if len(live_eval) else 0.0
            print(f"shadow merge hit rate: {shadow_match_rate:.4f}")
            print(f"live merge hit rate: {live_match_rate:.4f}")

            if shadow_match_rate == 0.0 or live_match_rate == 0.0:
                shadow_probe_cols = [
                    c for c in [
                        "league", "fixture_key", "market", "bookie_pick", "selection",
                        "_join_market", "_join_bookie_pick", "_k", "_outcome_k"
                    ] if c in shadow_eval.columns
                ]
                live_probe_cols = [
                    c for c in [
                        "league", "fixture_key", "market", "bookie_pick", "selection",
                        "_join_market", "_join_bookie_pick", "_k", "_outcome_k"
                    ] if c in live_eval.columns
                ]
                resolved_probe_cols = [
                    c for c in [
                        "league", "fixture_key", "market", "bookie_pick", "selection",
                        "_outcome_market", "_outcome_k"
                    ] if c in resolved_full.columns
                ]
                print("\nShadow key sample:")
                print(shadow_eval[shadow_probe_cols].head(5).to_string(index=False))
                print("\nLive key sample:")
                print(live_eval[live_probe_cols].head(5).to_string(index=False))
                print("\nResolved key sample:")
                print(resolved_full[resolved_probe_cols].head(5).to_string(index=False))

                shadow_exact_keys = set(shadow_eval["_k"].astype("string").tolist())
                live_exact_keys = set(live_eval["_k"].astype("string").tolist())
                shadow_outcome_keys = set(shadow_eval["_outcome_k"].astype("string").tolist())
                live_outcome_keys = set(live_eval["_outcome_k"].astype("string").tolist())
                resolved_outcome_keys = set(resolved_join["_outcome_k"].astype("string").tolist())

                shadow_exact_overlap = shadow_exact_keys & resolved_outcome_keys
                live_exact_overlap = live_exact_keys & resolved_outcome_keys
                shadow_outcome_overlap = shadow_outcome_keys & resolved_outcome_keys
                live_outcome_overlap = live_outcome_keys & resolved_outcome_keys

                print(f"\nShadow exact-key overlap count: {len(shadow_exact_overlap)}")
                print(f"Live exact-key overlap count: {len(live_exact_overlap)}")
                print(f"Shadow outcome-key overlap count: {len(shadow_outcome_overlap)}")
                print(f"Live outcome-key overlap count: {len(live_outcome_overlap)}")

                if shadow_outcome_overlap:
                    print("Shadow outcome-overlap sample:")
                    for k in sorted(list(shadow_outcome_overlap))[:10]:
                        print(k)
                if live_outcome_overlap:
                    print("Live outcome-overlap sample:")
                    for k in sorted(list(live_outcome_overlap))[:10]:
                        print(k)

            shadow_eval["win"] = (
                _series_upper(shadow_eval, "selection").eq(_series_upper(shadow_eval, "actual_btts"))
                & shadow_eval["actual_btts"].notna()
            ).astype("boolean")
            live_eval["win"] = (
                _series_upper(live_eval, "selection").eq(_series_upper(live_eval, "actual_btts"))
                & live_eval["actual_btts"].notna()
            ).astype("boolean")
        else:
            shadow_eval["actual_btts"] = pd.Series(pd.NA, index=shadow_eval.index, dtype="string")
            shadow_eval["win"] = pd.Series(pd.NA, index=shadow_eval.index, dtype="boolean")
            live_eval["actual_btts"] = pd.Series(pd.NA, index=live_eval.index, dtype="string")
            live_eval["win"] = pd.Series(pd.NA, index=live_eval.index, dtype="boolean")

        md = build_markdown(
            shadow_eval=shadow_eval,
            live_eval=live_eval,
            shadow_path=shadow_path,
            live_path=live_path,
            results_path=results_path,
            resolved_full=resolved_full,
        )
        out_path.write_text(md, encoding="utf-8")

        print("saved markdown:", out_path)
        print("shadow rows:", len(shadow_eval))
        print("live rows:", len(live_eval))
        if args.results:
            s_res = int(shadow_eval["actual_btts"].notna().sum())
            l_res = int(live_eval["actual_btts"].notna().sum())
            s_hit = shadow_eval.loc[shadow_eval["actual_btts"].notna(), "win"].mean() if s_res else float("nan")
            l_hit = live_eval.loc[live_eval["actual_btts"].notna(), "win"].mean() if l_res else float("nan")
            print("shadow resolved:", s_res, "| hit_rate:", "n/a" if pd.isna(s_hit) else round(float(s_hit), 4))
            print("live resolved:", l_res, "| hit_rate:", "n/a" if pd.isna(l_hit) else round(float(l_hit), 4))

        unresolved_shadow = _unresolved_rows(shadow_eval, n=max(1, int(args.print_top)))
        unresolved_live = _unresolved_rows(live_eval, n=max(1, int(args.print_top)))

        if not unresolved_shadow.empty:
            cols = [
                c
                for c in [
                    "league",
                    "fixture_key",
                    "home_team_name",
                    "away_team_name",
                    "selection",
                    "model_top_pick",
                    "signal_btts_runtime",
                    "edge",
                    "model_p_for_bookie",
                    "actual_btts",
                    "_outcome_k",
                ]
                if c in unresolved_shadow.columns
            ]
            print("\nUnresolved shadow rows:")
            print(unresolved_shadow[cols].to_string(index=False))

        if not unresolved_live.empty:
            cols = [
                c
                for c in [
                    "league",
                    "fixture_key",
                    "home_team_name",
                    "away_team_name",
                    "selection",
                    "model_top_pick",
                    "signal_btts_runtime",
                    "edge",
                    "model_p_for_bookie",
                    "actual_btts",
                    "_outcome_k",
                ]
                if c in unresolved_live.columns
            ]
            print("\nUnresolved live rows:")
            print(unresolved_live[cols].to_string(index=False))

        top_shadow = _top_rows(shadow_eval, n=max(1, int(args.print_top)))
        if not top_shadow.empty:
            cols = [
                c
                for c in [
                    "league",
                    "home_team_name",
                    "away_team_name",
                    "selection",
                    "model_top_pick",
                    "signal_btts_runtime",
                    "edge",
                    "model_p_for_bookie",
                    "actual_btts",
                    "win",
                ]
                if c in top_shadow.columns
            ]
            print("\nTop shadow rows:")
            print(top_shadow[cols].to_string(index=False))

        return 0
    except Exception as e:
        print(f"[audit_btts_shadow_bucket] failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())