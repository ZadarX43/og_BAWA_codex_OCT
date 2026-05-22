#!/usr/bin/env python3
"""Accumulate player-event live shadow outcomes across boards.

Research-only sidecar. This turns repeated outcome-tracker exports into a
single rolling ledger so player-event watch labels can be evaluated over time.
It does not create priced odds, deploy picks, slips, or production routing
changes.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "reports" / "player_events" / "live_shadow_outcomes" / "PLAYER_EVENT_LIVE_OUTCOME_LEDGER.csv"
DEFAULT_OUTDIR = ROOT / "reports" / "player_events" / "live_shadow_outcomes"

KEY_COLUMNS = [
    "shadow_stage",
    "fixture_key",
    "match_date",
    "league",
    "home_team_name",
    "away_team_name",
    "team_name_norm",
    "player_name_norm",
]

SUMMARY_GROUPS = {
    "stage_priority": ["shadow_stage", "watch_priority"],
    "stage_league": ["shadow_stage", "league"],
    "stage_role": ["shadow_stage", "tactical_role"],
    "stage_player": ["shadow_stage", "league", "team_name", "player_name"],
    "stage_team": ["shadow_stage", "league", "team_name"],
    "stage_context": ["shadow_stage", "fouled_context_cell_label"],
    "stage_red_card_context": ["shadow_stage", "red_card_context_flag"],
    "stage_substitution_context": ["shadow_stage", "substitution_context_flag"],
    "stage_sub_swap_review": ["shadow_stage", "player_sub_swap_review_mode"],
    "stage_month": ["shadow_stage", "match_month"],
}


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[col].astype("string").fillna("")


def stable_hash(parts: list[Any]) -> str:
    raw = "||".join("" if pd.isna(part) else str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def normalize_tracker(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df

    for col in KEY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if "player_name_norm" not in df.columns or text_series(df, "player_name_norm").eq("").all():
        df["player_name_norm"] = text_series(df, "player_name").str.lower().str.strip()
    if "team_name_norm" not in df.columns or text_series(df, "team_name_norm").eq("").all():
        df["team_name_norm"] = text_series(df, "team_name").str.lower().str.strip()

    df["match_date"] = pd.to_datetime(df.get("match_date"), errors="coerce").dt.date.astype("string")
    df["match_month"] = pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m")
    df["outcome_status"] = text_series(df, "outcome_status").replace("", "PENDING_NO_MATCH")
    df["actual_hit"] = num(df.get("actual_hit", pd.Series(np.nan, index=df.index)))
    df["actual_stat_value"] = num(df.get("actual_stat_value", pd.Series(np.nan, index=df.index)))
    df["actual_threshold"] = num(df.get("actual_threshold", pd.Series(np.nan, index=df.index)))
    df["predicted_hit_rate"] = num(df.get("predicted_hit_rate", pd.Series(np.nan, index=df.index)))
    df["backtest_hit_rate"] = num(df.get("backtest_hit_rate", pd.Series(np.nan, index=df.index)))
    df["source_tracker_path"] = str(path)
    df["source_tracker_mtime_utc"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    df["source_snapshot_id"] = path.parents[1].name if len(path.parents) > 1 else path.parent.name
    df["accumulated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df["player_event_row_key"] = [
        stable_hash([row.get(col, "") for col in KEY_COLUMNS])
        for _, row in df[KEY_COLUMNS].iterrows()
    ]
    df["_graded_rank"] = np.where(df["outcome_status"].eq("GRADED"), 1, 0)
    return df


def load_inputs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.exists():
            frames.append(normalize_tracker(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def discover_tracker_rows(search_root: Path) -> list[Path]:
    if search_root.is_file():
        return [search_root]
    return sorted(search_root.glob("**/PLAYER_EVENT_SHADOW_OUTCOME_TRACKER_ROWS.csv"))


def merge_ledger(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = incoming.copy()
    elif incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)

    if combined.empty:
        return combined
    if "_graded_rank" not in combined.columns:
        combined["_graded_rank"] = np.where(combined.get("outcome_status", "").astype(str).eq("GRADED"), 1, 0)
    if "source_tracker_mtime_utc" not in combined.columns:
        combined["source_tracker_mtime_utc"] = ""

    combined = combined.sort_values(
        ["player_event_row_key", "_graded_rank", "source_tracker_mtime_utc", "accumulated_at_utc"],
        ascending=[True, True, True, True],
    )
    return combined.drop_duplicates("player_event_row_key", keep="last").reset_index(drop=True)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["rows", "graded", "hits", "hit_rate", "pending"])
    for col in group_cols:
        if col not in df.columns:
            df[col] = ""
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        graded_mask = group["outcome_status"].astype(str).eq("GRADED")
        graded = group[graded_mask]
        hits = float(num(graded.get("actual_hit", pd.Series(dtype=float))).sum()) if not graded.empty else 0.0
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "rows": int(len(group)),
                "graded": int(len(graded)),
                "hits": int(hits),
                "hit_rate": float(hits / len(graded)) if len(graded) else np.nan,
                "pending": int((~graded_mask).sum()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["graded", "rows"], ascending=[False, False]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            work[col] = work[col].astype("string").fillna("")
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in work.columns) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, ledger: pd.DataFrame, summaries: dict[str, pd.DataFrame], input_paths: list[Path]) -> None:
    graded = int(ledger["outcome_status"].astype(str).eq("GRADED").sum()) if not ledger.empty else 0
    pending = int(len(ledger) - graded)
    lines = [
        "# Player Event Live Outcome Accumulator",
        "",
        "Research-only rolling ledger for player-event shadow outcomes.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Ledger rows are deduped by fixture/player/team/market and prefer graded outcomes over pending repeats.",
        "- Player Sub Swap evidence is kept as separate review context, not mixed into named-player prop grading.",
        "",
        "## Inputs",
        *[f"- `{path}`" for path in input_paths[:25]],
        "",
        "## Overall",
        f"- ledger rows: `{len(ledger)}`",
        f"- graded rows: `{graded}`",
        f"- pending rows: `{pending}`",
        "",
        "## Stage / Priority",
        markdown_table(summaries.get("stage_priority", pd.DataFrame())),
        "",
        "## Stage / League",
        markdown_table(summaries.get("stage_league", pd.DataFrame()), max_rows=60),
        "",
        "## Stage / Month",
        markdown_table(summaries.get("stage_month", pd.DataFrame()), max_rows=60),
        "",
        "## Stage / Role",
        markdown_table(summaries.get("stage_role", pd.DataFrame()), max_rows=60),
        "",
        "## Stage / Fouled Context",
        markdown_table(summaries.get("stage_context", pd.DataFrame()), max_rows=60),
        "",
        "## Stage / Red Card Context",
        markdown_table(summaries.get("stage_red_card_context", pd.DataFrame()), max_rows=60),
        "",
        "## Stage / Substitution Context",
        markdown_table(summaries.get("stage_substitution_context", pd.DataFrame()), max_rows=60),
        "",
        "## Stage / Player Sub Swap Review",
        markdown_table(summaries.get("stage_sub_swap_review", pd.DataFrame()), max_rows=60),
    ]
    (outdir / "PLAYER_EVENT_LIVE_OUTCOME_ACCUMULATOR.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-rows", type=Path, action="append", default=[])
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    input_paths = list(args.tracker_rows)
    for root in args.search_root:
        input_paths.extend(discover_tracker_rows(root))
    input_paths = sorted({path.resolve() for path in input_paths})
    if not input_paths:
        raise SystemExit("No tracker rows found. Pass --tracker-rows or --search-root.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)

    incoming = load_inputs(input_paths)
    existing = pd.DataFrame()
    if args.ledger.exists() and not args.reset:
        existing = pd.read_csv(args.ledger, low_memory=False)
    ledger = merge_ledger(existing, incoming)

    ledger.drop(columns=["_graded_rank"], errors="ignore").to_csv(args.ledger, index=False)
    ledger.drop(columns=["_graded_rank"], errors="ignore").to_csv(args.outdir / "PLAYER_EVENT_LIVE_OUTCOME_LEDGER.csv", index=False)

    summaries = {name: summarize(ledger, cols) for name, cols in SUMMARY_GROUPS.items()}
    for name, summary in summaries.items():
        summary.to_csv(args.outdir / f"PLAYER_EVENT_LIVE_OUTCOME_{name.upper()}_SUMMARY.csv", index=False)
    write_report(args.outdir, ledger, summaries, input_paths)

    print(f"WROTE {args.outdir}")
    print(f"ledger_rows={len(ledger)} graded={int(ledger['outcome_status'].astype(str).eq('GRADED').sum()) if not ledger.empty else 0}")
    print(f"ledger={args.ledger}")


if __name__ == "__main__":
    main()
