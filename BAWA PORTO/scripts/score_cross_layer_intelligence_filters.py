#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORE_ROWS = ROOT / "reports/latest/weekend_deploy_tier_scoring_2026_05_14_to_2026_05_19/DEPLOY_TIER_SCORE_ROWS.csv"
CROSS_LAYER = ROOT / "reports/latest/full_cross_layer_analysis_2026_05_15_to_2026_05_19/FULL_CROSS_LAYER_ANALYSIS.csv"
OUTDIR = ROOT / "reports/latest/weekend_deploy_tier_scoring_2026_05_14_to_2026_05_19/intelligence_overlay_filters"

STRICT_SUPPORT_READS = {"FULL_CONSENSUS", "STRONG_SUPPORT", "SPORTSMOLE_RESCUES_BTTS_SHAPE"}
NO_CONFLICT_READS = STRICT_SUPPORT_READS | {"SUPPORTED_BUT_MIXED"}
CAUTION_READS = {"PREVIEW_COUNTER_CAUTION", "INTERNAL_COUNTER_CAUTION"}


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def market_norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"over25", "over_25", "ou25", "over_2_5"}:
        return "ou25"
    return text


def key(row: pd.Series, market_col: str) -> str:
    return "|".join(
        [
            str(row.get("match_date") or "")[:10],
            norm(row.get("home_team_name")),
            norm(row.get("away_team_name")),
            market_norm(row.get(market_col)),
        ]
    )


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def summarize(rows: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    out: list[dict[str, Any]] = []
    if group_cols:
        iterator = rows.groupby(group_cols, dropna=False)
    else:
        iterator = [((), rows)]
    for group_key, group in iterator:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        settled = group[group["result_status"].isin(["won", "lost"])]
        wins = int((settled["result_status"] == "won").sum())
        losses = int((settled["result_status"] == "lost").sum())
        profits = pd.to_numeric(settled.get("profit_units"), errors="coerce").dropna()
        item = {col: group_key[idx] for idx, col in enumerate(group_cols)}
        item.update(
            {
                "rows": int(len(group)),
                "settled": int(len(settled)),
                "missing_actual": int((group["result_status"] == "missing_actual").sum()),
                "wins": wins,
                "losses": losses,
                "hit_rate": round(wins / len(settled), 4) if len(settled) else None,
                "profit_units": round(float(profits.sum()), 4) if not profits.empty else None,
                "roi": round(float(profits.sum()) / len(profits), 4) if not profits.empty else None,
            }
        )
        out.append(item)
    return out


def add_filter_rows(rows: pd.DataFrame, name: str, mask: pd.Series) -> list[dict[str, Any]]:
    subset = rows[mask].copy()
    summary = summarize(subset, [])[0] if not subset.empty else {
        "rows": 0,
        "settled": 0,
        "missing_actual": 0,
        "wins": 0,
        "losses": 0,
        "hit_rate": None,
        "profit_units": None,
        "roi": None,
    }
    summary["filter_name"] = name
    return [summary]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for col in row:
            if col not in fieldnames:
                fieldnames.append(col)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Intelligence Overlay Filter Scoring",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Raw ELITE/STANDARD Deploy",
        "",
        "| Tier | Market | Wins | Settled | Hit Rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["raw_by_tier_market"]:
        lines.append(
            f"| {row['score_tier']} | {row['score_market']} | {row['wins']} | {row['settled']} | {pct(row['hit_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Cross-Layer Reads",
            "",
            "| Cross-Layer Read | Wins | Settled | Hit Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["overlay_by_read"]:
        lines.append(
            f"| {row['cross_layer_read']} | {row['wins']} | {row['settled']} | {pct(row['hit_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Filter Views",
            "",
            "| Filter | Wins | Settled | Hit Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["filter_summary"]:
        lines.append(f"| {row['filter_name']} | {row['wins']} | {row['settled']} | {pct(row['hit_rate'])} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The full overlay file covers the May 15-19 ELITE/STANDARD forward board, not the May 14 MLS rows.",
            "- Strict intelligence support is useful for goal markets, but FTR still needs a separate trained-league / injury-shock / motivation-state gate.",
            "- OBSERVE is deliberately excluded from this overlay filter scorecard.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    score = pd.read_csv(SCORE_ROWS)
    cross = pd.read_csv(CROSS_LAYER)

    raw = score[score["score_tier"].isin(["ELITE", "STANDARD"])].copy()
    cross = cross.copy()
    raw["join_key"] = raw.apply(lambda row: key(row, "score_market"), axis=1)
    cross["join_key"] = cross.apply(lambda row: key(row, "market_l"), axis=1)

    scored = cross.merge(
        raw[
            [
                "join_key",
                "score_tier",
                "score_market",
                "actual",
                "home_goals",
                "away_goals",
                "result_status",
                "provider_status",
                "actual_source",
                "profit_units",
            ]
        ],
        on="join_key",
        how="left",
        validate="one_to_one",
    )

    settled_mask = scored["result_status"].isin(["won", "lost"])
    goal_mask = scored["market_l"].isin(["btts", "ou25"])
    ftr_mask = scored["market_l"].eq("ftr")

    filter_summary: list[dict[str, Any]] = []
    filter_summary += add_filter_rows(scored, "overlay_scope_all_markets", pd.Series(True, index=scored.index))
    filter_summary += add_filter_rows(scored, "strict_support_all_markets", scored["cross_layer_read"].isin(STRICT_SUPPORT_READS))
    filter_summary += add_filter_rows(scored, "strict_support_goal_markets_only", scored["cross_layer_read"].isin(STRICT_SUPPORT_READS) & goal_mask)
    filter_summary += add_filter_rows(scored, "no_conflict_all_markets", scored["cross_layer_read"].isin(NO_CONFLICT_READS))
    filter_summary += add_filter_rows(scored, "no_conflict_goal_markets_only", scored["cross_layer_read"].isin(NO_CONFLICT_READS) & goal_mask)
    filter_summary += add_filter_rows(scored, "caution_reads_all_markets", scored["cross_layer_read"].isin(CAUTION_READS))
    filter_summary += add_filter_rows(scored, "weak_or_unresolved_all_markets", scored["cross_layer_read"].eq("WEAK_OR_UNRESOLVED"))
    filter_summary += add_filter_rows(scored, "ftr_overlay_scope", ftr_mask)
    filter_summary += add_filter_rows(scored, "goal_overlay_scope", goal_mask)

    summary = {
        "generated_at": now_utc(),
        "score_rows": str(SCORE_ROWS.relative_to(ROOT)),
        "cross_layer_rows": str(CROSS_LAYER.relative_to(ROOT)),
        "raw_by_tier": summarize(raw, ["score_tier"]),
        "raw_by_tier_market": summarize(raw, ["score_tier", "score_market"]),
        "overlay_by_read": summarize(scored[settled_mask | scored["result_status"].eq("missing_actual")], ["cross_layer_read"]),
        "overlay_by_tier_market_read": summarize(scored, ["tier_file", "market_l", "cross_layer_read"]),
        "filter_summary": filter_summary,
        "outputs": {
            "scored_rows": str((OUTDIR / "SCORED_FULL_CROSS_LAYER_ROWS.csv").relative_to(ROOT)),
            "filter_summary": str((OUTDIR / "INTELLIGENCE_FILTER_SCORE_SUMMARY.csv").relative_to(ROOT)),
            "summary_json": str((OUTDIR / "summary.json").relative_to(ROOT)),
            "summary_md": str((OUTDIR / "SUMMARY.md").relative_to(ROOT)),
        },
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUTDIR / "SCORED_FULL_CROSS_LAYER_ROWS.csv", index=False)
    write_csv(OUTDIR / "INTELLIGENCE_FILTER_SCORE_SUMMARY.csv", filter_summary)
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (OUTDIR / "SUMMARY.md").write_text(markdown(summary), encoding="utf-8")

    print(json.dumps(summary["filter_summary"], indent=2, ensure_ascii=False))
    print(f"Outputs: {OUTDIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
