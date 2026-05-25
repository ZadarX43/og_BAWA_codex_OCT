#!/usr/bin/env python3
"""Build product-market league proof tables for website/investor onboarding.

This is a reporting script only. It reads settled walk-forward artifacts and
does not alter production routing, deploy gates, or model outputs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PHASE8H_ROWS = Path(
    "reports/2026-05-06/phase8h_full_estate_c4_sweeps/phase8h_replay_row_level_scored.csv"
)
DEFAULT_CS_PREMIUM = Path(
    "reports/2026-04-23/CS_BACKTEST__V2_PREMIUM/CS_BACKTEST__LEAGUE_AWARE_PREMIUM_SHORTLIST.csv"
)
DEFAULT_TG15_LEAGUE_SPLITS = Path(
    "reports/latest/team_goal_15_promotion_candidate_tracker/TEAM_GOAL_15_PROMOTION_CANDIDATE_LEAGUE_SPLITS.csv"
)
DEFAULT_OUTPUT_DIR = Path("reports/latest/product_market_league_breakdown")

CALIBRATED_MARKET_SUMMARY = (
    {
        "product_key": "btts_calibrated",
        "display_name": "BTTS Calibrated",
        "correct": 3164,
        "graded": 3382,
        "hit_rate": 0.9355,
        "status": "locked calibrated threshold layer",
    },
    {
        "product_key": "ou25_calibrated",
        "display_name": "Over 2.5 Calibrated",
        "correct": 3650,
        "graded": 3828,
        "hit_rate": 0.9535,
        "status": "locked calibrated threshold layer",
    },
)

BTTS_CALIBRATED_LEAGUES = (
    ("England FA Cup", 439, 0.9977, "0.80"),
    ("Europa Conference", 461, 0.9826, "0.88"),
    ("Champions League", 258, 0.9806, "0.80"),
    ("Brazil Serie A", 83, 0.9759, "0.85"),
    ("Italy Serie A", 179, 0.9665, "0.85"),
    ("Spain La Liga", 248, 0.9556, "0.80"),
    ("France Ligue 1", 121, 0.9174, "0.88"),
    ("Scotland Premiership", 110, 0.9091, "0.80"),
    ("Belgium Pro", 158, 0.9051, "0.88"),
    ("Norway Eliteserien", 138, 0.8986, "0.80"),
    ("Netherlands Eredivisie", 382, 0.8953, "0.88"),
    ("Japan J1", 293, 0.8908, "0.80"),
    ("USA MLS", 512, 0.8750, "0.85"),
)

OU25_CALIBRATED_LEAGUES = (
    ("England FA Cup", 532, 0.9962, "0.80"),
    ("Europa Conference", 501, 0.9741, "0.80"),
    ("Germany Bundesliga", 276, 0.9710, "0.88"),
    ("Europa League", 243, 0.9630, "0.85"),
    ("Brazil Serie A", 69, 0.9565, "0.80"),
    ("Champions League", 385, 0.9532, "0.80"),
    ("Portugal Liga", 213, 0.9531, "0.90"),
    ("Netherlands Eredivisie", 301, 0.9468, "0.88"),
    ("Scotland Premiership", 72, 0.9444, "0.90"),
    ("Spain La Liga", 352, 0.9375, "0.80"),
    ("Japan J1", 241, 0.9336, "0.80"),
    ("Norway Eliteserien", 117, 0.9145, "0.85"),
    ("USA MLS", 526, 0.9106, "0.80"),
)

VALUE_EDGE_SUMMARY = (
    {
        "value_edge_tier": "STANDARD",
        "rows": 1982,
        "hit_rate": 0.6221,
        "profit": 272.59,
        "roi": 0.1375,
    },
    {
        "value_edge_tier": "STRONG",
        "rows": 1927,
        "hit_rate": 0.6513,
        "profit": 373.58,
        "roi": 0.1939,
    },
    {
        "value_edge_tier": "PREMIUM",
        "rows": 15203,
        "hit_rate": 0.8331,
        "profit": 8194.99,
        "roi": 0.5390,
    },
)

VALUE_RESPONSE_LANES = (
    {
        "lane": "BTTS_NO PREMIUM",
        "rows": 4968,
        "hit_rate": 0.8060,
        "profit": 3027.15,
        "roi": 0.6093,
    },
    {
        "lane": "OU25_OVER PREMIUM",
        "rows": 4532,
        "hit_rate": 0.8652,
        "profit": None,
        "roi": 0.5285,
    },
    {
        "lane": "BTTS_YES PREMIUM",
        "rows": 5653,
        "hit_rate": 0.8344,
        "profit": None,
        "roi": 0.4921,
    },
    {
        "lane": "BTTS_NO META_ELITE",
        "rows": 2712,
        "hit_rate": 0.9015,
        "profit": 1989.19,
        "roi": 0.7335,
        "mean_odds": 1.922,
    },
)


MARKET_CONFIGS = (
    {
        "product_key": "over_25",
        "display_name": "Over 2.5",
        "market_norm": "ou25",
        "selection": "OVER25",
        "tiers": {"STANDARD"},
        "status": "production",
    },
    {
        "product_key": "btts_yes",
        "display_name": "BTTS Yes",
        "market_norm": "btts",
        "selection": "YES",
        "tiers": {"ELITE", "STANDARD"},
        "status": "production",
    },
    {
        "product_key": "btts_no",
        "display_name": "BTTS No",
        "market_norm": "btts",
        "selection": "NO",
        "tiers": {"ELITE", "STANDARD"},
        "status": "production / policy lane",
    },
    {
        "product_key": "ftr",
        "display_name": "FTR",
        "market_norm": "ftr",
        "selection": "",
        "tiers": {"ELITE", "STANDARD"},
        "status": "production",
    },
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig", errors="ignore") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def as_int(value: Any) -> int:
    try:
        return int(float(str(value or "0").strip()))
    except ValueError:
        return 0


def as_float(value: Any) -> float:
    try:
        return float(str(value or "0").strip())
    except ValueError:
        return 0.0


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def aggregate_phase8h_markets(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    league_rows: list[dict[str, Any]] = []
    for config in MARKET_CONFIGS:
        aggregate: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "graded": 0,
                "wins": 0,
                "losses": 0,
                "profit": 0.0,
                "elite_graded": 0,
                "elite_wins": 0,
                "standard_graded": 0,
                "standard_wins": 0,
            }
        )
        for row in rows:
            if row.get("market_norm") != config["market_norm"]:
                continue
            if config["selection"] and row.get("selection") != config["selection"]:
                continue
            if row.get("tier") not in config["tiers"] or row.get("correct") == "":
                continue
            league = row.get("league") or "Unknown"
            win = as_int(row.get("correct"))
            bookie_od = as_float(row.get("bookie_od"))
            profit = bookie_od - 1 if win and bookie_od else (-1 if not win else 0)
            bucket = aggregate[league]
            bucket["graded"] += 1
            bucket["wins"] += win
            bucket["losses"] += 1 - win
            bucket["profit"] += profit
            if row.get("tier") == "ELITE":
                bucket["elite_graded"] += 1
                bucket["elite_wins"] += win
            elif row.get("tier") == "STANDARD":
                bucket["standard_graded"] += 1
                bucket["standard_wins"] += win

        total_graded = sum(item["graded"] for item in aggregate.values())
        total_wins = sum(item["wins"] for item in aggregate.values())
        total_profit = sum(item["profit"] for item in aggregate.values())
        summary_rows.append(
            {
                "product_key": config["product_key"],
                "display_name": config["display_name"],
                "status": config["status"],
                "correct": total_wins,
                "graded": total_graded,
                "hit_rate": total_wins / total_graded if total_graded else None,
                "profit": total_profit,
                "league_count": len(aggregate),
            }
        )
        for league, item in sorted(aggregate.items(), key=lambda pair: pair[1]["graded"], reverse=True):
            league_rows.append(
                {
                    "product_key": config["product_key"],
                    "display_name": config["display_name"],
                    "league": league,
                    "correct": item["wins"],
                    "graded": item["graded"],
                    "hit_rate": item["wins"] / item["graded"] if item["graded"] else None,
                    "profit": item["profit"],
                    "elite_correct": item["elite_wins"],
                    "elite_graded": item["elite_graded"],
                    "standard_correct": item["standard_wins"],
                    "standard_graded": item["standard_graded"],
                }
            )
    return summary_rows, league_rows


def aggregate_correct_score(rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate: dict[str, dict[str, int]] = defaultdict(lambda: {"fixtures": 0, "exact": 0, "top3": 0})
    for row in rows:
        if not row.get("actual_score"):
            continue
        item = aggregate[row.get("league") or "Unknown"]
        item["fixtures"] += 1
        item["exact"] += as_int(row.get("exact_hit_flag"))
        item["top3"] += as_int(row.get("top3_hit_flag"))
    fixtures = sum(item["fixtures"] for item in aggregate.values())
    exact = sum(item["exact"] for item in aggregate.values())
    top3 = sum(item["top3"] for item in aggregate.values())
    summary = {
        "product_key": "correct_score",
        "display_name": "Correct Score Premium Top 3",
        "status": "premium product layer",
        "exact": exact,
        "top3": top3,
        "fixtures": fixtures,
        "exact_hit_rate": exact / fixtures if fixtures else None,
        "top3_hit_rate": top3 / fixtures if fixtures else None,
        "league_count": len(aggregate),
    }
    league_rows = [
        {
            "product_key": "correct_score",
            "display_name": "Correct Score Premium Top 3",
            "league": league,
            "exact": item["exact"],
            "top3": item["top3"],
            "fixtures": item["fixtures"],
            "exact_hit_rate": item["exact"] / item["fixtures"] if item["fixtures"] else None,
            "top3_hit_rate": item["top3"] / item["fixtures"] if item["fixtures"] else None,
        }
        for league, item in sorted(aggregate.items(), key=lambda pair: pair[1]["fixtures"], reverse=True)
    ]
    return summary, league_rows


def aggregate_tg15(rows: list[dict[str, str]], policy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate: dict[str, dict[str, int]] = defaultdict(
        lambda: {"graded": 0, "wins": 0, "losses": 0, "home_graded": 0, "home_wins": 0, "away_graded": 0, "away_wins": 0}
    )
    for row in rows:
        parts = [part.strip() for part in (row.get("value") or "").split("|")]
        if len(parts) != 3:
            continue
        product, row_policy, league = parts
        if row_policy != policy:
            continue
        graded = as_int(row.get("graded_rows"))
        wins = as_int(row.get("wins"))
        item = aggregate[league]
        item["graded"] += graded
        item["wins"] += wins
        item["losses"] += as_int(row.get("losses"))
        if product.startswith("HOME"):
            item["home_graded"] += graded
            item["home_wins"] += wins
        elif product.startswith("AWAY"):
            item["away_graded"] += graded
            item["away_wins"] += wins
    total_graded = sum(item["graded"] for item in aggregate.values())
    total_wins = sum(item["wins"] for item in aggregate.values())
    summary = {
        "product_key": f"team_goal_15_{policy.lower()}",
        "display_name": f"Team Goals 1.5 {policy}",
        "status": "research/watch",
        "correct": total_wins,
        "graded": total_graded,
        "hit_rate": total_wins / total_graded if total_graded else None,
        "league_count": len(aggregate),
    }
    league_rows = [
        {
            "product_key": f"team_goal_15_{policy.lower()}",
            "display_name": f"Team Goals 1.5 {policy}",
            "league": league,
            "correct": item["wins"],
            "graded": item["graded"],
            "hit_rate": item["wins"] / item["graded"] if item["graded"] else None,
            "home_correct": item["home_wins"],
            "home_graded": item["home_graded"],
            "away_correct": item["away_wins"],
            "away_graded": item["away_graded"],
        }
        for league, item in sorted(aggregate.items(), key=lambda pair: pair[1]["graded"], reverse=True)
    ]
    return summary, league_rows


def calibrated_league_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product_key, display_name, source_rows in (
        ("btts_calibrated", "BTTS Calibrated", BTTS_CALIBRATED_LEAGUES),
        ("ou25_calibrated", "Over 2.5 Calibrated", OU25_CALIBRATED_LEAGUES),
    ):
        for league, graded, hit_rate, threshold in source_rows:
            rows.append(
                {
                    "product_key": product_key,
                    "display_name": display_name,
                    "league": league,
                    "rows": graded,
                    "hit_rate": hit_rate,
                    "threshold_used": threshold,
                    "status": "locked calibrated threshold layer",
                    "source_artifact": "reports/2026-04-21/PHASE8H_VALUE_LAYER__LOCKED.md",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(field for row in rows for field in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(
    market_summary: list[dict[str, Any]],
    market_leagues: list[dict[str, Any]],
    cs_summary: dict[str, Any],
    cs_leagues: list[dict[str, Any]],
    tg15_summaries: list[dict[str, Any]],
    tg15_leagues: list[dict[str, Any]],
    calibrated_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Product Market League Breakdown",
        "",
        "Date: 2026-05-25",
        "",
        "Purpose: website onboarding and investor proof tables by product, league, volume, and hit rate.",
        "",
        "## Strongest Calibrated Numbers",
        "",
        "These are the strongest locked numbers in the system. They come from the calibrated threshold layer, not the broader production deploy-lane table below.",
        "",
        markdown_table(
            ["Product", "Correct / Rows", "Hit Rate", "State"],
            [
                [
                    row["display_name"],
                    f"{int(round(row['graded'] * row['hit_rate']))} / {int(row['graded'])}",
                    pct(row["hit_rate"]),
                    row["status"],
                ]
                for row in CALIBRATED_MARKET_SUMMARY
            ],
        ),
        "",
        "## Value Edge System",
        "",
        "The value layer is additive only: it ranks commercial edge inside already-strong lanes and does not override deploy gates.",
        "",
        markdown_table(
            ["Tier", "Rows", "Hit Rate", "Artifact Profit", "ROI"],
            [
                [
                    row["value_edge_tier"],
                    f"{int(row['rows']):,}",
                    pct(row["hit_rate"]),
                    signed(row["profit"]),
                    pct(row["roi"]),
                ]
                for row in VALUE_EDGE_SUMMARY
            ],
        ),
        "",
        "Best commercial value response by lane:",
        "",
        markdown_table(
            ["Lane", "Rows", "Hit Rate", "Artifact Profit", "ROI", "Mean Odds"],
            [
                [
                    row["lane"],
                    f"{int(row['rows']):,}",
                    pct(row["hit_rate"]),
                    signed(row.get("profit")),
                    pct(row["roi"]),
                    f"{row['mean_odds']:.3f}" if row.get("mean_odds") else "n/a",
                ]
                for row in VALUE_RESPONSE_LANES
            ],
        ),
        "",
        "## Production Market Summary",
        "",
        markdown_table(
            ["Product", "State", "Correct / Graded", "Hit Rate", "Artifact Profit", "Leagues"],
            [
                [
                    row["display_name"],
                    row["status"],
                    f"{int(row['correct'])} / {int(row['graded'])}",
                    pct(row["hit_rate"]),
                    signed(row["profit"]),
                    str(row["league_count"]),
                ]
                for row in market_summary
            ],
        ),
        "",
        "## Correct Score Summary",
        "",
        markdown_table(
            ["Product", "State", "Exact / Fixtures", "Exact Hit", "Top 3 / Fixtures", "Top 3 Hit", "Leagues"],
            [
                [
                    cs_summary["display_name"],
                    cs_summary["status"],
                    f"{int(cs_summary['exact'])} / {int(cs_summary['fixtures'])}",
                    pct(cs_summary["exact_hit_rate"]),
                    f"{int(cs_summary['top3'])} / {int(cs_summary['fixtures'])}",
                    pct(cs_summary["top3_hit_rate"]),
                    str(cs_summary["league_count"]),
                ]
            ],
        ),
        "",
        "## Team Goals 1.5 Summary",
        "",
        "Team Goals 1.5 remains research/watch until future out-of-sample promotion checks are complete.",
        "",
        markdown_table(
            ["Product", "State", "Correct / Graded", "Hit Rate", "Leagues"],
            [
                [
                    row["display_name"],
                    row["status"],
                    f"{int(row['correct'])} / {int(row['graded'])}",
                    pct(row["hit_rate"]),
                    str(row["league_count"]),
                ]
                for row in tg15_summaries
            ],
        ),
        "",
    ]

    for product_key, title in (
        ("btts_calibrated", "BTTS Calibrated By League"),
        ("ou25_calibrated", "Over 2.5 Calibrated By League"),
    ):
        rows = [row for row in calibrated_rows if row["product_key"] == product_key]
        lines.extend(
            [
                f"## {title}",
                "",
                markdown_table(
                    ["League", "Rows", "Hit Rate", "Threshold Used"],
                    [
                        [
                            row["league"],
                            f"{int(row['rows']):,}",
                            pct(row["hit_rate"]),
                            str(row["threshold_used"]),
                        ]
                        for row in rows
                    ],
                ),
                "",
            ]
        )

    for product in ("over_25", "btts_yes", "btts_no", "ftr"):
        product_rows = [row for row in market_leagues if row["product_key"] == product]
        if not product_rows:
            continue
        lines.extend(
            [
                f"## {product_rows[0]['display_name']} By League",
                "",
                markdown_table(
                    ["League", "Correct / Graded", "Hit Rate", "Artifact Profit", "ELITE", "STANDARD"],
                    [
                        [
                            row["league"],
                            f"{int(row['correct'])} / {int(row['graded'])}",
                            pct(row["hit_rate"]),
                            signed(row["profit"]),
                            f"{int(row['elite_correct'])} / {int(row['elite_graded'])}",
                            f"{int(row['standard_correct'])} / {int(row['standard_graded'])}",
                        ]
                        for row in product_rows
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Correct Score Premium By League",
            "",
            markdown_table(
                ["League", "Exact / Fixtures", "Exact Hit", "Top 3 / Fixtures", "Top 3 Hit"],
                [
                    [
                        row["league"],
                        f"{int(row['exact'])} / {int(row['fixtures'])}",
                        pct(row["exact_hit_rate"]),
                        f"{int(row['top3'])} / {int(row['fixtures'])}",
                        pct(row["top3_hit_rate"]),
                    ]
                    for row in cs_leagues
                ],
            ),
            "",
        ]
    )

    for product in ("team_goal_15_tg15_premium", "team_goal_15_tg15_core_watch"):
        product_rows = [row for row in tg15_leagues if row["product_key"] == product]
        if not product_rows:
            continue
        lines.extend(
            [
                f"## {product_rows[0]['display_name']} By League",
                "",
                markdown_table(
                    ["League", "Correct / Graded", "Hit Rate", "Home", "Away"],
                    [
                        [
                            row["league"],
                            f"{int(row['correct'])} / {int(row['graded'])}",
                            pct(row["hit_rate"]),
                            f"{int(row['home_correct'])} / {int(row['home_graded'])}",
                            f"{int(row['away_correct'])} / {int(row['away_graded'])}",
                        ]
                        for row in product_rows
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Source Artifacts",
            "",
            f"- `{DEFAULT_PHASE8H_ROWS}`",
            f"- `{DEFAULT_CS_PREMIUM}`",
            f"- `{DEFAULT_TG15_LEAGUE_SPLITS}`",
            "",
            "## Product Notes",
            "",
            "- FTR, Over 2.5, BTTS Yes, and BTTS No are production/proven lanes in this report.",
            "- The calibrated BTTS and Over 2.5 sections are the headline accuracy layer for subscriber/investor proof.",
            "- The production market summary is the broader settled deploy-lane view, so its hit rates can be lower than the calibrated threshold proof layer.",
            "- Artifact profit is carried through where the settled Phase 8H artifact contains odds; website/investor headline copy should lead with hit rates and volume unless the odds source is explicitly disclosed.",
            "- The value edge system is additive only: it helps rank commercial edge but does not rescue weak lanes or override production gates.",
            "- Correct Score is a premium product layer: top-3 accuracy is the main user-facing proof; exact-score hit rate is premium context.",
            "- Team Goals 1.5 is promising but remains research/watch until future out-of-sample windows confirm promotion.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build product market league breakdown report.")
    parser.add_argument("--phase8h-rows", default=str(DEFAULT_PHASE8H_ROWS))
    parser.add_argument("--correct-score-premium", default=str(DEFAULT_CS_PREMIUM))
    parser.add_argument("--tg15-league-splits", default=str(DEFAULT_TG15_LEAGUE_SPLITS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    market_summary, market_leagues = aggregate_phase8h_markets(read_csv_rows(Path(args.phase8h_rows)))
    cs_summary, cs_leagues = aggregate_correct_score(read_csv_rows(Path(args.correct_score_premium)))
    tg15_summaries = []
    tg15_leagues: list[dict[str, Any]] = []
    tg15_rows = read_csv_rows(Path(args.tg15_league_splits))
    for policy in ("TG15_PREMIUM", "TG15_CORE_WATCH"):
        summary, league_rows = aggregate_tg15(tg15_rows, policy)
        tg15_summaries.append(summary)
        tg15_leagues.extend(league_rows)
    calibrated_rows = calibrated_league_rows()

    write_csv(output_dir / "PRODUCT_MARKET_SUMMARY.csv", market_summary + [cs_summary] + tg15_summaries)
    write_csv(output_dir / "PRODUCT_MARKET_LEAGUE_BREAKDOWN.csv", market_leagues + cs_leagues + tg15_leagues)
    write_csv(output_dir / "PRODUCT_CALIBRATED_LEAGUE_BREAKDOWN.csv", calibrated_rows)
    write_csv(output_dir / "PRODUCT_VALUE_EDGE_SUMMARY.csv", list(VALUE_EDGE_SUMMARY) + list(VALUE_RESPONSE_LANES))
    (output_dir / "PRODUCT_MARKET_LEAGUE_BREAKDOWN.md").write_text(
        build_markdown(
            market_summary,
            market_leagues,
            cs_summary,
            cs_leagues,
            tg15_summaries,
            tg15_leagues,
            calibrated_rows,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
