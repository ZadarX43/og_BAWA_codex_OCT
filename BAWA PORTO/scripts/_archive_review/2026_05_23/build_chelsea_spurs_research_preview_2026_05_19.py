#!/usr/bin/env python3
"""Build a one-fixture Odds Genius research preview for Chelsea vs Tottenham.

This is deliberately report-only: it does not touch deploy routing or slip
generation, and player-event outputs remain beta/research.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "reports/latest/chelsea_spurs_2026_05_19_odds_genius_preview"
ALLMARKETS = ROOT / (
    "predictions_output/chelsea_spurs_2026_05_19_research/"
    "BOOKIE_IMP20_ALLMARKETS_2026-05-19_to_2026-05-19.csv"
)
PLAYER_STATS = ROOT / (
    "Players/England Premier League/"
    "england-premier-league-players-2025-to-2026-stats.csv"
)
MERGED = ROOT / "Matches/__merged__/England_Premier_League__merged.csv"


POSSIBLE_XI = [
    ("Chelsea", "Robert Sanchez", "Goalkeeper"),
    ("Chelsea", "Wesley Fofana", "Defender"),
    ("Chelsea", "Trevoh Chalobah", "Defender"),
    ("Chelsea", "Jorrel Hato", "Defender"),
    ("Chelsea", "Malo Gusto", "Defender"),
    ("Chelsea", "Moisés Caicedo", "Midfielder"),
    ("Chelsea", "Enzo Fernández", "Midfielder"),
    ("Chelsea", "Marc Cucurella", "Defender"),
    ("Chelsea", "Cole Palmer", "Midfielder"),
    ("Chelsea", "Pedro Neto", "Forward"),
    ("Chelsea", "Liam Delap", "Forward"),
    ("Tottenham Hotspur", "Antonin Kinsky", "Goalkeeper"),
    ("Tottenham Hotspur", "Pedro Porro", "Defender"),
    ("Tottenham Hotspur", "Kevin Danso", "Defender"),
    ("Tottenham Hotspur", "Micky van de Ven", "Defender"),
    ("Tottenham Hotspur", "Destiny Udogie", "Defender"),
    ("Tottenham Hotspur", "Rodrigo Bentancur", "Midfielder"),
    ("Tottenham Hotspur", "João Palhinha", "Midfielder"),
    ("Tottenham Hotspur", "Randal Kolo Muani", "Forward"),
    ("Tottenham Hotspur", "Conor Gallagher", "Midfielder"),
    ("Tottenham Hotspur", "Mathys Tel", "Forward"),
    ("Tottenham Hotspur", "Richarlison", "Forward"),
]


def poisson_ge(lam: float, threshold: int) -> float:
    """P(X >= threshold) for Poisson(lambda)."""
    return 1.0 - sum(math.exp(-lam) * lam**k / math.factorial(k) for k in range(threshold))


def poisson_total_over(lam_total: float, line: float) -> float:
    if line != 2.5:
        raise ValueError("only over 2.5 is needed for this fixture preview")
    return poisson_ge(lam_total, 3)


def norm_name(name: str) -> str:
    return (
        name.lower()
        .replace("í", "i")
        .replace("é", "e")
        .replace("ã", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ó", "o")
    )


def load_player_row(stats: pd.DataFrame, player_name: str) -> pd.Series | None:
    target = norm_name(player_name)
    names = stats["full_name"].astype(str).map(norm_name)
    exact = stats[names == target]
    if not exact.empty:
        return exact.iloc[0]
    parts = [p for p in target.split() if len(p) > 2]
    if not parts:
        return None
    mask = names.apply(lambda n: all(p in n for p in parts[-2:]))
    found = stats[mask]
    if not found.empty:
        return found.iloc[0]
    mask = names.apply(lambda n: parts[-1] in n)
    found = stats[mask]
    if not found.empty:
        return found.iloc[0]
    return None


def val(row: pd.Series | None, col: str, default: float = 0.0) -> float:
    if row is None or col not in row or pd.isna(row[col]):
        return default
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return default


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row.get(col, "")).replace("|", "/") for col in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    allmarkets = pd.read_csv(ALLMARKETS)
    player_stats = pd.read_csv(PLAYER_STATS)

    fixture = allmarkets[
        (allmarkets["home_team_name"].astype(str) == "Chelsea")
        & (allmarkets["away_team_name"].astype(str) == "Tottenham Hotspur")
    ].iloc[0]

    lam_home = float(fixture["lambda_home"])
    lam_away = float(fixture["lambda_away"])
    lam_total = lam_home + lam_away
    p_home_ge1 = poisson_ge(lam_home, 1)
    p_away_ge1 = poisson_ge(lam_away, 1)
    p_home_ge2 = poisson_ge(lam_home, 2)
    p_away_ge2 = poisson_ge(lam_away, 2)
    p_btts = 1 - math.exp(-lam_home) - math.exp(-lam_away) + math.exp(-lam_total)
    p_over25 = poisson_total_over(lam_total, 2.5)

    core_rows = [
        {
            "market": "FTR",
            "prediction": "Chelsea lean, but audit-only as a straight FTR single",
            "model_probability": round(float(fixture["p_home_pois"]), 4),
            "confidence": "LOW-MEDIUM",
            "source": "Odds Genius FTR + Poisson goal-mass",
            "notes": (
                "Cat/Poisson lean home; XGB warns away. Top scoreline is 1-1, "
                "so this is not a clean deploy-style home single."
            ),
        },
        {
            "market": "BTTS",
            "prediction": "YES lean",
            "model_probability": round(p_btts, 4),
            "confidence": "MEDIUM",
            "source": "Goal-mass derived from allmarkets lambdas",
            "notes": "Both season BTTS rates are 61%; lambdas imply both sides over 0.5 are live.",
        },
        {
            "market": "Over 2.5",
            "prediction": "OVER 2.5 slight lean, but not a premium single",
            "model_probability": round(p_over25, 4),
            "confidence": "LOW-MEDIUM",
            "source": "Goal-mass derived from allmarkets lambdas",
            "notes": "2.93 expected goals supports over, but team-news context and 1-1 top scoreline cap conviction.",
        },
        {
            "market": "Chelsea team goals over 0.5",
            "prediction": "YES",
            "model_probability": round(p_home_ge1, 4),
            "confidence": "MEDIUM-HIGH",
            "source": "Poisson goal-mass",
            "notes": "Chelsea lambda 1.71.",
        },
        {
            "market": "Tottenham team goals over 0.5",
            "prediction": "YES",
            "model_probability": round(p_away_ge1, 4),
            "confidence": "MEDIUM",
            "source": "Poisson goal-mass",
            "notes": "Tottenham lambda 1.22.",
        },
        {
            "market": "Chelsea team goals over 1.5",
            "prediction": "WATCH only",
            "model_probability": round(p_home_ge2, 4),
            "confidence": "LOW-MEDIUM",
            "source": "Poisson goal-mass",
            "notes": "Barely above coin-flip; Joao Pedro doubt lowers confidence.",
        },
        {
            "market": "Tottenham team goals over 1.5",
            "prediction": "NO lean",
            "model_probability": round(p_away_ge2, 4),
            "confidence": "MEDIUM",
            "source": "Poisson goal-mass",
            "notes": "Spurs attacking absences make 2+ less attractive despite away form.",
        },
        {
            "market": "Correct score cluster",
            "prediction": f"{fixture['cs1']}, {fixture['cs2']}, {fixture['cs3']}",
            "model_probability": round(float(fixture["cs1_p"]), 4),
            "confidence": "Research",
            "source": "Allmarkets score-mass",
            "notes": "Top scoreline probability shown for first listed score only.",
        },
    ]
    core_df = pd.DataFrame(core_rows)
    core_path = OUTDIR / "chelsea_spurs_core_markets.csv"
    core_df.to_csv(core_path, index=False)

    event_rows: list[dict[str, object]] = []
    for team, player, role in POSSIBLE_XI:
        row = load_player_row(player_stats, player)
        shots90 = val(row, "shots_per_90_overall")
        sot90 = val(row, "shots_on_target_per_90_overall")
        tackles90 = val(row, "tackles_per_90_overall")
        cards90 = val(row, "cards_per_90_overall")
        booked_pct = val(row, "booked_over05_percentage_overall")
        rating = val(row, "average_rating_overall")
        xg = val(row, "xg_per_game_overall")
        key_pass = val(row, "key_passes_per_game_overall")
        saves = val(row, "saves_per_game_overall")
        shots_faced = val(row, "shots_faced_per_game_overall")

        markets = []
        if role != "Goalkeeper":
            if shots90 >= 2.0:
                markets.append(("shots_over_1_5", "STRONG"))
            elif shots90 >= 1.0:
                markets.append(("shots_over_0_5", "WATCH"))
            if sot90 >= 0.75:
                markets.append(("sot_over_0_5", "STRONG"))
            elif sot90 >= 0.35:
                markets.append(("sot_over_0_5", "WATCH"))
            if tackles90 >= 2.4:
                markets.append(("tackles_over_1_5", "STRONG"))
            elif tackles90 >= 1.7:
                markets.append(("tackles_over_1_5", "WATCH"))
            if cards90 >= 0.30 or booked_pct >= 25:
                markets.append(("card_0_5_hazard", "HIGH"))
            elif cards90 >= 0.18 or booked_pct >= 15:
                markets.append(("card_0_5_hazard", "WATCH"))
        else:
            if team == "Tottenham Hotspur":
                markets.append(("keeper_saves_over_1_5", "WATCH"))
                markets.append(("keeper_saves_over_2_5", "LOW-WATCH"))
            elif team == "Chelsea":
                markets.append(("keeper_saves_over_1_5", "LOW-WATCH"))

        if not markets:
            markets.append(("context_only", "LOW"))

        for market, grade in markets:
            event_rows.append(
                {
                    "team": team,
                    "player": player,
                    "position_group": role,
                    "event_market": market,
                    "grade": grade,
                    "shots_per90": round(shots90, 2),
                    "sot_per90": round(sot90, 2),
                    "tackles_per90": round(tackles90, 2),
                    "cards_per90": round(cards90, 2),
                    "booked_over05_pct": round(booked_pct, 1),
                    "saves_per_game": round(saves, 2),
                    "shots_faced_per_game": round(shots_faced, 2),
                    "xg_per_game": round(xg, 2),
                    "key_passes_per_game": round(key_pass, 2),
                    "avg_rating": round(rating, 2),
                    "source_status": "matched_player_stats" if row is not None else "lineup_context_only",
                }
            )

    event_df = pd.DataFrame(event_rows)
    event_path = OUTDIR / "chelsea_spurs_player_event_board.csv"
    event_df.to_csv(event_path, index=False)

    top_event = event_df[event_df["event_market"] != "context_only"].copy()
    grade_order = {"STRONG": 0, "HIGH": 1, "MEDIUM": 2, "WATCH": 3, "LOW-WATCH": 4, "LOW": 5}
    top_event["_grade_order"] = top_event["grade"].map(grade_order).fillna(9)
    top_event = top_event.sort_values(["_grade_order", "event_market", "player"]).drop(columns="_grade_order")

    report = OUTDIR / "CHELSEA_SPURS_ODDS_GENIUS_PREVIEW.md"
    report.write_text(
        "\n".join(
            [
                "# Chelsea vs Tottenham Hotspur - Odds Genius Research Preview",
                "",
                "**Fixture:** Chelsea vs Tottenham Hotspur, 2026-05-19 19:15",
                "",
                "**Status:** Research/live-preview pack. QA gate passed with unrelated warnings; player-event outputs remain beta and pre-lineup.",
                "",
                "## Source Files",
                "",
                f"- Allmarkets run: `{ALLMARKETS.relative_to(ROOT)}`",
                f"- Canonical merged input: `{MERGED.relative_to(ROOT)}`",
                f"- Player stats input: `{PLAYER_STATS.relative_to(ROOT)}`",
                "- Team news / possible XI: user-supplied SportsMole preview, not confirmed lineups.",
                "",
                "## Core Market Read",
                "",
                f"- Goal-mass: Chelsea {lam_home:.2f}, Tottenham {lam_away:.2f}, total {lam_total:.2f}.",
                f"- FTR Poisson: home {float(fixture['p_home_pois']):.1%}, draw {float(fixture['p_draw_pois']):.1%}, away {float(fixture['p_away_pois']):.1%}.",
                f"- Cat FTR pick: {fixture['model_top_pick']} ({float(fixture['model_p_for_bookie']):.1%} for the home book pick).",
                f"- XGB FTR warning: {fixture['model_top_pick_xgb']} lane; this blocks a clean Chelsea single.",
                f"- Score-mass cluster: {fixture['cs1']} ({float(fixture['cs1_p']):.1%}), {fixture['cs2']} ({float(fixture['cs2_p']):.1%}), {fixture['cs3']} ({float(fixture['cs3_p']):.1%}).",
                "",
                "### Decision",
                "",
                "- **FTR:** Chelsea lean, audit-only as a straight single. Safer interpretation is Chelsea/draw protection rather than a hard home win.",
                "- **BTTS:** Yes lean. This is cleaner than the FTR single.",
                "- **Over 2.5:** Slight over lean from 2.93 goal-mass, but not premium because 1-1 is the top scoreline and Chelsea have recent goal drag.",
                "- **Team goals:** Chelsea over 0.5 and Spurs over 0.5 are the cleanest goal-market shapes; Chelsea over 1.5 is watch-only.",
                "",
                "## Player Event Board",
                "",
                "Top beta/research signals from possible XI and 2025/26 player profiles:",
                "",
                markdown_table(top_event.head(30)),
                "",
                "## Caveats",
                "",
                "- OU25 and BTTS dedicated allmarkets rows did not emit because the merged fixture lacks those book odds. The read above uses the same Odds Genius goal-mass lambdas instead.",
                "- Player-event predictions are not priced bets and should be refreshed after confirmed lineups.",
                "- Joao Pedro doubt, Spurs attacking injuries, and the derby/card environment materially affect conviction.",
                "",
                "## Saved Outputs",
                "",
                f"- `{core_path.relative_to(ROOT)}`",
                f"- `{event_path.relative_to(ROOT)}`",
                f"- `{report.relative_to(ROOT)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"WROTE {core_path.relative_to(ROOT)} rows={len(core_df)}")
    print(f"WROTE {event_path.relative_to(ROOT)} rows={len(event_df)}")
    print(f"WROTE {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
