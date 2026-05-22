#!/usr/bin/env python3
"""Build the first World Cup 2026 goal-market prediction shortlist.

Research-only launch prep. This takes the manually curated shortlist crosscheck
and writes a clean goal-market board across FTR, Over 2.5, BTTS, team goals,
and sensible combo candidates. It deliberately marks all rows as pending
official squad/injury/lineup truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_sources" / "footystats_world_cup" / "shortlist_user_crosscheck_2026_05_19.csv"
DEFAULT_OUTDIR = ROOT / "data_sources" / "footystats_world_cup" / "initial_goal_market_predictions_2026"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def side_for_market(row: pd.Series, market: str) -> str | None:
    market = str(market)
    if market in {"home_win", "france_tg15", "germany_tg15", "spain_tg15", "brazil_tg15"}:
        return "HOME"
    if market == "away_win":
        return "AWAY"
    return None


def o25_band(row: pd.Series) -> str:
    prob = as_float(row.get("prob_o25"))
    hint = str(row.get("goal_hint", ""))
    if prob >= 0.56 or "STRONG_BTTS_OU25" in hint:
        return "SUPPORTED"
    if prob >= 0.52 or "BTTS_OU25" in hint:
        return "WATCH"
    return "AVOID_OR_PRICE_ONLY"


def btts_band(row: pd.Series) -> str:
    prob = as_float(row.get("prob_btts"))
    hint = str(row.get("goal_hint", ""))
    if prob >= 0.54 or "STRONG_BTTS_OU25" in hint:
        return "SUPPORTED"
    if prob >= 0.50 or "BTTS_OU25" in hint:
        return "WATCH"
    return "AVOID_OR_PRICE_ONLY"


def tg15_side(row: pd.Series) -> tuple[str, str]:
    hint = str(row.get("goal_hint", ""))
    if "HOME_TG15" in hint:
        return str(row.get("api_home")), "SUPPORTED"
    if "AWAY_TG15" in hint:
        return str(row.get("api_away")), "SUPPORTED"
    if str(row.get("pp_edge")) == "HOME_PLAYER_POWER_EDGE":
        return str(row.get("api_home")), "WATCH"
    if str(row.get("pp_edge")) == "AWAY_PLAYER_POWER_EDGE":
        return str(row.get("api_away")), "WATCH"
    return "", "NO_CLEAR_EDGE"


def manual_tg15_side(row: pd.Series, fixture_markets: set[str]) -> tuple[str, str]:
    home = str(row.get("api_home", ""))
    away = str(row.get("api_away", ""))
    home_slug = home.lower().replace(" ", "_")
    away_slug = away.lower().replace(" ", "_")
    for market in fixture_markets:
        text = str(market).lower()
        if not text.endswith("_tg15"):
            continue
        prefix = text.removesuffix("_tg15")
        if prefix and (home_slug.startswith(prefix) or prefix in home_slug):
            return home, "WATCH"
        if prefix and (away_slug.startswith(prefix) or prefix in away_slug):
            return away, "WATCH"
    return "", "NO_CLEAR_EDGE"


def ftr_recommendation(row: pd.Series) -> tuple[str, str]:
    pick = str(row.get("macro_pick", "")).upper()
    if pick == "HOME":
        side = str(row.get("api_home"))
    elif pick == "AWAY":
        side = str(row.get("api_away"))
    else:
        return "", "NO_FTR_EDGE"
    risk = str(row.get("ftr_risk", ""))
    draw_risk = int(as_float(row.get("draw_risk")))
    hint = str(row.get("ftr_hint", ""))
    if risk == "STRONG" and draw_risk == 0 and ("SIDE_PLAYER_POWER_SUPPORT" in hint or "WEAK_SOURCE" in hint):
        return side, "SUPPORTED"
    if risk == "STRONG" and draw_risk == 0:
        return side, "WATCH"
    return side, "CAUTION"


def rank_label(*bands: str) -> str:
    bands = tuple(str(b) for b in bands)
    if any(b.startswith("AVOID") or b in {"NO_CLEAR_EDGE", "CAUTION"} for b in bands):
        return "PRICE_ONLY_OR_AVOID"
    if all(b == "SUPPORTED" for b in bands):
        return "STRONG_COMBO"
    if "SUPPORTED" in bands and "WATCH" in bands:
        return "WATCH_COMBO"
    return "RESEARCH_ONLY"


def build_boards(input_csv: Path, outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_csv, low_memory=False)
    raw = raw[raw["found"].astype(bool)].copy()
    fixtures = raw.drop_duplicates(["api_home", "api_away"]).copy()

    fixture_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    for _, row in fixtures.iterrows():
        fixture_markets = set(
            raw[
                raw["api_home"].eq(row["api_home"])
                & raw["api_away"].eq(row["api_away"])
            ]["market"].astype(str)
        )
        fixture_key = f"{row['api_date']}_{row['api_home']}_{row['api_away']}"
        over25 = o25_band(row)
        btts = btts_band(row)
        tg_team, tg_band = tg15_side(row)
        manual_tg_team, manual_tg_band = manual_tg15_side(row, fixture_markets)
        if tg_band == "NO_CLEAR_EDGE" and manual_tg_team:
            tg_team, tg_band = manual_tg_team, manual_tg_band
        ftr_side, ftr_band = ftr_recommendation(row)
        fixture_rows.append(
            {
                "fixture_key": fixture_key,
                "api_date": row.get("api_date"),
                "api_round": row.get("api_round"),
                "home_team": row.get("api_home"),
                "away_team": row.get("api_away"),
                "macro_ftr_pick": row.get("macro_pick"),
                "macro_ftr_confidence": row.get("macro_conf"),
                "ftr_risk": row.get("ftr_risk"),
                "draw_stalemate_risk": row.get("draw_risk"),
                "macro_prob_over25": row.get("prob_o25"),
                "over25_band": over25,
                "macro_prob_btts_yes": row.get("prob_btts"),
                "btts_yes_band": btts,
                "tg15_team": tg_team,
                "tg15_band": tg_band,
                "ftr_team": ftr_side,
                "ftr_band": ftr_band,
                "player_power_edge": row.get("pp_edge"),
                "player_power_goal_hint": row.get("goal_hint"),
                "player_power_ftr_hint": row.get("ftr_hint"),
                "lineup_truth_status": "PENDING_OFFICIAL_SQUAD_INJURY_LINEUP_LAYER",
                "research_status": "PRE_TOURNAMENT_RESEARCH_ONLY",
            }
        )
        markets = [
            ("FTR", ftr_side, ftr_band, as_float(row.get("macro_conf"))),
            ("TEAM_OVER_1_5", tg_team, tg_band, pd.NA),
            ("OVER_2_5", "MATCH", over25, as_float(row.get("prob_o25"))),
            ("BTTS_YES", "MATCH", btts, as_float(row.get("prob_btts"))),
        ]
        for market, selection, band, score in markets:
            market_rows.append(
                {
                    "fixture_key": fixture_key,
                    "api_date": row.get("api_date"),
                    "api_round": row.get("api_round"),
                    "home_team": row.get("api_home"),
                    "away_team": row.get("api_away"),
                    "market": market,
                    "selection": selection,
                    "prediction_band": band,
                    "model_score": score,
                    "lineup_truth_status": "PENDING_OFFICIAL_SQUAD_INJURY_LINEUP_LAYER",
                    "research_status": "PRE_TOURNAMENT_RESEARCH_ONLY",
                }
            )
        combo_specs = []
        if ftr_side and tg_team and ftr_side == tg_team:
            combo_specs.append(("FTR_PLUS_TG15", f"{ftr_side} win + {tg_team} over 1.5 team goals", rank_label(ftr_band, tg_band)))
        combo_specs += [
            ("FTR_PLUS_OVER25", f"{ftr_side} win + Over 2.5 match goals", rank_label(ftr_band, over25)),
            ("TG15_PLUS_OVER25", f"{tg_team} over 1.5 team goals + Over 2.5", rank_label(tg_band, over25)),
            ("FTR_PLUS_BTTS", f"{ftr_side} win + BTTS Yes", rank_label(ftr_band, btts)),
        ]
        for combo_market, selection, band in combo_specs:
            if not selection or selection.startswith(" win") or selection.startswith(" over"):
                continue
            combo_rows.append(
                {
                    "fixture_key": fixture_key,
                    "api_date": row.get("api_date"),
                    "api_round": row.get("api_round"),
                    "home_team": row.get("api_home"),
                    "away_team": row.get("api_away"),
                    "combo_market": combo_market,
                    "combo_selection": selection,
                    "combo_band": band,
                    "lineup_truth_status": "PENDING_OFFICIAL_SQUAD_INJURY_LINEUP_LAYER",
                    "research_status": "PRE_TOURNAMENT_RESEARCH_ONLY",
                }
            )

    fixtures_df = pd.DataFrame(fixture_rows)
    markets_df = pd.DataFrame(market_rows)
    combos_df = pd.DataFrame(combo_rows)
    fixtures_path = outdir / "world_cup_2026_initial_goal_market_fixture_board.csv"
    markets_path = outdir / "world_cup_2026_initial_goal_market_predictions_long.csv"
    combos_path = outdir / "world_cup_2026_initial_goal_market_combo_predictions.csv"
    fixtures_df.to_csv(fixtures_path, index=False)
    markets_df.to_csv(markets_path, index=False)
    combos_df.to_csv(combos_path, index=False)
    write_summary(outdir, fixtures_df, markets_df, combos_df)
    return {"fixtures": fixtures_path, "markets": markets_path, "combos": combos_path, "summary": outdir / "SUMMARY.md"}


def write_summary(outdir: Path, fixtures: pd.DataFrame, markets: pd.DataFrame, combos: pd.DataFrame) -> None:
    lines = [
        "# World Cup 2026 Initial Goal-Market Prediction List",
        "",
        "- First-pass research shortlist from the manually curated candidate list plus current macro/player-power context.",
        "- Not priced bets. Not production routing. All rows await official squad, injury, and lineup updates.",
        "- Designed to be refreshed after player announcements and again after confirmed lineups.",
        "",
        "## Market Counts",
    ]
    market_counts = markets.groupby(["market", "prediction_band"], dropna=False).size().reset_index(name="rows")
    for _, row in market_counts.iterrows():
        lines.append(f"- {row['market']} | {row['prediction_band']} | rows={int(row['rows'])}")
    lines.append("")
    lines.append("## Combo Counts")
    combo_counts = combos.groupby(["combo_market", "combo_band"], dropna=False).size().reset_index(name="rows")
    for _, row in combo_counts.iterrows():
        lines.append(f"- {row['combo_market']} | {row['combo_band']} | rows={int(row['rows'])}")
    lines.append("")
    lines.append("## Supported Goal-Market Reads")
    supported = markets[markets["prediction_band"].eq("SUPPORTED")].copy()
    if supported.empty:
        lines.append("- None yet.")
    else:
        for _, row in supported.sort_values(["market", "api_date"]).iterrows():
            lines.append(
                f"- {row['home_team']} vs {row['away_team']} | {row['market']} | {row['selection']}"
            )
    lines.append("")
    lines.append("## Next Update Trigger")
    lines.append("- Refresh once official squads/player announcements land.")
    lines.append("- Refresh again after confirmed lineups and injuries for each fixture.")
    lines.append("- During the tournament, upgrade after every matchday with same-tournament player ratings, SOT, saves, tackles, fouls, and bookings.")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build initial World Cup 2026 goal-market prediction shortlist.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()
    outputs = build_boards(Path(args.input), Path(args.outdir))
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
