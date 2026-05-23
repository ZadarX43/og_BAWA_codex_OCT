from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_PREVIEW_DIR = Path("reports/latest/chelsea_spurs_2026_05_19_odds_genius_preview")
DEFAULT_API_DIR = Path("reports/latest/chelsea_spurs_2026_05_19_api_fixture_audit/normalized")
DEFAULT_OUTDIR = Path("reports/latest/chelsea_spurs_2026_05_19_post_match_prediction_audit")


def norm_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return text


def load_player_stats(api_dir: Path) -> pd.DataFrame:
    path = api_dir / "match_player_stats__England_Premier_League__2025.csv"
    df = pd.read_csv(path)
    df["player_key"] = df["player_name"].map(norm_name)
    df["cards_total"] = df["yellow_cards"].fillna(0) + df["red_cards"].fillna(0)
    return df


def load_team_stats(api_dir: Path) -> pd.DataFrame:
    return pd.read_csv(api_dir / "match_team_stats__England_Premier_League__2025.csv")


def actual_for_market(row: pd.Series, market: str) -> tuple[float | None, bool | None, str]:
    if pd.isna(row.get("player_name")):
        return None, None, "NO_PLAYER_MATCH"
    market = str(market)
    if market == "context_only":
        return None, None, "CONTEXT_ONLY"
    checks = {
        "shots_over_0_5": ("shots_total", 1),
        "shots_over_1_5": ("shots_total", 2),
        "sot_over_0_5": ("shots_on_target", 1),
        "tackles_over_1_5": ("tackles", 2),
        "card_0_5_hazard": ("cards_total", 1),
        "keeper_saves_over_1_5": ("saves", 2),
        "keeper_saves_over_2_5": ("saves", 3),
    }
    if market not in checks:
        return None, None, f"UNSUPPORTED_MARKET:{market}"
    col, threshold = checks[market]
    actual = float(row.get(col, 0) or 0)
    return actual, bool(actual >= threshold), f"{col}>={threshold}"


def score_light_profile(preview_dir: Path, players: pd.DataFrame) -> pd.DataFrame:
    board = pd.read_csv(preview_dir / "chelsea_spurs_player_event_board.csv")
    board["player_key"] = board["player"].map(norm_name)
    actual = players[
        [
            "player_key",
            "player_name",
            "team_id",
            "minutes",
            "shots_total",
            "shots_on_target",
            "tackles",
            "fouls_drawn",
            "fouls_committed",
            "cards_total",
            "saves",
        ]
    ]
    out = board.merge(actual, on="player_key", how="left", suffixes=("", "_api"))
    scored = out.apply(lambda r: actual_for_market(r, r["event_market"]), axis=1, result_type="expand")
    out["actual_value"] = scored[0]
    out["hit"] = scored[1]
    out["scoring_rule"] = scored[2]
    return out


def parse_expression(expression: str) -> tuple[str, float] | tuple[None, None]:
    expr = str(expression)
    m = re.search(r"Player (Shots|SOT|Fouled) ([0-9.]+)\+", expr)
    if not m:
        return None, None
    return m.group(1), float(m.group(2))


def score_upgraded_interactions(preview_dir: Path, players: pd.DataFrame) -> pd.DataFrame:
    path = (
        preview_dir
        / "upgraded_interaction_confirmed_lineups"
        / "player_event_interaction_live_shadow_board_exact"
        / "PLAYER_EVENT_INTERACTION_LIVE_SHADOW_BOARD.csv"
    )
    board = pd.read_csv(path)
    board["player_key"] = board["source_selection"].map(norm_name)
    actual = players[
        [
            "player_key",
            "player_name",
            "minutes",
            "shots_total",
            "shots_on_target",
            "fouls_drawn",
            "fouls_committed",
            "tackles",
            "cards_total",
        ]
    ]
    out = board.merge(actual, on="player_key", how="left", suffixes=("", "_api"))
    values = []
    hits = []
    rules = []
    for _, row in out.iterrows():
        family, threshold = parse_expression(row.get("expression", ""))
        if family is None:
            values.append(None)
            hits.append(None)
            rules.append("UNSUPPORTED_EXPRESSION")
            continue
        if pd.isna(row.get("player_name")):
            values.append(None)
            hits.append(None)
            rules.append("NO_PLAYER_MATCH")
            continue
        col = {"Shots": "shots_total", "SOT": "shots_on_target", "Fouled": "fouls_drawn"}[family]
        required = int(threshold + 0.5)
        actual = float(row.get(col, 0) or 0)
        values.append(actual)
        hits.append(bool(actual >= required))
        rules.append(f"{col}>={required}")
    out["actual_value"] = values
    out["hit"] = hits
    out["scoring_rule"] = rules
    return out


def score_comparison_tips(preview_dir: Path, players: pd.DataFrame) -> pd.DataFrame:
    path = preview_dir / "upgraded_interaction_confirmed_lineups" / "LIGHT_PROFILE_BOARD_vs_UPGRADED_INTERACTION_BOARD.csv"
    tips = pd.read_csv(path)
    tips["player_key"] = tips["player"].map(norm_name)
    actual = players[
        [
            "player_key",
            "player_name",
            "minutes",
            "shots_total",
            "shots_on_target",
            "fouls_drawn",
            "fouls_committed",
            "tackles",
            "cards_total",
        ]
    ]
    out = tips.merge(actual, on="player_key", how="left", suffixes=("", "_api"))
    rows = []
    for _, row in out.iterrows():
        family = str(row.get("market_family", ""))
        threshold = float(row.get("threshold", 0.5) or 0.5)
        if pd.isna(row.get("player_name")):
            rows.append((None, None, "NO_PLAYER_MATCH"))
            continue
        if family == "PLAYER_SHOTS":
            col = "shots_total"
        elif family == "PLAYER_SOT":
            col = "shots_on_target"
        elif family == "PLAYER_CARDS":
            col = "cards_total"
        elif family == "PLAYER_FOULED":
            col = "fouls_drawn"
        else:
            rows.append((None, None, f"UNSUPPORTED_MARKET_FAMILY:{family}"))
            continue
        required = int(threshold + 0.5)
        actual = float(row.get(col, 0) or 0)
        rows.append((actual, bool(actual >= required), f"{col}>={required}"))
    scored = pd.DataFrame(rows, columns=["actual_value", "hit", "scoring_rule"])
    return pd.concat([out, scored], axis=1)


def score_core_markets(preview_dir: Path, team_stats: pd.DataFrame) -> pd.DataFrame:
    core = pd.read_csv(preview_dir / "chelsea_spurs_core_markets.csv")
    home = team_stats[team_stats["is_home"] == 1].iloc[0]
    away = team_stats[team_stats["is_home"] == 0].iloc[0]
    hg = int(home["goals_for"])
    ag = int(away["goals_for"])
    total = hg + ag
    actuals = {
        "FTR": ("HOME", "Chelsea win", hg > ag),
        "BTTS": ("YES", "Both teams scored", hg > 0 and ag > 0),
        "Over 2.5": ("OVER", "Total goals >= 3", total >= 3),
        "Chelsea team goals over 0.5": ("YES", "Chelsea goals >= 1", hg >= 1),
        "Tottenham team goals over 0.5": ("YES", "Tottenham goals >= 1", ag >= 1),
        "Chelsea team goals over 1.5": ("YES", "Chelsea goals >= 2", hg >= 2),
        "Tottenham team goals over 1.5": ("NO", "Tottenham goals < 2", ag < 2),
        "Correct score cluster": ("2-1", "Final score in listed cluster", f"{hg}-{ag}" in {"1-1", "2-1", "1-0"}),
    }
    out = core.copy()
    out["actual_result"] = out["market"].map(lambda m: actuals.get(m, ("", "", None))[0])
    out["actual_detail"] = out["market"].map(lambda m: actuals.get(m, ("", "", None))[1])
    out["hit"] = out["market"].map(lambda m: actuals.get(m, ("", "", None))[2])
    out["final_score"] = f"{hg}-{ag}"
    return out


def write_summary(
    outdir: Path,
    core: pd.DataFrame,
    light: pd.DataFrame,
    upgraded: pd.DataFrame,
    tips: pd.DataFrame,
    players: pd.DataFrame,
    team_stats: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        view = df.fillna("")
        cols = list(view.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in view.iterrows():
            vals = [str(row[c]).replace("|", "\\|") for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def hit_rate(df: pd.DataFrame, mask: pd.Series) -> tuple[int, int, float]:
        subset = df[mask & df["hit"].notna()]
        total = len(subset)
        hits = int(subset["hit"].sum()) if total else 0
        return hits, total, (hits / total if total else 0.0)

    light_scored = light["hit"].notna()
    upgraded_scored = upgraded["hit"].notna()
    strong_light = light["grade"].isin(["STRONG", "HIGH"])
    watch_light = light["grade"].isin(["WATCH", "LOW-WATCH"])
    core_hits, core_total, core_rate = hit_rate(core, core["hit"].notna())
    light_hits, light_total, light_rate = hit_rate(light, light_scored)
    strong_hits, strong_total, strong_rate = hit_rate(light, strong_light)
    watch_hits, watch_total, watch_rate = hit_rate(light, watch_light)
    up_hits, up_total, up_rate = hit_rate(upgraded, upgraded_scored)
    tip_hits, tip_total, tip_rate = hit_rate(tips, tips["hit"].notna())

    cards = players.loc[players["cards_total"] > 0, ["player_name", "team_id", "minutes", "cards_total"]]
    top_events = players[
        [
            "player_name",
            "team_id",
            "minutes",
            "shots_total",
            "shots_on_target",
            "tackles",
            "fouls_drawn",
            "fouls_committed",
            "cards_total",
            "saves",
        ]
    ].sort_values(["shots_total", "shots_on_target", "tackles"], ascending=False)

    report = [
        "# Chelsea vs Tottenham Post-Match Prediction Audit",
        "",
        "Fixture: Chelsea 2-1 Tottenham, 2026-05-19, API-Football fixture `1379333`.",
        "",
        "## Scorecard",
        "",
        f"- Core goal markets: {core_hits}/{core_total} ({core_rate:.1%}).",
        f"- Light profile player-event board: {light_hits}/{light_total} ({light_rate:.1%}).",
        f"- Light profile STRONG/HIGH rows: {strong_hits}/{strong_total} ({strong_rate:.1%}).",
        f"- Light profile WATCH/LOW-WATCH rows: {watch_hits}/{watch_total} ({watch_rate:.1%}).",
        f"- Upgraded interaction board: {up_hits}/{up_total} ({up_rate:.1%}).",
        f"- Named comparison tip set: {tip_hits}/{tip_total} ({tip_rate:.1%}).",
        "",
        "## Market Read",
        "",
        "- Goal environment was correct: BTTS, over 2.5, Chelsea 2+ team goals, Tottenham 1+ team goal, and the 2-1 score cluster all landed.",
        "- Shots/SOT were the strongest player-event family: Palmer SOT, Richarlison SOT, Enzo SOT, Richarlison 2+ shots, Gallagher 1+ shot, and the Maddison live/super-sub shot all landed.",
        "- Contact volume was correct: Chelsea 11 fouls, Tottenham 18 fouls, 7 total yellows, and key fouled/foul legs landed late.",
        "- Card targeting was noisy: the referee/card environment was right, but Fofana/Caicedo/Palhinha avoided cards while Hato, Cucurella, Delap, Dario Essugo, Porro, Van de Ven, and Udogie took them.",
        "",
        "## Official API Team Stats",
        "",
        md_table(team_stats),
        "",
        "## Booked Players",
        "",
        md_table(cards) if not cards.empty else "No booked players in normalized API rows.",
        "",
        "## Top Player Event Actuals",
        "",
        md_table(top_events.head(20)),
        "",
        "## Files",
        "",
        "- `scored_core_markets.csv`",
        "- `scored_light_profile_player_event_board.csv`",
        "- `scored_upgraded_interaction_player_event_board.csv`",
        "- `scored_confirmed_lineup_comparison_tips.csv`",
        "",
        "## Notes",
        "",
        "- Player props remain beta/research and are not production deploy rows.",
        "- Super-sub slip settlement is book-specific; this audit scores official player rows and separately preserves live-read interpretation.",
    ]
    (outdir / "CHELSEA_SPURS_POST_MATCH_PREDICTION_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--api-dir", type=Path, default=DEFAULT_API_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    players = load_player_stats(args.api_dir)
    team_stats = load_team_stats(args.api_dir)
    core = score_core_markets(args.preview_dir, team_stats)
    light = score_light_profile(args.preview_dir, players)
    upgraded = score_upgraded_interactions(args.preview_dir, players)
    tips = score_comparison_tips(args.preview_dir, players)

    core.to_csv(args.outdir / "scored_core_markets.csv", index=False)
    light.to_csv(args.outdir / "scored_light_profile_player_event_board.csv", index=False)
    upgraded.to_csv(args.outdir / "scored_upgraded_interaction_player_event_board.csv", index=False)
    tips.to_csv(args.outdir / "scored_confirmed_lineup_comparison_tips.csv", index=False)
    write_summary(args.outdir, core, light, upgraded, tips, players, team_stats)

    print(f"[ok] wrote {args.outdir}")
    print(f"[ok] core={int(core['hit'].sum())}/{int(core['hit'].notna().sum())}")
    print(f"[ok] light={int(light['hit'].eq(True).sum())}/{int(light['hit'].notna().sum())}")
    print(f"[ok] upgraded={int(upgraded['hit'].eq(True).sum())}/{int(upgraded['hit'].notna().sum())}")
    print(f"[ok] tips={int(tips['hit'].eq(True).sum())}/{int(tips['hit'].notna().sum())}")


if __name__ == "__main__":
    main()
