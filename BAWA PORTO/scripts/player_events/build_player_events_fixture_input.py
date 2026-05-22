from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


SCHEMA_COLUMNS = list(pd.read_csv("docs/PLAYER_EVENTS_INPUT_SCHEMA.csv").columns)
MANUAL_STRING_DEFAULTS = {
    "manual_pitch_side": "UNSET",
    "manual_flank_role": "UNSET",
    "manual_overload_target_side": "UNSET",
}
DIRECT_LEAGUE_HINTS = {
    "england_championship",
    "championship",
    "england_efl_league_1",
    "league_one",
    "scotland_premiership",
    "belgium_pro",
    "norway_eliteserien",
    "brazil_serie_a",
    "major_league_soccer",
    "usa_mls",
    "japan_j1",
    "portugal_liga",
    "france_ligue_1",
    "netherlands_eredivisie",
    "europa_conference",
}


def _safe_div(num: float, den: float) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _safe_int(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
    except TypeError:
        pass
    try:
        return int(float(value or 0.0))
    except (TypeError, ValueError):
        return 0


def _manual_clean(value: object, default: str = "UNSET") -> str:
    text = str(value or "").strip().upper()
    if text in {"", "NAN", "NONE", "NULL"}:
        return default
    return text


def _safe_slug(value: object) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _sum(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) if sample else 0.0


def _mean(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    return (_sum(records, key, n) / len(sample)) if sample else 0.0


def _position_group(position: str) -> str:
    pos = str(position or "").strip().upper()
    return {
        "G": "Goalkeeper",
        "D": "Defender",
        "M": "Midfielder",
        "F": "Forward",
    }.get(pos, "Unknown")


def _tactical_role(position: str) -> str:
    pos = str(position or "").strip().upper()
    return {
        "G": "Goalkeeper",
        "D": "Defensive line enforcer",
        "M": "Central disruptor",
        "F": "Front-foot presser",
    }.get(pos, "General role")


def _infer_role_and_flank(
    position: str,
    tackles_per90: float,
    interceptions_per90: float,
    dribbles_faced_per90: float,
    fouls_won_per90: float,
    ground_duel_loss_rate: float,
    blocks_per90: float = 0.0,
    duels_total_per90: float = 0.0,
    aerial_duel_loss_rate: float = 0.0,
) -> tuple[str, str, float, float, float]:
    pos = str(position or "").strip().upper()
    left_flank = 0.0
    right_flank = 0.0
    central = 0.35

    if pos == "D":
        centre_back_profile = (
            blocks_per90 >= 0.7
            or duels_total_per90 >= 6.5
            or interceptions_per90 >= 1.4
            or aerial_duel_loss_rate >= 0.38
        )
        if centre_back_profile and dribbles_faced_per90 <= 1.3:
            role = "Centre-back enforcer"
            central = 0.72
        elif dribbles_faced_per90 >= 0.75 or fouls_won_per90 >= 1.2:
            role = "Wide defender / wing-back"
            left_flank = 0.65
            right_flank = 0.65
            central = 0.20
        else:
            role = "Centre-back enforcer"
            central = 0.65
    elif pos == "M":
        if tackles_per90 >= 1.9 or interceptions_per90 >= 1.4:
            role = "Holding midfielder"
            central = 0.80
        elif fouls_won_per90 >= 1.6:
            role = "Wide midfielder / winger"
            left_flank = 0.70
            right_flank = 0.70
            central = 0.25
        else:
            role = "Central midfielder"
            central = 0.70
    elif pos == "F":
        if fouls_won_per90 >= 1.5 or ground_duel_loss_rate >= 0.55:
            role = "Wide forward"
            left_flank = 0.75
            right_flank = 0.75
            central = 0.20
        else:
            role = "Central striker"
            central = 0.60
    else:
        role = _tactical_role(position)
    return role, ("WIDE" if max(left_flank, right_flank) > central else "CENTRAL"), left_flank, right_flank, central


def _central_battle_flag(position: str) -> int:
    return int(str(position or "").strip().upper() == "M")


def _counterattack_defender_flag(position: str) -> int:
    return int(str(position or "").strip().upper() == "D")


def _temperament_flag(yellow_cards_per90: float, fouls_per90: float) -> float:
    if yellow_cards_per90 >= 0.30 or fouls_per90 >= 2.6:
        return 1.0
    if yellow_cards_per90 >= 0.20 or fouls_per90 >= 1.8:
        return 0.6
    return 0.2


def _expected_minutes(minutes_l5: float, is_starting_xi: int, position: str) -> float:
    if not is_starting_xi:
        return 25.0
    floor = 60.0 if str(position).upper() == "F" else 70.0
    return round(min(90.0, max(floor, minutes_l5 or 0.0)), 1)


def _match_stakes_score(match_date: str) -> float:
    ts = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(ts):
        return 2.0
    if ts.month >= 4:
        return 3.5
    if ts.month >= 2:
        return 2.8
    return 2.0


def _analyst_note(position: str, fouls_per90: float, yellows_per90: float, ref_cards_per_match: float) -> str:
    role = _position_group(position).lower()
    if role == "defender":
        shape = "defensive workload and recovery fouls"
    elif role == "midfielder":
        shape = "central duel load and tactical fouling risk"
    elif role == "forward":
        shape = "pressing/frustration risk"
    else:
        shape = "discipline profile still light"
    return (
        f"{role} profile with {fouls_per90:.2f} fouls/90, {yellows_per90:.2f} yellows/90, "
        f"referee baseline {ref_cards_per_match:.2f} cards/match; watch {shape}."
    )


def _formation_parts(formation: str) -> list[int]:
    text = str(formation or "").strip()
    if not text:
        return []
    parts: list[int] = []
    for chunk in text.split("-"):
        try:
            parts.append(int(chunk))
        except ValueError:
            return []
    return parts


def _formation_shape_scores(formation: str) -> tuple[float, float, float]:
    parts = _formation_parts(formation)
    if not parts:
        return 0.0, 0.0, 0.0
    back_line = float(parts[0])
    front_line = float(parts[-1]) if len(parts) >= 2 else 0.0
    if len(parts) >= 4:
        primary_midfield = float(parts[1])
        advanced_midfield = float(parts[-2])
        midfield = primary_midfield + 0.55 * advanced_midfield
    elif len(parts) == 3:
        midfield = float(parts[1])
    elif len(parts) == 2:
        midfield = float(parts[1])
    else:
        midfield = 0.0
    wing_aggression = 0.0
    if front_line >= 3:
        wing_aggression += 0.55
    if back_line == 3:
        wing_aggression += 0.25
    if len(parts) >= 4:
        wing_aggression += 0.10
    midfield_density = min(1.0, midfield / 4.5)
    defensive_exposure = 0.0
    if back_line <= 3:
        defensive_exposure = 0.75
    elif back_line == 4:
        defensive_exposure = 0.45
    else:
        defensive_exposure = 0.20
    return min(1.0, wing_aggression), midfield_density, defensive_exposure


def _opponent_striker_profile(
    league: str,
    opponent_form: str,
    fixture_attacking_style_label: str,
    fixture_corner_pressure_score: float,
    opponent_possession: float,
    og_xg_total: float,
) -> tuple[str, str, str, str]:
    league_slug = _safe_slug(league)
    parts = _formation_parts(opponent_form)
    front_line = parts[-1] if parts else 0
    attack_style = str(fixture_attacking_style_label or "").upper()
    direct_league = league_slug in DIRECT_LEAGUE_HINTS

    if direct_league and (attack_style in {"CORNER_SIEGE", "TERRITORY_TILT"} or fixture_corner_pressure_score >= 0.66):
        return (
            "AERIAL_BOX_NINE",
            "AERIAL_PIN_PRESSURE",
            "Opponent profile leans direct and aerial, so the centre-back should expect repeated first-contact box duels.",
            "Aerial-9 subtype: expect crosses, long entries, first-contact contests, and second-ball chaos around the box.",
        )
    if direct_league and front_line <= 2:
        return (
            "DIRECT_TARGET_STRIKER",
            "DIRECT_PIN_PRESSURE",
            "Opponent shape points to a direct target-forward lane that can pin the centre-back into front-foot contact.",
            "Target-forward subtype: back-to-goal reference point who pins the centre-back and invites body-duel fouls.",
        )
    if front_line >= 3 and opponent_possession <= 50.0 and og_xg_total >= 2.35:
        return (
            "CHANNEL_RUNNER_STRIKER",
            "CHANNEL_STRETCH_PRESSURE",
            "Opponent pressure looks more like channel running and recovery defending than static aerial wrestling.",
            "Channel-runner subtype: threatens the outside shoulder and space behind, forcing recovery turns and stretched tackles.",
        )
    if front_line >= 3 and og_xg_total >= 2.6:
        return (
            "MOBILE_PRESSING_9",
            "MOBILE_PRESSURE_FRONT",
            "Opponent front line looks mobile enough to pull the centre-back into space rather than just body duels.",
            "Mobile-9 subtype: presses actively, drifts into channels, and forces the centre-back into front-foot recovery steps.",
        )
    return (
        "UNSET",
        "UNSET",
        "No clear opponent striker-profile signal yet; keep the centre-back lane in research/watch mode.",
        "Subtype note unset until the opponent striker profile clears a stronger shape/pressure threshold.",
    )


def _build_player_pre_match_features(player_stats: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    merged = player_stats.merge(
        fixtures[
            [
                "fixture_id",
                "fixture_key",
                "league",
                "season",
                "match_date",
                "kickoff_ts_utc",
                "home_team_id",
                "away_team_id",
                "home_team_name",
                "away_team_name",
            ]
        ],
        on="fixture_id",
        how="left",
    )
    merged["kickoff_ts_utc"] = pd.to_datetime(merged["kickoff_ts_utc"], errors="coerce", utc=True)
    merged = merged.sort_values(["kickoff_ts_utc", "fixture_id", "team_id", "player_id"]).reset_index(drop=True)

    history: dict[int, list[dict]] = defaultdict(list)
    out_rows: list[dict] = []
    for _, row in merged.iterrows():
        player_id = int(row["player_id"])
        prev = list(reversed(history.get(player_id, [])))
        mins5 = _sum(prev, "minutes", 5)
        mins10 = _sum(prev, "minutes", 10)
        fouls10 = _sum(prev, "fouls_committed", 10)
        yellows10 = _sum(prev, "yellow_cards", 10)
        tackles5 = _sum(prev, "tackles", 5)
        interceptions5 = _sum(prev, "interceptions", 5)
        blocks5 = _sum(prev, "blocks", 5)
        dribbled_past5 = _sum(prev, "dribbled_past", 5)
        fouls_drawn5 = _sum(prev, "fouls_drawn", 5)
        shots5 = _sum(prev, "shots_total", 5)
        shots_on_target5 = _sum(prev, "shots_on_target", 5)
        goals5 = _sum(prev, "goals", 5)
        assists5 = _sum(prev, "assists", 5)
        key_passes5 = _sum(prev, "passes_key", 5)
        rating5 = _mean(prev, "rating", 5)
        passes_total5 = _sum(prev, "passes_total", 5)
        passes_accurate5 = _sum(prev, "passes_accurate", 5)
        duels_total5 = _sum(prev, "duels_total", 5)
        duels_won5 = _sum(prev, "duels_won", 5)
        duel_loss_rate = (1.0 - _safe_div(duels_won5, duels_total5)) if duels_total5 else 0.0
        dribble_pressure_share = _safe_div(dribbled_past5, duels_total5) if duels_total5 else 0.0
        block_relief_share = min(1.0, _safe_div(blocks5, max(duels_total5, 1.0)))
        aerial_duel_loss_proxy = 0.0
        position = str(row.get("position", "") or "").strip().upper()
        if duels_total5:
            aerial_duel_loss_proxy = duel_loss_rate
            if position in {"D", "F"}:
                aerial_duel_loss_proxy = min(
                    1.0,
                    max(
                        0.0,
                        0.72 * duel_loss_rate
                        + 0.18 * dribble_pressure_share
                        + 0.10 * (1.0 - block_relief_share),
                    ),
                )
        last_dates = [pd.to_datetime(r["kickoff_ts_utc"], utc=True) for r in prev[:1] if r.get("kickoff_ts_utc")]
        current_ts = row.get("kickoff_ts_utc")
        days_rest = 7.0
        if last_dates and pd.notna(current_ts):
            days_rest = max(0.0, (pd.to_datetime(current_ts, utc=True) - last_dates[0]).total_seconds() / 86400.0)

        out_rows.append(
            {
                "fixture_id": int(row["fixture_id"]),
                "player_id": player_id,
                "player_name": row.get("player_name", ""),
                "team_id": int(row["team_id"]),
                "position": row.get("position", ""),
                "fouls_per90": round(_safe_div(fouls10 * 90.0, mins10), 4),
                "yellow_cards_per90": round(_safe_div(yellows10 * 90.0, mins10), 4),
                "booking_efficiency": round(_safe_div(fouls10, max(yellows10, 1.0)), 4) if fouls10 else 6.0,
                "tackles_per90": round(_safe_div(tackles5 * 90.0, mins5), 4),
                "interceptions_per90": round(_safe_div(interceptions5 * 90.0, mins5), 4),
                "blocks_per90": round(_safe_div(blocks5 * 90.0, mins5), 4),
                "duels_total_per90": round(_safe_div(duels_total5 * 90.0, mins5), 4),
                "duels_won_per90": round(_safe_div(duels_won5 * 90.0, mins5), 4),
                "ground_duel_loss_rate": round(duel_loss_rate, 4) if duels_total5 else 0.0,
                "aerial_duel_loss_rate": round(aerial_duel_loss_proxy, 4),
                "dribbles_faced_per90": round(_safe_div(dribbled_past5 * 90.0, mins5), 4),
                "fouls_won_per90": round(_safe_div(fouls_drawn5 * 90.0, mins5), 4),
                "shots_per90": round(_safe_div(shots5 * 90.0, mins5), 4),
                "shots_on_target_per90": round(_safe_div(shots_on_target5 * 90.0, mins5), 4),
                "goals_per90": round(_safe_div(goals5 * 90.0, mins5), 4),
                "assists_per90": round(_safe_div(assists5 * 90.0, mins5), 4),
                "key_passes_per90": round(_safe_div(key_passes5 * 90.0, mins5), 4),
                "player_form_rating_l5": round(rating5, 4),
                "pass_accuracy_pct_l5": round(_safe_div(passes_accurate5 * 100.0, passes_total5), 4),
                "minutes_last_3_matches": round(_sum(prev, "minutes", 3), 1),
                "days_rest": round(days_rest, 2),
                "recent_injury_return_flag": 0,
                "player_minutes_l5": round(_mean(prev, "minutes", 5), 2),
            }
        )

        history[player_id].append(row.to_dict())

    return pd.DataFrame(out_rows)


def _build_team_pre_match_features(team_stats: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    merged = team_stats.merge(
        fixtures[["fixture_id", "fixture_key", "league", "season", "match_date", "kickoff_ts_utc"]],
        on="fixture_id",
        how="left",
    )
    merged["kickoff_ts_utc"] = pd.to_datetime(merged["kickoff_ts_utc"], errors="coerce", utc=True)
    merged = merged.sort_values(["kickoff_ts_utc", "fixture_id", "team_id"]).reset_index(drop=True)

    history: dict[int, list[dict]] = defaultdict(list)
    out_rows: list[dict] = []
    for _, row in merged.iterrows():
        team_id = int(row["team_id"])
        prev = list(reversed(history.get(team_id, [])))
        out_rows.append(
            {
                "fixture_id": int(row["fixture_id"]),
                "team_id": team_id,
                "team_name": row.get("team_name", ""),
                "team_avg_fouls": round(_mean(prev, "fouls_for", 5), 4),
                "team_avg_yellows": round(_mean(prev, "yellow_cards", 5), 4),
                "team_avg_possession": round(_mean(prev, "possession_pct", 5), 4),
            }
        )
        history[team_id].append(row.to_dict())
    return pd.DataFrame(out_rows)


def build_player_events_fixture_input(
    league_tag: str,
    season: int,
    fixtures_csv: str,
    player_stats_csv: str,
    team_stats_csv: str,
    lineups_csv: str,
    injuries_csv: str,
    referee_csv: str,
    og_overlay_csv: str,
    style_overlay_csv: str,
    quality_overlay_csv: str,
    lineup_features_csv: str,
    manual_side_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    player_stats = pd.read_csv(player_stats_csv)
    team_stats = pd.read_csv(team_stats_csv)
    lineups = pd.read_csv(lineups_csv)
    injuries = pd.read_csv(injuries_csv)
    referee = pd.read_csv(referee_csv)
    og_overlay = pd.read_csv(og_overlay_csv) if og_overlay_csv and Path(og_overlay_csv).exists() else pd.DataFrame()
    style_overlay = pd.read_csv(style_overlay_csv) if style_overlay_csv and Path(style_overlay_csv).exists() else pd.DataFrame()
    quality_overlay = pd.read_csv(quality_overlay_csv) if quality_overlay_csv and Path(quality_overlay_csv).exists() else pd.DataFrame()
    lineup_features = pd.read_csv(lineup_features_csv) if lineup_features_csv and Path(lineup_features_csv).exists() else pd.DataFrame()
    manual_side = pd.read_csv(manual_side_csv) if manual_side_csv and Path(manual_side_csv).exists() else pd.DataFrame()
    manual_lookup_by_ids: dict[tuple[int, int, int], dict] = {}
    manual_lookup_by_names: dict[tuple[str, str, str], dict] = {}

    fixtures["kickoff_ts_utc"] = pd.to_datetime(fixtures["kickoff_ts_utc"], errors="coerce", utc=True)
    fixtures = fixtures.sort_values(["kickoff_ts_utc", "fixture_id"]).reset_index(drop=True)

    player_roll = _build_player_pre_match_features(player_stats, fixtures)
    team_roll = _build_team_pre_match_features(team_stats, fixtures)

    lineup_base = lineups.copy()
    lineup_base["is_starting_xi"] = pd.to_numeric(lineup_base["is_starting_xi"], errors="coerce").fillna(0).astype(int)
    lineup_base = lineup_base[lineup_base["is_starting_xi"].eq(1)].copy()
    if lineup_base.empty:
        lineup_base = player_stats[player_stats["started_flag"].eq(1)][["fixture_id", "team_id", "player_id", "player_name", "position"]].copy()
        lineup_base["is_starting_xi"] = 1

    injury_map = (
        injuries.sort_values(["fixture_id", "team_id", "player_id"])
        .drop_duplicates(subset=["fixture_id", "team_id", "player_id"], keep="last")
        [["fixture_id", "team_id", "player_id", "status"]]
        .rename(columns={"status": "injury_status"})
    )

    player_actual_names = (
        player_stats.sort_values(["fixture_id", "team_id", "player_id"])
        .drop_duplicates(subset=["fixture_id", "team_id", "player_id"], keep="last")
        [["fixture_id", "team_id", "player_id", "player_name", "position", "started_flag"]]
    )
    formation_map = (
        lineups.sort_values(["fixture_id", "team_id", "lineup_published_ts_utc"])
        .dropna(subset=["formation"])
        .drop_duplicates(subset=["fixture_id", "team_id"], keep="last")[["fixture_id", "team_id", "formation"]]
        .rename(columns={"formation": "team_formation"})
    )

    base = (
        lineup_base.merge(player_actual_names, on=["fixture_id", "team_id", "player_id"], how="left", suffixes=("_lineup", ""))
        .merge(player_roll, on=["fixture_id", "team_id", "player_id"], how="left")
        .merge(team_roll, on=["fixture_id", "team_id"], how="left")
        .merge(referee, on=["fixture_id"], how="left", suffixes=("", "_ref"))
        .merge(injury_map, on=["fixture_id", "team_id", "player_id"], how="left")
        .merge(formation_map, on=["fixture_id", "team_id"], how="left")
        .merge(fixtures, on="fixture_id", how="left", suffixes=("", "_fx"))
    )
    if not og_overlay.empty:
        og_cols = [
            c
            for c in og_overlay.columns
            if c
            not in {
                "fixture_id",
                "team_id",
                "player_id",
                "fixture_key",
                "league",
                "season",
                "match_date",
                "home_team_name",
                "away_team_name",
            }
        ]
        base = base.merge(og_overlay[["fixture_id"] + og_cols].drop_duplicates(subset=["fixture_id"]), on="fixture_id", how="left")
    if not style_overlay.empty:
        style_cols = [
            c
            for c in style_overlay.columns
            if c
            not in {
                "fixture_id",
                "team_id",
                "player_id",
                "fixture_key",
                "league",
                "season",
                "match_date",
                "home_team_name",
                "away_team_name",
            }
        ]
        base = base.merge(style_overlay[["fixture_id"] + style_cols].drop_duplicates(subset=["fixture_id"]), on="fixture_id", how="left")
    if not quality_overlay.empty:
        quality_cols = [
            c
            for c in quality_overlay.columns
            if c
            not in {
                "fixture_id",
                "team_id",
                "player_id",
                "fixture_key",
                "league",
                "season",
                "match_date",
                "player_name",
                "position",
                "position_group",
                "player_team_side",
            }
        ]
        base = base.merge(
            quality_overlay[["fixture_id", "team_id", "player_id"] + quality_cols].drop_duplicates(
                subset=["fixture_id", "team_id", "player_id"]
            ),
            on=["fixture_id", "team_id", "player_id"],
            how="left",
        )
    if not lineup_features.empty:
        lineup_cols = [
            c
            for c in lineup_features.columns
            if c
            not in {
                "fixture_id",
                "fixture_key",
                "league",
                "league_id",
                "season",
                "match_date",
                "home_team_id",
                "away_team_id",
                "home_team_name",
                "away_team_name",
            }
        ]
        base = base.merge(
            lineup_features[["fixture_id"] + lineup_cols].drop_duplicates(subset=["fixture_id"]),
            on="fixture_id",
            how="left",
        )
    if not manual_side.empty:
        manual = manual_side.copy()
        for col, default in MANUAL_STRING_DEFAULTS.items():
            if col in manual.columns:
                manual[col] = manual[col].map(lambda v: _manual_clean(v, default=default))
        keep_cols = [c for c in ["manual_pitch_side", "manual_flank_role", "manual_overload_target_side", "manual_side_notes"] if c in manual.columns]
        for key in ["fixture_id", "team_id", "player_id"]:
            if key in manual.columns:
                manual[key] = pd.to_numeric(manual[key], errors="coerce")
        if {"fixture_id", "team_id", "player_id"}.issubset(manual.columns):
            tmp = manual.dropna(subset=["fixture_id", "team_id", "player_id"]).copy()
            tmp = tmp.sort_values(["fixture_id", "team_id", "player_id"]).drop_duplicates(
                subset=["fixture_id", "team_id", "player_id"],
                keep="last",
            )
            manual_lookup_by_ids = {
                (int(r["fixture_id"]), int(r["team_id"]), int(r["player_id"])): {c: r.get(c, "") for c in keep_cols}
                for _, r in tmp.iterrows()
            }
        if {"fixture_key", "team_name", "player_name"}.issubset(manual.columns):
            tmp = manual.sort_values(["fixture_key", "team_name", "player_name"]).drop_duplicates(
                subset=["fixture_key", "team_name", "player_name"],
                keep="last",
            )
            manual_lookup_by_names = {
                (str(r["fixture_key"]), str(r["team_name"]), str(r["player_name"])): {c: r.get(c, "") for c in keep_cols}
                for _, r in tmp.iterrows()
            }

    player_name_candidates = [c for c in ["player_name", "player_name_x", "player_name_y", "player_name_lineup"] if c in base.columns]
    position_candidates = [c for c in ["position", "position_x", "position_y", "position_lineup"] if c in base.columns]
    if player_name_candidates:
        base["player_name_final"] = base[player_name_candidates].bfill(axis=1).iloc[:, 0]
    else:
        base["player_name_final"] = ""
    if position_candidates:
        base["position_final"] = base[position_candidates].bfill(axis=1).iloc[:, 0]
    else:
        base["position_final"] = ""

    records: list[dict] = []
    for _, row in base.iterrows():
        home_team_id = int(row["home_team_id"])
        away_team_id = int(row["away_team_id"])
        team_id = int(row["team_id"])
        is_home = team_id == home_team_id
        team_name = row["home_team_name"] if is_home else row["away_team_name"]
        opponent_team_id = away_team_id if is_home else home_team_id
        opponent_roll = team_roll[(team_roll["fixture_id"].eq(int(row["fixture_id"]))) & (team_roll["team_id"].eq(opponent_team_id))]
        opponent_possession = float(opponent_roll["team_avg_possession"].iloc[0]) if not opponent_roll.empty else 50.0
        fixture_formation_rows = formation_map[formation_map["fixture_id"].eq(int(row["fixture_id"]))]
        opponent_form = ""
        if not fixture_formation_rows.empty:
            opp_form_row = fixture_formation_rows[fixture_formation_rows["team_id"].eq(opponent_team_id)]
            if not opp_form_row.empty:
                opponent_form = str(opp_form_row["team_formation"].iloc[0] or "")

        position = str(row.get("position_final", "") or "")
        team_formation = str(row.get("team_formation", "") or "")
        fouls_per90 = float(row.get("fouls_per90", 0.0) or 0.0)
        yellow_cards_per90 = float(row.get("yellow_cards_per90", 0.0) or 0.0)
        ref_cards_per_match = float(row.get("ref_cards_per_match", 0.0) or 0.0)
        player_minutes_l5 = float(row.get("player_minutes_l5", 0.0) or 0.0)
        expected_start_flag = int(row.get("is_starting_xi", 0) or 0)
        tackles_per90 = float(row.get("tackles_per90", 0.0) or 0.0)
        interceptions_per90 = float(row.get("interceptions_per90", 0.0) or 0.0)
        blocks_per90 = float(row.get("blocks_per90", 0.0) or 0.0)
        duels_total_per90 = float(row.get("duels_total_per90", 0.0) or 0.0)
        duels_won_per90 = float(row.get("duels_won_per90", 0.0) or 0.0)
        ground_duel_loss_rate = float(row.get("ground_duel_loss_rate", 0.0) or 0.0)
        aerial_duel_loss_rate = float(row.get("aerial_duel_loss_rate", 0.0) or 0.0)
        fouls_won_per90 = float(row.get("fouls_won_per90", 0.0) or 0.0)
        dribbles_faced_per90 = float(row.get("dribbles_faced_per90", 0.0) or 0.0)
        tactical_role, flank_zone, left_flank_dom, right_flank_dom, central_attack_dom = _infer_role_and_flank(
            position=position,
            tackles_per90=tackles_per90,
            interceptions_per90=interceptions_per90,
            dribbles_faced_per90=dribbles_faced_per90,
            fouls_won_per90=fouls_won_per90,
            ground_duel_loss_rate=ground_duel_loss_rate,
            blocks_per90=blocks_per90,
            duels_total_per90=duels_total_per90,
            aerial_duel_loss_rate=aerial_duel_loss_rate,
        )
        manual_values = manual_lookup_by_ids.get(
            (int(row["fixture_id"]), int(row["team_id"]), int(row["player_id"])),
            {},
        )
        if not manual_values:
            manual_values = manual_lookup_by_names.get(
                (str(row.get("fixture_key", "")), str(team_name), str(row.get("player_name_final", ""))),
                {},
            )
        manual_pitch_side = _manual_clean(manual_values.get("manual_pitch_side", ""), default="UNSET")
        manual_flank_role = _manual_clean(manual_values.get("manual_flank_role", ""), default="UNSET")
        manual_overload_target_side = _manual_clean(manual_values.get("manual_overload_target_side", ""), default="UNSET")
        manual_side_override_flag = int(manual_pitch_side in {"LEFT", "RIGHT", "CENTRAL"})
        manual_left_side_flag = int(manual_pitch_side == "LEFT")
        manual_right_side_flag = int(manual_pitch_side == "RIGHT")
        central_battle_flag = int("Holding" in tactical_role or "Central" in tactical_role or "Centre-back" in tactical_role)
        counterattack_defender_flag = int("defender" in tactical_role.lower() or "Centre-back" in tactical_role)
        home_power = float(row.get("og_home_power_rating", 0.0) or 0.0)
        away_power = float(row.get("og_away_power_rating", 0.0) or 0.0)
        lineup_same_formation_flag = int(row.get("same_formation_flag", 0) or 0)
        lineup_formation_mismatch_flag = int(row.get("formation_mismatch_flag", 0) or 0)
        lineup_attack_delta_raw = float(row.get("formation_attack_delta", 0.0) or 0.0)
        lineup_defence_delta_raw = float(row.get("formation_defence_delta", 0.0) or 0.0)
        lineup_xi_shot_power_delta_raw = float(row.get("xi_shot_power_delta", 0.0) or 0.0)
        lineup_xi_tackle_pressure_delta_raw = float(row.get("xi_tackle_pressure_delta", 0.0) or 0.0)
        lineup_xi_card_risk_delta_raw = float(row.get("xi_card_risk_delta", 0.0) or 0.0)
        lineup_attack_delta = lineup_attack_delta_raw if is_home else -lineup_attack_delta_raw
        lineup_defence_delta = lineup_defence_delta_raw if is_home else -lineup_defence_delta_raw
        lineup_xi_shot_power_delta = lineup_xi_shot_power_delta_raw if is_home else -lineup_xi_shot_power_delta_raw
        lineup_xi_tackle_pressure_delta = lineup_xi_tackle_pressure_delta_raw if is_home else -lineup_xi_tackle_pressure_delta_raw
        lineup_xi_card_risk_delta = lineup_xi_card_risk_delta_raw if is_home else -lineup_xi_card_risk_delta_raw
        team_power_rating = home_power if is_home else away_power
        opponent_power_rating = away_power if is_home else home_power
        team_power_edge = team_power_rating - opponent_power_rating
        weaker_side_under_pressure_flag = int(
            (team_power_edge <= -4.0)
            and (
                opponent_possession >= 55.0
                or max(left_flank_dom, right_flank_dom, central_attack_dom) >= 0.60
            )
        )
        base_power_pressure = max(0.0, min(1.0, (-team_power_edge) / 20.0))
        power_gap_directional_pressure_score = (
            0.45 * base_power_pressure
            + 0.25 * min(1.0, opponent_possession / 70.0)
            + 0.10 * left_flank_dom
            + 0.10 * right_flank_dom
            + 0.10 * central_attack_dom
        )
        if not weaker_side_under_pressure_flag:
            power_gap_directional_pressure_score *= 0.45
        weak_flank_overload_flag = int(
            weaker_side_under_pressure_flag
            and str(flank_zone).upper() == "WIDE"
            and max(left_flank_dom, right_flank_dom) >= 0.60
        )
        weak_midfield_overload_flag = int(
            weaker_side_under_pressure_flag
            and central_battle_flag
            and central_attack_dom >= 0.58
        )
        weak_territory_protection_flag = int(
            weaker_side_under_pressure_flag
            and opponent_possession >= 58.0
        )
        team_wing_aggr, team_mid_density, team_def_exposure = _formation_shape_scores(team_formation)
        opp_wing_aggr, opp_mid_density, opp_def_exposure = _formation_shape_scores(opponent_form)
        formation_wide_overload_flag = int(
            (
                (opp_wing_aggr >= 0.55 and team_def_exposure >= 0.45)
                or (opp_wing_aggr >= 0.65 and team_wing_aggr <= 0.40)
            )
        )
        formation_left_wide_overload_score = (
            0.55 * left_flank_dom
            + 0.30 * opp_wing_aggr
            + 0.15 * team_def_exposure
        ) if formation_wide_overload_flag else 0.0
        formation_right_wide_overload_score = (
            0.55 * right_flank_dom
            + 0.30 * opp_wing_aggr
            + 0.15 * team_def_exposure
        ) if formation_wide_overload_flag else 0.0
        formation_left_wide_overload_flag = int(
            formation_wide_overload_flag
            and formation_left_wide_overload_score >= 0.58
            and formation_left_wide_overload_score > formation_right_wide_overload_score + 0.03
        )
        formation_right_wide_overload_flag = int(
            formation_wide_overload_flag
            and formation_right_wide_overload_score >= 0.58
            and formation_right_wide_overload_score > formation_left_wide_overload_score + 0.03
        )
        if formation_left_wide_overload_flag and not formation_right_wide_overload_flag:
            wide_overload_target_side = "LEFT"
        elif formation_right_wide_overload_flag and not formation_left_wide_overload_flag:
            wide_overload_target_side = "RIGHT"
        elif formation_wide_overload_flag:
            wide_overload_target_side = "BOTH"
        else:
            wide_overload_target_side = "NONE"
        if manual_overload_target_side in {"LEFT", "RIGHT", "BOTH", "NONE"}:
            wide_overload_target_side = manual_overload_target_side
        midfield_density_floor = min(team_mid_density, opp_mid_density)
        midfield_density_gap = abs(team_mid_density - opp_mid_density)
        formation_midfield_grind_flag = int(
            (
                midfield_density_floor >= 0.82
                and midfield_density_gap <= 0.18
                and abs(team_power_edge) <= 8.0
            )
            or (
                central_battle_flag
                and midfield_density_floor >= 0.72
                and midfield_density_gap <= 0.12
                and weak_midfield_overload_flag
            )
        )
        underdog_fullback_wide_overload_flag = int(
            weaker_side_under_pressure_flag
            and "wide defender" in tactical_role.lower()
            and formation_wide_overload_flag
        )
        if underdog_fullback_wide_overload_flag and manual_side_override_flag:
            if manual_left_side_flag and wide_overload_target_side not in {"LEFT", "BOTH"}:
                underdog_fullback_wide_overload_flag = 0
            if manual_right_side_flag and wide_overload_target_side not in {"RIGHT", "BOTH"}:
                underdog_fullback_wide_overload_flag = 0
        weak_left_flank_overload_flag = int(
            weak_flank_overload_flag
            and wide_overload_target_side in {"LEFT", "BOTH"}
        )
        weak_right_flank_overload_flag = int(
            weak_flank_overload_flag
            and wide_overload_target_side in {"RIGHT", "BOTH"}
        )
        underdog_dm_midfield_grind_flag = int(
            weaker_side_under_pressure_flag
            and tactical_role.lower() in {"holding midfielder", "central midfielder"}
            and formation_midfield_grind_flag
        )
        formation_pressure_score = (
            0.35 * formation_wide_overload_flag
            + 0.25 * formation_midfield_grind_flag
            + 0.20 * underdog_fullback_wide_overload_flag
            + 0.20 * underdog_dm_midfield_grind_flag
        )
        formation_pressure_score += 0.10 * max(0.0, min(1.0, lineup_formation_mismatch_flag))
        formation_pressure_score += 0.12 * max(0.0, min(1.0, (-lineup_xi_tackle_pressure_delta) / 4.0))
        formation_pressure_score += 0.10 * max(0.0, min(1.0, (-lineup_xi_card_risk_delta) / 1.5))
        formation_pressure_score += 0.08 * max(0.0, min(1.0, (-lineup_defence_delta) / 3.0))
        formation_pressure_score = min(1.0, formation_pressure_score)
        formation_matchup_label = (
            f"{team_formation} vs {opponent_form}".strip()
            if team_formation or opponent_form
            else "UNSET"
        )
        opponent_striker_profile, opponent_striker_pressure_tag, opponent_striker_context_note, opponent_striker_subtype_note = _opponent_striker_profile(
            league=str(row.get("league", league_tag.replace("_", " "))),
            opponent_form=opponent_form,
            fixture_attacking_style_label=str(row.get("fixture_attacking_style_label", "") or ""),
            fixture_corner_pressure_score=float(row.get("fixture_corner_pressure_score", 0.0) or 0.0),
            opponent_possession=opponent_possession,
            og_xg_total=float(row.get("og_xg_total", 0.0) or 0.0),
        )
        cb_duel_pressure_score = 0.0
        cb_front_foot_duel_flag = 0
        if tactical_role == "Centre-back enforcer":
            cb_duel_pressure_score = min(
                1.0,
                0.22 * min(1.0, blocks_per90 / 1.6)
                + 0.20 * min(1.0, duels_total_per90 / 9.0)
                + 0.16 * min(1.0, duels_won_per90 / 6.0)
                + 0.16 * min(1.0, dribbles_faced_per90 / 1.8)
                + 0.12 * min(1.0, ground_duel_loss_rate)
                + 0.08 * min(1.0, aerial_duel_loss_rate)
                + 0.06 * min(1.0, opponent_possession / 65.0)
            )
            if opponent_striker_profile in {"DIRECT_TARGET_STRIKER", "AERIAL_BOX_NINE"}:
                cb_duel_pressure_score = min(1.0, cb_duel_pressure_score + 0.08)
            elif opponent_striker_profile in {"CHANNEL_RUNNER_STRIKER", "MOBILE_PRESSING_9"}:
                cb_duel_pressure_score = min(1.0, cb_duel_pressure_score + 0.05)
            cb_front_foot_duel_flag = int(
                cb_duel_pressure_score >= 0.58
                and (
                    blocks_per90 >= 0.7
                    or duels_total_per90 >= 6.5
                    or dribbles_faced_per90 >= 0.65
                )
                and opponent_striker_profile != "UNSET"
            )

        rec = {
            "fixture_key": row.get("fixture_key", ""),
            "match_date": row.get("match_date", ""),
            "competition": row.get("league", league_tag.replace("_", " ")),
            "league": row.get("league", league_tag.replace("_", " ")),
            "home_team_name": row.get("home_team_name", ""),
            "away_team_name": row.get("away_team_name", ""),
            "venue": row.get("venue_name", ""),
            "referee_name": row.get("referee_name", ""),
            "ref_cards_per_match": round(ref_cards_per_match, 4),
            "ref_foul_to_card_ratio": round(float(row.get("ref_foul_to_card_ratio", 0.0) or 0.0), 4),
            "ref_dissent_strictness": round(float(row.get("ref_dissent_strictness", 0.0) or 0.0), 4),
            "ref_timewasting_strictness": round(float(row.get("ref_timewasting_strictness", 0.0) or 0.0), 4),
            "weather_summary": "",
            "pitch_condition": "",
            "market_yellow_cards_available": 0,
            "market_fouls_available": 0,
            "market_team_cards_available": 0,
            "team_name": team_name,
            "player_name": row.get("player_name_final", ""),
            "player_team_side": "HOME" if is_home else "AWAY",
            "expected_start_flag": expected_start_flag,
            "expected_minutes": _expected_minutes(player_minutes_l5, expected_start_flag, position),
            "position_group": _position_group(position),
            "tactical_role": tactical_role,
            "likely_marking_assignment": opponent_striker_profile if tactical_role == "Centre-back enforcer" else "",
            "manual_pitch_side": manual_pitch_side,
            "manual_flank_role": manual_flank_role,
            "manual_overload_target_side": manual_overload_target_side,
            "manual_side_override_flag": manual_side_override_flag,
            "team_formation": team_formation,
            "opponent_formation": opponent_form,
            "formation_matchup_label": formation_matchup_label,
            "flank_zone": flank_zone,
            "central_battle_flag": central_battle_flag,
            "counterattack_defender_flag": counterattack_defender_flag,
            "same_formation_flag": lineup_same_formation_flag,
            "formation_mismatch_flag": lineup_formation_mismatch_flag,
            "lineup_formation_attack_delta": round(lineup_attack_delta, 4),
            "lineup_formation_defence_delta": round(lineup_defence_delta, 4),
            "lineup_xi_shot_power_delta": round(lineup_xi_shot_power_delta, 4),
            "lineup_xi_tackle_pressure_delta": round(lineup_xi_tackle_pressure_delta, 4),
            "lineup_xi_card_risk_delta": round(lineup_xi_card_risk_delta, 4),
            "formation_wide_overload_flag": formation_wide_overload_flag,
            "formation_left_wide_overload_score": round(formation_left_wide_overload_score, 4),
            "formation_right_wide_overload_score": round(formation_right_wide_overload_score, 4),
            "formation_left_wide_overload_flag": formation_left_wide_overload_flag,
            "formation_right_wide_overload_flag": formation_right_wide_overload_flag,
            "wide_overload_target_side": wide_overload_target_side,
            "formation_midfield_grind_flag": formation_midfield_grind_flag,
            "underdog_fullback_wide_overload_flag": underdog_fullback_wide_overload_flag,
            "underdog_dm_midfield_grind_flag": underdog_dm_midfield_grind_flag,
            "formation_pressure_score": round(formation_pressure_score, 4),
            "fouls_per90": round(fouls_per90, 4),
            "yellow_cards_per90": round(yellow_cards_per90, 4),
            "booking_efficiency": round(float(row.get("booking_efficiency", 6.0) or 6.0), 4),
            "tackles_per90": round(tackles_per90, 4),
            "interceptions_per90": round(interceptions_per90, 4),
            "blocks_per90": round(blocks_per90, 4),
            "duels_total_per90": round(duels_total_per90, 4),
            "duels_won_per90": round(duels_won_per90, 4),
            "ground_duel_loss_rate": round(ground_duel_loss_rate, 4),
            "aerial_duel_loss_rate": round(aerial_duel_loss_rate, 4),
            "dribbles_faced_per90": round(dribbles_faced_per90, 4),
            "cb_duel_pressure_score": round(cb_duel_pressure_score, 4),
            "cb_front_foot_duel_flag": cb_front_foot_duel_flag,
            "opponent_striker_profile": opponent_striker_profile,
            "opponent_striker_pressure_tag": opponent_striker_pressure_tag,
            "opponent_striker_context_note": opponent_striker_context_note,
            "opponent_striker_subtype_note": opponent_striker_subtype_note,
            "fouls_won_per90": round(fouls_won_per90, 4),
            "shots_per90": round(float(row.get("shots_per90", 0.0) or 0.0), 4),
            "shots_on_target_per90": round(float(row.get("shots_on_target_per90", 0.0) or 0.0), 4),
            "goals_per90": round(float(row.get("goals_per90", 0.0) or 0.0), 4),
            "assists_per90": round(float(row.get("assists_per90", 0.0) or 0.0), 4),
            "key_passes_per90": round(float(row.get("key_passes_per90", 0.0) or 0.0), 4),
            "player_form_rating_l5": round(float(row.get("player_form_rating_l5", 0.0) or 0.0), 4),
            "player_quality_score_l5": round(float(row.get("player_quality_score_l5", 0.0) or 0.0), 4),
            "player_form_tier": str(row.get("player_form_tier", "") or ""),
            "player_quality_rank_in_position": _safe_int(row.get("player_quality_rank_in_position", 0.0)),
            "player_quality_percentile_in_position": round(float(row.get("player_quality_percentile_in_position", 0.0) or 0.0), 4),
            "pass_accuracy_pct_l5": round(float(row.get("pass_accuracy_pct_l5", 0.0) or 0.0), 4),
            "minutes_last_3_matches": round(float(row.get("minutes_last_3_matches", 0.0) or 0.0), 1),
            "days_rest": round(float(row.get("days_rest", 7.0) or 7.0), 2),
            "recent_injury_return_flag": int(str(row.get("injury_status", "") or "").strip().lower() in {"doubtful", "questionable", "fit"}),
            "temperament_flag": _temperament_flag(fouls_per90=fouls_per90, yellow_cards_per90=yellow_cards_per90),
            "suspension_risk_flag": 0,
            "match_stakes_score": _match_stakes_score(str(row.get("match_date", ""))),
            "rivalry_flag": 0,
            "team_avg_fouls": round(float(row.get("team_avg_fouls", 0.0) or 0.0), 4),
            "team_avg_yellows": round(float(row.get("team_avg_yellows", 0.0) or 0.0), 4),
            "opponent_possession_projection": round(opponent_possession, 4),
            "home_team_fouls_l5": round(float(row.get("home_team_fouls_l5", 0.0) or 0.0), 4),
            "home_team_fouls_l10": round(float(row.get("home_team_fouls_l10", 0.0) or 0.0), 4),
            "away_team_fouls_l5": round(float(row.get("away_team_fouls_l5", 0.0) or 0.0), 4),
            "away_team_fouls_l10": round(float(row.get("away_team_fouls_l10", 0.0) or 0.0), 4),
            "home_team_tackles_l5": round(float(row.get("home_team_tackles_l5", 0.0) or 0.0), 4),
            "home_team_tackles_l10": round(float(row.get("home_team_tackles_l10", 0.0) or 0.0), 4),
            "away_team_tackles_l5": round(float(row.get("away_team_tackles_l5", 0.0) or 0.0), 4),
            "away_team_tackles_l10": round(float(row.get("away_team_tackles_l10", 0.0) or 0.0), 4),
            "home_team_interceptions_l5": round(float(row.get("home_team_interceptions_l5", 0.0) or 0.0), 4),
            "away_team_interceptions_l5": round(float(row.get("away_team_interceptions_l5", 0.0) or 0.0), 4),
            "home_team_dribbled_past_l5": round(float(row.get("home_team_dribbled_past_l5", 0.0) or 0.0), 4),
            "away_team_dribbled_past_l5": round(float(row.get("away_team_dribbled_past_l5", 0.0) or 0.0), 4),
            "home_team_shots_l5": round(float(row.get("home_team_shots_l5", 0.0) or 0.0), 4),
            "home_team_shots_l10": round(float(row.get("home_team_shots_l10", 0.0) or 0.0), 4),
            "away_team_shots_l5": round(float(row.get("away_team_shots_l5", 0.0) or 0.0), 4),
            "away_team_shots_l10": round(float(row.get("away_team_shots_l10", 0.0) or 0.0), 4),
            "home_team_shots_on_goal_l5": round(float(row.get("home_team_shots_on_goal_l5", 0.0) or 0.0), 4),
            "home_team_shots_on_goal_l10": round(float(row.get("home_team_shots_on_goal_l10", 0.0) or 0.0), 4),
            "away_team_shots_on_goal_l5": round(float(row.get("away_team_shots_on_goal_l5", 0.0) or 0.0), 4),
            "away_team_shots_on_goal_l10": round(float(row.get("away_team_shots_on_goal_l10", 0.0) or 0.0), 4),
            "home_team_corners_for_l5": round(float(row.get("home_team_corners_for_l5", 0.0) or 0.0), 4),
            "home_team_corners_for_l10": round(float(row.get("home_team_corners_for_l10", 0.0) or 0.0), 4),
            "away_team_corners_for_l5": round(float(row.get("away_team_corners_for_l5", 0.0) or 0.0), 4),
            "away_team_corners_for_l10": round(float(row.get("away_team_corners_for_l10", 0.0) or 0.0), 4),
            "home_team_corners_against_l5": round(float(row.get("home_team_corners_against_l5", 0.0) or 0.0), 4),
            "home_team_corners_against_l10": round(float(row.get("home_team_corners_against_l10", 0.0) or 0.0), 4),
            "away_team_corners_against_l5": round(float(row.get("away_team_corners_against_l5", 0.0) or 0.0), 4),
            "away_team_corners_against_l10": round(float(row.get("away_team_corners_against_l10", 0.0) or 0.0), 4),
            "home_team_possession_l5": round(float(row.get("home_team_possession_l5", 0.0) or 0.0), 4),
            "away_team_possession_l5": round(float(row.get("away_team_possession_l5", 0.0) or 0.0), 4),
            "home_team_passes_l5": round(float(row.get("home_team_passes_l5", 0.0) or 0.0), 4),
            "away_team_passes_l5": round(float(row.get("away_team_passes_l5", 0.0) or 0.0), 4),
            "h2h_total_fouls_l5": round(float(row.get("h2h_total_fouls_l5", 0.0) or 0.0), 4),
            "h2h_total_fouls_l10": round(float(row.get("h2h_total_fouls_l10", 0.0) or 0.0), 4),
            "h2h_total_tackles_l5": round(float(row.get("h2h_total_tackles_l5", 0.0) or 0.0), 4),
            "h2h_total_tackles_l10": round(float(row.get("h2h_total_tackles_l10", 0.0) or 0.0), 4),
            "h2h_total_shots_l5": round(float(row.get("h2h_total_shots_l5", 0.0) or 0.0), 4),
            "h2h_total_shots_on_goal_l5": round(float(row.get("h2h_total_shots_on_goal_l5", 0.0) or 0.0), 4),
            "h2h_total_corners_l5": round(float(row.get("h2h_total_corners_l5", 0.0) or 0.0), 4),
            "fixture_foul_density_score": round(float(row.get("fixture_foul_density_score", 0.0) or 0.0), 4),
            "fixture_tackle_density_score": round(float(row.get("fixture_tackle_density_score", 0.0) or 0.0), 4),
            "fixture_midfield_grind_score": round(float(row.get("fixture_midfield_grind_score", 0.0) or 0.0), 4),
            "fixture_wide_duel_score": round(float(row.get("fixture_wide_duel_score", 0.0) or 0.0), 4),
            "fixture_style_label": str(row.get("fixture_style_label", "") or ""),
            "fixture_attack_pressure_score": round(float(row.get("fixture_attack_pressure_score", 0.0) or 0.0), 4),
            "fixture_corner_pressure_score": round(float(row.get("fixture_corner_pressure_score", 0.0) or 0.0), 4),
            "fixture_territorial_stress_score": round(float(row.get("fixture_territorial_stress_score", 0.0) or 0.0), 4),
            "fixture_attacking_style_label": str(row.get("fixture_attacking_style_label", "") or ""),
            "book_yellow_odds": "",
            "book_foul_line": "",
            "book_foul_over_odds": "",
            "left_flank_dominance": left_flank_dom,
            "right_flank_dominance": right_flank_dom,
            "central_attack_dominance": central_attack_dom,
            "late_game_pressure_risk": 0.8 if str(position).upper() in {"D", "M"} else 0.4,
            "lead_protection_foul_risk": 0.9 if counterattack_defender_flag else (0.65 if central_battle_flag else 0.25),
            "trailing_frustration_risk": 0.75 if str(position).upper() == "F" else (0.45 if "Wide" in tactical_role else 0.35),
            "og_pre_match_xg_home": round(float(row.get("og_pre_match_xg_home", 0.0) or 0.0), 4),
            "og_pre_match_xg_away": round(float(row.get("og_pre_match_xg_away", 0.0) or 0.0), 4),
            "og_xg_total": round(float(row.get("og_xg_total", 0.0) or 0.0), 4),
            "og_xg_weaker_side": round(float(row.get("og_xg_weaker_side", 0.0) or 0.0), 4),
            "og_btts_pre": round(float(row.get("og_btts_pre", 0.0) or 0.0), 4),
            "og_over25_pre": round(float(row.get("og_over25_pre", 0.0) or 0.0), 4),
            "og_snap_over25_avg": round(float(row.get("og_snap_over25_avg", 0.0) or 0.0), 4),
            "og_home_power_rating": round(home_power, 4),
            "og_away_power_rating": round(away_power, 4),
            "og_power_gap_abs": round(float(row.get("og_power_gap_abs", 0.0) or 0.0), 4),
            "og_balance_score": round(float(row.get("og_balance_score", 0.0) or 0.0), 4),
            "og_goal_environment_score": round(float(row.get("og_goal_environment_score", 0.0) or 0.0), 4),
            "og_battle_on_score": round(float(row.get("og_battle_on_score", 0.0) or 0.0), 4),
            "og_goal_support_flag": int(float(row.get("og_goal_support_flag", 0.0) or 0.0) >= 1.0),
            "og_battle_on_flag": int(float(row.get("og_battle_on_flag", 0.0) or 0.0) >= 1.0),
            "og_goal_environment_label": str(row.get("og_goal_environment_label", "") or ""),
            "team_power_rating": round(team_power_rating, 4),
            "opponent_power_rating": round(opponent_power_rating, 4),
            "team_power_edge": round(team_power_edge, 4),
            "starting_xi_team_quality_score": round(float(row.get("starting_xi_team_quality_score", 0.0) or 0.0), 4),
            "starting_xi_attack_quality_score": round(float(row.get("starting_xi_attack_quality_score", 0.0) or 0.0), 4),
            "starting_xi_defensive_quality_score": round(float(row.get("starting_xi_defensive_quality_score", 0.0) or 0.0), 4),
            "starting_xi_avg_form_rating_l5": round(float(row.get("starting_xi_avg_form_rating_l5", 0.0) or 0.0), 4),
            "starting_xi_team_quality_rank_league": _safe_int(row.get("starting_xi_team_quality_rank_league", 0.0)),
            "starting_xi_team_quality_percentile_league": round(float(row.get("starting_xi_team_quality_percentile_league", 0.0) or 0.0), 4),
            "opponent_starting_xi_team_quality_score": round(float(row.get("opponent_starting_xi_team_quality_score", 0.0) or 0.0), 4),
            "starting_xi_quality_edge": round(float(row.get("starting_xi_quality_edge", 0.0) or 0.0), 4),
            "weaker_side_under_pressure_flag": weaker_side_under_pressure_flag,
            "power_gap_directional_pressure_score": round(power_gap_directional_pressure_score, 4),
            "weak_flank_overload_flag": weak_flank_overload_flag,
            "weak_left_flank_overload_flag": weak_left_flank_overload_flag,
            "weak_right_flank_overload_flag": weak_right_flank_overload_flag,
            "weak_midfield_overload_flag": weak_midfield_overload_flag,
            "weak_territory_protection_flag": weak_territory_protection_flag,
            "analyst_notes": _analyst_note(position, fouls_per90, yellow_cards_per90, ref_cards_per_match),
        }
        records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        out = pd.DataFrame(columns=SCHEMA_COLUMNS)
    else:
        out = out.reindex(columns=SCHEMA_COLUMNS)
        for col, default in MANUAL_STRING_DEFAULTS.items():
            if col in out.columns:
                out[col] = out[col].fillna(default).astype(str)
        out = out.sort_values(["match_date", "fixture_key", "player_team_side", "team_name", "player_name"]).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _default_path(league_tag: str, season: int) -> Path:
    return Path("data_sources/api_football/features/player_events") / f"player_events_fixture_input__{league_tag}__{season}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player-events fixture input from API-Football normalized tables.")
    parser.add_argument("--league-tag", required=True, help="League tag like Italy_Serie_A")
    parser.add_argument("--season", type=int, required=True, help="Season integer, e.g. 2024")
    parser.add_argument("--fixtures-csv", default="", help="Override fixtures csv path")
    parser.add_argument("--player-stats-csv", default="", help="Override player stats csv path")
    parser.add_argument("--team-stats-csv", default="", help="Override team stats csv path")
    parser.add_argument("--lineups-csv", default="", help="Override lineups csv path")
    parser.add_argument("--injuries-csv", default="", help="Override injuries csv path")
    parser.add_argument("--referee-csv", default="", help="Override referee profile csv path")
    parser.add_argument("--og-overlay-csv", default="", help="Optional OG goal-environment overlay csv path")
    parser.add_argument("--style-overlay-csv", default="", help="Optional fixture-style overlay csv path")
    parser.add_argument("--quality-overlay-csv", default="", help="Optional player form quality overlay csv path")
    parser.add_argument("--lineup-features-csv", default="", help="Optional fixture lineup feature csv path")
    parser.add_argument("--manual-side-csv", default="", help="Optional manual side enrichment csv path for elite fixtures")
    parser.add_argument("--output-csv", default="", help="Override output csv path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized = Path("data_sources/api_football/normalized")
    features = Path("data_sources/api_football/features/player_events")
    fixtures_csv = args.fixtures_csv or str(normalized / f"fixtures_master__{args.league_tag}__{args.season}.csv")
    player_stats_csv = args.player_stats_csv or str(normalized / f"match_player_stats__{args.league_tag}__{args.season}.csv")
    team_stats_csv = args.team_stats_csv or str(normalized / f"match_team_stats__{args.league_tag}__{args.season}.csv")
    lineups_csv = args.lineups_csv or str(normalized / f"lineups__{args.league_tag}__{args.season}.csv")
    injuries_csv = args.injuries_csv or str(normalized / f"injuries__{args.league_tag}__{args.season}.csv")
    referee_csv = args.referee_csv or str(features / f"referee_profiles__{args.league_tag}__{args.season}.csv")
    og_overlay_csv = args.og_overlay_csv or str(features / f"og_goal_environment_overlay__{args.league_tag}__{args.season}.csv")
    style_overlay_csv = args.style_overlay_csv or str(features / f"fixture_style_overlay__{args.league_tag}__{args.season}.csv")
    quality_overlay_csv = args.quality_overlay_csv or str(features / f"player_form_quality_overlay__{args.league_tag}__{args.season}.csv")
    lineup_features_csv = args.lineup_features_csv or str(Path("data_sources/api_football/features") / f"api_lineup_features__{args.league_tag}__{args.season}.csv")
    manual_side_csv = args.manual_side_csv or str(features / f"manual_side_enrichment__{args.league_tag}__{args.season}.csv")
    output_csv = args.output_csv or str(_default_path(args.league_tag, args.season))

    df = build_player_events_fixture_input(
        args.league_tag,
        args.season,
        fixtures_csv,
        player_stats_csv,
        team_stats_csv,
        lineups_csv,
        injuries_csv,
        referee_csv,
        og_overlay_csv,
        style_overlay_csv,
        quality_overlay_csv,
        lineup_features_csv,
        manual_side_csv,
        output_csv,
    )
    print(f"WROTE: {output_csv}")
    print(f"rows: {len(df)} | fixtures: {df['fixture_key'].nunique() if not df.empty else 0}")


if __name__ == "__main__":
    main()
