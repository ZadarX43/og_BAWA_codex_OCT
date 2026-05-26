#!/usr/bin/env python3
"""Fantasy football rules and scoring helpers.

Research/product foundation only. This module does not ingest, publish, or
redistribute official fantasy data. It encodes configurable squad constraints
and scoring rules so Odds Genius-derived projections can be turned into fantasy
points, squad validation, and optimiser constraints.

The default config is a versioned official-rule contract for the current FPL
rules surface captured from the public help/rules page. Future seasons should
create a new config/ruleset name rather than mutating old saved season state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


POSITION_GK = "GK"
POSITION_DEF = "DEF"
POSITION_MID = "MID"
POSITION_FWD = "FWD"
POSITIONS = (POSITION_GK, POSITION_DEF, POSITION_MID, POSITION_FWD)
CHIP_BENCH_BOOST = "BENCH_BOOST"
CHIP_FREE_HIT = "FREE_HIT"
CHIP_TRIPLE_CAPTAIN = "TRIPLE_CAPTAIN"
CHIP_WILDCARD = "WILDCARD"
CHIPS = (CHIP_BENCH_BOOST, CHIP_FREE_HIT, CHIP_TRIPLE_CAPTAIN, CHIP_WILDCARD)


def official_deadlines_2025_26() -> dict[int, str]:
    """Return official local deadline labels for the current published ruleset.

    FPL deadlines are season-specific and can move until 24 hours before the
    scheduled time. Store them as source labels in saved state, then refresh
    from the official game-state adapter for live operation.
    """

    return {
        1: "Fri 15 Aug 18:30",
        2: "Fri 22 Aug 18:30",
        3: "Sat 30 Aug 11:00",
        4: "Sat 13 Sep 11:00",
        5: "Sat 20 Sep 11:00",
        6: "Sat 27 Sep 11:00",
        7: "Fri 3 Oct 18:30",
        8: "Sat 18 Oct 11:00",
        9: "Fri 24 Oct 18:30",
        10: "Sat 1 Nov 13:30",
        11: "Sat 8 Nov 11:00",
        12: "Sat 22 Nov 11:00",
        13: "Sat 29 Nov 13:30",
        14: "Tue 2 Dec 18:00",
        15: "Sat 6 Dec 11:00",
        16: "Sat 13 Dec 13:30",
        17: "Sat 20 Dec 11:00",
        18: "Fri 26 Dec 18:30",
        19: "Tue 30 Dec 18:00",
        20: "Sat 3 Jan 11:00",
        21: "Tue 6 Jan 18:30",
        22: "Sat 17 Jan 11:00",
        23: "Sat 24 Jan 11:00",
        24: "Sat 31 Jan 13:30",
        25: "Fri 6 Feb 18:30",
        26: "Tue 10 Feb 18:00",
        27: "Sat 21 Feb 13:30",
        28: "Fri 27 Feb 18:30",
        29: "Tue 3 Mar 18:00",
        30: "Sat 14 Mar 13:30",
        31: "Fri 20 Mar 18:30",
        32: "Fri 10 Apr 18:30",
        33: "Sat 18 Apr 11:00",
        34: "Fri 24 Apr 18:30",
        35: "Fri 1 May 18:30",
        36: "Sat 9 May 11:00",
        37: "Fri 15 May 18:30",
        38: "Sun 24 May 14:30",
    }


def official_bps_points() -> dict[str, int]:
    """Bonus Points System event weights.

    These are exposed for explainability and future projection/backtest work.
    Tie allocation is handled separately because it depends on match rankings.
    """

    return {
        "playing_1_to_60_minutes": 3,
        "playing_over_60_minutes": 6,
        "penalty_goal": 12,
        "goal_gk_def": 12,
        "goal_mid": 18,
        "goal_fwd": 24,
        "assist": 9,
        "clean_sheet_gk_def": 12,
        "penalty_save": 8,
        "save_inside_box": 3,
        "save_outside_box": 2,
        "successful_open_play_cross": 1,
        "big_chance_created": 3,
        "two_cbi": 1,
        "three_recoveries": 1,
        "key_pass": 1,
        "successful_tackle": 2,
        "successful_dribble": 1,
        "winning_goal": 3,
        "goal_line_clearance": 9,
        "foul_won": 1,
        "shot_on_target": 2,
        "pass_completion_70_79": 2,
        "pass_completion_80_89": 4,
        "pass_completion_90_plus": 6,
        "goal_conceded_gk_def": -4,
        "penalty_conceded": -3,
        "penalty_miss": -6,
        "yellow_card": -3,
        "red_card": -9,
        "own_goal": -6,
        "big_chance_missed": -3,
        "error_leads_to_goal": -3,
        "error_leads_to_attempt": -1,
        "tackled": -1,
        "foul_conceded": -1,
        "offside": -1,
        "shot_off_target": -1,
    }


@dataclass(frozen=True)
class FPLRulesConfig:
    """Configurable fantasy ruleset.

    Prices are represented in tenths of a million to avoid float drift:
    45 means 4.5m, 1000 means 100.0m.
    """

    ruleset_name: str = "FPL_OFFICIAL_2025_26_HELP_CONTRACT_V1"
    season: str = "2025/26"
    budget_tenths: int = 1000
    max_players_per_club: int = 3
    squad_size: int = 15
    squad_position_counts: dict[str, int] = field(
        default_factory=lambda: {
            POSITION_GK: 2,
            POSITION_DEF: 5,
            POSITION_MID: 5,
            POSITION_FWD: 3,
        }
    )
    starting_xi_size: int = 11
    starting_minimums: dict[str, int] = field(
        default_factory=lambda: {
            POSITION_GK: 1,
            POSITION_DEF: 3,
            POSITION_MID: 2,
            POSITION_FWD: 1,
        }
    )
    starting_maximums: dict[str, int] = field(
        default_factory=lambda: {
            POSITION_GK: 1,
            POSITION_DEF: 5,
            POSITION_MID: 5,
            POSITION_FWD: 3,
        }
    )
    free_transfer_rollover_cap: int = 5
    free_transfers_per_gameweek: int = 1
    transfer_hit_cost: int = -4
    max_transfers_per_gameweek: int = 20
    unlimited_transfers_before_first_deadline: bool = True
    afcon_top_up_after_gameweek: int = 15
    afcon_top_up_target_gameweek: int = 16
    afcon_top_up_free_transfers: int = 5
    chip_counts: dict[str, int] = field(
        default_factory=lambda: {
            CHIP_BENCH_BOOST: 2,
            CHIP_FREE_HIT: 2,
            CHIP_TRIPLE_CAPTAIN: 2,
            CHIP_WILDCARD: 2,
        }
    )
    first_chip_deadline_gameweek: int = 19
    second_chip_start_gameweek: int = 20
    chips_one_per_gameweek: bool = True
    free_hit_cannot_be_consecutive: bool = True
    free_hit_cancellable: bool = False
    wildcard_cancellable: bool = False
    bench_boost_cancellable_before_deadline: bool = True
    triple_captain_cancellable_before_deadline: bool = True
    saved_free_transfers_retained_after_wildcard_or_free_hit: bool = True
    gameweek_deadlines_local: dict[int, str] = field(default_factory=official_deadlines_2025_26)
    minutes_under_60_points: int = 1
    minutes_60_plus_points: int = 2
    goal_points: dict[str, int] = field(
        default_factory=lambda: {
            POSITION_GK: 10,
            POSITION_DEF: 6,
            POSITION_MID: 5,
            POSITION_FWD: 4,
        }
    )
    assist_points: int = 3
    clean_sheet_points: dict[str, int] = field(
        default_factory=lambda: {
            POSITION_GK: 4,
            POSITION_DEF: 4,
            POSITION_MID: 1,
            POSITION_FWD: 0,
        }
    )
    saves_per_point: int = 3
    penalty_save_points: int = 5
    penalty_miss_points: int = -2
    yellow_card_points: int = -1
    red_card_points: int = -3
    own_goal_points: int = -2
    goals_conceded_per_minus: int = 2
    goals_conceded_points: int = -1
    defender_defcon_threshold: int = 10
    attacker_defcon_threshold: int = 12
    defensive_contribution_points: int = 2
    captain_multiplier: int = 2
    triple_captain_multiplier: int = 3
    bps_points: dict[str, int] = field(default_factory=official_bps_points)


@dataclass(frozen=True)
class PlayerPick:
    player_id: str
    team_id: str
    position: str
    price_tenths: int


@dataclass(frozen=True)
class PlayerEventLine:
    player_id: str
    position: str
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    clean_sheet: bool = False
    saves: int = 0
    penalty_saves: int = 0
    penalty_misses: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    goals_conceded: int = 0
    bonus_points: int = 0
    defensive_contributions: int = 0


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def normalize_position(position: object) -> str:
    text = str(position or "").strip().upper()
    aliases = {
        "GKP": POSITION_GK,
        "GOALKEEPER": POSITION_GK,
        "D": POSITION_DEF,
        "DEFENDER": POSITION_DEF,
        "M": POSITION_MID,
        "MIDFIELDER": POSITION_MID,
        "F": POSITION_FWD,
        "FORWARD": POSITION_FWD,
        "STRIKER": POSITION_FWD,
    }
    return aliases.get(text, text)


def count_by_position(players: Iterable[PlayerPick]) -> dict[str, int]:
    counts = {position: 0 for position in POSITIONS}
    for player in players:
        position = normalize_position(player.position)
        counts[position] = counts.get(position, 0) + 1
    return counts


def validate_squad(players: Iterable[PlayerPick], config: FPLRulesConfig | None = None) -> ValidationResult:
    cfg = config or FPLRulesConfig()
    squad = list(players)
    errors: list[str] = []
    warnings: list[str] = []

    if len(squad) != cfg.squad_size:
        errors.append(f"squad_size_expected_{cfg.squad_size}_got_{len(squad)}")

    duplicate_ids = sorted({player.player_id for player in squad if [p.player_id for p in squad].count(player.player_id) > 1})
    if duplicate_ids:
        errors.append("duplicate_player_ids:" + ",".join(duplicate_ids))

    total_cost = sum(int(player.price_tenths) for player in squad)
    if total_cost > cfg.budget_tenths:
        errors.append(f"budget_exceeded_{total_cost}_gt_{cfg.budget_tenths}")

    position_counts = count_by_position(squad)
    for position, expected in cfg.squad_position_counts.items():
        actual = position_counts.get(position, 0)
        if actual != expected:
            errors.append(f"{position.lower()}_count_expected_{expected}_got_{actual}")

    club_counts: dict[str, int] = {}
    for player in squad:
        club_counts[player.team_id] = club_counts.get(player.team_id, 0) + 1
    for club, count in sorted(club_counts.items()):
        if count > cfg.max_players_per_club:
            errors.append(f"club_limit_exceeded_{club}_{count}_gt_{cfg.max_players_per_club}")

    unknown_positions = sorted({normalize_position(player.position) for player in squad if normalize_position(player.position) not in POSITIONS})
    if unknown_positions:
        errors.append("unknown_positions:" + ",".join(unknown_positions))

    if total_cost <= cfg.budget_tenths - 150:
        warnings.append("large_budget_left_unspent")

    return ValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def validate_starting_xi(
    starting_player_ids: Iterable[str],
    squad: Iterable[PlayerPick],
    config: FPLRulesConfig | None = None,
) -> ValidationResult:
    cfg = config or FPLRulesConfig()
    starting_ids = list(starting_player_ids)
    squad_by_id = {player.player_id: player for player in squad}
    errors: list[str] = []

    if len(starting_ids) != cfg.starting_xi_size:
        errors.append(f"starting_xi_size_expected_{cfg.starting_xi_size}_got_{len(starting_ids)}")
    if len(set(starting_ids)) != len(starting_ids):
        errors.append("duplicate_starting_player_ids")

    missing = sorted(player_id for player_id in starting_ids if player_id not in squad_by_id)
    if missing:
        errors.append("starting_players_not_in_squad:" + ",".join(missing))

    starters = [squad_by_id[player_id] for player_id in starting_ids if player_id in squad_by_id]
    counts = count_by_position(starters)
    for position, minimum in cfg.starting_minimums.items():
        if counts.get(position, 0) < minimum:
            errors.append(f"{position.lower()}_starters_min_{minimum}_got_{counts.get(position, 0)}")
    for position, maximum in cfg.starting_maximums.items():
        if counts.get(position, 0) > maximum:
            errors.append(f"{position.lower()}_starters_max_{maximum}_got_{counts.get(position, 0)}")

    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_captaincy(
    captain_id: str,
    vice_captain_id: str,
    starting_player_ids: Iterable[str],
) -> ValidationResult:
    """Validate captain/vice-captain selection against the starting XI."""

    starters = {str(player_id) for player_id in starting_player_ids}
    captain = str(captain_id or "")
    vice = str(vice_captain_id or "")
    errors: list[str] = []

    if not captain:
        errors.append("captain_required")
    if not vice:
        errors.append("vice_captain_required")
    if captain and captain not in starters:
        errors.append("captain_must_be_starter")
    if vice and vice not in starters:
        errors.append("vice_captain_must_be_starter")
    if captain and vice and captain == vice:
        errors.append("captain_and_vice_must_differ")

    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_bench_order(
    bench_player_ids: Iterable[str],
    squad: Iterable[PlayerPick],
    starting_player_ids: Iterable[str],
    config: FPLRulesConfig | None = None,
) -> ValidationResult:
    """Validate the four-player bench order used for automatic substitutions."""

    cfg = config or FPLRulesConfig()
    bench_ids = [str(player_id) for player_id in bench_player_ids]
    starters = {str(player_id) for player_id in starting_player_ids}
    squad_by_id = {str(player.player_id): player for player in squad}
    expected_bench_size = cfg.squad_size - cfg.starting_xi_size
    errors: list[str] = []

    if len(bench_ids) != expected_bench_size:
        errors.append(f"bench_size_expected_{expected_bench_size}_got_{len(bench_ids)}")
    if len(set(bench_ids)) != len(bench_ids):
        errors.append("duplicate_bench_player_ids")

    missing = sorted(player_id for player_id in bench_ids if player_id not in squad_by_id)
    if missing:
        errors.append("bench_players_not_in_squad:" + ",".join(missing))

    overlap = sorted(player_id for player_id in bench_ids if player_id in starters)
    if overlap:
        errors.append("bench_players_also_starting:" + ",".join(overlap))

    expected = set(squad_by_id) - starters
    actual = set(bench_ids)
    omitted = sorted(expected - actual)
    if omitted:
        errors.append("bench_players_omitted:" + ",".join(omitted))

    return ValidationResult(ok=not errors, errors=tuple(errors))


def selling_price_tenths(purchase_price_tenths: int, current_price_tenths: int) -> int:
    """Return FPL selling price in tenths of a million.

    Managers keep half the profit on price rises, rounded down to the nearest
    0.1m. Price falls sell at the current lower price.
    """

    purchase = int(purchase_price_tenths)
    current = int(current_price_tenths)
    if current <= purchase:
        return current
    return purchase + ((current - purchase) // 2)


def transfer_cost_points(
    transfer_count: int,
    free_transfers: int,
    *,
    chip: str | None = None,
    before_first_deadline: bool = False,
    config: FPLRulesConfig | None = None,
) -> int:
    """Return net points cost for transfers preparing the next Gameweek."""

    cfg = config or FPLRulesConfig()
    normalized_chip = str(chip or "").strip().upper()
    count = max(0, int(transfer_count))
    free = max(0, int(free_transfers))
    if before_first_deadline and cfg.unlimited_transfers_before_first_deadline:
        return 0
    if normalized_chip in {CHIP_WILDCARD, CHIP_FREE_HIT}:
        return 0
    paid = max(0, count - free)
    return paid * int(cfg.transfer_hit_cost)


def next_free_transfers(
    current_free_transfers: int,
    transfer_count: int,
    *,
    chip: str | None = None,
    afcon_top_up: bool = False,
    config: FPLRulesConfig | None = None,
) -> int:
    """Project free transfers available next Gameweek after deadline processing."""

    cfg = config or FPLRulesConfig()
    current = min(max(0, int(current_free_transfers)), cfg.free_transfer_rollover_cap)
    count = max(0, int(transfer_count))
    normalized_chip = str(chip or "").strip().upper()

    if afcon_top_up:
        return cfg.afcon_top_up_free_transfers
    if normalized_chip in {CHIP_WILDCARD, CHIP_FREE_HIT} and cfg.saved_free_transfers_retained_after_wildcard_or_free_hit:
        return current
    if count <= 0:
        return min(cfg.free_transfer_rollover_cap, current + cfg.free_transfers_per_gameweek)
    return min(cfg.free_transfer_rollover_cap, max(cfg.free_transfers_per_gameweek, current - count + cfg.free_transfers_per_gameweek))


def chip_window(chip: str, gameweek: int, config: FPLRulesConfig | None = None) -> str:
    """Return FIRST_HALF, SECOND_HALF, or UNAVAILABLE for a chip/GW pair."""

    cfg = config or FPLRulesConfig()
    normalized = str(chip or "").strip().upper()
    gw = int(gameweek)
    if normalized not in CHIPS:
        return "UNAVAILABLE"
    if normalized in {CHIP_FREE_HIT, CHIP_WILDCARD} and gw <= 1:
        return "UNAVAILABLE"
    if gw <= cfg.first_chip_deadline_gameweek:
        return "FIRST_HALF"
    if gw >= cfg.second_chip_start_gameweek:
        return "SECOND_HALF"
    return "UNAVAILABLE"


def validate_chip_use(
    chip: str,
    gameweek: int,
    used_chip_gameweeks: dict[str, Iterable[int]] | None = None,
    *,
    already_active_chip: str | None = None,
    config: FPLRulesConfig | None = None,
) -> ValidationResult:
    """Validate season chip availability and one-chip-per-GW constraints."""

    cfg = config or FPLRulesConfig()
    normalized = str(chip or "").strip().upper()
    used = {str(k).upper(): {int(gw) for gw in v} for k, v in (used_chip_gameweeks or {}).items()}
    errors: list[str] = []

    if normalized not in CHIPS:
        errors.append("unknown_chip")
        return ValidationResult(False, tuple(errors))

    if already_active_chip and cfg.chips_one_per_gameweek and str(already_active_chip).strip().upper() != normalized:
        errors.append("only_one_chip_per_gameweek")

    window = chip_window(normalized, gameweek, cfg)
    if window == "UNAVAILABLE":
        errors.append(f"{normalized.lower()}_unavailable_gw_{int(gameweek)}")

    used_for_chip = used.get(normalized, set())
    if len(used_for_chip) >= cfg.chip_counts.get(normalized, 0):
        errors.append(f"{normalized.lower()}_uses_exhausted")

    if window == "FIRST_HALF" and any(gw <= cfg.first_chip_deadline_gameweek for gw in used_for_chip):
        errors.append(f"{normalized.lower()}_first_half_already_used")
    if window == "SECOND_HALF" and any(gw >= cfg.second_chip_start_gameweek for gw in used_for_chip):
        errors.append(f"{normalized.lower()}_second_half_already_used")

    if normalized == CHIP_FREE_HIT and cfg.free_hit_cannot_be_consecutive:
        previous = int(gameweek) - 1
        if previous in used_for_chip:
            errors.append("free_hit_cannot_be_consecutive")

    return ValidationResult(ok=not errors, errors=tuple(errors))


def played_in_gameweek(line: PlayerEventLine) -> bool:
    """FPL playing-status helper for captaincy and autosubs.

    The official rule treats an appearance or receiving a card as playing.
    """

    return int(line.minutes) > 0 or int(line.yellow_cards) > 0 or int(line.red_cards) > 0


def apply_auto_substitutions(
    starting_player_ids: Iterable[str],
    bench_player_ids: Iterable[str],
    squad: Iterable[PlayerPick],
    played_player_ids: Iterable[str],
    config: FPLRulesConfig | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Apply FPL automatic substitutions without breaking formation rules."""

    cfg = config or FPLRulesConfig()
    squad_list = list(squad)
    squad_by_id = {str(player.player_id): player for player in squad_list}
    played = {str(player_id) for player_id in played_player_ids}
    active = [str(player_id) for player_id in starting_player_ids]
    bench = [str(player_id) for player_id in bench_player_ids]
    notes: list[str] = []
    used_bench: set[str] = set()

    for starter_id in list(active):
        if starter_id in played:
            continue
        starter = squad_by_id.get(starter_id)
        if starter is None:
            continue
        starter_position = normalize_position(starter.position)
        candidates = [
            bench_id
            for bench_id in bench
            if bench_id not in used_bench
            and bench_id in played
            and bench_id in squad_by_id
            and (
                (starter_position == POSITION_GK and normalize_position(squad_by_id[bench_id].position) == POSITION_GK)
                or (starter_position != POSITION_GK and normalize_position(squad_by_id[bench_id].position) != POSITION_GK)
            )
        ]
        for bench_id in candidates:
            trial = [bench_id if player_id == starter_id else player_id for player_id in active]
            if validate_starting_xi(trial, squad_list, cfg).ok:
                active = trial
                used_bench.add(bench_id)
                notes.append(f"auto_sub:{starter_id}->{bench_id}")
                break

    return tuple(active), tuple(notes)


def score_event_line(line: PlayerEventLine, config: FPLRulesConfig | None = None) -> dict[str, int]:
    cfg = config or FPLRulesConfig()
    position = normalize_position(line.position)
    points: dict[str, int] = {}

    if line.minutes <= 0:
        points["minutes"] = 0
    elif line.minutes < 60:
        points["minutes"] = cfg.minutes_under_60_points
    else:
        points["minutes"] = cfg.minutes_60_plus_points

    points["goals"] = int(line.goals) * cfg.goal_points.get(position, 0)
    points["assists"] = int(line.assists) * cfg.assist_points
    points["clean_sheet"] = cfg.clean_sheet_points.get(position, 0) if line.clean_sheet and line.minutes >= 60 else 0
    points["saves"] = int(line.saves // cfg.saves_per_point) if position == POSITION_GK and cfg.saves_per_point else 0
    points["penalty_saves"] = int(line.penalty_saves) * cfg.penalty_save_points
    points["penalty_misses"] = int(line.penalty_misses) * cfg.penalty_miss_points
    points["yellow_cards"] = int(line.yellow_cards) * cfg.yellow_card_points
    points["red_cards"] = int(line.red_cards) * cfg.red_card_points
    points["own_goals"] = int(line.own_goals) * cfg.own_goal_points
    if position in {POSITION_GK, POSITION_DEF} and cfg.goals_conceded_per_minus:
        points["goals_conceded"] = int(line.goals_conceded // cfg.goals_conceded_per_minus) * cfg.goals_conceded_points
    else:
        points["goals_conceded"] = 0

    threshold = cfg.defender_defcon_threshold if position == POSITION_DEF else cfg.attacker_defcon_threshold
    if position in {POSITION_DEF, POSITION_MID, POSITION_FWD} and int(line.defensive_contributions) >= threshold:
        points["defensive_contributions"] = cfg.defensive_contribution_points
    else:
        points["defensive_contributions"] = 0

    points["bonus"] = int(line.bonus_points)
    points["total"] = sum(points.values())
    return points


def apply_captaincy(
    player_points: dict[str, int],
    captain_id: str,
    vice_captain_id: str,
    played_player_ids: Iterable[str],
    *,
    triple_captain: bool = False,
    config: FPLRulesConfig | None = None,
) -> dict[str, int]:
    cfg = config or FPLRulesConfig()
    out = dict(player_points)
    played = set(played_player_ids)
    multiplier = cfg.triple_captain_multiplier if triple_captain else cfg.captain_multiplier
    active_captain = captain_id if captain_id in played else vice_captain_id if vice_captain_id in played else ""
    if active_captain and active_captain in out:
        out[active_captain] = int(out[active_captain]) * multiplier
    return out
