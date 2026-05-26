import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "fpl"))

from fpl_rules_engine import (  # noqa: E402
    CHIP_BENCH_BOOST,
    CHIP_FREE_HIT,
    CHIP_TRIPLE_CAPTAIN,
    CHIP_WILDCARD,
    FPLRulesConfig,
    PlayerEventLine,
    PlayerPick,
    apply_auto_substitutions,
    chip_window,
    next_free_transfers,
    played_in_gameweek,
    score_event_line,
    selling_price_tenths,
    transfer_cost_points,
    validate_bench_order,
    validate_captaincy,
    validate_chip_use,
    validate_squad,
    validate_starting_xi,
)


def make_legal_squad():
    players = []
    idx = 1
    for position, count in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        for _ in range(count):
            players.append(PlayerPick(f"p{idx}", f"t{idx}", position, 50))
            idx += 1
    return players


def test_official_squad_and_lineup_contract():
    squad = make_legal_squad()
    starters = ["p1", "p3", "p4", "p5", "p8", "p9", "p10", "p11", "p13", "p14", "p15"]
    bench = ["p2", "p6", "p7", "p12"]

    assert validate_squad(squad).ok
    assert validate_starting_xi(starters, squad).ok
    assert validate_bench_order(bench, squad, starters).ok
    assert validate_captaincy("p13", "p8", starters).ok

    assert not validate_captaincy("p2", "p8", starters).ok
    assert not validate_captaincy("p13", "p13", starters).ok


def test_price_transfer_and_rollover_rules():
    assert selling_price_tenths(75, 78) == 76
    assert selling_price_tenths(75, 79) == 77
    assert selling_price_tenths(75, 73) == 73

    assert transfer_cost_points(1, 1) == 0
    assert transfer_cost_points(3, 1) == -8
    assert transfer_cost_points(1, 0) == -4
    assert transfer_cost_points(20, 5) == -60
    assert transfer_cost_points(12, 0, chip=CHIP_WILDCARD) == 0
    assert transfer_cost_points(12, 0, chip=CHIP_FREE_HIT) == 0
    assert transfer_cost_points(12, 0, before_first_deadline=True) == 0

    assert next_free_transfers(1, 0) == 2
    assert next_free_transfers(5, 0) == 5
    assert next_free_transfers(2, 1) == 2
    assert next_free_transfers(5, 5) == 1
    assert next_free_transfers(2, 12, chip=CHIP_FREE_HIT) == 2
    assert next_free_transfers(3, 12, chip=CHIP_WILDCARD) == 3
    assert next_free_transfers(1, 0, afcon_top_up=True) == 5


def test_chip_window_and_availability_rules():
    assert chip_window(CHIP_BENCH_BOOST, 1) == "FIRST_HALF"
    assert chip_window(CHIP_TRIPLE_CAPTAIN, 20) == "SECOND_HALF"
    assert chip_window(CHIP_FREE_HIT, 1) == "UNAVAILABLE"
    assert chip_window(CHIP_WILDCARD, 2) == "FIRST_HALF"

    assert validate_chip_use(CHIP_BENCH_BOOST, 18, {}).ok
    assert not validate_chip_use(CHIP_BENCH_BOOST, 18, {CHIP_BENCH_BOOST: [4]}).ok
    assert validate_chip_use(CHIP_BENCH_BOOST, 22, {CHIP_BENCH_BOOST: [4]}).ok
    assert not validate_chip_use(CHIP_FREE_HIT, 21, {CHIP_FREE_HIT: [20]}).ok
    assert not validate_chip_use(CHIP_WILDCARD, 2, {}, already_active_chip=CHIP_BENCH_BOOST).ok


def test_scoring_and_auto_substitution_contract():
    defender = PlayerEventLine(
        "p3",
        "DEF",
        minutes=90,
        goals=1,
        clean_sheet=True,
        goals_conceded=1,
        defensive_contributions=10,
        bonus_points=2,
    )
    scored = score_event_line(defender)
    assert scored["minutes"] == 2
    assert scored["goals"] == 6
    assert scored["clean_sheet"] == 4
    assert scored["defensive_contributions"] == 2
    assert scored["total"] == 16

    assert played_in_gameweek(PlayerEventLine("p1", "GK", minutes=0, yellow_cards=1))
    assert not played_in_gameweek(PlayerEventLine("p1", "GK", minutes=0))

    squad = make_legal_squad()
    starters = ["p1", "p3", "p4", "p5", "p8", "p9", "p10", "p11", "p13", "p14", "p15"]
    bench = ["p2", "p6", "p7", "p12"]
    played = {"p2", "p4", "p5", "p6", "p8", "p9", "p10", "p11", "p13", "p14", "p15"}
    active, notes = apply_auto_substitutions(starters, bench, squad, played)
    assert "p2" in active
    assert "p6" in active
    assert "p1" not in active
    assert "p3" not in active
    assert "auto_sub:p1->p2" in notes
    assert "auto_sub:p3->p6" in notes


if __name__ == "__main__":
    test_official_squad_and_lineup_contract()
    test_price_transfer_and_rollover_rules()
    test_chip_window_and_availability_rules()
    test_scoring_and_auto_substitution_contract()
