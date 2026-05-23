#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTOPSY = ROOT / "reports/latest/ftr_loss_autopsy_2026_05_14_to_2026_05_19/FTR_LOSS_AUTOPSY__2026-05-14_to_2026-05-19.csv"
DEFAULT_OUTDIR = ROOT / "reports/latest/ftr_shadow_throttle_2026_05_14_to_2026_05_19"

DEPLOY_TIERS = {"ELITE", "STANDARD"}
SETTLED_STATUSES = {"won", "lost"}


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    rule: Callable[[pd.Series], tuple[bool, list[str]]]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def flag(row: pd.Series, col: str) -> bool:
    return to_float(row.get(col), 0.0) > 0


def text(row: pd.Series, col: str) -> str:
    return str(row.get(col) or "").strip()


def tier(row: pd.Series) -> str:
    return text(row, "score_tier").upper()


def health_bucket(row: pd.Series) -> str:
    return text(row, "ftr_league_health_bucket").upper()


def reason_codes(row: pd.Series) -> set[str]:
    return {part for part in text(row, "autopsy_reason_codes").split("|") if part}


def picked_side_absence_hit(row: pd.Series) -> bool:
    return (
        to_float(row.get("picked_side_attack_absence_score"), 0.0) >= 10.0
        or to_float(row.get("picked_side_midfield_absence_score"), 0.0) >= 10.0
        or to_float(row.get("picked_side_defence_absence_score"), 0.0) >= 12.0
        or to_float(row.get("picked_side_lineup_confidence_score"), 100.0) < 97.0
    )


def unsafe_league_only(row: pd.Series) -> tuple[bool, list[str]]:
    block = health_bucket(row) == "UNSAFE" or "NOT_TRAINED_UNSAFE_FTR" in reason_codes(row)
    return block, ["UNSAFE_OR_UNTRAINED_LEAGUE"] if block else []


def red_or_unsafe_league(row: pd.Series) -> tuple[bool, list[str]]:
    block = health_bucket(row) in {"UNSAFE", "RED"}
    return block, [f"LEAGUE_HEALTH_{health_bucket(row)}"] if block else []


def weak_or_unknown_league(row: pd.Series) -> tuple[bool, list[str]]:
    block = health_bucket(row) in {"UNSAFE", "RED", "AMBER", "UNKNOWN"}
    return block, [f"LEAGUE_HEALTH_{health_bucket(row)}"] if block else []


def draw_shape_conflict(row: pd.Series) -> tuple[bool, list[str]]:
    if not flag(row, "draw_stalemate_risk_flag"):
        return False, []
    reasons: list[str] = ["DRAW_STALEMATE_RISK"]
    confirmers = [
        ("cs_conflict_flag", "CS_CONFLICT"),
        ("tg15_no_help_flag", "TG15_NO_HELP"),
        ("preview_counter_flag", "PREVIEW_COUNTER_SIGNAL"),
        ("market_price_weakness_flag", "MARKET_PRICE_WEAKNESS"),
    ]
    hits = [label for col, label in confirmers if flag(row, col)]
    block = bool(hits)
    return block, reasons + hits if block else []


def injury_picked_side(row: pd.Series) -> tuple[bool, list[str]]:
    block = flag(row, "injury_lineup_shock_flag") and picked_side_absence_hit(row)
    reasons = ["PICKED_SIDE_INJURY_OR_LINEUP_SHOCK"] if block else []
    return block, reasons


def end_season_requires_clean_state(row: pd.Series) -> tuple[bool, list[str]]:
    if not flag(row, "motivation_volatility_flag"):
        return False, []
    dirty = flag(row, "injury_lineup_shock_flag") or flag(row, "lineup_uncertainty_flag") or health_bucket(row) in {"UNSAFE", "RED", "AMBER"}
    reasons = ["MOTIVATION_VOLATILITY_WITH_DIRTY_TEAM_STATE"] if dirty else []
    return dirty, reasons


def preview_counter(row: pd.Series) -> tuple[bool, list[str]]:
    block = flag(row, "preview_counter_flag")
    return block, ["PREVIEW_COUNTER_SIGNAL"] if block else []


def conservative_stack(row: pd.Series) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for rule in [
        unsafe_league_only,
        red_or_unsafe_league,
        draw_shape_conflict,
        injury_picked_side,
        end_season_requires_clean_state,
        preview_counter,
    ]:
        blocked, rule_reasons = rule(row)
        if blocked:
            reasons.extend(rule_reasons)
    block = bool(reasons)
    return block, sorted(set(reasons))


def aggressive_stack(row: pd.Series) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    checks = [
        ("weak_or_unsafe_league_flag", "WEAK_OR_UNSAFE_LEAGUE"),
        ("draw_stalemate_risk_flag", "DRAW_STALEMATE_RISK"),
        ("injury_lineup_shock_flag", "INJURY_LINEUP_SHOCK"),
        ("lineup_uncertainty_flag", "LINEUP_UNCERTAINTY"),
        ("motivation_volatility_flag", "MOTIVATION_VOLATILITY"),
        ("cs_conflict_flag", "CS_CONFLICT"),
        ("tg15_no_help_flag", "TG15_NO_HELP"),
        ("preview_counter_flag", "PREVIEW_COUNTER"),
    ]
    for col, label in checks:
        if flag(row, col):
            reasons.append(label)
    block = bool(reasons)
    return block, reasons


def baseline(row: pd.Series) -> tuple[bool, list[str]]:
    return False, []


VARIANTS = [
    Variant("baseline_no_throttle", "No rows blocked; current FTR board reference.", baseline),
    Variant("unsafe_league_only", "Block FTR from untrained or explicitly unsafe leagues.", unsafe_league_only),
    Variant("league_red_or_unsafe", "Block unsafe plus low-health RED league FTR.", red_or_unsafe_league),
    Variant("league_weak_unknown_strict", "Block unsafe, RED, AMBER, or unknown league-health FTR.", weak_or_unknown_league),
    Variant("draw_shape_conflict", "Block draw/stalemate risk only when a second market-shape conflict confirms it.", draw_shape_conflict),
    Variant("injury_picked_side_shock", "Block picked-side injury/lineup shock when the selected side is materially affected.", injury_picked_side),
    Variant("end_season_clean_state", "Block motivation/end-season volatility unless team state and league health are clean.", end_season_requires_clean_state),
    Variant("preview_counter_signal", "Block rows with external preview counter-signal.", preview_counter),
    Variant("ftr_safety_stack_conservative", "Combine unsafe/RED league, confirmed draw conflict, picked-side shock, dirty motivation state, and preview counters.", conservative_stack),
    Variant("ftr_safety_stack_aggressive", "Block any major sidecar risk flag. Useful as an upper-bound suppression test.", aggressive_stack),
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fixture_label(row: pd.Series) -> str:
    return f"{text(row, 'home_team_name')} vs {text(row, 'away_team_name')}"


def profit_for(row: pd.Series) -> float:
    status = text(row, "result_status")
    if status not in SETTLED_STATUSES:
        return 0.0
    existing = row.get("profit_units")
    if str(existing).strip().lower() not in {"", "nan", "none"}:
        return to_float(existing, 0.0)
    if status == "won":
        return max(to_float(row.get("bookie_od"), 1.0) - 1.0, 0.0)
    return -1.0


def metric_summary(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    settled = [row for row in rows if row["result_status"] in SETTLED_STATUSES]
    wins = [row for row in settled if row["result_status"] == "won"]
    losses = [row for row in settled if row["result_status"] == "lost"]
    profit = sum(float(row["profit_units_calc"]) for row in settled)
    return {
        f"{prefix}_rows": len(rows),
        f"{prefix}_settled": len(settled),
        f"{prefix}_wins": len(wins),
        f"{prefix}_losses": len(losses),
        f"{prefix}_hit_rate": round(len(wins) / len(settled), 4) if settled else None,
        f"{prefix}_profit_units": round(profit, 4),
        f"{prefix}_roi": round(profit / len(settled), 4) if settled else None,
    }


def apply_variant(df: pd.DataFrame, variant: Variant, scope_name: str, scope_mask: pd.Series) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scoped = df.loc[scope_mask].copy()
    decisions: list[dict[str, Any]] = []
    for _, row in scoped.iterrows():
        blocked, reasons = variant.rule(row)
        status = text(row, "result_status")
        decisions.append(
            {
                "scope": scope_name,
                "variant": variant.name,
                "variant_description": variant.description,
                "fixture_key": text(row, "fixture_key"),
                "match_date": text(row, "match_date"),
                "league": text(row, "league"),
                "score_tier": tier(row),
                "fixture": fixture_label(row),
                "pick": text(row, "pick"),
                "actual": text(row, "actual"),
                "result_status": status,
                "bookie_od": to_float(row.get("bookie_od"), 0.0),
                "profit_units_calc": profit_for(row),
                "blocked_flag": int(blocked),
                "retained_flag": int(not blocked),
                "prevented_loss_flag": int(blocked and status == "lost"),
                "missed_winner_flag": int(blocked and status == "won"),
                "block_reason_codes": "|".join(reasons),
                "autopsy_reason_codes": text(row, "autopsy_reason_codes"),
                "ftr_league_health_bucket": health_bucket(row),
                "draw_stalemate_risk_flag": int(flag(row, "draw_stalemate_risk_flag")),
                "injury_lineup_shock_flag": int(flag(row, "injury_lineup_shock_flag")),
                "motivation_volatility_flag": int(flag(row, "motivation_volatility_flag")),
                "cs_conflict_flag": int(flag(row, "cs_conflict_flag")),
                "tg15_no_help_flag": int(flag(row, "tg15_no_help_flag")),
                "lineup_uncertainty_flag": int(flag(row, "lineup_uncertainty_flag")),
                "preview_counter_flag": int(flag(row, "preview_counter_flag")),
                "market_price_weakness_flag": int(flag(row, "market_price_weakness_flag")),
            }
        )

    original = metric_summary(decisions, prefix="original")
    retained = metric_summary([row for row in decisions if row["retained_flag"]], prefix="retained")
    blocked = metric_summary([row for row in decisions if row["blocked_flag"]], prefix="blocked")
    prevented = sum(row["prevented_loss_flag"] for row in decisions)
    missed = sum(row["missed_winner_flag"] for row in decisions)
    summary = {
        "scope": scope_name,
        "variant": variant.name,
        "variant_description": variant.description,
        **original,
        **retained,
        **blocked,
        "prevented_losses": prevented,
        "missed_winners": missed,
        "net_settled_saves": prevented - missed,
        "retained_hit_rate_delta": (
            round((retained["retained_hit_rate"] or 0.0) - (original["original_hit_rate"] or 0.0), 4)
            if retained["retained_hit_rate"] is not None and original["original_hit_rate"] is not None
            else None
        ),
        "retained_roi_delta": (
            round((retained["retained_roi"] or 0.0) - (original["original_roi"] or 0.0), 4)
            if retained["retained_roi"] is not None and original["original_roi"] is not None
            else None
        ),
    }
    return summary, decisions


def group_impact(decisions: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, rows in groups.items():
        item = {key: group_key[idx] for idx, key in enumerate(keys)}
        item.update(metric_summary(rows, prefix="original"))
        item.update(metric_summary([row for row in rows if row["retained_flag"]], prefix="retained"))
        item["blocked_rows"] = sum(row["blocked_flag"] for row in rows)
        item["prevented_losses"] = sum(row["prevented_loss_flag"] for row in rows)
        item["missed_winners"] = sum(row["missed_winner_flag"] for row in rows)
        out.append(item)
    return sorted(out, key=lambda row: tuple(str(row.get(key, "")) for key in keys))


def reason_impact(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    prevented: Counter[tuple[str, str, str]] = Counter()
    missed: Counter[tuple[str, str, str]] = Counter()
    for row in decisions:
        reasons = [part for part in str(row.get("block_reason_codes") or "").split("|") if part]
        for reason in reasons:
            key = (row["scope"], row["variant"], reason)
            counts[key] += 1
            if row["prevented_loss_flag"]:
                prevented[key] += 1
            if row["missed_winner_flag"]:
                missed[key] += 1
    return [
        {
            "scope": scope,
            "variant": variant,
            "block_reason": reason,
            "blocked_rows": count,
            "prevented_losses": prevented[(scope, variant, reason)],
            "missed_winners": missed[(scope, variant, reason)],
        }
        for (scope, variant, reason), count in sorted(counts.items())
    ]


def markdown(summary_rows: list[dict[str, Any]], assumptions: list[str], run_label: str) -> str:
    deploy_rows = [row for row in summary_rows if row["scope"] == "DEPLOY_ELITE_STANDARD"]
    audit_rows = [row for row in summary_rows if row["scope"] == "AUDIT_ALL_TIERS_OBSERVE_INCLUDED"]
    best = sorted(
        [
            row
            for row in deploy_rows
            if row["variant"] != "baseline_no_throttle" and int(row.get("retained_settled") or 0) > 0
        ],
        key=lambda row: (
            row.get("retained_hit_rate_delta") if row.get("retained_hit_rate_delta") is not None else -999,
            row.get("retained_roi_delta") if row.get("retained_roi_delta") is not None else -999,
            row.get("retained_settled", 0),
        ),
        reverse=True,
    )[:5]
    audit_best = sorted(
        [
            row
            for row in audit_rows
            if row["variant"] != "baseline_no_throttle" and int(row.get("retained_settled") or 0) > 0
        ],
        key=lambda row: (
            row.get("retained_roi_delta") if row.get("retained_roi_delta") is not None else -999,
            row.get("retained_hit_rate_delta") if row.get("retained_hit_rate_delta") is not None else -999,
            row.get("retained_settled", 0),
        ),
        reverse=True,
    )[:5]

    lines = [
        "# FTR Shadow Throttle Backtest",
        "",
        f"Generated: `{now_utc()}`",
        f"Run: `{run_label}`",
        "",
        "Research-only sidecar. No production deploy gates or live routing were changed.",
        "",
        "## Deploy Scope Results",
        "",
        "| Variant | Original Settled | Retained Settled | Blocked Settled | Original Hit | Retained Hit | Original ROI | Retained ROI | Prevented Losses | Missed Winners |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in deploy_rows:
        lines.append(
            "| {variant} | {orig_settled} | {ret_settled} | {blk_settled} | {orig_hit} | {ret_hit} | {orig_roi} | {ret_roi} | {prevented} | {missed} |".format(
                variant=row["variant"],
                orig_settled=row["original_settled"],
                ret_settled=row["retained_settled"],
                blk_settled=row["blocked_settled"],
                orig_hit="n/a" if row["original_hit_rate"] is None else f"{row['original_hit_rate'] * 100:.1f}%",
                ret_hit="n/a" if row["retained_hit_rate"] is None else f"{row['retained_hit_rate'] * 100:.1f}%",
                orig_roi="n/a" if row["original_roi"] is None else f"{row['original_roi']:.3f}",
                ret_roi="n/a" if row["retained_roi"] is None else f"{row['retained_roi']:.3f}",
                prevented=row["prevented_losses"],
                missed=row["missed_winners"],
            )
        )

    lines.extend(["", "## Best Deploy-Scope Variants By Hit-Rate Lift", ""])
    if not best:
        lines.append("No non-baseline deploy-scope variant retained a positive settled sample with hit-rate lift. Variants that block all five deploy FTR rows should be treated as suppression tests, not deploy policies.")
    else:
        for row in best:
            lines.append(
                f"- `{row['variant']}` retained `{row['retained_settled']}` settled picks, "
                f"prevented `{row['prevented_losses']}` losses, missed `{row['missed_winners']}` winners, "
                f"retained hit `{row['retained_hit_rate']}` and ROI `{row['retained_roi']}`."
            )

    lines.extend(
        [
            "",
            "## Audit Scope Results",
            "",
            "OBSERVE is included here only to inspect signal behavior. These rows remain non-deployable.",
            "",
            "| Variant | Original Settled | Retained Settled | Original Hit | Retained Hit | Original ROI | Retained ROI | Prevented Losses | Missed Winners |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in audit_rows:
        lines.append(
            "| {variant} | {orig_settled} | {ret_settled} | {orig_hit} | {ret_hit} | {orig_roi} | {ret_roi} | {prevented} | {missed} |".format(
                variant=row["variant"],
                orig_settled=row["original_settled"],
                ret_settled=row["retained_settled"],
                orig_hit="n/a" if row["original_hit_rate"] is None else f"{row['original_hit_rate'] * 100:.1f}%",
                ret_hit="n/a" if row["retained_hit_rate"] is None else f"{row['retained_hit_rate'] * 100:.1f}%",
                orig_roi="n/a" if row["original_roi"] is None else f"{row['original_roi']:.3f}",
                ret_roi="n/a" if row["retained_roi"] is None else f"{row['retained_roi']:.3f}",
                prevented=row["prevented_losses"],
                missed=row["missed_winners"],
            )
        )
    lines.extend(["", "## Strongest Audit-Scope Variants", ""])
    for row in audit_best:
        lines.append(
            f"- `{row['variant']}` retained `{row['retained_settled']}` settled audit rows, "
            f"moved hit rate from `{row['original_hit_rate']}` to `{row['retained_hit_rate']}`, "
            f"moved ROI from `{row['original_roi']}` to `{row['retained_roi']}`, "
            f"prevented `{row['prevented_losses']}` losses and missed `{row['missed_winners']}` winners."
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `FTR_SHADOW_THROTTLE_BACKTEST__{run_label}.csv`",
            f"- `FTR_SHADOW_THROTTLE_DECISIONS__{run_label}.csv`",
            f"- `FTR_SHADOW_THROTTLE_LEAGUE_IMPACT__{run_label}.csv`",
            f"- `FTR_SHADOW_THROTTLE_TIER_IMPACT__{run_label}.csv`",
            f"- `FTR_SHADOW_THROTTLE_REASON_IMPACT__{run_label}.csv`",
            "",
            "## Assumptions And Missing Proof",
            "",
        ]
    )
    for item in assumptions:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-only FTR shadow throttle simulator.")
    parser.add_argument("--autopsy-csv", type=Path, default=DEFAULT_AUTOPSY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--run-label", default="2026-05-14_to_2026-05-19")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    autopsy_csv = args.autopsy_csv if args.autopsy_csv.is_absolute() else ROOT / args.autopsy_csv
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(autopsy_csv, low_memory=False)
    df = df[df["score_market"].astype(str).str.upper().eq("FTR")].copy()
    df["profit_units_calc"] = df.apply(profit_for, axis=1)

    scopes = {
        "DEPLOY_ELITE_STANDARD": df["score_tier"].astype(str).str.upper().isin(DEPLOY_TIERS),
        "AUDIT_ALL_TIERS_OBSERVE_INCLUDED": pd.Series(True, index=df.index),
    }

    summary_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for scope_name, scope_mask in scopes.items():
        for variant in VARIANTS:
            summary, decisions = apply_variant(df, variant, scope_name, scope_mask)
            summary_rows.append(summary)
            decision_rows.extend(decisions)

    league_rows = group_impact(decision_rows, ["scope", "variant", "league"])
    tier_rows = group_impact(decision_rows, ["scope", "variant", "score_tier"])
    reason_rows = reason_impact(decision_rows)

    assumptions = [
        "Input flags are inherited from the FTR loss-autopsy sidecar, not recomputed from raw provider snapshots.",
        "May 14-19 injury and sidelined evidence is useful for proof-of-concept, but every historical expansion must prove pre-kickoff or pre-deploy timestamp eligibility.",
        "League-health buckets are seeded from current FTR health observations and must become rolling, window-safe league-health features for the 3-year study.",
        "OBSERVE rows are included only in the audit scope and remain non-deployable.",
        "External preview counters are optional sidecar evidence and should be excluded when publish-time evidence cannot be proved.",
    ]

    run = args.run_label
    write_csv(outdir / f"FTR_SHADOW_THROTTLE_BACKTEST__{run}.csv", summary_rows)
    write_csv(outdir / f"FTR_SHADOW_THROTTLE_DECISIONS__{run}.csv", decision_rows)
    write_csv(outdir / f"FTR_SHADOW_THROTTLE_LEAGUE_IMPACT__{run}.csv", league_rows)
    write_csv(outdir / f"FTR_SHADOW_THROTTLE_TIER_IMPACT__{run}.csv", tier_rows)
    write_csv(outdir / f"FTR_SHADOW_THROTTLE_REASON_IMPACT__{run}.csv", reason_rows)
    (outdir / "SUMMARY.md").write_text(markdown(summary_rows, assumptions, run), encoding="utf-8")
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": now_utc(),
                "run_label": run,
                "autopsy_csv": str(autopsy_csv.relative_to(ROOT)),
                "rows": int(len(df)),
                "variants": [variant.name for variant in VARIANTS],
                "scopes": list(scopes),
                "assumptions": assumptions,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote shadow throttle report: {outdir.relative_to(ROOT)}")
    deploy = [row for row in summary_rows if row["scope"] == "DEPLOY_ELITE_STANDARD"]
    for row in deploy:
        print(
            f"{row['variant']}: retained {row['retained_settled']}/{row['original_settled']} settled, "
            f"hit={row['retained_hit_rate']} roi={row['retained_roi']} "
            f"prevented={row['prevented_losses']} missed={row['missed_winners']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
