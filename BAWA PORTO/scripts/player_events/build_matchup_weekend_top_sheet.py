from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def _risk_note_for_row(row: pd.Series) -> str:
    lane = str(row.get("matchup_lane", ""))
    role = str(row.get("tactical_role", "UNSET"))
    team = str(row.get("team_name", "This team"))
    if lane == "DM_SCREEN":
        return f"Missing DM risk: if {team}'s holding midfielder is absent pre-kickoff, central screen pressure can collapse and upset/goal-volatility risk should be reviewed."
    if lane == "WINGER_ISOLATION":
        return f"Missing full-back risk: if {team}'s wide defender/wing-back changes late, flank isolation pressure may flip and the contact/bookings read should be rechecked."
    if lane == "CB_DUEL_REVIEW":
        return f"Missing CB duel anchor risk: if {team}'s centre-back duel anchor is absent, direct striker pressure may be redistributed and both contact and pre-match goal assumptions can change."
    return f"Role-risk note: if the expected {role.lower()} changes late, recheck the matchup lane before review."


def _safe_text(value: object, default: str = "UNSET") -> str:
    text = str(value).strip()
    return default if text == "" or text.lower() == "nan" else text


def _fixture_risk_summary(sub: pd.DataFrame) -> str:
    notes: list[str] = []
    lanes = set(sub["matchup_lane"].astype(str))
    if "DM_SCREEN" in lanes:
        notes.append("missing DM")
    if "WINGER_ISOLATION" in lanes:
        notes.append("missing full-back")
    if "CB_DUEL_REVIEW" in lanes:
        notes.append("missing CB duel anchor")
    return ", ".join(notes) if notes else "general lineup recheck"


def build_top_sheet(dm_csv: str, winger_csv: str, output_csv: str, output_md: str, cb_csv: str = "") -> pd.DataFrame:
    dm = _read_csv(dm_csv)
    winger = _read_csv(winger_csv)
    cb = _read_csv(cb_csv) if cb_csv else pd.DataFrame()

    if not dm.empty:
        dm = dm.copy()
        dm["matchup_lane"] = "DM_SCREEN"
    if not winger.empty:
        winger = winger.copy()
        winger["matchup_lane"] = "WINGER_ISOLATION"
    if not cb.empty:
        cb = cb.copy()
        cb["matchup_lane"] = "CB_DUEL_REVIEW"

    combined = pd.concat([x for x in [dm, winger, cb] if not x.empty], ignore_index=True) if (not dm.empty or not winger.empty or not cb.empty) else pd.DataFrame()
    out_dir = Path(output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if combined.empty:
        combined.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Matchup Weekend Top Sheet\n\nNo rows matched.\n")
        return combined

    combined["override_bias"] = pd.to_numeric(
        combined.get("manual_side_override_active", pd.Series(0, index=combined.index)),
        errors="coerce",
    ).fillna(0.0)
    combined["summary_priority"] = (
        8.0 * pd.to_numeric(combined.get("cascade_strength", pd.Series(0.0, index=combined.index)), errors="coerce").fillna(0.0)
        + 5.0 * pd.to_numeric(combined.get("market_hit_rate", pd.Series(0.0, index=combined.index)), errors="coerce").fillna(0.0)
        + 3.0 * pd.to_numeric(combined.get("role_hit_rate", pd.Series(0.0, index=combined.index)), errors="coerce").fillna(0.0)
        + pd.to_numeric(combined.get("score", combined.get("market_score", pd.Series(0.0, index=combined.index))), errors="coerce").fillna(0.0) / 25.0
        + 2.0 * combined["override_bias"]
    )
    combined["prematch_risk_feedback_note"] = combined.apply(_risk_note_for_row, axis=1)

    combined = combined.sort_values(
        ["summary_priority", "cascade_strength", "fixture_key", "team_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    combined.to_csv(output_csv, index=False)

    lines = [
        "# Matchup Weekend Top Sheet",
        "",
        f"- fixtures: {combined['fixture_key'].nunique()} | rows: {len(combined)}",
        f"- lanes: DM screen={len(combined[combined['matchup_lane'].eq('DM_SCREEN')])} | winger isolation={len(combined[combined['matchup_lane'].eq('WINGER_ISOLATION')])} | cb duel review={len(combined[combined['matchup_lane'].eq('CB_DUEL_REVIEW')])}",
        "",
    ]

    for fixture_key, sub in combined.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(
            f"- {first['home_team_name']} vs {first['away_team_name']} | top_lane={first['matchup_lane']} | cascade_strength={float(first.get('cascade_strength', 0.0)):.1f}"
        )
        lines.append(f"- prematch_risk_focus: {_fixture_risk_summary(sub)}")
        for _, row in sub.iterrows():
            lines.append(
                f"- [{row['matchup_lane']}] {row['player_name']} ({row['team_name']}) | {row['market']} | {row['tactical_role']} | market_hit={float(row.get('market_hit_rate', 0.0)):.3f} | role_hit={float(row.get('role_hit_rate', 0.0)):.3f}"
            )
            lines.append(
                f"  context={row.get('opponent_flank_profile', 'UNSET')} | matchup={row.get('player_vs_player_matchup_tag', 'UNSET')} | priority={float(row.get('summary_priority', 0.0)):.2f}"
            )
            if str(row.get("matchup_lane", "")) == "CB_DUEL_REVIEW" and _safe_text(row.get("opponent_striker_profile", "UNSET")) != "UNSET":
                lines.append(
                    f"  striker_profile={_safe_text(row.get('opponent_striker_profile', 'UNSET'))} | pressure_tag={_safe_text(row.get('opponent_striker_pressure_tag','UNSET'))} | cb_duel_pressure={float(row.get('cb_duel_pressure_score', 0.0)):.3f} | subtype_rank={_safe_text(row.get('cb_subtype_rank_label','UNSET'))}"
                )
                lines.append(f"  subtype_note={_safe_text(row.get('opponent_striker_subtype_note','UNSET'))}")
            lines.append(f"  prematch_risk_note={row.get('prematch_risk_feedback_note', 'UNSET')}")
            if int(float(row.get("manual_side_override_active", 0) or 0)) == 1:
                lines.append(
                    f"  manual_override=YES | pitch_side={row.get('manual_pitch_side', 'UNSET')} | overload_target={row.get('manual_overload_target_side', 'UNSET')}"
                )
        lines.append("")

    Path(output_md).write_text("\n".join(lines) + "\n")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one combined matchup weekend top sheet from the DM-screen and winger-isolation lanes.")
    parser.add_argument("--dm-csv", required=True)
    parser.add_argument("--winger-csv", required=True)
    parser.add_argument("--cb-csv", default="")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_top_sheet(args.dm_csv, args.winger_csv, args.output_csv, args.output_md, args.cb_csv)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
