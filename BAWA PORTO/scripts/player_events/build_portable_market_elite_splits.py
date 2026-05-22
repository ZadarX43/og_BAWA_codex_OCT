from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ATTACK_MARKETS = {"shots", "shots_on_target"}
CONTACT_MARKETS = {"fouls_committed", "tackles"}


def _risk_focus_for_role(role: str) -> str:
    role_text = str(role or "")
    if role_text == "Holding midfielder":
        return "missing DM"
    if role_text == "Wide defender / wing-back":
        return "missing full-back"
    if role_text == "Centre-back enforcer":
        return "missing CB duel anchor"
    return "no core structural flag"


def _write_md(df: pd.DataFrame, output_md: str, title: str) -> None:
    lines = [f"# {title}", ""]
    if df.empty:
        lines.append("No rows matched.")
        Path(output_md).write_text("\n".join(lines) + "\n")
        return

    lines.append(
        f"- rows: {len(df)} | fixtures: {df['fixture_key'].nunique()} | markets: {', '.join(sorted(df['market'].astype(str).unique()))}"
    )
    lines.append("")
    for market, sub in df.groupby("market", sort=False):
        lines.append(f"## {market}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['fixture_key']} | {row['player_name']} ({row['team_name']}) | family={row['source_family_tag']} | score={row['market_score']:.1f} | quality={row['fixture_quality_score']:.3f} | bucket={row['priority_bucket']}"
            )
            lines.append(f"  prematch_risk_focus={row['prematch_risk_focus']} | {row['prematch_risk_note']}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")


def build_splits(
    input_csv: str,
    attack_csv: str,
    attack_md: str,
    contact_csv: str,
    contact_md: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        attack = pd.DataFrame()
        contact = pd.DataFrame()
    else:
        attack = (
            df[df["market"].astype(str).isin(ATTACK_MARKETS)]
            .sort_values(["priority_rank", "market_score", "fixture_quality_score"], ascending=[True, False, False])
            .reset_index(drop=True)
        )
        contact = (
            df[df["market"].astype(str).isin(CONTACT_MARKETS)]
            .sort_values(["priority_rank", "market_score", "fixture_quality_score"], ascending=[True, False, False])
            .reset_index(drop=True)
        )
        for part in [attack, contact]:
            part["prematch_risk_focus"] = part["tactical_role"].astype(str).map(_risk_focus_for_role)
            part["prematch_risk_note"] = part.apply(
                lambda row: (
                    f"Structural role check only: if the expected {row['tactical_role'].lower()} changes late, rerun this elite lane before review."
                    if row["prematch_risk_focus"] != "no core structural flag"
                    else "No core DM/full-back/CB structural flag on this row; inherit broader fixture risk from the weekend sheet."
                ),
                axis=1,
            )

    Path(attack_csv).parent.mkdir(parents=True, exist_ok=True)
    attack.to_csv(attack_csv, index=False)
    contact.to_csv(contact_csv, index=False)
    _write_md(attack, attack_md, "Portable Attack Elite")
    _write_md(contact, contact_md, "Portable Contact Elite")
    return attack, contact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split the portable specialist elite preset board into attack and contact exports.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--attack-csv", required=True)
    parser.add_argument("--attack-md", required=True)
    parser.add_argument("--contact-csv", required=True)
    parser.add_argument("--contact-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    attack, contact = build_splits(
        input_csv=args.input_csv,
        attack_csv=args.attack_csv,
        attack_md=args.attack_md,
        contact_csv=args.contact_csv,
        contact_md=args.contact_md,
    )
    print(f"WROTE: {args.attack_csv}")
    print(f"attack_rows: {len(attack)} | contact_rows: {len(contact)}")
