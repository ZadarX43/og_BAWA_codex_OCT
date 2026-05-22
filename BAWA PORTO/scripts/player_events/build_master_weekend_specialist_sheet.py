from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEEP_BUCKETS = ["P1_SUPER_ELITE", "P2_CONTACT_STACK", "P2_ATTACK_STACK"]
FIXTURE_BUCKET_SCORES = {"P1_SUPER_ELITE": 5, "P2_CONTACT_STACK": 3, "P2_ATTACK_STACK": 3}


def _fixture_note(row: pd.Series) -> str:
    families = int(row.get("specialist_family_count", 1) or 1)
    p1 = int(row.get("p1_rows", 0) or 0)
    if p1 >= 2 and families >= 2:
        return "Top-end agreement across families with multiple elite rows."
    if p1 >= 1:
        return "Has at least one super-elite survivor."
    if families >= 2:
        return "Strong cross-family support without a P1 row."
    return "Useful specialist fixture, but agreement is narrower."


def _risk_focus_for_role(role: str) -> str:
    role_text = str(role or "")
    if role_text == "Holding midfielder":
        return "missing DM"
    if role_text == "Wide defender / wing-back":
        return "missing full-back"
    if role_text == "Centre-back enforcer":
        return "missing CB duel anchor"
    return "no core structural flag"


def build_sheet(input_csv: str, output_csv: str, output_md: str, max_fixtures: int) -> pd.DataFrame:
    board = pd.read_csv(input_csv)
    if board.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Master Weekend Specialist Sheet\n\nNo rows matched.\n")
        return board

    shortlist = board[board["priority_bucket"].isin(KEEP_BUCKETS)].copy()
    if shortlist.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        shortlist.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Master Weekend Specialist Sheet\n\nNo shortlist rows matched.\n")
        return shortlist

    shortlist["fixture_bucket_points"] = shortlist["priority_bucket"].map(FIXTURE_BUCKET_SCORES).fillna(0).astype(int)
    fixture_rank = (
        shortlist.groupby(["fixture_key", "home_team_name", "away_team_name"], as_index=False)
        .agg(
            fixture_quality_score=("fixture_quality_score", "max"),
            top_market_score=("market_score", "max"),
            total_rows=("market", "count"),
            p1_rows=("priority_bucket", lambda s: int((pd.Series(s) == "P1_SUPER_ELITE").sum())),
            bucket_points=("fixture_bucket_points", "sum"),
            specialist_family_count=("specialist_family_count", "max"),
            specialist_families=("specialist_families", "max"),
            source_family_tag=("source_family_tag", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique()))),
        )
        .sort_values(
            ["bucket_points", "p1_rows", "fixture_quality_score", "top_market_score", "total_rows"],
            ascending=[False, False, False, False, False],
        )
        .reset_index(drop=True)
    )
    fixture_rank["fixture_confidence_note"] = fixture_rank.apply(_fixture_note, axis=1)
    keep_fixtures = fixture_rank.head(max_fixtures)["fixture_key"].tolist()

    out = shortlist[shortlist["fixture_key"].isin(keep_fixtures)].copy()
    out = out.merge(
        fixture_rank[["fixture_key", "fixture_confidence_note"]],
        on="fixture_key",
        how="left",
    )
    out["prematch_risk_focus"] = out["tactical_role"].astype(str).map(_risk_focus_for_role)
    out["prematch_risk_note"] = out.apply(
        lambda row: (
            f"Structural role check only: if the expected {row['tactical_role'].lower()} changes late, recheck this fixture before trusting the shortlist."
            if row["prematch_risk_focus"] != "no core structural flag"
            else "No core DM/full-back/CB structural flag on this row; inherit the wider fixture context instead."
        ),
        axis=1,
    )
    bucket_order = {bucket: idx for idx, bucket in enumerate(KEEP_BUCKETS, start=1)}
    out["priority_rank"] = out["priority_bucket"].map(bucket_order).fillna(99).astype(int)
    out = out.sort_values(["priority_rank", "fixture_quality_score", "market_score"], ascending=[True, False, False]).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Master Weekend Specialist Sheet",
        "",
        f"- top fixtures kept: {len(keep_fixtures)}",
        f"- kept buckets: {', '.join(KEEP_BUCKETS)}",
        "",
    ]
    for fixture_key in keep_fixtures:
        sub = out[out["fixture_key"] == fixture_key].copy()
        if sub.empty:
            continue
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(
            f"- fixture: {first['home_team_name']} vs {first['away_team_name']} | quality={first['fixture_quality_score']:.3f} | note={first['fixture_confidence_note']}"
        )
        lines.append(f"- families: {first.get('specialist_families', '')}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['priority_bucket']}: {row['player_name']} ({row['team_name']}) | {row['market']} | score={row['market_score']:.1f} | family={row.get('source_family_tag', '')}"
            )
            lines.append(f"  prematch_risk_focus={row['prematch_risk_focus']} | {row['prematch_risk_note']}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a trimmed master weekend specialist sheet from the master specialist board.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-fixtures", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = build_sheet(args.input_csv, args.output_csv, args.output_md, args.max_fixtures)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(df)} | fixtures: {df['fixture_key'].nunique() if not df.empty else 0}")
