from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONFIDENCE_MAP = {
    "LOW": 0.35,
    "MEDIUM": 0.55,
    "HIGH": 0.75,
}


def build_audit(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# CB Subtype Walkforward Audit\n\nNo rows matched.\n")
        return df

    cb = df[
        df.get("tactical_role", pd.Series("", index=df.index)).astype(str).eq("Centre-back enforcer")
        & df.get("opponent_striker_profile", pd.Series("UNSET", index=df.index)).astype(str).ne("UNSET")
    ].copy()

    if cb.empty:
        cb.to_csv(output_csv, index=False)
        Path(output_md).write_text("# CB Subtype Walkforward Audit\n\nNo centre-back enforcer rows with live striker profiles matched.\n")
        return cb

    family_col = "review_family" if "review_family" in cb.columns else "source_family"
    cb[family_col] = cb.get(family_col, pd.Series("UNSET", index=cb.index)).astype(str)
    market_hit_col = "market_hit_rate" if "market_hit_rate" in cb.columns else "__market_hit_proxy"
    role_hit_col = "role_hit_rate" if "role_hit_rate" in cb.columns else "__role_hit_proxy"
    score_col = "score" if "score" in cb.columns else "market_score"
    if market_hit_col == "__market_hit_proxy":
        cb[market_hit_col] = (
            cb.get("market_confidence", pd.Series("", index=cb.index))
            .astype(str)
            .str.upper()
            .map(CONFIDENCE_MAP)
            .fillna(0.0)
        )
    if role_hit_col == "__role_hit_proxy":
        cb[role_hit_col] = cb[market_hit_col]
    audit = (
        cb.groupby(["opponent_striker_profile", "market", family_col], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", pd.Series.nunique),
            avg_market_hit_rate=(market_hit_col, lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_role_hit_rate=(role_hit_col, lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_score=(score_col, lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_cb_duel_pressure=("cb_duel_pressure_score", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
        .sort_values(
            ["avg_market_hit_rate", "avg_cb_duel_pressure", "rows"],
            ascending=[False, False, False],
        )
    )
    audit.to_csv(output_csv, index=False)

    lines = [
        "# CB Subtype Walkforward Audit",
        "",
        "- Beta audit only: this is a scored hit-rate review proxy until we wire a fuller 3-year walkforward pass.",
        f"- cb_rows={len(cb)} | cb_fixtures={cb['fixture_key'].nunique()} | subtype_count={cb['opponent_striker_profile'].nunique()}",
        "",
    ]

    for subtype, sub in audit.groupby("opponent_striker_profile", sort=False):
        lines.append(f"## {subtype}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['market']} | {row[family_col]} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={row['avg_market_hit_rate']:.3f} | role_hit={row['avg_role_hit_rate']:.3f} | cb_pressure={row['avg_cb_duel_pressure']:.3f} | avg_score={row['avg_score']:.2f}"
            )
        lines.append("")

    Path(output_md).write_text("\n".join(lines) + "\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny CB subtype walkforward-style audit by subtype, market, and fixture family.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_audit(args.input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
