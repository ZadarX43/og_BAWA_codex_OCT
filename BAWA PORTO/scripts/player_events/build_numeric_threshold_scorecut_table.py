from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def round_half(value: float) -> float:
    return round(value * 2) / 2.0


def propose_shift(row: pd.Series) -> float:
    delta = float(pd.to_numeric(row.get("avg_score_delta"), errors="coerce") or 0.0)
    signal = row.get("tuning_signal", "")
    if signal == "LOWER_SCORE_GATE":
        return round_half(max(-12.0, min(-1.0, delta * 0.5)))
    if signal == "RAISE_SCORE_GATE":
        return round_half(min(8.0, max(1.0, delta / 3.0)))
    return 0.0


def rationale(row: pd.Series) -> str:
    signal = row.get("tuning_signal", "")
    if signal == "LOWER_SCORE_GATE":
        return "Missed correct selections are clustering below the current 3Y expectation, so we lower the score cut by roughly half the average miss gap."
    if signal == "RAISE_SCORE_GATE":
        return "Near misses are still leaking through above expectation, so we tighten the score cut by about one-third of the average excess gap."
    return "No directional cut shift yet; keep the current gate and review with more sample."


def build_table(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Numeric Threshold Score-Cut Table\n\nNo rows matched.\n")
        return df
    out = df.copy()
    out["proposed_score_cut_shift"] = out.apply(propose_shift, axis=1)
    out["proposed_gate_action"] = out["proposed_score_cut_shift"].apply(
        lambda value: "LOWER" if value < 0 else "RAISE" if value > 0 else "HOLD"
    )
    out["rationale"] = out.apply(rationale, axis=1)
    out = out.sort_values(["market", "review_family", "prematch_risk_focus", "rows"], ascending=[True, True, True, False])
    out.to_csv(output_csv, index=False)

    lines = [
        "# Numeric Threshold Score-Cut Table",
        "",
        "- Proposed shifts are expressed in raw score points against the current player-market gate.",
        "- Negative values lower the gate to catch more valid survivors; positive values tighten the gate to cut leak-through.",
        "",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"- {row['market']} | {row['review_family']} | risk={row['prematch_risk_focus']} | signal={row['tuning_signal']} | shift={row['proposed_score_cut_shift']:+.1f} | rows={int(row['rows'])} | avg_score_delta={row['avg_score_delta']:.2f}"
        )
        lines.append(f"  rationale: {row['rationale']}")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turn threshold tuning signals into exact numeric score-cut proposals.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_table(args.input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
