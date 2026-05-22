from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CAUTION_TEXT = {
    "missing DM": "Review FTR and BTTS caution: central screen loss can raise transition volume, weaken underdog resistance, and widen upset-risk tails.",
    "missing full-back": "Review BTTS and OU25 caution: flank isolation changes can swing crossing volume, recovery defending, and one-sided chance creation.",
    "missing CB duel anchor": "Review FTR and OU25 caution: direct striker pressure can redistribute across the line and change box-duel stability quickly.",
}


def build_caution(input_md: str, output_md: str) -> None:
    path = Path(input_md)
    lines = ["# Pre-Match Goal-Market Structural Caution", ""]
    if not path.exists():
        lines.append("No structural risk flags source file matched.")
        Path(output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(output_md).write_text("\n".join(lines) + "\n")
        return

    raw = path.read_text().splitlines()
    fixtures: list[tuple[str, str]] = []
    current_fixture = ""
    for line in raw:
        if line.startswith("## "):
            current_fixture = line.replace("## ", "").strip()
        elif line.startswith("- ") and " | focus=" in line and current_fixture:
            fixtures.append((current_fixture, line.split("| focus=", 1)[1].strip()))

    if not fixtures:
        lines.append("No fixture risk flags matched.")
        Path(output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(output_md).write_text("\n".join(lines) + "\n")
        return

    lines.append("- Beta caution only: these are structural prompts for pre-match goal-market review, not deploy overrides.")
    lines.append("")
    for fixture_key, focus_text in fixtures:
        focuses = [part.strip() for part in focus_text.split(",")]
        lines.append(f"## {fixture_key}")
        lines.append(f"- structural_flags={focus_text}")
        for focus in focuses:
            note = CAUTION_TEXT.get(focus, "General structural caution: recheck lineup-driven goal-market assumptions.")
            lines.append(f"- {focus}: {note}")
        lines.append("")

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny pre-match goal-market structural caution markdown from the broader structural risk flags.")
    parser.add_argument("--input-md", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_caution(args.input_md, args.output_md)
    print(f"WROTE: {args.output_md}")
