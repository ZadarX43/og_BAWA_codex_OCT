from __future__ import annotations

import argparse
from pathlib import Path


def build_checklist(helper_csv: str, refresh_script: str, dm_csv: str, winger_csv: str, top_sheet_csv: str, output_md: str) -> None:
    lines = [
        "# Pre-Kickoff Matchup Override Checklist",
        "",
        "1. Review the manual side helper queue:",
        f"   - `{helper_csv}`",
        "2. Update manual side tags for the flagged fixtures in the manual enrichment CSVs.",
        "3. Run the hotspot refresh after confirmed lineups:",
        f"   - `{refresh_script}`",
        "4. Review the matchup deploy sheets:",
        f"   - `{dm_csv}`",
        f"   - `{winger_csv}`",
        "5. Review the combined matchup top sheet:",
        f"   - `{top_sheet_csv}`",
        "",
        "Notes:",
        "- Prioritize winger-isolation fixtures first when side tags are still `UNSET`.",
        "- Keep `DM screen` rows live even without flank overrides; they are less side-sensitive.",
        "- Treat any `WATCH` matchup tag as research-only until it repeats.",
        "",
    ]
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small pre-kickoff checklist for matchup overrides.")
    parser.add_argument("--helper-csv", required=True)
    parser.add_argument("--refresh-script", required=True)
    parser.add_argument("--dm-csv", required=True)
    parser.add_argument("--winger-csv", required=True)
    parser.add_argument("--top-sheet-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_checklist(
        helper_csv=args.helper_csv,
        refresh_script=args.refresh_script,
        dm_csv=args.dm_csv,
        winger_csv=args.winger_csv,
        top_sheet_csv=args.top_sheet_csv,
        output_md=args.output_md,
    )
    print(f"WROTE: {args.output_md}")
