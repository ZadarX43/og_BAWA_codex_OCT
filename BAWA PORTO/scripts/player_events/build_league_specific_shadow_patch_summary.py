from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_LEAGUES = {'La Liga', 'Serie A', 'UEFA Europa League'}


def build(runner_csv: str, shadow_csv: str, regime_csv: str, output_md: str) -> None:
    runner = pd.read_csv(runner_csv, low_memory=False)
    shadow = pd.read_csv(shadow_csv, low_memory=False)
    regime = pd.read_csv(regime_csv, low_memory=False)
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)

    runner = runner[runner['league'].astype(str).isin(TARGET_LEAGUES)].copy()
    shadow = shadow.copy()

    lines = [
        '# League-Specific Shadow Patch Summary',
        '',
        '- Focused on `Serie A`, `La Liga`, and `UEFA Europa League` because those are the current live regime-learning leagues.',
        '- Research only: this summarizes how the shadow threshold trials interact with the current league posture board.',
        '',
    ]

    for league in ['Serie A', 'La Liga', 'UEFA Europa League']:
        lines.append(f'## {league}')
        league_regime = regime[regime['league'].astype(str) == league]
        if not league_regime.empty:
            reg = league_regime.iloc[0]
            lines.append(
                f"- regime={reg['regime_posture']} | avg_outlier={reg['avg_outlier_score']:.3f} | avg_patch_pressure={reg['avg_patch_pressure']:+.2f} | rows={int(reg['rows'])}"
            )
        league_rows = runner[runner['league'].astype(str) == league]
        if league_rows.empty:
            lines.append('- No runner rows matched this league in the current walkforward slice.')
            lines.append('')
            continue
        cohort_rows = []
        for _, patch in shadow.iterrows():
            cohort = league_rows[
                (league_rows['market'].astype(str) == str(patch['market']))
                & (league_rows['review_family'].astype(str) == str(patch['review_family']))
                & (league_rows['prematch_risk_focus'].astype(str) == str(patch['prematch_risk_focus']))
            ]
            if cohort.empty:
                continue
            cohort_rows.append(
                f"- {patch['market']} | {patch['review_family']} | risk={patch['prematch_risk_focus']} | patch={patch['patch_confidence']} | shift={patch['proposed_score_cut_shift']:+.1f} | league_rows={len(cohort)}"
            )
        if cohort_rows:
            lines.extend(cohort_rows)
        else:
            lines.append('- No shadow patch cohorts currently intersect this league directly.')
        lines.append('')

    Path(output_md).write_text('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a league-specific shadow patch summary for Serie A, La Liga, and Europa League.')
    parser.add_argument('--runner-csv', required=True)
    parser.add_argument('--shadow-csv', required=True)
    parser.add_argument('--regime-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build(args.runner_csv, args.shadow_csv, args.regime_csv, args.output_md)
    print(f'WROTE: {args.output_md}')
