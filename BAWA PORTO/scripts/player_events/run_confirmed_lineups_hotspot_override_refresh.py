from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYER_EVENTS_DIR = REPO_ROOT / 'scripts' / 'player_events'
COMBINED_BOARDS_DIR = REPO_ROOT / 'reports' / 'player_events' / 'combined_boards'
QUALITY_AUDITS_DIR = REPO_ROOT / 'reports' / 'player_events' / 'quality_audits'


def _run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(PLAYER_EVENTS_DIR / script), *args], cwd=REPO_ROOT, check=True)


def run_refresh() -> Path:
    _run('run_confirmed_lineups_override_refresh.py')
    _run(
        'build_team_specific_weekend_sheet.py',
        '--contact-csv', str(COMBINED_BOARDS_DIR / 'portable_contact_elite.csv'),
        '--bookings-csv', str(COMBINED_BOARDS_DIR / 'portable_bookings_super_elite.csv'),
        '--team-role-csv', str(QUALITY_AUDITS_DIR / 'team_family_role_audit.csv'),
        '--team-market-csv', str(QUALITY_AUDITS_DIR / 'team_family_role_audit__team_market.csv'),
        '--output-csv', str(COMBINED_BOARDS_DIR / 'team_specific_weekend_sheet.csv'),
        '--output-md', str(COMBINED_BOARDS_DIR / 'team_specific_weekend_sheet.md'),
    )
    _run(
        'build_team_role_bookings_guide.py',
        '--bookings-csv', str(COMBINED_BOARDS_DIR / 'portable_bookings_super_elite.csv'),
        '--team-role-csv', str(QUALITY_AUDITS_DIR / 'team_family_role_audit.csv'),
        '--team-market-csv', str(QUALITY_AUDITS_DIR / 'team_family_role_audit__team_market.csv'),
        '--output-csv', str(QUALITY_AUDITS_DIR / 'team_role_bookings_guide.csv'),
        '--output-md', str(QUALITY_AUDITS_DIR / 'team_role_bookings_guide.md'),
    )
    _run(
        'build_team_specific_specialist_deploy_guide.py',
        '--team-family-role-csv', str(QUALITY_AUDITS_DIR / 'team_family_role_audit.csv'),
        '--team-market-csv', str(QUALITY_AUDITS_DIR / 'team_family_role_audit__team_market.csv'),
        '--context-csv', str(COMBINED_BOARDS_DIR / 'team_specific_weekend_sheet.csv'),
        '--output-md', str(QUALITY_AUDITS_DIR / 'team_specific_specialist_deploy_guide.md'),
    )
    _run(
        'build_team_specific_contact_deploy_sheet.py',
        '--input-csv', str(COMBINED_BOARDS_DIR / 'team_specific_weekend_sheet.csv'),
        '--output-csv', str(COMBINED_BOARDS_DIR / 'team_specific_contact_deploy_sheet.csv'),
        '--output-md', str(COMBINED_BOARDS_DIR / 'team_specific_contact_deploy_sheet.md'),
    )
    _run(
        'build_team_specific_bookings_doubles_shortlist.py',
        '--guide-csv', str(QUALITY_AUDITS_DIR / 'team_role_bookings_guide.csv'),
        '--elite-csv', str(COMBINED_BOARDS_DIR / 'portable_bookings_super_elite.csv'),
        '--output-csv', str(COMBINED_BOARDS_DIR / 'team_specific_bookings_doubles_shortlist.csv'),
        '--output-md', str(COMBINED_BOARDS_DIR / 'team_specific_bookings_doubles_shortlist.md'),
    )
    _run(
        'build_team_specific_contact_bookings_cascade_board.py',
        '--input-csv', str(COMBINED_BOARDS_DIR / 'team_specific_weekend_sheet.csv'),
        '--output-csv', str(COMBINED_BOARDS_DIR / 'team_specific_contact_bookings_cascade_board.csv'),
        '--output-md', str(COMBINED_BOARDS_DIR / 'team_specific_contact_bookings_cascade_board.md'),
    )
    _run(
        'build_player_vs_player_hotspot_registry.py',
        '--input-csv', str(COMBINED_BOARDS_DIR / 'team_specific_weekend_sheet.csv'),
        '--output-csv', str(QUALITY_AUDITS_DIR / 'player_vs_player_hotspot_registry.csv'),
        '--output-md', str(QUALITY_AUDITS_DIR / 'player_vs_player_hotspot_registry.md'),
    )
    _run(
        'build_team_specific_contact_bookings_weekend_shortlist.py',
        '--input-csv', str(COMBINED_BOARDS_DIR / 'team_specific_contact_bookings_cascade_board.csv'),
        '--output-csv', str(COMBINED_BOARDS_DIR / 'team_specific_contact_bookings_weekend_shortlist.csv'),
        '--output-md', str(COMBINED_BOARDS_DIR / 'team_specific_contact_bookings_weekend_shortlist.md'),
        '--max-fixtures', '8',
    )

    summary = COMBINED_BOARDS_DIR / 'confirmed_lineups_hotspot_override_refresh_summary.md'
    summary.write_text(
        '\n'.join([
            '# Confirmed-Lineups Hotspot Override Refresh Summary',
            '',
            '- refreshed the base confirmed-lineups override flow',
            '- regenerated team-specific weekend, contact deploy, cascade, hotspot registry, and weekend shortlist outputs',
            '- use this runner after updating manual side tags for elite fixtures',
            '',
        ]) + '\n'
    )
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description='Refresh hotspot/cascade outputs after manual side overrides.').parse_args()


if __name__ == '__main__':
    parse_args()
    summary = run_refresh()
    print(f'WROTE: {summary}')
