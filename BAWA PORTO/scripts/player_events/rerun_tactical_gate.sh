#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"
cd "$ROOT"

python3 scripts/player_events/build_tactical_lane_accumulation_tracker.py
python3 scripts/player_events/build_tactical_accumulation_delta_audit.py
python3 scripts/player_events/build_tactical_layer_completion_checklist.py

echo
echo "--- DELTA AUDIT ---"
sed -n '1,220p' reports/player_events/quality_audits/tactical_accumulation_delta_audit.md

echo
echo "--- COMPLETION CHECKLIST ---"
sed -n '1,240p' reports/player_events/quality_audits/TACTICAL_LAYER_COMPLETION_CHECKLIST.md
