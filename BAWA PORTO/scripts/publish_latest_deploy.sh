#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

resolve_latest_src() {
  local latest_dir
  latest_dir="$(find predictions_output -mindepth 1 -maxdepth 1 -type d | grep -E '/[0-9]{4}-[0-9]{2}-[0-9]{2}$' | sort | tail -n 1)"

  if [[ -z "${latest_dir:-}" ]]; then
    echo "[publish] ERROR: no dated predictions_output run directories found" >&2
    return 1
  fi

  local preset_src
  preset_src="$(find "$latest_dir" -maxdepth 1 -type f -name 'BOOKIE_*__DEPLOY_PRESET__*.csv' ! -name '*SCORED*' | sort | tail -n 1)"

  if [[ -z "${preset_src:-}" ]]; then
    echo "[publish] ERROR: no publishable DEPLOY_PRESET source found under $latest_dir" >&2
    echo "[publish] Use ./scripts/publish_latest_deploy.sh --src <deploy-csv>" >&2
    return 1
  fi

  printf '%s\n' "$preset_src"
}

if [[ "${1:-}" == "--src" ]]; then
  publish_args=("$@")
else
  latest_src="$(resolve_latest_src)"
  publish_args=(--src "$latest_src")
fi

echo "[publish] Running publish_predictions.py ${publish_args[*]}"
python3 publish_predictions.py "${publish_args[@]}"

echo "[publish] Validating exported JSON"
python3 validate_public_export.py

echo "[publish] Running static frontend smoke"
python3 scripts/smoke_frontend_static.py

if [[ -f "frontend/public/data/weekly_results.json" ]]; then
  echo "[publish] Validating weekly results snapshot"
  python3 validate_weekly_results.py
fi

echo
echo "[publish] Complete."
echo "Next recommended steps:"
echo "  1. review frontend/public/data/public_predictions.json"
echo "  2. review frontend/public/data/premium_predictions.json"
echo "  3. git add frontend/public/data publish_predictions.py reports/latest/PUBLISH_REPORT.md"
echo "  4. git commit and push dev"
