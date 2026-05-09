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

resolve_allmarkets_src() {
  local src="$1"

  if [[ "$src" == *"__DEPLOY_"* ]]; then
    local prefix="${src%%__DEPLOY_*}"
    local candidate="${prefix}.csv"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  if [[ "$src" == *"ALLMARKETS"* && -f "$src" ]]; then
    printf '%s\n' "$src"
    return 0
  fi

  return 1
}

if [[ "${1:-}" == "--src" ]]; then
  publish_args=("$@")
else
  latest_src="$(resolve_latest_src)"
  publish_args=(--src "$latest_src")
fi

src_for_coverage=""
if [[ "${publish_args[0]:-}" == "--src" ]]; then
  src_for_coverage="${publish_args[1]:-}"
fi

if [[ -n "$src_for_coverage" ]]; then
  if allmarkets_src="$(resolve_allmarkets_src "$src_for_coverage")"; then
    echo "[publish] Running pre-ALLMARKETS fixture loss coverage report"
    python3 scripts/build_pre_allmarkets_fixture_loss_report.py --allmarkets-csv "$allmarkets_src"
  else
    echo "[publish] WARN: could not resolve ALLMARKETS source for fixture loss coverage report"
  fi
fi

echo "[publish] Running publish_predictions.py ${publish_args[*]}"
python3 publish_predictions.py "${publish_args[@]}"

echo "[publish] Running publish_fixture_intelligence.py ${publish_args[*]}"
python3 publish_fixture_intelligence.py "${publish_args[@]}"

echo "[publish] Running build_covered_fixture_universe.py ${publish_args[*]}"
python3 build_covered_fixture_universe.py "${publish_args[@]}"

echo "[publish] Validating exported JSON"
python3 validate_public_export.py
python3 validate_fixture_intelligence.py
python3 validate_covered_fixture_universe.py

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
echo "  3. review frontend/public/data/fixture_intelligence_public.json"
echo "  4. review frontend/public/data/covered_fixture_universe.json"
echo "  5. git add frontend/public/data publish_predictions.py publish_fixture_intelligence.py build_covered_fixture_universe.py reports/latest"
echo "  6. git commit and push dev"
