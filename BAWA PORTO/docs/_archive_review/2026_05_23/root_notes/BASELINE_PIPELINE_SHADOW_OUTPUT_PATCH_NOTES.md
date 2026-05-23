# Baseline Pipeline Shadow Output Patch Notes

## Summary
Added support for a shadow ModelStore output root for goal ensemble artifacts and
feature-health reports. Default production behavior is unchanged.

## Behavior
- Default remains **ModelStore/** when no env var is set.
- If `MODELSTORE_SHADOW_ROOT` is set, goal ensemble artifacts **and** goal feature-health
  reports are written there.

## Env var
- `MODELSTORE_SHADOW_ROOT=/path/to/ModelStore_shadow_goal_rebuild_2026_04_12`

## Affected artifacts (shadow-aware)
- `goal_ensembles/home_goals_fold5.pkl`
- `goal_ensembles/away_goals_fold5.pkl`
- `goal_ensembles/lambda_models_manifest.json`
- season-stamped goal ensemble manifests (if written)
- goal feature-health reports (`<league>_feature_health.csv`)

## Files patched
- `_baseline_ftr_pipeline.py`

## Shadow rebuild command (example)
```bash
MODELSTORE_SHADOW_ROOT="/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/ModelStore_shadow_goal_rebuild_2026_04_12" \
LEAGUES_FILE="/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/leagues_shadow_goal.txt" \
.venv/bin/python shadow_goal_rebuild_runner.py
```

## Notes
- No production overwrite occurs unless `MODELSTORE_SHADOW_ROOT` is set.
- Reading paths remain unchanged by default.
