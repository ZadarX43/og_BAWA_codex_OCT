# Goal Ensemble Shadow Rebuild Plan

## Target
Build only goal ensemble artifacts into:
ModelStore_shadow_goal_rebuild_2026_04_12/

## Rebuild artifacts (per league)
- goal_ensembles/home_goals_fold5.pkl
- goal_ensembles/away_goals_fold5.pkl
- goal_ensembles/lambda_models_manifest.json

## Do NOT retrain market models yet
No FTR / BTTS / OU25 retrain until goal layer is validated.

## Steps
1. Clean duplicate tags in ModelStore (quarantine).
2. Build shadow goal ensembles only.
3. Audit manifests in shadow folder.
4. Run a small sample deployment and compare outputs.
5. Promote shadow folder only if validated.

## Pending decision
Select unified schema target (legacy or upgraded safe schema).
