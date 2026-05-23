# Promote Shadow Goal Ensembles → Production (Plan)

## Scope
Goal-ensemble artifacts only. No market model files touched.

## Shadow source
`ModelStore_shadow_goal_rebuild_2026_04_12/`

## Production target
`ModelStore/`

## League scope (safe leagues only)
- Australia A-League (shadow tag: Australia_A-League → prod tag: Australia_A_League)
- Austria Bundesliga
- Belgium Pro
- Brazil Serie A
- Champions League
- Czech First League
- Denmark Superliga
- England Championship
- England EFL League 1
- England FA Cup
- Europa Conference
- Europa League
- France Ligue 1
- Germany Bundesliga
- Germany Bundesliga 2
- Italy Serie A
- Japan J1
- Netherlands Eredivisie
- Norway Eliteserien
- Portugal Liga
- Saudi Pro League
- South Korea K League
- Swiss Super League
- Turkey Super Lig

## Explicit exclusions
- England Premier League
- Scotland Premiership
- Spain La Liga
- USA MLS
- Sweden Allsvenskan
- England U21

## Artifacts copied
- `goal_ensembles/home_goals_fold5.pkl`
- `goal_ensembles/away_goals_fold5.pkl`
- `goal_ensembles/lambda_models_manifest.json`
- any season-stamped goal ensemble manifests in `goal_ensembles/`

## Backups
Before each copy, the existing production `goal_ensembles/` folder is backed up into:
`ModelStore_goal_ensembles_backup_<timestamp>/`

## Outputs (when run)
- Promotion log file
- League-by-league actions CSV (updated with status)
- Rollback script (already prepared): `rollback_shadow_goal_ensembles.sh`

## Safety
- No writes outside `goal_ensembles/`
- No changes to `cat/`, `xgb/`, `ftr_v2.pkl`, `ftr_v2_xgb.pkl`, calibrators, thresholds

