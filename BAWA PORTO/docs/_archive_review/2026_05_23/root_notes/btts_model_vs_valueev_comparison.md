# BTTS Model vs ValueEV Comparison

- Walk-forward root: `predictions_output/walk_forward`
- Deployment months requested: `['2024-11', '2024-12', '2025-01', '2025-02', '2025-03', '2025-04']`
- Shared months compared: `['2024-11', '2024-12', '2025-01', '2025-02', '2025-03', '2025-04']`
- Months compared: `6`
- League rows: `14`

## League head-to-head

                league league_bucket  months_present  shared_months_present  shared_total_rows  valueev_total_rows  valueev_weighted_hit  valueev_weighted_roi  model_total_rows  model_weighted_hit  model_weighted_roi winner_overall_by_roi winner_overall_by_hit
    Germany Bundesliga  other_league               4                      3                  8                   8                   1.0                0.4350                11            0.909091            0.301818          btts_valueev          btts_valueev
    Norway Eliteserien  other_league               2                      2                  4                   4                   1.0                0.4250                 5            0.800000            0.140000          btts_valueev          btts_valueev
           Belgium Pro  other_league               2                      2                  2                   2                   1.0                0.4400                 2            1.000000            0.440000               no_call               no_call
               USA MLS  other_league               2                      2                  2                   2                   1.0                0.4400                 2            1.000000            0.440000               no_call               no_call
     Europa Conference  other_league               3                      2                  4                   4                   1.0                0.4025                 5            0.800000            0.122000               no_call               no_call
              Japan J1  other_league               1                      1                  2                   2                   1.0                0.3850                 2            1.000000            0.385000               no_call               no_call
  Scotland Premiership    bad_league               1                      0                  0                   0                   NaN                   NaN                 1            1.000000            0.440000        not_comparable        not_comparable
        France Ligue 1  other_league               2                      0                  0                   0                   NaN                   NaN                 4            1.000000            0.430000        not_comparable        not_comparable
        England FA Cup    bad_league               2                      0                  0                   0                   NaN                   NaN                 2            1.000000            0.420000        not_comparable        not_comparable
         Europa League  other_league               4                      0                  0                   0                   NaN                   NaN                10            0.900000            0.276000        not_comparable        not_comparable
         Spain La Liga    bad_league               4                      0                  0                   0                   NaN                   NaN                 7            0.857143            0.222857        not_comparable        not_comparable
      Champions League  other_league               6                      0                  0                   0                   NaN                   NaN                 9            0.777778            0.098889        not_comparable        not_comparable
England Premier League    bad_league               5                      0                  0                   0                   NaN                   NaN                23            0.521739           -0.266957        not_comparable        not_comparable
  England EFL League 1    bad_league               2                      0                  0                   0                   NaN                   NaN                 2            0.500000           -0.300000        not_comparable        not_comparable

## Final winner table

            league league_bucket  shared_months_present  shared_total_rows  shared_valueev_rows  shared_model_rows winner_overall_by_roi head_to_head_call       deployment_interpretation automatic_recommendation       recommended_live_lane         comparison_outcome    comparison_vs_policy_status                   policy_summary     evidence_state
Germany Bundesliga  other_league                      3                  8                    8                 10          btts_valueev      btts_valueev model_primary_valueev_watchlist            watch valueEV model_primary_watch_valueev valueev_outperformed_model valueev_won_but_not_deployable Model primary; valueEV watchlist sparse_shared_rows
Norway Eliteserien  other_league                      2                  4                    4                  5          btts_valueev      btts_valueev model_primary_valueev_watchlist            watch valueEV model_primary_watch_valueev valueev_outperformed_model valueev_won_but_not_deployable Model primary; valueEV watchlist sparse_shared_rows
 Europa Conference  other_league                      2                  4                    4                  4               no_call           no_call            no_deploy_conclusion     no deploy conclusion                     no_call              no_clear_edge                  no_clear_edge             No deploy conclusion sparse_shared_rows
       Belgium Pro  other_league                      2                  2                    2                  2               no_call           no_call            no_deploy_conclusion     no deploy conclusion                     no_call              no_clear_edge                  no_clear_edge             No deploy conclusion sparse_shared_rows
           USA MLS  other_league                      2                  2                    2                  2               no_call           no_call            no_deploy_conclusion     no deploy conclusion                     no_call              no_clear_edge                  no_clear_edge             No deploy conclusion sparse_shared_rows
          Japan J1  other_league                      1                  2                    2                  2               no_call           no_call            no_deploy_conclusion     no deploy conclusion                     no_call              no_clear_edge                  no_clear_edge             No deploy conclusion sparse_shared_rows
## Policy footer

- BTTS deployment decision: Model live; ValueEV shadow/watch only; no ValueEV live promotion at this time.


## Monthly head-to-head

  month             league league_bucket  valueev_rows  valueev_hit  valueev_roi  model_rows  model_hit  model_roi winner_by_roi
2024-11  Europa Conference  other_league           3.0          1.0     0.390000           3   1.000000   0.390000           tie
2024-11 Germany Bundesliga  other_league           5.0          1.0     0.432000           6   1.000000   0.426667  btts_valueev
2024-11           Japan J1  other_league           2.0          1.0     0.385000           2   1.000000   0.385000           tie
2024-11 Norway Eliteserien  other_league           3.0          1.0     0.433333           4   0.750000   0.075000  btts_valueev
2024-12        Belgium Pro  other_league           1.0          1.0     0.420000           1   1.000000   0.420000           tie
2024-12 Norway Eliteserien  other_league           1.0          1.0     0.400000           1   1.000000   0.400000           tie
2025-01        Belgium Pro  other_league           1.0          1.0     0.460000           1   1.000000   0.460000           tie
2025-01 Germany Bundesliga  other_league           1.0          1.0     0.440000           1   1.000000   0.440000           tie
2025-02  Europa Conference  other_league           1.0          1.0     0.440000           1   1.000000   0.440000           tie
2025-02            USA MLS  other_league           1.0          1.0     0.440000           1   1.000000   0.440000           tie
2025-03            USA MLS  other_league           1.0          1.0     0.440000           1   1.000000   0.440000           tie
2025-04 Germany Bundesliga  other_league           2.0          1.0     0.440000           3   0.666667  -0.040000  btts_valueev
