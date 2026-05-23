# Over 2.5 From Existing OU25 Walk-Forward Audit

- Run root: `predictions_output/ou25_walkforward/rulebook_ftr_validation_3yr_19lg_v1`
- Filter applied: `market == ou25` and `bookie_pick == OVER`
- Purpose: determine whether filtering existing OU25 outputs is enough for Over 2.5 deployment.

## Branch leaderboard

                branch  months_present  total_rows  weighted_hit  weighted_roi  weighted_avg_odds  profitable_months  losing_months  worst_roi  max_drawdown
ou25_combined_topq_080               6         339      0.814159      0.424631           1.760206                  6              0   0.342308           0.0
    ou25_band1_124_176               6         577      0.814558      0.404523           1.736066                  6              0   0.283295           0.0
ou25_combined_baseline               6         501      0.812375      0.395788           1.731317                  6              0   0.288974           0.0
    ou25_band2_178_195               6         610      0.798361      0.394213           1.759361                  6              0   0.301250           0.0
   ou25_mode_over_only               6         516      0.792636      0.361550           1.730155                  6              0   0.243176           0.0

## Month-by-month detail

  month                 branch  rows      hit      roi  avg_odds
2024-11     ou25_band1_124_176   118 0.822034 0.402542  1.727712
2024-12     ou25_band1_124_176   113 0.814159 0.420796  1.745221
2025-01     ou25_band1_124_176    93 0.827957 0.425806  1.731720
2025-02     ou25_band1_124_176    88 0.750000 0.283295  1.735682
2025-03     ou25_band1_124_176    85 0.847059 0.457529  1.729176
2025-04     ou25_band1_124_176    80 0.825000 0.436750  1.748250
2024-11     ou25_band2_178_195   118 0.813559 0.401271  1.744153
2024-12     ou25_band2_178_195   113 0.787611 0.383894  1.755752
2025-01     ou25_band2_178_195    98 0.775510 0.345714  1.755816
2025-02     ou25_band2_178_195    96 0.750000 0.301250  1.759583
2025-03     ou25_band2_178_195    94 0.829787 0.468298  1.774468
2025-04     ou25_band2_178_195    91 0.835165 0.471648  1.771538
2024-11 ou25_combined_baseline   101 0.831683 0.407228  1.717426
2024-12 ou25_combined_baseline    97 0.773196 0.342268  1.734948
2025-01 ou25_combined_baseline    81 0.839506 0.441728  1.726667
2025-02 ou25_combined_baseline    78 0.756410 0.288974  1.732564
2025-03 ou25_combined_baseline    74 0.851351 0.468378  1.733108
2025-04 ou25_combined_baseline    70 0.828571 0.442571  1.748429
2024-11 ou25_combined_topq_080    65 0.815385 0.417538  1.761231
2024-12 ou25_combined_topq_080    70 0.771429 0.355429  1.757714
2025-01 ou25_combined_topq_080    54 0.851852 0.482037  1.752963
2025-02 ou25_combined_topq_080    52 0.769231 0.342308  1.764808
2025-03 ou25_combined_topq_080    49 0.836735 0.477551  1.763061
2025-04 ou25_combined_topq_080    49 0.857143 0.504082  1.762653
2024-11    ou25_mode_over_only   105 0.828571 0.403619  1.718381
2024-12    ou25_mode_over_only    97 0.752577 0.306598  1.731340
2025-01    ou25_mode_over_only    80 0.812500 0.395250  1.729750
2025-02    ou25_mode_over_only    85 0.729412 0.243176  1.731882
2025-03    ou25_mode_over_only    78 0.846154 0.460897  1.732564
2025-04    ou25_mode_over_only    71 0.788732 0.369014  1.741690
