# Over25 Model vs Filtered OU25 Comparison

- Filtered OU25 run root: `predictions_output/ou25_walkforward/rulebook_ftr_validation_3yr_19lg_v1`
- Upstream walk-forward root: `predictions_output/walk_forward`
- Filtered main branch compared: `ou25_band1_124_176`
- Filtered premium reference branch available upstream: `ou25_combined_topq_080`
- Shared comparison months: `2024-11, 2024-12, 2025-01, 2025-02, 2025-03, 2025-04`

## Purpose

Head-to-head comparison between filtered OU25 OVER selections and the dedicated league-specific `over25_v3` lane over the same shared walk-forward months only.

## League winners by ROI

                league league_bucket  months_present  filtered_weighted_roi  dedicated_weighted_roi  filtered_weighted_hit  dedicated_weighted_hit  winner_overall_by_roi  roi_month_wins_filtered  roi_month_wins_dedicated
England Premier League    bad_league               6              -0.077843               -0.045000               0.529412                0.695652 dedicated_over25_model                        2                         3
              Japan J1  other_league               5               0.658462                0.380000               0.923077                1.000000          filtered_ou25                        5                         0
         Europa League  other_league               6               0.578750                0.257778               0.906250                0.888889          filtered_ou25                        5                         1
Netherlands Eredivisie  other_league               6               0.374884                0.100952               0.813953                0.809524          filtered_ou25                        5                         1
         Portugal Liga  other_league               6               0.550417                0.093125               0.916667                0.812500          filtered_ou25                        6                         0
      Champions League  other_league               6               0.598000                0.062593               0.971429                0.777778          filtered_ou25                        6                         0
           Belgium Pro  other_league               6               0.301622                0.045000               0.783784                0.750000          filtered_ou25                        4                         2
  Scotland Premiership    bad_league               6               0.491667                0.040909               0.833333                0.772727          filtered_ou25                        3                         3
        England FA Cup    bad_league               5               0.227407               -0.047778               0.666667                0.722222          filtered_ou25                        3                         2
    Germany Bundesliga  other_league               6               0.613396               -0.061373               1.000000                0.686275          filtered_ou25                        6                         0
         Italy Serie A  other_league               6               0.560606               -0.066667               0.909091                0.666667          filtered_ou25                        6                         0
        France Ligue 1  other_league               6               0.392432               -0.113571               0.783784                0.642857          filtered_ou25                        4                         2
         Spain La Liga    bad_league               6               0.168824               -0.162105               0.647059                0.605263          filtered_ou25                        4                         2
     Europa Conference  other_league               4               0.311429               -0.191000               0.761905                0.600000          filtered_ou25                        3                         1
    Norway Eliteserien  other_league               4               0.629412               -0.249231               1.000000                0.538462          filtered_ou25                        4                         0
  England EFL League 1    bad_league               6               0.188077               -0.275000               0.634615                0.500000          filtered_ou25                        5                         1
               USA MLS  other_league               2               0.631579               -0.637500               1.000000                0.250000          filtered_ou25                        2                         0
  England Championship  other_league               6               0.606444                     NaN               0.888889                     NaN          filtered_ou25                        6                         0
        Brazil Serie A    bad_league               4               0.338667                     NaN               0.733333                     NaN          filtered_ou25                        4                         0

## Bad-league focus

                league  filtered_weighted_roi  dedicated_weighted_roi  filtered_worst_roi  dedicated_worst_roi  filtered_max_drawdown  dedicated_max_drawdown  winner_overall_by_roi
England Premier League              -0.077843               -0.045000           -1.000000              -1.0000               -1.00000               -1.000000 dedicated_over25_model
  Scotland Premiership               0.491667                0.040909           -0.070000              -0.3600                0.00000               -0.083333          filtered_ou25
        England FA Cup               0.227407               -0.047778           -0.620000              -0.3125               -0.67130               -0.472344          filtered_ou25
         Spain La Liga               0.168824               -0.162105           -0.075000              -0.5400               -0.08659               -0.696666          filtered_ou25
  England EFL League 1               0.188077               -0.275000           -0.212857              -1.0000                0.00000               -1.000000          filtered_ou25
        Brazil Serie A               0.338667                     NaN           -1.000000                  NaN               -1.00000                0.000000          filtered_ou25

## Month-by-month winners

  month                 league  filtered_rows  filtered_hit  filtered_roi  dedicated_rows  dedicated_hit  dedicated_roi          winner_by_roi
2024-11            Belgium Pro              7      0.857143      0.364286               3       1.000000       0.416667 dedicated_over25_model
2024-12            Belgium Pro              8      0.750000      0.240000               2       1.000000       0.405000 dedicated_over25_model
2025-01            Belgium Pro              4      0.750000      0.210000               2       0.500000      -0.350000          filtered_ou25
2025-02            Belgium Pro              7      0.714286      0.164286               1       0.000000      -1.000000          filtered_ou25
2025-03            Belgium Pro              5      0.800000      0.418000               0            NaN            NaN          filtered_ou25
2025-04            Belgium Pro              6      0.833333      0.435000               0            NaN            NaN          filtered_ou25
2024-11         Brazil Serie A              4      0.500000     -0.047500               0            NaN            NaN          filtered_ou25
2024-12         Brazil Serie A              5      1.000000      0.806000               0            NaN            NaN          filtered_ou25
2025-03         Brazil Serie A              1      0.000000     -1.000000               0            NaN            NaN          filtered_ou25
2025-04         Brazil Serie A              5      0.800000      0.448000               0            NaN            NaN          filtered_ou25
2024-11       Champions League             10      1.000000      0.608000               7       0.571429      -0.228571          filtered_ou25
2024-12       Champions League              5      1.000000      0.618000               4       1.000000       0.372500          filtered_ou25
2025-01       Champions League              9      0.888889      0.452222              11       0.909091       0.246364          filtered_ou25
2025-02       Champions League              4      1.000000      0.555000               4       0.750000       0.022500          filtered_ou25
2025-03       Champions League              4      1.000000      0.800000               1       0.000000      -1.000000          filtered_ou25
2025-04       Champions League              3      1.000000      0.756667               0            NaN            NaN          filtered_ou25
2024-11   England Championship              8      1.000000      0.767500               0            NaN            NaN          filtered_ou25
2024-12   England Championship             10      0.800000      0.460000               0            NaN            NaN          filtered_ou25
2025-01   England Championship              8      1.000000      0.836250               0            NaN            NaN          filtered_ou25
2025-02   England Championship              7      0.571429      0.025714               0            NaN            NaN          filtered_ou25
2025-03   England Championship              7      1.000000      0.855714               0            NaN            NaN          filtered_ou25
2025-04   England Championship              5      1.000000      0.738000               0            NaN            NaN          filtered_ou25
2024-11   England EFL League 1              7      0.428571     -0.212857               1       1.000000       0.450000 dedicated_over25_model
2024-12   England EFL League 1             12      0.583333      0.100833               1       0.000000      -1.000000          filtered_ou25
2025-01   England EFL League 1              8      0.875000      0.655000               0            NaN            NaN          filtered_ou25
2025-02   England EFL League 1              8      0.625000      0.143750               0            NaN            NaN          filtered_ou25
2025-03   England EFL League 1              7      0.571429      0.080000               0            NaN            NaN          filtered_ou25
2025-04   England EFL League 1             10      0.700000      0.311000               0            NaN            NaN          filtered_ou25
2024-11         England FA Cup             12      0.750000      0.402500               4       1.000000       0.397500          filtered_ou25
2024-12         England FA Cup              1      1.000000      0.730000               1       1.000000       0.370000          filtered_ou25
2025-01         England FA Cup              7      0.857143      0.564286               8       0.625000      -0.232500          filtered_ou25
2025-02         England FA Cup              5      0.200000     -0.620000               4       0.500000      -0.312500 dedicated_over25_model
2025-03         England FA Cup              2      0.500000     -0.135000               1       1.000000       0.290000 dedicated_over25_model
2024-11 England Premier League             10      0.400000     -0.299000               7       0.857143       0.177143 dedicated_over25_model
2024-12 England Premier League             17      0.705882      0.239412              17       0.647059      -0.108235          filtered_ou25
2025-01 England Premier League             12      0.500000     -0.154167              11       0.909091       0.257273 dedicated_over25_model
2025-02 England Premier League              6      0.666667      0.183333               5       0.400000      -0.470000          filtered_ou25
2025-03 England Premier League              3      0.333333     -0.433333               4       0.750000       0.012500 dedicated_over25_model
2025-04 England Premier League              3      0.000000     -1.000000               2       0.000000      -1.000000                    tie
2024-11      Europa Conference              7      0.571429     -0.035714               4       0.750000       0.010000 dedicated_over25_model
2024-12      Europa Conference              8      0.750000      0.263750               5       0.600000      -0.190000          filtered_ou25
2025-02      Europa Conference              4      1.000000      0.825000               1       0.000000      -1.000000          filtered_ou25
2025-03      Europa Conference              2      1.000000      0.690000               0            NaN            NaN          filtered_ou25
2024-11          Europa League              8      1.000000      0.676250               5       0.800000       0.148000          filtered_ou25
2024-12          Europa League              4      1.000000      0.832500               0            NaN            NaN          filtered_ou25
2025-01          Europa League             10      0.700000      0.194000               3       1.000000       0.386667 dedicated_over25_model
2025-02          Europa League              4      1.000000      0.817500               1       1.000000       0.420000          filtered_ou25
2025-03          Europa League              4      1.000000      0.757500               0            NaN            NaN          filtered_ou25
2025-04          Europa League              2      1.000000      0.770000               0            NaN            NaN          filtered_ou25
2024-11         France Ligue 1              6      1.000000      0.755000               5       0.400000      -0.430000          filtered_ou25
2024-12         France Ligue 1              5      0.600000      0.056000               3       0.666667      -0.096667          filtered_ou25
2025-01         France Ligue 1              7      0.857143      0.554286               3       0.333333      -0.593333          filtered_ou25
2025-02         France Ligue 1              7      0.714286      0.304286               6       1.000000       0.418333 dedicated_over25_model
2025-03         France Ligue 1              5      0.800000      0.322000               3       1.000000       0.346667 dedicated_over25_model
2025-04         France Ligue 1              7      0.714286      0.298571               8       0.500000      -0.313750          filtered_ou25
2024-11     Germany Bundesliga              9      1.000000      0.605556              14       0.571429      -0.205714          filtered_ou25
2024-12     Germany Bundesliga              8      1.000000      0.683750               5       0.600000      -0.186000          filtered_ou25
2025-01     Germany Bundesliga             10      1.000000      0.586000               8       0.750000       0.012500          filtered_ou25
2025-02     Germany Bundesliga              8      1.000000      0.523750              10       0.800000       0.072000          filtered_ou25
2025-03     Germany Bundesliga              9      1.000000      0.658889               5       0.600000      -0.160000          filtered_ou25
2025-04     Germany Bundesliga              9      1.000000      0.623333               9       0.777778       0.073333          filtered_ou25
2024-11          Italy Serie A              5      0.800000      0.342000               1       0.000000      -1.000000          filtered_ou25
2024-12          Italy Serie A              9      1.000000      0.733333               1       1.000000       0.360000          filtered_ou25
2025-01          Italy Serie A              8      0.875000      0.495000               1       1.000000       0.440000          filtered_ou25
2025-02          Italy Serie A              5      0.800000      0.390000               0            NaN            NaN          filtered_ou25
2025-03          Italy Serie A              3      1.000000      0.776667               0            NaN            NaN          filtered_ou25
2025-04          Italy Serie A              3      1.000000      0.650000               0            NaN            NaN          filtered_ou25
2024-11               Japan J1              5      1.000000      0.750000               1       1.000000       0.360000          filtered_ou25
2024-12               Japan J1              3      1.000000      0.803333               1       1.000000       0.400000          filtered_ou25
2025-02               Japan J1              1      1.000000      0.850000               0            NaN            NaN          filtered_ou25
2025-03               Japan J1              2      0.500000     -0.075000               0            NaN            NaN          filtered_ou25
2025-04               Japan J1              2      1.000000      0.850000               0            NaN            NaN          filtered_ou25
2024-11 Netherlands Eredivisie              7      1.000000      0.772857               3       0.666667      -0.073333          filtered_ou25
2024-12 Netherlands Eredivisie              7      0.857143      0.404286               3       1.000000       0.396667          filtered_ou25
2025-01 Netherlands Eredivisie              5      0.800000      0.328000               4       1.000000       0.310000          filtered_ou25
2025-02 Netherlands Eredivisie              8      0.750000      0.245000               5       0.600000      -0.166000          filtered_ou25
2025-03 Netherlands Eredivisie              8      0.750000      0.267500               4       0.750000       0.005000          filtered_ou25
2025-04 Netherlands Eredivisie              8      0.750000      0.267500               2       1.000000       0.360000 dedicated_over25_model
2024-11     Norway Eliteserien              6      1.000000      0.521667               6       0.500000      -0.286667          filtered_ou25
2024-12     Norway Eliteserien              2      1.000000      0.700000               3       0.333333      -0.573333          filtered_ou25
2025-03     Norway Eliteserien              2      1.000000      0.675000               1       0.000000      -1.000000          filtered_ou25
2025-04     Norway Eliteserien              7      1.000000      0.688571               3       1.000000       0.400000          filtered_ou25
2024-11          Portugal Liga              2      1.000000      0.510000               3       1.000000       0.366667          filtered_ou25
2024-12          Portugal Liga              5      0.800000      0.394000               5       0.600000      -0.198000          filtered_ou25
2025-01          Portugal Liga              3      1.000000      0.646667               2       0.500000      -0.265000          filtered_ou25
2025-02          Portugal Liga              5      0.800000      0.344000               3       1.000000       0.260000          filtered_ou25
2025-03          Portugal Liga              4      1.000000      0.695000               0            NaN            NaN          filtered_ou25
2025-04          Portugal Liga              5      1.000000      0.756000               3       1.000000       0.376667          filtered_ou25
2024-11   Scotland Premiership              2      0.500000     -0.070000               2       0.500000      -0.360000          filtered_ou25
2024-12   Scotland Premiership              2      1.000000      0.825000               4       1.000000       0.347500          filtered_ou25
2025-01   Scotland Premiership              2      1.000000      0.720000               6       0.666667      -0.083333          filtered_ou25
2025-02   Scotland Premiership              0           NaN           NaN               5       0.800000       0.058000 dedicated_over25_model
2025-03   Scotland Premiership              0           NaN           NaN               2       1.000000       0.330000 dedicated_over25_model
2025-04   Scotland Premiership              0           NaN           NaN               3       0.666667      -0.073333 dedicated_over25_model
2024-11          Spain La Liga              3      1.000000      0.850000               4       0.750000      -0.020000          filtered_ou25
2024-12          Spain La Liga              2      0.500000     -0.075000               5       0.600000      -0.162000          filtered_ou25
2025-01          Spain La Liga              0           NaN           NaN               6       0.833333       0.175000 dedicated_over25_model
2025-02          Spain La Liga              5      0.600000      0.034000              10       0.400000      -0.429000          filtered_ou25
2025-03          Spain La Liga              2      0.500000     -0.045000               7       0.857143       0.172857 dedicated_over25_model
2025-04          Spain La Liga              5      0.600000      0.078000               6       0.333333      -0.540000          filtered_ou25
2025-02                USA MLS              4      1.000000      0.672500               2       0.000000      -1.000000          filtered_ou25
2025-03                USA MLS             15      1.000000      0.620667               2       0.500000      -0.275000          filtered_ou25
