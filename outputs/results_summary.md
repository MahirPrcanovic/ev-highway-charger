# Simulation Findings

Warm-up method: welch.

Warm-up period excluded from KPI analysis: 0.0 minutes.

## Queue Metrics by Scenario

|   chargers |   avg_wait_min |   avg_max_wait_min |   avg_p95_wait_min |   avg_served_vehicles |   std_wait_min |   replications |   wait_ci95_halfwidth |   wait_ci95_low |   wait_ci95_high |   avg_total_charge_time_min |   avg_utilization |
|-----------:|---------------:|-------------------:|-------------------:|----------------------:|---------------:|---------------:|----------------------:|----------------:|-----------------:|----------------------------:|------------------:|
|          2 |        61.5537 |           125.702  |           120.753  |                 15.25 |       14.9931  |              4 |               23.8574 |        37.6963  |          85.4111 |                     419.293 |          0.873527 |
|          4 |        10.9598 |            28.9032 |            26.0256 |                 29.5  |        6.19869 |              4 |                9.8635 |         1.09626 |          20.8233 |                     790.943 |          0.823898 |

## Cost-Benefit What-If

|   chargers |   avg_wait_min |   avg_max_wait_min |   avg_p95_wait_min |   avg_served_vehicles |   std_wait_min |   replications |   wait_ci95_halfwidth |   wait_ci95_low |   wait_ci95_high |   avg_total_charge_time_min |   avg_utilization |   added_chargers |   delta_avg_wait_min |   delta_avg_max_wait_min |   relative_wait_reduction_pct |   incremental_cost_eur |   minutes_saved_per_1k_eur |
|-----------:|---------------:|-------------------:|-------------------:|----------------------:|---------------:|---------------:|----------------------:|----------------:|-----------------:|----------------------------:|------------------:|-----------------:|---------------------:|-------------------------:|------------------------------:|-----------------------:|---------------------------:|
|          2 |        61.5537 |           125.702  |           120.753  |                 15.25 |       14.9931  |              4 |               23.8574 |        37.6963  |          85.4111 |                     419.293 |          0.873527 |                0 |               0      |                   0      |                        0      |                      0 |                   0        |
|          4 |        10.9598 |            28.9032 |            26.0256 |                 29.5  |        6.19869 |              4 |                9.8635 |         1.09626 |          20.8233 |                     790.943 |          0.823898 |                2 |              50.5939 |                  96.7987 |                       82.1948 |                  84000 |                   0.602309 |

## Metamodel Quality

| model             | metric   |    value |
|:------------------|:---------|---------:|
| linear_regression | R2       | -2.17268 |
| linear_regression | MAE      | 18.1252  |
| random_forest     | R2       | -1.93264 |
| random_forest     | MAE      | 17.3632  |