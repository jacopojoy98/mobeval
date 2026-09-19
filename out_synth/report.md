# Synthetic smoke run

## Setup
```
dataset=synthetic split_by=time grid=69x50 cells @ 500 m
  train: points=  59976 windows=  1429 staypoints=   740 visit_seqs=   560
  val  : points=   6512 windows=   148 staypoints=    82 visit_seqs=    82
  test : points=   7381 windows=   180 staypoints=    59 visit_seqs=    59
eval_seeds=[0, 1]  n_boot=200
```

Cells: mean over eval seeds / run tags (± std across them). Brackets: skill vs. the task's primary baseline on identical samples ([+x%] = share of baseline error removed or gap to perfect closed; [Δ] = nats/sample better than baseline). 95% bootstrap CIs are in results.jsonl.

## Summary: median headline skill per task family (%, clipped to ±100)

| model        |   classification |   continuous |   generation |   location |   recovery |
|:-------------|-----------------:|-------------:|-------------:|-----------:|-----------:|
| KinematicRef |              6.4 |         -0.7 |         97.1 |       -4   |       20.3 |
| WeakRef      |             18.4 |        -74.1 |        nan   |        1.6 |     -100   |

## Efficiency (Pareto front)

| model        |   mean_skill |   n_parameters |   latency_ms | pareto_optimal   |
|:-------------|-------------:|---------------:|-------------:|:-----------------|
| KinematicRef |     0.238219 |            352 |   0.604422   | True             |
| WeakRef      |    -0.385147 |              5 |   0.00382343 | True             |

## Recovery

|                                                            | KinematicRef          | WeakRef                | baseline:last_observed   | baseline:linear_interp   |
|:-----------------------------------------------------------|:----------------------|:-----------------------|:-------------------------|:-------------------------|
| ('recovery/block@0.25 · acc_100m ↑ (fraction)', 'native')  | 0.97 ±0.0034 [+56%]   | 0.203 ±0.0015 [-1102%] | 0.225 ±0.013             | 0.934 ±0.0015            |
| ('recovery/block@0.25 · acc_500m ↑ (fraction)', 'native')  | 0.995 ±0.00098 [+41%] | 0.987 ±0.0054 [-51%]   | 0.562 ±0.0069            | 0.992 ±0.002             |
| ('recovery/block@0.25 · ade_m ↓ (m)', 'native')            | 32.4 ±2 [+21%]        | 191 ±4 [-369%]         | 538 ±6                   | 40.9 ±2.5                |
| ('recovery/block@0.25 · dtw_m ↓ (m)', 'native')            | 26.9 ±0.74 [+22%]     | 177 ±2.6 [-411%]       | 538 ±6                   | 34.6 ±0.94               |
| ('recovery/block@0.25 · fde_m ↓ (m)', 'native')            | 22.6 ±1.8 [+20%]      | 193 ±6.4 [-583%]       | 943 ±10                  | 28.4 ±3.6                |
| ('recovery/block@0.25 · grid_acc ↑ (fraction)', 'native')  | 0.92 ±0.0044 [+9%]    | 0.575 ±0.018 [-388%]   | 0.322 ±0.0059            | 0.913 ±0.0054            |
| ('recovery/block@0.25 · median_ade_m ↓ (m)', 'native')     | 22.2 ±0.73 [-7%]      | 183 ±1.9 [-785%]       | 505 ±11                  | 20.7 ±1.2                |
| ('recovery/block@0.25 · p90_ade_m ↓ (m)', 'native')        | 39.9 ±0.49 [+43%]     | 233 ±3.6 [-235%]       | 1,035 ±60                | 69.7 ±1.7                |
| ('recovery/block@0.25 · rmse_m ↓ (m)', 'native')           | 67.5 ±7.1 [+24%]      | 220 ±5.6 [-150%]       | 744 ±0.28                | 88.3 ±7.6                |
| ('recovery/block@0.5 · acc_100m ↑ (fraction)', 'native')   | 0.865 ±0.014 [+49%]   | 0.169 ±0.0049 [-216%]  | 0.19 ±0.002              | 0.737 ±0.011             |
| ('recovery/block@0.5 · acc_500m ↑ (fraction)', 'native')   | 0.959 ±0.0034 [+36%]  | 0.923 ±0.013 [-19%]    | 0.367 ±0.002             | 0.935 ±0.01              |
| ('recovery/block@0.5 · ade_m ↓ (m)', 'native')             | 96.8 ±4.6 [+34%]      | 269 ±8 [-83%]          | 1,045 ±1.5               | 147 ±7.6                 |
| ('recovery/block@0.5 · dtw_m ↓ (m)', 'native')             | 49.3 ±1.3 [+34%]      | 189 ±1.9 [-152%]       | 1,045 ±1.5               | 75 ±2.4                  |
| ('recovery/block@0.5 · fde_m ↓ (m)', 'native')             | 36.2 ±4.4 [+44%]      | 218 ±6.4 [-236%]       | 1,898 ±6.2               | 65.1 ±6.3                |
| ('recovery/block@0.5 · grid_acc ↑ (fraction)', 'native')   | 0.84 ±0.012 [+29%]    | 0.522 ±0.013 [-111%]   | 0.234 ±0.0088            | 0.774 ±0.01              |
| ('recovery/block@0.5 · median_ade_m ↓ (m)', 'native')      | 34.9 ±1.8 [+21%]      | 196 ±0.44 [-341%]      | 957 ±11                  | 44.4 ±0.052              |
| ('recovery/block@0.5 · p90_ade_m ↓ (m)', 'native')         | 136 ±19 [+57%]        | 360 ±28 [-14%]         | 2,060 ±76                | 321 ±63                  |
| ('recovery/block@0.5 · rmse_m ↓ (m)', 'native')            | 273 ±1.2 [+24%]       | 412 ±5.5 [-14%]        | 1,448 ±2.5               | 360 ±1.4                 |
| ('recovery/random@0.25 · acc_100m ↑ (fraction)', 'native') | 0.994 ±0.0029 [+38%]  | 0.199 ±0.01 [-9495%]   | 0.364 ±0.012             | 0.991 ±0.0025            |
| ('recovery/random@0.25 · acc_500m ↑ (fraction)', 'native') | 1 ±0.00049 [+50%]     | 0.998 ±0.00049 [-100%] | 0.96 ±0.0059             | 0.999 ±0.00098           |
| ('recovery/random@0.25 · ade_m ↓ (m)', 'native')           | 17.4 ±0.65 [-2%]      | 180 ±0.92 [-955%]      | 167 ±4                   | 17.1 ±0.81               |
| ('recovery/random@0.25 · dtw_m ↓ (m)', 'native')           | 16.4 ±0.18 [-3%]      | 176 ±0.12 [-999%]      | 158 ±3                   | 16 ±0.53                 |
| ('recovery/random@0.25 · fde_m ↓ (m)', 'native')           | 17 ±0.53 [-5%]        | 181 ±2.8 [-1014%]      | 173 ±3.6                 | 16.3 ±0.46               |
| ('recovery/random@0.25 · grid_acc ↑ (fraction)', 'native') | 0.959 ±0.00049 [-3%]  | 0.61 ±0.0034 [-868%]   | 0.649 ±0.0098            | 0.96                     |
| ('recovery/random@0.25 · median_ade_m ↓ (m)', 'native')    | 15.3 ±0.39 [-10%]     | 178 ±1.7 [-1178%]      | 143 ±0.41                | 13.9 ±0.25               |
| ('recovery/random@0.25 · p90_ade_m ↓ (m)', 'native')       | 24.7 ±1.7 [-1%]       | 221 ±3 [-809%]         | 334 ±17                  | 24.4 ±0.92               |
| ('recovery/random@0.25 · rmse_m ↓ (m)', 'native')          | 24.8 ±5.6 [+13%]      | 202 ±0.87 [-630%]      | 236 ±11                  | 28.4 ±6.3                |
| ('recovery/random@0.5 · acc_100m ↑ (fraction)', 'native')  | 0.987 ±0.0015 [+11%]  | 0.203 ±0.0022 [-5341%] | 0.314 ±0.0083            | 0.985 ±0.0029            |
| ('recovery/random@0.5 · acc_500m ↑ (fraction)', 'native')  | 0.999 ±0.00098 [+42%] | 0.996 ±0.00098 [-85%]  | 0.887 ±0.0039            | 0.998 ±0.0012            |
| ('recovery/random@0.5 · ade_m ↓ (m)', 'native')            | 20.6 ±1 [+1%]         | 182 ±1.8 [-771%]       | 237 ±5.8                 | 20.9 ±1.4                |
| ('recovery/random@0.5 · dtw_m ↓ (m)', 'native')            | 17.7 ±0.52 [+0%]      | 170 ±0.0096 [-854%]    | 185 ±1.8                 | 17.8 ±0.66               |
| ('recovery/random@0.5 · fde_m ↓ (m)', 'native')            | 18.2 ±0.062 [-2%]     | 181 ±1.3 [-920%]       | 250 ±8.1                 | 17.8 ±0.18               |
| ('recovery/random@0.5 · grid_acc ↑ (fraction)', 'native')  | 0.955 ±0.0015 [+3%]   | 0.598 ±0.011 [-770%]   | 0.562 ±0.02              | 0.954 ±0.0025            |
| ('recovery/random@0.5 · median_ade_m ↓ (m)', 'native')     | 16 ±0.056 [-7%]       | 179 ±0.81 [-1100%]     | 215 ±2.9                 | 14.9 ±0.41               |
| ('recovery/random@0.5 · p90_ade_m ↓ (m)', 'native')        | 27.7 ±0.43 [+3%]      | 211 ±6.7 [-637%]       | 447 ±25                  | 28.7 ±0.75               |
| ('recovery/random@0.5 · rmse_m ↓ (m)', 'native')           | 41.2 ±6.5 [+11%]      | 205 ±2.2 [-350%]       | 341 ±9.4                 | 46.2 ±7.4                |

## Location

|                                                       | KinematicRef   | WeakRef     |   baseline:global_popular |   baseline:markov1 |   baseline:user_frequent |
|:------------------------------------------------------|:---------------|:------------|--------------------------:|-------------------:|-------------------------:|
| ('next_location · acc@1 ↑ (fraction)', 'native')      | 0.119 [-4%]    | 0.136 [-2%] |                    0.0339 |              0.153 |                    0.102 |
| ('next_location · acc@5 ↑ (fraction)', 'native')      | 0.593 [-33%]   | n/a         |                    0.0678 |              0.695 |                    0.763 |
| ('next_location · acc_1km ↑ (fraction)', 'native')    | 0.153 [-2%]    | 0.136 [-4%] |                    0.0508 |              0.169 |                    0.119 |
| ('next_location · dist_err_m ↓ (m)', 'native')        | 7,851 [+4%]    | 7,733 [+5%] |                 6496      |           8166     |                 7989     |
| ('next_location · loc_nll ↓ (nats)', 'native')        | 5.24 [Δ-0.93]  | n/a         |                    5.22   |              4.31  |                    4.5   |
| ('next_location · median_dist_err_m ↓ (m)', 'native') | 7,983 [+9%]    | 8,085 [+8%] |                 5966      |           8806     |                 7878     |
| ('next_location · mrr@20 ↑ (fraction)', 'native')     | 0.282 [-13%]   | n/a         |                    0.052  |              0.363 |                    0.358 |

## Continuous

|                                                                | KinematicRef   | WeakRef      | baseline:train_marginal   |
|:---------------------------------------------------------------|:---------------|:-------------|:--------------------------|
| ('continuous/duration · coverage80 ↑ (fraction)', 'native')    | 0.864          | n/a          | 0.881                     |
| ('continuous/duration · crps_min ↓ (min)', 'native')           | 159 [-5%]      | 121 [+20%]   | 151                       |
| ('continuous/duration · mae_min ↓ (min)', 'native')            | 152 [-86%]     | 121 [-48%]   | 81.6                      |
| ('continuous/duration · nll ↓ (nats)', 'native')               | 6.38 [Δ-0.18]  | n/a          | 6.2                       |
| ('continuous/duration · pit_ks ↓ (stat)', 'native')            | 0.428          | n/a          | 0.393                     |
| ('continuous/duration · pseudo_nll ↓ (nats)', 'native')        | n/a            | 7.15         | n/a                       |
| ('continuous/duration · rmse_min ↓ (min)', 'native')           | 180 [-73%]     | 149 [-44%]   | 104                       |
| ('continuous/travel_time · coverage80 ↑ (fraction)', 'native') | 0.864          | n/a          | 0.644                     |
| ('continuous/travel_time · crps_min ↓ (min)', 'native')        | 5.36 [+4%]     | 18.1 [-225%] | 5.58                      |
| ('continuous/travel_time · mae_min ↓ (min)', 'native')         | 7.37 [+8%]     | 18.1 [-125%] | 8.05                      |
| ('continuous/travel_time · nll ↓ (nats)', 'native')            | 3.36 [Δ+0.04]  | n/a          | 3.39                      |
| ('continuous/travel_time · pit_ks ↓ (stat)', 'native')         | 0.143          | n/a          | 0.18                      |
| ('continuous/travel_time · pseudo_nll ↓ (nats)', 'native')     | n/a            | 5.41         | n/a                       |
| ('continuous/travel_time · rmse_min ↓ (min)', 'native')        | 10.6 [-1%]     | 19.6 [-86%]  | 10.5                      |

## Classification

|                                                                            | KinematicRef           | WeakRef        | baseline:handcrafted_gbdt   |   baseline:majority |
|:---------------------------------------------------------------------------|:-----------------------|:---------------|:----------------------------|--------------------:|
| ('mode/linear_probe@0.2 · accuracy ↑ (fraction)', 'linear_probe')          | 0.746 ±0.011 [+7%]     | n/a            | 0.724 ±0.032                |              0.59   |
| ('mode/linear_probe@0.2 · balanced_accuracy ↑ (fraction)', 'linear_probe') | 0.626 ±0.062 [+13%]    | n/a            | 0.569 ±0.0092               |              0.25   |
| ('mode/linear_probe@0.2 · cls_nll ↓ (nats)', 'linear_probe')               | 0.629 ±0.0068 [Δ+1.18] | n/a            | 1.8 ±0.0081                 |              0.946  |
| ('mode/linear_probe@0.2 · ece ↓ (fraction)', 'linear_probe')               | 0.087 ±0.0092 [+62%]   | n/a            | 0.233 ±0.044                |              0.0632 |
| ('mode/linear_probe@0.2 · macro_f1 ↑ (fraction)', 'linear_probe')          | 0.637 ±0.088 [+18%]    | n/a            | 0.56 ±0.03                  |              0.185  |
| ('mode/linear_probe@1 · accuracy ↑ (fraction)', 'linear_probe')            | 0.739 [-17%]           | n/a            | 0.776                       |              0.59   |
| ('mode/linear_probe@1 · balanced_accuracy ↑ (fraction)', 'linear_probe')   | 0.584 [-0%]            | n/a            | 0.586                       |              0.25   |
| ('mode/linear_probe@1 · cls_nll ↓ (nats)', 'linear_probe')                 | 0.59 [Δ+0.77]          | n/a            | 1.36                        |              0.942  |
| ('mode/linear_probe@1 · ece ↓ (fraction)', 'linear_probe')                 | 0.0664 [+64%]          | n/a            | 0.182                       |              0.0612 |
| ('mode/linear_probe@1 · macro_f1 ↑ (fraction)', 'linear_probe')            | 0.558 [-8%]            | n/a            | 0.592                       |              0.185  |
| ('mode/native@0.2 · accuracy ↑ (fraction)', 'native')                      | n/a                    | 0.619 [-39%]   | 0.724 ±0.032                |              0.59   |
| ('mode/native@0.2 · balanced_accuracy ↑ (fraction)', 'native')             | n/a                    | 0.769 [+46%]   | 0.569 ±0.0092               |              0.25   |
| ('mode/native@0.2 · cls_nll ↓ (nats)', 'native')                           | n/a                    | 0.906 [Δ+0.90] | 1.8 ±0.0081                 |              0.946  |
| ('mode/native@0.2 · ece ↓ (fraction)', 'native')                           | n/a                    | 0.114 [+50%]   | 0.233 ±0.044                |              0.0632 |
| ('mode/native@0.2 · macro_f1 ↑ (fraction)', 'native')                      | n/a                    | 0.529 [-7%]    | 0.56 ±0.03                  |              0.185  |
| ('mode/native@1 · accuracy ↑ (fraction)', 'native')                        | n/a                    | 0.619 [-70%]   | 0.776                       |              0.59   |
| ('mode/native@1 · balanced_accuracy ↑ (fraction)', 'native')               | n/a                    | 0.769 [+44%]   | 0.586                       |              0.25   |
| ('mode/native@1 · cls_nll ↓ (nats)', 'native')                             | n/a                    | 0.906 [Δ+0.45] | 1.36                        |              0.942  |
| ('mode/native@1 · ece ↓ (fraction)', 'native')                             | n/a                    | 0.114 [+38%]   | 0.182                       |              0.0612 |
| ('mode/native@1 · macro_f1 ↑ (fraction)', 'native')                        | n/a                    | 0.529 [-15%]   | 0.592                       |              0.185  |

## Generation

|                                                                                          | KinematicRef         | baseline:real_noise_floor   | baseline:uniform_bbox   |
|:-----------------------------------------------------------------------------------------|:---------------------|:----------------------------|:------------------------|
| ('generation/daily_locations · jsd:daily_locations ↓ (bits)', 'native')                  | 0.0171 ±0.0044       | 0.0856                      | 0.0593                  |
| ('generation/daily_locations · w1:daily_locations ↓ (native)', 'native')                 | 0.19 ±0.013 [+429%]  | 0.4                         | 0.467                   |
| ('generation/jump_length · jsd:jump_length ↓ (bits)', 'native')                          | 0.104 ±0.027 [+127%] | 0.118                       | 0.255                   |
| ('generation/jump_length · w1:jump_length ↓ (native)', 'native')                         | 1,111 ±7e+02 [+97%]  | 870                         | 5,972                   |
| ('generation/memorisation · copy_rate ↓ (fraction)', 'native')                           | 1                    | n/a                         | n/a                     |
| ('generation/memorisation · nn_train_dist_m ↑ (m)', 'native')                            | 20.7 ±0.29           | n/a                         | n/a                     |
| ('generation/radius_of_gyration · jsd:radius_of_gyration ↓ (bits)', 'native')            | 0.296 ±0.037 [+121%] | 0.355                       | 0.634                   |
| ('generation/radius_of_gyration · spearman_paired:radius_of_gyration ↑ (rho)', 'native') | 0.684 ±0.062         | n/a                         | n/a                     |
| ('generation/radius_of_gyration · w1:radius_of_gyration ↓ (native)', 'native')           | 434 ±1.4e+02 [+104%] | 577                         | 5,085                   |
| ('generation/stay_duration · jsd:stay_duration ↓ (bits)', 'native')                      | 0.0951 ±0.02 [+113%] | 0.102                       | 0.164                   |
| ('generation/stay_duration · w1:stay_duration ↓ (native)', 'native')                     | 12.2 ±1.2 [+76%]     | 7.44                        | 25.4                    |
| ('generation/visited_cells · jsd:visited_cells ↓ (bits)', 'native')                      | 0.437 ±0.077 [+84%]  | n/a                         | n/a                     |

## Efficiency

|                                                                                | KinematicRef   | WeakRef   |
|:-------------------------------------------------------------------------------|:---------------|:----------|
| ('efficiency · n_parameters ↓ (count)', 'native')                              | 352            | 5         |
| ('efficiency/continuous/duration · latency_ms_per_sample ↓ (ms)', 'native')    | 0.00585        | 0.000291  |
| ('efficiency/continuous/travel_time · latency_ms_per_sample ↓ (ms)', 'native') | 0.00171        | 0.000391  |
| ('efficiency/embedding · latency_ms_per_sample ↓ (ms)', 'native')              | 0.0997         | n/a       |
| ('efficiency/generation · latency_ms_per_sample ↓ (ms)', 'native')             | 3.22           | n/a       |
| ('efficiency/mode_classification · latency_ms_per_sample ↓ (ms)', 'native')    | n/a            | 0.00796   |
| ('efficiency/next_location · latency_ms_per_sample ↓ (ms)', 'native')          | 0.0274         | 0.000504  |
| ('efficiency/recovery · latency_ms_per_sample ↓ (ms)', 'native')               | 0.267          | 0.00997   |

## Skipped / failed

- WeakRef@default · generation: skipped (capability not declared)
