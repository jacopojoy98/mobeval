# mobeval smoke test

## Setup
```
dataset=synthetic split_by=time grid=58x57 cells @ 500 m
  train: points=  31189 windows=   745 staypoints=   390 visit_seqs=   310
  val  : points=   4290 windows=   103 staypoints=    49 visit_seqs=    49
  test : points=   4034 windows=    96 staypoints=    34 visit_seqs=    34
eval_seeds=[0]  n_boot=50
```

Cells: mean over eval seeds / run tags (± std across them). Brackets: skill vs. the task's primary baseline on identical samples ([+x%] = share of baseline error removed or gap to perfect closed; [Δ] = nats/sample better than baseline). 95% bootstrap CIs are in results.jsonl.

## Summary: median headline skill per task family (%, clipped to ±100)

| model        |   classification |   continuous |   generation |   location |   recovery |
|:-------------|-----------------:|-------------:|-------------:|-----------:|-----------:|
| CLIP-tiny    |            -21.8 |        -44.3 |        nan   |      -36.1 |     -100   |
| KinematicRef |             -1.4 |         -2.8 |        100   |      -19.4 |       39.4 |
| TrajGPT-tiny |            nan   |         -0.6 |         39.7 |      -72.8 |      nan   |
| UniTraj-tiny |            -44.5 |        nan   |        nan   |      nan   |     -100   |

## Efficiency (Pareto front)

| model        |   mean_skill |    n_parameters |   latency_ms | pareto_optimal   |
|:-------------|-------------:|----------------:|-------------:|:-----------------|
| KinematicRef |     0.231603 |   352           |    0.743595  | True             |
| TrajGPT-tiny |    -0.112671 |     1.27744e+06 |   13.6058    | False            |
| CLIP-tiny    |    -0.505436 | 21194           |    1.48488   | False            |
| UniTraj-tiny |    -0.722373 | 25602           |    0.0877811 | True             |

## Recovery

|                                                          | CLIP-tiny       | KinematicRef   | UniTraj-tiny    |   baseline:last_observed |   baseline:linear_interp |
|:---------------------------------------------------------|:----------------|:---------------|:----------------|-------------------------:|-------------------------:|
| ('recovery/block@0.5 · acc_100m ↑ (fraction)', 'native') | 0.0026 [-225%]  | 0.831 [+45%]   | 0.00326 [-225%] |                    0.124 |                    0.693 |
| ('recovery/block@0.5 · acc_500m ↑ (fraction)', 'native') | 0.0762 [-1134%] | 0.947 [+29%]   | 0.109 [-1090%]  |                    0.334 |                    0.925 |
| ('recovery/block@0.5 · ade_m ↓ (m)', 'native')           | 1,855 [-1013%]  | 113 [+32%]     | 3,204 [-1823%]  |                 1084     |                  167     |
| ('recovery/block@0.5 · dtw_m ↓ (m)', 'native')           | 1,814 [-2283%]  | 50.5 [+34%]    | 3,204 [-4109%]  |                 1084     |                   76.1   |
| ('recovery/block@0.5 · fde_m ↓ (m)', 'native')           | 2,641 [-2809%]  | 47.6 [+48%]    | 3,622 [-3889%]  |                 1973     |                   90.8   |
| ('recovery/block@0.5 · grid_acc ↑ (fraction)', 'native') | 0.0286 [-294%]  | 0.824 [+28%]   | 0.00195 [-304%] |                    0.195 |                    0.753 |
| ('recovery/block@0.5 · median_ade_m ↓ (m)', 'native')    | 1,639 [-2843%]  | 41.5 [+25%]    | 2,902 [-5110%]  |                 1007     |                   55.7   |
| ('recovery/block@0.5 · p90_ade_m ↓ (m)', 'native')       | 3,011 [-713%]   | 261 [+30%]     | 5,644 [-1423%]  |                 1832     |                  371     |
| ('recovery/block@0.5 · rmse_m ↓ (m)', 'native')          | 2,270 [-502%]   | 283 [+25%]     | 3,837 [-918%]   |                 1469     |                  377     |

## Location

|                                                       | CLIP-tiny      | KinematicRef   | TrajGPT-tiny   |   baseline:global_popular |   baseline:markov1 |   baseline:user_frequent |
|:------------------------------------------------------|:---------------|:---------------|:---------------|--------------------------:|-------------------:|-------------------------:|
| ('next_location · acc@1 ↑ (fraction)', 'native')      | 0 [-31%]       | 0.0882 [-19%]  | 0 [-31%]       |                   0       |              0.235 |                    0.118 |
| ('next_location · acc@5 ↑ (fraction)', 'native')      | 0.0294 [-136%] | 0.441 [-36%]   | 0.176 [-100%]  |                   0       |              0.588 |                    0.618 |
| ('next_location · acc_1km ↑ (fraction)', 'native')    | 0 [-31%]       | 0.147 [-12%]   | 0.0294 [-27%]  |                   0.0294  |              0.235 |                    0.118 |
| ('next_location · dist_err_m ↓ (m)', 'native')        | 8,603 [-36%]   | 7,543 [-19%]   | 10,922 [-73%]  |                6740       |           6320     |                 7126     |
| ('next_location · loc_nll ↓ (nats)', 'native')        | 7.89 [Δ-2.45]  | 6.14 [Δ-0.69]  | 16.5 [Δ-11.05] |                   5.38    |              5.45  |                    6.14  |
| ('next_location · median_dist_err_m ↓ (m)', 'native') | 7,838 [-42%]   | 6,135 [-11%]   | 10,035 [-82%]  |                5772       |           5507     |                 5525     |
| ('next_location · mrr@20 ↑ (fraction)', 'native')     | 0.0257 [-52%]  | 0.25 [-17%]    | 0.0679 [-46%]  |                   0.00577 |              0.36  |                    0.316 |

## Continuous

|                                                                     | CLIP-tiny     | KinematicRef   | TrajGPT-tiny   |   baseline:train_marginal |
|:--------------------------------------------------------------------|:--------------|:---------------|:---------------|--------------------------:|
| ('continuous/duration · coverage80 ↑ (fraction)', 'native')         | 0.735         | 0.882          | 0.735          |                     0.735 |
| ('continuous/duration · crps_min ↓ (min)', 'native')                | 52.2 [+66%]   | 174 [-12%]     | 153 [+2%]      |                   156     |
| ('continuous/duration · mae_min ↓ (min)', 'native')                 | 73.7 [+11%]   | 164 [-97%]     | 93.8 [-13%]    |                    83.2   |
| ('continuous/duration · nll ↓ (nats)', 'native')                    | 5.93 [Δ+0.10] | 6.34 [Δ-0.31]  | 6.19 [Δ-0.16]  |                     6.03  |
| ('continuous/duration · pit_ks ↓ (stat)', 'native')                 | 0.332         | 0.399          | 0.44           |                     0.419 |
| ('continuous/duration · rmse_min ↓ (min)', 'native')                | 94.4 [+8%]    | 205 [-101%]    | 113 [-10%]     |                   102     |
| ('continuous/travel_time|<=4h · coverage80 ↑ (fraction)', 'native') | 0.294         | 0.853          | 0.853          |                     0.765 |
| ('continuous/travel_time|<=4h · crps_min ↓ (min)', 'native')        | 23.1 [-345%]  | 4.88 [+6%]     | 5.32 [-2%]     |                     5.2   |
| ('continuous/travel_time|<=4h · mae_min ↓ (min)', 'native')         | 31.2 [-302%]  | 7.02 [+10%]    | 7.69 [+1%]     |                     7.76  |
| ('continuous/travel_time|<=4h · nll ↓ (nats)', 'native')            | 4.89 [Δ-1.48] | 3.4 [Δ+0.00]   | 3.42 [Δ-0.02]  |                     3.4   |
| ('continuous/travel_time|<=4h · pit_ks ↓ (stat)', 'native')         | 0.648         | 0.103          | 0.126          |                     0.149 |
| ('continuous/travel_time|<=4h · rmse_min ↓ (min)', 'native')        | 33 [-238%]    | 9.65 [+1%]     | 9.67 [+1%]     |                     9.77  |

## Classification

|                                                                          | CLIP-tiny      | KinematicRef   | UniTraj-tiny   |   baseline:handcrafted_gbdt |   baseline:majority |
|:-------------------------------------------------------------------------|:---------------|:---------------|:---------------|----------------------------:|--------------------:|
| ('mode/linear_probe@1 · accuracy ↑ (fraction)', 'linear_probe')          | 0.662 [-39%]   | 0.743 [-6%]    | 0.595 [-67%]   |                       0.757 |              0.541  |
| ('mode/linear_probe@1 · balanced_accuracy ↑ (fraction)', 'linear_probe') | 0.383 [-17%]   | 0.466 [-2%]    | 0.27 [-39%]    |                       0.474 |              0.2    |
| ('mode/linear_probe@1 · cls_nll ↓ (nats)', 'linear_probe')               | 0.859 [Δ+0.85] | 0.59 [Δ+1.12]  | 0.961 [Δ+0.75] |                       1.71  |              1.1    |
| ('mode/linear_probe@1 · ece ↓ (fraction)', 'linear_probe')               | 0.131 [+43%]   | 0.08 [+65%]    | 0.136 [+41%]   |                       0.229 |              0.0634 |
| ('mode/linear_probe@1 · macro_f1 ↑ (fraction)', 'linear_probe')          | 0.388 [-15%]   | 0.459 [-1%]    | 0.263 [-38%]   |                       0.466 |              0.14   |
| ('mode/native@1 · accuracy ↑ (fraction)', 'native')                      | 0.324 [-178%]  | n/a            | 0.486 [-111%]  |                       0.757 |              0.541  |
| ('mode/native@1 · balanced_accuracy ↑ (fraction)', 'native')             | 0.336 [-26%]   | n/a            | 0.21 [-50%]    |                       0.474 |              0.2    |
| ('mode/native@1 · cls_nll ↓ (nats)', 'native')                           | 1.5 [Δ+0.21]   | n/a            | 1.31 [Δ+0.40]  |                       1.71  |              1.1    |
| ('mode/native@1 · ece ↓ (fraction)', 'native')                           | 0.0564 [+75%]  | n/a            | 0.159 [+30%]   |                       0.229 |              0.0634 |
| ('mode/native@1 · macro_f1 ↑ (fraction)', 'native')                      | 0.187 [-52%]   | n/a            | 0.198 [-50%]   |                       0.466 |              0.14   |

## Generation

|                                                                                          | KinematicRef   | TrajGPT-tiny   | baseline:real_noise_floor   | baseline:uniform_bbox   |
|:-----------------------------------------------------------------------------------------|:---------------|:---------------|:----------------------------|:------------------------|
| ('generation/daily_locations · jsd:daily_locations ↓ (bits)', 'native')                  | 0.0138         | 0.244          | 0.0292                      | 0                       |
| ('generation/daily_locations · w1:daily_locations ↓ (native)', 'native')                 | 0.125          | 0.724          | 0.25                        | 0                       |
| ('generation/jump_length · jsd:jump_length ↓ (bits)', 'native')                          | 0.129 [+208%]  | 0.15 [+188%]   | 0.241                       | 0.344                   |
| ('generation/jump_length · w1:jump_length ↓ (native)', 'native')                         | 1,251 [+103%]  | 1,168 [+104%]  | 1,425                       | 7,629                   |
| ('generation/memorisation · copy_rate ↓ (fraction)', 'native')                           | 1              | 0              | n/a                         | n/a                     |
| ('generation/memorisation · nn_train_dist_m ↑ (m)', 'native')                            | 21.3           | 5,063          | n/a                         | n/a                     |
| ('generation/radius_of_gyration · jsd:radius_of_gyration ↓ (bits)', 'native')            | 0.426 [+85%]   | 0.786 [-22%]   | 0.375                       | 0.711                   |
| ('generation/radius_of_gyration · spearman_paired:radius_of_gyration ↑ (rho)', 'native') | 0.667          | 0.533          | n/a                         | n/a                     |
| ('generation/radius_of_gyration · w1:radius_of_gyration ↓ (native)', 'native')           | 464 [+100%]    | 3,227 [+37%]   | 476                         | 4,844                   |
| ('generation/stay_duration · jsd:stay_duration ↓ (bits)', 'native')                      | 0.269 [+106%]  | 0.393 [+34%]   | 0.279                       | 0.452                   |
| ('generation/stay_duration · w1:stay_duration ↓ (native)', 'native')                     | 20.2 [+76%]    | 341 [-955%]    | 12.6                        | 43.8                    |
| ('generation/visited_cells · jsd:visited_cells ↓ (bits)', 'native')                      | 0.476 [+107%]  | 0.778 [+45%]   | n/a                         | n/a                     |

## Efficiency

|                                                                                | CLIP-tiny   | KinematicRef   | TrajGPT-tiny   | UniTraj-tiny   |
|:-------------------------------------------------------------------------------|:------------|:---------------|:---------------|:---------------|
| ('efficiency · n_parameters ↓ (count)', 'native')                              | 21,194      | 352            | 1,277,440      | 25,602         |
| ('efficiency/continuous/duration · latency_ms_per_sample ↓ (ms)', 'native')    | 0.803       | 0.00306        | 3.58           | n/a            |
| ('efficiency/continuous/travel_time · latency_ms_per_sample ↓ (ms)', 'native') | 0.862       | 0.00358        | 1.77           | n/a            |
| ('efficiency/embedding · latency_ms_per_sample ↓ (ms)', 'native')              | 0.0103      | 0.0927         | n/a            | 0.00988        |
| ('efficiency/generation · latency_ms_per_sample ↓ (ms)', 'native')             | n/a         | 3.87           | 48.8           | n/a            |
| ('efficiency/mode_classification · latency_ms_per_sample ↓ (ms)', 'native')    | 0.0385      | n/a            | n/a            | 0.126          |
| ('efficiency/next_location · latency_ms_per_sample ↓ (ms)', 'native')          | 6.75        | 0.0325         | 0.222          | n/a            |
| ('efficiency/recovery · latency_ms_per_sample ↓ (ms)', 'native')               | 0.448       | 0.465          | n/a            | 0.127          |

## Skipped / failed

- UniTraj-tiny@default · next_location: skipped (capability not declared)
- UniTraj-tiny@default · continuous/travel_time: skipped (capability not declared)
- UniTraj-tiny@default · continuous/duration: skipped (capability not declared)
- UniTraj-tiny@default · generation: skipped (capability not declared)
- TrajGPT-tiny@default · recovery: skipped (capability not declared)
- TrajGPT-tiny@default · mode_classification: skipped (capability not declared)
- CLIP-tiny@default · generation: skipped (capability not declared)
