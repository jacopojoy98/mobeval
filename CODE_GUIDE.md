# Reading the code

A guide to answering two questions from the source:

- **How is the data prepared for the training of model X?** → section 4
- **How is metric A calculated for model B?** → sections 5 and 6

Sections 1–3 give the map you need first. References are `file :: function`, not line numbers,
so they stay valid as the code changes. `examples/trace_walkthrough.py` runs the path of section 6
by hand on synthetic data, so you can print every intermediate array.

---

## 1. The whole run in one page

`mobeval run --config X.yaml` is `cmd_train` followed by `cmd_evaluate` (`cli.py`). Both start
from the same object, the **context**, built once by `cli.py :: _context`:

```
YAML ──config.load_config──▶ cfg
cfg["dataset"] ──config.load_dataset──▶ MobilityDataset      one table of GPS points
cfg["eval"]    ──config.eval_config──▶ EvalConfig            every evaluation setting
                         │
                         ▼
        context.py :: EvalContext.__init__                    the shared views of the data
          splits      train / val / test                      data.py :: MobilityDataset.split
          windows     fixed-length GPS windows per split      data.py :: make_windows
          staypoints  stays (visits) per split                data.py :: staypoints_from_trips | detect_staypoints
          visits      context of C visits → next visit        data.py :: make_visit_sequences
          grid        the shared location label space         data.py :: SpatialGrid | H3Grid
                         │
        ┌────────────────┴─────────────────┐
        ▼                                  ▼
 TRAINING (cmd_train)               EVALUATION (cmd_evaluate)
 registry.py :: train_model         registry.py :: load_model  → one adapter per model
   → Adapter.pretrain / .train      runner.py :: EvaluationPipeline.run
   → nn/common.py :: fit              for adapter, for task:  task.run(adapter, ctx)
   → checkpoint                        → tasks.py :: Task.emit → stats.py :: evaluate_with_ci
                                       → ResultRecord → results.jsonl
                                    report.py → report.md, leaderboard.csv ; make_table.py → LaTeX
```

Three ideas explain almost everything else:

1. **The context is the only source of data.** Training and evaluation read `ctx.windows`,
   `ctx.visits`, `ctx.staypoints`, `ctx.splits`. No model loads data on its own, so every model
   sees the same split and the same test samples.
2. **An adapter translates.** Each model has one adapter (`adapters/<model>.py`) that turns the
   common views into that model's input tensors and turns its output back into degrees, seconds or
   scores over the shared grid. The network itself lives in `nn/<model>_net.py`.
3. **A task owns the metric.** Tasks (`tasks.py`) hide the targets, call an adapter method, and
   compute the metric from the answer. Metrics are never computed inside an adapter, so the formula
   is identical for every model.

## 2. The data views (read these two classes first)

Everything a model or a task receives is one of two containers in `data.py`.

**`TrajectoryBatch`** — N windows of L consecutive GPS points: `lat`, `lon`, `t` are `(N, L)`
arrays (degrees, unix seconds), plus `user_id`, `traj_id`, optional `mode`. Built by
`make_windows(ds, length, stride, max_gap_s)`: each trajectory is cut wherever two points are more
than `max_gap_s` apart, and each remaining piece is cut into non-overlapping windows of exactly
`length` points. A piece shorter than `length` gives no window at all.

**`VisitBatch`** — N samples of "C context visits → one target visit": `ctx_lat/lon/cell`,
`ctx_t_arrive`, `ctx_t_leave` are `(N, C)`; `tgt_lat/lon/cell`, `tgt_t_arrive`,
`tgt_travel_time_s`, `tgt_duration_s` are `(N,)`. Built by `make_visit_sequences`: a sliding
window over each user's staypoints. The target must belong to the split; the context may reach
back into the user's earlier visits from another split, and `ctx_in_split` records which context
visits are the split's own.

How `EvalContext.__init__` fills them, in order:

| Step | Code | Setting in `eval:` |
|---|---|---|
| Split | `MobilityDataset.split` | `split_by` (`time`, `user`, `predefined`), `val_by`, `val_fraction` |
| Staypoints, over the whole dataset | `EvalContext.detect_staypoints` | `staypoint_method` (`trips` = gaps between trips, `points` = dwell), `staypoint_time_s` |
| Shared grid | `SpatialGrid.from_dataset` or `H3Grid.from_dataset` | `grid_backend`, `grid_cell_m` |
| Windows per split | `make_windows`, then `_cap` (val/test) or `_cap_train` | `window_length`, `max_gap_s`, `max_eval_samples`, `max_train_samples` |
| Visits per split | `make_visit_sequences`, then `_cap_visits` or `_cap_train` | `visit_context`, `visit_stride` |

The caps are seeded subsamples (`split_seed`), so val/test samples are the same for every model.

> The last run had only 13.6k training windows from 40M points. With no `max_train_samples` in the
> config, `_cap_train` is not the cause. The place to look is `make_windows`: with
> `window_length: 64` and `max_gap_s: 300`, a trip piece shorter than 64 points produces nothing.
> `mobeval info` prints points and windows per split (`EvalContext.summary`), so a lower
> `window_length` or a higher `max_gap_s` can be compared quickly. This is the likely cause, not a
> confirmed one: it has not been checked on the panel data.

## 3. The adapter contract

`adapters/base.py :: MobilityModelAdapter` lists the methods a task may call. An adapter declares
which it supports in `capabilities`, and `Task.applicable` checks that set.

| Method | Receives | Returns | Capability |
|---|---|---|---|
| `reconstruct(batch, mask)` | windows with hidden points set to NaN | `lat, lon` `(N, L)` in degrees | `RECOVERY` |
| `embed(batch)` | windows | `(N, d)` vectors | `EMBEDDING` |
| `predict_location(visits, grid)` | visits with the target blanked | `LocationPrediction` | `NEXT_LOCATION` |
| `predict_continuous(visits, target)` | same | `ContinuousPrediction` (seconds) | `CONTINUOUS` |
| `classify_mode(batch, classes)` | windows | class probabilities | `MODE_CLASSIFICATION` |
| `generate(reference, n, seed)` | train data | a `MobilityDataset` | `GENERATION` |
| `embed_query`, `embed_database`, `elements` | windows | vectors / sets | `CROSS_MODAL` (OmniTraj) |

`adapters/torch_base.py :: TorchAdapter` adds what all five models share: device handling, the
embedding cache in `embed` (keyed on trajectory id and first/last time; subclasses implement
`_embed_batch`), and a mode-classification head fitted on frozen embeddings.

| Model | Adapter | Network | Capabilities |
|---|---|---|---|
| UniTraj | `adapters/unitraj.py` | `nn/unitraj_net.py` | recovery, embedding, mode |
| TrajGPT | `adapters/trajgpt.py` | `nn/trajgpt_net.py` | next location, continuous, generation |
| TransferTraj | `adapters/transfertraj.py` | `nn/transfertraj_net.py` | recovery, embedding, mode |
| CLIP-Mobility | `adapters/clip_mobility.py` | `nn/clip_net.py` | recovery, embedding, next location, continuous, mode |
| OmniTraj | `adapters/omnitraj.py` | `nn/omnitraj/` (original files) | embedding, mode, cross-modal |

A task a model has no head for is still run when the model has `EMBEDDING`: a linear probe is
fitted on its frozen embedding (`probes.py`) and the result is stored with
`protocol="linear_probe"`.

---

## 4. How is the data prepared for training model X?

**The common path.** `cli.py :: cmd_train` trains every model that has a `train:` section.
`registry.py :: train_model` applies the recipe (`recipes.py :: apply`, for `recipe: paper | code`),
resolves `init_from`, and calls the adapter's training classmethod (`TRAIN_METHOD`: `pretrain`,
or `train` for TrajGPT) with `ctx`, the `train:` dict, and `train.options` as keyword arguments.

Every training method has the same shape, so they read alike:

1. build the adapter (new, or `from_checkpoint(init_from)`), fitting any normalisation, tokenizer
   or projection **on the train split only**;
2. define `loss_fn(idx, training)`: build the batch for sample indices `idx` from the train
   (`training=True`) or validation data, run the network, return a scalar loss;
3. hand `loss_fn` to `nn/common.py :: fit`, which shuffles indices, steps the optimiser, computes
   the validation loss each epoch, early-stops, and restores the best weights. The checkpoint is
   saved on every improvement (`TorchAdapter.epoch_checkpointer`).

So the answer to "how is the data prepared" is always: **read `loss_fn` and the helper it calls to
build a batch.** Hyper-parameters come from `nn/common.py :: TrainConfig`.

### UniTraj — `UniTrajAdapter.pretrain`

Source: `ctx.windows["train"]` and `["val"]`.

- **Normalisation.** A fresh model fits mean/std of the lon/lat offsets from each window's first
  point on the train windows. With `init_from` it keeps the checkpoint's statistics
  (`PUBLIC_NORM` for the public weights) unless `fit_norm: true`.
- **Encoding — `_encode(lat, lon, t, hidden)`.** Per window: offsets in degrees from the first
  *visible* point, in (lon, lat) order, z-normalised, zero at hidden points; right-padded to the
  fixed length 200; `intervals` = time differences in seconds; `hidden` is True for masked and for
  padding positions.
- **Masks (default, `sampling="mobeval"`).** `masks(n)` in `pretrain`: `mask_ratio` (0.5) of the
  points, random positions, or one contiguous block with probability `block_prob` (0.3); endpoints
  stay visible (`data.py :: make_mask`). Validation masks are random with a seed fixed by the batch.
- **Loss.** `loss_fn` encodes the window twice (unmasked = target, masked = input), runs the
  network, and takes the mean squared error in normalised units over the masked real points.
- **Original methodology (`sampling="original"`, set by the recipes).** `_pretrain_original` builds
  batches with `nn/unitraj_sampling.py :: build_batch`: optional adaptive resampling
  (`atr_resample`), one of four mask strategies per sample (`strategy_mask`: random, block,
  key-points via `rdp_keypoints`, last-n), and the original's top-up to a fixed number of hidden
  tokens. `sample_unit: trajectory` uses whole trips (`trajectories`) instead of windows.

### TrajGPT — `TrajGPTAdapter.train`

Source: `ctx.staypoints["train"]` and `ctx.visits["train"]` / `["val"]`.

- **Fitted on train staypoints.** `nn/features.py :: RegionTokenizer.fit` (region vocabulary; 1 km
  grid by default, 4 special tokens in front), `LocalProjection.from_points` (metric x/y), the
  99th percentiles of travel time and duration (`max_travel_h`, `max_duration_h`), and `t0` (the
  first arrival).
- **A sample.** `tensors(v)` appends the target to the C context visits (`_with_target`), giving a
  sequence of C+1 visits. Output position k predicts visit k+1. Targets per position: travel time
  = arrival − previous departure, duration = departure − arrival, both in hours; duration is
  clipped at `max_duration_h`.
- **Loss masks.** `own` excludes positions whose visit belongs to another split (from
  `ctx_in_split`). The travel-time loss also excludes gaps above `max_valid_travel_h`
  (`travel_loss_mask`).
- **Encoding — `_sequence(lat, lon, ta, tl)`.** Region token ids, projected x/y, arrival and
  departure as time since a reference (`time_reference`: the dataset's first arrival, or the
  Monday before the sequence) divided by `time_input_unit_s`.
- **Loss.** `loss_fn`: cross-entropy on the next region + Gaussian-mixture negative log-likelihood
  on travel time and on duration (`nn/trajgpt_net.py :: gmm_nll`).
- **Original methodology (`sequences="original"`).** `_train_original` with `_original_instances`:
  per user, windows of `seq_len` (128) visits over the full staypoint history, left-padded; the
  nested `batch` function builds the tensors and the same masks.

### TransferTraj — `TransferTrajAdapter.pretrain`

Source: `ctx.windows["train"]` and `["val"]`.

- **Fitted on train.** Only the projection centre (mean lat/lon of the train points).
- **Encoding — `_encode`.** Coordinates projected to metres and divided by `coord_scale` (1000),
  taken relative to the first visible point; features per point `[x, y, t, t − t0]`, each paired
  with a token channel (known / masked). Shape `(B, L, 4, 2)`.
- **Masks — `_pretrain_masks`.** `masking="code"`: the window is cut at `span_div_ratio·L` random
  places and `span_mask_ratio` of the spans are hidden completely; `masking="paper"`: one random
  span. Then each point loses its spatial *or* its temporal part with probability
  `feature_mask_prob`. Validation uses span masks only unless `val_feature_masking`.
- **Batch — `_pretrain_batch`.** Input = masked features with the mask token; target = the full
  sequence with masked positions marked as unknown. The loss is `net.loss(...)` in
  `nn/transfertraj_net.py`.
- **Fine-tuning objectives.** `objective="tp"` hides the spatial part of the last `pred_len`
  points; `"trec"` keeps every `keep_every`-th point and the last. Both use `make_mask`.

### CLIP-Mobility — `CLIPMobilityAdapter.pretrain`

Source: `ctx.windows` (train, val) paired with staypoints of **all** splits as history.

- **Pairing — `pair_windows_with_visits`.** For each window, the same user's last `visit_context`
  staypoints that ended before the window starts, right-padded; `build(split)` drops windows with
  fewer than `min_visits` (2) previous visits and logs how many remain.
- **Point tokens — `nn/features.py :: point_tokens`.** 9 values per GPS point: dx, dy in km from
  the first point, log time step, speed, heading sin/cos, time-of-day sin/cos, weekday.
- **Visit tokens — `visit_tokens`.** 9 values per visit: projected x, y (10 km units), arrival and
  departure time-of-day sin/cos, log duration, log travel time from the previous visit, weekday.
- **Loss.** `loss_fn`: next-point prediction (MSE between the trajectory transformer's output for
  points `0..L-2` and the tokens `1..L-1`) plus the contrastive loss between the window embedding
  and the visit-history embedding, weighted by `prediction_weight` and `clip_weight`.

### OmniTraj — `OmniTrajAdapter.pretrain`

Source: `ctx.splits` (whole trips) or `ctx.windows`, plus the map-matching table `roads_file`.

- **Units.** `sample_unit="trajectory"`: each trip, split at gaps longer than `max_gap_s`, kept if
  it has at least `min_points` points. `"window"`: the context windows.
- **Fitted on train.** Normalisation (mean/std of resampled coordinates),
  `nn/omnitraj_prep.py :: RegionGrid.fit` (16×16 over the train area, or `grid_cell_m`), and
  `RoadVocab.fit` on the matched segment ids. Road segments per point come from `_segments`,
  which looks up `(traj_id, t)` in the table written by `mobeval mapmatch`.
- **A sample — `_samples` → `omnitraj_prep.py :: build_sample`.** Four views of one trip:
  `trajectory` (resampled to 200 points, `resample`), `topology` (its simplified shape,
  `topology`), `region` (de-duplicated grid ids), `road` (de-duplicated segment tokens between
  begin and end tokens). In training the region and road sequences are augmented
  (`augment_region`, `augment_road`).
- **Loss.** `loss_fn` calls the original network's forward, which returns the contrastive loss
  over the modality pairs and fusions (`_build_net`, options `loss` and `pairs`).

The preprocessing here is reconstructed from the paper and one sample file, because the original
preprocessing code is not public; `MODELS.md` says which parts are exact and which are inferred.

---

## 5. How is metric A calculated for model B?

Every number in the report is produced by the same five steps. To trace one, answer five
questions in order.

| # | Question | Where to look |
|---|---|---|
| 1 | Which task and protocol does the row belong to? | the row's `task` and `protocol` in `results.jsonl` / `report.md` → the task class in `tasks.py` (list in `default_tasks`) |
| 2 | Which test samples, and what is hidden? | the start of that task's `run`: `ctx.windows["test"]` or `ctx.visits["test"]`, then `make_mask` / `TargetGuard` (`adapters/base.py`) |
| 3 | What does the model return? | the adapter method the task calls (table in section 3) |
| 4 | Which function turns it into one value per sample? | the metric function the task calls (`metrics/*.py`, table below) |
| 5 | How do per-sample values become the reported number? | `tasks.py :: Task.emit` → `stats.py :: evaluate_with_ci`, using the metric's entry in `metrics/registry.py` |

**Step 5 is the same for every metric.** A metric function returns one value per test sample.
`evaluate_with_ci` then:

- aggregates them as the registry says (`aggregate`: mean, median, p90, or `rmse` = square root of
  the mean, used when the per-sample value is a squared error);
- bootstraps a 95% interval by resampling the test samples `n_boot` times;
- computes the **skill** against the first baseline in the task's `preferred` list that has this
  metric, on the same samples and the same bootstrap resamples (`stats.py :: skill_score`; the
  formula depends on the metric's `skill` type: `1 − v/b` for errors, `(v − b)/(1 − b)` for
  fractions, `b − v` for log scores).

Metrics that only exist for a set of samples (macro-F1, ROC-AUC, PIT) are passed as functions of
the sample indices instead of arrays; the rest is identical. `Task.emit` also sets the ★/☆ mark
(`paper_metrics.py :: paper_level`) and emits the baselines' own rows once per task.

**Task → code map**

| Task name in the report | Task class | Adapter call | Metric function | Baselines (`baselines.py`) |
|---|---|---|---|---|
| `recovery/<kind>@<ratio>`, `recovery/last:5`, `recovery/keep_every:8` | `RecoveryTask` | `reconstruct` | `metrics/reconstruction.py :: recovery_metrics` | `linear_interpolation`, `last_observed`, `constant_velocity` |
| `next_location` | `NextLocationTask` | `predict_location` | `metrics/classification.py :: ranking_metrics` + distance in `NextLocationTask._score` | `LocationBaselines` (markov1, user_frequent, global_popular) |
| `next_location/linear_probe` | `NextLocationTask._probe_score` | `embed` | `probes.py :: location_probe` → `candidate_ranking_metrics` | same |
| `continuous/travel_time…`, `continuous/duration…` | `ContinuousValueTask._run` | `predict_continuous`, or `embed` + `probes.py :: continuous_probe` | `metrics/probabilistic.py :: continuous_metrics` | `ContinuousBaselines` (train_marginal) |
| `mode/<protocol>@<fraction>` | `ModeClassificationTask` | `classify_mode` or `embed` | `metrics/classification.py :: classification_metrics` | handcrafted_gbdt, majority |
| `generation/<statistic>` | `GenerationTask` | `generate`, or `rollout.py :: masked_rollout` over `reconstruct` | `metrics/generative.py :: trajectory_stats`, `compare_stat` | uniform_bbox, seed_only, real_noise_floor |
| `user_identification@Kusers` | `UserIdentificationTask` | `embed` | `baselines.py :: linear_probe` → `ranking_metrics` | mean_location, majority |
| `anomaly/<kind>` | `AnomalyDetectionTask` | `embed` or `reconstruct` | `metrics/detection.py :: knn_distance`, `detection_metrics` | kinematic_knn, max_step |
| `retrieval/odd_even`, `retrieval/cross_modal:<m>`, `retrieval/condition:<m>` | `RetrievalTask` | `embed`, `embed_query`, `embed_database`, `elements` | `RetrievalTask._ranks`, `_scores` | hausdorff, random |
| `efficiency…` | `EfficiencyTask` | — | timings collected by `ctx.timed` | — |

**What "native" and "linear_probe" mean in the code.** In `NextLocationTask.run` and
`ContinuousValueTask._run` the loop over protocols has two branches. `native` calls the model's
own head. `linear_probe` embeds the context visits as a short trajectory
(`tasks.py :: visits_as_windows`: C points at the visit coordinates and arrival times), calls
`adapter.embed` once per split (`_context_embeddings`), and fits a linear model on the train
embeddings. This is how UniTraj, TransferTraj and OmniTraj get a next-location or travel-time
number without having such a head.

## 6. Four worked traces

### a. `ade_m` for UniTraj on `recovery/block@0.5`

1. `RecoveryTask.run`: `batch = ctx.windows["test"]` (at most `max_eval_samples` windows). For
   each kind × ratio and each seed in `eval_seeds`: `make_mask(n, L, 0.5, "block", seed)` → a
   boolean `(N, L)` array, True = hidden, endpoints kept.
2. `TargetGuard.hide_masked` sets the hidden lat/lon to NaN, so a model cannot read them.
3. `UniTrajAdapter.reconstruct`: `_encode` (offsets from the first visible point, normalised,
   padded to 200) → network → `_decode` (de-normalise, add the origin back) → degrees.
4. `recovery_metrics`: haversine distance in metres between prediction and truth at every point;
   `ade_m` is the mean over the **masked** points of each window, giving one value per window.
   The same function is run on `linear_interpolation(hidden, mask)`.
5. `Task.emit` → `evaluate_with_ci("ade_m", …)`: mean over windows, bootstrap interval, skill =
   `1 − ADE_model / ADE_linear_interp`.

Same function, other metrics: `rmse_m` is the per-window mean squared distance, aggregated with
`rmse`; `fde_m` uses only the last point of each hidden block (`block_ends`); `dtw_m` is
`dtw_masked`; `acc_100m` is the share of masked points within 100 m.

For TransferTraj or CLIP-Mobility only step 3 changes (`TransferTrajAdapter.reconstruct`,
`CLIPMobilityAdapter.reconstruct`, which fills hidden points one at a time from left to right).

### b. `acc@1` for TrajGPT on `next_location` (native)

1. `NextLocationTask.run`: `v = ctx.visits["test"]`; `TargetGuard.hide_visits(v)` blanks the target.
2. `TrajGPTAdapter.predict_location`: appends a placeholder visit after the context (a copy of the
   last context visit, arriving when that visit ended), runs the network, and takes the region
   logits at the last position (`_region_logits`). It returns `token_scores` over its own regions
   and `token_latlon`, the region centres.
3. `NextLocationTask._model_scores`: softmax (`normalise_probs`), then the mass of every region is
   added to the shared-grid cell containing its centre. Every model is scored on that same grid.
4. `ranking_metrics` → `rank_of_true`: the rank of the true cell `v.tgt_cell`, ties counted against
   the model. `acc@1` = 1 if the rank is 1. `dist_err_m` is the haversine distance from the centre
   of the model's top region to the true visit.
5. `emit`: mean over samples; skill against `markov1`.

### c. `acc@1` for UniTraj on `next_location/linear_probe`

1. Same samples as (b).
2. `_context_embeddings`: for train, val and test, `hide_visits` → `visits_as_windows` →
   `UniTrajAdapter.embed` → `_embed_batch` (nothing hidden, C points padded to 200, `cls` pooling).
3. `probes.py :: location_probe`: the `probe_top_k` most visited train cells are the classes
   (`candidate_cells`); a single linear layer with softmax is trained on up to `probe_max_train`
   train embeddings (`_linear_softmax`).
4. `candidate_ranking_metrics`: a target inside the candidate set is ranked among the candidates;
   a target outside it counts as a miss. `probe_coverage` reports the share of targets that were
   reachable, which is the ceiling for `acc@1`.
5. `emit` with `protocol="linear_probe"`.

### d. `crps_min` for CLIP-Mobility on `continuous/travel_time/linear_probe|given:location|<=4h`

1. `ContinuousValueTask.run` runs `_run` once per conditioning (`reveal`): the configured one and
   the paper's (`PAPER_REVEAL`, the `|given:…` suffix). `_valid` keeps samples with a finite,
   positive target of at most `travel_time_max_h` (the `|<=4h` suffix).
2. `_probe`: context embeddings as in (c), with the revealed fields appended as extra columns
   (`_revealed_features`).
3. `probes.py :: continuous_probe`: ridge regression on log(minutes); the offset and spread come
   from the **validation** residuals; the result is a log-normal distribution per sample
   (`Mixture`, `space="log"`) and its median as the point forecast.
4. `continuous_metrics`: `crps_min` = `crps_mixture` (closed form for linear-space mixtures, 500
   samples for log-space ones); `mae_min`, `rmse_min`, `mape` use the median; `nll`, `coverage80`
   and `pit_ks` use the distribution; `p_within_*min` is `p_within`.
5. `emit`: mean over samples; skill against `train_marginal` (`ContinuousBaselines`).

The native path for CLIP-Mobility replaces steps 2–3 with
`CLIPMobilityAdapter.predict_continuous`, which fits a mixture head on its frozen visit-encoder
state; for TrajGPT it is `TrajGPTAdapter._continuous_batch`.

## 7. From records to the report

- `results.py :: ResultRecord` is one row: model, task, metric, value, CI, protocol, baseline,
  skill, seed, `paper_match`. Unit, direction and family are filled from the registry, and values
  outside the metric's valid range are flagged.
- `runner.py :: EvaluationPipeline.run` writes `results.jsonl` after every finished task.
- `report.py :: leaderboard` averages over seeds and run tags; `markdown_report` writes `report.md`.
- `make_table.py` reads `leaderboard.csv` and writes the LaTeX tables.

## 8. Reading it yourself

- **Find where a metric is computed:** `grep -rn '"crps_min"' mobeval/` gives its registry entry
  and the function that fills it.
- **Find what a config key does:** `grep -rn 'window_length' mobeval/`; every `eval:` key is a
  field of `context.py :: EvalConfig`, every `train:` key a field of `nn/common.py :: TrainConfig`,
  and `train.options` are the keyword arguments of the adapter's training method.
- **Print the intermediate arrays:** `python examples/trace_walkthrough.py` (synthetic data, CPU),
  then put a `breakpoint()` in the adapter's `_encode` or in the metric function.
- **Run on a small real sample:** add `max_eval_samples: 200` and `tasks: [recovery]` under `eval:`
  in a copy of the config and submit it as a job; on the cluster nothing runs outside PBS.
- **Check a claim in this guide:** the tests are executable examples — `tests/test_core.py`
  (metrics, masks, statistics), `tests/test_models.py` (adapters end to end),
  `tests/test_paper.py` (recipes and original-methodology sampling).

Related documents: `EVALUATION_PROTOCOL.md` (why each metric and baseline was chosen),
`MODELS.md` (how each reimplementation differs from the original), `DATASETS.md` (loaders and
paper datasets).
