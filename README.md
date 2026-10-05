# mobeval: unified evaluation and (re)training of human-mobility foundation models

`mobeval` trains and evaluates models with very different representations (masked GPS
reconstruction, visit-token generators with probabilistic time heads, dual-view contrastive
encoders) on the **same data, the same splits, the same samples and the same baselines**.
Built-in models: **UniTraj**, **TrajGPT** and the **CLIP mobility model**. See
`EVALUATION_PROTOCOL.md` for the evaluation design and `MODELS.md` for how each model is
wrapped, what was changed and why. `CODE_GUIDE.md` walks through the code: how each model's
training data is prepared and how each metric is computed.

`python plot_results.py <run>/leaderboard.csv --out figures` draws the results (models against each
other and against the baselines); `python make_table.py <run>/leaderboard.csv` writes the LaTeX tables.

## Install

```bash
pip install -e ".[models,test]"      # core + PyTorch + pytest
mobeval smoke --device cpu           # trains 3 tiny models on synthetic data and evaluates them (~30 s)
python -m pytest -q tests            # 22 tests
```

The evaluation core (metrics, baselines, reports) does not need PyTorch; the model adapters do.

## The workflow: one config file

```bash
mobeval info     --config examples/configs/geolife.yaml   # dataset/split summary and split fingerprint
mobeval train    --config examples/configs/geolife.yaml   # train every model with a `train:` section
mobeval evaluate --config examples/configs/geolife.yaml   # evaluate all models, write the report
mobeval run      --config examples/configs/geolife.yaml   # train missing checkpoints, then evaluate
```

Useful flags: `--models TrajGPT CLIPMobility` (subset), `--device cuda`, `--out DIR`, `-v`.

A config has four parts (full example: `examples/configs/geolife.yaml`):

```yaml
output_dir: results/geolife
dataset: {loader: geolife, path: /data/geolife}        # or loader: csv / synthetic
eval:    {split_by: time, window_length: 64, eval_seeds: [0, 1, 2]}   # any EvalConfig field
models:
  - {name: UniTraj-zeroshot, type: unitraj, checkpoint: model.pt, external_pretraining: true}
  - {name: UniTraj-ft, type: unitraj, train: {init_from: model.pt, epochs: 30, device: cuda}}
  - {name: TrajGPT, type: trajgpt, train: {epochs: 200, device: cuda}}
  - {name: CLIPMobility, type: clip_mobility, train: {epochs: 50, device: cuda}}
```

Outputs land in `<output_dir>/runs/<run_id>/` — `report.md`, `results.jsonl` (every metric with
bootstrap CIs), `leaderboard.csv`, `family_summary.csv`, `run_info.json` — with
`<output_dir>/latest` pointing at the newest run, and `checkpoints/<name>.pt` (+ a readable
`.json` with config, training history and provenance) shared across runs. See
[one directory per run](#one-directory-per-run), and
[surviving a crash](#surviving-a-crash-or-a-walltime-kill) for `--persist-dir` and `--resume`.

## Using your own data

Any GPS table works (CSV or Parquet). Map your columns to `user_id, traj_id, t, lat, lon` in the config;
see `examples/configs/vehicle_panel.yaml` for a complete example with pre-split files:

```yaml
dataset:
  loader: csv
  train_path: data/train.csv      # or `path:` for a single file that mobeval splits
  test_path: data/test.csv
  time_col: datetime
  user_id: uid
  traj_id: trip_id
  lon: lng
  query: "QUALITY >= 2"           # optional row filter
  clean: {max_speed_mps: 70}      # optional GPS cleaning
eval:
  split_by: predefined            # validation is carved from the train file (val_by: time | user)
  staypoint_method: trips         # for data recorded only while moving (vehicle black boxes)
```

Use `staypoint_method: points` (default) when the device also records while people stay somewhere, and
`trips` when it records only during trips, so stays have to be inferred from the gaps between them. Run
`mobeval info` first to check the number of windows, staypoints and visit sequences per split. The
transport-mode task is skipped automatically when the data has no `mode` column.

## POI and road-network features (optional, used by TransferTraj)

`mobeval context` builds them for your data's area and prints the config block to paste:

```bash
# with internet (login node / laptop): downloads from OpenStreetMap via osmnx
mobeval context --config exp.yaml --out data/context

# offline: from files you exported once (Overpass Turbo, QGIS, a Geofabrik extract)
mobeval context --config exp.yaml --out data/context \
    --poi-file pois.geojson --road-file roads.geojson
```

It writes `poi_embed.npy`, `poi_latlon.npy`, `road_embed.npy`, `road_latlon.npy` (embeddings are
one-hot over the most frequent OSM categories, so nothing needs downloading or training) plus a
`*_categories.json` listing what each column means. Useful flags: `--dim`, `--max-features`,
`--road-spacing`, `--pad-km`. Supply your own embeddings instead by writing those `.npy` files
yourself — any real-valued `(N, d)` matrix works, paired with `(N, 2)` (lat, lon) coordinates.

The model compares every trajectory point with every context entry, so cost grows with the number of
entries; `--max-features` caps it. Only TransferTraj uses these today; the other models ignore them.

## Protocols: native heads vs. linear probes

A model is only asked natively for what it declares. UniTraj and TransferTraj are masked
reconstruction encoders with no next-location or time head, so on those tasks they used to
appear as "skipped (capability not declared)". The **linear probe** puts them on the same axis
without inventing a head for them:

```yaml
eval:
  location_protocols:   [native, linear_probe]
  continuous_protocols: [native, linear_probe]
  probe_top_k: 1000            # candidate cells the location probe may predict
  probe_max_train: 100000      # cap on probe training samples (speed only)
```

The probe is **linear** (no hidden layer, so the score is a property of the representation, not
of capacity bolted on top), the encoder is **frozen**, and every model gets the same probe on
the same data. Results carry `protocol: linear_probe` and land in tasks named
`next_location/linear_probe`, so a probe number can never be read as a native capability.

Two details worth knowing:

* **The location probe scores a candidate set.** One output per grid cell is impossible when the
  shared grid has half a million of them (a dense score matrix would be tens of gigabytes), so
  the probe predicts the `probe_top_k` most-visited training cells and everything else falls
  back to training popularity. Targets outside that set count as misses, and the log reports
  what share of test targets the candidates cover — that share is the protocol's accuracy
  ceiling, and acc@1 should be read against it rather than against 1.0.
* **The continuous probe returns a distribution**, not just a point: ridge on log(minutes), with
  the spread fitted on the *validation* residuals, so CRPS, NLL and PIT are all defined and
  comparable with the native heads.

### Generation by rollout

A reconstruction model has no `generate` method, but hiding **the future** rather than a random
subset turns filling-in into generation: seed it with the first few real points of a trajectory,
have it fill the rest, feed its own output back, continue. UniTraj, TransferTraj and
CLIP-Mobility are all evaluated on generation this way, under `protocol: rollout`, never mixed
with a native generator's numbers.

```yaml
eval:
  generation_protocols: [native, rollout]
  rollout_seed_points: 4       # real points the model is seeded with
  rollout_block: 8             # points committed per step before re-feeding
  rollout_noise_m: null        # optional; see the caveat below
```

Two things make the numbers mean something:

* **Comparability.** The metrics compare per-trajectory statistics against *real trajectories*,
  so generating fixed-length windows would compare 64-point fragments with whole trips and every
  model would look wrong for a reason unrelated to the model. The rollout reproduces each seed
  trajectory's own length and its own timestamps, sliding the model's fixed window forward, so a
  generated trajectory is the same kind of object as a real one.
* **Attribution.** The seed prefix is real data and already fixes much of a trajectory's
  statistics, so `seed_only` — the same prefix, then standing still — is reported as a baseline
  for this protocol. A model that does not beat it has added nothing to the seed it was given.

Two caveats worth carrying into a write-up. A model trained with a squared error predicts a
conditional **mean**, not a sample, so its rollouts are smoother and shorter than real
trajectories and the distributional metrics charge it for that; measured on synthetic data, the
radius-of-gyration spread came out ~3× narrower than the real one. `rollout_noise_m` injects
calibrated noise if you want to correct the marginal spread, and is off by default because the
right scale is a modelling choice rather than something the pipeline should pick. And rolling
forward is pure **extrapolation** — every hidden position follows the last visible one — which
some reconstruction models handle badly; positions that run outside twice the data's extent are
clipped and the count is logged, so "the decoder diverged" stays distinguishable from "the model
generates poorly".

## Representation-level tasks

Two tasks score the embedding itself rather than a prediction head. Both are applicable to any
model declaring `embedding`.

### User identification

Can a linear model recover **who** produced a window from its frozen embedding? Closed-set, over
the users with enough windows in both splits.

```yaml
eval:
  user_id_max_users: 100
  user_id_min_windows: 8
```

Read it against the **`mean_location` baseline**, never against chance. Mobility identity is
largely home and work location, so an embedding that merely records absolute position will
re-identify users very well while having learned nothing about behaviour. That baseline is
exactly such an "embedding" — latitude/longitude statistics and nothing else. A model only
demonstrates that it captures individual mobility style if it clears that line. In practice it
often does not, and that is the useful finding.

### Anomaly detection

There are no anomaly labels in a GPS panel, so anomalies are injected into the test windows and
each kind is scored separately. Both protocols are unsupervised — fitted on normal training
data, with the labels used only to score:

| protocol | needs | score |
|---|---|---|
| `embedding_knn` | `embedding` | distance to the nearest training embeddings |
| `reconstruction` | `recovery` | error when filling in masked points |

The kinds are graded by how much of the signal is kinematic, and this is the whole point:

| kind | what it does | step lengths |
|---|---|---|
| `teleport` | displaces a chunk far away | changed — trivially detectable |
| `speed` | scales displacements up | changed — trivially detectable |
| `noise` | heavy GPS jitter | changed — trivially detectable |
| `detour` | rotates the middle heading | **preserved exactly** |
| `loop` | retraces the middle in reverse | **preserved exactly** |

`detour` and `loop` are permutations and rotations of the original displacements, so every step
length survives (verified to ~0.2%, the projection round-trip) and mean/median/percentile speed,
acceleration and total distance are unchanged. A speed check is therefore at chance on them *by
construction* — `max_step` measures 0.54.

They are not invisible to *all* kinematics, and this is the part to be careful about when
reading results: splicing in a rotated or reversed span inserts a sharp turn, so turning-angle
features do carry signal — `turn.mean` alone reaches 0.66 on detour and 0.70 on loop, and the
full `kinematic_knn` baseline reaches 0.60–0.64. **That is the bar, not 0.5.** A model has only
demonstrated learned route plausibility if it clears `kinematic_knn`. Timestamps are never
touched, so nothing can be inferred from the time axis.

```yaml
eval:
  anomaly_kinds: [teleport, detour, loop]
  anomaly_rate: 0.1
  anomaly_protocols: [embedding_knn, reconstruction]
```

Two baselines are always reported: `max_step` (the classic GPS-jump check, no training) and
`kinematic_knn` (distance to training windows in handcrafted speed/acceleration/turn feature
space). Beating chance on `teleport` means nothing — `max_step` gets 1.0 there. Beating
`kinematic_knn` on `detour` or `loop` is the result that means something. A ROC-AUC clearly
*below* 0.5 is also informative: it means the model finds the corrupted windows easier than
normal ones, which is what happens when a retraced route is more predictable than a real one.

### Trajectory retrieval

`retrieval` asks each model to find a trajectory in a database of `retrieval_db_size` test windows,
from a query derived from that same trajectory. Scores are mean rank (MR), mean reciprocal rank
(MRR) and HR@1/5/10. There are three protocols:

- **`odd_even`** runs for every model with embeddings. The query is the window's odd-indexed
  points, the database the even-indexed points of every window, as in t2vec and TrajCL.
- **`cross_modal`** (OmniTraj) queries with the window's topology, road segments, regions or a
  fusion of them, against the GPS embeddings (the OmniTraj paper's Table 2).
- **`condition`** (OmniTraj) queries with the SET of a window's regions or road segments.
  CR@k is the share of those elements found in the top-k retrieved windows (Table 3).

The baselines are scored on the same queries: the Hausdorff distance between point sets, which is
exact and pruned by bounding boxes so that 20,000-window databases stay tractable, and random
ranking. On odd/even halves Hausdorff is hard to beat: a learned embedding has to add something
beyond geometry.

## Checking that train and test cover the same places

`mobeval info` now prints a split-overlap line, and it is worth reading before any location
metric:

```
  overlap: 94.2% of test staypoints are in cells seen in train (48,113 train cells,
           31,904 test cells); 86.1% of test users appear in train
```

Low overlap is not a subtle problem. A model cannot predict a cell it has never seen, and
TrajGPT's region tokenizer will silently **snap** such a target onto its nearest known
region — which may be a hundred kilometres away — so its distance errors are bounded from
below by that snap distance and no amount of training will fix it. When the median snap
exceeds 2 km the tokenizer now says so directly:

```
region tokenizer: 4,812/5,000 (96.2%) of these locations are in cells never seen in training
and were snapped to the nearest known region - a median of 131.4 km away (p95 166.0 km).
Location metrics for this model are bounded by that distance.
```

If you see that, check the split before trusting anything in the Location table. Widening
`tokenizer.cell_m` enlarges each region but does not help when the test area simply is not in
the training data.

## If the job is killed (exit 137)

`Killed` with no traceback is the memory limit, not a crash — the process is SIGKILLed, so
nothing is raised in Python and nothing reaches the progress file. The job scripts now print
what to do; the settings that actually drive peak memory are:

| setting | why it matters |
|---|---|
| `max_eval_samples` | location scoring holds a (samples × grid cells) score matrix per chunk |
| `grid_cell_m` | a **larger** cell means far fewer cells — halving the resolution quarters the matrix |
| `generation_max_real_trajectories` | trajectories behind the generation reference statistics |
| `generation_nn_max_train` / `_max_query` | size of the memorisation comparison |

Two things used to dominate and no longer do. The location task scores in chunks of 512
samples (`NextLocationTask.SCORE_CHUNK`), which is exact — every location metric is per-sample,
and the chunked and unchunked numbers are bit-identical — and takes the peak from 18 GB to
1.9 GB on a 450k-cell grid with 5,000 samples. The generation task used to resample *every*
trajectory in the panel before discarding all but a few thousand, and sized its pairwise
comparison by the full test-set count; both sides are now subsampled first, so its cost no
longer grows with the dataset.

## Training is always on the evaluation split

`train` builds the same `EvalContext` as `evaluate` and trains only on its `train` split, with early
stopping on `val`. Every checkpoint stores a fingerprint of the split assignment; `evaluate` refuses a
checkpoint whose fingerprint differs from the current split (`check_provenance: error | warn | off`),
because such a model may have been trained on today's test data. Checkpoints pre-trained elsewhere (e.g.
the public UniTraj weights) must be declared with `external_pretraining: true`.

## Models

| type | what it does in mobeval | training routine |
|---|---|---|
| `unitraj` | recovery, embeddings, mode classification (head on frozen embeddings) | masked reconstruction; from scratch or `init_from` the public `model.pt` |
| `trajgpt` | next location, travel time, duration (Gaussian mixtures), generation | region CE + travel/duration NLL on visit sequences |
| `transfertraj` | recovery, embeddings, mode classification | span-masked pre-training (optional POI / road features) |
| `omnitraj` | embeddings (retrieval, probes, mode classification); cross-modal and condition-based retrieval from topology / road segments / regions | contrastive alignment of trajectory, topology, road and region encoders (roads need map matching, below) |
| `clip_mobility` | recovery (autoregressive), embeddings, mode classification, next location, travel time, duration (heads on the frozen visit encoder) | next-token regression + InfoNCE between trajectory and visit views |
| `kinematic_ref`, `weak_ref` | non-neural references | none |

Model options go under three keys:

- `arch:` for the architecture.
- `train:` for training: `epochs, batch_size, lr, weight_decay, patience, grad_clip, optimizer,
  scheduler, restore_best, max_steps_per_epoch, device, seed`, plus `init_from` and `options:` for
  model-specific training arguments.
- `adapter:` for loading and inference: `device, batch_size, head_train, head_hidden,
  head_class_weighted`, ...

## Reproducing the original methodology

Each published model can be retrained exactly as its authors did, via `recipe:`:

```yaml
models:
  - {name: UniTraj-paper, type: unitraj, recipe: paper}         # the publication
  - {name: TrajGPT-code,  type: trajgpt, recipe: code}          # the released code, as is
```

- **`paper`** follows the publication. Where the paper is silent it follows the released code.
- **`code`** follows the released code, including where it departs from the paper.

A recipe sets the architecture, the optimiser and schedule (Adam without weight decay, no gradient
clipping, the papers' batch sizes and epochs), and each model's own sample construction:

- **UniTraj:** ATR resampling of whole trajectories; the random/RDP/block/last-n masking mix;
  exactly 100 of 200 tokens hidden.
- **TrajGPT:** 128-visit instances at every start position, left-padded; H3 resolution-7 regions;
  times in hours; no loss masking or GMM initialisation.
- **TransferTraj:** the paper's single-span masking or the repository's multi-span one; a context
  radius of 100 m (paper) or 10 m (code); raw metres; per-task fine-tuning (`options: {objective: tp | trec}`).

Two kinds of difference are logged and stored in the checkpoint's provenance:

- **Deliberate omissions.** Defects that would corrupt an evaluation are never reproduced: target
  leaks, statistics computed on the test data, a training loop that stops after one batch.
- **Your overrides.** Anything set explicitly in the config overrides the recipe and is reported.

`mobeval recipes` prints every setting and every omission. The paper's data and evaluation settings
are in `examples/configs/paper_trajgpt.yaml`, `paper_transfertraj.yaml` and
`paper_unitraj_{pretrain,eval}.yaml`. DATASETS.md explains how to download each dataset.
CLIP-Mobility has no publication and so no recipe.

**Paper metrics in the results.** Cells whose metric is one the model's own paper reported are
marked, in `report.md` and by `make_table.py`:

- **★ in `report.md`, $^\star$ in LaTeX:** the same formula and protocol as the paper.
- **☆ in `report.md`, $^\dagger$ in LaTeX:** the same quantity under a protocol that differs as
  stated. For example, TrajGPT's P(±t) uses the paper's formula, but the released code's time
  heads read the target's own times, and an honest evaluation cannot reproduce that leak.

Most marks are ☆. ★ is only given where mobeval can actually run the paper's protocol: UniTraj's
recovery and prediction error, when the test windows are 200 points at 3 s with maskable
endpoints (`paper_unitraj_eval.yaml`).

The report's "Paper metrics" section lists each one with the reason. The mark concerns the
metric only: a number is comparable with the paper's table only on the paper's data. Several
metrics were added so that the papers' own are available:

- Acc@10/20;
- P(±5/10/20 min), the forecast mass within t minutes of the truth, conditioned on the next visit as
  TrajGPT evaluates it;
- MAPE;
- recovery of the last 5 points (`recovery/last:5`) and TRec's keep-every-8th
  (`recovery/keep_every:8`), with a constant-velocity baseline for the former.

`eval.tasks: [recovery, ...]` restricts a run to some tasks. Without a visit task
(`next_location`, `continuous`, `generation`) no staypoints are computed, which matters on
WorldTrace-sized data.

## Controlling how much data training sees

Visit sequences are built with a sliding window over each user's staypoints, so with the default
`visit_stride: 1` consecutive samples share `visit_context - 1` visits. On a panel dataset that can
mean millions of nearly identical training sequences (and `mobeval info` warns when it does).

```yaml
eval:
  visit_context: 32        # history per sample; also how many targets TrajGPT learns from per sample
  visit_stride: 8          # thin overlapping training sequences (train split only)
  max_train_samples: 200000   # hard cap on training windows and visit sequences
  max_eval_samples: 5000      # val/test views (unchanged by the two above)
```

`visit_stride` and `max_train_samples` affect only the training split, so evaluation stays comparable
and the split fingerprint is unchanged - existing checkpoints remain valid. A third lever lives in the
model's own `train:` section and bounds work per epoch without changing the dataset:

```yaml
    train: {max_steps_per_epoch: 2000, ...}
```

Run `mobeval info --config exp.yaml` to see the resulting sizes before submitting anything.

## Watching a run in progress

Batch jobs are opaque: the work happens on a compute node, possibly for hours. Every run therefore
writes live progress that you can read from anywhere:

```bash
mobeval status --progress-dir ~/results/progress      # one look
mobeval status --config exp.yaml --watch                    # refresh every 10 s
mobeval status --progress-dir DIR --events 20               # plus the recent event log
```

```
* 20260919-135540-8821-run  [running]  job 446898.pbs01 on daneel02
  started Sat 13:55:40   elapsed 1h 12m   last update 3s ago
  phase: training TransferTraj
  steps: [████████░░░░░░░░░░░░░░░░] 3/9   eta ~2h 04m
    TransferTraj           training   epoch 47/200  val 4.31  best 4.29
    UniTraj                trained    -> UniTraj.pt   trained in 12m 31s
    TrajGPT                pending
  records: 124   skipped: 8
```

Reporting starts before the configuration is read whenever the directory is known from
`MOBEVAL_PROGRESS_DIR` or `--progress-dir`, so even a broken config leaves a `failed` run with the
reason. An empty progress directory therefore means the command never started at all - a wrong
interpreter, a failed import, or the command missing from the job script.

A run is marked `stale` when it stops updating while still claiming to run, which is what a killed or
crashed job looks like; `failed` runs show the reason. Several jobs can share one progress directory
(`submit_all.sh` submits one per model), and `status` lists them together, newest first.

Each run writes two files: `<run_id>.status.json`, a snapshot rewritten atomically so reading it mid-run
is safe, and `<run_id>.events.jsonl`, an append-only log (stage boundaries, every training epoch with
its losses, every task with its record count and duration) that is easy to parse for your own plots.

**Where it goes.** `MOBEVAL_PROGRESS_DIR`, `--progress-dir`, or `progress_dir:` in the config, falling
back to `<output_dir>/progress`. On a cluster this must be a shared filesystem (home or project
folder): jobs compute on node-local `/scratch`, which the login node cannot see. The supplied PBS
scripts already set it to `$RESULTS_DIR/progress`.

Nothing in the status path imports PyTorch, so it works in a plain shell on a login node.

## Running on an HPC cluster (PBS)

`jobs/` contains ready-made PBS scripts (written for the SNS HPC cluster: no default queue, GPUs on the
daneel queues, mandatory use of `/scratch`). Edit `jobs/env.sh` once with your paths and virtual
environment, then:

```bash
qsub jobs/all_in_one.pbs                              # train missing checkpoints + evaluate, one job
bash jobs/submit_all.sh UniTraj-finetuned TrajGPT CLIPMobility   # one job per model + evaluation after
qsub -v MODEL=TrajGPT jobs/train_model.pbs            # a single model
qsub jobs/evaluate.pbs                                # evaluation only, from existing checkpoints
qsub -v RESUME=latest jobs/all_in_one.pbs             # continue where a killed job stopped
qstat -u $USER                                        # is it queued or running?
python -m mobeval status --progress-dir ~/results/progress --watch   # what is it doing?
```

The scripts copy their inputs to `/scratch/$USER/mobeval/<job id>` and run there, as the cluster
requires. On the daneel nodes `/scratch` is a disk local to each node, so it only holds one job's
working copy: keep every input (datasets, context files, road networks, OSM extracts) in home or a
project folder. `stage_in` in `jobs/env.sh` copies the dataset paths and every other absolute input
path in the config (a model's `context:` files, `roads_file`, ...) and rewrites the config to point at
the copies; it stops with an error if the config points at a `/scratch` file the node does not have.

## Surviving a crash or a walltime kill

Nothing finished waits for the end of the job. `MOBEVAL_PERSIST_DIR` (set to `$RESULTS_DIR` by
`jobs/env.sh`, or `--persist-dir` / `persist_dir:`) is a durable directory outside scratch that is
written to *as the work completes*:

| what | when it is written |
|---|---|
| a model's checkpoint | after **every epoch that improves**, not at the end of training |
| a task's metrics | after **every task**, not at the end of the evaluation |
| report, leaderboard, `run_info.json` | when the evaluation finishes |

So a job that dies at hour 11 of 12 leaves behind every model that finished training, the best epoch
of the one that was still training, and every metric computed so far. Writes are atomic (temporary
file plus rename), so an interrupted write cannot corrupt the previous good copy, and a truncated
last line in `results.jsonl` costs one record rather than the file.

PBS sends `SIGTERM` before `SIGKILL` when the walltime expires. mobeval catches it, finishes its
current write, marks the run `interrupted` in the progress file with a pointer to the results so far,
and exits 143 (so `depend=afterok` chains do not continue on half-finished work).

**Continuing.** Re-submit with `RESUME`:

```bash
qsub -v RESUME=latest jobs/all_in_one.pbs        # continue the newest run
qsub -v RESUME=20260921-100122-994-run jobs/all_in_one.pbs   # or a specific one
mobeval run --config exp.yaml --resume           # the same thing, directly
```

A resumed run reuses the *same* run directory and:

* skips models whose checkpoint is marked complete;
* **continues** a model whose checkpoint is marked incomplete, from its best epoch — an
  interrupted model is never silently accepted as trained;
* keeps every evaluation task that already has results and computes only what is missing.

It works even when scratch is empty, which is the normal case for the next job: checkpoints and the
earlier run's `results.jsonl` are restored from the persist directory first.

## One directory per run

Each invocation writes into `<output_dir>/runs/<run_id>/`, with `<output_dir>/latest` pointing at the
newest:

```
$RESULTS_DIR/
  checkpoints/                 one per model, SHARED by every run so re-runs do not retrain
    TrajGPT.pt  TrajGPT.json   (the .json sidecar holds config, history and complete: true/false)
  runs/
    20260921-100122-994-run/   report.md  results.jsonl  leaderboard.csv  family_summary.csv  run_info.json
    20260921-143010-1002-run/
  latest -> runs/20260921-143010-1002-run
  progress/                    live status files (see above)
```

The run id matches the one `mobeval status` shows, so a progress line and a results directory are
easy to line up. Checkpoints are deliberately *not* per-run: training is the expensive part, and a
second evaluation must be able to reuse the first one's weights. Set `run_dirs: false` in the config
for the old flat layout, or `checkpoint_dir:` to put weights somewhere else entirely.

## Python API

```python
from mobeval import EvalConfig, EvaluationPipeline, markdown_report
from mobeval.loaders import load_geolife
from mobeval.adapters.unitraj import UniTrajAdapter
from mobeval.adapters.trajgpt import TrajGPTAdapter
from mobeval.adapters.clip_mobility import CLIPMobilityAdapter

pipe = EvaluationPipeline(EvalConfig(window_length=64, eval_seeds=(0, 1, 2)))
ctx = pipe.prepare(load_geolife("/data/geolife"))

unitraj = UniTrajAdapter.pretrain(ctx, init_from="model.pt", train={"epochs": 30, "device": "cuda"},
                                  out="ckpt/unitraj_ft.pt")
trajgpt = TrajGPTAdapter.train(ctx, train={"epochs": 200, "device": "cuda"}, out="ckpt/trajgpt.pt")
clip = CLIPMobilityAdapter.from_checkpoint("ckpt/clip.pt", device="cuda")

store = pipe.run([unitraj, trajgpt, clip], ctx)
open("report.md", "w").write(markdown_report(store, ctx))
```

## Adding another model

Subclass `MobilityModelAdapter` (or `adapters.torch_base.TorchAdapter` for PyTorch models), declare
`capabilities`, implement the matching methods in the canonical formats (degrees, seconds, shared-grid
or token scores, `Mixture` distributions), optionally a `pretrain`/`train` classmethod that uses
`nn.common.fit`, and register it in `registry.MODEL_TYPES`. The adapter contract is documented in
`adapters/base.py`; the three built-in adapters are complete examples.

## Layout

```
mobeval/
  cli.py, config.py, registry.py   command line, config files, model registry
  data.py, loaders.py, geo.py      canonical data, splits, windows, staypoints, loaders
  context.py, runner.py, tasks.py  EvalConfig / shared context, pipeline, tasks
  metrics/, baselines.py, stats.py metrics registry and implementations, baselines, bootstrap
  results.py, report.py            result schema with sanity flags, reports
  layout.py, progress.py           per-run directories + durable mirroring, live status
  recipes.py, paper_metrics.py     original training recipes; which metrics each paper reported
  roads.py, mapmatch.py            OSM road segments, HMM map matching, FMM import/export (OmniTraj roads)
  adapters/  base.py, torch_base.py, unitraj.py, trajgpt.py, transfertraj.py, omnitraj.py, clip_mobility.py, reference.py
  nn/        common.py (training loop, checkpoints, heads), features.py (tokenizers),
             unitraj_net.py, trajgpt_net.py, transfertraj_net.py, clip_net.py (networks,
             checkpoint-compatible), unitraj_sampling.py (UniTraj's ATR resampling and masking),
             omnitraj/ (OmniTraj's network, vendored unmodified), omnitraj_prep.py (its inputs)
make_table.py                      LaTeX tables from leaderboard.csv (paper metrics marked)
DATASETS.md                        where to get each paper's data
```
