# mobeval: unified evaluation and (re)training of human-mobility foundation models

`mobeval` trains and evaluates models with very different representations (masked GPS
reconstruction, visit-token generators with probabilistic time heads, dual-view contrastive
encoders) on the **same data, the same splits, the same samples and the same baselines**.
Built-in models: **UniTraj**, **TrajGPT** and the **CLIP mobility model**. See
`EVALUATION_PROTOCOL.md` for the evaluation design and `MODELS.md` for how each model is
wrapped, what was changed and why.

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

Outputs in `output_dir`: `report.md`, `results.jsonl` (every metric with bootstrap CIs),
`leaderboard.csv`, `family_summary.csv`, `run_info.json`, and `checkpoints/<name>.pt` (+ a
readable `.json` with config, training history and provenance).

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
| `clip_mobility` | recovery (autoregressive), embeddings, mode classification, next location, travel time, duration (heads on the frozen visit encoder) | next-token regression + InfoNCE between trajectory and visit views |
| `kinematic_ref`, `weak_ref` | non-neural references | none |

Model options go under `arch:` (architecture), `train:` (`epochs, batch_size, lr, weight_decay,
patience, grad_clip, max_steps_per_epoch, device, seed`, plus `init_from` and `options:` for
model-specific training arguments) and `adapter:` (`device, batch_size, head_train, head_hidden,
head_class_weighted`, ...).

## Running on an HPC cluster (PBS)

`jobs/` contains ready-made PBS scripts (written for the SNS HPC cluster: no default queue, GPUs on the
daneel queues, mandatory use of `/scratch`). Edit `jobs/env.sh` once with your paths and virtual
environment, then:

```bash
qsub jobs/all_in_one.pbs                              # train missing checkpoints + evaluate, one job
bash jobs/submit_all.sh UniTraj-finetuned TrajGPT CLIPMobility   # one job per model + evaluation after
qsub -v MODEL=TrajGPT jobs/train_model.pbs            # a single model
qsub jobs/evaluate.pbs                                # evaluation only, from existing checkpoints
qstat -u $USER                                        # follow, qdel <id> to cancel
```

The scripts copy data to `/scratch/$USER`, run there, and copy `results/` (report, metrics, checkpoints)
back to `RESULTS_DIR`. Checkpoints from earlier jobs are restored first, so a job killed by the walltime
can be re-submitted and continues with the models that are already trained.

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
  adapters/  base.py, torch_base.py, unitraj.py, trajgpt.py, clip_mobility.py, reference.py
  nn/        common.py (training loop, checkpoints, heads), features.py (tokenizers),
             unitraj_net.py, trajgpt_net.py, clip_net.py (networks, checkpoint-compatible)
```
