# mobeval: unified evaluation of human-mobility foundation models

`mobeval` evaluates models with different output representations (continuous GPS reconstruction,
discrete visit/region tokens, probabilistic time heads, embeddings, generators) on the same data,
the same samples and the same baselines. See `EVALUATION_PROTOCOL.md` for the rationale and a review
of the previous evaluation run.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # 13 tests, ~3 s
python examples/run_synthetic.py out_synth     # end-to-end run with two reference models, ~10 s
```

`out_synth/` then contains `results.jsonl` (one record per model × task × metric × seed, with CIs),
`leaderboard.csv` and `report.md`.

For real data, fill in the adapters (below) and run:

```bash
python examples/run_real.py --geolife /data/geolife --out results \
    --mymodel ckpt_seed0.pt ckpt_seed1.pt --unitraj unitraj.pt --trajgpt trajgpt.pt
```

## Layout

```
mobeval/
  data.py            canonical dataset, splits, shared grid, windows, masks, staypoints, visit sequences
  loaders.py         CSV/Parquet and GeoLife loaders
  geo.py             haversine, local projection, radius of gyration
  metrics/
    registry.py      direction, unit, valid range, skill formula for every metric (single source)
    reconstruction.py  ADE / FDE / DTW / RMSE / within-d accuracy / grid accuracy
    classification.py  top-k, MRR, NLL, balanced accuracy, macro-F1, ECE
    probabilistic.py   Gaussian (log-)mixtures, closed-form CRPS, NLL with unit Jacobian, PIT
    generative.py      mobility statistics, JSD / W1 on fixed bins, paired Spearman, copy detection
  baselines.py       interpolation, Markov/frequency, train marginal, handcrafted-feature classifier, generators
  adapters/
    base.py          the adapter contract and canonical prediction formats
    reference.py     two small non-neural models that exercise every code path
    templates.py     UniTraj, TrajGPT and MyModel adapters (model calls marked TODO(model))
  context.py         EvalConfig and the shared EvalContext
  tasks.py           recovery, next location, travel time, duration, mode classification, generation, efficiency
  stats.py           bootstrap CIs, paired skill-score CIs
  results.py         ResultRecord schema with automatic sanity flags
  report.py          leaderboards, family summary, Pareto front, Markdown report
  runner.py          EvaluationPipeline
```

## Writing an adapter

Subclass `MobilityModelAdapter`, set `name` and `capabilities`, and implement only the methods the model
supports. The pipeline skips undeclared capabilities and records them as skipped.

| Capability | Method | Return |
|---|---|---|
| `recovery` | `reconstruct(batch, mask)` | `(lat, lon)` arrays `(N, L)` in degrees |
| `next_location` | `predict_location(visits, grid)` | `LocationPrediction` with grid scores, own-token scores plus token centroids, or coordinates |
| `continuous` | `predict_continuous(visits, target)` | `ContinuousPrediction` with a point, a `Mixture` (linear or log space) or samples, in seconds |
| `mode_classification` | `classify_mode(batch, classes)` | `(N, K)` probabilities or logits |
| `embedding` | `embed(batch)` | `(N, d)`; used for linear probing |
| `generation` | `generate(reference_train, n, seed)` | `MobilityDataset` point table |

Checklist that prevents the inconsistencies seen previously:

1. Convert outputs to the canonical formats inside the adapter: degrees (de-normalised with the training
   statistics), seconds, and the correct mixture space. Never report errors in normalised units.
2. For models that reorder tokens (TrajGPT infilling with a separator), put predictions back at their
   original positions before returning.
3. Use `prepare(...)` to fit task heads on the `train` subset the pipeline passes, and re-fit for each label
   fraction and seed so few-shot comparisons use identical labels.
4. Evaluate several training seeds by creating one adapter per checkpoint with a distinct `run_tag`;
   reports aggregate over them.
5. Implement `num_parameters()` so the efficiency comparison is populated.

Masked GPS positions and target visit fields arrive as NaN, so an adapter that reads ground truth fails
loudly instead of producing optimistic numbers.

## Configuration

All settings live in `EvalConfig` (`context.py`): split type and ratios, window length, grid cell size,
staypoint thresholds, visit context length, evaluation seeds, bootstrap size, mask ratios and kinds,
continuous targets and conditioning (`continuous_reveal`), classification protocols and label fractions,
and generation sample size. Custom task lists can be passed to `EvaluationPipeline(tasks=[...])`.

## Adding a metric or task

Register the metric in `metrics/registry.py` (direction, unit, valid range, skill type), compute per-sample
values (or a set-level function of sample indices), and emit records through `Task.emit`, which pairs them
with the task's baselines and computes the CIs. Tasks subclass `Task` and declare the capability they use.
