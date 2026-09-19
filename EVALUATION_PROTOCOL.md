# Evaluation protocol: review of the current run and what the pipeline changes

This document reviews *Evaluation Methodology for Human Mobility Foundation Models* and the results it
logs, then describes the protocol implemented in `mobeval`. Recommendations draw on the UniTE pipeline
(Lin et al., 2024) and the STFM pipeline survey (Fang et al., 2025), which identifies the lack of
systematic benchmarks as an open problem for trajectory foundation models.

## 1. Problems in the current results, and what they most likely mean

### 1.1 Reconstruction numbers are not in the same space

UniTraj reports ADE = 0.539 and MSE = 2.135; My Model reports ADE = 953.9 and MSE = 5.4 × 10⁸.
The methodology treats both as metres, but neither pair is consistent with that reading.

An ADE of 0.54 m is below the noise floor of consumer GPS (typically several metres), so UniTraj's value is
almost certainly in normalised coordinate units, not metres. My Model's values look like metres, but the
ratio between MSE and ADE is diagnostic. If a fraction *f* of points has a large error *E* and the rest are
small, then MSE ≈ f·E² and ADE ≈ f·E, so E ≈ MSE / ADE. For My Model that gives E ≈ 570 km affecting
roughly 0.2 % of points. This pattern points to a small number of catastrophic points (padding positions
included in the mask, coordinates at (0, 0), a lat/lon swap, or de-normalisation with the wrong statistics)
rather than a model that is uniformly 1 km off. Note also the definitional ambiguity: MSE can be per
coordinate or per squared Euclidean distance, which differ by a factor of 2.

What the pipeline does: adapters must return de-normalised WGS84 degrees; every spatial error is computed
by the pipeline as haversine metres on masked points only. It reports median and 90th-percentile ADE next
to the mean so heavy tails are visible, and flags values above plausibility bounds. The `TargetGuard` sets
hidden positions to NaN before the adapter sees them, so padding or masked values can neither leak in nor
silently be scored.

### 1.2 TrajGPT's continuous outputs have unit and estimator issues

A travel-time MAE of 138.8 hours is very likely a unit mismatch, but two other effects can inflate it even
with correct units. First, the point estimate is the mixture *mean*; for a skewed (log-normal-like) mixture
the mean is pulled far into the tail, and the MAE-optimal point is the median. In the synthetic smoke run,
switching from mean to median reduced the MAE of a log-normal head from 624 to 152 minutes on identical
predictions. Second, "travel time" between consecutive staypoints includes unobserved periods (overnight,
phone off), so its distribution has an extreme tail that dominates any mean-based error.

The duration NLL of −5.575 implies densities above 1, which happens when the target is on a small scale
(hours, normalised, or log-time). NLL is not unit-invariant: changing seconds to minutes shifts it by
log 60 ≈ 4.09 nats. A negative NLL is therefore not wrong, but it cannot be compared with anything unless
the unit and space are fixed.

What the pipeline does: adapters return mixtures in seconds, declaring whether the mixture is over the
value or over log(value); the pipeline converts to minutes with the correct Jacobian, uses the mixture
median as the point forecast, and reports CRPS, NLL, 80 % interval coverage and a PIT calibration statistic.

### 1.3 Scale and direction conventions

UniTraj's next-location accuracy was logged as 66.0 / 92.0 (percent) while the rest used fractions, and
the recovery metrics were tagged "higher is better". Both are logging bugs that should be impossible by
construction. In `mobeval`, direction, unit, valid range and skill formula are defined once in
`metrics/registry.py`; records outside the valid range (e.g. accuracy 66.0) or above plausibility bounds
(e.g. travel-time MAE above 24 h) are flagged in the report instead of silently normalised.

### 1.4 The uniform 0.05 baseline, and the relative-improvement formula itself

The placeholder baseline already invalidates the relative improvements, but the formula (b − v)/|b| has
problems even with real baselines. For NLL it is meaningless, because NLL can be negative and shifts with
units. For bounded scores it compresses gains near the top (going from 0.90 to 0.95 halves the error but
reads as +5.6 %). The pipeline replaces it with skill scores against baselines scored on the same samples:

| Metric type | Skill score | Interpretation |
|---|---|---|
| Errors ≥ 0 (m, min, CRPS) | 1 − v / b | Share of baseline error removed |
| Scores in [0, 1] (accuracy, F1) | (v − b) / (1 − b) | Share of the gap to perfect closed |
| Log scores (NLL) | b − v | Nats per sample (a log-likelihood ratio), not a percentage |
| Divergences (JSD, W1) | (b − v) / (b − f) | 1 = as close as real data is to real data (noise floor f) |

Baselines per task: time-aware linear interpolation and last-observed for recovery; first-order Markov,
user-frequency and global popularity (fitted on train) for next location; the train-set marginal
distribution (log-space GMM, median as point) for travel time and duration; majority class and a
gradient-boosted classifier on speed, acceleration and turning features for transport mode; a
uniform-location generator (lower) and held-out real trajectories (noise floor) for generation.

### 1.5 Transport-mode results cannot currently be distinguished

UniTraj 0.506 versus My Model 0.505 is a 0.001 difference with no uncertainty estimate. On GeoLife-style
labels the class distribution is strongly imbalanced, so plain accuracy can be close to the majority-class
rate. Recommended: report balanced accuracy and macro-F1, a majority-class and a handcrafted-feature
baseline on identical windows, bootstrap confidence intervals, and results over several training seeds.
Top-5 accuracy is trivially 1.0 with 5 classes and should be dropped for this task.

### 1.6 Table 1 compares unrelated label spaces

TrajGPT's 0.308 top-1 over roughly 2,500 H3 regions and UniTraj's 0.660 next-location top-1 come from
different vocabularies, and accuracy depends heavily on cell size. The pipeline defines one shared metric
grid; models with their own tokeniser return token scores plus token centroids, and point predictors
return coordinates. Everything is mapped onto the same grid, and distance error (top-1 location to true
visit, in metres) is always reported alongside accuracy because it does not depend on cell size.

### 1.7 Generation: only one model, and no reference points

A JSD of 0.234 on radius of gyration is uninterpretable without a noise floor (how different two real
samples of the same size are) and a lower reference. Bins must be fixed from real data so every model is
histogrammed identically, and "Spearman correlation" needs a definition; the pipeline uses a per-user
paired rank correlation of the statistic when generated trajectories are conditioned on users.
Distributional metrics also cannot distinguish a good generator from one that copies training data: the
reference generator in this package, which jitters training trips, scores near the noise floor on every
distribution. The pipeline therefore adds a copy rate (share of generated trajectories closer to a training
trajectory than 95 % of real test trajectories are).

## 2. Protocol implemented in `mobeval`

**One source of truth for data.** A single raw GPS table is split once (chronologically 8:1:1 following
UniTE, or by user for generalisation to unseen people). GPS windows, staypoints (detected by the pipeline
with fixed thresholds) and visit sequences are all derived from it. Visit-sequence targets belong to one
split while context may use the user's earlier visits, which avoids empty test sets under chronological
splits without letting targets cross splits.

**Shared inputs.** Masks (random and block, several ratios, endpoints observed) are generated once per
seed and passed to every model. Label subsets for few-shot classification are stratified and seeded, and
baselines are trained on the same subset.

**Evaluation protocols.** Following UniTE's training strategies, representation quality is measured by
linear probing of frozen embeddings, and native task heads by the adapter's `prepare` hook, at label
fractions such as 100 %, 10 % and 1 %. Pre-training benefit ("w/o pretrain" versus "full" in UniTE) needs
each model's training code and plugs into the same hook.

**Tasks and metrics.** Recovery: ADE, median and p90 ADE, RMSE, FDE per masked block, length-normalised DTW
on masked points, accuracy within 100 m and 500 m, shared-grid accuracy. Next location: acc@1, acc@5,
MRR@20, NLL, distance error, accuracy within 1 km. Travel time and duration: MAE, RMSE, CRPS, NLL (or
pseudo-NLL for point models), coverage and PIT. Mode: accuracy, balanced accuracy, macro-F1, NLL, ECE.
Generation: JSD and W1 for radius of gyration, jump length, stay duration and daily locations, JSD of
visited cells, paired Spearman, copy rate. Efficiency: parameters and latency per sample.

**Why CRPS is the cross-model bridge.** CRPS equals absolute error for a point forecast and has a closed
form for Gaussian mixtures, so TrajGPT's probabilistic head and a point predictor are scored on the same
scale in minutes with a proper scoring rule. The proposed Gaussian pseudo-NLL is kept but with the variance
fitted on validation residuals; fitting it on the evaluated batch chooses the best variance after seeing
the errors and is optimistic.

**Travel-time gaps.** Gaps longer than 4 h between consecutive staypoints are treated as missing data (phone off, overnight) rather than travel and are excluded from the travel-time task for all models and baselines (`travel_time_max_h`, the same rule TrajGPT uses in its own metrics); the task name records the threshold.

**Uncertainty.** Every metric has a bootstrap CI over evaluation samples, and every skill score a paired
bootstrap CI (model and baseline resampled on the same indices). Results are recorded per evaluation seed
and per training run tag; reports show the mean with the spread across runs.

**Aggregation.** Cross-family summaries take the median of clipped skill scores over a non-redundant set of
headline metrics, never raw values. Efficiency is shown as a Pareto front over skill, parameters and
latency rather than a single efficiency score, since collapsing them requires arbitrary weights.

## 3. Remaining recommendations outside the code

Report dataset, split, sampling interval and preprocessing for every model, and evaluate all models on the
same dataset; pre-training corpora should be checked for overlap with the test period. Match the shared grid
cell size to the models' token resolution (H3 resolution 8 has edges of roughly 460 m) and report at a
second resolution to show sensitivity. Where a model was pre-trained elsewhere, use its own normalisation
statistics in the adapter. Add a cross-city transfer setting (train on one dataset, evaluate on another)
once a second dataset is available; generalisation is one of the open problems the STFM survey highlights.
