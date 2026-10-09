# Model integrations: what was wrapped, verified and changed

All three networks were re-implemented inside `mobeval/nn/` with unchanged parameter names, so
existing checkpoints load, and checked against the original code: with identical weights and
inputs the outputs match exactly (maximum absolute difference 0.0). Everything below that deviates
from the original repositories is deliberate and listed with its reason.

The *networks* match. mobeval's default training settings do not: they differ from the papers
in masking schedules, optimisers, batch sizes and epochs. Each model section ends with a list of
those differences. `recipe: paper` and `recipe: code` reproduce the original training exactly,
apart from the deliberate omissions listed by `mobeval recipes` (see the README).

## UniTraj (`type: unitraj`)

Source: github.com/Yasoz/UniTraj (Apache-2.0). Dependencies on `timm` and `einops` were removed.

**Conventions reproduced.** (longitude, latitude) channel order; offsets in degrees from a reference
point (the original subtracts `trajectory[0]` whether or not that point is masked, which hands the
model the true position of a masked first point; mobeval uses the first *visible* point, which is the
same point whenever the first point is visible); z-normalisation with the pre-training statistics (the public checkpoint's are built in);
time intervals in seconds; fixed length 200 with patch size 1.

**Changes.** Windows shorter than 200 points are right-padded and the padding positions are always
hidden from the encoder, so meaningless zero tokens are never visible. Masking uses an explicit random
generator instead of global NumPy state, making evaluation reproducible. When retraining from scratch,
normalisation statistics are fitted on the train windows and stored in the checkpoint; when continuing
from existing weights they are kept.

**Verification.** On UniTraj's own WorldTrace sample (1 s sampling), the public checkpoint reconstructs
50 % randomly masked points with a mean error of about 52 m. On synthetic data with a very different
sampling rate and spatial extent it reached 1.8 km; 150 CPU steps of continued pre-training with
`init_from` reduced that to 0.7 km. Zero-shot results on data unlike WorldTrace should therefore be
read as a transfer test, and a fine-tuned variant reported alongside.

**Default training vs the paper** (`utils/dataset.py`, `main.py`, paper Table 5). All of the
following is reproduced by `recipe: paper` / `code`, which uses `nn/unitraj_sampling.py` (RDP
verified identical to the `rdp` package):
- Masking: the original mixes four strategies per trajectory (random 70 %, RDP key points 15 %,
  a 5-15-point block 5 %, the last 3-8 points 10 %) at ratio 0.5, and the first and last points
  can be masked. mobeval trains with its own random/block masks and always keeps the endpoints.
- Resampling: the original applies ATR resampling (bin averaging with probability 0.3 for L >= 360,
  otherwise a length-dependent keep ratio). mobeval does not.
- Optimisation: Adam, lr 1e-3, no weight decay, plateau schedule (factor 0.5, patience 2), batch
  1024, 200 epochs in the paper (1000 in the repository), early-stopping patience 20, no gradient
  clipping. mobeval's defaults are AdamW with weight decay, batch 128, 50 epochs and clipping.
- For 64-point windows padded to 200 the hidden/visible counts are 168/32, not the original 100/100.
- The repository's training loop has a `break` that ends every epoch after one batch.
- A per-row check now rejects masks with unequal visible counts per row, which the encoder's
  fixed-length gather cannot represent.

## TrajGPT (`type: trajgpt`)

Source: github.com/ktxlh/TrajGPT (MIT).

**Target leakage in the original time heads.** The visit embedding is concatenated as
[location, arrival, departure, region], but the travel-time decoder reads the first two blocks of the
*target* visit and the duration decoder the first three. The travel head therefore sees the target's
arrival time (travel = arrival − previous departure) and the duration head its arrival and departure
(duration = departure − arrival). The code comments and the paper's factorisation
p(region)·p(travel | region)·p(duration | region, travel) indicate the intended order
[region, location, arrival, departure], which is `input_order: fixed`, the default for training. A test
shows the legacy duration head responds to the target's departure time while the fixed one does not.
This leak very likely explains the strongly negative duration NLL reported earlier.

**Checkpoint fidelity.** The original `PositionalEncoding` adds in place (`x += pe`) to tensor views,
which silently also shifts positional encodings into the decoder targets and into the encoder memory
used by the time heads. `input_order: legacy` reproduces both side effects exactly, so original
checkpoints behave identically (`TrajGPTAdapter.from_original_state_dict`, which needs the H3 region
list, scales and reference time from the original preprocessing).

Loading was broken until this revision and is now verified. State dicts saved by the original code
at three commits load with `strict=True`: HEAD, the paper-era 49aad40, and 2d47f78. Region, travel
and duration outputs match the original modules exactly (0.0). Two quirks of the original had to be
reproduced:
- main.py passes `num_regions + 4` and the modules add 4 again, so the embedding has nr+8 rows and
  the head nr+8 outputs at HEAD but nr+4 at 49aad40. The loader reads both sizes from the state dict
  and scores only the real regions.
- The commits differ in ways the weights cannot reveal, so `revision=` must name the code that
  trained the checkpoint. A wrong value loads without error.
  - `"49aad40"` (paper era): Space2Vec scales are g^(s/S - 1) instead of g^(s/(S-1)), and times
    are fed to Time2Vec in hours.
  - `"2d47f78"` (2d47f78 and b9f1ae2): times in days, and travel time predicted in days, so it is
    converted.
  - `"cf959ca"` (the default, alias `"HEAD"`): times in days, both targets in hours.

  HEAD's own training loop reshapes the nr+8 head to nr+4 and cannot run, so a working HEAD
  checkpoint is unlikely to exist.

**Training fixes** (each was necessary; without them validation loss diverged and region accuracy
stayed at chance level):

1. `time_reference: week`. The original feeds absolute days since the dataset start into Time2Vec.
   Under a chronological split, validation and test days lie outside the training range and the linear
   Time2Vec component extrapolates. Times are now measured from the Monday 00:00 before each sequence,
   which keeps time-of-day and day-of-week phase but stays bounded. `global` reproduces the original.
2. Travel gaps above 4 h are excluded from the travel loss. TrajGPT's own metrics already treat them as
   missing spans; the pipeline's travel-time task applies the same rule to every model and baseline
   (`EvalConfig.travel_time_max_h`).
3. Minimum mixture scale of 1 minute (the original floor of 1e-6 h lets a single out-of-range value
   dominate the validation loss).
4. Mixture heads are initialised at the training data's quantiles and spread. With durations of tens to
   hundreds of hours against an initial location of about 0, the NLL gradients were so large that, after
   clipping, the region cross-entropy on the shared encoder barely moved.
5. Split hygiene. A visit context can contain visits from another split when the splits interleave
   in time (for example user-defined splits). Those visits stay visible as history but are excluded
   from the training loss (`ctx_in_split`). Before this fix they were trained on, so TrajGPT
   checkpoints trained on interleaving splits should be retrained.
6. Durations are clipped at the TRAIN 99th percentile. The original clips at the 99th percentile of
   all splits, which reads the test data. Travel times are not clipped: gaps over 4 h are masked out.

**Context length.** mobeval's `visit_context` defines the task for every model ("predict the next visit
given the last C visits"), and TrajGPT is trained teacher-forced on exactly that shape, so one sample
yields C prediction targets. The original repository instead trains on sequences of up to
RAW_SEQ_LEN = 128 visits and evaluates at the last position. The difference is not the mask size -
`sequence_len` only dimensions the pre-computed causal masks, no parameter depends on it, and the masks
are non-persistent so checkpoints are portable across values (it is kept at >= 128 for that reason).
The difference is `visit_context`, which is worth raising on dense data: a larger context gives every
model more history AND gives TrajGPT more targets per forward pass. Combined with `visit_stride` it
also shrinks the epoch dramatically - with 5.9M staypoints, context 8 / stride 1 gives 5.65M samples
and 45M targets per epoch, whereas context 32 / stride 32 gives 184k samples and the same 5.9M targets
in 30x fewer steps. Checkpoints record the context they were trained with, and evaluating at a
different one warns.

**Evaluation.** Location predictions are region-token scores mapped onto the shared grid through region
centroids. When the target location is hidden, travel-time and duration distributions marginalise over
the top-5 predicted regions (the result is still a Gaussian mixture); for duration the unknown arrival is
set to the last departure plus the median predicted travel time. Outputs are in hours (linear space) and
converted to seconds by the adapter. Regions use a metric grid by default (no extra dependency) or H3
(`options: {tokenizer: {backend: h3, h3_resolution: 7}}`); unseen test cells map to the nearest known
region.

**Default training vs the paper** (paper-era commit 49aad40 and the paper's GeoLife settings;
reproduced by `recipe: paper` / `code` with `examples/configs/paper_trajgpt.yaml`):
- Staypoints: trackintel with 100 m / 5 min in the code, 200 m / 10 min in the paper. mobeval uses
  200 m / 20 min.
- Regions: H3 resolution 7, with the vocabulary built over all splits. mobeval's default is a 1 km
  grid built on train.
- Sequences: up to RAW_SEQ_LEN = 128 visits, left-padded; the infilling task uses p = 0.2 and
  SEQ_LEN 384.
- Loss and model: the three losses weighted 1:1:1, 3 GMM components, 2 layers / 8 heads /
  feed-forward 32, dropout 0.1.
- Optimisation: Adam, lr 1e-4, batch 64, up to 2000 epochs with early-stopping patience 50 in the
  49aad40 code (the paper states patience 10 and seed 0); HEAD later switched to Adafactor.
- The paper's headline task is visit infilling (Table 3), which mobeval does not implement; mobeval
  evaluates next-visit prediction, the paper's second task.

## CLIP mobility model (`type: clip_mobility`)

Source: `Model/Models/clip_mobility_model.py` and `mobility_transformer_vector.py`.

**Token layouts.** The original tokenizer produces semantic, coordinate-free point tokens (speed and turn
bins, time of day, road type, POI density, land-use diversity, network centrality) that require OSM data.
Without coordinates the model cannot be scored on recovery, and the original layouts are not recorded in
checkpoints, so mobeval defines its own documented layouts (`nn/features.py`). The trajectory view has
local offsets in km, time step, speed, heading, time of day and weekday; the visit view has position,
arrival and departure time of day, duration, travel time and weekday. OSM or other features can be
appended with `extra_point_features` / `extra_dim`; they must be zero or computable for masked points.
Checkpoints from the original scripts are therefore not loadable: retrain with `pretrain`.

**Pairing.** Each GPS window is paired with the same user's last `visit_context` staypoints that ended
before the window started (tested), so the visit view never contains future information. Shorter histories
are padded on the right: with a causal mask, left padding leaves positions with nothing to attend to, which
PyTorch's fast inference path turns into NaN (this produced `val nan` in an earlier version). Point-token
features are clipped to physical bounds so single GPS glitches cannot create extreme inputs.

**Contrastive loss.** The original training script uses unpaired visit sequences in half of the batches
with a negative CLIP weight. That rewards driving the contrastive cross-entropy towards infinity and
makes the objective unbounded below. Standard symmetric InfoNCE already uses the other pairs in the batch
as negatives; it is used here, with the logit scale clamped as in CLIP.

**Capabilities.** Recovery uses the native next-token head autoregressively: masked points are filled left
to right from the points before them. The causal model cannot use points after a gap, a structural
disadvantage against bidirectional models such as UniTraj and against interpolation. Next location and
the two time tasks use heads on the frozen visit encoder's last-token state, trained on the train visit
sequences when first needed; the embedding task uses the normalised CLIP embedding.

## TransferTraj (`type: transfertraj`)

Source: github.com/wtl52656/TransferTraj. Verified against the original implementation: identical
weights and inputs give identical hidden states, spatial predictions and loss (maximum absolute
difference 0.0). The einops dependency was removed.

**Conventions reproduced.** Metre coordinates relative to each trajectory's first point (the absolute
first point is passed separately, since the POI/road lookup needs it); the four features
[x, y, timestamp, delta_t], each paired with a token channel (known / mask / pad); causal attention
whose rotary embedding is driven by the coordinates; the mixture-of-experts encoder layer; and the
pre-training objective of span masking plus per-point single-modality masking, with the original
spatial + temporal + token loss.

**Deviations, all recorded in the checkpoint.**
1. *Projection.* The original converts to UTM with pyproj, choosing the zone by city name. mobeval uses
   its own local metric projection centred on the training data: no extra dependency, still metres,
   and sub-percent distortion at city scale.
2. *coord_scale* (default 1000, i.e. kilometres). The original feeds raw metres, so the spatial MSE
   term starts around 1e5 and dominates the gradient. On synthetic data, raw metres moved the
   validation loss from 6.17M to 6.14M over six epochs while the scaled version went from 1,868 to
   1,145. The architecture is unchanged; set `coord_scale: 1.0` to reproduce the original exactly, and
   keep 1.0 when loading original checkpoints.
3. *Deterministic inference.* The mixture-of-experts router adds Gaussian routing noise on every
   forward pass in the original, including evaluation, so repeated runs of the same checkpoint give
   different predictions. Noisy top-k gating is a training-time regulariser (Shazeer et al.), so the
   noise is applied only in training mode here; set `NoisyTopkRouter.noise_in_eval = True` to restore
   the original behaviour. Numerical equivalence with the original was verified with the noise active.
4. *Memory.* To average the POI embeddings near each point, the original materialises a
   (batch, length, n_context, d_model) tensor — 6.5 GB for 16 x 64 points against the 12k POIs of its
   own Chengdu sample at d_model 128. A masked sum
   over the context axis is exactly a matrix product of the 0/1 mask with the embedding matrix, so
   mobeval computes it that way, in chunks of `context_chunk` (default 4096) entries. Results match
   the original to float32 rounding (3.6e-07) at any chunk size, with about 128x less memory.
5. *POI and road-network features are optional.* Without them the parameter shapes are unchanged and
   the two context pathways contribute only their token embedding. Supply them per adapter with
   `context: {poi_embed: pois.npy, poi_latlon: poi_latlon.npy, road_embed: ..., road_latlon: ...}`,
   where the embeddings are (N, d) arrays and the coordinates (N, 2) arrays of (lat, lon); mobeval
   projects them with the same projection as the trajectories. `poi_dist`/`rn_dist` are thresholds on
   SQUARED distance in metres², as in the original: its default of 100 is a 10 m radius, whereas the
   paper describes a 100 m neighbourhood (10,000). A bug made the threshold wrong whenever
   `coord_scale` was not 1: with the default 1000 it was compared with squared kilometres, so almost
   every POI in the city counted as nearby (12,439 of 12,439 on the Chengdu sample, instead of about
   7.5 at 100 m). This is fixed; TransferTraj checkpoints trained with context features before the fix
   should be retrained. `mobeval context` prints a value suited to the density of your area.

**Where the features come from.** The original ships 64-d embeddings for Chengdu and Xi'an: one row
per POI (12,439 of them, nearly all distinct; the paper describes text embeddings of the POI name, type
and address) and one
per road segment (4,315 rows but only 1,410 distinct, so segments of the same street share a vector).
Neither the embedding model nor a builder is included, and both cities are Chinese, so for any other
region the features have to be rebuilt. `mobeval context` does that from OpenStreetMap: POIs from the
usual tags (amenity, shop, tourism, leisure, office, public_transport), road sample points from the
drivable network at a fixed spacing, and one-hot category embeddings that need no model download. See
the README for the command and `mobeval/context_features.py` for the readers if you prefer to supply
your own vectors.

**Capabilities.** Recovery (spatial features masked, timestamps kept, one forward pass, as in the
repository's TRec padder), embeddings (mean over the encoder states), and mode classification through
the shared frozen-embedding head. Its trajectory-prediction and travel-time tasks are point-level and
have no counterpart among mobeval's visit-level tasks, so they are not exposed; the pipeline lists
them as skipped capabilities rather than silently scoring something different.

**Default training vs the paper** (reproduced by `recipe: paper` / `code` with
`examples/configs/paper_transfertraj.yaml`, including the per-task fine-tuning via
`options: {objective: tp | trec}`):
- Masking: the repository cuts each trajectory into ceil(0.2 L) spans, fully masks 40 % of them and
  masks one modality of 20 % of the points. The paper describes a single masked span, with the
  remaining points masked 50/50 spatially or temporally.
- Data: three-hop resampling (>= 6 s), trips of 5-120 points, and a chronological 8:1:1 split.
- Model: 8 experts, top-4 routing, embed_size 64, d_model 128, 2 layers.
- Optimisation: Adam, lr 1e-3, batch 64, 30 epochs, no validation or early stopping (the last
  epoch is kept); fine-tuning with StepLR(5, 0.5). The paper fine-tunes on each task before testing
  it; mobeval's default evaluation corresponds to its "w/o ft" variant.
- The paper's recovery masks both modalities of the hidden points; the repository's TRec padder (and
  mobeval) masks only the spatial part.
- Evaluation: recovery keeps every 8th point plus the last one (paper: mu = 4/8/16 epsilon);
  prediction targets the last 5 points; OD travel time is reported as MAE/RMSE in minutes plus
  MAPE, averaged over 5 repeats, including zero- and few-shot transfer across cities.

**Sanity check.** Overfitting 16 windows drives recovery error from 2,190 m to 161 m, so the
encode/decode path is sound; short CPU runs on small data remain far from converged (about 1.5 km
after 40 epochs with a 1-layer, 32-dimensional model), as expected.

## OmniTraj (`type: omnitraj`)

Source: github.com/Yasoz/OmniTraj (KDD 2025; no licence file, used with the authors' code as
published). The network (four encoders, fusion layers, loss) is **vendored unmodified** in
`mobeval/nn/omnitraj/`, so it is the original by construction. It needs `timm`, `einops` and
`transformers`. No pretrained weights were released.

**Preprocessing, reconstructed.** The repository ships no preprocessing code, only 1,000
preprocessed Chengdu trips. `nn/omnitraj_prep.py` rebuilds each step from the paper and that
sample, and states how well it matches:

| step | source | agreement with the sample |
|---|---|---|
| 200 points by "cubic spline" | paper Sec. 3.2.1 | PCHIP (piecewise-cubic Hermite) over the point index, inferred. A quarter of the sample's steps are exactly 0 at stops, which arc-length parametrisation, scipy's CubicSpline (overshoot ripples) and linear-over-index all fail to produce |
| topology = RDP, epsilon 1e-4 degrees, on the 200 resampled points | inferred | **exact for 1000/1000 trips** |
| regions: 16 x 16 square cells, id = 1 + lon_index x 16 + lat_index | paper B.1 + fit | **99.9 %** of 200,000 cell ids |
| roads: one map-matched segment id per original point, deduplicated in order | paper Def. 3 + sample | matcher not stated in the paper |
| z-normalised (lon, lat); padding, BOS/EOS, truncation, augmentations | repository `utils/dataset.py` | line by line |

Road and region ids are compacted to the training vocabulary: the original feeds raw ids and notes
that segment 0 collides with padding. On a large area, only the most frequent `max_regions`
cells keep their own id.

**Paper vs code.** The released training differs from the paper's text, and the recipes follow
each side:

| | paper | code |
|---|---|---|
| loss | InfoNCE on cosine similarity, both directions (Eqs. 9-10) | cross-entropy against soft targets built from within-modality similarities, on unnormalised projections, averaged over directions |
| contrast pairs | trajectory with each modality (Eq. 10) | trajectory-topology, topology-road, topology-region |
| optimiser | Adam, lr 2e-4 (App. A) | AdamW, lr 2e-4, weight decay 1e-4 |

Where the paper is silent, both recipes use the code's settings: cosine schedule over 500 epochs
(eta_min 1e-5), batch 1536, gradient clipping 1.0, learnable temperature initialised at 1.0, and
best model by validation loss. Both contrast the trajectory against each of the four fusions. The original augments the
validation set as well (its dataset settings are shared); `augment_val` reproduces that.

**Memory.** A training step at batch 1536 holds about 73 GB of activations, more than any single
GPU. Training therefore checkpoints every encoder layer (`gradient_checkpointing`, on by default):
each layer is recomputed in the backward pass with the same dropout draws, so the gradients are
identical, and the step needs about 10 GB, at roughly 35% more time. It changes nothing about
the model or the recipe; `train: {options: {gradient_checkpointing: false}}` turns it off for small batches.

**Evaluation.** `embed()` returns the L2-normalised projection of the GPS trajectory, which is the
retrieval space the model is trained for. Windows are spline-resampled to 200 points like trips.
`embed_query(batch, "topology" | "road" | "region" | "region+topology" | ...)` embeds the other
representations in the same space. For fusions it uses the repository's `get_embeddings` path:
each modality normalised, fused, then normalised again. Training feeds the fusion layers
unnormalised projections, so fused queries see inputs of a different scale than in training; this
is the original's behaviour and is kept. The retrieval task (README) uses these
for the paper's Table 2 (cross-modal retrieval: MR, MRR, HR@k) and Table 3 (condition-based
retrieval: CR@1/5).

**Scale.** OmniTraj is a city model: the Chengdu and Xi'an grids are about 9 km across. For a
national panel, restrict the run to one city with `prepare: {bbox: ...}`
(`examples/configs/omnitraj_city.yaml`). Two recipe values then no longer mean what they meant in
the paper:
- **Grid:** 16 x 16 cells over a larger city gives larger cells. Set `grid_cell_m: 571` to keep
  the paper's cell size instead of its cell count.
- **Normalisation:** the mean and std are fitted on the city's training trips.

**Roads.** `mobeval roads` builds segments between intersections from an OpenStreetMap extract.
`mobeval mapmatch` then assigns one segment per GPS point, using the built-in HMM matcher (Newson
& Krumm) or FMM output (DATASETS.md). Without a `roads_file` the road encoder is left out, and
the model and its retrieval queries use trajectory, topology and regions only. This is logged and
stored in the checkpoint.

**Not reproducible:** the paper's numbers. The Chengdu and Xi'an datasets (1.2M trips each) are no
longer available, so every ☆ mark on OmniTraj is a comparison of method, not of data.

**Training speed.** 500 epochs over a city's trips is long, and three things shorten it without
changing what is computed:
- *Preprocessing once.* Resampling, topology (RDP), region and road ids depend only on the trip, so
  they are computed once before training, in parallel over the job's CPUs (`prep_workers`, default
  `$NCPUS`); each step then only augments and pads. The batches are identical to rebuilding every
  sample each step (tested), and this is how the original is organised: preprocessed trips on disk,
  augmented by the dataset class. Before, rebuilding the samples took about 3 ms per trip per epoch
  on one CPU core, which left the GPU idle most of the time.
- *Mixed precision* (`train: {amp: true}`): the encoders run in float16 on a V100 (bfloat16 on A100
  and newer) with loss scaling; the weights and the contrastive loss stay float32. The paper trained
  on an A100, whose PyTorch 1.8 defaults already ran matrix products in TF32. Recorded in the
  checkpoint's provenance (`compute`).
- *Two GPUs* (`train: {options: {gpus: 2}}`, and `ngpus=2` in the PBS request): the four encoders
  split each batch, and their outputs are gathered before the loss, which therefore still contrasts
  every trip with all 1,535 others. DataParallel or DDP over the whole model would compute the loss
  per GPU, halving the negatives - a different objective for a contrastive model.

Each run logs, for the first epoch and every 25th, how many seconds went into building batches and
how many into the network (`batch_seconds` in the checkpoint history).

## Shared components

**Mode-classification head.** The native head is trained on frozen embeddings and re-fitted for every
label fraction and seed. By default it is linear and unweighted, matching the pipeline's logistic
regression probe, so the two protocols differ only in the optimiser. A class-weighted head trades
accuracy for balanced accuracy (on synthetic data: accuracy 0.65 vs 0.83, balanced accuracy 0.80 vs
0.45); enable it with `head_class_weighted: true` if balanced metrics are the priority, and report it.

**Training loop** (`nn/common.py`): AdamW, plateau learning-rate schedule, gradient clipping, early
stopping on validation loss with restoration of the best weights, and checkpoints that store the
architecture, model-specific metadata (normalisation, vocabulary, scales), the training history and the
split fingerprint.

## What this means for the earlier results

The previous "My Model" recovery results came from `Trajectory_transformer` with a randomly initialised
reconstruction head (its script never loaded weights), so they carry no information about the model.
The TrajGPT duration NLL was produced by a head that could see the answer. Neither should be reported.


## What each model can be asked for

| model | native | via linear probe on frozen embeddings |
|---|---|---|
| UniTraj | recovery | next location, travel time, duration, user identification, anomaly detection, generation (rollout) |
| TransferTraj | recovery | next location, travel time, duration, user identification, anomaly detection, generation (rollout) |
| TrajGPT | next location, travel time, duration, generation | — (declares no embedding) |
| CLIPMobility | recovery, next location, travel time, duration | user identification, anomaly detection, generation (rollout) |
| OmniTraj | cross-modal and condition-based retrieval | next location, travel time, duration, user identification, anomaly detection (no decoder: no recovery or generation) |

Every model with embeddings also takes part in similarity retrieval (`retrieval/odd_even`).

A model is never given a head it does not have. The probe is linear, the encoder is frozen, and
every probe result is tagged `protocol: linear_probe` in `results.jsonl` and named
`<task>/linear_probe` in the report, so it cannot be confused with a native capability. See the
README for the candidate-set accounting on the location probe and for the baselines that make
user identification and anomaly detection interpretable.
