# Model integrations: what was wrapped, verified and changed

All three networks were re-implemented inside `mobeval/nn/` with unchanged parameter names, so
existing checkpoints load, and checked against the original code: with identical weights and
inputs the outputs match exactly (maximum absolute difference 0.0). Everything below that deviates
from the original repositories is deliberate and listed with its reason.

## UniTraj (`type: unitraj`)

Source: github.com/Yasoz/UniTraj (Apache-2.0). Dependencies on `timm` and `einops` were removed.

**Conventions reproduced.** (longitude, latitude) channel order; offsets from the first visible point
in degrees; z-normalisation with the pre-training statistics (the public checkpoint's are built in);
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
   own Chengdu sample at d_model 128, which is why its settings use a batch size of 16. A masked sum
   over the context axis is exactly a matrix product of the 0/1 mask with the embedding matrix, so
   mobeval computes it that way, in chunks of `context_chunk` (default 4096) entries. Results match
   the original to float32 rounding (3.6e-07) at any chunk size, with about 128x less memory.
5. *POI and road-network features are optional.* Without them the parameter shapes are unchanged and
   the two context pathways contribute only their token embedding. Supply them per adapter with
   `context: {poi_embed: pois.npy, poi_latlon: poi_latlon.npy, road_embed: ..., road_latlon: ...}`,
   where the embeddings are (N, d) arrays and the coordinates (N, 2) arrays of (lat, lon); mobeval
   projects them with the same projection as the trajectories. Note the original compares SQUARED
   distances against `poi_dist`/`rn_dist`, so the default of 100 means a 10 m radius; `mobeval context`
   prints a value suited to the density of your area.

**Where the features come from.** The original ships 64-d embeddings for Chengdu and Xi'an: one row
per POI (12,439 of them, nearly all distinct, so text embeddings of the POI name and category) and one
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

**Sanity check.** Overfitting 16 windows drives recovery error from 2,190 m to 161 m, so the
encode/decode path is sound; short CPU runs on small data remain far from converged (about 1.5 km
after 40 epochs with a 1-layer, 32-dimensional model), as expected.

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
| UniTraj | recovery | next location, travel time, duration, user identification, anomaly detection |
| TransferTraj | recovery | next location, travel time, duration, user identification, anomaly detection |
| TrajGPT | next location, travel time, duration, generation | — (declares no embedding) |
| CLIPMobility | recovery, next location, travel time, duration | user identification, anomaly detection |

A model is never given a head it does not have. The probe is linear, the encoder is frozen, and
every probe result is tagged `protocol: linear_probe` in `results.jsonl` and named
`<task>/linear_probe` in the report, so it cannot be confused with a native capability. See the
README for the candidate-set accounting on the location probe and for the baselines that make
user identification and anomaly detection interpretable.
