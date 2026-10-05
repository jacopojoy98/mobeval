# The original papers' datasets

Each model's reproduction config (`examples/configs/paper_*.yaml`) reads the dataset its paper
used. mobeval does not download anything. Compute nodes have no internet access, so download
on a **login node** into `/scratch/$USER/data`, then submit the run as a PBS job. Heavy
preprocessing (staypoints, windows) happens inside the job, never on the login node.

| Model | Dataset | Where | Access | Loader |
|---|---|---|---|---|
| UniTraj | WorldTrace (2.45M trajectories, 1 s, 70 countries, ~35 GB) | huggingface.co/datasets/OpenTrace/WorldTrace | open, ODbL 1.0 | `loader: worldtrace` |
| TrajGPT | GeoLife 1.3 (182 users, Beijing, 2007-2012) | microsoft.com/en-us/download/details.aspx?id=52367 | open | `loader: geolife` |
| TransferTraj | DiDi Chengdu / Xi'an (~6 s) | outreach.didichuxing.com (GAIA) | registration | `loader: transfertraj_h5` |
| TransferTraj | Porto taxi (15 s) | kaggle.com/competitions/pkdd-15-predict-taxi-service-trajectory-i | Kaggle account | `loader: porto` |
| OmniTraj | DiDi Chengdu / Xi'an (1.2M trips each, map-matched) | no longer available | - | any city subset: `prepare: {bbox: ...}` |
| CLIP-Mobility | - | no publication | - | - |

## WorldTrace (UniTraj)

    # on a login node
    pip install --user -U "huggingface_hub[cli]"
    huggingface-cli download OpenTrace/WorldTrace --repo-type dataset --local-dir /scratch/$USER/data/WorldTrace

The download is archives of per-trajectory CSV files (`time`, `latitude`, `longitude`, map-matched
columns). Extract them in a PBS job, not on the login node. `load_worldtrace` reads every `*.csv`
below `path`; it also reads UniTraj's own pickle format (`data/worldtrace_sample.pkl` in the
repository), which is a quick way to check the setup.

- The paper trained on a curated 1.1M-trajectory subset that is not released. Use
  `max_trajectories` to draw a random subset: the full set is ~880M points, far more than one
  job's memory.
- WorldTrace has no users, so each trajectory is its own user. User-level tasks are meaningless on
  it, and the UniTraj configs run recovery only.
- The paper evaluates on trajectories resampled to 3 s, while pre-training reads the 1 s data
  through ATR resampling. That is why there are two configs: `paper_unitraj_pretrain.yaml` for
  1 s training and `paper_unitraj_eval.yaml` for evaluation, with `prepare: {min_interval_s: 3}`.

## GeoLife (TrajGPT)

    wget -O geolife.zip "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
    unzip geolife.zip -d /scratch/$USER/data/geolife

`path` is the folder containing `Data/`. TrajGPT's preprocessing, as set in `paper_trajgpt.yaml`:

- trackintel reading, so no speed filter (`max_speed_mps: null`);
- visits from 2007-2008 only (`prepare: {time_from, time_to}`);
- staypoints of 200 m / 10 min (paper) or 100 m / 5 min (the 49aad40 code);
- H3 resolution-7 regions;
- a chronological 8:1:1 split.

## DiDi Chengdu / Xi'an (TransferTraj)

Registration at DiDi's GAIA initiative is required. The TransferTraj repository works on a processed
HDF5 layout (`samples/small_chengdu.h5` shows it):

- `/trips`: trip, seq_i, time, lng, lat;
- `/trip_info`: trip, driver, ...;
- `/pois` and `/road_info`: coordinates of the POIs and road segments, row-aligned with the shipped
  `*_poi_embed.npy` / `*_road_embed.npy`.

`loader: transfertraj_h5` reads that layout. Running

    python -c "from mobeval.loaders import load_transfertraj_h5; load_transfertraj_h5('chengdu.h5', context_out='chengdu_ctx')"

writes `poi_latlon.npy` / `road_latlon.npy` next to it, for the adapter's `context:` option.

The paper's "three-hop resampling" (about 6 s between points) is already applied in these files.
For raw GAIA CSV exports use `loader: csv` with a column mapping plus `prepare: {every_nth: 3}`.
Either way `prepare: {min_traj_points: 5, max_traj_points: 120}` applies the trip-length filter.

The coordinates are in GCJ-02, as distributed, and TransferTraj does not convert them.
OpenStreetMap-based context (`mobeval context`) is in WGS84, so for a DiDi city use the
repository's POI/road embeddings, not rebuilt ones.

## Porto (TransferTraj)

    kaggle competitions download -c pkdd-15-predict-taxi-service-trajectory-i -p $HOME/data/porto
    unzip $HOME/data/porto/*.zip -d $HOME/data/porto

`loader: porto`, `path: .../train.csv`. Points are every 15 s from `TIMESTAMP`, and trips flagged
`MISSING_DATA` are dropped. The TransferTraj code has no UTM zone for Porto, so Porto could not
have been run through the released code as is. mobeval's local projection has no such limit.

## What "same data" does and does not cover

Loaders and `prepare:` reproduce the papers' **selection** of data: sources, date ranges, trip
lengths and resampling. The run's `eval:` section reproduces their **views**: staypoint parameters,
region resolution, splits and window length. `mobeval recipes` lists each paper's settings, and a
model trained with `recipe:` warns about every `eval:` setting that differs from its paper.

Some differences remain. mobeval assigns splits by trajectory or target visit, never by
overlapping instances. It fits vocabularies and clipping on the train split only. It scores
fixed-length windows rather than whole trips. MODELS.md lists these per model.

## OmniTraj: a city subset plus a road network

OmniTraj's datasets (DiDi Chengdu and Xi'an as processed by the authors: 200-point trips, topology,
map-matched roads, 16 x 16 regions) are no longer downloadable. `examples/configs/omnitraj_city.yaml`
runs it on one city of your own data instead. Every other model is evaluated on the same subset,
so the table stays like-for-like.

1. **OpenStreetMap extract.** On a login node:
   `wget -P $HOME/data/osm https://download.geofabrik.de/europe/italy-latest.osm.pbf`
   Geofabrik also has regional extracts (e.g. `europe/italy/centro-latest.osm.pbf`), which are
   smaller.
2. **Road network and map matching**, as a CPU job: `jobs/mapmatch.pbs`. It computes in the job's
   scratch directory and copies `network.npz` and `matched.csv.gz` to `ROADS_DIR`, which must be in
   home or a project folder: on the daneel nodes `/scratch` is a local disk, so a training job on
   another node would not find files left there.
   - `mobeval roads` cuts the drivable ways (motorway to residential, links, living streets) at
     intersections, which is the paper's Definition 3. One-way streets keep their direction.
   - `mobeval mapmatch` gives every GPS point of the configured dataset (after `prepare:`) a
     segment id, with the built-in HMM matcher. Its parameters are `--sigma-m` (GPS noise,
     default 20 m) and `--radius-m` (candidate radius, 60 m).
   - It is pure Python, parallel over `--workers`, and takes roughly 0.1-0.5 s per trip: a city's
     100k trips need a few hours on 32 cores.
3. **FMM instead of the built-in matcher.** FMM (github.com/cyang-kth/fmm) is much faster, which
   helps for many cities or full coverage:
   - `mobeval roads ... --fmm net.gpkg` writes the network in FMM's format;
   - `mobeval mapmatch --config c.yaml --fmm-export gps.csv` writes the points;
   - run FMM with `--output_fields opath`;
   - `mobeval mapmatch --config c.yaml --fmm-import out.csv --fmm-network net.gpkg --out matched.csv.gz`
     reads the result back.

   The same config must be used for export and import.
4. Point `train.options.roads_file` at the matched table (`.csv.gz`, or `.parquet` with pyarrow).
   The config's filters must match those used for map matching, because points are joined on
   `(traj_id, t)`.

The paper also drops trips under 20 points (`prepare: {min_traj_points: 20}`) and "trajectories
recorded outside urban areas": `bbox` keeps only trips lying entirely inside the box.
