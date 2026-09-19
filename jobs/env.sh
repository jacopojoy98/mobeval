# Shared environment for all mobeval PBS jobs. Edit this file once; the job scripts source it.
# ---------------------------------------------------------------------------------------------
# Where the code, the data and the results live (home or a project folder, NOT scratch)
MOBEVAL_DIR="$HOME/MobFM/BMdir/mobeval"
CONFIG="$MOBEVAL_DIR/examples/configs/vehicle_panel.yaml"
DATA_DIR="$HOME/MobFM/data"                 # the CSV/Parquet files referenced by the config
RESULTS_DIR="$HOME/MobFM/results"           # final outputs are copied back here

# Scratch: all job I/O happens here (mandatory on this cluster; not backed up)
SCRATCH="/scratch/$USER/mobeval/$PBS_JOBID"

# Live progress. MUST be on a shared filesystem so you can read it from trantor while the job runs
# on a compute node - scratch is local to the node. Inspect with:
#   python -m mobeval status --progress-dir "$RESULTS_DIR/progress" --watch
export MOBEVAL_PROGRESS_DIR="$RESULTS_DIR/progress"

# Python environment. Check `module avail` on the cluster for the exact module names;
# if there are no modules, just create the venv with the system python once:
#   python3 -m venv ~/venvs/mobeval
#   ~/venvs/mobeval/bin/pip install -e "$MOBEVAL_DIR[models]"
# module load python/3.11
# module load cuda/12.1
VENV="$HOME/venvs/mobeval"

setup_env() {
    set -euo pipefail
    source "$VENV/bin/activate"
    # PBS reserves NCPUS cores; keep the numeric libraries inside that allocation
    export OMP_NUM_THREADS="${NCPUS:-1}"
    export MKL_NUM_THREADS="${NCPUS:-1}"
    export PYTHONUNBUFFERED=1
    mkdir -p "$SCRATCH" "$RESULTS_DIR" "$MOBEVAL_PROGRESS_DIR"
    echo "host=$(hostname) job=$PBS_JOBID cpus=${NCPUS:-?} scratch=$SCRATCH"
    echo "progress: python -m mobeval status --progress-dir $MOBEVAL_PROGRESS_DIR --watch"
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
}

# Copy inputs to the local scratch disk and rewrite the config to point at them
stage_in() {
    cp -r "$DATA_DIR" "$SCRATCH/data"
    mkdir -p "$SCRATCH/results"
    python - "$CONFIG" "$SCRATCH" > "$SCRATCH/config.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])); scratch = sys.argv[2]
cfg["output_dir"] = f"{scratch}/results"
d = cfg["dataset"]
for key in ("path", "train_path", "val_path", "test_path"):
    if d.get(key):
        d[key] = f"{scratch}/data/" + d[key].rsplit("/", 1)[-1]
yaml.safe_dump(cfg, sys.stdout, sort_keys=False)
PY
}

# Copy results (including checkpoints) back; scratch can be wiped at any time
stage_out() {
    mkdir -p "$RESULTS_DIR"
    cp -r "$SCRATCH/results/." "$RESULTS_DIR/"
    echo "results copied to $RESULTS_DIR"
    rm -rf "$SCRATCH"
}

# Reuse checkpoints from previous jobs so a re-submission does not retrain everything
restore_checkpoints() {
    if [ -d "$RESULTS_DIR/checkpoints" ]; then
        mkdir -p "$SCRATCH/results/checkpoints"
        cp -r "$RESULTS_DIR/checkpoints/." "$SCRATCH/results/checkpoints/"
        echo "restored checkpoints: $(ls "$SCRATCH/results/checkpoints" | tr '\n' ' ')"
    fi
}
