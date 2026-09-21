# Shared environment for all mobeval PBS jobs. Edit this file once; the job scripts source it.
# ---------------------------------------------------------------------------------------------
# Where the code, the data and the results live (home or a project folder, NOT scratch)
MOBEVAL_DIR="$HOME/MobFM/BMdir/mobeval"
CONFIG="$MOBEVAL_DIR/examples/configs/vehicle_panel.yaml"
DATA_DIR="$HOME/MobFM/data"                 # the CSV/Parquet files referenced by the config
RESULTS_DIR="$HOME/MobFM/results"           # final outputs are copied back here

# Scratch: all job I/O happens here (mandatory on this cluster; not backed up)
SCRATCH="/scratch/$USER/mobeval/${PBS_JOBID:-manual-$$}"   # always a per-job subdirectory

# Live progress. MUST be on a shared filesystem so you can read it from trantor while the job runs
# on a compute node - scratch is local to the node. Inspect with:
#   python -m mobeval status --progress-dir "$RESULTS_DIR/progress" --watch
export MOBEVAL_PROGRESS_DIR="$RESULTS_DIR/progress"

# Durable directory. The job computes in scratch (mandatory) but copies anything finished here
# STRAIGHT AWAY rather than at the end: a checkpoint as soon as its epoch improves, results after
# every task. If the job crashes, hits its walltime, or the node dies, whatever had finished is
# already in $RESULTS_DIR - stage_out below becomes a safety net rather than the only copy.
#   $RESULTS_DIR/checkpoints/      one per model, shared by every run
#   $RESULTS_DIR/runs/<run_id>/    report.md, results.jsonl, leaderboard.csv for that run
#   $RESULTS_DIR/latest ->         the newest run
export MOBEVAL_PERSIST_DIR="$RESULTS_DIR"

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
    mkdir -p "$SCRATCH" "$RESULTS_DIR" "$MOBEVAL_PROGRESS_DIR" "$MOBEVAL_PERSIST_DIR"
    echo "host=$(hostname) job=$PBS_JOBID cpus=${NCPUS:-?} scratch=$SCRATCH"
    echo "progress: python -m mobeval status --progress-dir $MOBEVAL_PROGRESS_DIR --watch"
    echo "results are kept in $MOBEVAL_PERSIST_DIR as they are produced"
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

# Final sync. mobeval already copied checkpoints and results to $MOBEVAL_PERSIST_DIR as it went,
# so this only picks up anything left over - it is no longer the step your results depend on.
stage_out() {
    mkdir -p "$RESULTS_DIR"
    cp -r "$SCRATCH/results/." "$RESULTS_DIR/" 2>/dev/null || true
    echo "results copied to $RESULTS_DIR"
    # Delete ONLY this job's own subdirectory - never /scratch/$USER itself, which the cluster owns
    # and which holds other jobs' data. A failure here must not fail the job: the results are safe,
    # and under `set -e` a non-zero rm would abort the script and break `depend=afterok` chains.
    case "$SCRATCH" in
        /scratch/"$USER"/mobeval/?*)
            rm -rf "$SCRATCH" 2>/dev/null || echo "note: could not fully remove $SCRATCH - clean it up later"
            ;;
        *)
            echo "note: leaving $SCRATCH in place (not a per-job mobeval directory)"
            ;;
    esac
    return 0
}

# Reuse checkpoints from previous jobs so a re-submission does not retrain everything.
# mobeval does this itself from $MOBEVAL_PERSIST_DIR; this stays for jobs that set neither.
restore_checkpoints() {
    if [ -d "$RESULTS_DIR/checkpoints" ]; then
        mkdir -p "$SCRATCH/results/checkpoints"
        cp -r "$RESULTS_DIR/checkpoints/." "$SCRATCH/results/checkpoints/"
        echo "restored checkpoints: $(ls "$SCRATCH/results/checkpoints" | tr '\n' ' ')"
    fi
}

# Trap the walltime kill. PBS sends SIGTERM first; passing it on lets mobeval finish its current
# write, mark the run "interrupted" in the progress file, and print where to resume from.
on_term() {
    echo "received SIGTERM (walltime?) - letting mobeval close down; finished work is in $RESULTS_DIR"
    kill -TERM "$MOBEVAL_PID" 2>/dev/null || true
    wait "$MOBEVAL_PID" 2>/dev/null
    stage_out
    exit 143
}

# Run mobeval so that the trap above can reach it, instead of blocking the shell. Returns
# mobeval's exit status instead of aborting under `set -e`, so the caller can still stage out.
run_mobeval() {
    trap on_term TERM
    python -m mobeval "$@" &
    MOBEVAL_PID=$!
    local status=0
    wait "$MOBEVAL_PID" || status=$?
    trap - TERM
    return $status
}
