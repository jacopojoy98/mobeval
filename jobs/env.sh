# Shared environment for all mobeval PBS jobs. Edit this file once; the job scripts source it.
# ---------------------------------------------------------------------------------------------
# Where the code, the data and the results live (home or a project folder, NOT scratch)
MOBEVAL_DIR="$HOME/mobeval"
CONFIG="$MOBEVAL_DIR/examples/configs/omnitraj_city.yaml"
DATA_DIR="$HOME/data"                 # the CSV/Parquet files referenced by the config
RESULTS_DIR="$HOME/results/omnitraj_city"           # final outputs are copied back here
export MOBEVAL_PROGRESS_DIR="$RESULTS_DIR/progress"

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
export MOBEVAL_PERSIST_DIR="$RESULTS_DIR/latest"
# Python environment. Check `module avail` on the cluster for the exact module names;
# if there are no modules, just create the venv with the system python once:
#   python3 -m venv ~/venvs/mobeval
module load gcc/10.2.0
module load openssl/1.1.1w
module load python/3.13.11-pytorch
VENV="$HOME/.venv-own"

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
    cp -r "$DATA_DIR" "$SCRATCH"
    mkdir -p "$SCRATCH/results"
    python - "$CONFIG" "$SCRATCH" > "$SCRATCH/config.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])); scratch = sys.argv[2]
cfg["output_dir"] = f"{scratch}/results"
d = cfg["dataset"]
for key in ("path", "train_path", "val_path", "test_path"):
    if d.get(key):
        d[key] = f"{scratch}/data/" + d[key].rsplit("data/", 1)[-1]
yaml.safe_dump(cfg, sys.stdout, sort_keys=False)
PY
}

# Copy results (including checkpoints) back; scratch can be wiped at any time
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

# Reuse checkpoints from previous jobs so a re-submission does not retrain everything
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

# Turn a bare "Killed" into something actionable. A job that exceeds its memory allowance is
# SIGKILLed by the kernel or by PBS: there is no Python traceback, no exception and nothing in
# the progress file, just exit 137 and a one-word message.
explain_exit() {
    if [ "${1:-0}" -eq 137 ]; then
        cat <<'MSG'

=== the job was KILLED (exit 137) ===
That is almost always the memory limit, not a bug in your data. Nothing was raised in Python,
so there is no traceback to look for. Options, cheapest first:

  1. Ask for more memory in the #PBS -l select=... line (e.g. mem=64gb).
  2. Lower the settings that drive peak memory, in the `eval:` block of the config:
       max_eval_samples                 fewer test samples scored at once
       generation_max_real_trajectories fewer trajectories behind the generation reference
       generation_nn_max_train / _query smaller memorisation comparison
       grid_cell_m                      a LARGER cell size means far fewer grid cells
  3. Evaluate one model at a time with --models, so only one model's caches are live.

Whatever had finished is already in $RESULTS_DIR; re-submit with RESUME to continue.
MSG
    fi
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
