# Shared environment for all mobeval PBS jobs. Edit this file once; the job scripts source it.
# ---------------------------------------------------------------------------------------------
# Where the code, the data and the results live (home or a project folder, NOT scratch)
# The jobs are submitted from the mobeval folder (`qsub jobs/...`), so that folder is the default.
MOBEVAL_DIR="${MOBEVAL_DIR:-${PBS_O_WORKDIR:-$PWD}}"
# Override per job, e.g. for a paper reproduction:
#   qsub -v CONFIG=$PWD/examples/configs/paper_trajgpt.yaml,RESULTS_DIR=$HOME/results_trajgpt jobs/all_in_one.pbs
CONFIG="${CONFIG:-$MOBEVAL_DIR/examples/configs/vehicle_panel.yaml}"
# The data are read from the paths in the config (dataset.path / train_path / test_path). DATA_DIR is
# only where a RELATIVE dataset path in the config is looked up. Keep every input (datasets, context
# files, road networks, OSM extracts) here or in a project folder, never in /scratch: on the daneel
# nodes /scratch is a local disk, so a file left there by one job is not visible to the next one.
DATA_DIR="${DATA_DIR:-$HOME/data/data_by_fua_final}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/results}"   # final outputs are copied back here

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

# Copy every input file the config names to this job's scratch directory and rewrite the config
# to point at the copies: dataset.path / train_path / val_path / test_path, and any other absolute
# path that exists anywhere in the config (a model's context: files, roads_file, a checkpoint to
# start from, ...). Only those files are copied, not the folder around them. On the daneel nodes
# /scratch is a disk local to each node, so nothing is read from /scratch in place: data kept
# there by an earlier job is on whichever node ran it. Keep inputs in home or a project folder.
# Output locations (output_dir, persist_dir, progress_dir, checkpoint_dir, a model's checkpoint
# or train.out) are never copied: they are where results go, not inputs.
# A model's train.options.context / roads_file is moved to its adapter: section (see the end of the
# script below), so a later job loads the checkpoint with its own copies of those files.
stage_in() {
    mkdir -p "$SCRATCH/data" "$SCRATCH/results"
    python - "$CONFIG" "$SCRATCH" "$DATA_DIR" > "$SCRATCH/config.yaml" <<'PY'
import os, shutil, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])); scratch, data_dir = sys.argv[2], sys.argv[3]
node = os.uname().nodename

def expand(n):                                       # $USER, $HOME, ~ in paths: YAML does not expand them
    for k, v in (n.items() if isinstance(n, dict) else enumerate(n)):
        if isinstance(v, (dict, list)):
            expand(v)
        elif isinstance(v, str) and ("$" in v or v.startswith("~")):
            n[k] = os.path.expanduser(os.path.expandvars(v))
expand(cfg)
OUTPUT_KEYS = {"output_dir", "persist_dir", "progress_dir", "checkpoint_dir", "checkpoint", "out"}
copied = {}                                          # realpath of a source -> its copy in scratch

def stage(src, where):
    real = os.path.realpath(src)
    if real.startswith("/scratch/") and not real.startswith(scratch + "/"):
        print(f"stage_in: WARNING {where} = {src} is on /scratch, which is local to the node that wrote "
              f"it; move it to home or a project folder", file=sys.stderr)
    if real in copied:
        return copied[real]
    name = os.path.basename(src.rstrip("/"))
    dst, i = os.path.join(scratch, "data", name), 1
    while os.path.exists(dst):                       # two different files with the same name
        dst, i = os.path.join(scratch, "data", f"{i}_{name}"), i + 1
    shutil.copytree(src, dst) if os.path.isdir(src) else shutil.copy2(src, dst)
    copied[real] = dst
    print(f"stage_in: {where} {src} -> {dst}", file=sys.stderr)
    return dst

cfg["output_dir"] = f"{scratch}/results"
d = cfg["dataset"]
for key in ("path", "train_path", "val_path", "test_path"):
    src = d.get(key)
    if not src:
        continue
    if not os.path.isabs(src):                      # relative: looked up in DATA_DIR by file name
        src = os.path.join(data_dir, os.path.basename(src.rstrip("/")))
    if not os.path.exists(src):
        sys.exit(f"stage_in: dataset.{key} = {src} does not exist on {node}")
    d[key] = stage(src, f"dataset.{key}")

def walk(node_, where):                              # every other absolute path that exists
    items = node_.items() if isinstance(node_, dict) else enumerate(node_)
    for k, v in items:
        here = f"{where}.{k}" if where else str(k)
        if isinstance(v, (dict, list)):
            walk(v, here)
        elif (isinstance(v, str) and k not in OUTPUT_KEYS and v.startswith("/")
              and not v.startswith(scratch + "/")):         # not one of the copies made above
            if os.path.exists(v):
                node_[k] = stage(v, here)
            elif v.startswith("/scratch/"):
                sys.exit(f"stage_in: {here} = {v} does not exist on {node}. /scratch is local to each "
                         f"daneel node; keep inputs in home or a project folder")
walk(cfg, "")

# A checkpoint records the data files its model was trained with (TransferTraj's context files,
# OmniTraj's roads_file) by path, and that path is now this job's scratch copy, gone when the job
# ends. A later job (evaluate.pbs after train_model.pbs, or a RESUME) must therefore be told where
# ITS copies are: options under `adapter:` are passed both to training and to loading a checkpoint,
# where they replace the recorded path. So these two are moved from train.options to adapter.
for m in cfg.get("models", []):
    opts = (m.get("train") or {}).get("options") or {}
    for key in ("context", "roads_file"):
        if key in opts:
            m.setdefault("adapter", {}).setdefault(key, opts[key])
            del opts[key]
yaml.safe_dump(cfg, sys.stdout, sort_keys=False)
PY
}

# Copy results (including checkpoints) back; scratch can be wiped at any time
stage_out() {
    mkdir -p "$RESULTS_DIR"
    # -u: never put back an older file over a newer one. Two jobs may share RESULTS_DIR (e.g. the two
    # OmniTraj variants trained side by side): each restored the other's checkpoint at its start, and
    # a plain copy here would overwrite the other job's newer checkpoint with that stale copy.
    cp -r -u "$SCRATCH/results/." "$RESULTS_DIR/" 2>/dev/null || true
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
        # -p keeps the modification times, so mobeval's own restore (from $MOBEVAL_PERSIST_DIR, where
        # every improving epoch is copied at once) can tell which of the two copies is newer
        cp -r -p "$RESULTS_DIR/checkpoints/." "$SCRATCH/results/checkpoints/"
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
