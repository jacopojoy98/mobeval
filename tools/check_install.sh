#!/bin/bash
# Is this copy of mobeval complete? Each line names a feature and the file that must contain it.
# Run from the mobeval folder:  bash tools/check_install.sh
ok=1
check() {   # file, text, what
    if [ -f "$1" ] && grep -q -- "$2" "$1"; then echo "ok       $3"; else echo "MISSING  $3   ($1)"; ok=0; fi
}
check mobeval/loaders.py               "WT_COLUMNS"            "WorldTrace read from Trajectory.zip"
check mobeval/loaders.py               "on_split_overlap"      "split-overlap diagnosis"
check mobeval/config.py                "def expand_paths"      "\$USER / ~ in config paths"
check mobeval/context.py               "def split_view"        "evaluation on the training data"
check mobeval/context.py               "visualize_samples"     "sample figures (config option)"
check mobeval/visualize.py             "def make_samples"      "sample figures (module)"
check mobeval/report.py                "def train_vs_test"     "train-vs-test report section"
check mobeval/adapters/transfertraj.py "def context_coverage"  "TransferTraj context coverage check"
check jobs/env.sh                      "expand(cfg)"           "staging expands \$USER / ~"
check jobs/env.sh                      'for key in ("context", "roads_file")' "staging keeps context / roads paths valid across jobs"
check jobs/mapmatch.pbs                'CITY="${CITY:-milan}"' "map-matching job defaults"
check tools/worldtrace_subset.py       "zip_index"             "WorldTrace subset tool"
check mobeval/nn/omnitraj_prep.py      "def prepare_units"     "OmniTraj inputs prepared once"
check mobeval/adapters/omnitraj.py     "def parallelize"       "OmniTraj on several GPUs"
check mobeval/adapters/omnitraj.py     "def _checkpointed_class" "gradient checkpointing on several GPUs"
check mobeval/nn/common.py             "def resuming"          "resumed training continues the same run"
check mobeval/nn/common.py             "amp: bool"             "mixed precision (train: amp)"
check mobeval/nn/common.py             "def write_loss_log"    "training log (<checkpoint>.losses.csv every epoch)"
check mobeval/adapters/trajgpt.py      "report_loss_parts"     "training log: TrajGPT loss terms"
check mobeval/adapters/clip_mobility.py "report_loss_parts"    "training log: CLIP-Mobility loss terms"
check mobeval/nn/transfertraj_net.py   "report_loss_parts"     "training log: TransferTraj loss terms"
check mobeval/adapters/omnitraj.py     "report_loss_parts"     "training log: OmniTraj loss per contrasted pair"
check mobeval/adapters/unitraj.py      "log_to=out"            "training log: UniTraj"
check mobeval/nn/common.py             "save_every: int"       "snapshots every N epochs (train: save_every)"
check mobeval/snapshots.py            "def summarize"         "which epoch is best (mobeval snapshots)"
check mobeval/tasks.py                'scored = getattr(ctx, "eval_split"' "generation scored on the split being evaluated"
check jobs/evaluate_snapshots.pbs     "run_mobeval snapshots" "snapshot evaluation job"
check plot_snapshots.py               "def fig_ranking"       "snapshot figures (plot_snapshots.py 40-42)"
check plot_fit.py                      "def fig_terms"         "training figures (plot_fit.py 22-24)"
check jobs/env.sh                      'cp -r -u'              "jobs sharing RESULTS_DIR keep each other's checkpoints"
check jobs/env.sh                      'cp -r -p'              "restore keeps the newer of the two checkpoint copies"
[ $ok = 1 ] && echo "all present" || echo "some files are older than the rest: see the MISSING lines"
