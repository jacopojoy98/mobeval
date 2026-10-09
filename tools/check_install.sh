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
check jobs/env.sh                      'cp -r -u'              "jobs sharing RESULTS_DIR keep each other's checkpoints"
check jobs/env.sh                      'cp -r -p'              "restore keeps the newer of the two checkpoint copies"
[ $ok = 1 ] && echo "all present" || echo "some files are older than the rest: see the MISSING lines"
