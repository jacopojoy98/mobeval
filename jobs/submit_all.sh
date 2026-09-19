#!/bin/bash
# Submit one training job per model, then an evaluation job that starts only if they all succeed.
# Usage (from the mobeval directory):  bash jobs/submit_all.sh UniTraj-finetuned TrajGPT CLIPMobility
# NOTE: q07daneel allows 2 running jobs per user (q02daneel allows 6 but has a 2-day limit);
# extra jobs simply wait in the queue.
set -euo pipefail
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || { echo "usage: bash jobs/submit_all.sh MODEL [MODEL ...]"; exit 1; }

DEPS=""
for m in "${MODELS[@]}"; do
    id=$(qsub -N "train_$m" -v "MODEL=$m" jobs/train_model.pbs)
    echo "submitted training $m: $id"
    DEPS="${DEPS:+$DEPS:}$id"
done
eval_id=$(qsub -W "depend=afterok:$DEPS" jobs/evaluate.pbs)
echo "submitted evaluation (after the training jobs): $eval_id"
echo "track with: qstat -u $USER   |   cancel with: qdel <job id>"
