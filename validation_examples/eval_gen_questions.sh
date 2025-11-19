#!/bin/bash

set -x

set -euo pipefail

experiment_name=$1
model_path=$2

export PYTHONUNBUFFERED=1


export WANDB_MODE=offline
export WANDB_DIR=yourpath/vr-zero/wandb
export STORAGE_PATH="yourpath/vr-zero/storage"

# DATASETS=(
#   "yourpath/datasets/mm-vet/test.parquet"
#   "yourpath/datasets/MLLM_test/test.parquet"
#   "yourpath/datasets/visnumbench/data/test-00000-of-00001.parquet"
#   "yourpath/datasets/mmmu_pro_10options/data/test-00000-of-00001.parquet"
#   "yourpath/datasets/mmmu-pro-vision/data/test-00000-of-00001.parquet"
#   "yourpath/datasets/hallusionbench/data/test-00000-of-00001.parquet"
#   "yourpath/datasets/MMMU/data/test-00000-of-00001.parquet"
# )

DATASETS=(
  "yourpath/datasets/MLLM_test/test.parquet"
  "yourpath/datasets/visnumbench/data/test-00000-of-00001.parquet"
  "yourpath/datasets/hallusionbench/data/test-00000-of-00001.parquet"
  "yourpath/datasets/MMMU/data/test-00000-of-00001.parquet"
)

# ------------------------------------------------------------------
# STATIC pieces of the command line (everything that never changes)
# ------------------------------------------------------------------
BASE_CMD="python3 -m verl.trainer.main \
  config=validation_examples/eval_config.yaml \
  data.train_files=yourpath/datasets/geometry3k/data/test-00000-of-00001.parquet \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.image_key=images \
  worker.actor.model.model_path=${model_path} \
  worker.rollout.max_model_len=25600 \
  worker.rollout.n=8 \
  trainer.total_epochs=1 \
  trainer.experiment_name=${experiment_name} \
  trainer.save_checkpoint_path=./Evaluation/Raw-Outputs \
  trainer.n_gpus_per_node=8 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.global_batch_size=8 \
  data.format_prompt=./train_examples/format_prompt/solver.jinja \
  trainer.val_only=true \
  trainer.logger=[console]"

# ------------------------------------------------------------------
# LOOP over datasets
# ------------------------------------------------------------------
for DS in "${DATASETS[@]}"; do
  # extract the dataset name from the path
  SHORT_NAME=$(echo "${DS}" | cut -d'/' -f5)

  echo ">>> Evaluating on ${DS}"
  CMD="${BASE_CMD} \
    data.val_files=${DS} \
    trainer.response_path=yourpath/vr-zero/Evaluation/Raw-Outputs/${experiment_name}/${SHORT_NAME}.jsonl"

  # show the command (optional)
  echo "$CMD" | sed 's/  */ /g'
  echo "------------------------------------------------------------"

  # run it
  eval $CMD
  
  # sleep for a few seconds before next dataset (except for the last one)
  sleep 10
done

