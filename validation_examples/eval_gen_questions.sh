#!/bin/bash

set -x

set -euo pipefail

experiment_name=$1
model_path=$2

export PYTHONUNBUFFERED=1


export WANDB_MODE=offline
export WANDB_DIR=/data/hezhuangzhuang-p/vr-zero/wandb
export STORAGE_PATH="/data/hezhuangzhuang-p/vr-zero/storage"

# MODEL_PATH=/data/hezhuangzhuang-p/llm/Qwen2.5-VL-3B-Instruct
# MODEL_PATH=/data/hezhuangzhuang-p/vr-zero/storage/models/Qwen2.5-VL-3B-Instruct_solver_v1/global_step_60/actor/huggingface
# SAVE_PATH=3B-VL-Qwen2.5-solver_1


# DATASETS=(
#   "/data/hezhuangzhuang-p/datasets/mm-vet/test.parquet"
#   "/data/hezhuangzhuang-p/datasets/MLLM_test/test.parquet"
#   "/data/hezhuangzhuang-p/datasets/visnumbench/data/test-00000-of-00001.parquet"
#   "/data/hezhuangzhuang-p/datasets/mmmu_pro_10options/data/test-00000-of-00001.parquet"
#   "/data/hezhuangzhuang-p/datasets/mmmu-pro-vision/data/test-00000-of-00001.parquet"
#   "/data/hezhuangzhuang-p/datasets/hallusionbench/data/test-00000-of-00001.parquet"
#   "/data/hezhuangzhuang-p/datasets/MMMU/data/test-00000-of-00001.parquet"
# )

DATASETS=(
  "/data/hezhuangzhuang-p/datasets/MLLM_test/test.parquet"
  "/data/hezhuangzhuang-p/datasets/visnumbench/data/test-00000-of-00001.parquet"
  "/data/hezhuangzhuang-p/datasets/hallusionbench/data/test-00000-of-00001.parquet"
  "/data/hezhuangzhuang-p/datasets/MMMU/data/test-00000-of-00001.parquet"
)

# ------------------------------------------------------------------
# STATIC pieces of the command line (everything that never changes)
# ------------------------------------------------------------------
BASE_CMD="python3 -m verl.trainer.main \
  config=validation_examples/eval_config.yaml \
  data.train_files=/data/hezhuangzhuang-p/datasets/geometry3k/data/test-00000-of-00001.parquet \
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
    trainer.response_path=/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/${experiment_name}/${SHORT_NAME}.jsonl"

  # show the command (optional)
  echo "$CMD" | sed 's/  */ /g'
  echo "------------------------------------------------------------"

  # run it
  eval $CMD
  
  # sleep for a few seconds before next dataset (except for the last one)
  sleep 10
done

