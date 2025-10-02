#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

export WANDB_MODE=offline
export WANDB_DIR=/home/hezhuangzhuang-p/vr-zero/wandb

MODEL_PATH=/data/hezhuangzhuang-p/llm/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=train_examples/cot_config.yaml \
    data.train_files=/data/hezhuangzhuang-p/datasets/Vision-SR1-47K \
    data.val_files=/data/hezhuangzhuang-p/datasets/mmstar/test.parquet \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.max_model_len=8192 \
    worker.rollout.n=8 \
    trainer.total_epochs=1 \
    trainer.experiment_name=qwen2_5_vl_3b_vrzero_grpo \
    trainer.save_checkpoint_path=./saves/3b_grpo_vrzero \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=true \
    trainer.val_only=false