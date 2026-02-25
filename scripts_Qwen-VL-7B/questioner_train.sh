#!/bin/bash

solver_model_path=$1
questioner_model_path=$2
save_path=$3
echo "save_path: $save_path"
# 生成唯一 RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID

echo "RUN_ID=$RUN_ID"

# 启动 vllm 服务（记录 PID）
bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

# 开始训练 Questioner
echo "Start training questioner: $questioner_model_path -> $save_path"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=train_examples/cot_config.yaml \
    data.train_files=yourpath/datasets/Vision-SR1-47K \
    data.val_files=yourpath/datasets/mmstar/test.parquet \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    worker.actor.model.model_path=$questioner_model_path \
    worker.rollout.max_model_len=8192 \
    worker.rollout.n=8 \
    trainer.max_steps=20  \
    trainer.save_freq=10 \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=10 \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=false \
    trainer.val_only=false

sleep 5

echo "merging model"
python scripts_Qwen-VL-7B/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_20/actor

sleep 10

pkill python

echo "questioner training finished"
