export WANDB_MODE=offline
export WANDB_DIR=/data/hezhuangzhuang-p/vr-zero/wandb

export STORAGE_PATH="/data/hezhuangzhuang-p/vr-zero/storage"
export HUGGINGFACENAME="bruno888"

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"


Base_model=/data/hezhuangzhuang-p/llm/Qwen2.5-VL-3B-Instruct
Model_abbr=Qwen2.5-VL-3B-Instruct
echo "Model_abbr: $Model_abbr"
# Initialize first iteration with base model
# bash scripts_Qwen-VL-3B/questioner_train.sh  $Base_model $Base_model ${Model_abbr}_questioner_v1
# bash scripts_Qwen-VL-3B/solver_train.sh $Base_model /data/hezhuangzhuang-p/vr-zero/saves/3b_grpo_vrzero/global_step_40/actor/huggingface ${Model_abbr}_solver_v1
# bash scripts_Qwen-VL-3B/solver_train.sh $Base_model ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_20/actor/huggingface ${Model_abbr}_solver_v1

for i in {2..3}; do
    prev=$((i-1))
    
    bash scripts_Qwen-VL-3B/questioner_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_40/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_20/actor/huggingface \
        ${Model_abbr}_questioner_v${i}

    # Train solver
    bash scripts_Qwen-VL-3B/solver_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_40/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_20/actor/huggingface \
        ${Model_abbr}_solver_v${i}
done

# bash evaluation/eval_math.sh $Base_model
