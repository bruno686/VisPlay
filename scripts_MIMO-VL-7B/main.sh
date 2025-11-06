export WANDB_MODE=offline
export WANDB_DIR=/data/hezhuangzhuang-p/vr-zero/wandb

export STORAGE_PATH="/data/hezhuangzhuang-p/vr-zero/storage"
export HUGGINGFACENAME="bruno888"

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"


Base_model=/data/hezhuangzhuang-p/llm/MiMo-VL-7B-SFT/
Model_abbr=MiMo-VL-7B-SFT
echo "Model_abbr: $Model_abbr"
# Initialize first iteration with base model
bash scripts_MIMO-VL-7B/questioner_train.sh  $Base_model $Base_model ${Model_abbr}_questioner_v1
bash scripts_MIMO-VL-7B/solver_train.sh $Base_model ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_20/actor/huggingface ${Model_abbr}_solver_v1

for i in {2..3}; do
    prev=$((i-1))
    
    bash scripts_MIMO-VL-7B/questioner_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_20/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_20/actor/huggingface \
        ${Model_abbr}_questioner_v${i}

    # Train solver
    bash scripts_MIMO-VL-7B/solver_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_20/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_20/actor/huggingface \
        ${Model_abbr}_solver_v${i}
done

# bash evaluation/eval_math.sh $Base_model
