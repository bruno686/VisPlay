export WANDB_MODE=offline
export WANDB_DIR=/data/hezhuangzhuang-p/vr-zero/wandb

export STORAGE_PATH="/mnt/shared-storage-user/hezhuangzhuang-p/vr-zero/storage"
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
# bash scripts_MIMO-VL-7B/questioner_train.sh  $Base_model $Base_model ${Model_abbr}_questioner_v1
# bash scripts_MIMO-VL-7B/solver_train.sh $Base_model ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_10/actor/huggingface ${Model_abbr}_solver_v1

# bash scripts_MIMO-VL-7B/solver_train.sh $Base_model $Base_model ${Model_abbr}_solver_v0


for i in {2..3}; do
    prev=$((i-1))
    
    # Check if questioner model already exists
    if [ -d "${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_10/actor/huggingface" ]; then
        echo "Questioner model ${Model_abbr}_questioner_v${i} already exists, skipping training..."
    else
        bash scripts_MIMO-VL-7B/questioner_train.sh \
            ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_10/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_10/actor/huggingface \
            ${Model_abbr}_questioner_v${i}
    fi

    # Check if solver model already exists
    if [ -d "${STORAGE_PATH}/models/${Model_abbr}_solver_v${i}/global_step_10/actor/huggingface" ]; then
        echo "Solver model ${Model_abbr}_solver_v${i} already exists, skipping training..."
    else
        # Train solver
        bash scripts_MIMO-VL-7B/solver_train.sh \
            ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_10/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_10/actor/huggingface \
            ${Model_abbr}_solver_v${i}
    fi
done

# bash evaluation/eval_math.sh $Base_model
