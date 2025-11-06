solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

echo $STORAGE_PATH

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
bash question_generate/question_generate.bash $questioner_model_path 1000 $experiment_name
echo 'start evaluate generated question'
bash question_evaluate/evaluate.sh $solver_model_path $experiment_name

# python question_evaluate/upload_hf.py --repo_name ${experiment_name} --max_score 0.8 --min_score 0.3 --experiment_name ${experiment_name} --batch_size 200 --max_samples 1000

echo 'start upload'
python question_evaluate/upload.py --max_score 0.8 --min_score 0.3 --save_name ${experiment_name}
echo 'start train'

python3 -m verl.trainer.main \
    config=train_examples/cot_config.yaml \
    data.max_response_length=4096 \
    data.train_files=${STORAGE_PATH}/local_parquet/${experiment_name}_train.parquet \
    data.val_files=/data/hezhuangzhuang-p/datasets/mmstar/test.parquet \
    data.format_prompt=./train_examples/format_prompt/solver.jinja \
    worker.actor.model.model_path=$solver_model_path \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.max_num_batched_tokens=20000 \
    worker.reward.reward_function=./train_examples/reward_function/cot_val_solver.py:compute_score \
    worker.val_reward.reward_function=./train_examples/reward_function/cot_val_solver.py:compute_score \
    trainer.total_epochs=100 \
    trainer.max_steps=41 \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    trainer.val_before_train=false

sleep 5

echo "merging model"
python scripts_Qwen-VL-3B/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_40/actor

sleep 10

echo "solver training finished"

# bash /data/hezhuangzhuang-p/vr-zero/validation_examples/eval_gen_questions.sh $experiment_name ${STORAGE_PATH}/models/${experiment_name}/