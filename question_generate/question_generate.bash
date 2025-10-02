# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
export VLLM_DISABLE_COMPILE_CACHE=1
CUDA_VISIBLE_DEVICES=0 python question_generate/question_generate.py --model $model_name --suffix 0 --num_samples $num_samples --save_name $save_name &
CUDA_VISIBLE_DEVICES=1 python question_generate/question_generate.py --model $model_name --suffix 1 --num_samples $num_samples --save_name $save_name &
CUDA_VISIBLE_DEVICES=2 python question_generate/question_generate.py --model $model_name --suffix 2 --num_samples $num_samples --save_name $save_name &
CUDA_VISIBLE_DEVICES=3 python question_generate/question_generate.py --model $model_name --suffix 3 --num_samples $num_samples --save_name $save_name &
CUDA_VISIBLE_DEVICES=4 python question_generate/question_generate.py --model $model_name --suffix 4 --num_samples $num_samples --save_name $save_name &
CUDA_VISIBLE_DEVICES=5 python question_generate/question_generate.py --model $model_name --suffix 5 --num_samples $num_samples --save_name $save_name &
CUDA_VISIBLE_DEVICES=6 python question_generate/question_generate.py --model $model_name --suffix 6 --num_samples $num_samples --save_name $save_name &
CUDA_VISIBLE_DEVICES=7 python question_generate/question_generate.py --model $model_name --suffix 7 --num_samples $num_samples --save_name $save_name &

wait
