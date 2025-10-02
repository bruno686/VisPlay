model_path=$1
run_id=$2
export VLLM_DISABLE_COMPILE_CACHE=1
CUDA_VISIBLE_DEVICES=4 python vllm_service_init/start_vllm_server.py --port 6000 --model_path $model_path &
CUDA_VISIBLE_DEVICES=5 python vllm_service_init/start_vllm_server.py --port 6001 --model_path $model_path &
CUDA_VISIBLE_DEVICES=6 python vllm_service_init/start_vllm_server.py --port 6002 --model_path $model_path &
CUDA_VISIBLE_DEVICES=7 python vllm_service_init/start_vllm_server.py --port 6003 --model_path $model_path &