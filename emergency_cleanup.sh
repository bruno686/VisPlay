#!/bin/bash

echo "=== 开始紧急清理 CUDA/Ray 资源 ==="

# 1. 杀死所有相关进程
echo ">>> 1. 终止所有相关进程..."
pkill -9 -f "verl.trainer.main" || true
pkill -9 -f "ray::" || true
pkill -9 -f "vllm" || true
pkill -9 -f "python.*verl" || true

sleep 2

# 2. 清理 Ray 临时文件和会话
echo ">>> 2. 清理 Ray 会话..."
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/* 2>/dev/null || true
rm -rf /tmp/ray_* 2>/dev/null || true
rm -rf ~/ray_results/* 2>/dev/null || true

# 3. 清理共享内存（vLLM 的关键资源）
echo ">>> 3. 清理共享内存..."
rm -rf /dev/shm/torch_* 2>/dev/null || true
rm -rf /dev/shm/vllm_* 2>/dev/null || true
rm -rf /dev/shm/ray_* 2>/dev/null || true
ipcs -m | grep $USER | awk '{print $2}' | xargs -I {} ipcrm -m {} 2>/dev/null || true

# 4. 清理 CUDA 僵尸进程
echo ">>> 4. 清理 CUDA 进程..."
nvidia-smi | grep python | awk '{print $5}' | xargs -I {} kill -9 {} 2>/dev/null || true

# 5. 重置所有 GPU（这个很关键！）
echo ">>> 5. 重置 GPU 设备..."
for i in {0..7}; do
    nvidia-smi -i $i --gpu-reset 2>/dev/null || echo "GPU $i 重置失败或不存在"
done

# 6. 等待 GPU 完全空闲
echo ">>> 6. 等待 GPU 完全空闲..."
sleep 3

# 7. 验证清理结果
echo ">>> 7. 验证清理状态..."
echo "=== GPU 使用情况 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

echo ""
echo "=== Ray 进程 ==="
ps aux | grep ray | grep -v grep || echo "无 Ray 进程"

echo ""
echo "=== Python/CUDA 进程 ==="
ps aux | grep python | grep -v grep | grep -E "(verl|vllm|ray)" || echo "无相关 Python 进程"

echo ""
echo "=== 共享内存使用 ==="
ls -lh /dev/shm/ | grep -E "(torch|vllm|ray)" || echo "无相关共享内存文件"

echo ""
echo "=== 清理完成！==="
echo "现在可以重新运行你的脚本了"

