#!/bin/bash

echo "=== 清理 CUDA/Ray 资源 (无需 root) ==="

# 1. 杀死所有相关进程
echo ">>> 1. 终止相关进程..."
ps aux | grep "$USER" | grep -E "(verl.trainer|ray::|vllm)" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
sleep 2

# 2. 清理 Ray
echo ">>> 2. 清理 Ray..."
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/* 2>/dev/null || true
rm -rf /tmp/ray_* 2>/dev/null || true
rm -rf ~/ray_results/* 2>/dev/null || true

# 3. 清理共享内存
echo ">>> 3. 清理共享内存..."
rm -rf /dev/shm/torch_* 2>/dev/null || true
rm -rf /dev/shm/vllm_* 2>/dev/null || true
rm -rf /dev/shm/ray_* 2>/dev/null || true

# 4. 等待 GPU 进程自然结束
echo ">>> 4. 等待 CUDA 上下文清理..."
sleep 3

# 5. 验证
echo ">>> 5. GPU 状态："
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

echo ""
echo "=== 清理完成！==="

