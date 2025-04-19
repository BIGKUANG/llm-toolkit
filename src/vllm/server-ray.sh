#!/bin/bash

# vLLM服务启动脚本（使用Ray）
## --worker-use-ray 在 ​​vLLM 0.9.0+​​ 中正式支持，建议升级：

# 停止现有服务
pkill -9 python
pkill -9 ray

# 设置环境变量
export PYTHONPATH=/home/dakuang.shen/vfarm-share/framwork/vllm-gpu:$PYTHONPATH

# 模型路径和端口配置
MODEL_PATH="/topsmodels/data-llm/qwen/Qwen2-0.5B"
PORT=8089

# Ray启动参数
RAY_NUM_GPUS=1
RAY_NUM_CPUS=8

# 启动Ray集群
nohup ray start --head \
    --num-gpus=$RAY_NUM_GPUS \
    --num-cpus=$RAY_NUM_CPUS \
    --include-dashboard=true \
    > ray.log 2>&1 &

# 等待Ray集群启动
sleep 10

# 使用Ray启动vLLM服务
nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size $RAY_NUM_GPUS \
    --gpu-memory-utilization 0.95 \
    --max-model-len 6144 \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --worker-use-ray \
    > vllm_service_ray.log 2>&1 &

echo "vLLM服务已通过Ray启动，日志文件: vllm_service_ray.log"