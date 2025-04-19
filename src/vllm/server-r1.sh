#!/bin/bash
# start_service.sh


pkill -9 python
export PYTHONPATH=/home/dakuang.shen/vfarm-share/framwork/vllm-gpu:$PYTHONPATH

# MODEL_PATH="/datasets/deepseek-v3-awq"
MODEL_PATH="/topsmodels/data-llm/qwen/Qwen2-0.5B"
PORT=8089


## 元宝提供
# python -m vllm.entrypoints.openai.api_server \
#     --model "/topsmodels/data-llm/qwen/Qwen2-0.5B" \
#     --trust-remote-code \
#     --host 0.0.0.0 \
#     --port 8089


nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 6144 \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    > vllm_service.log 2>&1 &

