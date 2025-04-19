
export PYTHONPATH=/home/jie.zhang/dsv3-debug/MoTS:$PYTHONPATH
# export PYTHONPATH=/home/jie.zhang/dev/dsv3-debug/vllm_gpu/vllm:$PYTHONPATH
# export PYTHONPATH=/home/jie.zhang/dev/dsv3-debug/vllm_gcu/vllm:$PYTHONPATH
# 定义合法IP列表
# valid_ips=("10.9.113.132" "10.9.113.134" "10.9.113.135" "10.9.113.192")       # A组机器
# valid_ips=("10.9.113.137" "10.9.113.190" "10.9.113.191" "10.9.113.193")       # B组机器
# valid_ips=("10.9.113.190" "10.9.113.191")       # B组机器
# valid_ips=("10.12.116.186" "10.12.116.187")       # C组机器
# valid_ips=("10.12.116.186" "10.12.116.187" "10.12.116.188" "10.12.116.189")       # C组机器
# valid_ips=("10.12.116.186" "10.12.116.189")       # C组机器
valid_ips=("10.12.116.216" "10.12.116.247" "10.12.116.250" "10.12.116.252") # 内部机器
# valid_ips=("10.9.112.84" "10.9.112.95" "10.9.113.194" "10.9.113.195")       # D组机器
dp_size=$((${#valid_ips[@]} * 2))
 
VISIBLE_DEVICES=("0,1,2,3" "4,5,6,7")
 
max_num_batched_tokens=32768
 
# log_folder="./logs/logs_offline/offline_412_layer6_1"
# log_folder="./logs/logs_offline/offline_412_layer6_2"
# log_folder="./logs/logs_offline/offline_412_layer6_1_graph"
# log_folder="./logs/logs_offline/offline_412_layer6_2_graph"
# log_folder="./logs/logs_offline/offline_412_layer61_1_graph"
# log_folder="./logs/logs_offline/offline_412_layer61_2"
log_folder="./logs/logs_offline/offline_412_layer6_2_graph"
mkdir -p "$log_folder"
 
# 获取本机IP地址和网卡名
HOST_IP=$(hostname -I | awk '{print $1}')
INTERFACE_NAME=$(ifconfig -a | grep -B1 "inet ${HOST_IP}[^0-9]" | head -n 1 | awk '{print $1}' | cut -d: -f1)
echo ${HOST_IP}
echo ${INTERFACE_NAME}
 
# 检查当前IP是否在合法列表中
valid=false
host_index=-1
for idx in "${!valid_ips[@]}"; do
    if [[ "$HOST_IP" == "${valid_ips[idx]}" ]]; then
        valid=true
        host_index=$idx
        break
    fi
done
 
if ! $valid; then
    echo "Error: Unsupported IP address $HOST_IP"
    exit 1
fi
 
# for i in $(seq 0 1); do
#     rank=$((host_index * 2 + i))
#     echo rank: ${rank}
 
#     ECCL_SHM_DISABLE=1 \
#     ECCL_ALLTOALLV_MAXSIZE=$((4 * dp_size *(7168+16+16)*2)) \
#     TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
#     GLOO_SOCKET_IFNAME=${INTERFACE_NAME} \
#     TOPS_VISIBLE_DEVICES=${VISIBLE_DEVICES[i]} \
#     VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1 \
#     VLLM_USE_V1=0 \
#     VLLM_DP_MASTER_IP=${valid_ips[0]} \
#     VLLM_DP_MASTER_PORT=54333 \
#     VLLM_DP_SIZE=${dp_size} \
#     VLLM_DP_RANK=${rank} \
#     python3.10 dsv3_offline.py \
#     --model /home/pretrained_models/DeepSeek-R1-awq/ \
#     --max_model_len=${max_num_batched_tokens} \
#     --quantization='moe_wna16_gcu' \
#     --dtype='bfloat16' \
#     --max_num_seqs=1 \
#     --enforce_eager \
#     --enable-expert-parallel \
#     --gpu_memory_utilization=0.9 &> ${log_folder}/rank_${rank}_${HOST_IP}.log &
# done

for i in $(seq 0 1); do
    rank=$((host_index * 2 + i))
    echo rank: ${rank}
 
    ECCL_SHM_DISABLE=1 \
    ECCL_ALLTOALLV_MAXSIZE=$((4 * dp_size *(7168+16+16)*2)) \
    TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
    GLOO_SOCKET_IFNAME=${INTERFACE_NAME} \
    TOPS_VISIBLE_DEVICES=${VISIBLE_DEVICES[i]} \
    VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1 \
    VLLM_USE_V1=0 \
    VLLM_DP_MASTER_IP=${valid_ips[0]} \
    VLLM_DP_MASTER_PORT=54333 \
    VLLM_DP_SIZE=${dp_size} \
    VLLM_DP_RANK=${rank} \
    python3.10 dsv3_offline.py \
    --model /home/pretrained_models/DeepSeek-R1-awq/ \
    --max_model_len=${max_num_batched_tokens} \
    --quantization='moe_wna16_gcu' \
    --dtype='bfloat16' \
    --max_num_seqs=1 \
    --enable-expert-parallel \
    --gpu_memory_utilization=0.9 &> ${log_folder}/rank_${rank}_${HOST_IP}.log &
done

# for i in $(seq 0 1); do
#     rank=$((host_index * 2 + i))
#     echo rank: ${rank}
 
#     ECCL_SHM_DISABLE=1 \
#     ECCL_ALLTOALLV_MAXSIZE=$((4 * dp_size *(7168+16+16)*2)) \
#     TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
#     GLOO_SOCKET_IFNAME=${INTERFACE_NAME} \
#     TOPS_VISIBLE_DEVICES=${VISIBLE_DEVICES[i]} \
#     VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1 \
#     VLLM_USE_V1=0 \
#     VLLM_DP_MASTER_IP=${valid_ips[0]} \
#     VLLM_DP_MASTER_PORT=54666 \
#     VLLM_DP_SIZE=${dp_size} \
#     VLLM_DP_RANK=${rank} \
#     python3.10 dsv3_offline.py \
#     --model /home/jenkins/inference/scorpio/vllm/deepseek_r1_awq_int4/DeepSeek-R1-awq/ \
#     --max_model_len=${max_num_batched_tokens} \
#     --quantization='moe_wna16_gcu' \
#     --dtype='bfloat16' \
#     --max_num_seqs=1 \
#     --enforce_eager \
#     --enable-expert-parallel \
#     --gpu_memory_utilization=0.7 &> ${log_folder}/rank_${rank}_${HOST_IP}.log &
# done