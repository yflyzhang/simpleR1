# 3 * 80 G (2 for training, 1 for rollout)

export HF_HOME=/mnt/sgnfsdata/tolo-02-95/yafei/.cache/huggingface

export CUDA_VISIBLE_DEVICES=2

python src/vllm_serve.py \
    --model Qwen/Qwen2.5-3B \
    --seed 42 \
    --gpu_memory_utilization 0.9


# Remember to set other parameters as needed.
# For example:
# export CUDA_VISIBLE_DEVICES=2,3
# python src/vllm_serve.py \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --gpu_memory_utilization 0.9 \
#     --tensor_parallel_size 2 \
#     --data_parallel_size 1 \
#     --host 0.0.0.0 \
#     --port 8001

