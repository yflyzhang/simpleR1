# Train via command line

model_name_or_path=Qwen/Qwen2.5-1.5B
# model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B-Instruct
# model_name_or_path=Qwen/Qwen3-1.7B

# model_name_or_path=meta-llama/Llama-3.2-1B
# model_name_or_path=meta-llama/Llama-3.2-1B-Instruct

model_name_or_path=Qwen/Qwen2.5-3B
# model_name_or_path=Qwen/Qwen2.5-3B-Instruct

# model_name_or_path=Qwen/Qwen3-1.7B

# model_name_or_path=Qwen/Qwen3-4B-Base
# model_name_or_path=Qwen/Qwen3-4B

# Trained model
# model_name_or_path=outputs/models/Qwen2.5-3B_data-hendrycks-MATH-benchmark_date-2025-06-23


model_name=$(basename $model_name_or_path)
# run_name=$model_name-$(date +%Y-%m-%d)
# run_name=${model_name}_data-$(basename $eval_dataset)_date-$(date +%Y-%m-%d)
run_name=${model_name}_eval_date-$(date +%Y-%m-%d)


OUTPUT_DIR=outputs/models/$run_name
LOG_FILE="$OUTPUT_DIR/train_log_$(date +%Y-%m-%d_%H:%M:%S.log)"

mkdir -p $OUTPUT_DIR

echo "current file is: $0"
cp "$0" "$OUTPUT_DIR"/run.sh


echo
echo "==================== Training: ===================="
echo "[INFO] run name: $run_name"
echo "[INFO] logs are saved to: $LOG_FILE"
echo

# sleep 7h


MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/sgnfsdata/tolo-02-95/yafei/.cache/huggingface
# export HF_HOME=/xxx/.cache/huggingface
# Set `HF_HOME` when necessary, by default HF_HOME=~/.cache/huggingface
# where `~` is user's home directory.

accelerate launch \
    --main_process_port $MASTER_PORT \
    --config_file configs/accelerate_configs/ddp.yaml \
    --num_processes=2 \
src/run_grpo.py \
    --do_eval True \
    --config configs/grpo_config.yaml \
    --output_dir $OUTPUT_DIR \
    --check_gpu_idle True \
    --model_name_or_path model_name_or_path=Qwen/Qwen2.5-3B \
    --eval_dataset_name HuggingFaceH4/MATH-500 \
    --num_eval_generations 1 \
    --per_device_eval_batch_size 128 \
    --num_test_samples_per_dataset -1 \
    --max_eval_completion_length 4096 \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.75 \
    --reward_funcs accuracy format tag \
    --reward_weights 8 1 1 \
    --mask_truncated_completions True \
    --eval_temperature 0.7 \
    --eval_top_p 0.95 \
    --log_level info \
    --wandb_project simpleR1-eval \
    --run_name $run_name \
    2>&1 | tee $LOG_FILE



# accelerate launch \
#     --main_process_port $MASTER_PORT \
#     --config_file configs/accelerate_configs/zero3.yaml \
#     --num_processes=2 \
# src/run_grpo.py \
#     --do_eval True \
#     --config configs/grpo_config.yaml \
#     --output_dir $OUTPUT_DIR \
#     --check_gpu_idle True \
#     --model_name_or_path $model_name_or_path \
#     --eval_dataset_name $eval_dataset \
#     --num_eval_generations 1 \
#     --per_device_eval_batch_size 128 \
#     --num_test_samples_per_dataset -1 \
#     --max_eval_completion_length 4096 \
#     --use_vllm True \
#     --vllm_mode server \
#     --vllm_server_host 0.0.0.0 \
#     --vllm_server_port 8000 \
#     --reward_funcs accuracy format tag \
#     --reward_weights 8 1 1 \
#     --mask_truncated_completions True \
#     --eval_temperature 0.7 \
#     --eval_top_p 0.95 \
#     --log_level info \
#     --wandb_project simpleR1-eval \
#     --run_name $run_name \
#     2>&1 | tee $LOG_FILE


