# Train via command line

model_name_or_path=Qwen/Qwen2.5-1.5B
# model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B-Instruct
# model_name_or_path=Qwen/Qwen3-1.7B

# model_name_or_path=meta-llama/Llama-3.2-1B
# model_name_or_path=meta-llama/Llama-3.2-1B-Instruct

# model_name_or_path=Qwen/Qwen2.5-3B
# model_name_or_path=Qwen/Qwen2.5-3B-Instruct

# model_name_or_path=Qwen/Qwen3-4B

# train_dataset=openai/gsm8k
train_dataset=nlile/hendrycks-MATH-benchmark
# train_dataset=meta-math/MetaMathQA
# train_dataset=SynthLabsAI/Big-Math-RL-Verified
# train_dataset=hiyouga/math12k
# train_dataset=gneubig/aime-1983-2024
# train_dataset=open-r1/OpenR1-Math-220k
# train_dataset=agentica-org/DeepScaleR-Preview-Dataset
# train_dataset=RUC-AIBOX/STILL-3-Preview-RL-Data
# train_dataset=Maxwell-Jia/AIME_2024


eval_dataset=HuggingFaceH4/MATH-500
# eval_dataset=opencompass/AIME2025


model_name=$(basename $model_name_or_path)
# run_name=$model_name-$(date +%Y-%m-%d)
# run_name=${model_name}_data-$(basename $train_dataset)_date-$(date +%Y-%m-%d)
run_name=${model_name}_date-$(date +%Y-%m-%d)


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
    --do_train True \
    --config configs/grpo_config.yaml \
    --output_dir $OUTPUT_DIR \
    --check_gpu_idle True \
    --model_name_or_path $model_name_or_path \
    --train_dataset_name $train_dataset \
    --eval_dataset_name $eval_dataset \
    --num_train_epochs 1 \
    --num_generations 10 \
    --num_eval_generations 1 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 128 \
    --dynamic_sampling True \
    --max_resample_attempts 3 \
    --gradient_accumulation_steps 1 \
    --num_iterations 3 \
    --torch_empty_cache_steps 1 \
    --num_train_samples_per_dataset 2000 \
    --num_test_samples_per_dataset -1 \
    --max_completion_length 2048 \
    --max_eval_completion_length 4096 \
    --use_vllm True \
    --vllm_mode server \
    --vllm_server_host 0.0.0.0 \
    --vllm_server_port 8000 \
    --reward_funcs accuracy format tag \
    --reward_weights 8 1 1 \
    --loss_type bnpo \
    --scale_rewards False \
    --mask_truncated_completions True \
    --epsilon 0.2 \
    --epsilon_high 0.3 \
    --temperature 1.0 \
    --top_p 0.95 \
    --eval_temperature 0.7 \
    --eval_top_p 0.95 \
    --repetition_penalty 1.02 \
    --beta 1e-5 \
    --lr_scheduler_type constant \
    --learning_rate 5e-6 \
    --save_strategy steps \
    --save_steps 100 \
    --eval_strategy steps \
    --eval_steps 10 \
    --eval_on_start True \
    --log_level info \
    --wandb_project simpleR1-train \
    --run_name $run_name \
    2>&1 | tee $LOG_FILE





