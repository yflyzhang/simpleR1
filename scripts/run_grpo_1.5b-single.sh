# Train via command line

# model_name_or_path=Qwen/Qwen2.5-1.5B
model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B-Instruct
# model_name_or_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
model_name_or_path=Qwen/Qwen3-1.7B


train_dataset=openai/gsm8k
# train_dataset=nlile/hendrycks-MATH-benchmark
# train_dataset=meta-math/MetaMathQA
# train_dataset=SynthLabsAI/Big-Math-RL-Verified
# train_dataset=HuggingFaceH4/MATH-500
# train_dataset=hiyouga/math12k
# train_dataset=gneubig/aime-1983-2024
# train_dataset=open-r1/OpenR1-Math-220k
# train_dataset=agentica-org/DeepScaleR-Preview-Dataset
# train_dataset=RUC-AIBOX/STILL-3-Preview-RL-Data
# train_dataset=Maxwell-Jia/AIME_2024


model_name=$(basename $model_name_or_path)
# run_name=$model_name-$(date +%Y-%m-%d)
run_name=${model_name}_data-$(basename $train_dataset)_date-$(date +%Y-%m-%d)
# run_name=$model_name+


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

# export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/mnt/sgnfsdata/tolo-02-95/yafei/.cache/huggingface

accelerate launch \
    --main_process_port $MASTER_PORT \
    --config_file configs/accelerate_configs/ddp.yaml \
    --num_processes=1 \
src/run_grpo.py \
    --config configs/grpo_config.yaml \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $model_name_or_path \
    --train_dataset_name $train_dataset \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.2 \
    --num_train_epochs 1 \
    --num_generations 12 \
    --max_resample_attempts 3 \
    --gradient_accumulation_steps 3 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --num_iterations 3 \
    --torch_empty_cache_steps 1 \
    --max_num_train_samples 10000 \
    --max_completion_length 1024 \
    --reward_funcs accuracy format tag \
    --reward_weights 8 1 1 \
    --scale_rewards True \
    --mask_truncated_completions True \
    --epsilon 0.2 \
    --epsilon_high 0.3 \
    --top_p 0.95 \
    --temperature 1.0 \
    --beta 0.0 \
    --compute_kl True \
    --learning_rate 1e-6 \
    --save_strategy steps \
    --log_level debug \
    --wandb_project test \
    --run_name $run_name \
    2>&1 | tee $LOG_FILE

    # --wandb_project simpleR1-$(basename $train_dataset) \
# --eval_strategy steps \
#     --eval_on_start True \
    # --seed 95 \
    

