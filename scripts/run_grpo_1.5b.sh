# Train via command line

# model_name_or_path=Qwen/Qwen2.5-1.5B
# model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct
# model_name_or_path=Qwen/Qwen2.5-Math-1.5B-Instruct
model_name_or_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

dataset=HuggingFaceH4/MATH-500
# dataset=gneubig/aime-1983-2024
# dataset=open-r1/OpenR1-Math-220k
# dataset=agentica-org/DeepScaleR-Preview-Dataset
# dataset=RUC-AIBOX/STILL-3-Preview-RL-Data


model_name=$(basename $model_name_or_path)
run_name=$model_name-$(date +%Y-%m-%d)
# run_name=$model_name-data_$(basename $dataset)-$(date +%Y-%m-%d)
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

# sleep 5.1h


MASTER_PORT=$(shuf -n 1 -i 10000-65535)

CUDA_VISIBLE_DEVICES=0,1,2 \
HF_HOME=/mnt/sgnfsdata/tolo-02-95/yafei/.cache/huggingface \
accelerate launch \
    --main_process_port $MASTER_PORT \
    --config_file configs/accelerate_configs/zero1.yaml \
    --num_processes=2 \
src/run_grpo.py \
    --config configs/grpo_template.yaml \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $model_name_or_path \
    --dataset_name $dataset \
    --vllm_gpu_memory_utilization 0.75 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 6 \
    --per_device_train_batch_size 5 \
    --num_generations 5 \
    --num_iterations 4 \
    --torch_empty_cache_steps 1 \
    --num_train_samples 1000 \
    --max_completion_length 3200 \
    --top_p 0.95 \
    --temperature 1.0 \
    --beta 0.05 \
    --learning_rate 5e-5 \
    --save_strategy epoch \
    --log_level debug \
    --wandb_project simpleR1 \
    --run_name $run_name \
    &> $LOG_FILE
    

