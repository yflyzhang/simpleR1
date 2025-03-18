
# simpleR1: A simple implementation of DeepSeek R1 using GRPO

**simpleR1** is a simple implementation of DeepSeek R1, a large language model designed for reasoning tasks like math and code. This repository builds upon Hugging Face's TRL GRPO Trainer and the [open-r1](https://github.com/huggingface/open-r1) project, with a focus on ease of use and enhanced training features. 


<a src="https://wandb.ai/yflyzhang/DeepSeekR1/reports/SimpleR1-Examples--VmlldzoxMTg1Njc0NQ" style="border:none;height:1024px;width:100%"></a>



## ⚡ Key Features

-  Better multi GRPO iteration support (`num_iterations`).

-  Better progress control with accurate epoch and execution time estimation.

-  Compatible with Hugging Face TRL and open-r1 workflows and scripts.


<!-- 
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/simpleR1.git
   cd simpleR1


## Features
- Built on Hugging Face's TRL library for reinforcement learning.
- Enhanced GRPO trainer with multi-iteration support and time estimation.
 -->


## 📁 Repository Structure


```
├── configs/
│   ├── accelerate_configs/    # Deepspeed configs
│   │   ├── zero2.yamal        # Deepspeed zero2 config
│   │   └── ...                
│   └── grpo_template.yaml     # Template for specifying arguments
│       └── ...     
│           
├── scripts/                   # Bash scripts to run
│   ├── run_grpo_1.5b.sh       # Shell for running a 1.5b model
│   └── ...         
│           
├── src/                       # Python codes
│   ├── arguments.py           # Model, scripts, training arguments
│   ├── rewards.py             # Reward functions
│   ├── grpo_trainer.py        # Trainer for GRPO [core part]
│   ├── run_grpo.py            # Python scripts to run GRPO
│   └── utils.py               # Other supporting functions
│
├── requirements.txt           # Full list of requirements
├── LICENSE
└── README.md                  # This document
```




## 🚀 Usage

Example training command:

```bash
bash scripts/run_grpo_1.5b.sh
```


Or override additional parameters via command line:

```bash
# HF_HOME=/xxx/xxx/.cache/huggingface \
CUDA_VISIBLE_DEVICES=0,1,2 \  # assume we have 3 cards
accelerate launch \
    --main_process_port $MASTER_PORT \
    --config_file configs/accelerate_configs/zero1.yaml \
    --num_processes=2 \       # cuda:2 is reserved for vllm generation
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
```



## 🤝 Contributing

Contributions are welcome! Feel free to open issues, suggest improvements, or submit pull requests.



## 🙏 Acknowledgements

Special thanks to the [Open-R1 project](https://github.com/huggingface/open-r1) by Hugging Face and the broader open-source AI community for their foundational work.

