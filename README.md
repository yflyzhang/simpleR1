<h1 align="center"> simpleR1: A simple framework for training R1-like models</h1>


**simpleR1** is a simple framework for training R1-like models, aiming to improve llm's reasoning abilities in math and code. 
This repository builds upon Hugging Face's TRL GRPO Trainer and the [open-r1](https://github.com/huggingface/open-r1) project, with a focus on ease of use and enhanced training features. 


## Key Features

The latest version includes an upgraded GRPO Trainer with a custom evaluate function for simultaneous training and evaluation, modularized model completion and reward score estimation.

- Enhanced GRPO trainer with multi-iteration support, precise time estimation (tqdm), custom evaluate block, and wandb logging.

- Modularized generate, score, and log for completions, enabling more user-defined controls.

- Implementation of a simple reject sampling approach for generation.


- Compatible with Hugging Face TRL and open-r1 workflows and scripts.





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
│   ├── run_grpo_qwen2.5-1.5b-single.sh   # Shell for running Qwen2.5-1.5B with a single gpu
│   ├── run_grpo_qwen3-0.6b-single.sh     # Shell for running Qwen3-0.6B with a single gpu
│   └── ...         
│           
├── src/                       # Python codes
│   ├── arguments.py           # Model, scripts, and training arguments
│   ├── rewards.py             # Reward functions
│   ├── grpo_trainer.py        # Trainer for GRPO [core part]
│   ├── run_grpo.py            # Python scripts to run GRPO
│   └── utils.py               # Supporting utils
│
├── requirements.txt           # Full list of requirements
├── LICENSE
└── README.md                  # This document
```


## 📚 Examples


### Training on MATH-benchmark and evaluating on MATH-500

We trained four `Qwen2.5-1.5B*` models and one `Qwen3-0.6B*` model on [MATH-benchmark](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark), with real-time evaluation on [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) using the `evaluate` function.

- **Models**: 
`Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-Math-1.5B`, `Qwen/Qwen2.5-Math-1.5B-Instruct`, `Qwen/Qwen3-0.6B`.

- **Setup**: Trained for 1 epoch, 3 grpo iterations, with one NVIDIA A100-80G GPU and parameters as shown in the usage example. The max completion length is set as 2,048 and 4,096 for train and eval mode, respectively. Train and evaluation accuracy and completion length were logged via wandb.

- **Rewards**: Rule-based rewards were adopted in the example, including 'accuracy reward', 'format reward', and 'tag count reward', with different weights to arrive at the final reward signal. For example, `reward = 8 * accuracy_reward + 1 * format_reward + 1 * tag_count_reward`.

- **Results**: 
  - All models improved accuracy, with Math-specific models (`Qwen2.5-Math-1.5B*`) outperforming general models (`Qwen2.5-1.5B*`). 
  - Completion length varied, with Instruct models often producing shorter outputs. 
  - `Qwen3-0.6B` outperforms all the tested `Qwen2.5*-1.5B*` variants except for `Qwen2.5-Math-1.5B-Instruct`, but this superior performance comes at the expense of generating significantly more tokens. 

### Model Accuracy Comparison

Below is the plot comparing evaluation accuracy across the four models:

<p align="center">
  <img src="imgs/wandb_log-math.png" width="800" />
  <br>
  <strong>Fig 1. SimpleR1 running example (eval on MATH-500).</strong>
  <br>
  📈 More train and eval logs are available at WandB project: 
      <a href="https://api.wandb.ai/links/yflyzhang/wkdme9ea">wandb log</a>.
</p>


### Results Table

<!-- | Model | Initial Accuracy | Best Accuracy | Completion Length Trend | Token Efficiency |
| --- | --- | --- | --- | --- |
| `Qwen2.5-1.5B` | 0.150 | 0.478 | 2,044 → 550 tokens | ↑ |
| `Qwen2.5-1.5B-Instruct` | 0.368 | 0.492 | 455 → 552 tokens |  ↓ |
| `Qwen2.5-Math-1.5B` | 0.432 | 0.606 | 1425 → 812 tokens | ↑ |
| `Qwen2.5-Math-1.5B-Instruct` | 0.738 | 0.766 | 591 → 581 tokens | -- | -->


| Model | Initial Accuracy | Best Accuracy | Completion Length Trend | Token Efficiency | Runtime |
| --- | --- | --- | --- | --- | --- |
| `Qwen2.5-1.5B` | 0.150 | 0.478 | 2,044 → 550 tokens | ↑ | 3h 37m 55 |
| `Qwen2.5-1.5B-Instruct` | 0.368 | 0.492 | 455 → 552 tokens |  ↓ | 2h 18m 43 |
| `Qwen2.5-Math-1.5B` | 0.432 | 0.606 |  1425 → 812 tokens | ↑ | 3h 6m 20s |
| `Qwen2.5-Math-1.5B-Instruct` | 0.738 | 0.766 | 591 → 581 tokens | -- | 5h 39m 41 |
| `Qwen3-0.6B` | 0.576 | 0.612 | 2096 → 1799 tokens | ↑ | 11h 3m 10 |



### Training on `gsm8k` and evaluating on `MATH-500`

We also trained the models on [gsm8k](https://huggingface.co/datasets/openai/gsm8k) (with 1000 samples for fast test) and evaluated on [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500). The evaluation results can be found as follows:

<p align="center">
<img src="imgs/wandb_log-gsm.png" width="800" />
<br>
<strong>Fig 2. Train on gsm8k and evaluate on MATH-500.</strong>
<br>
📈 More train and eval logs are available at WandB project: 
<a href="https://wandb.ai/yflyzhang/simpleR1-gsm8k/reports/SimpleR1-Training-Examples-gsm8k---VmlldzoxMjg1MjE4Mw?accessToken=ifm2s40kzp8a5i5n94jozsli50mdp978s9hz8ukjrqezo0frsj684l2aayvn5r7d">wandb log</a>
</p>



### Case Examples


Below is an response example from trained `Qwen2.5-1.5B` on a MATH-500 problem:
<details>
  <summary>Click to expand/collapse</summary>

  - **Question**:
  "Paul and Jesse each choose a number at random from the first six primes. What is the probability that the sum of the numbers they choose is even?"
  
  - **Reference Answer**: "The only way for the sum of the numbers Paul and Jesse choose to be odd is if one of them chooses 2 and the other chooses an odd prime. There are five ways for Paul to choose 2 and Jesse to choose an odd prime, and there are five ways for Jesse to choose 2 and Paul to choose an odd prime. Since there are $6\cdot 6=36$ total possible ways for Paul and Jesse to choose their numbers, the probability that the sum of the numbers Paul and Jesse choose is NOT even is $\frac{10}{36}=\frac{5}{18}$. Therefore, the probability that the sum of the numbers Paul and Jesse choose IS even is $1-\frac{5}{18}=\boxed{\frac{13}{18}}$."

  - **After Training**: "\<think>
    To determine the probability that the sum of the numbers chosen by Paul and Jesse is even, we need to consider the properties of even and odd numbers. The sum of two numbers is even if and only if both numbers are either even or both are odd.

    First, let's identify the first six prime numbers: 2, 3, 5, 7, 11, and 13.

    Next, we count the number of even and odd primes:
    - Even primes: 2
    - Odd primes: 3, 5, 7, 11, 13 (total of 5 odd primes)

    Now, let's calculate the total number of possible outcomes when both Paul and Jesse choose a number:
    - There are 6 choices for Paul.
    - There are 6 choices for Jesse.
    - Therefore, the total number of possible outcomes is $6 \times 6 = 36$.

    Next, we calculate the number of favorable outcomes where the sum is even:
    - Both Paul and Jesse choose an even prime: $1 \times 1 = 1$ (since Paul must choose 2 and Jesse must choose one of 3, 5, 7, 11, 13).
    - Both Paul and Jesse choose an odd prime: $5 \times 5 = 25$ (since Paul can choose any of the 5 odd primes and Jesse can choose any of the 5 odd primes).

    So, the total number of favorable outcomes is $1 + 25 = 26$.

    Finally, the probability that the sum of the numbers chosen is even is the ratio of the number of favorable outcomes to the total number of possible outcomes:
    $\text{Probability} = \frac{26}{36} = \frac{13}{18}$

    Therefore, the probability that the sum of the numbers they choose is even is $\frac{13}{18}$.\
    \</think>\
    \<answer>
    $\frac{13}{18}$
    \</answer>"

</details>




## Insights
- [x] TODO

## 🚀 Usage

1. Clone the repository:
```bash
git clone https://github.com/yflyzhang/simpleR1.git
cd simpleR1
```


2. Example training command:

```bash
bash scripts/run_grpo_qwen2.5-1.5b-single.sh
```

> [!NOTE]
> `run_grpo_qwen2.5-1.5b-single.sh` provides a concrete runing example using only one A100-80G GPU, please change the parameters therein accordingly.

Or override additional parameters via command line. For example,

```bash
# HF_HOME=/xxx/xxx/.cache/huggingface \
CUDA_VISIBLE_DEVICES=0,1,2 \  # assume we have 3 gpus
accelerate launch \
    --main_process_port $MASTER_PORT \
    --config_file configs/accelerate_configs/zero1.yaml \
    --num_processes=2 \       # cuda:2 is reserved for vllm generation
src/run_grpo.py \
    --config configs/grpo_template.yaml \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $model_name_or_path \
    --train_dataset_name $train_dataset \
    --eval_dataset_name $eval_dataset \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.2 \
    --num_train_epochs 1 \
    --num_generations 7 \
    --num_eval_generations 1 \
    --per_device_train_batch_size 7 \
    --per_device_eval_batch_size 64 \
    --max_resample_attempts 3 \
    --gradient_accumulation_steps 3 \
    --num_iterations 3 \
    --torch_empty_cache_steps 1 \
    --max_num_train_samples 1000 \  # restrict max train/test samples for fast test
    --max_num_test_samples -1 \
    --max_completion_length 2048 \
    --max_eval_completion_length 4096 \
    --reward_funcs accuracy format tag \
    --reward_weights 8 1 1 \
    --loss_type bnpo \
    --scale_rewards False \
    --mask_truncated_completions True \
    --epsilon 0.2 \
    --epsilon_high 0.3 \
    --temperature 1.0 \
    --top_p 0.95 \
    --eval_temperature 0.7 \    # eval rollout hyperparameter
    --eval_top_p 0.95 \
    --beta 0.0001 \
    --compute_kl True \
    --learning_rate 3e-6 \
    --save_strategy steps \
    --save_steps 100 \
    --eval_strategy steps \
    --eval_steps 10 \
    --eval_on_start True \
    --log_level info \
    --wandb_project simpleR1-$(basename $train_dataset) \
    --run_name $run_name \
    2>&1 | tee $LOG_FILE
```


## Dependencies
See `requirements.txt` for a full list, but generally you don't need to install all of them. <br>
Key dependencies include and can be installed as follows:

```bash
# 1. Create and activate a new env named `simpler1`
conda create --prefix simpler1 python==3.12
# Activate the env, for example:
conda activate /path/simpler1

# 2. Install the key dependencies
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install transformers==4.51.0 accelerate==1.4.0 trl==0.16.0 deepspeed==0.16.4
pip install flash-attn --no-build-isolation
pip install vllm==0.7.2
pip install math-verify==0.5.2 latex2sympy2_extended==1.0.6
pip install wandb==0.19.7
```


## Contributing

Contributions are welcome! Feel free to open issues, suggest improvements, or submit pull requests.



## Acknowledgements

Special thanks to the [Open-R1 project](https://github.com/huggingface/open-r1) by Hugging Face and the broader open-source AI community for their foundational work.

