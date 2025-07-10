import os
import contextlib
import functools
import time
import random
from typing import Generator
from transformers import (
    AutoTokenizer, 
    PreTrainedTokenizer, 
    Trainer, 
    is_wandb_available
)
from datasets import load_dataset

if is_wandb_available():
    import wandb


##################
# Setup wandb
##################
"""
By default, WandbCallback (in transformers) gets the project name from 
`os.environ['WANDB_PROJECT']` or defaults to "huggingface"
See: 
https://github.com/huggingface/transformers/blob/v4.44.1/src/transformers/integrations/integration_utils.py#L834
project=os.getenv("WANDB_PROJECT", "huggingface")
"""
def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project


#####################
# Check if messages
#####################
# Adapted from: https://github.com/huggingface/trl/blob/v0.15.1/trl/data_utils.py#L24
def is_messages(examples):
    # examples = [
    #     {"role": "user", "content": "What color is the sky?"},
    #     {"role": "assitant", "content": "The sky is blue."}
    # ]
    # It must be a list of messages.
    if isinstance(examples, list):
        maybe_message = examples[0]
        # Each message must a list of dictionaries with keys "role" and "content"
        if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
            return True
    return False


##################
# Get tokenizer
##################
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

# TODO: revise
def get_tokenizer(
    model_args, 
    training_args, 
    auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    
    return tokenizer


########################
# Get train/eval dataset
########################

def get_dataset(
    dataset_name,                   # Name of the dataset. Can be a single dataset or a list of datasets.
    # *,                              # Arguments after this are keyword-only (no positional arguments allowed)
    split='train',                  # Data split, e.g., 'train' or 'test'
    num_samples_per_dataset=None,   # Number of max samples used per dataset
    system_prompt=None,             # System prompt
):
    
    # Handle multiple datasets
    # Convert list of dataset names to dict of `Dataset` if necessary
    if isinstance(dataset_name, list):
        datasets = {name: get_dataset(name, split, num_samples_per_dataset, system_prompt) for name in dataset_name}
        return datasets
    
    # Check if the input is a local file or directory
    if os.path.exists(dataset_name):
        print(f"Loading local dataset from: {dataset_name}")
        # Load local dataset
        dataset = load_dataset(
            'json' if dataset_name.endswith('.jsonl') else 'csv',  # Infer format
            data_files=dataset_name,
            split=split
        )
    else:
        print(f"Loading remote dataset from Hugging Face Hub: {dataset_name}")
        # Change the following accordingly
        if dataset_name == 'openai/gsm8k':
            dataset = load_dataset(dataset_name, name='main', split=split)
        elif dataset_name == 'opencompass/AIME2025':
            dataset = load_dataset(dataset_name, name='AIME2025-I', split=split)
        elif dataset_name == 'HuggingFaceH4/aime_2024':
            # Note: only train split is available for this dataset
            dataset = load_dataset(dataset_name, split='train')
        else:
            dataset = load_dataset(dataset_name, split=split)
        # TODO: add support for other datasets accordingly
    
    columns = dataset.column_names
    
    # Check if 'problem' is in columns (may change it accordingly):
    if 'problem' not in columns:
        for feture in columns:
            # 'problem', 'query' can be considered as 'problem' column
            # (may change or add columns accordingly)
            if feture.lower() in ['question', 'problem', 'query']:
                dataset = dataset.rename_column(feture, 'problem')
                break
        else:
            raise ValueError("No column named 'problem' in the datset!")
    
    # Check if 'solution' is in columns:
    if 'solution' not in columns:
        for feture in columns:
            # 'answer', 'response' can be considered as 'solution' column
            if feture.lower() in ['answer', 'solution', 'response']:
                dataset = dataset.rename_column(feture, 'solution')
                break
        else:
            raise ValueError("No column named 'solution' in the datset!")
    
    # Format into conversation
    def make_conversation(example):
        prompt = []
        if system_prompt is not None:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}
    
    dataset = dataset.map(make_conversation)
    
    # if "messages" in dataset.column_names:
    #     dataset = dataset.remove_columns("messages")
    
    # Chose a sample (by `num_samples_per_dataset`) from the dataset
    # Note: Can use a small dataset for fast check. Can change the random strategy accordingly.
    # Make sure it's called after data preprocessing.
    if num_samples_per_dataset is not None and num_samples_per_dataset > 0:
        num_samples = min(num_samples_per_dataset, len(dataset))
        sample_ids = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(sample_ids)
    
    return dataset


######################
# Check gpu idle
######################
import subprocess
import pandas as pd
from io import StringIO
import re

def get_visible_gpus():
    """
    Get the list of visible GPU IDs from CUDA_VISIBLE_DEVICES.
    Returns a list of GPU indices or None if not set or empty.
    """
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible is None or cuda_visible.strip() == "":
        # If CUDA_VISIBLE_DEVICES is not set or empty, return None to check all GPUs
        return None
    try:
        # Parse comma-separated GPU IDs
        gpu_ids = [int(gpu_id.strip()) for gpu_id in cuda_visible.split(',')]
        return gpu_ids if gpu_ids else None
    except ValueError:
        print("Invalid CUDA_VISIBLE_DEVICES format, treating as no restriction.")
        return None


def check_gpu_free(gpu_memory_threshold=10000, gpu_util_threshold=10):
    """
    Check if the GPU is idle.
    gpu_memory_threshold: GPU memory usage threshold (MB), below which the GPU is considered idle.
    gpu_util_threshold: GPU utilization threshold (%), below which the GPU is considered idle.
    Returns True if the GPU is idle, False if it is busy.
    """
    try:
        # Run nvidia-smi command to get GPU information. Note: nvidia-smi will get all GPU information.
        gpu_stat = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv'],
            stdout=subprocess.PIPE, text=True, check=True
        )
        # Parse CSV output
        df = pd.read_csv(
            StringIO(gpu_stat.stdout),
            names=['memory.used [MiB]', 'utilization.gpu [%]'],
            skiprows=1
        )
        # Get visible gpus if any
        visible_gpus = get_visible_gpus()
        if visible_gpus is not None:
            df = df[df.index.astype(int).isin(visible_gpus)]
        print(df)
        # Check memory usage and utilization for each GPU
        for i, row in df.iterrows():
            # Extract memory usage (MB)
            memory_used = int(re.search(r'\d+', row['memory.used [MiB]']).group())
            # Extract utilization (%)
            utilization = int(re.search(r'\d+', row['utilization.gpu [%]']).group())
            # If any GPU's memory usage or utilization exceeds the threshold, consider it busy
            if memory_used > gpu_memory_threshold or utilization > gpu_util_threshold:
                return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return False




######################
# Get open port
######################
import socket
def _get_open_port(port) -> int:
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                print(f"Port is set as {port}")
                return port
        except OSError:
            port += 1  # Increment port number if already in use
            print(f"Port {port-1} is already in use, trying port {port}")


####################
# Profiling context
####################
@contextlib.contextmanager
def profiling_context(trainer: Trainer, name: str) -> Generator[None, None, None]:
    """
    A context manager function for profiling a block of code. Results are logged to Weights & Biases if enabled.

    Args:
        trainer (`~transformers.Trainer`):
            Trainer object.
        name (`str`):
            Name of the block to be profiled. Used as a key in the logged dictionary.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_context

    class MyTrainer(Trainer):
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            with profiling_context(self, "matrix_multiplication"):
                # Code to profile: simulate a computationally expensive operation
                result = A @ B  # Matrix multiplication
    ```
    """
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time

    if "wandb" in trainer.args.report_to and wandb.run is not None and trainer.accelerator.is_main_process:
        wandb.log({f"profiling/Time taken: {trainer.__class__.__name__}.{name}": duration})


def profiling_decorator(func: callable) -> callable:
    """
    Decorator to profile a function and log execution time using [`extras.profiling.profiling_context`].

    Args:
        func (`callable`):
            Function to be profiled.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_decorator

    class MyTrainer(Trainer):
        @profiling_decorator
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            # Code to profile: simulate a computationally expensive operation
            result = A @ B
    ```
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper