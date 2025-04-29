import os
import contextlib
import functools
import time
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

def get_dataset(dataset_name, split='train', system_prompt=None):

    if dataset_name == 'openai/gsm8k':
        dataset = load_dataset(dataset_name, name='main', split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
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
    
    return dataset


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