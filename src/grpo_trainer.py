# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2025 yflyzhang. All rights reserved.
# Modifications are licensed under the Apache License, Version 2.0.


import contextlib
import functools
import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union, Iterator
from unittest.mock import patch

# --------------------------------
from transformers.trainer import *

# `_xxx`` cannot be imported by `from module import *`
from transformers.trainer import _is_peft_model
# --------------------------------

import gc
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import Sampler
import transformers
from accelerate import PartialState
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available


from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
# from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

from tqdm import tqdm
if is_wandb_available():
    import wandb



from utils import is_messages
from vllm_client import VLLMClient

# Custom ProgressCallback
from callbacks import GRPOProgressCallback
# DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def debug(rank=0):
    # Add a distributed breakpoint for debug for a specific rank
    torch.distributed.breakpoint(rank=rank)

# breakpoint()


class RepeatRandomSampler(Sampler):
    r"""
    Random sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).
    
    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    
    Ref: RandomSampler
        https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/data/sampler.py#L132
    """
    data_source: Sized
    repeat_count: int
    
    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None) -> None:
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)
    
    def __iter__(self) -> Iterator[int]:
        for idx in torch.randperm(self.num_samples, generator=self.generator).tolist():
            yield from [idx] * self.repeat_count
            # for _ in range(self.repeat_count):
            #     yield idx
    
    def __len__(self) -> int:
        return self.num_samples * self.repeat_count
    

# SequentialSampler for eval dataset
class RepeatSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order, with optional repetition.
    
    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index. Defaults to 1.
    
    Example:
    ```python
    >>> sampler = RepeatSequentialSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [0, 0, 1, 1, 2, 2, 3, 3]    # same order as the raw dataset
    ```

    Ref: SequentialSampler
        https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/data/sampler.py#L113
    """
    data_source: Sized
    repeat_count: int

    def __init__(self, data_source: Sized, repeat_count: int=1) -> None:
        self.data_source = data_source
        self.repeat_count = repeat_count
    
    def __iter__(self) -> Iterator[int]:
        # return iter(range(len(self.data_source)))
        for idx in range(len(self.data_source)):
            yield from [idx] * self.repeat_count
            # for _ in range(self.repeat_count):
            #     yield idx
    
    def __len__(self) -> int:
        return len(self.data_source) * self.repeat_count



class GRPOTrainer(Trainer):
    r"""
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).
    
    Trainer ref:
        https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer.py#L316
    
    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo", "simpleR1"]
    
    def __init__(
        self,
        model: Union[PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        ref_model: Optional[Union[str, PreTrainedModel]] = None,
        args: Union[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            raise ValueError(f"Invalid args. Expected to be GRPOConfig.")
        
        # TODO: 
        if args.gradient_accumulation_steps != 1:
            # only support gradient_accumulation_steps = 1
            logger.warning(f"Invalid gradient_accumulation_steps. Expected to be 1.")
            args.gradient_accumulation_steps = 1
        
        self.beta = args.beta
        self.ref_model = ref_model

        self.compute_kl = args.compute_kl   # compute kl even when beta=0 (helps to monitor model update)
        if args.beta > 0:                   # have to compute kl when beta > 0
            self.compute_kl = True
        
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if not isinstance(model, PreTrainedModel):
            raise ValueError(
                f"Invalid type. Expected to be type of 'PreTrainedModel', but got {type(model)}."
            )
        

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)
        
        # Disable dropout in the model?
        # Note: Qwen 2.5 and Llama 3.2 didn't use dropout.
        if args.disable_dropout:
            self.disable_dropout_in_model(model)
            if self.ref_model is not None:
                self.disable_dropout_in_model(self.ref_model)
        
        # Processing class
        if processing_class is None:
            logger.warning(
                f"`processing_class` is not defined, will load tokenizer from `{model.config._name_or_path}` by default."
            )
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.temperature = args.temperature

        # Eval arguments
        # If not explicitly specified, use parameters from training mode
        self.num_eval_generations = args.num_eval_generations if args.num_eval_generations is not None else args.num_generations
        self.max_eval_completion_length = args.max_eval_completion_length if args.max_eval_completion_length is not None else args.max_completion_length
        self.eval_temperature = args.eval_temperature if args.eval_temperature is not None else args.temperature
        self.eval_top_p = args.eval_top_p if args.eval_top_p is not None else args.top_p
        self.eval_top_k = args.eval_top_k if args.eval_top_k is not None else args.top_k
        self.eval_min_p = args.eval_min_p if args.eval_min_p is not None else args.min_p
        
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization     # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size         # only applies to colocation mode
        
        # Multi-step
        self.num_iterations = args.num_iterations   # = 𝜇 in the GRPO paper
        # --------------------------
        self.mode = None                            # train or eval mode
        self.grpo_iteration = 0                     # tracks current grpo iteration
        self.mini_batch_step = -1                   # tracks mini-batch step
        self.max_resample_attempts = args.max_resample_attempts         # max number of generation attempts
        self.scale_rewards = args.scale_rewards     # scale the rewards by std or not
        self.mask_truncated_completions = args.mask_truncated_completions   # mask truncated completions
        self.loss_type = args.loss_type
        
        # Clip higher
        self.epsilon = args.epsilon                 # clip value (pi_theta/pi_old)
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # --------------------------

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        # Completion examples (e.g., prompt + solution + score) for wandb log
        self._completion_examples = {"train": [], "eval": []}
        self.log_completions = args.log_completions
        
        # Use num_train_epochs to estimate max_steps
        if args.max_steps > 0 and args.num_train_epochs > 0:
            args.max_steps = -1
            logger.info("num_train_epochs is given, do not use max_steps!")
        
        # Call `Trainer.__int__`
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        
        # TODO: dynamic sampling or not
        # Remove `transformers.trainer_callback.ProgressCallback`,
        # and add custom `GRPOProgressCallback`
        from transformers.trainer_callback import ProgressCallback
        self.callback_handler.pop_callback(ProgressCallback)
        self.callback_handler.add_callback(GRPOProgressCallback)
        # self.callback_handler.callbacks
        
        # debug(0)
        
        # Note: the log level (of logger) is set depending on the node. 
        # By default, the main process (rank 0) logs at level INFO, while other processes log at level WARNING.
        # For further details, see:
        # https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer.py#L470.
        # https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/training_args.py#L2404
        
        
        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        effective_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            effective_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(1, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0]
            # Note: possible_values can start from 1 in eval mode.
            if self.num_eval_generations not in possible_values:
                raise ValueError(
                    f"The effective eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the eval number of generations per prompt ({self.num_eval_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm==xxx` to use it."
                )
            
            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                # `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
                # server is running (start with `trl vllm-serve`).
                if args.vllm_server_base_url is not None:
                    base_url = args.vllm_server_base_url
                else:
                    base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                self.vllm_client.init_communicator()
            
            elif self.vllm_mode == "colocate":
                # `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
                #   separate server but may cause resource contention with training.

                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )
                
                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )
                
                # Setup vLLM
                self.llm = LLM(
                    model=model.name_or_path,
                    gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    distributed_executor_backend="external_launcher",   # important for tp
                    # max_num_seqs=self.args.per_device_eval_batch_size 
                    #             * self.vllm_tensor_parallel_size
                    #             * self.args.gradient_accumulation_steps,
                    # max_model_len=self.max_prompt_length + self.max_eval_completion_length,
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    # seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    seed=self.args.seed,
                    # # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    # max_num_batched_tokens=4096,
                )
            
            # Guided decoding, if enabled
            if args.vllm_guided_decoding_regex is not None:
                guided_decoding = GuidedDecodingParams(backend="outlines", regex=args.vllm_guided_decoding_regex)
            else:
                guided_decoding = None
            
            # Sampling parameters for training
            self.sampling_params = SamplingParams(
                n=1 if self.vllm_mode == "colocate" else args.num_generations,  
                # vLLM on each GPU generates only 1 in colocate mode
                max_tokens=self.max_completion_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=-1 if args.top_k is None else args.top_k,
                min_p=0.0 if args.min_p is None else args.min_p,
                repetition_penalty=args.repetition_penalty,
                guided_decoding=guided_decoding,
            )
            
            # Sampling parameters for evaluation
            self.eval_sampling_params = SamplingParams(
                n=1 if self.vllm_mode == "colocate" else args.num_eval_generations,  
                # vLLM on each GPU generates only 1 in colocate mode
                max_tokens=self.max_eval_completion_length,
                temperature=args.eval_temperature,
                top_p=args.eval_top_p,
                top_k=-1 if args.eval_top_k is None else args.eval_top_k,
                min_p=0.0 if args.eval_min_p is None else args.eval_min_p,
                repetition_penalty=args.repetition_penalty,
                guided_decoding=guided_decoding,
            )
        

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
            )


        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
    

    def disable_dropout_in_model(self, model: torch.nn.Module) -> None:
        r"""Disable dropout in the model and reference model."""
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "problem", "solution"]
    
    # TODO: remove mini-batch
    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        
        # In the following illustration, values are the prompt indices. The first row indictaes the first sampled mini-batch, 
        # the second row indicates the second sampled mini-batch, and so on.
        #
        #                                                 |     GPU 0     |     GPU 1     |     GPU 2    |
        #                            grpo
        #            global_step   iteration   mini-batch   <───────>  num_generations=3
        #                                                   <───────────> per_device_train_batch_size=4
        #                ▲   0          0          1        0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          0          2        4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          0          3        8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    0          1          1        0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    0          1          2        4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    0          1          3        8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          0          1       12  12  12  13  13  13  14  14  14  15  15  15  │ 
        #                    1          0          2       16  16  16  17  17  17  18  18  18  19  19  19  │ Next batched samples
        #                    1          0          3       20  20  20  21  21  21  22  22  22  23  23  23  │
        #                                                  ...

        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)
    
    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Note: In eval mode, no need to shuffle the dataset sample order!
        # Returns a sequential sampler that ensures each prompt is repeated across multiple processes.
        # See `RepeatSequentialSampler` and `_get_train_sampler for an explanation of the sampler.
        return RepeatSequentialSampler(eval_dataset, self.num_eval_generations)
        
        
    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _move_model_to_vllm(self):
        r"""Load model state dict to vllm"""
        # logger.debug(f"[rank={self.accelerator.process_index}] Moving model to vllm")
        
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = contextlib.nullcontext
        
        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            raise ValueError("PEFT is not supported now.")
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(name, param.data)])
        
        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
        
            
    
    def get_train_dataloader(self) -> DataLoader:
        r"""
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        
        # train_dataloader = DataLoader(train_dataset, **dataloader_params)
        
        # debug(0)
        # Note: After `accelerator.prepare`, len(train_dataloader) may be changed if we use data parallel.
        
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        
    
    def training_step(
        self, model: nn.Module, inputs: list[str], num_items_in_batch=None
    ) -> torch.Tensor:
        r"""
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`list[str]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.

        Ref: 
            https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3668
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        
        # ----------------------------------------
        # Prepare grpo inputs: generate, score and log the completions
        self.mode = 'train'
        # inputs = self.prepare_grpo_inputs(inputs)
        # ----------------------------------------
        
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        gc.collect()

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()
        
        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()
    

    def check_model_generation(self, rewards, **kwargs):
        r"""
        Check if current generation meets specific criterion based on rewards.
        For example, make sure the generated sentences are diverse and contain positive samples.
        
        Add more condition checks if you want to inject some custom behavior.

        Args:
            rewards (`list[float]`): The rewards of the generated sentences.

        Return:
            `str`: The category of the generation.
                "easy": Current problems are easy to solve. Skip this generation.
                "hard": Current problems are hard to solve. Try again.
                "bad": Current generations are not diverse enough. Try again.
                "good": Current generations are diverse with at least one right answer.
        """

        # TODO: if no dynamic sampling, simply return 'good'

        # 1. Easy: full rewards for all generations
        # Note: the current question is too easy for the model as it can get full rewrds for all generations.
        # if min(rewards) == sum(self.reward_weights.tolist()):
        if min(rewards) >= sum(self.reward_weights.tolist()):
            return 'easy'
        
        # 2. Bad: Nearly identical rewards, e.g., std=0
        # example: [10, 10, 10, 8]
        # if np.std(rewards) == 0:
        if np.std(rewards) < 1e-2:
        # RSD = std / mean, RSD > 0.1 is good?
        # TODO: change to other values or set by user
        # if np.std(rewards) / (np.mean(rewards) + 1e-6) < 0.1:
            return 'bad'
        
        # 3. Hard: accuracy reward (the primary reward)
        # If no generation arrives at the right answer, try it again.
        if max(rewards) < max(self.reward_weights.tolist()):    
            return 'hard'
        # # If no generation get the full reward, try it again.
        # if max(rewards) < sum(self.reward_weights.tolist()):   
        #     return 'hard
        
        # TODO: 
        # Check other conditions, e.g., generation diversity, entropy
        
        # 4. Good: All rewards are within a certain range (with at least one right answer)
        return 'good'
    
    
    def _generate_completions(self, inputs, mode):
        r"""
        Generate completions for the given inputs.
        
        Args:
            inputs (`list[str]`):
                List of input batched data.
            mode (`str`):
                    The mode of generation. Can be `"train"` or `"eval"`.
        
        Return:
            `dict[str, Union[torch.Tensor, Any]]`.
        """
        # TODO: remove self.mode
        # mode = self.mode
        device = self.accelerator.device
        problems = [example["problem"] for example in inputs]   # raw problems
        prompts = [example["prompt"] for example in inputs]     # raw prompts
        solutions = [example["solution"] for example in inputs] # raw solutions

        # Apply chat template and tokenize
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        
        # Prepare `prompt_inputs` before feeding them to the model, converting them to tensors 
        # if they are not already and handling potential state.
        prompt_inputs = self._prepare_inputs(prompt_inputs)     # transformers.Trainer._prepare_inputs
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        
         # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            
            
            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    
                    # vllm generate
                    if mode == 'train':
                        sampling_params = self.sampling_params
                        num_generations = self.num_generations
                    else:       # eval mode
                        sampling_params = self.eval_sampling_params
                        num_generations = self.num_eval_generations
                    
                    ordered_set_of_prompts = all_prompts_text[::num_generations]
                    all_outputs = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        sampling_params=sampling_params
                    )
                    # Example output:
                    # all_outputs = {
                    #     'completion_ids': [[1,2,3,4], [5,6,7,8,9]],
                    #     'completion_texts': ['This is a test.', 'This is another test.']
                    # }
                    
                    all_completion_ids = all_outputs['completion_ids']
                    all_completion_texts = all_outputs['completion_texts']
                    
                else:
                    all_completion_ids = [None] * len(all_prompts_text)
                    all_completion_texts = [None] * len(all_prompts_text)
                
                # Broadcast the completions from the main process to all processes, 
                # ensuring each process receives its corresponding slice.
                all_completion_ids = broadcast_object_list(all_completion_ids, from_process=0)
                all_completion_texts = broadcast_object_list(all_completion_texts, from_process=0)
                
                # Keep only the local part of the data
                start = self.accelerator.process_index * len(prompts)
                completion_ids = all_completion_ids[start:start+len(prompts)]   # [i:i+len(prompts)]
                completion_texts = all_completion_texts[start:start+len(prompts)]


            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                # vllm generate
                if mode == 'train':
                    sampling_params = self.sampling_params
                else:       # eval mode
                    sampling_params = self.eval_sampling_params
                
                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text
                
                # Each process generates its own completions given the prompts
                all_outputs = self.llm.generate(
                    all_prompts_text, 
                    sampling_params=sampling_params, 
                    use_tqdm=False
                )
                
                # Each process receives its corresponding slice
                completion_ids = []
                completion_texts = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
                        completion_texts.append(output.text)
                
                
                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]
                
            # Pad the completions
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        else:
            # TODO: remove regular generation or add sglang support
            # Regular generation path
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Decode the generated completions
            completion_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            completion_mask = completion_mask * is_eos.any(dim=1).unsqueeze(1).int()
        
        # Process the generated completions
        # completion_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completion_texts):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])  # multi-turn support
                # completions.append({"role": "assistant", "content": bootstrap + completion})    # single-turn only
        else:
            completions = completion_texts

        
        # debug(0)

        return {
            # 1. text
            "problems": problems,
            "solutions": solutions,
            "prompts": prompts,
            "completions": completions,
            
            # 2. eos index
            "eos_idx": eos_idx,
            
            # 3. prompt and completion ids and masks
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        }
    
    
    def _score_completions(self, inputs, mode):
        r"""
        Score completions for the given generations.

        Args:
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                List of input batched data.
            mode (`str`):
                The mode of generation. Can be `"train"` or `"eval"`.
        
        Return:
            `tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]`.
        """
        # mode = self.mode
        if mode == 'train':
            num_generations = self.num_generations
            max_completion_length = self.max_completion_length
        else:       # eval mode
            num_generations = self.num_eval_generations
            max_completion_length = self.max_eval_completion_length

        device = self.accelerator.device
        prompts = inputs['prompts']
        completions = inputs['completions']
        solutions = inputs['solutions']
        
        # Compute rewards
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            
            # if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
            #     reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            # else:
            #     reward_func_name = reward_func.__name__
            
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                # if is_conversational(inputs[0]):    
                if is_messages(completions[0]): # to be verified here
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = self._prepare_inputs(reward_inputs)     # Trainer._prepare_inputs
                with torch.no_grad():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Get each rule-based reward
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, solutions=solutions
                )
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        all_rewards_per_func = gather(rewards_per_func)     # gather from all devices
        # Note: mask_truncated_completions is not applied to 'all_rewards_per_func', 
        # but may be applied to 'all_rewards'
        
        # Apply weights to each reward function's output and sum
        all_rewards = (all_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        
        # eos_idx can indicate the completion length
        eos_idx = inputs['eos_idx']
        all_eos_idx = gather(eos_idx)     # gather from all devices

        # If mask_truncated_completions is enabled, zero out rewards of truncated completions
        if self.mask_truncated_completions:
            all_rewards = all_rewards * (all_eos_idx < max_completion_length).int()
        
        # Do not compute advantages in eval mode
        all_advantages = None
        if mode == 'train':
            # Compute grouped-wise rewards
            mean_grouped_rewards = all_rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            
            # Note: num_generations>1 is enabled in train mode
            std_grouped_rewards = all_rewards.view(-1, num_generations).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
            
            # Compute the advantages
            all_advantages = all_rewards - mean_grouped_rewards
            if self.scale_rewards:
                all_advantages = all_advantages / (std_grouped_rewards + 1e-4)
        
        # Get the primary reward (i.e., accuracy reward)
        # Note: by default, the first reward function is the accuracy reward.
        all_accuracy_reward = all_rewards_per_func[:, 0]
        
        # Print reward info in the main process
        # if self.accelerator.is_main_process:
        all_completions_length = all_eos_idx + (all_eos_idx < max_completion_length).int()
        print()     # for better readability only
        
        # Note: By default, log level in the main process is set to INFO, while the rest processes are set to WARNING.
        if mode == 'train':
            logger.info(
                f"\n  [{mode}] global_step = {self.state.global_step+1},"
                f" grpo_iteration = {self.grpo_iteration+1},"
                # f" mini_batch (grad. acc.) = {self.mini_batch_step+1}"
                f"\n    completion length = {all_completions_length.cpu().tolist()}"
                f"\n    accuracy reward   = {all_accuracy_reward.cpu().tolist()}"
                f"\n    total reward      = {all_rewards.cpu().tolist()}"
                f"\n    advantage         = {[round(x, 3) for x in all_advantages.detach().cpu().tolist()]}"
            )
        else:
            # accuracy is the primary goal!
            logger.info(
                f"\n  [{mode}] global_step = {self.state.global_step}"
                f"\n    completion length = {all_completions_length.cpu().tolist()}"
                f"\n    accuracy reward   = {all_accuracy_reward.cpu().tolist()}"
                # f"\n    total reward      = {all_rewards.cpu().tolist()}"
            )
    
        
        # debug(0)
        
        # return all device rewards/advantages
        return (
            all_rewards_per_func,   # all rewards per function (from all devices)
            all_rewards,            # all rewards (from all devices)
            all_advantages          # all advantages (from all device)
        )
        # Note: 
        # all_rewards_per_func: tracks the the raw reward (0.0-1.0) for each reward function.
        # all_rewards: is the weighted sum of all_rewards_per_func and may be applied mask_truncated_completions then.
        # all_advantages: computed advantages (only enabled in train mode; is not needed in eval mode by default)
    
        

    def _log_completions(self, inputs, all_rewards_per_func, all_rewards, mode):
        r"""
        Log completions (append/extend) in list of buffers `_metrics` and `_completion_examples`.
        
        Args:
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                Dict of generated and scored completions.
            all_rewards_per_func (`torch.Tensor`):
                All device rewards per each reward function.
            all_rewards (`torch.Tensor`):
                All device rewards (weighted sum of each reward function, and maybe partially masked).
            mode (`str`):
                The mode of generation. Can be `"train"` or `"eval"`.
        """
        # mode = self.mode
        if mode == 'train':
            num_generations = self.num_generations
            max_completion_length = self.max_completion_length
        else:       # eval mode
            num_generations = self.num_eval_generations
            max_completion_length = self.max_eval_completion_length
        
        eos_idx = inputs['eos_idx']
        prompts = inputs['prompts']
        completions = inputs['completions']
        
        # 1. Problem text
        if is_messages(prompts[0]):
            # by default, the last one denotes the problem: prompts = [{}, {'role': 'user', 'content': problem_text}]
            problems_text = [example[-1]['content'] for example in prompts]
        else:
            # may apply chat template, e.g. with system prompt
            problems_text = [
                maybe_apply_chat_template({'prompt': p}, self.processing_class)['prompt'] for p in prompts
            ]
        
        # 2. Raw solution text (gold) (may change it to `answer`?)
        solutions_text = inputs['solutions']
        
        # 3. Completion text (may extract from conversational message)
        if is_messages(completions[0]):
            completion_texts = [example[0]['content'] for example in completions]
            # completion_texts = [example['content'] for example in completions]
        else:
            completion_texts = completions
        
        if (
            self.log_completions and 
            self.state.global_step % self.args.logging_steps == 0 and
            self.args.report_to and "wandb" in self.args.report_to
        ):
            # Note: Log in the unique sample level (i.e., num_completions or num_generations), since the last batch
            # may not be full (especially for the eval dataset) and the averaged values over all samples are thus 
            # not accurate. 
            # For example, suppose we have 3 unique samples, num_generations=2 (6 samples in total)
            # and batch_size=4 (batch_1 gets 4 samples, bacth_2 gets 2 samples).
            # When we get the final resutls: res = [[1,1, 1,1], [0,0]],
            #   batch-level mean:         [[1,1, 1,1], [0,0]] -> [1, 0] -> 0.5
            #   unique-sample-level mean: [[1,1, 1,1], [0,0]] -> [1, 1, 0] -> 2/3 = 0.67
            # This example explains why we use `view(-1, num_generations)` below.
            
            # 4. Completion length (including truncated completions): mean, min, max
            all_eos_idx = gather(eos_idx)
            all_completions_length = all_eos_idx + (all_eos_idx < max_completion_length).int()

            if self.accelerator.is_main_process:
                # (a) naive mean/min/max
                # self._metrics[mode]["completions/mean_length"].append(all_completions_length.float().mean().item())
                # self._metrics[mode]["completions/min_length"].append(all_completions_length.float().min().item())
                # self._metrics[mode]["completions/max_length"].append(all_completions_length.float().max().item())
                # (b) unique-sample-level mean/min/max
                reshaped = all_completions_length.float().view(-1, num_generations)
                self._metrics[mode]["completions/mean_length"].extend(reshaped.mean(-1).tolist())
                self._metrics[mode]["completions/min_length"].extend(reshaped.min(-1).values.tolist())
                self._metrics[mode]["completions/max_length"].extend(reshaped.max(-1).values.tolist())
                
                # 5. Ratio of truncated sequences (completions without EOS token).
                # Note: `eos_idx` is equal to `max_completion_length` for completions with EOS token.
                reshaped = (all_eos_idx == max_completion_length).float().view(-1, num_generations)
                self._metrics[mode]["completions/truncated_ratio"].extend(reshaped.mean(-1).tolist())
            
            
            # 6. Completion examples (table)
            global_step = self.state.global_step
            if mode == 'train':
                global_step += 1    # +1 trick for train
            # Completion table to be logged later
            completion_table = {
                "global_step": [global_step] * len(all_completions_length),
                # "mini_batch": [self.mini_batch_step+1] * len(all_completions_length),
                "problem": gather_object(problems_text),
                "solution": gather_object(solutions_text),
                "completion": gather_object(completion_texts),
                "completion_length": all_completions_length.tolist(),
                # "reward": all_rewards.tolist(),
            }
            
            if self.accelerator.is_main_process:
                # 7. Rewards
                # (7.1) log combined reward (e.g., accuracy reward + format reward)
                reshaped = all_rewards.view(-1, num_generations)    # [batch_size, num_generations]
                self._metrics[mode]["reward"].extend(reshaped.mean(-1).tolist())
                if mode == 'train' or (mode == 'eval' and self.num_eval_generations>1):
                    # reward std: enabled when num_generations > 1
                    self._metrics[mode]["reward_std"].extend(reshaped.std(-1).tolist())
                
                # (7.2) log each specific reward (e.g., accuracy reward and format reward)
                mean_rewards_per_func = all_rewards_per_func.view(-1, num_generations, len(self.reward_funcs)).mean(1)  # average over num_generations
                # Note: batch_size = num_unique_problems * num_generations
                # reshape: [batch_size, num_reward_funcs] -> [num_unique_problems, num_generations, num_reward_funcs]
                if mode == 'train' or (mode == 'eval' and self.num_eval_generations>1):
                    # reward std: enabled when num_generations > 1
                    std_rewards_per_func = all_rewards_per_func.view(-1, num_generations, len(self.reward_funcs)).std(1)
                
                for i, reward_func in enumerate(self.reward_funcs):
                    if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                        reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                    else:
                        reward_func_name = reward_func.__name__
                    
                    # all of each reward: [batch_size, num_reward_funcs]
                    completion_table[reward_func_name] = all_rewards_per_func[:, i].tolist()
                    
                    # mean of each reward: [num_unique_problems, num_reward_funcs]
                    self._metrics[mode][f"rewards/{reward_func_name}"].extend(mean_rewards_per_func[:, i].tolist())
                    # std of each reward: [num_unique_problems, num_reward_funcs]
                    if mode == 'train' or (mode == 'eval' and self.num_eval_generations>1):
                        self._metrics[mode][f"rewards/{reward_func_name}_std"].extend(std_rewards_per_func[:, i].tolist())
                
                # 6.continue Completion examples
                completion_table["reward"] = all_rewards.tolist()
                # if mode == 'eval':  # no 'mini_batch' in eval mode
                #     del completion_table['mini_batch']
                
                # Append current examples to the global buffer: `_completion_examples`
                self._completion_examples[mode].append(completion_table)
    
    
    def _get_batch_logps(self, prompt_completion_ids, attention_mask, logits_to_keep):
        r"""
        Get the bacth per-token log probabilities of prompts/completions under model/ref_model.
        """
        with torch.no_grad():
            # If num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if not self.compute_kl:
                ref_per_token_logps = None
            elif self.ref_model is not None:    # compute kl even when beta=0 (to monitor model updates)
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:   # model with adapter (untested)
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
        
        return old_per_token_logps, ref_per_token_logps
    
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        r"""Get the per-token log probabilities under the given model"""
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        
        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # logits /= (self.temperature + 1e-7) # scale logits by the sampling temperature
        logits /= self.temperature          # scale logits by the sampling temperature
        return selective_log_softmax(logits, input_ids)     # compute logprobs for the input tokens
    
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        r"""
        How the loss is computed by GRPOTrainer. Return loss only.
        """
        # mode = self.mode
        mode = 'train'

        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        # Get per_token_logps in train mode
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        
        # Compute the KL divergence between the model and the reference model
        if self.compute_kl:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
        
        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # `_generate_score_log_completions`) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        # Clip higher to promote diversity and precent entropy collapse (ref: DAPO)
        # coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon_high)
        per_token_adv1 = coef_1 * advantages.unsqueeze(1)
        per_token_adv2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_adv1, per_token_adv2)
        
        # Track policy gradient loss
        # TODO: Check if this is correct
        pg_loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["pg_loss"].append(self.accelerator.gather_for_metrics(pg_loss).mean().item())
        
        if self.beta > 0.0:    # add kl loss if beta>0
            per_token_loss += self.beta * per_token_kl
        
        # Note: `completion_mask` could be all zeros if `mask_truncated_completions` is enabled!
        # To mitigate this, consider to use 'completion_mask.sum().clamp(min=1.0)' as the divisor (or other appropriate operations).
        
        # Sequence-level loss: 
        if self.loss_type == "grpo":
            # Each sequence has an equal weight in the final loss computation
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        
        # Token-level loss:
        elif self.loss_type == "bnpo":
            # Normalization is performed over the local batch only.
            # Longer sequences can have more influence on the overall gradient update, but 
            # particular generation pattern may help to train the model regardless of the response length.
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        
        # Remove length bias by using masked_sum with a constant normalizer (ref: Dr. GRPO)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / self.max_completion_length
            # loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Compute kl even when beta=0, to monitor model update
        if self.compute_kl:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        is_clipped = (coef_1 < (1 - self.epsilon)) | (coef_1 > (1 + self.epsilon_high))
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["advantage_clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        
        return loss
    
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            # self.log(logs, start_time)
            self.log(logs, mode='train', start_time=start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def log(
        self, 
        logs: dict[str, float], 
        mode: str,
        start_time: Optional[float] = None
    ) -> None:
        r"""
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Note: 
            Log in rank 0 only! CallbackHandler will do it for you.
            `if state.is_world_process_zero`

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        
        Ref:
        Trainer.log(): 
            https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer.py#L3631
        """
        
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)
        
        # Log grpo train/eval metrics
        # mode = self.mode
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics
        # Note: averaged over unique-samples, so max/min completion length means the averaged value across unique-samples.
        
        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        
        # Log completion exmaples (i.e., prompt + solution + score) tables in main process
        if (
            self.accelerator.is_main_process and
            self.log_completions and 
            # self.args.report_to and "wandb" in self.args.report_to and
            self._completion_examples[mode]
        ):
            # Convert each dictionary in _completion_examples[mode] to a DataFrame and concatenate them
            df = pd.concat(
                [pd.DataFrame(d) for d in self._completion_examples[mode]], 
                ignore_index=True
            )
            wandb.log({f"{mode}_completions": wandb.Table(dataframe=df)})
        
        logs = {**logs, **metrics}      # raw logs + grpo metrics
        # output = {**logs, **metrics}    # raw logs + grpo metrics
        
        # reset `_metrics` and `_completion_examples` buffers
        self._metrics[mode].clear()
        self._completion_examples[mode].clear()
        
        self.state.log_history.append(logs)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        
        # Ref: 
        # 1. CallbackHandler.on_log
        #   self.call_event("on_log", args, state, control, logs=logs)
        #   https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#L555
        # 2. `ProgressCallback.on_log`
        #   def on_log(self, args, state, control, logs=None, **kwargs):
        #   https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#L677
        # 3. `WandbCallback.on_log`
        #   def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        #   https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/integrations/integration_utils.py#L957
    
    
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        r"""
        Run evaluation and returns metrics.
        
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        
        Ref:
            # https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer.py#L4086
            # https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer.py#L4254
        """
        
        mode = 'eval'
        self.mode = mode
        
        # handle multipe eval datasets
        # TODO: add support for list of eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics
        
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            dataloader = tpu_spmd_dataloader(dataloader)

        
        # debug(0)

        start_time = time.time()
        
        # `evaluation_loop`
        # Ref: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L4205
        description="Evaluation"
        args = self.args
        
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        
        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        
        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        
        # batch_size = self.args.eval_batch_size
        effective_batch_size = self.args.per_device_eval_batch_size * self.accelerator.num_processes
        logger.info("\n\n***** Evaluate *****")
        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            num_examples = self.num_examples(dataloader)
            logger.info(f"  Num examples (raw) = {num_examples}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Num generations = {self.num_eval_generations}")
        # logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_eval_batch_size}")
        logger.info(f"  Total eval batch size = {effective_batch_size}")
        if has_length(dataloader):
            num_update_steps_per_epoch = math.ceil(num_examples * self.num_eval_generations / effective_batch_size)
            logger.info(f"  Num updates per epoch (ceiled) = {num_update_steps_per_epoch}")
        logger.info(f"  Max completion length  = {self.max_eval_completion_length}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()
        
        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None
        
        # Initialize containers/metrics
        metrics = {}
        
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            
            # Prediction step: prepare grpo inputs
            # Ref: self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            #   https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer.py#L4487
            
            # generate, score, and log completions (`_generate_score_log_completions` in eval mode)
            # grpo_inputs = self.prepare_grpo_inputs(inputs)
            # TODO: return None if grpo_inputs is not used.

            # 1. Generate completions: augment inputs with generated completions
            processed_inputs = self._generate_completions(inputs, mode='eval')
            # 2. Score completions: get rewards for each completion
            all_rewards_per_func, all_rewards, all_advantages = self._score_completions(processed_inputs, mode='eval')
            # 3. Log completions
            self._log_completions(processed_inputs, all_rewards_per_func, all_rewards, mode='eval')

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            # Ref: ProgressCallback.on_prediction_step
            # https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer_callback.py#L656
            
            del inputs, processed_inputs
            gc.collect()
            torch.cuda.empty_cache()
        
        
        # debug(0)
        # torch.distributed.breakpoint(rank=1)
        
        # Remove redundant eval samples (in main process only)
        # Note: This is important for the eval metrics for multi-device eval.
        # Example:
        # Suppose we have 7 samples (e.g., [0, 1, 2, 3, 4, 5, 6]) in the eval dataset and 2 gpu devices,
        # if eval batch size is set to be 5, then we will have 2 batches: 
        # [0, 1, 2, 3, 4] and [5, 6, 0, 1, 2].
        # We need to remove the redundant samples in the last batch.
        if self.accelerator.is_main_process:
            num_samples = len(eval_dataset)
            keyword = list(self._metrics[mode].keys())[0]
            num_total_samples = len(self._metrics[mode][keyword])
            if num_total_samples > num_samples:
                new_dic = {k:v[:num_samples] for k,v in self._metrics[mode].items()}
                self._metrics[mode].update(new_dic)     # update eval dict
                
                
                num_redundant = num_total_samples - num_samples
                self._completion_examples[mode][-1] = {k:v[:-num_redundant] for k,v in self._completion_examples[mode][-1].items()}
                # new_dic = {k:v[:-num_redundant] for k,v in self._completion_examples[mode][-1].items()}
                # {k:len(v) for k,v in new_dic.items()}
        
        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")
        
        # total_batch_size = self.args.eval_batch_size * self.args.world_size
        # metrics.update(
        #     speed_metrics(
        #         metric_key_prefix,
        #         start_time,
        #         num_samples=num_samples,
        #         num_steps=math.ceil(num_samples / total_batch_size),
        #     )
        # )
        
        # log/print eval metrics
        if self.accelerator.is_main_process:
            # mode = 'eval'
            metrics['global_step'] = self.state.global_step
            # Prefix all keys with metric_key_prefix + '_'
            for k, vals in self._metrics[mode].items():
                if not k.startswith(f"{metric_key_prefix}_"):
                    k = f"{metric_key_prefix}_{k}"
                metrics[k] = np.mean(vals)
            
            print('\n\n')
            self.log_metrics(mode, metrics)
            # Note: To save the best model checkpoint in terms of accuracy, i.e., `save_strategy` is best,
            # 'args.metric_for_best_model' can be set to 'xxx/accuracy_reward'.
        
        
        # debug(0)
        
        # Log `_metrics` and `_completion_examples` in wandb
        # self.log(metrics)
        self.log({}, mode='eval')
        
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)       
        
        self._memory_tracker.stop_and_update_metrics(metrics)
        
        
        # debug(0)
        
        return metrics
    
    
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ):
        r"""
        Main training entry point.

        Here, we overwrite Trainer.train() to inject custom grpo iteration.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`list[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        
        Ref: Trainer.train: 
            https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer.py#L2139

        """
        # TODO: dynamic sampling: 
        # skip too simple, resample hard

        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train and not self.is_model_parallel:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None
        
        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size
        
        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model
        
        # `Trainer._inner_training_loop`
        # Ref: 
        #   https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/trainer.py#L2059
        #   https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer.py#L2252
        
        self.accelerator.free_memory()
        
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)
 
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        
        epoch_based = True      # at present, we only support epoch_based
        num_train_epochs = math.ceil(args.num_train_epochs)
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
        else:
            raise ValueError(
                "'train_dataloader' must have a length!"
            )
        # num_update_steps_per_epoch = max(len_dataloader // args.gradient_accumulation_steps, 1)   
        # Revised as below: ceiled
        # num_update_steps_per_epoch = (len_dataloader + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
        num_update_steps_per_epoch = math.ceil(len_dataloader / args.gradient_accumulation_steps)
        num_examples = self.num_examples(train_dataloader)  # get raw dataset length (no num_generations)
        num_train_samples = num_examples * self.num_generations * num_train_epochs  # multiply num_generations
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)   # max update steps, shown in traing bar
        
        
        
        # debug(0)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)
        
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.
        
        # TODO: clean gradient accumulation

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num epochs = {num_train_epochs:,}")
        logger.info(f"  Num examples (raw train dataset) = {num_examples:,}")
        logger.info(f"  Num generations = {self.num_generations:,}")
        logger.info(f"  Num examples (w. num_generations) = {num_examples*self.num_generations:,}")
        # TODO: grpo num_iterations
        # logger.info(f"  Num iterations (𝜇 in GRPO) = {self.num_iterations:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
        
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        
        # logger.info(f"  Total train sample size = {num_train_epochs*num_examples*self.num_generations:,}")
        
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        # logger.info(f"  Total train batch size = {total_train_batch_size:,}")
        # logger.info(f"                         = (world_size * per_device_train_batch_size * gradient_accumulation_steps)")
        
        logger.info(f"  Training steps per epoch (ceiled) = {num_update_steps_per_epoch:,}")
        logger.info(f"  Total training steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")
        
        
        # debug(0)

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None
        
        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        # self.state.init_training_references(self, train_dataloader, max_steps, num_train_epochs, trial)     # v4.49.0
        self.state.init_training_references(self, max_steps, num_train_epochs, trial)     # v4.51.0

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)
        
        # Training progress bar
        if self.is_world_process_zero():
            progress_bar = tqdm(
                total=max_steps, 
                dynamic_ncols=True, 
                desc='Training step'
            )
        
        # debug(0)
        
        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1   # note: should take num_iterations into consideration when using `step`
            epoch_iterator = iter(epoch_dataloader)
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                
                # TODO: deal with num_iterations (grpo iteration)
                do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                # do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch * self.num_iterations
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)
                
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_tokens = inputs[main_input_name].numel()
                        input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                        self.state.num_input_tokens_seen += (
                            self.accelerator.gather(input_tokens).sum().cpu().item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None
                
                # Should take `num_iterations` into consideration or move it before grpo iteration starts
                # if step % (args.gradient_accumulation_steps * self.num_iterations) == 0:
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                
                # TODO: generate and score and check here, 
                # skip this batch if it's bad

                # Rejection sampling for model generation
                max_attempts = max(self.max_resample_attempts, 1) # maximum number of generation attempts
                for attempt in range(1, max_attempts+1):
                    # grpo_inputs = self._generate_score_log_completions(inputs, attempt=attempt)
                    
                    # 1. Generate completions: augment inputs with generated completions
                    processed_inputs = self._generate_completions(inputs, mode='train')
                    
                    # 2. Score completions: get rewards for each completion
                    all_rewards_per_func, all_rewards, all_advantages = self._score_completions(processed_inputs, mode='train')
                    # Get the local slice of advantages (enabled only in train mode)
                    
                    # debug(0)
                    
                    start = self.accelerator.process_index * len(processed_inputs['prompt_ids'])
                    advantages = all_advantages[start:start+len(processed_inputs['prompt_ids'])]      # [i:i+len(prompts)]
                    
                    # TODO
                    processed_inputs['advantages'] = advantages   # for compute loss
                    # processed_inputs['rewards'] = all_rewards         # for check model generation
                    
                    
                    # 3. Check if current model generation meets specific criterion
                    checkout = self.check_model_generation(all_rewards.tolist(), **kwargs)
                    
                    if checkout == 'easy':
                        logger.info(f"\n  [{attempt=}]: Current problems are too easy for the model to solve, skip to the next batch.")
                        break
                    elif checkout == 'good':
                        # logger.info(f"\n  [rank={self.accelerator.process_index}][attempt={attempt}]: Current model generation is good.")
                        logger.info(f"\n  [{attempt=}]: Current model generation is good.")
                        break
                    elif checkout == 'bad':
                        if attempt < max_attempts:
                            logger.info(f"\n  [{attempt=}]: Current model generation is bad, try again...")
                        else:
                            logger.info(f"\n  [{attempt=}]: Current model generation is bad, skip to the next batch.")
                    elif checkout == 'hard':
                        if attempt < max_attempts:
                            logger.info(f"\n  [{attempt=}]: Current model generation is not good, try again...")
                        else:
                            logger.info(f"\n  [{attempt=}]: Current model generation is not good, but max_attempts reached!")
                
                if checkout in ['easy', 'bad']:
                    # Too easy/bad, skip to the next batch
                    # TODO: do progrees callback
                    # Only update progress_bar, while global_step remains unchanged!
                    # self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    # self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    if self.is_world_process_zero():
                        progress_bar.update(1)
                        print()

                    continue
                
                # checkout is good or hard at this point

                # 4. Log completions
                self._log_completions(processed_inputs, all_rewards_per_func, all_rewards, mode='train')

                # # Synchronize: wait for all processes to finish before continuing
                # self.accelerator.wait_for_everyone()

                # Get per token logps (old_per_token_logps and ref_per_token_logps)
                # TODO: Note: call `_get_batch_logps` after the last attempt.
                # Concatenate prompt with completion for logit computation: (batch, prompt + completion)
                prompt_completion_ids = torch.cat([processed_inputs['prompt_ids'], processed_inputs['completion_ids']], dim=1)  
                attention_mask = torch.cat([processed_inputs['prompt_mask'], processed_inputs['completion_mask']], dim=1)
                logits_to_keep = processed_inputs['completion_ids'].size(1)  # we only need to compute the logits for the completion tokens
                old_per_token_logps, ref_per_token_logps = self._get_batch_logps(
                    prompt_completion_ids, attention_mask, logits_to_keep
                )
                processed_inputs['old_per_token_logps'] = old_per_token_logps
                processed_inputs['ref_per_token_logps'] = ref_per_token_logps
                
                # model inputs are well prepared now, goto training_step

                # TODO: add grpo iteration, but should move global_step, epoch and on_step_end outside
                 # -----------------------------------------------
                # Train the batch sample in multi-grpo iterations
                for iteration in range(self.num_iterations):
                    self.grpo_iteration = iteration     # tracks current grpo_iteration
                # -----------------------------------------------

                    # TODO: clean context
                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    # context = (
                    #     functools.partial(self.accelerator.no_sync, model=model)
                    #     if i != len(batch_samples) - 1
                    #     and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                    #     else contextlib.nullcontext
                    # )
                    context = contextlib.nullcontext
                    with context():
                    # with self.accelerator.accumulate(model):
                        # Training step for one batch sample
                        tr_loss_step = self.training_step(model, processed_inputs)
                    
                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(processed_inputs))

                    
                    if do_sync_step:
                        logger.info(
                            f"optimizer.step: "
                            # f"grpo_iteration={self.grpo_iteration+1}"
                            # f"\n    global_step={self.state.global_step+1}, grpo_iteration={self.grpo_iteration+1}, mini_batch_step (grad. acc.)={self.mini_batch_step+1}"
                        )
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm
                        
                        # Update model parameter in each iteration
                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
                        
                        # if not self.accelerator.optimizer_step_was_skipped:
                        #     # Delay optimizer scheduling until metrics are generated
                        #     if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        #         self.lr_scheduler.step()

                        model.zero_grad()
                        
                        # # update global step (after optimizer step)
                        # self.state.global_step += 1
                        # self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        # self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        
                        # # update training progress bar
                        # if self.is_world_process_zero():
                        #     progress_bar.update(1)
                        
                        # self._maybe_log_save_evaluate(
                        #     tr_loss, 
                        #     grad_norm, 
                        #     model, 
                        #     trial, 
                        #     epoch, 
                        #     ignore_keys_for_eval, 
                        #     start_time
                        # )
                    
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                    
                # After multiple grpo iterations of one batch (instead of inside the loop)
                # Update `lr_scheduler`, `global_step`, `epoch`, and call `on_step_end`
                if not self.accelerator.optimizer_step_was_skipped:
                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()
                
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                
                # update training progress bar
                if self.is_world_process_zero():
                    progress_bar.update(1)
                
                self._maybe_log_save_evaluate(
                    tr_loss, 
                    grad_norm, 
                    model, 
                    trial, 
                    epoch, 
                    ignore_keys_for_eval, 
                    start_time
                )

                # PyTorch/XLA relies on the data loader to insert the mark_step for
                # each step. Since we are breaking the loop early, we need to manually
                # insert the mark_step here.
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            
            # We also need to break out of the nested loop
            if self.control.should_epoch_stop or self.control.should_training_stop:
                if is_torch_xla_available():
                    xm.mark_step()
                break

            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
            )
            
            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        # Note: Training is completed at this point.
        
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics, mode='train')

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    