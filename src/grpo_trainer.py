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
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch


import pandas as pd
# --------------------------------
from transformers.trainer import *

# `_xxx`` cannot be imported by `from module import *`
from transformers.trainer import _is_peft_model
# --------------------------------



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
from trl.import_utils import is_rich_available, is_vllm_available
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

from utils import profiling_context, profiling_decorator

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb


from utils import is_messages

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

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
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

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

    _tag_names = ["trl", "grpo"]

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

        # Eval parameters
        # If not explicitly specified, use parameters from training mode
        self.num_eval_generations = args.num_eval_generations if args.num_eval_generations is not None else args.num_generations
        self.max_eval_completion_length = args.max_eval_completion_length if args.max_eval_completion_length is not None else args.max_completion_length
        self.eval_temperature = args.eval_temperature if args.eval_temperature is not None else args.temperature
        self.eval_top_p = args.eval_top_p if args.eval_top_p is not None else args.top_p
        self.eval_top_k = args.eval_top_k if args.eval_top_k is not None else args.top_k
        self.eval_min_p = args.eval_min_p if args.eval_min_p is not None else args.min_p


        self.use_vllm = args.use_vllm

        # Multi-step
        self.num_iterations = args.num_iterations   # = 𝜇 in the GRPO paper
        # --------------------------
        self.mode = None                            # train or eval mode
        self.grpo_iteration = -1                    # tracks current grpo iteration
        self.mini_batch_step = -1                   # tracks mini-batch step
        self.max_resample_attempts = args.max_resample_attempts         # max number of generation attempts
        self.scale_rewards = args.scale_rewards     # scale the rewards by std or not
        self.mask_truncated_completions = args.mask_truncated_completions   # mask truncated completions
        self.loss_type = args.loss_type

        # Clip higher
        self.epsilon = args.epsilon                 # clip value (pi_theta/pi_old)
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # --------------------------
        
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `prepare_grpo_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

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
            possible_values = [n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0]
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

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                device_type = PartialState().default_device.type
                device_module = getattr(torch, device_type)
                if vllm_device == "auto":
                    if device_module.device_count() == 1:
                        vllm_device = f"{device_type}:0"  # particular case when training with only 1 device: share it
                    else:
                        vllm_device = f"{device_type}:{self.accelerator.num_processes}"  # take the next GPU idx
                
                logger.info(f"{vllm_device=}")
                
                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == f"{device_type}"
                    and int(vllm_device.split(":")[1]) >= device_module.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {device_module.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"{device_type}:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )

                # For Ascend NPU (torch-npu), collective communication requires the establishment of a communication
                # group, and different processes must hold the same group number. However, multiple process groups will
                # be created internally within vLLM. This will cause the group id of the communication group on rank 0
                # to be different from that of other ranks, causing backward to hang on because the communication
                # domain cannot be established. So we need to patch it to make sure the group id of different ranks in
                # the training phase are the same.
                @contextlib.contextmanager
                def new_group_context():
                    new_group = torch.distributed.new_group
                    try:
                        torch.distributed.new_group = functools.partial(new_group, use_local_synchronization=True)
                        torch.npu.mem_get_info = functools.partial(torch.npu.mem_get_info, device=vllm_device)
                        yield
                    finally:
                        torch.distributed.new_group = new_group

                new_group_patch = new_group_context() if device_type == "npu" else contextlib.nullcontext()
                with world_size_patch, profiling_patch, new_group_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        # dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                        max_model_len=self.args.vllm_max_model_len,
                    )

                # Guided decoding, if enabled
                if args.vllm_guided_decoding_regex is not None:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=args.vllm_guided_decoding_regex)
                else:
                    guided_decoding = None
                
                # Sampling parameters
                self.sampling_params = SamplingParams(
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                    n=args.num_generations,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=-1 if args.top_k is None else args.top_k,
                    min_p=0.0 if args.min_p is None else args.min_p,
                    repetition_penalty=args.repetition_penalty,
                )

                # Eval sampling parameters
                self.eval_sampling_params = SamplingParams(
                    max_tokens=self.max_eval_completion_length,
                    guided_decoding=guided_decoding,
                    n=args.num_eval_generations,
                    temperature=args.eval_temperature,
                    top_p=args.eval_top_p,
                    top_k=-1 if args.eval_top_k is None else args.eval_top_k,
                    min_p=0.0 if args.eval_min_p is None else args.eval_min_p,
                    repetition_penalty=args.repetition_penalty,
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
        
        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)

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
        # Disable dropout in the model and reference model.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        
        # In the following figure, the values are the prompt indices. The first row shows the first sampled mini-batch, 
        # the second row shows the second sampled mini-batch, and so on.
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
        # Returns a sampler that ensures each prompt is repeated across multiple processes.
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(eval_dataset, self.num_eval_generations, seed=self.args.seed)
    
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
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            
            # # >>>>> add a breakpoint for debug? <<<<<
            # torch.distributed.breakpoint(rank=0)

            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()
    
    def get_train_dataloader(self) -> DataLoader:
        """
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
        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        # Note: After `accelerator.prepare`, len(train_dataloader) may be change if we use data parallel.
        
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        
    
    # Ref: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3668
    def training_step(
        self, model: nn.Module, inputs: list[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        
        # ----------------------------------------
        # Prepare grpo inputs: generate and score the completions
        self.mode = 'train'
        inputs = self.prepare_grpo_inputs(inputs)
        # ----------------------------------------
        
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
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
    

    def check_model_generation(self, inputs):
        """
        Check if current generation meets specific criterion(s).
        e.g., make sure the generated sentences are diverse and contain positive samples.
        """
        
        # Check rewards
        rewards = inputs['rewards'].tolist()
        # 1. full rewards (good and meets criterion)
        # Note: the current question is too easy for the model as it can get full rewrds for all generations.
        if min(rewards) == sum(self.reward_weights.tolist()):
            return True
        
        # 2. accuracy reward (primary reward)
        # If no generation contains the right answer, try it again.
        # if max(rewards) != sum(self.reward_weights.tolist()):
        if max(rewards) < max(self.reward_weights.tolist()):    
            return False
        
        # 3. identical rewards (e.g., all zeros)
        if len(set(rewards)) == 1:
            return False
        
        # TODO: 
        # Check other conditions, e.g., entropy
        
        return True
    
    
    def prepare_grpo_inputs(
        self, 
        inputs: list[str, Union[torch.Tensor, Any]],
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model 
        """
        
        mode = self.mode
        if not mode:
            raise ValueError(f"`mode` should be specified clearly! It could be 'train' or 'eval'.")
        
        if mode == "train":     # train mode
            # if self.state.global_step % self.num_iterations == 0:
            if self.grpo_iteration == 0:
                # Generate and score completions in grpo_iteration=0, and reuse them for later grpo iterations
                # Rejection sampling for model generation
                max_attempts = max(self.max_resample_attempts, 1) # maximum number of generation attempts
                for attempt in range(1, max_attempts+1):
                    grpo_inputs = self._generate_score_log_completions(inputs, attempt=attempt)

                    # Check if current model generation meets specific criterion(s)
                    if self.check_model_generation(grpo_inputs):
                        logger.debug(f"\n  [attempt={attempt}]: Current model generation is good.")
                        break
                    elif attempt == max_attempts:
                        logger.warning(f"\n  [attempt={attempt}]: Current model generation is not good, but max_attempts reached!")
                    else:
                        logger.warning(f"\n  [attempt={attempt}]: Current model generation is not good, try again...")
                    
                    # # >>>>> add a breakpoint for debug? <<<<<
                    # torch.distributed.breakpoint(rank=0)                    
                
                # Get per token logps (old_per_token_logps and ref_per_token_logps)
                # Note: call `_get_batch_logps` after the last attempt.
                # Concatenate prompt with completion for logit computation: (B, P+C)
                prompt_completion_ids = torch.cat([grpo_inputs['prompt_ids'], grpo_inputs['completion_ids']], dim=1)  
                attention_mask = torch.cat([grpo_inputs['prompt_mask'], grpo_inputs['completion_mask']], dim=1)
                logits_to_keep = grpo_inputs['completion_ids'].size(1)  # we only need to compute the logits for the completion tokens
                old_per_token_logps, ref_per_token_logps = self._get_batch_logps(
                    prompt_completion_ids, attention_mask, logits_to_keep
                )
                grpo_inputs['old_per_token_logps'] = old_per_token_logps
                grpo_inputs['ref_per_token_logps'] = ref_per_token_logps

                # Put grpo inputs to the buffer for later reuse
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = grpo_inputs
            else:
                grpo_inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        
        else:           # eval mode
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            grpo_inputs = self._generate_score_log_completions(inputs)
        return grpo_inputs
    
    
    def _generate_completions(self, inputs):
        """
        Generate completions for the given inputs.
        """
        mode = self.mode
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
                all_outputs = self.llm.generate(
                    ordered_set_of_prompts, 
                    sampling_params=sampling_params, 
                    use_tqdm=False
                )

                all_completion_ids = []
                all_completion_texts = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        all_completion_ids.append(output.token_ids)
                        all_completion_texts.append(output.text)

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
            completions_text = all_completion_texts[start:start+len(prompts)]

            # Pad the completions, and concatenate them with the prompts
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
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
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
        # completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])  # multi-turn support
                # completions.append({"role": "assistant", "content": bootstrap + completion})    # single-turn only
        else:
            completions = completions_text

        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)

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
    
    
    def _score_completions(self, inputs):
        """
        Score completions for the given input generations.
        """
        mode = self.mode
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
        
        # If mask_truncated_completions is enabled, zero out rewards of truncated completions
        if self.mask_truncated_completions:
            # eos_idx can indicate the completion length
            eos_idx = inputs['eos_idx']
            all_eos_idx = gather(eos_idx)     # gather from all devices
            all_rewards = all_rewards * (all_eos_idx < max_completion_length).int()
        
        # Compute grouped-wise rewards
        mean_grouped_rewards = all_rewards.view(-1, num_generations).mean(dim=1)
        std_grouped_rewards = all_rewards.view(-1, num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
        
        # Compute advantages
        all_advantages = all_rewards - mean_grouped_rewards
        if self.scale_rewards:
            all_advantages = all_advantages / (std_grouped_rewards + 1e-4)
        
        # Get accuracy reward (i.e., the primary reward)
        # Note: by default, the first reward function is the accuracy reward.
        all_accuracy_reward = all_rewards_per_func[:, 0]
        
        # Print reward info in the main process
        if self.accelerator.is_main_process:
            all_completions_length = all_eos_idx + (all_eos_idx < max_completion_length).int()

            if mode == 'train':
                print()
                logger.debug(
                    f"\n  [{mode}] global_step={self.state.global_step+1},"
                    f" grpo_iteration={self.grpo_iteration+1},"
                    f" mini_batch (grad. acc.)={self.mini_batch_step+1}"
                    # f"\n  accuracy:            {all_accuracy_reward.cpu().tolist()}, "
                    f"\n  completion length:  {all_completions_length.cpu().tolist()}, "
                    f"\n  rewards:            {all_rewards.cpu().tolist()}, "
                    f"\n  advantages:         {[round(x, 3) for x in all_advantages.detach().cpu().tolist()]}"
                )
            else:
                # accuracy is the primary goal!
                print()
                logger.debug(
                    f"\n  [{mode}] global_step={self.state.global_step},"
                    f"\n  completion length:  {all_completions_length.cpu().tolist()}, "
                    f"\n  accuracy reward:    {all_accuracy_reward.cpu().tolist()}"
                    # f"\n  rewards:            {all_rewards.cpu().tolist()}"
                )
        
        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        
        # TODO: keep only the local part?
        return (
            all_rewards_per_func,   # all rewards per function (from all devices)
            all_rewards,            # all rewards (from all devices)
            all_advantages          # all advantages (from all device)
        )
        # Note: 
        # all_rewards_per_func: tracks the the raw reward (0.0-1.0) for each reward function.
        # all_rewards: is the weighted sum of all_rewards_per_func and may apply mask_truncated_completions then.



    def _generate_score_log_completions(
        self, 
        inputs: list[str, Union[torch.Tensor, Any]],
        attempt: int = 1
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Generate and score the completions for given inputs.

        Args:
            inputs (`list[str, Union[torch.Tensor, Any]]`):
                List of input dataset.

        Return:
            `dict[str, Union[torch.Tensor, Any]]`.
        """

        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)

        # Generate completions: augment inputs with generated completions
        inputs = self._generate_completions(inputs)
        
        # Score completions
        all_rewards_per_func, all_rewards, all_advantages = self._score_completions(inputs)
        # get the local slice of advantages
        start = self.accelerator.process_index * len(inputs['prompt_ids'])
        advantages = all_advantages[start:start+len(inputs['prompt_ids'])]      # [i:i+len(prompts)]
        inputs['advantages'] = advantages   # to compute loss
        inputs['rewards'] = all_rewards     # to check model generation

        # Get accuracy reward
        # Note: by default, the first reward function is the accuracy reward.
        all_accuracy_reward = all_rewards_per_func[:, 0]
        inputs['accuracy'] = all_accuracy_reward     # accuracy
        
        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        
        # Only log the first attempt (may change to other values accordingly)
        if attempt == 1:
            self._log_completions(inputs, all_rewards_per_func, all_rewards)
        
        return inputs 
        

    def _log_completions(self, inputs, all_rewards_per_func, all_rewards):
        
        # Log completions
        mode = self.mode
        if mode == 'train':
            num_generations = self.num_generations
            max_completion_length = self.max_completion_length
        else:       # eval mode
            num_generations = self.num_eval_generations
            max_completion_length = self.max_eval_completion_length

        eos_idx = inputs['eos_idx']
        prompts = inputs['prompts']
        completions = inputs['completions']
        # solutions = inputs['solutions']
        
        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        
        # Prompt text (may apply chat template, e.g. with system prompt)
        prompts_text = [
            maybe_apply_chat_template({'prompt': p}, self.processing_class)['prompt'] for p in prompts
        ]
        # Completion text (may extract from conversational message)
        # maybe_message = completions[0][0]
        # if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
        if is_messages(completions[0]):
            completions_text = [example[0]['content'] for example in completions]
            # completions_text = [example['content'] for example in completions]
        else:
            completions_text = completions
        
        # Raw solution text (gold) (may change it to answer only?)
        solutions_text = inputs['solutions']
        
        if (
            self.accelerator.is_main_process and
            self.log_completions and 
            self.state.global_step % self.args.logging_steps == 0 and
            self.args.report_to and "wandb" in self.args.report_to
        ):
            # Note: Log in the unique sample level (num_completions/num_generations), since the last 
            # batch may not be full (especially for eval dataset) and the averaged values are thus not accurate.
            # For example, 
            # suppose we have 3 unique samples, num_generations=2 (6 samples in total)
            # and batch_size=4 (batch_1 gets 4 samples, bacth_2 gets 2 samples).
            # When we get the final resutls: res = [[1,1, 1,1], [0,0]],
            #   batch-level mean:               [[1,1, 1,1], [0,0]] -> [1, 0] -> 0.5
            #   unique-sample-level mean: [[1,1, 1,1], [0,0]] -> [1, 1, 0] -> 2/3 = 0.67
            # This example explains why we use `view(-1, num_generations)` below.

            # 1. Log completion length (including truncated completions): mean, min, max
            all_eos_idx = gather(eos_idx)
            all_completions_length = all_eos_idx + (all_eos_idx < max_completion_length).int()
            # naive mean/min/max
            # self._metrics[mode]["completions/mean_length"].append(all_completions_length.float().mean().item())
            # self._metrics[mode]["completions/min_length"].append(all_completions_length.float().min().item())
            # self._metrics[mode]["completions/max_length"].append(all_completions_length.float().max().item())
            # unique-sample-level mean/min/max
            reshaped = all_completions_length.float().view(-1, num_generations)
            self._metrics[mode]["completions/mean_length"].extend(reshaped.mean(-1).tolist())
            self._metrics[mode]["completions/min_length"].extend(reshaped.min(-1).values.tolist())
            self._metrics[mode]["completions/max_length"].extend(reshaped.max(-1).values.tolist())
            
            
            # # >>>>> add a breakpoint for debug? <<<<<
            # torch.distributed.breakpoint(rank=0)

            # the ratio of truncated sequences (completions without EOS)
            # i.e., `eos_idx` is equal to `max_completion_length`
            # num_truncated = (all_eos_idx == max_completion_length).int().sum().item()
            # self._metrics[mode]["completions/truncated_ratio"].append(num_truncated/len(all_completions_length))
            reshaped = (all_eos_idx == max_completion_length).float().view(-1, num_generations)
            self._metrics[mode]["completions/truncated_ratio"].extend(reshaped.mean(-1).tolist())
            
            
            # 2. Log rewards
            # log combined reward (e.g., accuracy reward + format reward)
            # self._metrics[mode]["reward"].append(all_rewards.mean().item())
            reshaped = all_rewards.view(-1, num_generations)
            self._metrics[mode]["reward"].extend(reshaped.mean(-1).tolist())
            # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

            # log each specific reward (e.g., 1. accuracy reward; 2. format reward)
            # reward_per_func = all_rewards_per_func.mean(0)
            # for i, reward_func in enumerate(self.reward_funcs):
            #     if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
            #         reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            #     else:
            #         reward_func_name = reward_func.__name__
            #     self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            
            rewards_per_func = all_rewards_per_func.view(-1, num_generations, len(self.reward_funcs)).mean(1)  # average over num_generations
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                    reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                else:
                    reward_func_name = reward_func.__name__
                self._metrics[mode][f"rewards/{reward_func_name}"].extend(rewards_per_func[:, i].tolist())
            

            # 3. Log concrete completion examples
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            solutions_to_log = gather_object(solutions_text)
            rewards_to_log = all_rewards.tolist()
            completion_length_to_log = all_completions_length.tolist()
            
            # Completion table to be logged later
            completion_table = {
                "global_step": [self.state.global_step+1] * len(rewards_to_log),
                "mini_batch": [self.mini_batch_step+1] * len(rewards_to_log),
                "prompt": prompts_to_log,
                "solution": solutions_to_log,
                "completion": completions_to_log,
                "completion_length": completion_length_to_log,
                "reward": rewards_to_log,
            }

            # # >>>>> add a breakpoint for debug? <<<<<
            # torch.distributed.breakpoint(rank=0)
            # # {k: len(v) for k,v in completion_table.items()}

            self._completion_examples[mode].append(completion_table)
    
    
    def _get_batch_logps(self, prompt_completion_ids, attention_mask, logits_to_keep):
        
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
    
    
    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        logits /= self.temperature      # scale logits by the sampling temperature
        return selective_log_softmax(logits, input_ids)     # compute logprobs for the input tokens
    
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

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
        # _generate_score_log_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        # coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        # Clip higher to promote diversity and precent entropy collapse (ref: DAPO)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon_high)
        per_token_adv1 = coef_1 * advantages.unsqueeze(1)
        per_token_adv2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_adv1, per_token_adv2)
        if self.beta > 0.0:    # add kl loss when beta>0
            per_token_loss += self.beta * per_token_kl
        
        # Note: completion_mask could be all zeros if mask_truncated_completions is enabled!
        # To mitigate this, consider to use 'completion_mask.sum().clamp(min=1.0)' as the divisor (or other appropriate operations) 
        # instead of merely 'completion_mask.sum()' which may result in nan with 0 as the divisor.
        
        # Sequence-level loss: 
        if self.loss_type == "grpo":
            # Each sequence has an equal weight in the final loss computation
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        
        # Token-level loss:
        elif self.loss_type == "bnpo":
            # Longer sequences can have more influence on the overall gradient update, but 
            # particular generation pattern may help to train the model regardless of the response length.
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        
        # Remove length bias by using masked_sum with a constant normalizer (ref: Dr. GRPO)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / self.max_completion_length
            # loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Log the metrics
        # mode = "eval" if self.control.should_evaluate else "train"
        mode = self.mode
        
        # Compute kl even when beta=0, to monitor model update
        if self.compute_kl:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        
        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        
        # is_clipped = (coef_1 < (1 - self.epsilon)) | (coef_1 > (1 + self.epsilon))
        is_clipped = (coef_1 < (1 - self.epsilon)) | (coef_1 > (1 + self.epsilon_high))
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["advantage_clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
    
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
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
        """
        
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)
        
        # Log grpo train/eval metrics
        # mode = "eval" if self.control.should_evaluate else "train"
        mode = self.mode
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics
        # Note: averaged over unique-samples, so max/min completion length means the averaged value across unique-samples..

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        
        # Log completion exmaples (i.e., prompt + solution + score) tables in main process
        if (
            self.accelerator.is_main_process and
            self.log_completions and 
            # self.state.global_step % self.args.logging_steps == 0 and
            self.args.report_to and "wandb" in self.args.report_to and
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
        
        self._metrics[mode].clear()         # reset _metrics for next log
        self._completion_examples[mode].clear()    # reset completion_examples for next log
        
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
        
        # ------------------------------------
    

    # Ref: https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer.py#L4086
    # Ref: https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer.py#L4254
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
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
        """
        
        self.mode = 'eval'
        
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

        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)

        start_time = time.time()

        # eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        # output = eval_loop(
        #     eval_dataloader,
        #     description="Evaluation",
        #     # No point gathering the predictions if there are no metrics, otherwise we defer to
        #     # self.args.prediction_loss_only
        #     prediction_loss_only=True if self.compute_metrics is None else None,
        #     ignore_keys=ignore_keys,
        #     metric_key_prefix=metric_key_prefix,
        # )
        
        # Ref: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L4205
        # def evaluation_loop()
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """

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
        
        batch_size = self.args.eval_batch_size
        
        print(f"\n\n***** Running {description} *****")
        # logger.info(f"\n***** Running {description} *****")
        logger.info(f"  ***** Parameters*****")
        if has_length(dataloader):
            logger.info(f"  Num examples (raw) = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Num generations = {self.num_eval_generations}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Max completion length  = {self.max_eval_completion_length}")
        
        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()
        
        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None
        
        # >>> continue from here
        
        # Initialize containers/metrics
        metrics = None
        all_accuracies = []
        
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):

            # Prediction step
            # losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # main_input_name = getattr(self.model, "main_input_name", "input_ids")
            # inputs_decode = (
            #     self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            # )
            
            grpo_inputs = self.prepare_grpo_inputs(inputs)

            # Extract useful info from grpo_inputs
            # For example, to compute accuracy, or pass@k
            all_accuracy = grpo_inputs['accuracy'].tolist()
            all_accuracies.extend(all_accuracy)
            
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            # ref: ProgressCallback.on_prediction_step
            # https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/trainer_callback.py#L656

        
        if self.accelerator.is_main_process:
            # Compute custom metrics
            all_accuracies = np.array(all_accuracies).reshape(-1, self.num_eval_generations)
            # mode = 'eval'
            mode = self.mode
            accs = all_accuracies.mean(-1).tolist()
            self._metrics[mode][f"accuracy"].extend(accs)
            
            # # >>>>> add a breakpoint for debug? <<<<<
            # torch.distributed.breakpoint(rank=0)
            # # {k:len(v) for k,v in self._metrics[mode].items()}            
            
            # log the metrics
            metrics = {}
            # self.log(output.metrics)
            self.log(metrics)
            
            # logger.info(f"\n***** Eval results *****")
            print(f"\n\n***** Eval results *****")
            print(f"  Accuracy: {all_accuracies.mean()}")

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)            



    # Ref: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L2140
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

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
        """
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
        

        # inner_training_loop
        # Ref: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L2248
        
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
        
        # Revise num_update_steps_per_epoch by custom function 'set_initial_training_values'
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)
        
        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)

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
        
        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num epochs = {num_train_epochs:,}")
        logger.info(f"  Num examples (raw train dataset) = {num_examples:,}")
        logger.info(f"  Num examples (w. num_generations) = {num_examples*self.num_generations:,}")
        logger.info(f"  Num generations = {self.num_generations:,}")
        logger.info(f"  Num iterations (𝜇 in GRPO) = {self.num_iterations:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
        
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        
        # logger.info(f"  Total train sample size = {num_train_epochs*num_examples*self.num_generations:,}")
        
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        # logger.info(f"  Total train batch size = {total_train_batch_size:,}")
        # logger.info(f"                         = (world_size * per_device_train_batch_size * gradient_accumulation_steps)")
        
        logger.info(f"  Num update steps per epoch (ceiled) = {num_update_steps_per_epoch:,}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")
        
        # # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)

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
            
            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            # remainder = num_examples % args.gradient_accumulation_steps
            # TODO: the above estimation is not accurate in current setting, change it to the below:
            remainder = len_dataloader % args.gradient_accumulation_steps
            num_mini_batches = args.gradient_accumulation_steps
            for update_step in range(num_update_steps_per_epoch):
                # One batch sample
                if update_step == num_update_steps_per_epoch - 1 and remainder != 0:
                    num_mini_batches = remainder
                
                # Split one batch to multiple mini-batches
                # batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_mini_batches)
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_mini_batches, args.device)
                
                # -----------------------------------------------
                # Train the batch sample in multi-grpo iterations
                for iteration in range(self.num_iterations):
                    self.grpo_iteration = iteration     # tracks current grpo_iteration
                # -----------------------------------------------
                    # Train mini-batch (one batch contains multiple mini-batches)
                    for i, inputs in enumerate(batch_samples):  # inputs: one mini-batch
                        self.mini_batch_step = i
                        # One mini-batch (in total, there are `gradient_accumulation_steps` mini-batches in each batch)
                        # Back propagate when the last mini-batch is finished
                        step += 1   # note: should take num_iterations into consideration when using `step`
                        # do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                        do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch * self.num_iterations
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
                        
                        # if step % args.gradient_accumulation_steps == 0:
                        # Should take `num_iterations` into consideration or move it before grpo iteration starts
                        if step % (args.gradient_accumulation_steps * self.num_iterations) == 0:
                            self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                        # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                        context = (
                            functools.partial(self.accelerator.no_sync, model=model)
                            if i != len(batch_samples) - 1
                            and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                            else contextlib.nullcontext
                        )
                        with context():
                            # Training step for one mini-batch
                            tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

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

                        self.current_flos += float(self.floating_point_ops(inputs))

                        if do_sync_step:
                            logger.debug(
                                f"optimizer.step: "
                                f"grpo_iteration={self.grpo_iteration+1}"
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

                            # self.state.global_step += 1
                            # self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                            # self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                            # self._maybe_log_save_evaluate(
                            #     tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                            # )
                        else:
                            self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                        # PyTorch/XLA relies on the data loader to insert the mark_step for
                        # each step. Since we are breaking the loop early, we need to manually
                        # insert the mark_step here.
                        if self.control.should_epoch_stop or self.control.should_training_stop:
                            if is_torch_xla_available():
                                xm.mark_step()
                            break
                
                # -----------------------------------------------
                # End of one batch step (i.e., global_step): `lr_scheduler`, `state`, `_maybe_log_save_evaluate`
                if not self.accelerator.optimizer_step_was_skipped:
                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()
                
                # if do_sync_step:  # After each batched samples, `do_sync_step` is True.
                self.state.global_step += 1
                # self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                self.state.epoch = epoch + (step + 1 + steps_skipped) / self.num_iterations / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(
                    tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                )

                # num_update_steps_per_epoch = max(len_dataloader // args.gradient_accumulation_steps, 1)   # // -> floored?
                # max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                # -----------------------------------------------

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
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

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

        self.log(metrics)

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


        
    # Overwrite `set_initial_training_values` as it's not accurate in estimating: 
    # - num_update_steps_per_epoch
    # - num_train_samples
    def set_initial_training_values(
        self, args: 'TrainingArguments', train_dataloader: 'DataLoader', total_train_batch_size: int
    ):
        """
        Calculates and returns the following values:
        - `num_train_epochs`
        - `num_update_steps_per_epoch`
        - `num_examples`
        - `num_train_samples`
        - `epoch_based`
        - `len_dataloader`
        - `max_steps`
        """
        
        logger.warning(
            "We use a custom 'set_initial_training_values' function in this project. "
            "You may consider to change this function in other projects if the output is not desired."
        )

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
        
        return (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        )
    