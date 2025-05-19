import json
import warnings
from typing import Any, Optional
from dataclasses import dataclass, field, fields

import trl
from transformers import TrainingArguments
# from trl import ScriptArguments


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`):
            Dataset name.
        dataset_config (`str` or `None`, *optional*, defaults to `None`):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """
    
    # dataset_name: str = field(
    #     default=None,
    #     metadata={"help": "Dataset name."}
    # )
    
    train_dataset_name: str = field(
        default=None,
        metadata={"help": "Train dataset name."},
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Test dataset name."},
    )
    
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )
    # dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    # dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient checkpointing."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )

    max_num_train_samples: int = field(
        default=-1,
        metadata={"help": "Max number of samples used for training."},
    )
    
    max_num_test_samples: int = field(
        default=None,
        metadata={"help": "Max number of samples used for test."},
    )

    check_gpu_idle: bool = field(
        default=False,
        metadata={
            "help": "If set, wait for GPU to be idle before running. If not set, run scripts without checking GPU."
        },
    )

    gpu_memory_threshold: int = field(
        default=1000,
        metadata={
            "help": "GPU memory usage threshold in MB to consider GPU idle."
        },
    )
    
    gpu_util_threshold: int = field(
        default=10,
        metadata={
            "help": "GPU utilization threshold in percentage to consider GPU idle."
        },
    )
    
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return json.dumps(d, indent=2)



@dataclass
class ModelArguments:
    """
    Configuration class for the models.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        model_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Model checkpoint for weights initialization.
        model_revision (`str`, *optional*, defaults to `"main"`):
            Specific model version to use. It can be a branch name, a tag name, or a commit id.
        torch_dtype (`Literal["auto", "bfloat16", "float16", "float32"]` or `None`, *optional*, defaults to `None`):
            Override the default `torch.dtype` and load the model under this dtype. Possible values are

                - `"bfloat16"`: `torch.bfloat16`
                - `"float16"`: `torch.float16`
                - `"float32"`: `torch.float32`
                - `"auto"`: Automatically derive the dtype from the model's weights.

        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow for custom models defined on the Hub in their own modeling files. This option should only
            be set to `True` for repositories you trust and in which you have read the code, as it will execute code
            present on the Hub on your local machine.
        attn_implementation (`str` or `None`, *optional*, defaults to `None`):
            Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in which case
            you must install this manually by running `pip install flash-attn --no-build-isolation`.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether to use PEFT for training.
        lora_r (`int`, *optional*, defaults to `16`):
            LoRA R value.
        lora_alpha (`int`, *optional*, defaults to `32`):
            LoRA alpha.
        lora_dropout (`float`, *optional*, defaults to `0.05`):
            LoRA dropout.
        lora_target_modules (`Union[str, list[str]]` or `None`, *optional*, defaults to `None`):
            LoRA target modules.
        lora_modules_to_save (`list[str]` or `None`, *optional*, defaults to `None`):
            Model layers to unfreeze & train.
        lora_task_type (`str`, *optional*, defaults to `"CAUSAL_LM"`):
            Task type to pass for LoRA (use `"SEQ_CLS"` for reward modeling).
        use_rslora (`bool`, *optional*, defaults to `False`):
            Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, instead of
            the original default value of `lora_alpha/r`.
        use_dora (`bool`, *optional*, defaults to `False`):
            Enable [Weight-Decomposed Low-Rank Adaptation (DoRA)](https://huggingface.co/papers/2402.09353). This
            technique decomposes the updates of the weights into two parts, magnitude and direction. Direction is
            handled by normal LoRA, whereas the magnitude is handled by a separate learnable parameter. This can
            improve the performance of LoRA, especially at low ranks. Right now, DoRA only supports linear and Conv2D
            layers. DoRA introduces a bigger overhead than pure LoRA, so it is recommended to merge weights for
            inference.
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            Whether to use 8 bit precision for the base model. Works only with LoRA.
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            Whether to use 4 bit precision for the base model. Works only with LoRA.
        bnb_4bit_quant_type (`str`, *optional*, defaults to `"nf4"`):
            Quantization type (`"fp4"` or `"nf4"`).
        use_bnb_nested_quant (`bool`, *optional*, defaults to `False`):
            Whether to use nested quantization.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use. It can be a branch name, a tag name, or a commit id."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train."},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type to pass for LoRA (use 'SEQ_CLS' for reward modeling)."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, "
            "instead of the original default value of `lora_alpha/r`."
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": "Enable Weight-Decomposed Low-Rank Adaptation (DoRA). This technique decomposes the updates of "
            "the weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
            "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
            "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a "
            "bigger overhead than pure LoRA, so it is recommended to merge weights for inference."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8 bit precision for the base model. Works only with LoRA."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4 bit precision for the base model. Works only with LoRA."},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type.", "choices": ["fp4", "nf4"]},
    )
    use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use nested quantization."},
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if hasattr(self.lora_target_modules, "__len__") and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]
    
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return json.dumps(d, indent=2)
    

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )


    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return json.dumps(d, indent=2)


@dataclass
class GRPOTrainingArguments(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """
    
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )
    
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )

    max_resample_attempts: int = field(
        default=10,
        metadata={"help": "Max number of attempts per prompt in model generation."},
    )
    
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(
        default=None, 
        metadata={"help": "The chat template to use."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(
        default=False, 
        metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, 
        metadata={"help": "Whether to push to a Hub revision/branch."}
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )

    compute_kl: bool = field(
        default=False, 
        metadata={"help": "Whether to compute kl even when beta=0. This helps to monitor model update."}
    )

    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )

    loss_type: str = field(
        default="bnpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are `grpo`, `bnpo`, and `dr_grpo`. "
            "`'grpo'`: Aggregates sequence-level losses by normalizing over sequence length. Not recommended due to "
                "length bias—this approach tends to prefer shorter completions with positive advantages and longer ones "
                "with negative advantages. "
            "`'bnpo'`: Aggregates token-level losses by normalizing number of active token in the local batch. "
                "Note that normalization is performed over the local batch only, so results may slightly vary depending "
                "on the local batch size, despite a constant effective batch size. When using "
                "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss. "
            "`'dr_grpo'`: Aggregates token-level losses by normalizing with a global constant. This method was "
                "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
                "`max_completion_length`."
        },
    )

    # Eval parameters
    num_eval_generations: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    max_eval_completion_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of the generated completion."},
    )
    eval_temperature: float = field(
        default=None,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    eval_top_p: float = field(
        default=None,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    eval_top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled."
        },
    )
    eval_min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )



@dataclass
class SFTArguments(TrainingArguments):
    r"""
    Configuration class for the [`SFTTrainer`].

    Only the parameters specific to SFT training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`SFTTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the column that contains text data in the dataset.
        dataset_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Dictionary of optional keyword arguments for the dataset preparation. The only supported key is
            `skip_prepare_dataset`.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the right.
            If `None`, no truncation is applied. When packing is enabled, this value sets the sequence length.
        packing (`bool`, *optional*, defaults to `False`):
            Whether to pack multiple sequences into a fixed-length format. Uses `max_length` to define sequence length.
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the `flash_attention_2` attention implementation, which can efficiently handle the flattened
            batch structure.
        eval_packing (`bool` or `None`, *optional*, defaults to `None`):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `2e-5`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
    """

    # Parameters that control the model
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `SFTTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the column that contains text data in the dataset."},
    )
    dataset_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Dictionary of optional keyword arguments for the dataset preparation. The only supported key is "
            "`skip_prepare_dataset`."
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from"
            "the right. If `None`, no truncation is applied. When packing is enabled, this value sets the "
            "sequence length."
        },
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to pack multiple sequences into a fixed-length format. Uses `max_length` to define "
            "sequence length."
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, "
            "this is only supported with the `flash_attention_2` attention implementation, which can efficiently "
            "handle the flattened batch structure."
        },
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing`."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=2.0e-5,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`TrainingArguments`."
        },
    )
    
    # Benchamarks
    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )

    # Template
    chat_template: Optional[str] = field(
        default=None, 
        metadata={
            "help": "The chat template to use."
                    "Will override the default chat_template of tokenizer!"
        }
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    
    #  Wandb parameters
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


    # `use_liger_kernel` is already defined in `TrainingArguments`.
    # use_liger_kernel: Optional[bool] = field(
    #     default=False,
    #     metadata={"help": "Whether or not to enable the Liger Kernel for model training."},
    # )
    # use_liger_kernel (`bool`, *optional*, defaults to `False`):
    #     Whether enable [Liger](https://github.com/linkedin/Liger-Kernel) Kernel for LLM model training.
    #     It can effectively increase multi-GPU training throughput by ~20% and reduces memory usage by ~60%, works out of the box with
    #     flash attention, PyTorch FSDP, and Microsoft DeepSpeed. Currently, it supports llama, mistral, mixtral and gemma models.


    # Deprecated parameters
    dataset_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated. You can safely remove this parameter from your configuration."},
    )
    num_of_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated. Use `max_length` instead, which specifies the maximum length of the tokenized "
            "sequence, unlike `num_of_sequences`, which referred to string sequences."
        },
    )
    chars_per_token: Optional[float] = field(
        default=None,
        metadata={"help": "Deprecated. If you want to customize the packing length, use `max_length`."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated. Use `max_length` instead."},
    )
    use_liger: Optional[bool] = field(
        default=None,
        metadata={"help": "Deprecated. Use `use_liger_kernel` instead."},
    )

    
    def __post_init__(self):
        super().__post_init__()

        if self.dataset_batch_size is not None:
            warnings.warn(
                "`dataset_batch_size` is deprecated and will be remove in version 0.18.0. You can safely remove this "
                "parameter from your configuration.",
                DeprecationWarning,
            )

        if self.num_of_sequences is not None:
            warnings.warn(
                "`num_of_sequences` is deprecated and will be remove in version 0.18.0. Use `max_length` instead, "
                "which specifies the maximum length of the tokenized sequence, unlike `num_of_sequences`, which r"
                "eferred to string sequences.",
                DeprecationWarning,
            )

        if self.chars_per_token is not None:
            warnings.warn(
                "`chars_per_token` is deprecated and will be remove in version 0.18.0. If you want to customize the "
                "packing length, use `max_length`.",
                DeprecationWarning,
            )

        if self.max_seq_length is not None:
            warnings.warn(
                "`max_seq_length` is deprecated and will be remove in version 0.20.0. Use `max_length` instead.",
                DeprecationWarning,
            )
            self.max_length = self.max_seq_length

        if self.use_liger is not None:
            warnings.warn(
                "`use_liger` is deprecated and will be remove in version 0.18.0. Use `use_liger_kernel` instead.",
                DeprecationWarning,
            )
            self.use_liger_kernel = self.use_liger
    


# @dataclass
# class SFTArguments(trl.SFTConfig):
#     """
#     args for callbacks, benchmarks etc
#     """

#     benchmarks: list[str] = field(
#         default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
#     )
#     callbacks: list[str] = field(
#         default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
#     )
#     chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
#     system_prompt: Optional[str] = field(
#         default=None,
#         metadata={"help": "The optional system prompt to use for benchmarking."},
#     )
#     hub_model_revision: Optional[str] = field(
#         default="main",
#         metadata={"help": "The Hub model branch to push the model to."},
#     )
#     overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
#     push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
#     wandb_entity: Optional[str] = field(
#         default=None,
#         metadata={"help": ("The entity to store runs under.")},
#     )
#     wandb_project: Optional[str] = field(
#         default=None,
#         metadata={"help": ("The project to store runs under.")},
#     )

#     padding_free: bool = field(
#         default=False,
#         metadata={
#             "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
#             "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, "
#             "this is only supported with the `flash_attention_2` attention implementation, which can efficiently "
#             "handle the flattened batch structure."
#         },
#     )
    

#     def to_json_string(self):
#         """
#         Serializes this instance to a JSON string.
#         """
#         d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
#         return json.dumps(d, indent=2)


