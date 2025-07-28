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
    
    # train_dataset_name: str = field(
    #     default=None,
    #     metadata={"help": "Train dataset name."},
    # )
    # eval_dataset_name: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Test dataset name."},
    # )

    train_dataset_name: list[str] = field(
        default=None,
        # default_factory=lambda: [], 
        metadata={"help": "The list of train dataset names to run."},
    )
    
    eval_dataset_name: list[str] = field(
        default_factory=lambda: [], 
        metadata={"help": "The list of eval/test dataset names to run."}
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
    
    # Deprecated parameter, use `num_train_samples_per_dataset` instead
    max_num_train_samples: int = field(
        default=None,
        metadata={"help": "Max number of samples used per dataset for training."},
    )
    
    # Deprecated parameter, use `num_eval_samples_per_dataset` instead
    max_num_test_samples: int = field(
        default=None,
        metadata={"help": "Max number of samples used per dataset for eval/test."},
    )
    
    num_train_samples_per_dataset: int = field(
        default=None,
        metadata={"help": "Max number of samples used per dataset for training."},
    )
    
    num_test_samples_per_dataset: int = field(
        default=None,
        metadata={"help": "Max number of samples used per dataset for eval/test."},
    )

    check_gpu_idle: bool = field(
        default=False,
        metadata={
            "help": "If set, wait for GPU to be idle before running. If not set, run scripts without checking GPU."
        },
    )
    wait_time: int = field(
        default=60*5,
        metadata={
            "help": "Time interval to check gpu idle."
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
# class GRPOTrainingArguments(trl.GRPOConfig):
class GRPOTrainingArguments(TrainingArguments):
    """
    Configuration class for the [`GRPOTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.
    
    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.
    """
    
    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )
    shuffle_dataset: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )


    # Parameters that control generation
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."},
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: bool = field(
        default=True,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a vLLM server is "
            "running. To run the server, install vLLM (`pip install vllm`) and run `trl vllm-serve`."
        },
    )
    vllm_server_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` "
            "and `vllm_server_port` are ignored."
        },
    )
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `server` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a "
            "TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

     # Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )

    # Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_gpu_memory_utilization` flag."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_tensor_parallel_size` flag."
        },
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    loss_type: str = field(
        default="bnpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are `grpo`, `bnpo`, and `dr_grpo`. "
            "`'grpo'`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to "
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
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )
    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), "
            "the rewards are normalized by the standard deviation, ensuring they have unit variance. If `False`, no "
            "scaling is applied. The Dr. GRPO paper recommends not scaling the rewards, as scaling by the standard "
            "deviation introduces a question-level difficulty bias."
        },
    )
    dynamic_sampling: bool = field(
        default=False, 
        metadata={"help": "Whether to use dyamic sampling, which may filter out low-quality samples."}
    )
    max_resample_attempts: int = field(
        default=10,
        metadata={"help": "Max number of attempts per prompt in model generation."},
    )
    compute_kl: bool = field(
        default=False, 
        metadata={"help": "Whether to compute kl even when beta=0. This helps to monitor model update."}
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )

    
    # Parameters that control the evaluation
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


    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )

    # Deprecated parameters
    vllm_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To use vLLM, start a vLLM "
            "server with the `trl vllm-serve` command."
        },
    )
    vllm_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control the data type for "
            "vLLM generation, you should now use the `dtype` parameter in the vLLM server configuration."
        },
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control the "
            "`max_model_len` for vLLM, you should now use the `max_model_len` parameter in the vLLM server "
            "configuration."
        },
    )
    vllm_enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control prefix caching in "
            "vLLM, you should now use the `enable_prefix_caching` parameter in the vLLM server configuration."
        },
    )

    
    
    # Parameters
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


    def __post_init__(self):
        super().__post_init__()

        # Check if the effective batch size can be divided by the number of generations
        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
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


