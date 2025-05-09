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


import os
import sys
import random
import logging
# from dataclasses import dataclass, field


import torch
import datasets
import transformers
# from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint


from accelerate.utils import is_peft_model, set_seed
from trl.models import create_reference_model
from trl import ModelConfig, TrlParser, get_peft_config
# from trl import GRPOTrainer

from grpo_trainer import GRPOTrainer
from arguments import GRPOTrainingArguments, GRPOScriptArguments, ModelArguments
from rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    tag_count_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
)

from utils import init_wandb_training, get_tokenizer, get_dataset


logger = logging.getLogger(__name__)


def main():

    # Arguments
    parser = TrlParser((GRPOScriptArguments, GRPOTrainingArguments, ModelArguments))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        # format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        format="[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model arguments:\n  {model_args.to_json_string()}")
    logger.info(f"Script parameters:\n  {script_args.to_json_string()}")
    logger.info(f"Training parameters:\n  {training_args.to_json_string()}")
    
    # Set wandb
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)
    
    ##################
    # Load the dataset
    ##################
    system_prompt = training_args.system_prompt
    train_dataset = get_dataset(
        script_args.train_dataset_name, split='train', system_prompt=system_prompt
    )
    eval_dataset = None
    if script_args.eval_dataset_name:
        eval_dataset = get_dataset(
            script_args.eval_dataset_name, split='test', system_prompt=system_prompt
        )
    
    # >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)
    
    # Use a small dataset for fast check
    # Make sure it's called after data preprocessing
    if script_args.max_num_train_samples is not None and script_args.max_num_train_samples > 0:
        num_samples = min(script_args.max_num_train_samples, len(train_dataset))
        sample_ids = random.sample(range(len(train_dataset)), num_samples)
        train_dataset = train_dataset.select(sample_ids)    
    
    ################
    # Load tokenizer
    ################
    # Set tokenizer.chat_template when necessary, see `get_tokenizer` for details.
    tokenizer = get_tokenizer(model_args, training_args)
    
    # -----------------------------------
    # Add special tokens when necessary?
    print("-"*100)
    special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        # If token is not included in vocabulary, add it
        if token_id == tokenizer.unk_token_id:
            tokenizer.add_tokens([token])
            logger.warning(f"'{token}' is not in the vocabulary, add it now.")
        else:
            logger.info(f"'{token}' is already in the vocabulary.")
    print("-"*100)
    # -----------------------------------
    
    ######################
    # Get reward functions
    ######################
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "tag": tag_count_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    
    #####################
    # Initializing model
    #####################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_init_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        # Disable caching if gradient checkpointing is enabled (not compatible)
    )
    training_args.model_init_kwargs = model_init_kwargs
    
    # Create model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_init_kwargs)
    
    # Reference model
    if (training_args.beta == 0.0 and not training_args.compute_kl) or is_peft_model(model):
        # If beta is 0.0, the reference model is not needed
        # If PEFT is used, the reference model is not needed since the adapter can be disabled
        # to revert to the initial model.
        ref_model = None
    else:
        # Create a reference model based on the initial model.
        ref_model = create_reference_model(model)
    
    # # >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)
    
    # # Resize the model's embedding layer
    # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # print(f"Updated model token embedding: {model.model.embed_tokens}")
    # TODO: 
    # However, vllm load model and tokenize from model config (model: str,),
    # https://github.com/vllm-project/vllm/blob/v0.7.2/vllm/entrypoints/llm.py#L157
    # so, resize token embeddings may induce mismatch on embedding size between the reseized one and vllm loaded one
    # https://github.com/vllm-project/vllm/issues/5203
    # A compromise proposal:
    if model.config.vocab_size is not None:
        assert len(tokenizer) <= model.config.vocab_size, "Mismatch: model vocab_size < tokenizer vocab_size"
    
    
    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),  
    )
    
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    
    # Check for last checkpoint when necessary
    checkpoint = None
    if not training_args.overwrite_output_dir:  # defaults to not overwrite
        if training_args.resume_from_checkpoint is not None:
            # TrainingArguments.resume_from_checkpoint (`str`, *optional*): 
            # The path to a folder with a valid checkpoint for your model.
            if os.path.isdir(training_args.resume_from_checkpoint):
                checkpoint = training_args.resume_from_checkpoint
            else:
                logger.warning(
                    f"'resume_from_checkpoint' is not detected at {training_args.resume_from_checkpoint}."
                    "Will train from scratch."
                )
        elif os.path.isdir(training_args.output_dir):
            # Continue training if output_dir points to a checkpoint directory
            checkpoint = get_last_checkpoint(training_args.output_dir)
            if checkpoint is not None:
                logger.info(f"Checkpoint detected, resuming training at {checkpoint=}.")
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    #############
    # Save model
    #############
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    
    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    

if __name__ == "__main__":
    main()
