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
from dataclasses import dataclass, field


import torch
import datasets
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint


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


# from utils import get_tokenizer
from utils import init_wandb_training


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
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    # -----------------------------------
    # Check if there is a column named `problem` and `solution` (may change it accordingly)
    split_name = 'train' if 'train' in dataset else 'test'  # use 'test' split if 'train' split is not available
    columns = dataset[split_name].column_names
    # if 'problem' in columns and 'solution' in columns:
    if 'problem' not in columns:
        for feture in columns:
            if feture.lower() in ['problem', 'question']:
                dataset = dataset.rename_column(feture, 'problem')
                break
        else:
            raise ValueError("no column named 'problem' in the datset!")
    if 'solution' not in columns:
        for feture in columns:
            if feture.lower() in ['solution', 'answer']:
                dataset = dataset.rename_column(feture, 'solution')
                break
        else:
            raise ValueError("no column named 'solution' in the datset!")
    
    # >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)
    
    # Format into conversation
    def make_conversation(example):
        prompt = []
        
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        
        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}
    
    dataset = dataset.map(make_conversation)
    
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    
    
    # Use a small set for fast check
    # Make sure it's called after data preprocessing
    train_dataset = dataset[split_name]
    if script_args.num_train_samples is not None:
        num_samples = min(script_args.num_train_samples, len(train_dataset))
        sample_ids = random.sample(range(len(train_dataset)), num_samples)
        train_dataset = train_dataset.select(sample_ids)
    
    # # >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)
    
    
    ################
    # Load tokenizer
    ################
    # tokenizer = get_tokenizer(model_args, training_args)
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    elif tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    
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
    
    # # >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)
    
    # # Resize the model's embedding layer
    # TODO: vllm read vocab_size from config.json? 
    # so, reseize token embeddings may induce mismatch on embedding size
    # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # print(f"Updated model token embedding: {model.model.embed_tokens}")
    # A compromise proposal:
    if model.config.vocab_size is not None:
        assert len(tokenizer) <= model.config.vocab_size, "Mismatch: model vocab_size < tokenizer vocab_size"
    
    
    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
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

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        # trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    
    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # #############
    # # push to hub
    # #############
    # if training_args.push_to_hub:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()
