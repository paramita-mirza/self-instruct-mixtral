# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import torch

from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, PreTrainedModel, PreTrainedTokenizer

from dataclasses import dataclass

from trl import (
    KTOConfig,
    KTOTrainer,
    ModelConfig,
    ScriptArguments,
    get_peft_config,
    setup_chat_format,
    get_kbit_device_map,
    get_quantization_config
)

def load_data(dataset_name, load_hf_data_from_disk=False):
    if load_hf_data_from_disk:
        return load_from_disk(dataset_name)
    if os.path.isfile(dataset_name):
        file_type = dataset_name.split(".")[-1].replace("jsonl", "json")
        return load_dataset(file_type, data_files=dataset_name)
    else:
        return load_dataset(dataset_name)

@dataclass
class CustomArgs:
    """
    Custom arguments for our personal use case.
    
    load_data_from_disk (`bool`, *optional*, defaults to `False`):
        Whether to load hf dataset from disk.

    """
    
    load_hf_data_from_disk: bool = False
    dataset_name_eval: str = None
    tokenizer_name_or_path: str = None


@dataclass
class ChatMlSpecialTokens:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|im_end|>"

    @property
    def system(self):
        return f"{self.bos_token}system"

    @property
    def user(self):
        return f"{self.bos_token}user"

    @property
    def assistant(self):
        return f"{self.bos_token}assistant"

    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )


FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens}

def setup_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    format: Optional[Literal["chatml"]] = "chatml",
    resize_to_multiple_of: Optional[int] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Setup chat format by adding special tokens to the tokenizer, setting the correct format, and extending the embedding layer of the model based on the new special tokens.

    If the model already has a chat template, this will throw an error. If you want to overwrite it, please set `tokenizer.chat_template` to `None`.

    Args:
        model (`~transformers.PreTrainedModel`): The model to be modified.
        tokenizer (`~transformers.PreTrainedTokenizer`): The tokenizer to be modified.
        format (`Optional[Literal["chatml"]]`): The format to be set. Defaults to "chatml".
        resize_to_multiple_of (`Optional[int]`): Number to resize the embedding layer to. Defaults to None.

    Returns:
        model (`~transformers.PreTrainedModel`): The modified model.
        tokenizer (`~transformers.PreTrainedTokenizer`): The modified tokenizer.
    """
    # check if model already had a chat template
    if tokenizer.chat_template is not None:
        raise ValueError(
            "Chat template is already added to the tokenizer. If you want to overwrite it, please set it to None"
        )

    # check if format available and retrieve
    if format not in FORMAT_MAPPING:
        raise ValueError(f"Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}")

    chat_format = FORMAT_MAPPING[format]()

    # set special tokens and them
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token

    added_tokens = tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                *tokenizer.special_tokens_map_extended["additional_special_tokens"],
                chat_format.bos_token,
                chat_format.eos_token
            ]
        }
    )

    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )

    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    # Update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig, CustomArgs))
    script_args, training_args, model_args, custom_args = parser.parse_args_into_dataclasses()
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    if custom_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            custom_args.tokenizer_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
    
    ref_model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=None)
    
    # Load the dataset
    dataset = load_data(script_args.dataset_name, load_hf_data_from_disk=custom_args.load_hf_data_from_disk)

    # Set train dataset
    train_data = dataset[script_args.dataset_train_split]

    # Set eval dataset
    if custom_args.dataset_name_eval:
        eval_data = load_data(custom_args.dataset_name_eval, load_hf_data_from_disk=custom_args.load_hf_data_from_disk)[script_args.dataset_test_split]
    else:
        eval_data = dataset[script_args.dataset_test_split] if script_args.dataset_test_split in dataset else None
    
    # Initialize the KTO trainer
    trainer = KTOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
