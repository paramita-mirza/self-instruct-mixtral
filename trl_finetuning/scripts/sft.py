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
import pandas as pd
import re
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass
from accelerate import PartialState

from trl import (
    SFTConfig,
    SFTTrainer,
    ModelConfig,
    ScriptArguments,
    DataCollatorForCompletionOnlyLM,
    get_peft_config,
    get_kbit_device_map,
    get_quantization_config
)


def load_data(dataset_name, split=None, load_hf_data_from_disk=False):
    if load_hf_data_from_disk:
        return load_from_disk(dataset_name)
    if os.path.isfile(dataset_name):
        df = pd.read_json(dataset_name, lines=True)
        # dataset should be in one of the supported formats
        # https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
        if 'messages' in df.columns:
            df = df[['messages']]
        elif 'prompt' in df.columns and 'completion' in df.columns:
            df = df[['prompt', 'completion']]
        else:
            raise ValueError('Unsupported data format, see https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support')
        if split:
            dataset = Dataset.from_pandas(df).train_test_split(test_size=0.03, seed=42)
            return dataset[split]
        else:
            return Dataset.from_pandas(df)
    else:
        return load_dataset(dataset_name, split=split)

def apply_spectrum(model, spectrum_parameters):
    with open(spectrum_parameters,
            "r") as fin:
        yaml_parameters = fin.read()

    unfrozen_parameters = []
    for line in yaml_parameters.splitlines():
        if line.startswith("- "):
            unfrozen_parameters.append(line.split("- ")[1])

    def freeze_and_unfreeze_parameters(model, unfrozen_parameters):
        # freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze Spectrum parameters
        for name, param in model.named_parameters():
            if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
                param.requires_grad = True

    freeze_and_unfreeze_parameters(model, unfrozen_parameters)

    return model


@dataclass
class CustomArgs:
    """
    Custom arguments for our personal use case.

    load_data_from_disk (`bool`, *optional*, defaults to `False`):
        Whether to load hf dataset from disk.

    """

    load_hf_data_from_disk: bool = False
    dataset_name_eval: str = None
    spectrum_parameters: str = None
    tokenizer_name_or_path: str = None
    completion_only: bool = False

@dataclass
class ChatMlSpecialTokens:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<|im_start|>"
    # eos_token: str = "<|im_end|>"
    # pad_token: str = "<|im_end|>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"

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
            f"{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )

class LLamaSpecialTokens:
    """Dataclass for special tokens used in LLama, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"

    @property
    def chat_template(self):
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}"
            "{% set loop_messages = messages %}{% set system_message = false %}{% endif %}"
            "{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}"
            "{% else %}{% set content = message['content'] %}{% endif %}"
            "{% if message['role'] == 'user' %}"
            f"{{{{'{self.bos_token}' + '[INST] ' + content.strip() + ' [/INST]'}}}}"
            "{% elif message['role'] == 'assistant' %}"
            f"{{{{' ' + content.strip() + ' ' + '{self.eos_token}'}}}}"
            "{% endif %}{% endfor %}"
        )


FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens,
                  "llama": LLamaSpecialTokens}

def setup_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    format: Optional[Literal["chatml", "llama"]] = "chatml",
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
    # tokenizer.bos_token = chat_format.bos_token   # ChatML bos_token will not be regarded as a token

    tokens_to_add = [special_token for special_token in [chat_format.eos_token, chat_format.pad_token]
                     if special_token not in tokenizer.special_tokens_map_extended["additional_special_tokens"]]

    if "additional_special_tokens" in tokenizer.special_tokens_map_extended:
        added_tokens = tokenizer.add_special_tokens(
            {
                "additional_special_tokens": tokenizer.special_tokens_map_extended["additional_special_tokens"] +
                                             tokens_to_add
            }
        )
    else:
        added_tokens = tokenizer.add_special_tokens(
            {
                "additional_special_tokens": tokens_to_add
            }
        )

    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )

    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        # model.config.bos_token_id = tokenizer.bos_token_id    # ChatML bos_token will not be regarded as a token
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    # Update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        # model.generation_config.bos_token_id = tokenizer.bos_token_id     # ChatML bos_token will not be regarded as a token
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig, ModelConfig, CustomArgs))
    script_args, training_args, model_args, custom_args = parser.parse_args_into_dataclasses()
    print(training_args)
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    device_string = PartialState().process_index
    model_kwargs = dict(
        revision=model_args.model_revision,
        # attn_implementation="flash_attention_2",
        # torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False, # if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # print(model_args, model_kwargs, training_args, custom_args)

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    if custom_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            custom_args.tokenizer_name_or_path, trust_remote_code=model_args.trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code,
        )

    # Remove original chat template
    tokenizer.chat_template = None
    tokenizer.default_chat_template = None

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer, format="llama")

    # Sanity check of the tokenizer
    messages = [{"role": "user", "content": "Heya! How's it going?"}, {"role": "assistant", "content": "Good! You?"},
                {"role": "user", "content": "Fine, thanks.\n"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print(repr(text))
    tokenized_text = tokenizer(text).input_ids
    print(tokenizer.convert_ids_to_tokens(tokenized_text))

    # Load the training dataset
    train_data = load_data(script_args.dataset_name, split=script_args.dataset_train_split,
                           load_hf_data_from_disk=custom_args.load_hf_data_from_disk)

    # Load the eval dataset
    if custom_args.dataset_name_eval:
        eval_data = load_data(custom_args.dataset_name_eval, split=script_args.dataset_test_split,
                              load_hf_data_from_disk=custom_args.load_hf_data_from_disk)
    else:
        eval_data = load_data(script_args.dataset_name, split=script_args.dataset_test_split,
                           load_hf_data_from_disk=custom_args.load_hf_data_from_disk)

    train_data = train_data.map(lambda x: {
        "text": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)}).remove_columns("messages").shuffle(seed=42)
    eval_data = eval_data.map(lambda x: {
        "text": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)}).remove_columns("messages")

    # During training, pad_token is set to <pad> instead of eos token to prevent endless generation.
    # This change must be reverted after training.
    # if tokenizer.pad_token is None:
    #    tokenizer.pad_token = '<pad>'
    #    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    tokenizer.padding_side = 'right'

    # Apply Spectrum to freeze/unfreeze model parameters
    if custom_args.spectrum_parameters:
        model = apply_spectrum(model, custom_args.spectrum_parameters)

    if custom_args.completion_only:
        training_args.packing = False
        training_args.eval_packing = False

        # Set up Data Collator
        messages = [{"role": "user", "content": "Hi"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_text = tokenizer(text).input_ids
        print(tokenizer.convert_ids_to_tokens(tokenized_text))
        instruction_template_ids = tokenized_text[1:4]
        print(tokenizer.convert_ids_to_tokens(instruction_template_ids))
        response_template_ids = tokenized_text[5:]
        print(tokenizer.convert_ids_to_tokens(response_template_ids))
        collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template_ids,
                                                   response_template=response_template_ids,
                                                   tokenizer=tokenizer, mlm=False, padding_free=True)

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=collator,
            peft_config=get_peft_config(model_args),
        )
    else:
        # Initialize the SFT trainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=get_peft_config(model_args),
        )

    # Train and push the model to the Hub

    # trainer.evaluate()
    trainer.train()

    # During training, pad_token is set to unk instead of eos token to prevent endless generation.
    # This change must be reverted after training.
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    tokenizer.padding_side = 'left'

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

