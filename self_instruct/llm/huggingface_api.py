import json
import tqdm
import os
import re
import random
from datetime import datetime
import argparse
import logging
from typing import Union, Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
)


class HuggingFaceLLM():
    def __init__(
            self,
            model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
            cache_dir: str = "~/.cache/huggingface/hub/",
            device: str = "cuda",
            use_accelerate: bool = True,
            use_fast: bool = True,
    ):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info("Initiating HF Model Loading")

        self.__model_name = model_name
        self.__tokenizer_name = model_name

        self.__cache_dir = cache_dir
        self.__device = device
        self.__use_accelerate = use_accelerate

        self.__use_fast = use_fast

        self.__model_args = self.__get_model_args()
        self.__tokenizer_args = self.__get_tokenizer_args()

        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

        self.model.eval()
        self.model.config.use_cache = True

    def __get_model_args(self) -> Dict[str, str]:
        model_args = {
            'pretrained_model_name_or_path': self.__model_name,
            'torch_dtype': torch.float16,
            'attn_implementation': "flash_attention_2",
            #'load_in_8bit': False
        }

        # if "falcon" in self.__model_name:
        #    model_args["trust_remote_code"] = True

        if self.__use_accelerate:
            model_args["device_map"] = "auto"

        if self.__cache_dir : #and not os.environ["HF_HUB_CACHE"]:
            model_args['cache_dir'] = self.__cache_dir

        return model_args

    def __get_tokenizer_args(self) -> Dict[str, str]:
        tokenizer_args = {
            'pretrained_model_name_or_path': self.__tokenizer_name,
            'use_fast': self.__use_fast,
            # 'legacy': False,
        }

        if "falcon" in self.__model_name \
                or "vicuna" in self.__model_name \
                or "polylm" in self.__model_name \
                or "mistralai" in self.__model_name:
            tokenizer_args["padding_side"] = "left"

        if self.__cache_dir : #and not os.environ["HF_HUB_CACHE"]:
            tokenizer_args['cache_dir'] = self.__cache_dir

        return tokenizer_args

    def __load_model(self):
        try:
            model = AutoModelForCausalLM.from_pretrained(**self.__model_args)
        except:
            raise ValueError(
                f"The passed model type: AutoModelForCausalLM is not suitable for the model \"{self.__model_name}\"." \
                )

        if self.__device:
            if self.__use_accelerate:
                # os.environ["CUDA_VISIBLE_DEVICES"] = self.__device
                # print("Using CUDA devices:", os.environ["CUDA_VISIBLE_DEVICES"])
                pass
            else:
                self.__logger.info("loading model to", self.__device)
                model = model.to(self.__device)
                self.__logger.info("Using CUDA devices:", self.__device)

        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(**self.__tokenizer_args)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    @staticmethod
    def __get_model(self):
        return self.model

    def _postprocess_generated_ids(self, tokenized, generated_ids, stop_strings):
        new_ids = generated_ids
        if not self.model.config.is_encoder_decoder:
            # remove input ids from the genreated ids.
            # Note: This doesn't apply to AutoModelForSeq2SeqLM
            new_ids = generated_ids[:, tokenized.input_ids.shape[1]:]

        responses = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        responses = [response.replace("<pad>", "") for response in responses]  # LoRA output somehow contains <pad>
        exclusion = '|'.join([st + '$' for st in stop_strings]).replace('\n',
                                                                        '\s')  # removing stop string at the end of generation (if any)
        responses = [re.sub(exclusion, '', response) for response in responses]
        responses = [response.strip() for response in responses]

        return responses

    def run(self,
            prompts: Union[str, List[str]],
            max_new_tokens: int = 500,
            do_sample: bool = False,
            temperature: float = 0.7,
            top_p: float = 0.5,
            stop_strings: List[str] = None) -> List[str]:

        # if "mistralai" in self.__model_name and "Instruct" in self.__model_name:
        #     prompts = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False) for prompt in prompts]

        if self.__device:
            tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False).to(
                self.__device)
        else:
            tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False).to(0)

        generated_ids = self.model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            stop_strings=stop_strings,
            tokenizer=self.tokenizer,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        responses = self._postprocess_generated_ids(tokenized, generated_ids, stop_strings)
        return responses


    def make_requests(
            self, prompts, max_tokens, do_sample, temperature, top_p, stop_sequences
    ):
        response = self.run(prompts=prompts,
                           max_new_tokens=max_tokens,
                           do_sample=do_sample,
                           temperature=temperature,
                           top_p=top_p,
                           stop_strings=stop_sequences)

        if isinstance(prompts, list):
            results = []
            for j, prompt in enumerate(prompts):
                data = {
                    "prompt": prompt,
                    "response": response[j] if response else None,
                    "created_at": str(datetime.now()),
                }
                results.append(data)
            return results
        else:
            data = {
                "prompt": prompts,
                "response": response,
                "created_at": str(datetime.now()),
            }
            return [data]
        
    def magicoder_request(self, messages, max_tokens=10, temperature=1.0, model=None, n=1, seed=1):
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False) for m in messages]
        return self.make_requests(prompts=prompts, max_tokens=max_tokens, do_sample=True, temperature=temperature,top_p=1.0,stop_sequences=self.tokenizer.eos_token)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the prompts to Mixtral.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file to save the responses from Mixtral.",
    )
    parser.add_argument(
        "--hf_model_id",
        type=str,
        required=True,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="The HF model id.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        required=True,
        default="~/.cache/huggingface/hub/",
        help="The HF cache dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2000,
        help="The max_tokens parameter of Mixtral.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="The do_sample of Mixtral.",
    )
    parser.set_defaults(do_sample=True)
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temperature of Mixtral.",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="The `top_p` parameter of Mixtral.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=["\n\n"],
        nargs="+",
        help="The `stop_sequences` parameter of Mixtral.",
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists."
    )
    parser.add_argument(
        "--request_batch_size",
        default=20,
        type=int,
        help="The number of requests to send to Mixtral at a time."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    llm = HuggingFaceLLM(model_name=args.hf_model_id, cache_dir=args.hf_cache_dir)

    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            all_prompts = ["You are a helpful AI assistant. Follow the instruction and give a reasoning when necessary.\n### Instruction: " + json.loads(line)["instruction"] + "\n### Input: " + json.loads(line)["input"] + "\n### Output:" for line in fin]
        else:
            all_prompts = [line.strip().replace("\\n", "\n") for line in fin]


    for i in tqdm.tqdm(range(0, len(all_prompts), args.request_batch_size)):
        batch_prompts = all_prompts[i: i + args.request_batch_size]

        results = llm.make_requests(
            prompts=batch_prompts,
            max_tokens=args.max_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_sequences=args.stop_sequences,
        )
        with open(args.output_file, "a") as fout:
            for output in results:
                fout.write(json.dumps({"mixtral_output": output}) + "\n")
