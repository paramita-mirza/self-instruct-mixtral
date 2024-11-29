import os
import logging
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json
from datetime import datetime
import argparse

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
)


class VLLM():
    def __init__(
            self,
            model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
            num_devices: int = 1,
            cache_dir: str = '/raid/s3/opengptx/models',
            max_model_len: int = None,
            **kwargs
    ):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info("Initiating HF Model Loading (via vllm)")

        self.model_name = model_name
        if num_devices > 1:
            self.llm = LLM(model=self.model_name, tensor_parallel_size=num_devices, download_dir=cache_dir, trust_remote_code=True, max_model_len=max_model_len, **kwargs
                           )  # Create an LLM, multiple devices.
        else:
            self.llm = LLM(model=self.model_name, download_dir=cache_dir, trust_remote_code=True, max_model_len=max_model_len)  # Create an LLM.
        self.tokenizer = self.llm.get_tokenizer()

    @staticmethod
    def __get_model(self):
        return self.model

    def make_requests(
            self, prompts, max_tokens, do_sample, temperature, top_p, stop_sequences = []
    ):
        if not isinstance(prompts, list): prompts = [prompts]

        if "Llama-3" in self.model_name:
            # stop_sequences += [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            stop_sequences += ['<|eot_id|>'] #[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        if stop_sequences:
            self.sampling_params = SamplingParams(max_tokens=max_tokens,
                                                  temperature=temperature,
                                                  top_p=top_p,
                                                  stop=stop_sequences,
                                                  )
        else:
            self.sampling_params = SamplingParams(max_tokens=max_tokens,
                                                  temperature=temperature,
                                                  top_p=top_p,
                                                  )

        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for i, output in enumerate(outputs):
            try:
                output_str = output.outputs[0].text.strip()
                data = {
                    "response": output_str,
                    "created_at": str(datetime.now()),
                }
                results.append(data)
            except:
                data = {
                    "response": None,
                    "created_at": str(datetime.now()),
                }
                results.append(data)

        return results
    
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
        default=1.0,
        type=float,
        help="The temperature of Mixtral.",
    )
    parser.add_argument(
        "--top_p",
        default=1.0,
        type=float,
        help="The `top_p` parameter of Mixtral.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=None,
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
    parser.add_argument(
        "--num_devices",
        default=1,
        type=int,
        help="The number of GPU devices."
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="The starting index."
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="The ending index."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    llm = VLLM(model_name=args.hf_model_id, num_devices=args.num_devices)

    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            if "Llama-3" in args.hf_model_id:
                all_prompts = ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant. " \
                               "Follow the instruction and give a reasoning or an explanation when necessary.<|eot_id|>" \
                               "<|start_header_id|>user<|end_header_id|>### Instruction: " + json.loads(line)["instruction"] +
                               "\n### Input: " + json.loads(line)["input"] + "<|eot_id|>" \
                               "<|start_header_id|>assistant<|end_header_id|>" for line in fin]
            else:
                all_prompts = ["You are a helpful AI assistant. Follow the instruction and give a reasoning or an explanation when necessary.\n" \
                               "### Instruction: " + json.loads(line)["instruction"] + "\n### Input: " + json.loads(line)["input"]
                               + "\n\n### Assistant:" for line in fin]
        else:
            all_prompts = [line.strip().replace("\\n", "\n") for line in fin]
    all_prompts = all_prompts[args.start_idx:args.end_idx]

    for i in tqdm(range(0, len(all_prompts), args.request_batch_size)):
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
                resp = output["response"]
                fout.write(json.dumps({"output": resp, "output_generator": args.hf_model_id}) + "\n")
