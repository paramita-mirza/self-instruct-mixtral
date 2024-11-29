import os
import json
import argparse
import logging
from tqdm import tqdm
import time
from utils import TqdmToLogger
from typing import List, Dict
import sys
from datasets import Dataset, load_dataset
import random

logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler()
                    ]
                    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output file that contains instructions.",
    )
    parser.add_argument(
        "--model_deployment",
        type=str,
        default="vllm",
        help="How to load the HF model, e.g., hf, fastapi, vllm, nemo",
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
        "--max_tokens",
        type=int,
        default=2048,
        help="The max tokens parameter for the generation."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=20,
        help="The number of requests to send to Mixtral at a time."
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
        "--prompt",
        default="self_instruct_alpaca/data/prompt_everyday_conversation.txt",
        type=str,
        help="Path of the prompt instructions file",
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=1,
        help="The number of GPUs to use for inference."
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
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Enforce eager mode for vllm"
    )
    return parser.parse_args()

def generate_everyday_conversations(llm, batch_topics: List[str], everyday_conversations_template):

    greeting_examples = ["Hi!", "Hello", "Heya", "Good morning"]
    batch_prompts = [everyday_conversations_template.format(topics=topics,
                                                            greeting=random.choice(greeting_examples))
                     for topics in batch_topics]

    results = llm.make_requests(
        prompts=batch_prompts,
        max_tokens=args.max_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    outputs = []
    for i, result in enumerate(results):
        response = result['response']
        outputs.append(response)

    return outputs

if __name__ == "__main__":
    start_time = time.time()
    logger = logging.getLogger(os.path.basename(__file__))
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    args = parse_args()

    template = open(args.prompt_instructions, encoding='utf-8').read()

    # Load the model...
    if args.model_deployment == "fastapi":      # via FastAPI
        from llm.llm_requests import LLMRequests as llm

        # wait until the API server is up...
        patience = 200
        healthy = llm.check_api()
        while (not healthy and patience > 0):
            time.sleep(10)
            patience -= 10
            healthy = llm.check_api()
    elif args.model_deployment == "vllm":       # via VLLM
        from llm.vllm_api import VLLM
        llm = VLLM(model_name=args.hf_model_id, num_devices=args.num_devices, cache_dir=args.hf_cache_dir, enforce_eager=args.enforce_eager)
        healthy = True
    elif args.model_deployment == "hf":         # via HuggingFace
        from llm.huggingface_api import HuggingFaceLLM
        llm = HuggingFaceLLM(model_name=args.hf_model_id, cache_dir=args.hf_cache_dir)
        healthy = True
    elif args.model_deployment == "nemo":       # via Nemo framework
        # TODO: check how to start the server for requests
        from llm.nemo_api import LLMRequests_Nemo
        llm = LLMRequests_Nemo()
        healthy = True

    if healthy:
        dataset = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k")
        train_ds = dataset["train_sft"]
        topics = [example["full_topic"] for example in train_ds]
        topics = topics[args.start_idx:args.end_idx]

        # now let's generate everyday conversations!
        with open(args.output_file, "a") as fout:
            instances = []
            for i in tqdm(range(0, len(topics), args.request_batch_size)):
                batch_topics = topics[i: i + args.request_batch_size]
                outputs = generate_everyday_conversations(llm, batch_topics, template)

                for j, output in enumerate(outputs):
                    message = []
                    turns = output.split("User:")
                    user_request = ""
                    for turn in turns:
                        if "AI:" in turn:
                            parts = turn.split("AI:")
                            if len(parts) > 1:
                                user_request += parts[0].strip()
                                ai_response = parts[1].strip()
                            if len(parts) > 2:
                                ai_response += parts[2].strip()

                            message.append({"content": user_request, "role": "user"})
                            message.append({"content": ai_response, "role": "assistant"})
                            user_request = ""
                        else:
                            user_request = turn.strip()

                    instances.append({"message": message, "topics": batch_topics[j].split('/')})

            for inst in instances:
                fout.write(json.dumps(inst) + "\n")

        if args.model_deployment == "fastapi":
            from llm.llm_requests import LLMRequests as llm
            llm.termination_request()

        logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))
    
