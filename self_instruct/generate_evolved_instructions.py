import os
import json
import argparse
import logging
from tqdm import tqdm
import time
from utils import TqdmToLogger
from typing import List, Dict

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
        "--batch_dir",
        type=str,
        required=True,
        default="data/mixtral_8x22b_generations_test/",
        help="The directory where the batch is stored.",
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
        default=1,
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
        default="self_instruct_alpaca/data/prompt.txt",
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
    return parser.parse_args()

def generate_evol_instructions(llm, batch_instructions: List[Dict]):
    evol_template = (
        "You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version. "
        "Please follow the steps below to rewrite the given \"#Instruction#\" into a more complex version.\n\n"
        "# Step 1: Please read the \"#Instruction#\" carefully and list all the possible methods to make "
        "this instruction more complex (to make it a bit harder for well-known AI assistants such as "
        "ChatGPT and GPT4 to handle). Please do not provide methods to change the language of the instruction!\n\n"
        "Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 "
        "to make the #Instruction# more complex. The plan should include several methods from the #Methods List#.\n\n"
        "Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# "
        "can only add 10 to 20 words into the \"#Instruction#\".\n\n"
        "Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. "
        "Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#. "
        "Just provide the #Finally Rewritten Instruction# without any explanation.\n\n"
        "Please reply strictly in the following format:\n"
        "Step 1 #Methods List#:\n"
        "Step 2 #Plan#:\n"
        "Step 3 #Rewritten Instruction#:\n"
        "Step 4 #Finally Rewritten Instruction#:\n\n"
        "#Instruction#:\n"
        "{instruction}"
    )

    batch_prompts = [evol_template.format(instruction=inst['instruction']) for inst in batch_instructions]
    results = llm.make_requests(
        prompts=batch_prompts,
        max_tokens=args.max_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    evolved_instructions = []
    for i, result in enumerate(results):
        evolved = ""
        if "#Finally Rewritten Instruction#:" in result['response']:
            instruction = result['response'].split("#Finally Rewritten Instruction#:")[1].strip()
            if instruction:
                evolved = instruction
        evolved_instructions.append(evolved)

    return evolved_instructions

if __name__ == "__main__":
    start_time = time.time()
    logger = logging.getLogger(os.path.basename(__file__))
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    args = parse_args()

    # load the LM-generated instructions...
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                if instruction_info["input"] == "":
                    machine_instructions.append({
                        # "instruction" : instruction_info["instruction"] + '\n\n' + instruction_info["input"]
                        "instruction": instruction_info["instruction"]
                    })
                request_idx = instruction_info["request_idx"] + 1
        logger.info(f"Loaded {len(machine_instructions)} machine-generated instruction and response pairs.")
    machine_instructions = machine_instructions[args.start_idx:args.end_idx]

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
        llm = VLLM(model_name=args.hf_model_id, num_devices=args.num_devices)
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
        # now let's generate evolved instructions!
        with open(os.path.join(args.batch_dir, "evolved_instructions.jsonl"), "a") as fout:
            for i in tqdm(range(0, len(machine_instructions), args.request_batch_size)):
                batch_instructions = machine_instructions[i: i + args.request_batch_size]
                evolved_instructions = generate_evol_instructions(llm, batch_instructions)

                for inst in evolved_instructions:
                    fout.write(json.dumps({
                        "instruction": inst,
                        "input": "",
                        "evol_llm": args.hf_model_id
                    }) + "\n")

        if args.model_deployment == "fastapi":
            from llm.llm_requests import LLMRequests as llm
            llm.termination_request()

        logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))
    
