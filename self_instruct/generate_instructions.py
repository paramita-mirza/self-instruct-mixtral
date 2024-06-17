import os
import json
import random
import re
import string
import argparse
import numpy as np
from chromadb_utils import create_collection, populate_collection, query_collection
# from llm.llm_requests import LLMRequests as llm
from llm.huggingface_api import HuggingFaceLLM
import logging
from tqdm import tqdm
import time
from utils import TqdmToLogger

logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler()
                    ]
                    )
# random.seed(42)

def encode_prompt_alpaca(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(args.prompt_instructions).read() + "\n"
    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}<[EOI]>\n"
    prompt += f"###\n"
    prompt += f"{len(prompt_instructions)+1}. Instruction:"
    return prompt

def post_process_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction: " + response
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if "<[EOI]>" not in inst:
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip().replace("<[EOI]>", "")
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def filter_instructions(instructions):
    # this function is to filter out same instruction and input pairs in a single batch
    instruction_input_dict = {}
    for instruction in instructions:
        key = f"{instruction['instruction']} ### {instruction['input']}"
        if key not in instruction_input_dict:
            instruction_input_dict[key] = []
        instruction_input_dict[key].append(instruction)

    filtered_instructions = [random.choice(v) for k, v in instruction_input_dict.items()]
    return filtered_instructions 

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
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=300,
        help="Number of instructions to generate.",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="The HF model id.",
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=3,
        help="The number of instructions to use in the prompt."
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
        "--prompt_instructions",
        default="data/prompt.txt",
        type=str,
        help="Path of the prompt instructions file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    logger = logging.getLogger(os.path.basename(__file__))
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    args = parse_args()

    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if "dolly" in args.seed_tasks_path:
        seed_instructions = [
            {"instruction": t["instruction"], "input": t["instances"][0]["input"],
             "output": t["instances"][0]["output"]}
            for t in seed_tasks if t["category"] == "closed_qa"
        ]
    else:
        seed_instructions = [
            {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
            for t in seed_tasks
        ]
    logger.info(f"Loaded {len(seed_instructions)} human-written seed instruction and instance pairs")
    
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append({
                    "instruction" : instruction_info["instruction"],
                    "input" : instruction_info["input"],
                    "output" : instruction_info["output"],
                })
                request_idx = instruction_info["request_idx"] + 1
        logger.info(f"Loaded {len(machine_instructions)} machine-generated instruction and instance pairs ")

    documents = [d["instruction"] if len(d["input"]) > 100 else d["instruction"]+'\n'+d["input"] for d in seed_instructions] + \
                [d["instruction"] if len(d["input"]) > 100 else d["instruction"]+'\n'+d["input"] for d in machine_instructions]
    chromadb_collection = create_collection("instructions_embeddings")
    populate_collection(documents = documents, batch_size = 100, collection = chromadb_collection)

    
    # # wait until the API server is up...
    # patience = 200
    # healthy = llm.check_api()
    # while (not healthy and patience > 0):
    #     time.sleep(10)
    #     patience -= 10
    #     healthy = llm.check_api()

    llm = HuggingFaceLLM(model_name=args.hf_model_id)
    healthy = True
    
    if healthy:
        # now let's generate new instructions and instances!
        progress_bar = tqdm(total=args.num_instructions_to_generate, ascii=True, file=tqdm_out)
        if machine_instructions:
            progress_bar.update(len(machine_instructions))

        while len(machine_instructions) < args.num_instructions_to_generate:
            with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
                batch_inputs = []
                for _ in range(args.request_batch_size):
                    # # only sampling from the seed tasks
                    prompt_instructions = random.sample(seed_instructions, args.num_prompt_instructions)

                    # bootstrapping, 1/3 seed, 2/3 machine generated tasks
                    # prompt_instructions = random.sample(seed_instructions, args.num_prompt_instructions // 3)
                    # prompt_instructions += random.sample(machine_instructions, args.num_prompt_instructions // 3 * 2)

                    prompt = encode_prompt_alpaca(prompt_instructions)
                    batch_inputs.append(prompt)

                # print(batch_inputs)

                request_start = time.time()
                results = llm.make_requests(
                    prompts=batch_inputs,
                    max_tokens=3072,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop_sequences=["\n20", "20.", "20."]
                )
                request_duration = time.time() - request_start

                process_start = time.time()

                instructions = []
                all_metadata = []

                # print(results)
                
                for result in results:
                    new_instructions = post_process_response(args.num_prompt_instructions, result['response'])
                    instructions += new_instructions

                if len(instructions) > 0:
                    instructions = filter_instructions(instructions)
                
                selected_instructions = []
                total = len(instructions)
                keep = 0
                for inst in instructions:
                    most_similar_instructions, similarities = query_collection(
                        collection = chromadb_collection,
                        query_text = inst["instruction"],
                        limit = 10
                    )
                    if max(similarities) > 0.7:
                        continue
                    else:
                        keep += 1
                    
                    machine_instructions.append(inst)
                    if len(inst["input"]) > 100:
                        selected_instructions.append(inst["instruction"])
                    else:
                        selected_instructions.append(inst["instruction"]+'\n'+inst["input"])
                    fout.write(json.dumps({
                        "instruction": inst['instruction'],
                        "input": inst['input'],
                        "output": inst['output'],
                        "most_similar": most_similar_instructions,
                        "avg_similarity_score": float(np.mean(similarities)),
                        "request_idx": request_idx
                    }) + "\n")
                    progress_bar.update(1)

                populate_collection(documents = selected_instructions, batch_size = len(selected_instructions),
                                    collection = chromadb_collection)
                
                request_idx += 1
                process_duration = time.time() - process_start
                print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
                print(f"--- Generated {total} instructions, kept {keep} instructions")

        # llm.termination_request()
        logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))
    
