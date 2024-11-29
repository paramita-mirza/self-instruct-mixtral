import os
import json
import random
import re
import string
import argparse
from chromadb_utils import create_collection, populate_collection, add_to_collection
import logging
from tqdm import tqdm
import time
from utils import TqdmToLogger
import torch

from utils import language_map, embedding_models

logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler()
                    ]
                    )

def encode_prompt_alpaca(seed_instructions, translation=None):
    translation = language_map["EN"] if translation is None else translation
    prompt = open(args.prompt_instructions, encoding='utf-8').read()
    seed_instructions_str = ""
    for idx, task_dict in enumerate(seed_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        seed_instructions_str += f"#####\n"
        seed_instructions_str += f"{idx + 1}. {translation['Instruction']}: {instruction}\n"
        seed_instructions_str += f"{idx + 1}. {translation['Input']}:\n{input}\n"
        seed_instructions_str += f"{idx + 1}. {translation['Output']}:\n{output}##EOI##\n"
    return prompt.format(seed_instructions=seed_instructions_str)

def post_process_response(num_prompt_instructions, response, translation=None):
    translation = language_map["EN"] if translation is None else translation

    if response is None:
        return []
    raw_instructions = f"1. {translation['Instruction']}: " + response
    raw_instructions = re.split("#####", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        if inst.strip() == "":
            continue
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if "##EOI##" not in inst:
            continue
        inst = inst.strip()

        idx += 1
        #splitted_data = re.split(f"{idx}\.\s+({translation['Instruction']}|{translation['Input']}|{translation['Output']}):", inst)
        splitted_data = re.split(f"(?:{idx}\\.)?\\s+({translation['Instruction']}|{translation['Input']}|{translation['Output']}):", inst)


        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip().replace("##EOI##", "")

        # filter out too short instructions
        if len(inst.split()) <= 3: # or len(inst.split()) > 150:
            continue

        # filter out too long output
        if len(output.split()) > 200:
            continue

        # filter based on keywords that are not suitable for language models.
        blacklist = translation.get("blacklist", [])
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
    # this function is to filter out the same instruction and input pairs in a single batch
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
        help="The directory where the batch is stored. If --lang is not EN, the file is automatically renamed to <path>/<foldername>_<lang>/",
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
        "--chromadb_embedding",
        type=str,
        default="recommended",
        help="The embedding model for ChromaDB to compute instruction similarity. If 'recommended', it will use the recommended model for the specified language.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="The max tokens parameter for the generation."
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
        "--prompt_instructions",
        default="self_instruct_alpaca/data/prompt.txt",
        type=str,
        help="Path of the prompt instructions file. If --lang is not EN, the file is automatically renamed to <path>/<filename>_<lang>.txt",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Whether to use generated instructions as seed tasks (bootstrapping)",
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=1,
        help="The number of GPUs to use for inference."
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="EN",
        help="The language of the instructions."
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Whether to use `enforce_eager` option when using the vllm api."
    )
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    logger = logging.getLogger(os.path.basename(__file__))
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    args = parse_args()

    # set the language specific terms
    translation = language_map[args.lang]
    if args.lang != "EN":
        args.prompt_instructions = args.prompt_instructions.replace(".txt", f"_{args.lang}.txt")
        args.batch_dir = args.batch_dir.rstrip('/') + f"_{args.lang}/"
    chromadb_embedding = embedding_models[args.lang] if args.chromadb_embedding == "recommended" else args.chromadb_embedding

    # load the human-written seed instructions...
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    seed_instructions = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    random.shuffle(seed_instructions)
    logger.info(f"Loaded {len(seed_instructions)} human-written seed instruction and instance pairs.")
    
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0

    # load the LM-generated instructions...
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
        random.shuffle(machine_instructions)
        logger.info(f"Loaded {len(machine_instructions)} machine-generated instruction and response pairs.")

    documents = [d["instruction"]+'\n'+d["input"] for d in seed_instructions] + \
                [d["instruction"]+'\n'+d["input"] for d in machine_instructions]

    # store seed instructions and machine-generated ones in ChromaDB...
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chromadb_collection = create_collection(collection_name="instruction_embeddings", embedding_model=chromadb_embedding, device=device)
    populate_collection(collection=chromadb_collection, documents=documents, batch_size=100)

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
        # now let's generate new instructions and instances!
        progress_bar = tqdm(total=args.num_instructions_to_generate, ascii=True, file=tqdm_out)
        if machine_instructions:
            progress_bar.update(len(machine_instructions))
        num_generated_instructions = len(machine_instructions)

        while num_generated_instructions < args.num_instructions_to_generate:
            with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
                batch_inputs = []
                for _ in range(args.request_batch_size):

                    if args.bootstrap:
                        # bootstrapping, 1/3 seed, 2/3 machine generated tasks
                        prompt_instructions = random.sample(seed_instructions, args.num_prompt_instructions // 3)
                        prompt_instructions += random.sample(machine_instructions, args.num_prompt_instructions // 3 * 2)
                    else:
                        # only sampling from the seed tasks
                        prompt_instructions = random.sample(seed_instructions, args.num_prompt_instructions)

                    prompt = encode_prompt_alpaca(prompt_instructions, translation=translation)
                    batch_inputs.append(prompt)

                # print(batch_inputs)

                request_start = time.time()
                results = llm.make_requests(
                    prompts=batch_inputs,
                    max_tokens=args.max_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop_sequences=["\n###\n11."]
                )
                request_duration = time.time() - request_start

                process_start = time.time()

                generated_instructions = []
                all_metadata = []

                # print(results)

                for result in results:
                    instructions = post_process_response(args.num_prompt_instructions, result['response'], translation=translation)
                    generated_instructions += instructions

                if len(generated_instructions) > 0:
                    generated_instructions = filter_instructions(generated_instructions)

                # print(generated_instructions)
                
                selected_instructions = []
                total_generated = len(generated_instructions)
                keep = 0

                # query the ChromaDB for similar instructions
                results = chromadb_collection.query(
                    query_texts=[inst["instruction"]+'\n'+inst["input"] for inst in generated_instructions],  # Chroma will embed this for you
                    n_results=1  # how many results to return
                )
                for i, inst in enumerate(generated_instructions):
                    distance = results['distances'][i][0]
                    similar_instruction = results['documents'][i][0]
                    if distance > 0.3:  # not too similar with existing instructions
                        keep += 1
                        selected_instructions.append(inst["instruction"]+'\n'+inst["input"])
                        fout.write(json.dumps({
                            "instruction": inst['instruction'],
                            "input": inst['input'],
                            "output": inst['output'],
                            "generator": args.hf_model_id,
                            "most_similar": similar_instruction,
                            "cosine_distance": distance,
                            "request_idx": request_idx
                        }) + "\n")
                        progress_bar.update(1)

                if len(selected_instructions) > 0:
                    add_to_collection(
                        collection=chromadb_collection,
                        documents=selected_instructions
                    )
                    num_generated_instructions += len(selected_instructions)
                
                request_idx += 1
                process_duration = time.time() - process_start
                print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
                print(f"--- Generated {total_generated} instructions, kept {keep} instructions")

        if args.model_deployment == "fastapi":
            from llm.llm_requests import LLMRequests as llm
            llm.termination_request()

        logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))
    
