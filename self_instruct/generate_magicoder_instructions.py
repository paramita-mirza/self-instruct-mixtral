import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
import os

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from llm.vllm_api import VLLM
from llm.huggingface_api import HuggingFaceLLM

import magicoder_utils


# DO NOT CHANGE THE FOLLOWING
SYSTEM = "You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions."
ERROR_MARGIN = 10


@dataclass(frozen=True)
class Args:
    seed_code_start_index: int
    # `seed_code_start_index` + `max_new_data` is the last-to-end seed code index
    max_new_data: int
    continue_from: str | None = field(default=None)

    # Keep the following arguments unchanged for reproducibility
    seed: int = field(default=976)

    temperature: float = field(default=1.0)
    model: str = field(default="mistralai/Mixtral-8x22B-Instruct-v0.1")
    model_max_tokens: int = field(default=8192)
    max_new_tokens: int = field(default=2500)

    min_lines: int = field(default=1)
    max_lines: int = field(default=15)
    chunk_size: int = field(default=1000)

    dataset_name: str = field(default="bigcode/starcoderdata")
    data_dir: str | None = field(default="python")
    max_considered_data: int | None = field(default=150000)

    num_vllm_devices: int = field(default=1)
    model_deployment: str = field(default='vllm')
    enforce_eager: bool = field(default=False)
    request_batch_size: int = field(default=1)
    cache_dir: str = field(default='/raid/s3/opengptx/models')
    to_self_instruct: bool = field(default=True)

    tag: str = field(
        default="",
        metadata={
            "help": "Custom tag as part of the output filename, not affecting the fingerprint"
        },
    )

    def fingerprint(self, prompt_template: str) -> str:
        # The combination of arguments can uniquely determine the generation process
        args = (
            self.seed,
            self.temperature,
            self.model,
            self.model_max_tokens,
            self.min_lines,
            self.max_lines,
            self.chunk_size,
            self.dataset_name,
            self.data_dir,
            self.max_considered_data,
            prompt_template,
            SYSTEM,
            ERROR_MARGIN,
        )
        return magicoder_utils.compute_fingerprint(*args, hash_length=5)


def map_dataset(examples: dict, indices: list[int], args: Args) -> dict:
    random.seed(args.seed + indices[0])
    seed_snippets = [
        extract_seed_code(args, content) for content in examples["content"]
    ]
    return {
        "seed": seed_snippets,
        "raw_index": indices,
    }

def extract_seed_code(args: Args, document: str) -> str:
    lines = document.splitlines(keepends=True)
    start_index = random.choice(range(len(lines)))
    n_lines_to_consider = random.randint(args.min_lines, args.max_lines)
    code = "".join(lines[start_index : start_index + n_lines_to_consider])
    return code


def parse_problem_solution(response_text: str) -> tuple[str, str] | None:
    response_text = response_text[0] if isinstance(response_text, list) else response_text
    lines = response_text.splitlines(keepends=True)
    problem_start_index: int | None = None
    solution_start_index: int | None = None
    for idx, line in enumerate(lines):
        if ("[problem description]" in line.lower() or "**problem description**" in line.lower()) and problem_start_index is None:
            problem_start_index = idx
        if ("[solution]" in line.lower() or "**solution**" in line.lower()) and solution_start_index is None:
            solution_start_index = idx
    if problem_start_index is None or solution_start_index is None:
        return None
    if problem_start_index >= solution_start_index:
        return None
    problem = "".join(lines[problem_start_index + 1 : solution_start_index]).strip()
    solution = "".join(lines[solution_start_index + 1 :]).strip()
    return problem, solution


def main():

    args, *_ = cast(
        tuple[Args, ...], HfArgumentParser(Args).parse_args_into_dataclasses()
    )

    openai = False

    split = (
        f"train[:{args.max_considered_data}]"
        if args.max_considered_data is not None
        else "train"
    )
    dataset: Dataset = load_dataset(
        args.dataset_name,
        data_dir=args.data_dir,
        split=split,
        num_proc=magicoder_utils.N_CORES,
    )
    random.seed(args.seed)
    # map_fn = get_map_dataset(args)
    dataset = dataset.map(
        function=map_dataset,
        fn_kwargs=dict(args=args),
        with_indices=True,
        batched=True,
        batch_size=args.chunk_size,
    )
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.map(lambda _, index: {"index": index}, with_indices=True)

    # Every run should produce the same data as long as the default params are not changed
    start_index = args.seed_code_start_index
    end_index = min(start_index + args.max_new_data, len(dataset))
    dataset = dataset.select(range(start_index, end_index))

    if args.model_deployment == 'openai':
        CLIENT = None
    elif args.model_deployment == 'hf':
        CLIENT = HuggingFaceLLM(model_name=args.model, cache_dir=args.cache_dir)
    elif args.model_deployment == 'vllm':
        CLIENT = VLLM(model_name=args.model, num_devices=args.num_vllm_devices, cache_dir=args.cache_dir, max_model_len=args.model_max_tokens, enforce_eager=args.enforce_eager)
    else:
        raise ValueError(f"Unknown API type: {args.model_deployment}")

    prompt_template = Path("data/prompt_magicoder.txt").read_text()
    timestamp = magicoder_utils.timestamp()
    data_fingerprint = args.fingerprint(prompt_template)
    if args.continue_from is not None:
        assert data_fingerprint in args.continue_from, "Fingerprint mismatch"
        assert f"{start_index}_{end_index}" in args.continue_from, "Index mismatch"
        old_path = Path(args.continue_from)
        assert old_path.exists()
        old_data = magicoder_utils.read_jsonl(old_path)
        assert len(old_data) > 0
        last_index = old_data[-1]["index"]
        n_skipped = last_index - start_index + 1
        print("Continuing from", old_path)
        f_out = old_path.open("a")
    else:
        tag = "" if args.tag == "" else f"-{args.tag}"
        os.makedirs(f"data/{args.model.replace('/', '__')}", exist_ok=True)
        path = Path(
            f"data/{args.model.replace('/', '__')}/data{tag}-{data_fingerprint}-{start_index}_{end_index}-{timestamp}.jsonl"
        )
        assert not path.exists()
        f_out = path.open("w")
        print("Saving to", path)
        n_skipped = 0

    batched_messages = []
    examples = []

    for index, example in enumerate(tqdm(dataset)):
        if index < n_skipped:
            continue
        assert index + start_index == example["index"]

        prompt = prompt_template.format(code=example["seed"])
        
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]

        batched_messages.append(messages)
        examples.append(example)

        if index % args.request_batch_size != 0:
            continue

        responses = CLIENT.magicoder_request(
            model=args.model,
            messages=batched_messages,
            max_tokens=args.max_new_tokens,
            n=1,
            temperature=args.temperature,
        )
        
        for example, response in zip(examples, responses):
            if openai:
                choice = response.choices[0]
                if choice.finish_reason != "stop":
                    continue

            response_content = choice.message.content if openai else response['response']
            parsing_result = parse_problem_solution(response_content)

            if parsing_result is None:
                print('Output has incorrect formatting, discard!')
                continue
            problem, solution = parsing_result
            if len(problem) == 0 or len(solution) == 0:
                continue
            fingerprint = response.system_fingerprint if openai else response['created_at'] + ' by ' + args.model.replace('/', '__')
            assert fingerprint is not None
            # In this dict seed means "seed code snippet" instead of "random seed"
            data = dict(
                raw_index=example["raw_index"],
                index=example["index"],
                seed=example["seed"],
                openai_fingerprint=fingerprint,
                problem=problem,
                solution=solution,
            )

            f_out.write(json.dumps(data) + "\n")

        batched_messages = []
        examples = []
    if args.to_self_instruct:
        magicoder_utils.to_self_instruct_format(args.model, sample_size=(end_index-start_index), file=path.name)

if __name__ == "__main__":
    main()
