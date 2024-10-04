import logging
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import torch
import gc
from typing import List
from tqdm import tqdm
import json
from analyzer.utils import plot_histogram

logger = logging.getLogger(__name__)

class InsTagger(object):

    def __init__(self):
        self.model_name = "OFA-Sys/InsTagger"
        self.max_model_len = 2048
        self.sampling_params = SamplingParams(max_tokens=1000)
        self.tagger_template = (
            "You are a helpful assistant. Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {\"tag\": str, \"explanation\": str}.\nQuery: ##INSTRUCTION##\n\nASSISTANT:")

    def infer_tags(self, user_requests: List[str]):
        # Truncate input text
        tokenizer = self.llm.get_tokenizer()
        template_input_ids = tokenizer(self.tagger_template).input_ids
        num_token_tagger_template = len(template_input_ids)
        num_avail_token = (self.max_model_len - num_token_tagger_template)
        truncated_user_requests = []
        for user_request in user_requests:
            user_request_input_ids = tokenizer(user_request, truncation=True, max_length=num_avail_token).input_ids
            user_request = tokenizer.decode(user_request_input_ids, skip_special_tokens=True)
            truncated_user_requests.append(user_request)

        instructions = [self.tagger_template.replace("##INSTRUCTION##", inst) for inst in truncated_user_requests]

        outputs = self.llm.generate(instructions, self.sampling_params)
        output_tags = []
        for output in outputs:
            try:
                output_str = output.outputs[0].text
                if output_str.endswith(']'):
                    output_tags.append(json.loads(output_str))
                else:  # generation is not complete, not a proper list of {tag: ..., explanation: ...}
                    output_str = output.outputs[0].text[1:]
                    complete_tags = []
                    for tag in output_str.split(',')[:-1]:
                        complete_tags.append(json.loads(tag.strip()))
                    output_tags.append(complete_tags)
            except:
                output_tags.append([])
        return output_tags

    def run(self, instructions: List[str], responses: List[str], dataset_name: str, dataset_title: str, output_dir: str, request_batch_size: int=16):
        self.llm = LLM(model=self.model_name)  # Create an LLM.
        num_tags = []
        with open(f"{output_dir}/{dataset_name}.jsonl", "w") as fout:
            for i in tqdm(range(0, len(instructions), request_batch_size)):
                batch_instructions = instructions[i: i + request_batch_size]
                batch_tags = self.infer_tags(batch_instructions)
                for tags in batch_tags:
                    num_tags.append(len(tags))
                    fout.write(json.dumps(tags) + "\n")

        self.unload_llm()

        return num_tags

    def plot (self, scores: List[int], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str] = None):
        # Plot the histogram
        min_ylim = 1.0
        max_ylim = max(scores)
        plot_histogram(f"{output_dir}/{dataset_name}.png", dataset_title, scores, min_ylim, max_ylim)

    def unload_llm(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm.llm_engine.model_executor
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()

