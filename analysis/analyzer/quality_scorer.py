import logging
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import torch
import gc
from typing import List
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from analyzer.utils import plot_histogram, plot_histogram_per_category

logger = logging.getLogger(__name__)

class QualityScorer(object):

    def __init__(self):
        self.model_name = "hkust-nlp/deita-quality-scorer"
        self.max_model_len = 2048
        self.sampling_params = SamplingParams(max_tokens=2, logprobs=20)
        self.quality_template = (
            "You are a helpful assistant. Please identify the quality score of the Response corresponding to the Question. \n #Question#:\n{instruction}\n#Response#:\n{output} \n##Quality: ")


    def infer_quality(self, user_requests: List[str], system_responses: List[str]):
        # Truncate input text
        tokenizer = self.llm.get_tokenizer()
        template_input_ids = tokenizer(self.quality_template).input_ids
        num_token_complexity_template = len(template_input_ids)
        num_avail_token = (self.max_model_len - num_token_complexity_template)
        truncated_system_responses = []
        for i, system_response in enumerate(system_responses):
            user_request_input_ids = tokenizer(user_requests[i]).input_ids
            num_token_user_request = len(user_request_input_ids)
            num_avail_token_resp = num_avail_token - num_token_user_request
            if num_avail_token_resp > 0:
                response_input_ids = tokenizer(system_response, truncation=True, max_length=num_avail_token_resp).input_ids
                system_response = tokenizer.decode(response_input_ids, skip_special_tokens=True)
            truncated_system_responses.append(system_response)

        prompts = [self.quality_template.format(instruction=user_requests[i], output=resp) for i, resp in enumerate(truncated_system_responses)]

        scores = []
        outputs = self.llm.generate(prompts, self.sampling_params)
        for output in outputs:
            try:
                logprobs_list = output.outputs[0].logprobs[0]
                score_logits = []
                id2score = {
                    29896: "1",
                    29906: "2",
                    29941: "3",
                    29946: "4",
                    29945: "5",
                    29953: "6"
                }
                score_template = np.array([1, 2, 3, 4, 5, 6])
                for k in id2score:
                    if k in logprobs_list:
                        score_logits.append(logprobs_list[k].logprob)
                    else:
                        score_logits.append(-20.0)
                score_logits = np.array(score_logits)
                score_npy = softmax(score_logits, axis=0)
                score_npy = score_npy * score_template

                score_npy = np.sum(score_npy, axis=0)
                scores.append(score_npy)
            except:
                scores.append(3.0)

        return scores

    def run(self, instructions: List[str], responses: List[str], dataset_name: str, dataset_title: str, output_dir: str, request_batch_size: int=16):
        self.llm = LLM(model=self.model_name)  # Create an LLM.
        quality_scores = []
        for i in tqdm(range(0, len(instructions), request_batch_size)):
            batch_instructions = instructions[i: i + request_batch_size]
            batch_responses = responses[i: i + request_batch_size]
            scores = self.infer_quality(batch_instructions, batch_responses)
            quality_scores += scores

        # Write scores to file
        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            for score in quality_scores:
                fout.write(f"{score}\n")

        self.unload_llm()

        return quality_scores

    def plot(self, scores: List[float], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # Plot the histogram
        min_ylim = 1.0
        max_ylim = 6.0
        plot_histogram(f"{output_dir}/{dataset_name}.png", dataset_title + " (quality)", scores,
                       min_ylim, max_ylim)
        if categories:
            plot_histogram_per_category(f"{output_dir}/{dataset_name}_category.png",
                                        dataset_title + " (quality per category)",
                                        scores, categories)

    def unload_llm(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm.llm_engine.model_executor
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()

