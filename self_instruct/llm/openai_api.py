import os
from openai import OpenAI
import argparse
import logging
from typing import Union, List
import json
import tqdm

chat_models = ['gpt-4o', 'gpt-4o-mini']

class OpenAILLM():
    def __init__(
        self,
        model: str = "gpt-3.5-turbo-instruct"
    ):
        self.model_name = model
        self.endpoint = OpenAI().chat.completions if model in chat_models else OpenAI().completions

        self.domain_model_name = self.model_name
        self.intent_model_name = self.model_name
        self.slot_model_name = self.model_name

    def run(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 1024,
        split_lines: bool = False,
        temperature: float = 0,
        frequency_penalty: float = 2.0,
        presence_penalty: float = 2.0,
        stop_sequences: List[str] = ["\\n"]
    ) -> str:
        assert 'OPENAI_API_KEY' in os.environ.keys(), \
            "Please set your OPENAI_API_KEY!"
            
        if self.model_name in chat_models:
            inputs = {
                'model': self.model_name,
                'messages': prompts,
                'temperature': temperature,
                }
        else:
            inputs = {
                'model': self.model_name,
                'prompt': prompts,
                'temperature': temperature,
                'max_tokens': max_new_tokens,
                'top_p': 1,
                'frequency_penalty': frequency_penalty,
                'presence_penalty': presence_penalty,
                'stop': stop_sequences}
                
        responses = self.endpoint.create(**inputs).choices
        if self.model_name in chat_models:
            responses = [response.message.content for response in responses]
        else:
            responses = [response.text.strip() for response in responses]
        if split_lines:
            responses_post = []
            for response in responses:
                if response:
                    responses_post.append(response.splitlines()[0])
                else:
                    responses_post.append(response)

        return responses
    
    def make_requests(self, prompts, max_tokens=10, do_sample=True, temperature=.0, top_p=1., stop_sequences = ["##EOS##"]):
        if self.model_name in chat_models:
            prompts = [{"role": "user", "content": prompt} for prompt in prompts]
        return self.run(prompts,max_new_tokens=max_tokens, split_lines = False, temperature=temperature, stop_sequences=stop_sequences)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the prompts to OpenAI.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file to save the responses from OpenAI.",
    )
    parser.add_argument(
        "--request_batch_size",
        default=5,
        type=int,
        help="The number of requests to send to OpenAI at a time."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    llm = OpenAILLM()

    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            all_prompts = [json.loads(line)["instruction"] + "\n### Input: " + json.loads(line)["input"] + "\n### Output:" for line in fin]
        else:
            all_prompts = [line.strip().replace("\\n", "\n") for line in fin]

    with open(args.output_file, "w") as fout:
        for i in tqdm.tqdm(range(0, len(all_prompts), args.request_batch_size)):
            batch_prompts = all_prompts[i: i + args.request_batch_size]

            results = llm.run(
                prompts=batch_prompts
            )
            for output in results:
                fout.write(json.dumps({"openai_output": output}) + "\n")