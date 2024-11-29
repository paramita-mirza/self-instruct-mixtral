import json
import requests
import re
from datetime import datetime

# THIS IS THE DEFAULT PROMPTING TEMPLATE OF NEMOTRON-4
PROMPT_TEMPLATE = """<extra_id_0>System

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
"""

class LLMRequests_Nemo():
    def __init__(self) -> None:
        pass

    def _postprocess_responses(self, response, prompt, stop_strings):
        """ Note: placeholder, currently not used"""
        if response.endswith("<extra_id_1>"):
            response = response[:-len("<extra_id_1>")] 
        return response.strip()
    
    def _format_responses(self, responses, prompts, stop_sequences=[], postprocess=False, batch=True):
        if postprocess:
            responses = self._postprocess_responses(responses, prompts, stop_sequences)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        out = [{'prompt': prompt, 'response': response[len(prompt):], 'created_at': timestamp} for prompt, response in zip(prompts, responses)]
        return out

    def _text_generation(self, data, ip='localhost', port=None):
        resp = requests.put(f'http://{ip}:{port}/generate', data=json.dumps(data), headers={"Content-Type": "application/json"})
        return resp.json()

    def _get_generation(self, prompt, stop_sequences=["<|endoftext|>", "<extra_id_1>", "\x11", "<extra_id_1>User"],
                        do_sample=False, add_BOS=False, max_tokens=3072, min_tokens=1, temperature=1.0, top_p=1.0, 
                        top_k=0, repetition=1.0, batch=True):
        data = {
            "sentences": [prompt] if not batch else prompt,
            "tokens_to_generate": int(max_tokens),
            "temperature": temperature,
            "add_BOS": add_BOS,
            "top_k": top_k,
            "top_p": top_p,
            "greedy": not do_sample,
            "all_probs": False,
            "repetition_penalty": repetition,
            "min_tokens_to_generate": int(min_tokens),
            "end_strings": stop_sequences,
        }
        sentences = self._text_generation(data, port=1424)['sentences']
        return sentences[0] if not batch else sentences

    def make_requests(self, prompts, stop_sequences, **kwargs):
        responses = self._get_generation(prompts, stop_sequences,  **kwargs)
        results = self._format_responses(responses, prompts, stop_sequences=stop_sequences, postprocess=False)
        return results

if __name__ == '__main__':
    pass