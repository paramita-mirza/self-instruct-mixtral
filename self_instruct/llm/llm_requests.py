import json
import requests
import logging
import os

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
    )

class LLMRequests():

    host = "0.0.0.0"
    port = 8007
    logger = logging.getLogger(os.path.basename(__file__))

    @classmethod
    def make_requests(cls,
            prompts, max_tokens, do_sample, temperature, top_p, stop_sequences
    ):
        request_body = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop_sequences
        }
        response = requests.post(f"http://{cls.host}:{cls.port}/llm_run/", data=json.dumps(request_body))
        # cls.logger.info(response)
        # cls.logger.info(response.json())
        return response.json()

    @classmethod
    def check_api(cls):
        try:
            response = requests.get(f"http://{cls.host}:{cls.port}/health/")
        except Exception as e:
            cls.logger.info(f"NOT OK: {str(e)}")
        else:
            if response.status_code == 200:
                cls.logger.info(f"OK: API is live")
                return True
            else:
                cls.logger.info(f"NOT OK: HTTP response code {response.status_code}")
                return False

    @classmethod
    def termination_request(cls):
        response = requests.get(f"http://{cls.host}:{cls.port}/shutdown_uvicorn/")
        cls.logger.info(response.text)
