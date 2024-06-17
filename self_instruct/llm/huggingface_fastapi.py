import os
import signal
from datetime import datetime
import argparse
from pydantic import BaseModel
from fastapi import FastAPI, Depends, Response
import uvicorn
from contextlib import asynccontextmanager
from huggingface_api import HuggingFaceLLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_id",
        type=str,
        help="The model name.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the API.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8007,
        help="Port for the API.",
    )
    return parser.parse_args()

args = parse_args()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    app.state.llm = HuggingFaceLLM(model_name=args.hf_model_id)
    yield
    # Clean up the ML models and release the resources
    # app.state.llm.clear()
    del app.state.llm

app = FastAPI(lifespan=lifespan)

@app.get('/shutdown_uvicorn/')
async def shutdown_uvicorn():
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=200, content="Server shutting down .......")

@app.get('/health/')
async def check_api_status():
    return Response(status_code=200, content="FastAPI is up...")

class RequestModel(BaseModel):
    prompts: list = []
    max_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    stop_sequences: list = []

class ResponseModel(BaseModel):
    prompt: str
    response: str
    created_at: str

@app.post('/llm_run/')
async def llm_run(inputs: RequestModel) -> list[ResponseModel]:
    response = app.state.llm.run(prompts=inputs.prompts,
                       max_new_tokens=inputs.max_tokens,
                       do_sample=inputs.do_sample,
                       temperature=inputs.temperature,
                       top_p=inputs.top_p,
                       stop_strings=inputs.stop_sequences)
    
    if isinstance(inputs.prompts, list):
        results = []
        for j, prompt in enumerate(inputs.prompts):
            data = ResponseModel(
                prompt=prompt,
                response=response[j] if response else None,
                created_at=str(datetime.now())
            )
            results.append(data)
        return results
    else:
        data = ResponseModel(
            prompt=inputs.prompts,
            response=response,
            created_at=str(datetime.now()),
        )
        return [data]
    
if __name__ == "__main__":
    uvicorn.run("huggingface_fastapi:app", host=args.host, port=args.port, reload=False)


