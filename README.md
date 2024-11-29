# SIGMA: Synthetic Instruction Generation with Mistral AI

This repository contains code and data for the [Alpaca-style self-instruct](https://github.com/tatsu-lab/stanford_alpaca) 
to generate instruction-following data for fine-tuning LLMs, and made the following modifications:

- We use open-source LLMs (e.g., [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)) 
instead of OpenAI models.
- We use in-memory vector db ([Chroma](https://www.trychroma.com/)) for similarity comparison, which is much faster than 
calculating rouge score for each generated instruction.

**Self-instruct** is a framework that helps language models improve their ability to follow natural language instructions. 
It does this by distilling the instructional knowledge of powerful LLMs to create a large collection of instructional data. With self-instruct, it is possible to improve the instruction-following capabilities of language models without relying on extensive manual annotation.

Related papers and repos:

- Self-Instruct: Aligning LM with Self Generated Instructions [paper](https://arxiv.org/abs/2212.10560) | [code](https://github.com/yizhongw/self-instruct)
- Stanford Alpaca: An Instruction-following LLaMA Model [paper](https://crfm.stanford.edu/2023/03/13/alpaca.html) | [code](https://github.com/tatsu-lab/stanford_alpaca)
- airoboros: using large language models to fine-tune large language models [code](https://github.com/jondurbin/airoboros)

### Data Generation Process

#### Running the code

1. Create a virtual environment ``python -m venv sigma`` then activate `source sigma/bin/activate`
2. Set environment variables HF_TOKEN to your Huggingface access token
3. Install the dependencies with ``pip install -r requirements.txt``
4. Run ``python self_instruct_alpaca/generate_instructions.py``, e.g.,

```
batch_dir=self_instruct_alpaca/data/mixtral_8x22b_generations/
hf_model_id=mistralai/Mixtral-8x22B-Instruct-v0.1
hf_cache_dir=/raid/s3/opengptx/models/

python self_instruct_alpaca/generate_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 100 \
    --hf_model_id ${hf_model_id} \
    --hf_cache_dir ${hf_cache_dir} \
    --seed_tasks_path self_instruct_alpaca/data/seed_tasks_plus.jsonl \
    --temperature 1.0 \
    --top_p 1.0
```

The default deployment mode for inference is [vLLM](https://github.com/vllm-project/vllm). 
For inference with multiple GPUs set ``--num_devices <number of GPUs to use>``.

#### Deployment for inference via FastAPI
There is an option to deploy and then inference a model via [FastAPI](https://fastapi.tiangolo.com/) by running
`python self_instruct_alpaca/llm/huggingface_fastapi.py` and then adding the argument ``--fastapi`` when running 
`python self_instruct_alpaca/generate_instructions.py`, e.g., 

```
batch_dir=self_instruct_alpaca/data/mixtral_8x22b_generations/
hf_model_id=mistralai/Mixtral-8x22B-Instruct-v0.1
hf_cache_dir=/raid/s3/opengptx/models/
port=8007

nohup python self_instruct_alpaca/llm/huggingface_fastapi.py \
    --hf_model_id ${hf_model_id} \
    --hf_cache_dir ${hf_cache_dir} \
    --port ${port} &
    
python -u self_instruct_alpaca/generate_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 100 \
    --hf_model_id ${hf_model_id} \
    --hf_cache_dir ${hf_cache_dir} \
    --seed_tasks_path self_instruct_alpaca/data/seed_tasks_plus.jsonl \
    --temperature 1.0 \
    --top_p 1.0 \
    --model_deployment fastapi
```

#### Deployment for inference via NVIDIA NeMo Framework

...
