#!/bin/bash
LANGUAGE=DE
TASK=trans_evalset_v1
NUM_DEVICES=2
MODEL1=Meta-Llama-3.1-70B-Instruct #ALMA-13B 
MODEL2=deepl
JUDGE=meta-llama/Meta-Llama-3.1-70B-Instruct #gpt-4o #meta-llama/Meta-Llama-3.1-70B-Instruct
DEPLOYMENT=vllm_local #openai #
MAX_MODEL_LENGTH=6000

OUTPUT_DIR=output

OPENAI_API_KEY=$(head -n 1 openai-api-key.txt)
export OPENAI_API_KEY=$OPENAI_API_KEY

# This script uses the vllm and hf apis from self_instruct
export PYTHONPATH=$PYTHONPATH:../self_instruct_alpaca/

python judge_translations.py  --lang ${LANGUAGE} \
                --task_name ${TASK}  \
                --out_dir  ${OUTPUT_DIR} \
                --deployment ${DEPLOYMENT} \
                --num_devices ${NUM_DEVICES} \
                --judge ${JUDGE} \
                --model1 ${MODEL1} \
                --model2 ${MODEL2} \
                --max_model_len ${MAX_MODEL_LENGTH} 
