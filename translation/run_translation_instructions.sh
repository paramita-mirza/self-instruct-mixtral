#!/bin/bash
LANGUAGE=DE
TASKS=/raid/s3/opengptx/lucas/data/trans_evalset_v1.jsonl
LIMIT=185
NUM_DEVICES=4
TRANSLATOR=mistralai/Mixtral-8x22B-Instruct-v0.1 #haoranxu/ALMA-13B #meta-llama/Meta-Llama-3.1-70B-Instruct #mistralai/Mistral-7B-Instruct-v0.3 #deepl
DEPLOYMENT=vllm_local #deepl 

if [ "$TASKS" == "/raid/s3/opengptx/lucas/data/trans_evalset_v1.jsonl" ]; then
    OUTPUT_DIR=output/evalset_translations
else
    OUTPUT_DIR=/raid/s3/opengptx/lucas/data
fi

# This script uses the vllm and hf apis from self_instruct
export PYTHONPATH=$PYTHONPATH:../self_instruct_alpaca/

python main.py  --lang ${LANGUAGE} \
                --tasks ${TASKS}  \
                --out_dir  ${OUTPUT_DIR} \
                --deployment ${DEPLOYMENT} \
                --limit ${LIMIT} \
                --num_devices ${NUM_DEVICES} \
                --model ${TRANSLATOR}
