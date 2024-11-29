BATCH_SIZE=4
GRADIENT_ACCUMULATION=8 # BATCH_SIZE * GRADIENT_ACCUMULATION == 16
EPOCHS=1
N_GPUS=4

#MODEL='EuropeanLLM-Beta/Teuken-7B-instruct-commercial-v0.4'
MODEL='/raid/s3/opengptx/paramita/models/teuken7B_sigma-mix-v11'
OUTPUT='/raid/s3/opengptx/lucas/kto-aligned-model/Teuken-7B-instruct-commercial-v0.4_tau_kto_DE_mix'

DATASET='/raid/s3/opengptx/lucas/data/tau_kto.jsonl'
EVAL_DATASET='trl-lib/kto-mix-14k'

TOKENIZER='/raid/s3/opengptx/paramita/teuken_tokenizer'

RUN_NAME=$(basename ${OUTPUT})

#export PYTHONPATH=$PYTHONPATH:trl/
export HF_HOME="/raid/s3/opengptx/models"
export WANDB_PROJECT='trl_kto'

accelerate launch --config_file=trl/examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml  --main_process_port 29500 --num_processes ${N_GPUS} \
            scripts/kto.py \
                --dataset_name ${DATASET}  \
                --dataset_name_eval ${EVAL_DATASET}  \
                --model_name_or_path ${MODEL}  \
                --per_device_train_batch_size ${BATCH_SIZE}  \
                --num_train_epochs ${EPOCHS}  \
                --learning_rate 5e-7  \
                --lr_scheduler_type=cosine  \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION}  \
                --logging_steps 10  \
                --save_total_limit 4  \
                --eval_steps 100 \
                --output_dir=${OUTPUT}  \
                --warmup_steps 42  \
                --bf16  \
                --save_only_model \
                --logging_first_step \
                --report_to wandb \
                --run_name ${RUN_NAME} \
                --trust_remote_code \
                #--tokenizer_name_or_path ${TOKENIZER} \
                #--load_hf_data_from_disk
                #--warmup_ratio 0.1 --save_steps 421  \
