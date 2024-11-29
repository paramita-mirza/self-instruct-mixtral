BATCH_SIZE=1 
GRADIENT_ACCUMULATION=16 # BATCH_SIZE * GRADIENT_ACCUMULATION == 16
EPOCHS=1
N_GPUS=2

DATASET='argilla/dpo-mix-7k'
MODEL='/raid/s3/opengptx/lucas/out/finetune/mistral-7b-v0.3_sigma_v3.3_soft_limits_mp_hf1'
OUTPUT='/raid/s3/opengptx/lucas/dpo-aligned-model/mistral7B-v0.3_sigma-v3.3_soft_limits'

cd trl

accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml --main_process_port 29501 --num_processes ${N_GPUS} \
   examples/scripts/dpo.py \
    --dataset_name ${DATASET}   \
    --model_name_or_path ${MODEL} \
    --learning_rate 5.0e-7 \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 500 \
    --output_dir ${OUTPUT} \
    --no_remove_unused_columns \
    --bf16
