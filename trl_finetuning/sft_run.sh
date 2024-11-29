BATCH_SIZE=4
GRADIENT_ACCUMULATION=8 # BATCH_SIZE * GRADIENT_ACCUMULATION == 16
EPOCHS=2
N_GPUS=4

export CUDA_VISIBLE_DEVICES=2,4,5,6

#MODEL='mistralai/Mistral-7B-v0.3'
#OUTPUT='/raid/s3/opengptx/paramita/models/mistral7B-v0.3_spectrum_sigma-mix-v9'

#MODEL='EuropeanLLM-Beta/HalloEurope-7B'
MODEL='EuropeanLLM-Beta/Teuken-7B-base-v0.5'
TOKENIZER='/raid/s3/opengptx/paramita/teuken_tokenizer'
OUTPUT='/raid/s3/opengptx/paramita/models/teuken7B-base-v05_sigma-mix-v20-mp'

DATASET='/raid/s3/opengptx/lucas/data/sigma_mix_v20.jsonl'
EVAL_DATASET='/raid/s3/opengptx/lucas/data/tulu_v2_val.jsonl'

RUN_NAME=$(basename ${OUTPUT})

#export PYTHONPATH=$PYTHONPATH:trl/
export HF_HOME="/raid/s3/opengptx/models"
export WANDB_PROJECT='trl_sft'

#SPECTRUM_PARAMETERS='/raid/s3/opengptx/paramita/instruction_tuning/spectrum/snr_results_mistralai-Mistral-7B-v0.3_unfrozenparameters_50percent.yaml'
SPECTRUM_PARAMETERS='/raid/s3/opengptx/paramita/instruction_tuning/spectrum/snr_results_EuropeanLLM-Beta-HalloEurope-7B_unfrozenparameters_50percent.yaml'

#!!!NOTE!!!
# * learning_rate 1.0e-05 works well for Mistral, not so much for Teuken

#accelerate launch --main_process_port 29501 \
accelerate launch --config_file=trl/examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml  --main_process_port 29500 --num_processes ${N_GPUS} \
            scripts/sft.py \
                --dataset_name ${DATASET}  \
                --dataset_train_split 'train' \
                --dataset_test_split 'test' \
                --model_name_or_path ${MODEL} \
                --trust_remote_code \
                --num_train_epochs ${EPOCHS}  \
                --learning_rate 1.0e-05  \
                --lr_scheduler_type cosine  \
                --warmup_ratio 0.03 \
                --max_seq_length 2048 \
                --packing True \
                --per_device_train_batch_size ${BATCH_SIZE} \
                --per_device_eval_batch_size ${BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION}  \
                --gradient_checkpointing \
                --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
                --logging_steps 1  \
                --save_strategy steps \
                --eval_strategy steps \
                --eval_steps 20 \
                --output_dir ${OUTPUT}  \
                --overwrite_output_dir \
                --bf16 True \
                --tf32 True \
                --weight_decay 0.1 \
                --logging_first_step \
                --report_to wandb \
                --run_name ${RUN_NAME} \
                --neftune_noise_alpha=5 \
                --completion_only \
                --load_best_model_at_end \
                --save_total_limit 1 \
                #--save_only_model \
                #--dataset_name_eval ${EVAL_DATASET}  \
                #--tokenizer_name_or_path ${TOKENIZER} \
                #--eval_packing False \
                #--max_grad_norm 0.3 \
                #--spectrum_parameters ${SPECTRUM_PARAMETERS} \



