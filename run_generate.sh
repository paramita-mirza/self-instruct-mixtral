#!/bin/bash

#SBATCH --job-name="self-instruct"
#SBATCH --time=07-00:00:00
#SBATCH --partition=alpha
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=8
#SBATCH --mem-per-gpu=15GB
#SBATCH --nodes=1
#SBATCH --output=log_10k.out
#SBATCH --error=log_10k.err

module load release/23.04  GCC/11.3.0  Python/3.10.4
source /data/horse/ws/pami666g-transformers/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u self_instruct/generate_instructions.py --batch_dir data/mixtral_8x22b_generations_alpaca_10k/ --num_instructions_to_generate 10000 --seed_tasks_path data/seed_tasks.jsonl --hf_model_id mistralai/Mixtral-8x22B-Instruct-v0.1 --prompt_instructions data/prompt.txt

deactivate