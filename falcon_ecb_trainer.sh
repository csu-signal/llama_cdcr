#!/bin/bash

#SBATCH --job-name="LLAMA2_7B_coref_ECB"   # job name
#SBATCH --partition=peregrine-gpu                # partition to which job should be submitted
#SBATCH --qos=gpu_medium                                 # qos type
#SBATCH --nodes=1                                # node count
#SBATCH --ntasks=1                               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4                        # cpu-cores per task
#SBATCH --mem=40G                                # total memory per node
#SBATCH --gres=gpu:a100-sxm4-80gb:1 # Request 1 GPU (A100 80GB)
#SBATCH --time=48:00:00                                  #  wall time

source activate nlg
torchrun --nproc_per_node 2 example_chat_completion_falcon.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 2048 --max_batch_size 40
