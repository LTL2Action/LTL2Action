#!/bin/bash
#SBATCH --array=1-5
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=t4v2
#SBATCH --qos=normal

base_cmd=$1 #e.g. python3 train_agent.py --algo ppo --env Letter-7x7-v2 --save-interval 100 --frames 1000000000 --ltl-sampler UntilTasks_3_3_1_1 --lr 0.0001
seed=$SLURM_ARRAY_TASK_ID

eval "$base_cmd --seed $seed --checkpoint-dir /checkpoint/${USER}/${SLURM_JOB_ID}"
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE
