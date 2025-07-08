#!/bin/bash
#SBATCH --job-name=llm-lab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Example script for a CPU-only cluster. Remove GPU modules.

export MASTER_PORT=12345

srun torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node $SLURM_NTASKS_PER_NODE \
    labs/simple_model/train_simple.py --epochs 1

