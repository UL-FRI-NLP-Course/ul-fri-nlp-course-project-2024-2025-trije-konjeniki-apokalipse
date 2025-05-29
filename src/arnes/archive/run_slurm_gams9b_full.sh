#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --time=24:00:00
#SBATCH --output=logs/gams9b-%J.out
#SBATCH --error=logs/gams9b-%J.err
#SBATCH --job-name="GaMS-9B LoRA Full"

srun singularity exec --nv ./containers/container-pytorch-updated.sif \
    python run_gams9b.py