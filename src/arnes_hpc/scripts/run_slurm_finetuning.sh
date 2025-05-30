#!/bin/bash
#SBATCH --job-name=gams9b-h100
#SBATCH --partition=gpu
#SBATCH --nodelist=gwn[08-10]
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

srun singularity exec --nv ./containers/container-pytorch-updated.sif \
     python finetuning.py