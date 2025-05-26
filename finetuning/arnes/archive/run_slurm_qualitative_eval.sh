#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=wn[201-224]        # only the V100 nodes
#SBATCH --time=02:00:00
#SBATCH --output=logs/qual_eval-%J.out
#SBATCH --error=logs/qual_eval-%J.err
#SBATCH --job-name="Qualitative Eval"

srun singularity exec --nv ./containers/container-pytorch-updated.sif \
    python qualitative_eval.py
