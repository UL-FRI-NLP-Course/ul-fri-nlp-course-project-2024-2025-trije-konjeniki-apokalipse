#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=wn[201-224]        # only the V100 nodes
#SBATCH --time=02:00:00
#SBATCH --output=logs/gams2b-%J.out
#SBATCH --error=logs/gams2b-%J.err
#SBATCH --job-name="GaMS-2B LoRA Test"

srun singularity exec --nv ./containers/container-pytorch-updated.sif \
    python run_gams2b_json.py