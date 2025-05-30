#SBATCH --job-name=run-finetuned
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=wn[201-223]
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

srun singularity exec --nv ./containers/container-pytorch-updated.sif \
     python run_finetuned_model.py