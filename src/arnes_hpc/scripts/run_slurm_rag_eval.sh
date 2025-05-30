#SBATCH --job-name=run-rag
#SBATCH --partition=gpu
#SBATCH --nodelist=wn[201-223]
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

srun singularity exec --nv ./containers/container-pytorch-rag.sif \
     python run_rag_model.py