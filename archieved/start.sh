#!/bin/bash
#SBATCH --job-name=a2
#SBATCH --output=logs/a2_%A_%a.out
#SBATCH --error=logs/a2_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=40G
#SBATCH --time=12:00:00



# 这些环境变量会自动传进容器里，shard_llm.py 用 os.environ.get(...) 就能读到
#singularity exec --nv llm.sif python3 A2.py
singularity exec --nv \
     -B venv:/scratch/my_venv \
     llm.sif \
     /bin/bash -c "source /scratch/my_venv/bin/activate && python prompt_test.py"