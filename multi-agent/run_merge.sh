#!/bin/bash
#SBATCH --job-name=green_patent_mas
#SBATCH --output=/ceph/home/student.aau.dk/gy53xm/AppliedDL/logs/finetune_%A.out
#SBATCH --error=/ceph/home/student.aau.dk/gy53xm/AppliedDL/logs/finetune_%A.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --time=12:00:00


uv run --no-sync python merge_qlora.py

