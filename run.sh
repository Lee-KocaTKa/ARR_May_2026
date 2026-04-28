#!/bin/bash
#SBATCH --job-name=qwen_pp
#SBATCH --partition=public
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/home/sangmyeong-l/research/ARR_May_2026
#SBATCH -o slurm-%j.out

export PYTHONPATH=/mnt/home/sangmyeong-l/research/ARR_May_2026/src



# Run evaluation
python -m scripts.run_qwen --category ellip
# Stop server after evaluation
kill $SERVER_PID