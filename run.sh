#!/bin/bash
#SBATCH --job-name=llava_anaph
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/home/sangmyeong-l/research/ARR_May_2026
#SBATCH -o slurm-%j.out

export PYTHONPATH=/mnt/home/sangmyeong-l/research/ARR_May_2026/src
python -m scripts.run_llava_onevision --category anaph 