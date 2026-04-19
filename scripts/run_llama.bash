#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=public
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
python run_llama4_eval.py --category anaph 