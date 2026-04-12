#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
python run_llava_onevision.py --category anaph 
