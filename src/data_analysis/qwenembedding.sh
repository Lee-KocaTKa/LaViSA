#!/bin/bash
#SBATCH --job-name=qwen_embed
#SBATCH --partition=public
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out


export PYTHONPATH=$PWD/src:$PYTHONPATH

python qwenembeddingextract.py 