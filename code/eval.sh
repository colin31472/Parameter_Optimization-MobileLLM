#!/bin/bash
#SBATCH --job-name=eval_mobilellm
#SBATCH --partition=titanxp
#SBATCH --gres=gpu:TitanXP:1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/miniconda3/bin/activate

conda activate mobilellm

export PYTHONIOENCODING=utf8

python eval.py