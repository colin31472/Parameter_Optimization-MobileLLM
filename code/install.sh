#!/bin/bash
#SBATCH --job-name=install_env
#SBATCH --partition=titanxp   
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
rm Miniconda3-latest-Linux-x86_64.sh

conda create -n mobilellm python=3.9 -y
conda activate mobilellm
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install numpy matplotlib -y
pip install transformers datasets