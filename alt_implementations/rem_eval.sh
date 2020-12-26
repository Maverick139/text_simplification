#!/bin/bash
#BATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

USR="kvaditya"

cd /scratch/"$USR"/xfg_playground
bash evaluate.sh

scp -r eval "$USR"@ada.iiit.ac.in:/share1/"$USR"/fsq_eval/model17/
