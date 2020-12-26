#!/bin/bash
#BATCH -A research
#SBATCH -n 2
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=k.v.aditya@research.iiit.ac.in

cd /scratch/kvaditya/xfg_playground
scp -r eval kvaditya@ada.iiit.ac.in:/share1/kvaditya/fsq_eval/model11/

