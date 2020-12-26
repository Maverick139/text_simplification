#!/bin/bash
#SBATCH -A research
#SBATCH -n 30
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=k.v.aditya@research.iiit.ac.in

module add cuda/10.0
module add cudnn/7.6-cuda-10.0

source anaconda3/bin/activate gpu

sleep 120

rm -rf /scratch/kvaditya/
mkdir -p /scratch/kvaditya

cp -r xfg_playground /scratch/kvaditya/

cd /scratch/kvaditya/xfg_playground/

bash train.sh
