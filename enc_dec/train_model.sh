#!/bin/bash
#BATCH -A research
#SBATCH -n 30
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

module add cuda/10.0
module add cudnn/7.6-cuda-10.0

echo "\n >>> setting up fair-seq environment"

source anaconda3/bin/activate gpu

#python3 simplewiki.py

mkdir -p /scratch/kvaditya

cp -r fairseq_playground /scratch/kvaditya/

cd /scratch/kvaditya/fairseq_playground

echo "\n>>> augmenting data with specified feature tokens"
python3 feature_extract.py -s raw_data -d aug_data --nbchars

echo "\n>>> converting data to bin and idx format"
fairseq-preprocess --source-lang complex --target-lang simple --trainpref raw_data/train/wiki.train --validpref raw_data/valid/wiki.valid --testpref raw_data/test/wiki.test --destdir preproc_data --workers 20

echo "\n>>> training model"
fairseq-train preproc_data --save-dir checkPoints --lr 0.00011 --lr-scheduler 'fixed' --optimizer adam --adam-betas '(0.9, 0.999)' --dropout 0.2 --arch 'transformer' --warmup-updates 5000 --encoder-embed-dim 256 --encoder-ffn-embed-dim 1024 --encoder-layers 6 --encoder-attention-heads 8 --decoder-layers 6 --decoder-attention-heads 8 --seed 0 --scoring bleu --decoder-embed-dim 256 --max-tokens 1000

echo "\n>>> training complete"
