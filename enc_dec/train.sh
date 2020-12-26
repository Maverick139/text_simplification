#!/bin/bash
#SBATCH -A research
#SBATCH -n 30
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

module add cuda/10.0
module add cudnn/7.6-cuda-10.0
source anaconda3/bin/activate gpu
mkdir -p /scratch/$USER
cp -r fairseq_playground /scratch/$USER/
cd /scratch/$USER/fairseq_playground

# ^ For ADA Usage only

echo " Training New Model"
echo "========================================================="

echo ">>> setting up environment"
rm -rf data/aug_data
rm -rf data/preproc_data
mkdir data/aug_data
mkdir data/aug_data/train data/aug_data/test data/aug_data/valid

echo ">>> augmenting data with specified feature tokens"
python3 src/feature_extract.py -s data/raw_data -d data/aug_data --nbchars --deptree

echo ">>> setting control tokens for test set"
python3 src/create_controlled_eval_set.py --in-dir data/raw_data --out-dir data/aug_data --nbchars 1.00 --deptree 0.50

echo ">>> converting data to bin and idx format"
fairseq-preprocess --source-lang complex --target-lang simple --trainpref data/aug_data/train/wiki.train --validpref data/aug_data/valid/wiki.valid --testpref data/aug_data/test/wiki.test --destdir data/preproc_data --workers 30

echo ">>> training model"
fairseq-train data/preproc_data --save-dir checkPoints --lr 0.00011 --lr-scheduler 'fixed' --optimizer adam --adam-betas '(0.9, 0.999)' --dropout 0.2 --arch 'transformer' --warmup-updates 5000 --encoder-embed-dim 256 --encoder-ffn-embed-dim 1024 --encoder-layers 6 --encoder-attention-heads 8 --decoder-layers 6 --decoder-attention-heads 8 --seed 0 --scoring bleu --decoder-embed-dim 256 --max-tokens 1000
