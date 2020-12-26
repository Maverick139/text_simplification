#!/bin/bash

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
fairseq-train data/preproc_data --save-dir checkPoints --lr 0.00011 --lr-scheduler 'fixed' --optimizer adam --adam-betas '(0.9, 0.999)' --dropout 0.2 --arch 'transformer' --warmup-updates 5000 --encoder-embed-dim 128 --encoder-ffn-embed-dim 512 --encoder-layers 3 --encoder-attention-heads 4 --decoder-layers 3 --decoder-attention-heads 4 --seed 0 --scoring bleu --decoder-embed-dim 128 --max-tokens 1000
