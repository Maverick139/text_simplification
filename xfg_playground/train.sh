
echo " Training New Model"
echo "========================================================="

echo ">>> setting up environment"
rm -rf aug_data
rm -rf preproc_data
mkdir aug_data
mkdir aug_data/train aug_data/test aug_data/valid

echo ">>> augmenting data with specified feature tokens"
python3 src/feature_extract.py -s raw_data -d aug_data --nbchars --levsim

echo ">>> setting control tokens for test set"
python3 src/create_controlled_eval_set.py --in-dir raw_data --out-dir aug_data --nbchars 1.00 --levsim 0.75

echo ">>> converting data to bin and idx format"
fairseq-preprocess --source-lang complex --target-lang simple --trainpref aug_data/train/wiki.train --validpref aug_data/valid/wiki.valid --testpref aug_data/test/wiki.test --destdir preproc_data --workers 30

echo ">>> training model"
fairseq-train preproc_data --save-dir checkPoints --lr 0.00011 --lr-scheduler 'fixed' --optimizer adam --adam-betas '(0.9, 0.999)' --dropout 0.2 --arch 'transformer' --warmup-updates 5000 --encoder-embed-dim 256 --encoder-ffn-embed-dim 1024 --encoder-layers 6 --encoder-attention-heads 8 --decoder-layers 6 --decoder-attention-heads 8 --seed 0 --scoring bleu --decoder-embed-dim 256 --max-tokens 1000
