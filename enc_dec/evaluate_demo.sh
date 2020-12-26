#!/bin/bash

echo " Running Evaluation"
echo "============================"

rm -rf eval
mkdir eval

fcode="1110"	# model code (refer to README)

for i in {1..5}
do
echo ">>> evaluating file: checkPoints/checkpoint$i.pt"
fairseq-generate preproc_data --path checkPoints/checkpoint$i.pt --batch-size 64 --beam 5 --remove-bpe > eval/fsq_gen_test_"$fcode"_"$i".txt
echo ">>> saved generated report to file: eval/fsq_gen_test_"$fcode"_$i.txt"
done