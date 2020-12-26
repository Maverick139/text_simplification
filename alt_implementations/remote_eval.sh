#!/bin/bash
#BATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

cd /scratch/kvaditya/fairseq_playground

echo " Running Evaluation"
echo "============================"

rm -rf eval
mkdir eval

for i in {1..50}
do
echo ">>> evaluating file: checkPoints/checkpoint$i.pt"
fairseq-generate preproc_data --path checkPoints/checkpoint$i.pt --batch-size 64 --beam 5 --remove-bpe > eval/fsq_gen_test_1000_"$i".txt
echo ">>> saved generated report to file: eval/fsq_gen_test_1110_$i.txt"
done

echo ">>> evaluating file: checkPoints/checkpoint_best.pt"
fairseq-generate preproc_data --path checkPoints/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe > eval/fsq_gen_test_1000_"$i".txt
echo ">>> saved generated report to file: eval/fsq_gen_test_1110_best.txt"

#for file in checkPoints/*
#do
#echo ">>> evaluating file: $file"
#i=$(echo "$file"|sed -e 's/.*[^0-9]\([0-9]\+\)[^0-9]*$/\1/');
#fairseq-generate preproc_data --path "$file" --batch-size 64 --beam 5 --remove-bpe > eval/fsq_gen_test_1110_"$i".txt
#echo ">>> saved generated report to file: eval/fsq_gen_test_0100_$i.txt"
#done

