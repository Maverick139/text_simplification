#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

# ^ For ADA Usage only 

echo " Running Evaluation"
echo "============================"

rm -rf eval
mkdir eval

fcode="1001"	# model code (refer to README)

for i in {1..50}	# for first 50 epochs
do
echo ">>> evaluating file: checkPoints/checkpoint$i.pt"
fairseq-generate data/preproc_data --path checkPoints/checkpoint$i.pt --batch-size 64 --beam 5 --remove-bpe > eval/fsq_gen_test_"$fcode"_"$i".txt
echo ">>> saved generated report to file: eval/fsq_gen_test_"$fcode"_$i.txt"
done

echo ">>> evaluating file: checkPoints/checkpoint_best.pt"
fairseq-generate data/preproc_data --path checkPoints/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe > eval/fsq_gen_test_"$fcode"_"$i".txt
echo ">>> saved generated report to file: eval/fsq_gen_test_"$fcode"_best.txt"

