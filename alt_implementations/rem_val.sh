#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=k.v.aditya@research.iiit.ac.in

USR="kvaditya"
fcode="1001"

module add cuda/10.0
module add cudnn/7.6-cuda-10.0

echo " Running Validation Eval"
echo "============================"

cd /scratch/"$USR"/xfg_playground

rm -rf valid_data
cp -r preproc_data/ valid_data

rm valid_data/test.complex-simple.*
rm valid_data/train.complex-simple.*
mv valid_data/valid.complex-simple.complex.idx valid_data/test.complex-simple.complex.idx
mv valid_data/valid.complex-simple.complex.bin valid_data/test.complex-simple.complex.bin
mv valid_data/valid.complex-simple.simple.idx valid_data/test.complex-simple.simple.idx
mv valid_data/valid.complex-simple.simple.bin valid_data/test.complex-simple.simple.bin

rm -rf val
mkdir val

for i in {1..50}
do
echo ">>> evaluating file: checkPoints/checkpoint$i.pt"
fairseq-generate valid_data --path checkPoints/checkpoint$i.pt --batch-size 64 --beam 5 --remove-bpe > val/fsq_gen_test_"$fcode"_"$i".txt
echo ">>> saved generated report to file: eval/fsq_gen_test_"$fcode"_$i.txt"
done

echo ">>> evaluating file: checkPoints/checkpoint_best.pt"
fairseq-generate valid_data --path checkPoints/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe > val/fsq_gen_test_"$fcode"_"$i".txt
echo ">>> saved generated report to file: eval/fsq_gen_test_"$fcode"_best.txt"

scp -r val "$USR"@ada.iiit.ac.in:/share1/"$USR"/fsq_eval/model17/

