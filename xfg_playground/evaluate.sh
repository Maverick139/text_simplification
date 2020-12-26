
echo " Running Evaluation"
echo "============================"

rm -rf eval
mkdir eval

fcode="1001"

for i in {1..50}
do
echo ">>> evaluating file: checkPoints/checkpoint$i.pt"
fairseq-generate preproc_data --path checkPoints/checkpoint$i.pt --batch-size 64 --beam 5 --remove-bpe > eval/fsq_gen_test_"$fcode"_"$i".txt
echo ">>> saved generated report to file: eval/fsq_gen_test_"$fcode"_$i.txt"
done

echo ">>> evaluating file: checkPoints/checkpoint_best.pt"
fairseq-generate preproc_data --path checkPoints/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe > eval/fsq_gen_test_"$fcode"_"$i".txt
echo ">>> saved generated report to file: eval/fsq_gen_test_"$fcode"_best.txt"

#for file in checkPoints/*
#do
#echo ">>> evaluating file: $file"
#i=$(echo "$file"|sed -e 's/.*[^0-9]\([0-9]\+\)[^0-9]*$/\1/');
#fairseq-generate preproc_data --path "$file" --batch-size 64 --beam 5 --remove-bpe > eval/fsq_gen_test_1110_"$i".txt
#echo ">>> saved generated report to file: eval/fsq_gen_test_0100_$i.txt"
#done

