set -x

mkdir -p processed_logs/

for((i=1;$i<10;i++));do

python postprocess.py --in-file raw_logs/train_run${i}.log --out-file processed_logs/train_run${i}.log
sh logging_postprocess.sh processed_logs/train_run${i}.log raw_logs/eval_run${i}.stdout  result_run${i}.txt

done
