python dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" \
       --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset \
       --data-set=terabyte --raw-data-file=/data/day --processed-data-file=/data/day --loss-function=bce \
       --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2048 --print-time \
       --test-freq=102400 --test-mini-batch-size=16384 --test-num-workers=16 --memory-map --mlperf-logging \
       --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle \
       --mlperf-coalesce-sparse-grads --use-gpu