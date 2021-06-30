CONT=nvcr.io/nvdlfwea/mlperfv1/ssd:20210517.mxnet DATADIR=/mlperf/work/data/ssd LOGDIR=/mlperf/work/code/ssd/logs/logs2 DGXSYSTEM=NF5488M6 ./run_with_docker.sh 2>&1 | tee log_ssd.log
