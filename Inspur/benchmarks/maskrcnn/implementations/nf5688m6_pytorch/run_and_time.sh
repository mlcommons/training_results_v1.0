cd ../pytorch
source config_NF5688M6.sh
DGXSYSTEM=NF5688M6 CONT=mlperf-inspur:maskrcnn DATADIR=/path/to/preprocessed/data LOGDIR=LOGDIR=/path/to/logfile ./run_with_docker.sh
