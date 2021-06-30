cd ../pytorch
source config_NF5488A5.sh
DGXSYSTEM=NF5488A5 CONT=mlperf-inspur:maskrcnn DATADIR=/path/to/preprocessed/data LOGDIR=LOGDIR=/path/to/logfile ./run_with_docker.sh
