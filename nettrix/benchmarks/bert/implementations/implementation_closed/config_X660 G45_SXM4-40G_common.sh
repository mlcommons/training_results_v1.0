## System config params
export DGXNGPU=8
export DGXSOCKETCORES=40
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/mlperf-datasets/bert/2048_shards_uncompressed"
export EVALDIR="/mlperf-datasets/bert/eval_set_uncompressed"
export DATADIR_PHASE2="/mlperf-datasets/bert/2048_shards_uncompressed"
export CHECKPOINTDIR="/mlperf-datasets/bert/ck_pt/"
export CHECKPOINTDIR_PHASE1="/mlperf-datasets/bert/ck_pt/"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
