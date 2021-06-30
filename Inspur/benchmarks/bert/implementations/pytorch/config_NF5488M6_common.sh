## System config params
export DGXNGPU=8
export DGXSOCKETCORES=32
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/mlperf/work/data/bert/2048_shards_uncompressed"
export EVALDIR="/mlperf/work/data/bert/eval_uncompressed"
export DATADIR_PHASE2="/mlperf/work/data/bert/2048_shards_uncompressed"
export CHECKPOINTDIR="/mlperf/work/data/bert/checkpoints"
export CHECKPOINTDIR_PHASE1="/mlperf/work/data/bert/checkpoints"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
