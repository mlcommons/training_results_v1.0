## System config params
export DGXNGPU=8
export DGXSOCKETCORES=20
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/fs/sjc1-lcl01/ent/mlperf/bmark/bert/v1p0/2048_shards_uncompressed"
export EVALDIR="/fs/sjc1-lcl01/ent/mlperf/bmark/bert/v1p0/eval_uncompressed"
export DATADIR_PHASE2="/fs/sjc1-lcl01/ent/mlperf/bmark/bert/v1p0/2048_shards_uncompressed"
export CHECKPOINTDIR="$CI_BUILDS_DIR/$SLURM_ACCOUNT/$CI_JOB_ID/ci_checkpoints"
export CHECKPOINTDIR_PHASE1="/fs/sjc1-lcl01/ent/mlperf/bmark/bert/phase1_checkpoint/"
export UNITTESTDIR="/fs/sjc1-lcl01/ent/mlperf/bmark/bert/unit_test"
