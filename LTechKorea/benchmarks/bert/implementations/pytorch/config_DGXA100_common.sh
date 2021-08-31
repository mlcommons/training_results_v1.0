## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/raid/datasets/bert/hdf5/v1p0_ref/4096_shards_uncompressed"
export EVALDIR="/raid/datasets/bert/hdf5/v1p0_ref/eval_uncompressed"
export DATADIR_PHASE2="/raid/datasets/bert/hdf5/v1p0_ref/4096_shards_uncompressed"
export CHECKPOINTDIR="$CI_BUILDS_DIR/$SLURM_ACCOUNT/$CI_JOB_ID/ci_checkpoints"
export CHECKPOINTDIR_PHASE1="/raid/datasets/bert/checkpoints/checkpoint_phase1"
#Optional
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
