#/usr/bin/bash

RANDOM_SEED=`date +%s`

QUALITY=0.759

set -e

# Register the model as a source root

export PYTHONPATH="$(pwd):$(pwd)/../:${PYTHONPATH}"

# MLPerf

export PYTHONPATH="$(pwd)/../logging:${PYTHONPATH}"

echo $PYTHONPATH

mkdir -p /scratch/IntelMLPerf/

MODEL_DIR="/scratch/IntelMLPerf/resnet_imagenet_${RANDOM_SEED}"

export OMP_NUM_THREADS=12

export KMP_BLOCKTIME=1

pdsh -w ocpx04,ocpx07,ocpx09,ocpx10,ocpx06,ocpx12,ocpx13,ocpx14 cacheclr3
# changing to mini batch 64 to keep hyper-parameters the same
mpirun -n 128 -ppn 16 -host ocpx04,ocpx07,ocpx09,ocpx10,ocpx06,ocpx12,ocpx13,ocpx14  --map-by ppr:2:socket:pe=14 -genv FI_PROVIDER=psm3 -genv PSM3_ALLOW_ROUTERS=1 -genv PSM3_RDMA=0  python imagenet_main.py $RANDOM_SEED --data_dir /scratch/TF_Imagenet_FullData/ --model_dir $MODEL_DIR --train_epochs 48 --stop_threshold $QUALITY --batch_size 64 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.0002  2>&1 |tee lars8192-mb64-48epochs_eval_every_4_epochs_${RANDOM_SEED}.log
