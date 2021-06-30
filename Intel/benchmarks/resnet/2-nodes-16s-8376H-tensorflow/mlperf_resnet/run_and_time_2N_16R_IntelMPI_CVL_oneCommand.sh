#/usr/bin/bash

echo 3 > /proc/sys/vm/drop_caches 

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

export OMP_NUM_THREADS=24
#export OMP_NUM_THREADS=12

export KMP_BLOCKTIME=1

#mpirun -n 16  --map-by ppr:2:socket:pe=14 python imagenet_main.py $RANDOM_SEED --data_dir /tmp/TF_Imagenet_FullData/ --model_dir $MODEL_DIR --train_epochs 41 --stop_threshold $QUALITY --batch_size 128 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.0002  2>&1 |tee lars-41epochs_eval_every_4_epochs_${RANDOM_SEED}.log
#mpirun -n 16  --map-by ppr:2:socket:pe=14 python imagenet_main.py $RANDOM_SEED --data_dir /scratch/TF_Imagenet_FullData/ --model_dir $MODEL_DIR --train_epochs 41 --stop_threshold $QUALITY --batch_size 128 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.0002  2>&1 |tee lars-41epochs_eval_every_4_epochs_${RANDOM_SEED}.log
#mpirun -n 16  --map-by ppr:2:socket:pe=14 python imagenet_main.py $RANDOM_SEED --data_dir /admin/datasets/resnet50/TF_Imagenet_FullData/ --model_dir $MODEL_DIR --train_epochs 41 --stop_threshold $QUALITY --batch_size 128 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.0002  2>&1 |tee lars-41epochs_eval_every_4_epochs_${RANDOM_SEED}.log
#mpirun -n 16  --map-by ppr:2:socket:pe=14 python imagenet_main.py $RANDOM_SEED --data_dir /dev/shm/TF_Imagenet_FullData/ --model_dir $MODEL_DIR --train_epochs 41 --stop_threshold $QUALITY --batch_size 128 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.0002  2>&1 |tee lars-41epochs_eval_every_4_epochs_${RANDOM_SEED}.log
#mpirun -n 16  --map-by ppr:2:socket:pe=14 python imagenet_main.py $RANDOM_SEED --data_dir /scratch/TF_Imagenet_FullData/ --model_dir $MODEL_DIR --train_epochs 41 --stop_threshold $QUALITY --batch_size 128 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.0002  2>&1 |tee lars-41epochs_eval_every_4_epochs_${RANDOM_SEED}.log
#mpirun -n 16  --map-by ppr:2:socket:pe=14 python imagenet_main.py $RANDOM_SEED --data_dir /scratch/TF_Imagenet_FullData/ --model_dir $MODEL_DIR --train_epochs 41 --stop_threshold $QUALITY --batch_size 128 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.0002  2>&1 |tee lars-41epochs_eval_every_4_epochs_${RANDOM_SEED}.log
#mpirun -np 16 -ppn 8 -host ocpx05,ocpx07 -genv FI_PROVIDER=psm3 -genv PSM3_MULTIRAIL=1 -genv PSM3_ALLOW_ROUTERS=1  -genv I_MPI_PIN_DOMAIN=socket ./run_single_process.sh 2>&1 | tee lars-41epochs_eval_every_4epochs_${RANDOM_SEED}.log
#mpirun -np 32 -ppn 16 -host ocpx13,ocpx14 -genv PSM3_ALLOW_ROUTERS=1 -genv PSM3_RDMA=1 -genv I_MPI_PIN_DOMAIN=socket ./run_1S2P_IntelMPI.sh $RANDOM_SEED 2>&1 | tee lars-41epochs_eval_every_4epochs_${RANDOM_SEED}.log
#pdsh -w ocpx07,ocpx09 cacheclr3
#mpirun -n 8 -ppn 8 -host ocpx05  --map-by socket -genv FI_PROVIDER=psm3 -genv PSM3_ALLOW_ROUTERS=1 -genv PSM3_RDMA=0  python imagenet_main.py $RANDOM_SEED --data_dir /scratch/TF_Imagenet_FullData/ --model_dir $MODEL_DIR --train_epochs 37 --stop_threshold $QUALITY --batch_size 102 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.00005  2>&1 |tee lars-40epochs_eval_every_4_epochs_${RANDOM_SEED}.log
#mpirun -n 8 -ppn 8 -host ocpx05  --map-by socket -genv FI_PROVIDER=psm3 -genv PSM3_ALLOW_ROUTERS=1 -genv PSM3_RDMA=0  python imagenet_main.py $RANDOM_SEED --data_dir /scratch/TF_Imagenet_FullData/ --model_dir $MODEL_DIR --train_epochs 37 --stop_threshold $QUALITY --batch_size 408 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.00005  2>&1 |tee lars-40epochs_eval_every_4_epochs_${RANDOM_SEED}.log

pdsh -w ocpx04,ocpx06 cacheclr3
mpirun -n 16 -ppn 8 -host ocpx04,ocpx06 -genv PSM3_ALLOW_ROUTERS=1 -genv PSM3_RDMA=0 -genv I_MPI_PIN_DOMAIN=socket python imagenet_main.py $RANDOM_SEED --data_dir /scratch/TF_Imagenet_FullData/ --model_dir $MODEL_DIR --train_epochs 37 --stop_threshold $QUALITY --batch_size 204 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.00005  2>&1 |tee lars-40epochs_eval_every_4_epochs_${RANDOM_SEED}.log
