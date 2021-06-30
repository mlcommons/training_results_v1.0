#!/bin/bash

BASE_PATH=`dirname $(readlink -e ${BASH_SOURCE[0]})`
TRAIN_SCRIPT=${BASE_PATH}/TensorFlow/computer_vision/Resnets/resnet_keras/resnet_ctl_imagenet_main.py

DATA_DIR=/tf_records
MODEL_DIR=/tmp/resnet50
BATCH_SIZE=256
EPOCHS=40
EXPERIMENTAL_PRELOADING=1
DISPLAY_STEPS=125
STEPS_PER_LOOP=125
ENABLE_CHECKPOINT=false
ENABLE_TENSORBOARD=false
EPOCHS_BETWEEN_EVALS=4
EVAL_OFFSET_EPOCHS=2
REPORT_ACCURACY_METRICS=true
DIST_EVAL=true
TARGET_ACCURACY=0.759
LARS_WARMUP_EPOCHS=1
ENABLE_DEVICE_WARMUP=true
LARS_DECAY_EPOCHS=41

mkdir -p ${MODEL_DIR}

TRAIN_COMMAND="python3 ${TRAIN_SCRIPT}
    --model_dir=${MODEL_DIR}
    --data_dir=${DATA_DIR}
    --batch_size=${BATCH_SIZE}
    --distribution_strategy=off
    --num_gpus=0
    --data_format=channels_last
    --train_epochs=${EPOCHS}
    --experimental_preloading=${EXPERIMENTAL_PRELOADING}
    --log_steps=${DISPLAY_STEPS}
    --steps_per_loop=${STEPS_PER_LOOP}
    --enable_checkpoint_and_export=${ENABLE_CHECKPOINT}
    --enable_tensorboard=${ENABLE_TENSORBOARD}
    --epochs_between_evals=${EPOCHS_BETWEEN_EVALS}
    --base_learning_rate=9.5
    --warmup_epochs=${LARS_WARMUP_EPOCHS}
    --optimizer=LARS
    --lr_schedule=polynomial
    --label_smoothing=0.1
    --weight_decay=0.0001
    --single_l2_loss_op
    --use_horovod
    --data_loader_image_type=bf16
    --eval_offset_epochs=${EVAL_OFFSET_EPOCHS}
    --report_accuracy_metrics=${REPORT_ACCURACY_METRICS}
    --dist_eval=${DIST_EVAL}
    --target_accuracy=${TARGET_ACCURACY}
    --enable_device_warmup=${ENABLE_DEVICE_WARMUP}
    --lars_decay_epochs=${LARS_DECAY_EPOCHS}
"

NUM_WORKERS=${NUM_WORKERS:-8}
MPI_MAP_BY=socket
MPI_MAP_BY_PE=`lscpu | grep "^CPU(s):"| awk -v NUM=${NUM_WORKERS} '{print int($2/NUM/2)}'`

HVD_COMMAND="mpirun --allow-run-as-root
    -np $NUM_WORKERS
    --bind-to core --map-by $MPI_MAP_BY:PE=$MPI_MAP_BY_PE
    ${TRAIN_COMMAND}"


export TF_RECIPE_CACHE_PATH=/tmp/resnet50
mkdir -p ${TF_RECIPE_CACHE_PATH}

export PYTHONPATH=$BASE_PATH:${PYTHONPATH}
export HABANA_USE_STREAMS_FOR_HCL=true
export HBN_TF_REGISTER_DATASETOPS=1
export TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS=1
export TF_ENABLE_BF16_CONVERSION=1
export TF_PRELIMINARY_CLUSTER_SIZE_THRESHOLD=700
export TF_PRELIMINARY_CLUSTER_SIZE=20
export HCL_CONFIG_PATH=${HCL_CONFIG_PATH:-${BASE_PATH}/hls1.json}

if [ ! -f /usr/lib/habanalabs/dynpatch_prf_remote_call.so ]; then
  ln -s /usr/local/lib/python`python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'`/dist-packages/habana_frameworks/tensorflow/tf`python3 -c "import tensorflow as tf; print(tf.__version__.replace('.', '_'))"`/lib/habanalabs/dynpatch_prf_remote_call.so /usr/lib/habanalabs/dynpatch_prf_remote_call.so
fi
echo $HVD_COMMAND
LD_PRELOAD=/usr/lib/habanalabs/dynpatch_prf_remote_call.so $HVD_COMMAND
