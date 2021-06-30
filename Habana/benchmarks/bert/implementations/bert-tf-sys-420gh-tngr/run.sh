#!/bin/bash

BASE_PATH=`dirname "$(readlink -e "${BASH_SOURCE[0]}")"`
TRAIN_SCRIPT=${BASE_PATH}/TensorFlow/nlp/bert/demo_bert

export INPUT_FILES=/root/tensorflow_datasets/MLPerf_BERT_Wiki
echo "INPUT_FILES=$INPUT_FILES"
export EVAL_FILES=/root/tensorflow_datasets/mlperf_bert_eval_dataset
echo "EVAL_FILES=$EVAL_FILES"
export OUTPUT_MODEL_DIR=/tmp/bert_pretrain
echo "OUTPUT_MODEL_DIR=$OUTPUT_MODEL_DIR"
export INITIAL_CHECKPOINT=/root/MLPerf_BERT_checkpoint
echo "INITIAL_CHECKPOINT=$INITIAL_CHECKPOINT"
export BERT_CONFIG_DIR=/root/bert_config
echo "BERT_CONFIG_DIR=$BERT_CONFIG_DIR"

NUM_WORKERS_PER_HLS=${NUM_WORKERS_PER_HLS:-8}
BATCH_SIZE_P2=14
STEPS_P2=6365

export HABANA_INITIAL_WORKSPACE_SIZE_MB=4600
echo "HABANA_INITIAL_WORKSPACE_SIZE_MB=$HABANA_INITIAL_WORKSPACE_SIZE_MB"
TRAIN_COMMAND="${TRAIN_SCRIPT} pretraining \
    -dist_eval \
    -v ${NUM_WORKERS_PER_HLS} \
    -d bf16 \
    -i ${STEPS_P2} \
    -b ${BATCH_SIZE_P2} \
    --bert-config-dir ${BERT_CONFIG_DIR} \
    --output-dir ${OUTPUT_MODEL_DIR}

"

mkdir -p ${OUTPUT_MODEL_DIR}

export TF_RECIPE_CACHE_PATH=${OUTPUT_MODEL_DIR}/graph_dump_recipes
echo "TF_RECIPE_CACHE_PATH=$TF_RECIPE_CACHE_PATH"

export PYTHONPATH=$BASE_PATH:${PYTHONPATH}
echo "PYTHONPATH=$PYTHONPATH"

git clone https://github.com/mlperf/logging.git mlperf-logging
python3 -m pip install -e mlperf-logging

echo $TRAIN_COMMAND
$TRAIN_COMMAND
