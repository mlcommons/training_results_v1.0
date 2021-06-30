#!/bin/bash

BASE_PATH=`dirname $(readlink -e ${BASH_SOURCE[0]})`
IMPL_PATH=${BASE_PATH}/../implementations/bert-tf-sys-420gh-tngr/

NUM_WORKERS_PER_HLS=8
## Idealy the following data/output directories are locally on NVME/SSD drive for best performance.
# Fill with appropriate path to initial checkpoint
INITIAL_CHECKPOINT=${INITIAL_CHECKPOINT:-/scratch/MLPerf_BERT_checkpoint/}
# Fill with appropriate path to training dataset
DATA_DIR=${DATA_DIR:-/scratch/Bert-Large_pretrainingdatasetWiki/}
# Fill with appropriate path to eval dataset
EVAL_DIR=${EVAL_DIR:-/scratch/mlperf_bert_eval_dataset}
# Fill with appropriate path to docker container image
DOCKER_IMAGE=${DOCKER_IMAGE:-"vault.habana.ai/gaudi-docker-mlperf/ver1.0/tf/hls-1:0.14-mlperf"}
# Fill with appropriate path to where output checkpoint/logs will be placed
OUTPUT_MODEL_DIR=${OUTPUT_MODEL_DIR:-${BASE_PATH}/../../../results/SYS-420GH-TNGR/bert/}
# Fill with appropriate path to bert_config.json
BERT_CONFIG_DIR=${BERT_CONFIG_DIR:-/scratch/wwm_uncased_L-24_H-1024_A-16/}


if [[ ! -d "${BERT_CONFIG_DIR}" ]]; then
  BERT_CONFIG_MOUNT_STRING=""
else
  BERT_CONFIG_MOUNT_STRING="-v ${BERT_CONFIG_DIR}:/root/bert_config"
fi

sudo docker info|grep "Runtimes: habana" > /dev/null 2>&1
HAS_HABANA_CONTAINER_RUNTIME=$([ ${PIPESTATUS[1]} == 0 ] && (echo 1) || (echo 0))

if [ ${HAS_HABANA_CONTAINER_RUNTIME} == 1 ]; then
DOCKER_PARAMS=" --runtime=habana -e HABANA_VISIBLE_DEVICES=all "
else
DOCKER_PARAMS="
  --device=/dev/hl_controlD0:/dev/hl_controlD0
  --device=/dev/hl_controlD1:/dev/hl_controlD1
  --device=/dev/hl_controlD2:/dev/hl_controlD2
  --device=/dev/hl_controlD3:/dev/hl_controlD3
  --device=/dev/hl_controlD4:/dev/hl_controlD4
  --device=/dev/hl_controlD5:/dev/hl_controlD5
  --device=/dev/hl_controlD6:/dev/hl_controlD6
  --device=/dev/hl_controlD7:/dev/hl_controlD7
  --device=/dev/hl0:/dev/hl0
  --device=/dev/hl1:/dev/hl1
  --device=/dev/hl2:/dev/hl2
  --device=/dev/hl3:/dev/hl3
  --device=/dev/hl4:/dev/hl4
  --device=/dev/hl5:/dev/hl5
  --device=/dev/hl6:/dev/hl6
  --device=/dev/hl7:/dev/hl7
  "
fi

OUTPUT_MODEL_DIR_DOCKER=/tmp/bert_pretrain
TRAIN_CMD="/work/run.sh 2>&1 | tee ${OUTPUT_MODEL_DIR_DOCKER}/bert_mlperf.log"

echo "Flush host cache..."
sync
sudo bash -c "echo 3 > /proc/sys/vm/drop_caches"
sudo docker run --rm \
    ${DOCKER_PARAMS} \
    -v ${INITIAL_CHECKPOINT}:/root/MLPerf_BERT_checkpoint \
    -v ${DATA_DIR}:/root/tensorflow_datasets/MLPerf_BERT_Wiki \
    -v ${EVAL_DIR}:/root/tensorflow_datasets/mlperf_bert_eval_dataset \
    -v ${IMPL_PATH}:/work \
    ${BERT_CONFIG_MOUNT_STRING} \
    -v ${OUTPUT_MODEL_DIR}:${OUTPUT_MODEL_DIR_DOCKER} \
    -e NUM_WORKERS_PER_HLS=${NUM_WORKERS_PER_HLS} \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice \
    -v /sys/kernel/debug:/sys/kernel/debug \
    --net=host \
    --ulimit memlock=-1:-1 \
    $DOCKER_IMAGE \
    bash -c "${TRAIN_CMD}"
