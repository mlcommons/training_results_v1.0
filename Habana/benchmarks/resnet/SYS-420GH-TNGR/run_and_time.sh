#!/bin/bash

BASE_PATH=`dirname $(readlink -e ${BASH_SOURCE[0]})`
IMPL_PATH=${BASE_PATH}/../implementations/resnet-tf-sys-420gh-tngr/
RESULT_DIR=${RESULT_DIR:-${BASE_PATH}/../../../results/SYS-420GH-TNGR/resnet}

NUM_WORKERS=8
DATA_DIR=${DATA_DIR:-/data/tf_records/}
DOCKER_IMAGE=${DOCKER_IMAGE:-vault.habana.ai/gaudi-docker-mlperf/ver1.0/tf/hls-1:0.14-mlperf}
REPEAT_BASE=${REPEAT_BASE:-0}
REPEAT=${REPEAT:-1}
TIMESTAMP_LOGNAME=${TIMESTAMP_LOGNAME:-0}

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

TRAIN_CMD=/work/run.sh
DOCKER_CMD="sudo docker run --rm
    ${DOCKER_PARAMS}
    -v ${DATA_DIR}:/tf_records
    -v ${IMPL_PATH}:/work
    -e NUM_WORKERS=${NUM_WORKERS}
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none
    --cap-add=sys_nice
    --net=host
    --ulimit memlock=-1:-1
    $DOCKER_IMAGE
    ${TRAIN_CMD}"

if [ ! -f $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi

if [ ! -d $RESULT_DIR ]; then
  echo "$RESULT_DIR is not a directory. Set RESULT_DIR to a different path"
  exit 1
fi

for i in `seq $REPEAT`;
do
  if [ $TIMESTAMP_LOGNAME -eq 1 ]; then
    LOG_SUFFIX=`date +%s`.txt
  else
    LOG_SUFFIX=$((REPEAT_BASE+i-1)).txt
  fi
  LOG_FILE=log_${LOG_SUFFIX}
  RESULT_FILE=${RESULT_DIR}/result_${LOG_SUFFIX}

  echo "Flush host cache..."
  sync
  sudo bash -c "echo 3 > /proc/sys/vm/drop_caches"

  echo "Start training..."
  ${DOCKER_CMD} > $LOG_FILE 2>&1
  grep "^:::MLLOG.*worker0" $LOG_FILE > $RESULT_FILE
  echo "Result created in ${RESULT_FILE}"

  sleep 1
done
