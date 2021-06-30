#!/usr/bin/env bash

USER=`whoami`
: "${NRUNS:=1}"
: "${NEXP:=1}"
: "${CONFIG:=config_DGXA100_multi_08x08x64.sh}"
: "${BASE_LOG_DIR:=logs}"
: "${JOB_NAME:=job}"
: "${CONTAINER_IMAGE:=gitlab-master.nvidia.com/akiswani/images/mlperf_mxnet_ssd:master-py3-devel}"

while [ "$1" != "" ]; do
    case $1 in
        -n | --num-runs )       shift
                                NRUNS=$1
                                ;;
        -r | --num-exp  )       shift
                                NEXP=$1
                                ;;
        -c | --config )         shift
                                CONFIG=$1
                                ;;
        -l | --log-dir )        shift
                                BASE_LOG_DIR=$1
                                ;;
        -d | --container )      shift
                                CONTAINER_IMAGE=$1
                                ;;
    esac
    shift
done

source ${CONFIG}
CONFIG_NAME=`basename ${CONFIG} .sh`

GIT_SHA=`git rev-parse HEAD`
GIT_SHA_SHORT=${GIT_SHA::5}
SUFFIX=`date +%s`
JOB_NAME=${CONFIG_NAME}_${GIT_SHA_SHORT}_${SUFFIX}

BASE_LOG_DIR="${BASE_LOG_DIR}/${CONFIG_NAME}"
mkdir -p ${BASE_LOG_DIR}

# Run experiments
for i in $(seq 1 "${NRUNS}"); do
    echo "[${i}/${NRUNS}] Running config ${CONFIG}"
    # Run experiment
    sbatch \
        --account=mlperf \
        --partition=luna \
        --job-name="mlperf-ssd:${JOB_NAME}_${i}" \
        --nodes="${DGXNNODES}" \
        --ntasks-per-node="${DGXNGPU}" \
        --time="${WALLTIME}" \
        --output="${BASE_LOG_DIR}/%A_${GIT_SHA_SHORT}.out" \
        ./scripts/launchers/batch.sh "${CONTAINER_IMAGE}"
    sleep 1
done
echo ""

