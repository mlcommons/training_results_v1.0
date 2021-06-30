#!/bin/bash

# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -e

# Only rank print 
[ "${SLURM_LOCALID-}" -ne 0 ] && set +x


: "${AMP_LVL:=1}"
: "${DALIDEVICE:=gpu}"
: "${MODELCONFIG:=configs/baseline_v3-1023sp.yaml}"
: "${BATCHSIZE:=${BATCHSIZE}}"
: "${EVAL_BATCHSIZE:=${BATCHSIZE}}"
: "${EPOCH:=1}"
: "${SEED:=2021}"
: "${LR:=0.004}"
: "${WARMUP:=6}"
: "${GRAD_ACCUMULATION_STEPS:=1}"
: "${VAL_FREQUENCY:=1}"
: "${HOLD_EPOCHS:=40}"
: "${EMA:=0.999}"
: "${LR_DECAY_POWER:=0.935}"
: "${WEIGHTS_INIT_SCALE:=0.5}"
: "${DATASET_DIR:="/datasets/LibriSpeech"}"
: "${DALI_ONLY:=false}"
: "${MEMBIND:=true}"
: "${VECTORIZED_SA:=false}"
: "${VECTORIZED_SAMPLER=false}"
: "${LOG_FREQUENCY=1}"
: "${BETA1:=0.9}"
: "${BETA2:=0.999}"
: "${MAX_TRAIN_DURATION:=16.7}"

: ${TRAIN_MANIFESTS:="/metadata/librispeech-train-clean-100-wav-tokenized.pkl \
                      /metadata/librispeech-train-clean-360-wav-tokenized.pkl \
                      /metadata/librispeech-train-other-500-wav-tokenized.pkl"}
: ${VAL_MANIFESTS:="/metadata/librispeech-dev-clean-wav-tokenized.pkl"}
: ${OUTPUT_DIR:="/results"}
: ${TARGET:=0.058}

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
echo "running benchmark"

export DATASET_DIR
export TORCH_HOME="$(pwd)/torch-model-cache"

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    CMD=( './bind.sh' '--' 'python' '-u' )
  else
    CMD=( 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker; need to launch tasks with Pytorch's distributed launch
  # TODO: use bind.sh instead of bind_launch.py
  #       torch.distributed.launch only accepts Python programs (not bash scripts) to exec
  CMD=( 'python' '-u' '-m' 'bind_launch' "--nsockets_per_node=${DGXNSOCKET}" \
    "--ncores_per_socket=${DGXSOCKETCORES}" "--nproc_per_node=${DGXNGPU}" )
  [ "$MEMBIND" = false ] &&  CMD+=( "--no_membind" )
fi
echo "${CMD[@]}"

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  # TODO(ahmadki): track the apiLog.sh bug and remove the workaround
  # there is a bug in apiLog.sh preventing it from collecting
  # NCCL logs, the workaround is to log a single rank only
  # LOCAL_RANK is set with an enroot hook for Pytorch containers
  # SLURM_LOCALID is set by Slurm
  # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi


mkdir -p /results
# run training
ARGS="train.py \
  --batch_size=$BATCHSIZE \
  --beta1=${BETA1} \
  --beta2=${BETA2} \
  --max_duration=${MAX_TRAIN_DURATION} \
  --val_batch_size=$EVAL_BATCHSIZE \
  --target=${TARGET} \
  --lr=${LR} \
  --min_lr=1e-5 \
  --lr_exp_gamma=${LR_DECAY_POWER} \
  --epochs=$EPOCH \
  --warmup_epochs=$WARMUP \
  --hold_epochs=$HOLD_EPOCHS \
  --epochs_this_job=0 \
  --ema=$EMA \
  --output_dir ${OUTPUT_DIR} \
  --model_config=$MODELCONFIG \
  --seed $SEED \
  --dataset_dir=${DATASET_DIR} \
  --cudnn_benchmark \
  --dali_device $DALIDEVICE \
  --weight_decay=1e-3 \
  --log_frequency=${LOG_FREQUENCY} \
  --val_frequency=$VAL_FREQUENCY \
  --grad_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
  --prediction_frequency=1000000 \
  --weights_init_scale=${WEIGHTS_INIT_SCALE} \
  --val_manifests=${VAL_MANIFESTS} \
  --train_manifests ${TRAIN_MANIFESTS}"

if [ $BUCKET -ne 0 ]; then
  ARGS="${ARGS} --num_buckets=${BUCKET}"
fi
if [ $MAX_SYMBOL -gt 0 ]; then
  ARGS="${ARGS} --max_symbol_per_sample=${MAX_SYMBOL}"
fi
if [ "$APEX_LOSS" = "fp16" ] || [ "$APEX_LOSS" = "fp32" ]; then
  ARGS="${ARGS} --apex_transducer_loss=${APEX_LOSS}"
fi
if [ "$FUSE_RELU_DROPOUT" = true ]; then
  ARGS="${ARGS} --fuse_relu_dropout"
fi
if [ "$MULTI_TENSOR_EMA" = true ]; then
  ARGS="${ARGS} --multi_tensor_ema"
fi
if [ "$BATCH_EVAL_MODE" = "no_cg" ] || [ "$BATCH_EVAL_MODE" = "cg" ] || [ "$BATCH_EVAL_MODE" = "cg_unroll_pipeline" ]; then
  ARGS="${ARGS} --batch_eval_mode ${BATCH_EVAL_MODE}"
fi
if [ "$DIST_LAMB" = true ]; then
  ARGS="${ARGS} --dist_lamb"
fi
if [ "$APEX_JOINT" = "pack" ] || [ "$APEX_JOINT" = "not_pack" ]; then
  ARGS="${ARGS} --apex_transducer_joint=${APEX_JOINT}"
fi
if [ "$BUFFER_PREALLOC" = true ]; then
  ARGS="${ARGS} --buffer_pre_alloc"
fi
if [ "$EMA_UPDATE_TYPE" = "fp16" ] || [ "$EMA_UPDATE_TYPE" = "fp32" ]; then
  ARGS="${ARGS} --ema_update_type=${EMA_UPDATE_TYPE}"
fi

[ ! -z "${AMP_LVL}" ] && ARGS+=" --amp_level ${AMP_LVL}"
[ ! -z "${DATA_CPU_THREADS}" ] && ARGS+=" --data_cpu_threads ${DATA_CPU_THREADS}"
[ ! -z "${BATCH_SPLIT_FACTOR}" ] && ARGS+=" --batch_split_factor ${BATCH_SPLIT_FACTOR}"
[ ! -z "${NUM_CG}" ] && ARGS+=" --num_cg ${NUM_CG}"
[ ! -z "${MIN_SEQ_SPLIT_LEN}" ] && ARGS+=" --min_seq_split_len ${MIN_SEQ_SPLIT_LEN}"
[ ! -z "${DWU_GROUP_SIZE}" ] && ARGS+=" --dwu_group_size ${DWU_GROUP_SIZE}"
[ "${VECTORIZED_SA}" = true ] && ARGS+=" --vectorized_sa"
[ "${MULTILAYER_LSTM}" = true ] && ARGS+=" --multilayer_lstm"
[ "${IN_MEM_FILE_LIST}" = true ] && ARGS+=" --in_mem_file_list"
[ "${ENABLE_PREFETCH}" = true ] && ARGS+=" --enable_prefetch"
[ "${TOKENIZED_TRANSCRIPT}" = true ] && ARGS+=" --tokenized_transcript"
[ "${VECTORIZED_SAMPLER}" = true ] && ARGS+=" --vectorized_sampler"
[ "${SEQ_LEN_STATS}" = true ] && ARGS+=" --enable_seq_len_stats"
[ "${DIST_SAMPLER}" = true ] && ARGS+=" --dist_sampler"
[ "${APEX_MLP}" = true ] && ARGS+=" --apex_mlp"
[ "${PRE_SORT_FOR_SEQ_SPLIT}" = true ] && ARGS+=" --pre_sort_for_seq_split"
[ "${JIT_TENSOR_FORMATION}" = true ] && ARGS+=" --jit_tensor_formation"
[ "${DALI_DONT_USE_MMAP}" = true ] && ARGS+=" --dali_dont_use_mmap"

${LOGGER:-} "${CMD[@]}" $ARGS
ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="RNN_SPEECH_RECOGNITION"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

