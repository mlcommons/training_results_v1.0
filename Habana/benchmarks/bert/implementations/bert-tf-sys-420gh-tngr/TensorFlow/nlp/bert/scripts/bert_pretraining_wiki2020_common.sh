###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

export TF_CPP_MIN_LOG_LEVEL=6
export TF_CPP_MIN_VLOG_LEVEL=0

echo "EVAL_ONLY=$EVAL_ONLY"
echo "USE_DIS_EVAL=$USE_DIS_EVAL"
if [[ $EVAL_ONLY -eq 1 ]]; then
    eval_only=true
else
    eval_only=false
fi
if [[ $USE_DIS_EVAL -eq 1 ]]; then
    is_dist_eval_enabled=true
else
    is_dist_eval_enabled=false
fi

run_per_ip setup_libjemalloc

if [ "$USE_HOROVOD" == "true" ]; then
    export HOROVOD_STALL_CHECK_DISABLE=1
    echo HOROVOD_STALL_CHECK_DISABLE=$HOROVOD_STALL_CHECK_DISABLE
    # HCL Streams:ON by default
    export HABANA_USE_STREAMS_FOR_HCL=${HABANA_USE_STREAMS_FOR_HCL:-true}
    echo HABANA_USE_STREAMS_FOR_HCL=${HABANA_USE_STREAMS_FOR_HCL}

    # ART:ON by default
    export HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=${HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE:-false}
    echo HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=${HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE}

    # SAO:ON by default
    export TF_DISABLE_SCOPED_ALLOCATOR=${TF_DISABLE_SCOPED_ALLOCATOR:-false}
    echo TF_DISABLE_SCOPED_ALLOCATOR=$TF_DISABLE_SCOPED_ALLOCATOR

    export FORCE_WEIGHT_SYNC=${FORCE_WEIGHT_SYNC:-false}
    echo FORCE_WEIGHT_SYNC=$FORCE_WEIGHT_SYNC
fi

if [[ "$*" == --no_steps_accumulation ]]
then
    num_acc_steps_phase2=1
    learning_rate_phase2=5.0e-4
else
    if [ "$FAST_PERF_ONLY" == "1" ]; then
        local global_batch_size2=200
    else
        local global_batch_size2=448
    fi
    echo global_batch_sizes = $global_batch_size1, $global_batch_size2
    num_acc_steps_phase2=$(expr $global_batch_size2 \/ $NUM_WORKERS_TOTAL \/ $P2_BATCH)
    # Default value of learning rate argument (which is then scaled in run_pretraining.py
    # with the formula: effective_learning_rate = learning_rate * number_of_workers) for:
    # - 2nd phase with global batch size = 448 and 8 workers is 5.0e-5.
    # According to global_batch_size/learning_rate = const, to compute learning rate of
    # number of workers and global batch size, we first multiply default value by
    # (8 / $NUM_WORKERS_TOTAL) and then by ($global_batch_size / 448) for second phase.
    learning_rate_phase2=$(echo "0.00005  * ( 8 / $NUM_WORKERS_TOTAL ) * ( $global_batch_size2 / 448 )" | bc -l)
    echo "learning_rate_phase2 for MLPerf BERT-Large is $learning_rate_phase2"
    echo "global_batch_size2=$global_batch_size2"
    echo "num_acc_steps_phase2=$num_acc_steps_phase2"
    echo "NUM_WORKERS_TOTAL =$NUM_WORKERS_TOTAL "
fi


export RESULTS_DIR=${OUTPUT_DIR:-$HOME/bert_pretrain/tmp/}

echo "inside bert_pretraining_wiki2020_common.sh, and to source pretraining/scripts/run_pretraining_lamb.sh "
source ${SCRIPT_DIR}/scripts/run_pretraining_lamb.sh \
    $P2_BATCH \
    125 \
    $learning_rate_phase2 \
    fp32 \
    false \
    $NUM_WORKERS_PER_HLS \
    $P2_WARMUP \
    $P2_STEPS \
    335 \
    $num_acc_steps_phase2 \
    large \
    $is_dist_eval_enabled \
    $eval_only

