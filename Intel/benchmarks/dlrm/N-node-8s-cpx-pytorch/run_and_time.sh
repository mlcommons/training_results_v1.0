#!/bin/bash

set -x
source ~/anaconda3/bin/activate dlrm
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
DATA=mlperf_dataset/
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
export CCL_WORKER_COUNT=4
export OMP_NUM_THREADS=24
export CCL_WORKER_AFFINITY="0,1,2,3,28,29,30,31,56,57,58,59,84,85,86,87,112,113,114,115,140,141,142,143,168,169,170,171,196,197,198,199"
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=5117411328
export MASTER_ADDR=cpx01
BATCH_SIZE=262144
unset FI_PROVIDER_PATH
export LD_LIBRARY_PATH=/usr/lib64/libfabric:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=GCC-10.2.0/lib:GCC-10.2.0/lib64:${LD_LIBRARY_PATH}
seed_num=$(date +%s)
#need to enable ssh passwordless login
host_list="cpx01,cpx02,cpx03,cpx04,cpx05,cpx06,cpx07,cpx08"
NP=64
PPN=8
mpirun -np 8 -ppn 1 -hosts  $host_list  /usr/local/bin/transparent_hugepage_enabled_never
mpirun -np 8 -ppn 1 -hosts  $host_list  /usr/local/bin/transparent_hugepage_defrag_never
mpirun -np 8 -ppn 1 -hosts  $host_list  /usr/local/bin/transparent_hugepage_enabled_always
mpirun -np 8 -ppn 1 -hosts  $host_list  /usr/local/bin/transparent_hugepage_defrag_always
mpirun -np 8 -ppn 1 -hosts  $host_list pgrep -a python
mpirun -np 8 -ppn 1 -hosts  $host_list  pkill python
mpirun -np 8 -ppn 1 -hosts  $host_list  pkill python
mpirun -np 8 -ppn 1 -hosts  $host_list  cacheclr3
sleep 5
CCL_WORKER_COUNT=4 PSM3_RDMA=1 CCL_ATL_TRANSPORT=ofi FI_PROVIDER=psm3 PSM3_NIC=any PSM3_ALLOW_ROUTERS=1 mpiexec -l  -genv I_MPI_PIN_DOMAIN=[0xffffff0,0xffffff00000000,0xffffff000000000000000,0xffffff0000000000000000000000,0xffffff00000000000000000000000000000,0xffffff000000000000000000000000000000000000,0xffffff0000000000000000000000000000000000000000000,0xffffff00000000000000000000000000000000000000000000000000,] -n $NP -PPN 8 -hosts $host_list python -u dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=$DATA/day --processed-data-file=$DATA/terabyte_processed.npz --loss-function=bce --round-targets=True --num-workers=8 --test-num-workers=8 --use-ipex --dist-backend=ccl --learning-rate=16 --mini-batch-size=$BATCH_SIZE --print-freq=64 --print-time --test-freq=800 --test-mini-batch-size=524288 --lr-num-warmup-steps=5600 --lr-decay-start-step=6900 --lr-num-decay-steps=11100  --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --numpy-rand-seed=$seed_num $dlrm_extra_option  --bf16 --lamblr=30  --optimizer=1  
set +x

