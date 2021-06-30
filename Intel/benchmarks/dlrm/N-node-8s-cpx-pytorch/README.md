# DLRM MLPerf Training Intel Submission

## Before you start
# ResNet50-v1.5 MLPerf Training Intel Submission

## Before you start
Please note that all the v1.0 8380H results are the same as Intel v0.7 results. You can find our v0.7 code from the link [here](https://github.com/mlcommons/training_results_v0.7/tree/master/Intel/benchmarks/dlrm).
This readme has instructions for 8376H based v1.0 results.

## HW and SW requirements
### 1. HW requirements
| HW | configuration |
| -: | :- |
| CPU | CPX-8376 @ 8 sockets/Node |
| DDR | 192G/socket @ 3200 MT/s |
| SSD | 1 SSD/Node @ >= 1T |

### 2. SW requirements
| SW |configuration  |
|--|--|
| GCC | GCC 8.3  |

## Steps to run DLRM

### 1. Install anaconda 3.0
```
   wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
   chmod +x anaconda3.sh
   ./anaconda3.sh -b -p ~/anaconda3
   ~/anaconda3/bin/conda create -n dlrm python=3.7

   export PATH=~/anaconda3/bin:$PATH
   source ./anaconda3/bin/activate dlrm
```
### 2. Install dependency packages
```
   pip install sklearn onnx tqdm lark-parser tensorboardX psutil
   pip install -e git+https://github.com/mlperf/logging@1.0.0-rc4#egg=mlperf-logging

   conda config --append channels intel
   conda install ninja pyyaml setuptools cmake cffi typing
   conda install intel-openmp mkl mkl-include numpy -c intel --no-update-deps
   conda install -c conda-forge gperftools
```
### 3. Clone source code and Install
(1) Install PyTorch and Intel Extension for PyTorch
```
   # clone PyTorch
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch && git checkout tags/v1.5.0-rc3 -b v1.5-rc3
   git submodule sync && git submodule update --init --recursive

   # clone Intel Extension for PyTorch
   git clone https://github.com/intel/intel-extension-for-pytorch.git
   cd intel-extension-for-pytorch && git checkout tags/v0.2 -b v0.2
   git submodule update --init --recursive

   # install PyTorch
   cd {path/to/pytorch}
   cp {path/to/intel-pytorch-extension}/torch_patches/0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch .
   patch -p1 < 0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch
   python setup.py install

   # install Intel Extension for PyTorch
   cd {path/to/intel-pytorch-extension}
   #enable lamb fused for bf16
   wget https://patch-diff.githubusercontent.com/raw/intel/intel-extension-for-pytorch/pull/161.diff 
   git apply 161.diff
   python setup.py install
```
(2) Install torch-ccl
```
   git clone https://github.com/intel/torch-ccl.git
   cd torch-ccl && git checkout remotes/origin/ccl_torch1.5 -b ccl_torch1.5
   git submodule sync && git submodule update --init --recursive
   cd third_party/oneCCL
   git apply ../../patches/Fix_the_* && cd ../../   
   export PATH=/opt/crtdc/binutils/2.36/bin/:$PATH
   export CCL_BF16=avx512bf
   export LD_LIBRARY_PATH=GCC-10.2.0/lib:GCC-10.2.0/lib64:${LD_LIBRARY_PATH}
   CMAKE_C_COMPILER=GCC-10.2.0/bin/gcc CMAKE_CXX_COMPILER=GCC-10.2.0/bin/g++  python setup.py install
```
(3) Install DLRM
```
   git clone https://github.com/facebookresearch/dlrm.git
   cd dlrm && git checkout mlperf
   wget https://raw.githubusercontent.com/intel/intel-extension-for-pytorch/653f6464c651411e9e694265429e4b87402d434e/torch_patches/models/dlrm_mlperf_v1.0_training.diff
   git apply dlrm_mlperf_v1.0_training.diff
```
## 4. Run command
```
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
```

## 5. Log patch 
We added the following line at the begining of each result_*.txt to compliance with 1.0 log format:
``` 
[0] :::MLLOG {"namespace": "", "time_ms": 0, "event_type": "POINT_IN_TIME", "key": "gradient_accumulation_steps", "value": 1, "metadata": {"file": "dlrm_s_pytorch.py", "lineno": 0}}

```
