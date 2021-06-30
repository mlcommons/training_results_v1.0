# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


# NOTE: This implementation of DLRM uses the HugeCTR framework.  The benchmark
# leverages the NGC PyTorch image solely for the GPU-enabled build environment
# it provides, not for the PyTorch framework itself (HugeCTR as a framework is
# independent of PyTorch).  Other NGC DL Frameworks images such as
# nvcr.io/nvidia/tensorflow:21.05-tf1-py3 or nvcr.io/nvidia/mxnet:21.05-py3
# could have been used similarly as a base for this benchmark.

FROM nvcr.io/nvidia/pytorch:21.05-py3 AS devel

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && \
    apt-get install -y --no-install-recommends \
    vim gdb git wget tar python-dev python3-dev \
    zlib1g-dev lsb-release ca-certificates clang-format aptitude numactl libnuma-dev libaio-dev curl libtool

RUN DEBIAN_FRONTEND=noninteractive aptitude install -y libboost-all-dev && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*

ENV CPATH=/opt/conda/include:$CPATH \
    LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/opt/conda/lib:$LIBRARY_PATH \
    PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda \
    NCCL_LAUNCH_MODE=PARALLEL

RUN conda update -n base -c defaults conda && \
    conda install -c conda-forge cmake=3.19.6 pip ucx openmpi=4.1.0 openmpi-mpicc=4.1.0 mpi4py=3.0.3 && \
    rm -rfv /opt/conda/include/nccl.h /opt/conda/lib/libnccl* /opt/conda/include/google /opt/conda/include/*cudnn* /opt/conda/lib/*cudnn*

RUN rm -rf /opt/conda/lib/libtinfo.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libtinfo.so.6.2 /opt/conda/lib/libtinfo.so.6

ENV OMPI_MCA_plm_rsh_agent=sh
RUN pip3 install numpy pandas sklearn ortools jupyter tensorflow==2.4.0

RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch branch-0.19 https://github.com/rapidsai/rmm.git rmm && cd - && \
    cd /var/tmp/rmm && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DBUILD_TESTS=OFF && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/rmm

ENV SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0 \
    SHARP_COLL_LOCK_ON_COMM_INIT=1 \
    SHARP_COLL_LOG_LEVEL=3

## Build cuda-aware hwloc
RUN wget https://download.open-mpi.org/release/hwloc/v2.4/hwloc-2.4.1.tar.gz && \
    tar -xvzf hwloc-2.4.1.tar.gz && cd hwloc-2.4.1 && \
    ./configure CPPFLAGS="-I/usr/local/cuda-11.3/include/ -L/usr/local/cuda-11.3/lib64/" LDFLAGS="-L/usr/local/cuda-11.3/lib64" --enable-cuda && \
    make && make install
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-11.3/lib64/:$LD_LIBRARY_PATH

RUN ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1.11.32.1 /usr/lib/x86_64-linux-gnu/libibverbs.so

RUN git clone https://github.com/NVIDIA/HugeCTR HugeCTR
RUN cd HugeCTR && \
    git checkout v3.1_beta && \
    git submodule update --init --recursive && \
    mkdir build && cd build &&\
    cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" -DDISABLE_CUDF=ON -DDISABLE_EMBEDDING_PLUGIN=ON \
         -DVAL_MODE=OFF -DENABLE_MULTINODES=ON -DENABLE_MPI=ON -DNCCL_A2A=ON -DUCX_INCLUDE_DIR=/usr/local/ucx/include/ -DUCX_LIBRARIES=/usr/local/ucx/lib/ .. && \
    make -j$(nproc) &&\
    mkdir /usr/local/hugectr &&\
    make install &&\
    chmod +x /usr/local/hugectr/bin/* &&\
    rm -rf HugeCTR

RUN rm /usr/lib/x86_64-linux-gnu/libibverbs.so

ENV PATH=/usr/local/hugectr/bin:$PATH
ENV NCCL_COLLNET_ENABLE=0

WORKDIR /workspace/dlrm

RUN pip3 install --upgrade --no-cache-dir https://github.com/mlcommons/logging/archive/1.0.0-rc2.zip

COPY . .
