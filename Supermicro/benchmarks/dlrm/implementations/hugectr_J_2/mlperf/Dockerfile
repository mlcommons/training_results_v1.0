# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
# nvcr.io/nvidia/tensorflow:20.06-tf1-py3 or nvcr.io/nvidia/mxnet:20.06-py3
# could have been used similarly as a base for this benchmark.

ARG FROM_IMAGE_NAME=gitlab-master.nvidia.com:5005/dl/dgx/cuda:11.1-devel-ubuntu20.04--20.12

FROM ${FROM_IMAGE_NAME}
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        lsb-release \
        libboost-all-dev \
        libaio-dev \
        vim \
        wget \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# CMake version 3.14.3
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.sh && \
    mkdir -p /usr/local && \
    /bin/sh /var/tmp/cmake-3.17.0-Linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm -rf /var/tmp/cmake-3.17.0-Linux-x86_64.sh
ENV PATH=/usr/local/bin:$PATH

# pip
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        python3-wheel && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip && \
    pip3 install numpy pandas sklearn ortools tensorflow

RUN mkdir -p /opt/conda
ENV CONDA_PREFIX=/opt/conda

# Build apache arrow from source
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/compat/lib/:$LD_LIBRARY_PATH
RUN mkdir -p /var/tmp && cd /var/tmp && git clone -b apache-arrow-2.0.0 https://github.com/apache/arrow.git arrow && \ 
   cd arrow && git submodule init && git submodule update && \
   cd cpp && \
   mkdir -p build && cd build && \
   cmake -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs -DCMAKE_INSTALL_LIBDIR=lib  -DARROW_PARQUET=ON -DARROW_CUDA=ON -DARROW_DEPENDENCY_SOURCE=AUTO .. && \
   make -j$(nproc) && \
   make -j$(nproc) install && \
   rm -rf /var/tmp/arrow
 
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch branch-0.16 https://github.com/rapidsai/rmm.git rmm && cd - && \
    cd /var/tmp/rmm && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/rmm

# CUDF 
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch branch-0.16 https://github.com/rapidsai/cudf.git cudf && cd - && \
    git clone --depth=1 --branch main https://github.com/dmlc/dlpack.git /var/tmp/dlpack && \
    cd /var/tmp/cudf/cpp && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGPU_ARCHS="70;80" \
             -DARROW_INCLUDE_DIR=/usr/include/arrow/ -DDLPACK_INCLUDE=/var/tmp/dlpack/include \
             -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDISABLE_DEPRECATION_WARNING=ON && \ 
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/dlpack /var/tmp/cudf

# Install Python dependencies
RUN pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip \
 && pip install --no-cache-dir numpy pandas sklearn ortools

# Install HugeCTR framework
WORKDIR /opt
RUN ln -s /usr/local/cuda/compat/lib.real/libcuda.so.1 ${CONDA_PREFIX}/lib/libcuda.so.1

COPY . HugeCTR

RUN cd HugeCTR && \
    git submodule update --init --recursive && \
    mkdir build && cd build &&\
    cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" \
         -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
         -DVAL_MODE=OFF -DENABLE_MULTINODES=ON -DENABLE_MPI=ON -DNCCL_A2A=ON -DUCX_INCLUDE_DIR=/usr/local/ucx/include/ -DUCX_LIBRARIES=/usr/local/ucx/lib/ .. && \
    make -j$(nproc) &&\
    mkdir /usr/local/hugectr &&\
    make install &&\
    chmod +x /usr/local/hugectr/bin/* &&\
    rm -rf HugeCTR

ENV PATH=/usr/local/hugectr/bin:$PATH

# Install DLRM benchmark code
WORKDIR /workspace/dlrm
COPY . .
