FROM gitlab-master.nvidia.com:5005/dl/dgx/cuda:11.3-devel-ubuntu20.04--master AS devel


RUN apt-get update -y && apt-get upgrade -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends vim gdb git wget tar python-dev python3-dev zlib1g-dev lsb-release ca-certificates clang-format numactl libnuma-dev libaio-dev aptitude
RUN aptitude -y install libboost-all-dev
RUN rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp http://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh && \
    bash /var/tmp/Miniconda3-4.7.12-Linux-x86_64.sh -b -p /opt/conda && \
    /opt/conda/bin/conda init && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    /opt/conda/bin/conda clean -afy && \
    rm -rf /var/tmp/Miniconda3-4.7.12-Linux-x86_64.sh
ENV CPATH=/opt/conda/include:$CPATH \
    LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/opt/conda/lib:$LIBRARY_PATH \
    PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda \
    NCCL_LAUNCH_MODE=PARALLEL

RUN conda update -n base -c defaults conda && \
    conda install -c conda-forge cmake=3.19.6 pip ucx libhwloc=2.4.0 openmpi=4.1.0 openmpi-mpicc=4.1.0 mpi4py=3.0.3 && \
    rm -rfv /opt/conda/include/nccl.h /opt/conda/lib/libnccl* /opt/conda/include/google /opt/conda/include/*cudnn* /opt/conda/lib/*cudnn*
ENV OMPI_MCA_plm_rsh_agent=sh \
    OMPI_MCA_opal_cuda_support=true

RUN pip3 install numpy pandas sklearn ortools jupyter tensorflow==2.4.0

RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch branch-0.19 https://github.com/rapidsai/rmm.git rmm && cd - && \
    cd /var/tmp/rmm && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DBUILD_TESTS=OFF && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    rm -rf /var/tmp/rmm

COPY . HugeCTR
RUN cd HugeCTR && \
    git submodule update --init --recursive && \
    mkdir build && cd build &&\
    cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" -DDISABLE_CUDF=ON -DDISABLE_EMBEDDING_PLUGIN=ON \
         -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
         -DVAL_MODE=OFF -DENABLE_MULTINODES=ON -DENABLE_MPI=ON -DNCCL_A2A=ON -DUCX_INCLUDE_DIR=/usr/local/ucx/include/ -DUCX_LIBRARIES=/usr/local/ucx/lib/ .. && \
    make -j$(nproc) &&\
    mkdir /usr/local/hugectr &&\
    make install &&\
    chmod +x /usr/local/hugectr/bin/* &&\
    rm -rf HugeCTR
ENV PATH=/usr/local/hugectr/bin:$PATH
WORKDIR /workspace/dlrm
COPY . .
