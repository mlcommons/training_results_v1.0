ARG FROM_IMAGE_NAME=nvcr.io/nvidian/mxnet:21.05-py3
FROM ${FROM_IMAGE_NAME}
#FROM gitlab-master.nvidia.com/dl/dgx/mxnet:master-py3-devel

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d


RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt

# Install dependencies for system configuration logger
RUN apt-get update
RUN apt-get install -y --no-install-recommends infiniband-diags pciutils
RUN rm -rf /var/lib/apt/lists/*

# install zip
#RUN rm -rf /var/lib/apt/lists/* && apt-get update && apt-cache gencaches && apt-get install -y zip unzip


ENV MXNET_CUDNN_AUTOTUNE_DEFAULT=0
ENV MXNET_ENABLE_CUDA_GRAPHS=1
# SHARP
#ENV NCCL_COLLNET_ENABLE=1

# New dependency engine
ENV MXNET_ASYNC_GPU_ENGINE=1
# Fix for horovod with the new dependency engine
ENV HOROVOD_ENABLE_ASYNC_COMPLETION=1

ENV MXNET_INSTANCENORM_USE_DUAL_STREAMS=1
ENV MXNET_OPTIMIZER_AGGREGATION_SIZE=60
ENV MXNET_GPU_WORKER_NTHREADS=1
ENV MXNET_GPU_COPY_NTHREADS=1
ENV MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
ENV MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999
ENV MXNET_CUDNN_NHWC_BN_HEURISTIC_FWD=1
ENV MXNET_CUDNN_NHWC_BN_HEURISTIC_BWD=1
ENV MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_FWD=1
ENV MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_BWD=1

# DALI jitter
ENV MALLOC_ARENA_MAX=1
ENV MALLOC_MMAP_MAX_=0

#ENV NCCL_DEBUG=TRACE

