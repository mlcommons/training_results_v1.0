# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

FROM gitlab-master.nvidia.com/dl/dgx/pytorch:21.04-py3-stage

# Install dependencies for system configuration logger
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        infiniband-diags \
        pciutils \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libibumad3 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /workspace/object_detection

COPY requirements.txt .
RUN pip install --no-cache-dir https://github.com/mlperf/logging/archive/5bc16bf.zip \
 && pip install --no-cache-dir -r requirements.txt

# Copy detectron code and build
COPY . .
RUN pip install -e .
ENV OMP_NUM_THREADS=1
ENV OPENCV_FOR_THREADS_NUM=1

