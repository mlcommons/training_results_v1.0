FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /research

RUN apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python3 \
    python3-pip \
    python \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV HOME /research
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN apt-get update && apt-get install -y \
	    python-setuptools \
	    virtualenv \
	    htop \
	    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get install -y \
	    locales tzdata \
	    && locale-gen en_US.UTF-8 \
	    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2

RUN pip install --upgrade numpy scipy sklearn tf-nightly-gpu
#RUN pip install --upgrade numpy scipy sklearn tensorflow-gpu==2.2.0

RUN apt-get update && apt-get install -y \
	    nvidia-cuda-toolkit \
	    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Mount data into the docker
ADD . /research/resnet
ADD ./tensorflow2 /research/resnet-tf2

WORKDIR /research/resnet
RUN pip install -r official/requirements.txt

ENV DEBIAN_FRONTEND teletype

ENTRYPOINT ["/bin/bash"]
