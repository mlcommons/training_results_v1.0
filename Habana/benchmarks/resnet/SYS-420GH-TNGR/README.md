# ResNet

## Benchmark Information

ResNet is the
[ResNet-50 image classification](https://github.com/mlperf/training/tree/master/image_classification) benchmark

## Software

[Tensorflow 2](https://www.tensorflow.org/)

## Hardware

Habana Gaudi

## Dataset Preparation

[ImageNet dataset preparation](https://github.com/mlperf/training/tree/master/image_classification#3-datasetenvironment)

# Setup
## System Setup
Follow the steps in [Setup and Install](https://github.com/HabanaAI/Setup_and_Install) to setup your system.  
For this submission the following was used:  
OS: Ubuntu 18.04  
* [On Premises](https://github.com/HabanaAI/Setup_and_Install#on-premises) -- Check if Driver is installed.
* If not installed -- [Install Habana Driver](https://github.com/HabanaAI/Setup_and_Install#install-habana-driver)
* [Install habana-container-runtime package](https://github.com/HabanaAI/Setup_and_Install#install-habana-container-runtime-package-1)
* Continue with Docker pull and training below

## Docker pull and training

- Get the Habana TensorFlow docker image:
```
docker pull vault.habana.ai/gaudi-docker-mlperf/ver1.0/tf/hls-1:0.14-mlperf
```

- Run the training:
```
/root/benchmarks/resnet/SYS-420GH-TNGR/run_and_time.sh
```

- (optional) If the ImageNet dataset is in a different path, use this command
```
DATA_DIR=<Path to ImageNet dataset> <Path to run_and_tim.sh>/run_and_time.sh
```
NOTE: Idealy the dataset/output directories are placed locally on NVME/SSD drive for best performance.
- (optional) To run multiple training (e.g. 5 times)
```
REPEAT=5 <Path to run_and_time.sh>/run_and_time.sh
```

- (optional) Training with the docker container manually:
```
docker run -td -e DISPLAY=$DISPLAY -e LOG_LEVEL_ALL=6 --name habana_mlperf_resnet_v1.0 \
    -v #<Path to ImageNet dataset>:/tf_records -v #<PATH to implementations/resnet-tf-sys-420gh-tngr/:/work ....
docker exec -ti habana_mlperf_resnet_v1.0
(in the container)
/work/run.sh
```
