# Bert

## Benchmark Information

Bert is the
[Bert](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert) benchmark

## Software

[Tensorflow 2](https://www.tensorflow.org/)

## Hardware

Habana Gaudi

## Location to download Dataset and Checkpoint

[Dataset and Checkpoint download location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT)

## Dataset Preparation

[Bert dataset preparation](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets)

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
- Please follow instruction in run_and_time.sh to fill with appropriate path to dataset, init checkpoint and configuration file  
NOTE: Idealy the dataset/checkpoint/output directories are placed locally on NVME/SSD drive for best performance.
- Run the training:
```
/root/benchmarks/bert/SYS-420GH-TNGR/run_and_time.sh
```

- (optional) To run multiple training (e.g. 10 times)
```
for idx in $(seq 0 1 9)
do
  echo "run_idx=$idx"
  rm -rf $OUTPUT_MODEL_DIR
  mkdir -p ${OUTPUT_MODEL_DIR}
  /root/benchmarks/bert/SYS-420GH-TNGR/run_and_time.sh
done
```



