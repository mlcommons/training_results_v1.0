# Graphcore: packed BERT 

This readme dscribes how to run BERT on IPUs.

## Wikipedia datasets
Follow the `bert_data/README.md` to construct the packed BERT Wikipedia dataset for Mlperf.

## Pre-trained checkpoint
Follow the steps in the Mlcommons/Mlperf reference implementation 1.0.0 to download the tf1 checkpoint. 
Place the files in `tf-checkpoints`. More specifically the config files will expect to load the checkpoint from `tf-checkpoints/bs64k_32k_ckpt/model.ckpt-28252`.

## Prepare the environment

### 1) Download the Poplar SDK
  Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
  scripts for poplar and popART.

### 2) Install Boost and compile `custom_ops`

Install Boost:

```bash
apt-get update; apt-get install libboost-all-dev
```

Compile custom_ops:

```bash
make
```

This should create `custom_ops.so`.

### 3) Prepare Python3 virtual environment

Create a virtualenv and install the required packages:

```bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
pip install <path to gc_tensorflow.whl>
```

Note: TensorFlow is required by `bert_tf_loader.py`. You can use the standard TensorFlow version for this BERT example, however using the Graphcore TensorFlow version allows this virtual environment to be used for other TensorFlow programs targeting the IPU.

To instal the **mlperf-logging** Python package use:
```
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
```
Make sure to checkout the 1.0.0 branch of mlperf-logging

## Running BERT

Use `create_submission.py` with sudo to execute 10 runs with different seeds.
For example for pod16:
```
sudo python3 create_submission.py --pod=16 --submission-division=closed
```
and for pod64:
```
sudo python3 create_submission.py --pod=64 --submission-division=closed
```
The mlperf result logs are placed in `result/bert/result_*.txt`
During the first run, an executable will be compiled and saved for subsequent runs.
Individual runs can be launched using `bert.py` directly:
```
python bert.py --config=./configs/mk2/pod16-closed.json
```