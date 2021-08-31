This file contains the instructions for downloading and preprocessing the dataset,
specifying the location of the input files and building the docker image. Those steps are the same,
irrespective of the hardware platform. However, the actual commands for running the benchmark are different 
on single node and multiple nodes. Please refer to the other README files in this directories for those instructions.

### Dataset downloading and preprocessing 

Steps required to launch DLRM training with HugeCTR on a single NVIDIA DGX A100:

1. Prepare the input dataset.

1.1 Download the dataset from: https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/

1.2 Clone the reference implementation repository.

```
git clone https://github.com/facebookresearch/dlrm/
cd dlrm
git checkout mlperf
```

1.3 Build and run the reference docker image.
```
docker build -t dlrm_reference .
docker run -it --rm --network=host --ipc=host --shm-size=1g --ulimit memlock=-1 \
           --ulimit stack=67108864 --gpus=all  -v /data:/data dlrm_reference
```

1.4 Run the training script to obtain the preprocessed data.
This process can take up to several days and needs a few TB of fast storage space.
As a result, files named: "day_train.bin" and "day_test.bin" will be created.

After creating the preprocessed dataset the script will start training using the reference implementation.
This will be clearly visible in the logs e.g., by the script printing: "Finished training it" etc.
This can be safely interrupted with "Ctrl+C" as we only need this script to produce the preprocessed data
and not to complete the full training run. 

```
python dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" \
       --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset \
       --data-set=terabyte --raw-data-file=/data/day --processed-data-file=/data/day --loss-function=bce \
       --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2048 --print-time \
       --test-freq=102400 --test-mini-batch-size=16384 --test-num-workers=16 --memory-map --mlperf-logging \
       --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle \
       --mlperf-coalesce-sparse-grads --use-gpu
```

### Specify the preprocessed data paths in the JSON config files.

You will need to manually change the location of the datasets in the `dgx_a100.json` or `dgx2h.json` files.
The "source" parameter should contain the absolute path to the `day_train.bin` file and the `eval_source`
parameter should point to the `day_test.bin` file.

However, for launching with nvidia-docker, don't modify the json but rather rename `day_[train|test].bin` as `[train|test]_data.bin` and just make sure to set `DATADIR` as the path to the directory containing those two files.

### Build the container and push to a docker registry.
```
cd ../implementations/hugectr
docker build -t <docker/registry>/mlperf-nvidia:recommendation_hugectr .
docker push <docker/registry>/mlperf-nvidia:recommendation_hugectr
```
