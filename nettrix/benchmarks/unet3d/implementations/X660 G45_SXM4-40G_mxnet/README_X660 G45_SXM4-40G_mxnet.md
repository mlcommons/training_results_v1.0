## Steps to launch training

### Nettrix X660 G45 (single node)

Launch configuration and system-specific hyperparameters for the Nettrix X660 G45
single-node submission are in the `config_X660 G45_SXM4-40G_conv-dali_1x8x7.sh` script.

Steps required to launch single-node training on NVIDIA DGX A100 40G

1. Build the docker container and push to a docker registry

```
cd ../implementations/mxnet
docker build --pull -t <docker/registry>/mlperf-nvidia:image_segmentation-mxnet .
docker push <docker/registry>/mlperf-nvidia:image_segmentation-mxnet
```

2. Launch the training

```
source config_X660 G45_SXM4-40G_conv-dali_1x8x7.sh
CONT="<docker/registry>/mlperf-nvidia:image_segmentation-mxnet" DATASET_DIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

#### Alternative launch with nvidia-docker

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
source config_X660 G45_SXM4-40G_conv-dali_1x8x7.sh
CONT=<docker/registry>mlperf-nvidia:image_segmentation-mxnet DATASET_DIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_and_time.sh
