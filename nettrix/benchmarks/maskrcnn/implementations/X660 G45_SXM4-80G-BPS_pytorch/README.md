Steps to launch training on a single node

For single-node training, we use docker to run our container.

### Nettrix X660 G45 (single node)

Launch configuration and system-specific hyperparameters for the Nettrix X660 G45
single node submission are in the `config_DGXA100.sh` script.

Steps required to launch single node training on Nettrix X660 G45:

1. Build the container and push to a docker registry:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:object_detection .
docker push <docker/registry>/mlperf-nvidia:object_detection
```

2. Launch the training:

```
source config_X660\ G45_SXM4-80G-BPS.sh
CONT="<docker/registry>/mlperf-nvidia:object_detection" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run_X660\ G45_SXM4-80G-BPS.sub
```

#### Alternative launch with nvidia-docker

When generating results for the official v0.7 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX A100 (single node)](#nvidia-dgx-a100-single-node) explain
how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
docker build --pull -t mlperf-nvidia:object_detection .
source config_X660\ G45_SXM4-80G-BPS.sh
CONT=mlperf-nvidia:object_detection DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker_X660 G45_SXM4-80G-BPS.sh
```

