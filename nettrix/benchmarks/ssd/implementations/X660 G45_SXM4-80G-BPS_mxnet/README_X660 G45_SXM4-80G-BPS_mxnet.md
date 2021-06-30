## Steps to launch training

### Nettrix X660 G45 (single node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100
single node submission are in the `config_X660\ G45_SXM4-80G-BPS.sh` script.

Steps required to launch single node training on X660 G45

1. Build the docker container and push to a docker registry

```
cd ../implementations/mxnet
docker build --pull -t <docker/registry>/mlperf-nvidia:single_stage_detector-mxnet .
docker push <docker/registry>/mlperf-nvidia:single_stage_detector-mxnet
```

2. Launch the training

```
source config_X660\ G45_SXM4-80G-BPS.sh
CONT="<docker/registry>/mlperf-nvidia:single_stage_detector-mxnet" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
