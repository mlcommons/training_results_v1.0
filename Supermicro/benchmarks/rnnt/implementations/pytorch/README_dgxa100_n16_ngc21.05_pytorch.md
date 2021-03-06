## Steps to launch training

### NVIDIA DGX A100 (multi node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100
multi node submission are in the `config_DGXA100_16x8x16x1.sh` script.

Steps required to launch multi node training on NVIDIA DGX A100

1. Build the docker container and push to a docker registry

```
cd ../implementations/pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:rnn_speech_recognition-pytorch .
docker push <docker/registry>/mlperf-nvidia:rnn_speech_recognition-pytorch
```

2. Launch the training

```
source config_DGXA100_16x8x16x1.sh
CONT="<docker/registry>/mlperf-nvidia:rnn_speech_recognition-pytorch" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
