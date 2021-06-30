### Steps to download data
```
DATADIR=<path/to/data/dir> ./init_datasets.sh
```

### Download of ResNet-34 pretrained backbone weights
```
cd ../mxnet-fujitsu
source scripts/get_resnet34_backbone.sh
```

## Steps to launch training

### FUJITSU PRIMERGY GX2460M1 (single node)
Launch configuration and system-specific hyperparameters for the FUJITSU PRIMERGY GX2460M1
single node submission are in the `config_PG.sh` script.

Steps required to launch single node training on FUJITSU PRIMERGY GX2460M1:

```
BENCHMARK=<benchmark_test> ./setup.sh

cd ../mxnet-fujitsu
source config_PG.sh
CONT="mlperf-fujitsu:single_stage_dector" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
